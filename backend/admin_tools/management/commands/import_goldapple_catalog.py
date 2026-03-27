from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from django.core.management.base import BaseCommand, CommandError
from django.db import transaction

from admin_tools.goldapple_catalog import (
    _load_workbook_rows,
    _normalized_rows,
    audit_goldapple_catalog,
    build_normalized_product_payload,
    classify_row_mapping,
    export_products_backup,
    identify_legacy_synthetic_products,
)
from catalog.models import Product


SAFE_VERDICTS = {"fully_compatible", "compatible_with_normalization_layer"}


class Command(BaseCommand):
    help = "Guarded Goldapple catalog import. Audits first, writes only with --execute, never hard-replaces referenced catalog."

    def add_arguments(self, parser):
        parser.add_argument(
            "--xlsx",
            type=str,
            default="data/catalog/goldapple_300_products.xlsx",
            help="Path to Goldapple catalog XLSX file.",
        )
        parser.add_argument(
            "--sheet",
            type=str,
            default="",
            help="Optional sheet name. Empty means auto-detect best catalog sheet.",
        )
        parser.add_argument(
            "--strategy",
            type=str,
            default="archive_add",
            choices=["archive_add"],
            help="Safe import strategy. Hard replace is intentionally unsupported here.",
        )
        parser.add_argument("--limit", type=int, default=0, help="Optional row limit for testing.")
        parser.add_argument("--backup-dir", type=str, default="backups/catalog", help="Directory for product backup snapshot.")
        parser.add_argument("--execute", action="store_true", help="Apply changes. Without this flag the command is dry-run.")

    def handle(self, *args, **options):
        xlsx = options["xlsx"]
        sheet = (options.get("sheet") or "").strip()
        limit = int(options.get("limit") or 0)
        execute = bool(options.get("execute"))

        audit = audit_goldapple_catalog(xlsx, sheet_name=sheet)
        verdict = audit["overall_verdict"]
        migration_status = audit["migration_strategy"]["status"]

        self.stdout.write(f"audit_overall_verdict={verdict}")
        self.stdout.write(f"audit_runtime_verdict={audit['runtime_compatibility']['verdict']}")
        self.stdout.write(f"migration_status={migration_status}")
        self.stdout.write(f"migration_strategy={audit['migration_strategy']['chosen_strategy']}")

        if verdict not in SAFE_VERDICTS or migration_status == "blocked":
            message = (
                "Goldapple import blocked by audit: "
                f"overall_verdict={verdict}; blockers={audit['migration_strategy']['blockers']}"
            )
            if execute:
                raise CommandError(message)
            self.stdout.write(f"ready_to_import=False")
            self.stdout.write(message)
            return

        workbook = _load_workbook_rows(xlsx, sheet_name=sheet)
        rows = workbook["rows"][:limit] if limit else workbook["rows"]
        normalized_rows = _normalized_rows(rows, workbook["selected_sheet"])

        import_rows: list[dict] = []
        skipped_rows: list[dict] = []
        for raw_row, payload in zip(rows, normalized_rows):
            mapping = classify_row_mapping(raw_row, payload)
            if mapping["status"] != "confident":
                skipped_rows.append(
                    {
                        "source_row": raw_row.get("__source_row__"),
                        "name": payload.get("name"),
                        "reason": mapping["reason"],
                    }
                )
                continue
            if not payload.get("name") or not payload.get("product_type"):
                skipped_rows.append(
                    {
                        "source_row": raw_row.get("__source_row__"),
                        "name": payload.get("name"),
                        "reason": "missing_required_payload_fields",
                    }
                )
                continue
            payload = build_normalized_product_payload({**raw_row, "__sheet__": workbook["selected_sheet"]})
            source_id = str(payload.get("source_product_id") or raw_row.get("__source_row__"))
            payload["source_product_id"] = f"ga:{source_id}"
            payload["raw_meta"] = {
                **(payload.get("raw_meta") or {}),
                "imported_via": "import_goldapple_catalog",
            }
            import_rows.append(payload)

        backup_dir = Path(str(options.get("backup_dir") or "backups/catalog")).resolve()
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        backup_path = backup_dir / f"products_before_goldapple_{timestamp}.jsonl"

        legacy_qs = identify_legacy_synthetic_products()
        self.stdout.write(f"ready_to_import=True")
        self.stdout.write(f"strategy=archive_add")
        self.stdout.write(f"candidate_rows={len(rows)}")
        self.stdout.write(f"importable_rows={len(import_rows)}")
        self.stdout.write(f"skipped_rows={len(skipped_rows)}")
        self.stdout.write(f"legacy_synthetic_products={legacy_qs.count()}")
        self.stdout.write(f"backup_path={backup_path}")

        if skipped_rows:
            sample = skipped_rows[:10]
            self.stdout.write(f"skipped_sample={sample}")

        if not execute:
            return

        with transaction.atomic():
            backup_count = export_products_backup(Product.objects.all(), backup_path)
            archived_count = legacy_qs.update(in_stock=False)
            created = 0
            updated = 0
            for payload in import_rows:
                _, was_created = Product.objects.update_or_create(
                    source_product_id=payload["source_product_id"],
                    defaults=payload,
                )
                if was_created:
                    created += 1
                else:
                    updated += 1

        self.stdout.write(self.style.SUCCESS("Goldapple catalog import completed"))
        self.stdout.write(f"backup_rows={backup_count}")
        self.stdout.write(f"archived_legacy_products={archived_count}")
        self.stdout.write(f"created={created}")
        self.stdout.write(f"updated={updated}")
