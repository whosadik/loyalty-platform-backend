from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

from django.core.management.base import BaseCommand

from admin_tools.goldapple_catalog import (
    _attrs_quality,
    _category_product_type_coverage,
    _load_workbook_rows,
    _normalized_rows,
    audit_goldapple_catalog,
    classify_row_mapping,
    json_default,
)


def _import_status(mapping_status: str) -> str:
    if mapping_status == "confident":
        return "confident"
    if mapping_status == "ambiguous":
        return "ambiguous"
    return "rejected"


def _raw_type_key(row: dict) -> str:
    return str(row.get("product_type_raw") or row.get("Type") or "").strip()


class Command(BaseCommand):
    help = "Build curated Goldapple catalog artifact with confident/ambiguous/rejected normalization statuses."

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
            "--out-dir",
            type=str,
            default="data/catalog/curated/goldapple_300_products",
            help="Output directory for curated artifact and reports.",
        )

    def handle(self, *args, **options):
        xlsx = options["xlsx"]
        sheet = (options.get("sheet") or "").strip()
        out_dir = Path(str(options["out_dir"])).resolve()
        out_dir.mkdir(parents=True, exist_ok=True)

        audit = audit_goldapple_catalog(xlsx, sheet_name=sheet)
        workbook = _load_workbook_rows(xlsx, sheet_name=sheet)
        rows = workbook["rows"]
        normalized_rows = _normalized_rows(rows, workbook["selected_sheet"])
        attr_quality = _attrs_quality(normalized_rows)
        coverage = _category_product_type_coverage(normalized_rows)

        curated_rows: list[dict] = []
        normalization_groups: dict[tuple[str, str, str, str, str], Counter[str]] = defaultdict(Counter)
        status_counts = Counter()

        for raw_row, payload in zip(rows, normalized_rows):
            mapping = classify_row_mapping(raw_row, payload)
            import_status = _import_status(mapping["status"])
            status_counts[import_status] += 1
            curated = {
                "source_sheet": workbook["selected_sheet"],
                "source_row": int(raw_row.get("__source_row__") or 0),
                "source_product_id_raw": str(raw_row.get("id") or ""),
                "raw_name": str(raw_row.get("Name") or raw_row.get("name") or ""),
                "raw_brand": str(raw_row.get("бренд") or raw_row.get("brand") or raw_row.get("Name4") or ""),
                "raw_type": _raw_type_key(raw_row),
                "canonical_category": payload["category"],
                "canonical_product_type": payload["product_type"],
                "expected_category_from_raw": mapping.get("expected_category") or "",
                "expected_product_type_from_raw": mapping.get("expected_product_type") or "",
                "import_status": import_status,
                "import_reason": mapping["reason"],
                "name": payload["name"],
                "brand": payload["brand"],
                "price": payload["price"],
                "currency": payload["currency"],
                "category": payload["category"],
                "product_type": payload["product_type"],
                "concerns": payload["concerns"],
                "attrs": payload["attrs"],
                "actives": payload["actives"],
                "flags": payload["flags"],
                "supported_skin_types": payload["supported_skin_types"],
                "strength": payload["strength"],
                "in_stock": bool(payload["in_stock"]),
                "step": payload["step"],
                "image_url": payload["image_url"],
                "image_urls": payload["image_urls"],
                "description": payload["description"],
                "application_text": payload["application_text"],
                "ingredients_inci": payload["ingredients_inci"],
                "volume_raw": payload["volume_raw"],
                "raw_meta": payload["raw_meta"],
            }
            curated_rows.append(curated)
            key = (
                curated["raw_type"],
                curated["canonical_category"],
                curated["canonical_product_type"],
                curated["expected_category_from_raw"],
                curated["expected_product_type_from_raw"],
            )
            normalization_groups[key][import_status] += 1

        curated_all_path = out_dir / "goldapple_curated_catalog.jsonl"
        curated_confident_path = out_dir / "goldapple_curated_catalog_confident.jsonl"
        normalization_csv_path = out_dir / "goldapple_normalization_table.csv"
        report_json_path = out_dir / "goldapple_curation_report.json"
        report_md_path = out_dir / "goldapple_curation_report.md"

        with curated_all_path.open("w", encoding="utf-8") as handle:
            for row in curated_rows:
                handle.write(json.dumps(row, ensure_ascii=False, default=json_default) + "\n")

        with curated_confident_path.open("w", encoding="utf-8") as handle:
            for row in curated_rows:
                if row["import_status"] == "confident":
                    handle.write(json.dumps(row, ensure_ascii=False, default=json_default) + "\n")

        with normalization_csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                [
                    "raw_type",
                    "canonical_category",
                    "canonical_product_type",
                    "expected_category_from_raw",
                    "expected_product_type_from_raw",
                    "confident_rows",
                    "ambiguous_rows",
                    "rejected_rows",
                ]
            )
            for key, counts in sorted(normalization_groups.items(), key=lambda kv: (-sum(kv[1].values()), kv[0])):
                writer.writerow(
                    [
                        key[0],
                        key[1],
                        key[2],
                        key[3],
                        key[4],
                        counts.get("confident", 0),
                        counts.get("ambiguous", 0),
                        counts.get("rejected", 0),
                    ]
                )

        report = {
            "xlsx_path": audit["xlsx_path"],
            "selected_sheet": workbook["selected_sheet"],
            "rows_total": len(curated_rows),
            "import_status_counts": dict(status_counts),
            "coverage": coverage,
            "attrs_quality": attr_quality,
            "audit_overall_verdict": audit["overall_verdict"],
            "audit_runtime_verdict": audit["runtime_compatibility"]["verdict"],
            "suspicious_examples": audit["mapping_quality"]["examples"][:20],
            "artifact_paths": {
                "curated_all_jsonl": str(curated_all_path),
                "curated_confident_jsonl": str(curated_confident_path),
                "normalization_table_csv": str(normalization_csv_path),
            },
        }
        report_json_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, default=json_default),
            encoding="utf-8",
        )

        markdown = "\n".join(
            [
                "# Goldapple Curation Report",
                "",
                f"- File: `{audit['xlsx_path']}`",
                f"- Selected sheet: `{workbook['selected_sheet']}`",
                f"- Audit verdict: **{audit['overall_verdict']}**",
                f"- Rows total: **{len(curated_rows)}**",
                f"- confident / ambiguous / rejected: **{status_counts.get('confident', 0)} / {status_counts.get('ambiguous', 0)} / {status_counts.get('rejected', 0)}**",
                "",
                "## Coverage",
                *[
                    f"- {category}: total={payload['total_rows']}, missing={payload['missing'] or 'none'}"
                    for category, payload in coverage.items()
                ],
                "",
                "## Fragrance attrs",
                f"- scent_family: {attr_quality['fragrance']['coverage']['scent_family']['pct']}%",
                f"- notes: {attr_quality['fragrance']['coverage']['notes']['pct']}%",
                f"- intensity: {attr_quality['fragrance']['coverage']['intensity']['pct']}%",
                f"- slot distribution: {json.dumps(attr_quality['fragrance']['slot_distribution'], ensure_ascii=False, sort_keys=True)}",
                "",
                "## Artifact paths",
                f"- all rows: `{curated_all_path}`",
                f"- confident only: `{curated_confident_path}`",
                f"- normalization table: `{normalization_csv_path}`",
                f"- json report: `{report_json_path}`",
            ]
        ) + "\n"
        report_md_path.write_text(markdown, encoding="utf-8")

        self.stdout.write(self.style.SUCCESS(f"Curation report written: {report_md_path}"))
        self.stdout.write(self.style.SUCCESS(f"Curation JSON written: {report_json_path}"))
        self.stdout.write(self.style.SUCCESS(f"Curated artifact written: {curated_all_path}"))
        self.stdout.write(self.style.SUCCESS(f"Confident-only artifact written: {curated_confident_path}"))
        self.stdout.write(self.style.SUCCESS(f"Normalization table written: {normalization_csv_path}"))
        self.stdout.write(markdown)
