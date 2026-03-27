from __future__ import annotations

from pathlib import Path

from django.core.management.base import BaseCommand

from admin_tools.goldapple_catalog_curated_v2 import build_curated_catalog_v2, write_curated_catalog_v2_artifacts


class Command(BaseCommand):
    help = "Audit and build a curated v2 Goldapple workbook without touching the database."

    def add_arguments(self, parser):
        parser.add_argument(
            "--xlsx",
            type=str,
            default="data/catalog/goldapple_300_products.xlsx",
            help="Source Goldapple workbook path.",
        )
        parser.add_argument(
            "--output-xlsx",
            type=str,
            default="data/catalog/goldapple_300_products_curated_v2.xlsx",
            help="Path for the curated workbook.",
        )
        parser.add_argument(
            "--audit-md",
            type=str,
            default="reports/goldapple_catalog_curated_v2_audit.md",
            help="Path for the markdown audit report.",
        )
        parser.add_argument(
            "--audit-json",
            type=str,
            default="reports/goldapple_catalog_curated_v2_audit.json",
            help="Path for the JSON audit report.",
        )
        parser.add_argument(
            "--changes-csv",
            type=str,
            default="reports/goldapple_catalog_curated_v2_changes.csv",
            help="Path for the CSV with fixed rows.",
        )
        parser.add_argument(
            "--added-products-csv",
            type=str,
            default="reports/goldapple_catalog_curated_v2_added_products.csv",
            help="Path for the CSV with added products.",
        )

    def handle(self, *args, **options):
        result = build_curated_catalog_v2(options["xlsx"])
        artifacts = write_curated_catalog_v2_artifacts(
            result,
            workbook_path=options["output_xlsx"],
            audit_md_path=options["audit_md"],
            audit_json_path=options["audit_json"],
            changes_csv_path=options["changes_csv"],
            added_products_csv_path=options["added_products_csv"],
        )

        workbook_path = Path(artifacts["workbook"]).resolve()
        audit_md = Path(artifacts["audit_md"]).resolve()
        self.stdout.write(self.style.SUCCESS(f"Curated workbook: {workbook_path}"))
        self.stdout.write(self.style.SUCCESS(f"Audit report: {audit_md}"))
        self.stdout.write(
            f"correct={result['status_counts'].get('confident_correct', 0)} "
            f"fixed={result['status_counts'].get('fixable', 0)} "
            f"ambiguous={result['status_counts'].get('ambiguous', 0)} "
            f"rejected={result['status_counts'].get('reject', 0)} "
            f"added={len(result['added_rows'])}"
        )
