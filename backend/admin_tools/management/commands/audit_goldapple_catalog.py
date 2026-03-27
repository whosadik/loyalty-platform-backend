from __future__ import annotations

import json
from pathlib import Path

from django.core.management.base import BaseCommand

from admin_tools.goldapple_catalog import audit_goldapple_catalog, json_default, render_audit_markdown


class Command(BaseCommand):
    help = "Audit Goldapple catalog XLSX compatibility against Product model and runtime catalog expectations."

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
        parser.add_argument("--output-md", type=str, default="", help="Optional markdown report path.")
        parser.add_argument("--output-json", type=str, default="", help="Optional JSON report path.")

    def handle(self, *args, **options):
        audit = audit_goldapple_catalog(options["xlsx"], sheet_name=(options.get("sheet") or "").strip())
        markdown = render_audit_markdown(audit)

        output_md = str(options.get("output_md") or "").strip()
        if output_md:
            md_path = Path(output_md).resolve()
            md_path.parent.mkdir(parents=True, exist_ok=True)
            md_path.write_text(markdown, encoding="utf-8")
            self.stdout.write(self.style.SUCCESS(f"Markdown report written: {md_path}"))

        output_json = str(options.get("output_json") or "").strip()
        if output_json:
            json_path = Path(output_json).resolve()
            json_path.parent.mkdir(parents=True, exist_ok=True)
            json_path.write_text(
                json.dumps(audit, ensure_ascii=False, indent=2, default=json_default),
                encoding="utf-8",
            )
            self.stdout.write(self.style.SUCCESS(f"JSON report written: {json_path}"))

        self.stdout.write(markdown)
