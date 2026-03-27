from __future__ import annotations

from pathlib import Path

from django.core.management.base import BaseCommand

from admin_tools.roadmap_product_freeze import (
    PROJECT_ROOT,
    build_decision_freeze_payload,
    render_decision_freeze_md,
    write_report_bundle,
)


class Command(BaseCommand):
    help = "Generate the final roadmap decision freeze report."

    def add_arguments(self, parser):
        parser.add_argument(
            "--output-md",
            type=str,
            default=str((PROJECT_ROOT / "reports" / "roadmap_decision_freeze_report.md").resolve()),
        )
        parser.add_argument(
            "--output-json",
            type=str,
            default=str((PROJECT_ROOT / "reports" / "roadmap_decision_freeze_report.json").resolve()),
        )

    def handle(self, *args, **options):
        payload = build_decision_freeze_payload()
        markdown = render_decision_freeze_md(payload)
        output_md = Path(str(options["output_md"]).strip()).resolve()
        output_json = Path(str(options["output_json"]).strip()).resolve()
        write_report_bundle(payload=payload, markdown=markdown, output_md=output_md, output_json=output_json)
        self.stdout.write(self.style.SUCCESS(f"roadmap decision freeze report written to {output_md}"))
