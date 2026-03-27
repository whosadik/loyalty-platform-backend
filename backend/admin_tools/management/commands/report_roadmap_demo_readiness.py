from __future__ import annotations

from pathlib import Path

from django.core.management.base import BaseCommand

from admin_tools.roadmap_product_freeze import (
    PROJECT_ROOT,
    build_demo_readiness_payload,
    render_demo_readiness_md,
    write_report_bundle,
)


class Command(BaseCommand):
    help = "Generate roadmap demo readiness audit."

    def add_arguments(self, parser):
        parser.add_argument(
            "--output-md",
            type=str,
            default=str((PROJECT_ROOT / "reports" / "roadmap_demo_readiness_audit.md").resolve()),
        )
        parser.add_argument(
            "--output-json",
            type=str,
            default=str((PROJECT_ROOT / "reports" / "roadmap_demo_readiness_audit.json").resolve()),
        )

    def handle(self, *args, **options):
        payload = build_demo_readiness_payload()
        markdown = render_demo_readiness_md(payload)
        output_md = Path(str(options["output_md"]).strip()).resolve()
        output_json = Path(str(options["output_json"]).strip()).resolve()
        write_report_bundle(payload=payload, markdown=markdown, output_md=output_md, output_json=output_json)
        self.stdout.write(self.style.SUCCESS(f"roadmap demo readiness audit written to {output_md}"))
