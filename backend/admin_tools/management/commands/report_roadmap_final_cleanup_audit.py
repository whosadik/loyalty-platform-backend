from __future__ import annotations

from pathlib import Path

from django.core.management.base import BaseCommand

from admin_tools.roadmap_demo_artifacts import (
    PROJECT_ROOT,
    build_final_cleanup_audit_payload,
    render_final_cleanup_audit_md,
    write_json,
    write_markdown,
)


class Command(BaseCommand):
    help = "Generate the final roadmap cleanup audit."

    def add_arguments(self, parser):
        parser.add_argument(
            "--output-md",
            type=str,
            default=str((PROJECT_ROOT / "reports" / "roadmap_final_cleanup_audit.md").resolve()),
        )
        parser.add_argument(
            "--output-json",
            type=str,
            default=str((PROJECT_ROOT / "reports" / "roadmap_final_cleanup_audit.json").resolve()),
        )

    def handle(self, *args, **options):
        payload = build_final_cleanup_audit_payload()
        markdown = render_final_cleanup_audit_md(payload)
        output_md = Path(str(options["output_md"]).strip()).resolve()
        output_json = Path(str(options["output_json"]).strip()).resolve()
        write_markdown(output_md, markdown)
        write_json(output_json, payload)
        self.stdout.write(self.style.SUCCESS(f"roadmap final cleanup audit written to {output_md}"))
