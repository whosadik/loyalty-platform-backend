from __future__ import annotations

from pathlib import Path

from django.core.management.base import BaseCommand

from admin_tools.roadmap_demo_artifacts import (
    PROJECT_ROOT,
    build_demo_scenarios_payload,
    render_demo_script_md,
    write_markdown,
)


class Command(BaseCommand):
    help = "Generate the reproducible roadmap demo scenario pack."

    def add_arguments(self, parser):
        parser.add_argument(
            "--output-md",
            type=str,
            default=str((PROJECT_ROOT / "reports" / "roadmap_demo_script.md").resolve()),
        )

    def handle(self, *args, **options):
        payload = build_demo_scenarios_payload()
        markdown = render_demo_script_md(payload)
        output_md = Path(str(options["output_md"]).strip()).resolve()
        write_markdown(output_md, markdown)
        self.stdout.write(self.style.SUCCESS(f"roadmap demo script written to {output_md}"))
