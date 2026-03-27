from __future__ import annotations

from pathlib import Path

from django.core.management.base import BaseCommand

from admin_tools.roadmap_demo_artifacts import (
    PROJECT_ROOT,
    build_final_technical_summary_markdown,
    write_markdown,
)


class Command(BaseCommand):
    help = "Generate the final roadmap technical summary for diploma/project use."

    def add_arguments(self, parser):
        parser.add_argument(
            "--output-md",
            type=str,
            default=str((PROJECT_ROOT / "reports" / "roadmap_final_technical_summary.md").resolve()),
        )

    def handle(self, *args, **options):
        markdown = build_final_technical_summary_markdown()
        output_md = Path(str(options["output_md"]).strip()).resolve()
        write_markdown(output_md, markdown)
        self.stdout.write(self.style.SUCCESS(f"roadmap final technical summary written to {output_md}"))
