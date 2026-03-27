from __future__ import annotations

from pathlib import Path

from django.core.management.base import BaseCommand

from admin_tools.roadmap_product_freeze import PROJECT_ROOT, build_diploma_positioning_markdown


class Command(BaseCommand):
    help = "Generate a short roadmap technical note for diploma/project positioning."

    def add_arguments(self, parser):
        parser.add_argument(
            "--output-md",
            type=str,
            default=str((PROJECT_ROOT / "reports" / "roadmap_diploma_positioning.md").resolve()),
        )

    def handle(self, *args, **options):
        markdown = build_diploma_positioning_markdown()
        output_md = Path(str(options["output_md"]).strip()).resolve()
        output_md.parent.mkdir(parents=True, exist_ok=True)
        output_md.write_text(markdown, encoding="utf-8")
        self.stdout.write(self.style.SUCCESS(f"roadmap diploma positioning note written to {output_md}"))
