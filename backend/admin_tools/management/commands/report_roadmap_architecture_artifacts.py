from __future__ import annotations

from pathlib import Path

from django.core.management.base import BaseCommand

from admin_tools.roadmap_demo_artifacts import (
    PROJECT_ROOT,
    render_architecture_blocks_md,
    render_sequence_flows_md,
    write_markdown,
)


class Command(BaseCommand):
    help = "Generate text artifacts for roadmap architecture and sequence diagrams."

    def add_arguments(self, parser):
        parser.add_argument(
            "--blocks-md",
            type=str,
            default=str((PROJECT_ROOT / "reports" / "roadmap_architecture_blocks.md").resolve()),
        )
        parser.add_argument(
            "--flows-md",
            type=str,
            default=str((PROJECT_ROOT / "reports" / "roadmap_sequence_flows.md").resolve()),
        )

    def handle(self, *args, **options):
        blocks_md = Path(str(options["blocks_md"]).strip()).resolve()
        flows_md = Path(str(options["flows_md"]).strip()).resolve()
        write_markdown(blocks_md, render_architecture_blocks_md())
        write_markdown(flows_md, render_sequence_flows_md())
        self.stdout.write(self.style.SUCCESS(f"roadmap architecture artifacts written to {blocks_md} and {flows_md}"))
