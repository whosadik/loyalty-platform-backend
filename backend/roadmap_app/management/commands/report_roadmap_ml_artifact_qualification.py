from __future__ import annotations

import json
from pathlib import Path

from django.core.management.base import BaseCommand

from roadmap_app.ml_artifact_qualification import (
    build_roadmap_ml_artifact_qualification_payload,
    render_roadmap_ml_artifact_qualification_markdown,
)


class Command(BaseCommand):
    help = "Build artifact-aligned roadmap ML qualification report for exact configured runtime/shadow artifacts."

    def add_arguments(self, parser):
        parser.add_argument(
            "--output-md",
            type=str,
            default="reports/roadmap_ml_artifact_qualification.md",
        )
        parser.add_argument(
            "--output-json",
            type=str,
            default="reports/roadmap_ml_artifact_qualification.json",
        )

    def handle(self, *args, **options):
        payload = build_roadmap_ml_artifact_qualification_payload()
        markdown = render_roadmap_ml_artifact_qualification_markdown(payload)

        output_md = Path(str(options["output_md"]).strip()).resolve()
        output_json = Path(str(options["output_json"]).strip()).resolve()
        output_md.parent.mkdir(parents=True, exist_ok=True)
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_md.write_text(markdown, encoding="utf-8")
        output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

        self.stdout.write(f"[report_roadmap_ml_artifact_qualification] md={output_md}")
        self.stdout.write(f"[report_roadmap_ml_artifact_qualification] json={output_json}")
