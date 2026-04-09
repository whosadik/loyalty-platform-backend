from __future__ import annotations

import json
from pathlib import Path

from django.conf import settings
from django.core.management.base import BaseCommand, CommandError

from roadmap_app.nextstep_decision_quality import (
    DEFAULT_REPORT_STEM,
    build_nextstep_v4_decision_quality_payload,
    render_nextstep_v4_decision_quality_markdown,
)


FORMAT_CHOICES = ["md", "json", "both"]


class Command(BaseCommand):
    help = "Build decision-quality qualification for the exact roadmap nextstep_v4 artifact on recovered historical anchors."

    def add_arguments(self, parser):
        parser.add_argument("--model-path", default="", help="Exact artifact model.pkl to evaluate.")
        parser.add_argument("--days", type=int, default=30)
        parser.add_argument("--category", default="all")
        parser.add_argument("--include-ga", action="store_true")
        parser.add_argument("--min-slice-size", type=int, default=10)
        parser.add_argument("--format", choices=FORMAT_CHOICES, default="both")
        parser.add_argument(
            "--out",
            default=str(DEFAULT_REPORT_STEM),
            help="Output stem without extension. Defaults to reports/roadmap_nextstep_v4_decision_quality",
        )

    def handle(self, *args, **options):
        model_path = str(options.get("model_path") or "").strip() or str(
            getattr(settings, "ROADMAP_NEXTSTEP_V4_MODEL_PATH", "") or ""
        ).strip()
        if not model_path:
            raise CommandError("model_path is required")

        out_stem = Path(str(options.get("out") or DEFAULT_REPORT_STEM)).expanduser()
        payload = build_nextstep_v4_decision_quality_payload(
            model_path=model_path,
            days=int(options.get("days") or 30),
            category=str(options.get("category") or "all"),
            include_ga=bool(options.get("include_ga")),
            min_slice_size=int(options.get("min_slice_size") or 10),
        )
        md = render_nextstep_v4_decision_quality_markdown(payload)
        out_format = str(options.get("format") or "both").strip().lower()

        if out_format in {"json", "both"}:
            out_stem.parent.mkdir(parents=True, exist_ok=True)
            out_stem.with_suffix(".json").write_text(
                json.dumps(payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        if out_format in {"md", "both"}:
            out_stem.parent.mkdir(parents=True, exist_ok=True)
            out_stem.with_suffix(".md").write_text(md, encoding="utf-8")

        self.stdout.write(self.style.SUCCESS(f"Decision quality report written to `{out_stem}`"))
