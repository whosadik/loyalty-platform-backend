from __future__ import annotations

import json
from pathlib import Path

from django.conf import settings
from django.core.management.base import BaseCommand, CommandError

from roadmap_app.nextstep_targeted_retrain import (
    DEFAULT_COMPARE_REPORT_STEM,
    build_targeted_retrain_comparison_payload,
    render_targeted_retrain_comparison_markdown,
)


FORMAT_CHOICES = ["md", "json", "both"]


class Command(BaseCommand):
    help = "Compare current active roadmap nextstep_v4 artifact against a targeted-retrain candidate."

    def add_arguments(self, parser):
        parser.add_argument("--base-model-path", default=str(getattr(settings, "ROADMAP_NEXTSTEP_V4_MODEL_PATH", "") or ""))
        parser.add_argument("--candidate-model-path", required=True)
        parser.add_argument("--days", type=int, default=30)
        parser.add_argument("--format", choices=FORMAT_CHOICES, default="both")
        parser.add_argument("--out", default=str(DEFAULT_COMPARE_REPORT_STEM))

    def handle(self, *args, **options):
        base_model_path = str(options.get("base_model_path") or "").strip()
        candidate_model_path = str(options.get("candidate_model_path") or "").strip()
        if not base_model_path:
            raise CommandError("base_model_path is required")
        if not candidate_model_path:
            raise CommandError("candidate_model_path is required")

        out_stem = Path(str(options.get("out") or DEFAULT_COMPARE_REPORT_STEM)).expanduser()
        payload = build_targeted_retrain_comparison_payload(
            base_model_path=base_model_path,
            candidate_model_path=candidate_model_path,
            days=int(options.get("days") or 30),
        )
        md = render_targeted_retrain_comparison_markdown(payload)
        out_format = str(options.get("format") or "both").strip().lower()
        out_stem.parent.mkdir(parents=True, exist_ok=True)

        if out_format in {"json", "both"}:
            out_stem.with_suffix(".json").write_text(
                json.dumps(payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        if out_format in {"md", "both"}:
            out_stem.with_suffix(".md").write_text(md, encoding="utf-8")

        self.stdout.write(self.style.SUCCESS(f"Targeted retrain comparison written to `{out_stem}`"))
