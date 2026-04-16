from __future__ import annotations

import json
from pathlib import Path

from django.conf import settings
from django.core.management.base import BaseCommand, CommandError

from roadmap_app.nextstep_skincare_freeze_qualification import (
    DEFAULT_SKINCARE_FREEZE_QUALIFICATION_REPORT_STEM,
    build_v5_skincare_freeze_qualification_payload,
    render_v5_skincare_freeze_qualification_markdown,
)
from roadmap_app.nextstep_targeted_retrain import (
    DEFAULT_HISTORICAL_ANCHOR_COMPARE_REPORT_JSON,
    SOURCE_PREFERENCE_CHOICES,
)


FORMAT_CHOICES = ["md", "json", "both"]


class Command(BaseCommand):
    help = "Report the current skincare-only next qualification lane for v5 under runtime freeze."

    def add_arguments(self, parser):
        parser.add_argument(
            "--active-model-path",
            default=str(getattr(settings, "ROADMAP_NEXTSTEP_V4_MODEL_PATH", "") or ""),
        )
        parser.add_argument(
            "--retrain-v1-model-path",
            default="models/roadmap_next_step_v4_targeted_retrain_v1/model.pkl",
        )
        parser.add_argument(
            "--candidate-model-path",
            default="models/roadmap_next_step_v5_historical_anchor_targeted_v1/model.pkl",
        )
        parser.add_argument("--days", type=int, default=30)
        parser.add_argument("--source-preference", choices=SOURCE_PREFERENCE_CHOICES, default="auto")
        parser.add_argument(
            "--cached-comparison-json",
            default=str(DEFAULT_HISTORICAL_ANCHOR_COMPARE_REPORT_JSON),
        )
        parser.add_argument("--format", choices=FORMAT_CHOICES, default="both")
        parser.add_argument("--out", default=str(DEFAULT_SKINCARE_FREEZE_QUALIFICATION_REPORT_STEM))

    def handle(self, *args, **options):
        active_model_path = str(options.get("active_model_path") or "").strip()
        retrain_v1_model_path = str(options.get("retrain_v1_model_path") or "").strip()
        candidate_model_path = str(options.get("candidate_model_path") or "").strip()
        if not active_model_path:
            raise CommandError("active_model_path is required")
        if not retrain_v1_model_path:
            raise CommandError("retrain_v1_model_path is required")
        if not candidate_model_path:
            raise CommandError("candidate_model_path is required")

        out_stem = Path(
            str(options.get("out") or DEFAULT_SKINCARE_FREEZE_QUALIFICATION_REPORT_STEM)
        ).expanduser()
        payload = build_v5_skincare_freeze_qualification_payload(
            active_model_path=active_model_path,
            retrain_v1_model_path=retrain_v1_model_path,
            candidate_model_path=candidate_model_path,
            days=int(options.get("days") or 30),
            source_preference=str(options.get("source_preference") or "auto"),
            cached_comparison_json_path=str(
                options.get("cached_comparison_json") or DEFAULT_HISTORICAL_ANCHOR_COMPARE_REPORT_JSON
            ),
        )
        markdown = render_v5_skincare_freeze_qualification_markdown(payload)
        out_format = str(options.get("format") or "both").strip().lower()
        out_stem.parent.mkdir(parents=True, exist_ok=True)

        if out_format in {"json", "both"}:
            out_stem.with_suffix(".json").write_text(
                json.dumps(payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        if out_format in {"md", "both"}:
            out_stem.with_suffix(".md").write_text(markdown, encoding="utf-8")

        self.stdout.write(self.style.SUCCESS(f"Skincare freeze qualification report written to `{out_stem}`"))
