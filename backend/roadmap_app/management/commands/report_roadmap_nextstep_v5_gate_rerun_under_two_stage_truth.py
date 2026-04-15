from __future__ import annotations

import json
from pathlib import Path

from django.conf import settings
from django.core.management.base import BaseCommand, CommandError

from roadmap_app.nextstep_targeted_retrain import (
    DEFAULT_TWO_STAGE_GATE_RERUN_REPORT_STEM,
    build_historical_anchor_candidate_comparison_payload,
    render_historical_anchor_candidate_comparison_markdown,
)


FORMAT_CHOICES = ["md", "json", "both"]


class Command(BaseCommand):
    help = "Re-run v5 continuation comparison under the adopted D_two_stage shampoo truth."

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
        parser.add_argument("--format", choices=FORMAT_CHOICES, default="both")
        parser.add_argument("--out", default=str(DEFAULT_TWO_STAGE_GATE_RERUN_REPORT_STEM))

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

        out_stem = Path(str(options.get("out") or DEFAULT_TWO_STAGE_GATE_RERUN_REPORT_STEM)).expanduser()
        payload = build_historical_anchor_candidate_comparison_payload(
            active_model_path=active_model_path,
            retrain_v1_model_path=retrain_v1_model_path,
            candidate_model_path=candidate_model_path,
            days=int(options.get("days") or 30),
        )
        markdown = render_historical_anchor_candidate_comparison_markdown(payload)
        out_format = str(options.get("format") or "both").strip().lower()
        out_stem.parent.mkdir(parents=True, exist_ok=True)

        if out_format in {"json", "both"}:
            out_stem.with_suffix(".json").write_text(
                json.dumps(payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        if out_format in {"md", "both"}:
            out_stem.with_suffix(".md").write_text(markdown, encoding="utf-8")

        self.stdout.write(self.style.SUCCESS(f"Two-stage shampoo gate rerun written to `{out_stem}`"))
