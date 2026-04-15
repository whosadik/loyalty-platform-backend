from __future__ import annotations

import json
from pathlib import Path

from django.core.management.base import BaseCommand

from roadmap_app.nextstep_db_rerun_readiness import (
    DEFAULT_V5_DB_RERUN_READINESS_REPORT_STEM,
    build_v5_db_rerun_readiness_payload,
    render_v5_db_rerun_readiness_markdown,
)


FORMAT_CHOICES = ["md", "json", "both"]


class Command(BaseCommand):
    help = "Probe whether the v5 fresh DB-backed broader qualification rerun is currently ready to execute."

    def add_arguments(self, parser):
        parser.add_argument("--days", type=int, default=30)
        parser.add_argument("--category", default="all")
        parser.add_argument("--include-ga", action="store_true")
        parser.add_argument(
            "--candidate-model-path",
            default="models/roadmap_next_step_v5_historical_anchor_targeted_v1/model.pkl",
        )
        parser.add_argument("--format", choices=FORMAT_CHOICES, default="both")
        parser.add_argument("--out", default=str(DEFAULT_V5_DB_RERUN_READINESS_REPORT_STEM))

    def handle(self, *args, **options):
        out_stem = Path(
            str(options.get("out") or DEFAULT_V5_DB_RERUN_READINESS_REPORT_STEM)
        ).expanduser()
        payload = build_v5_db_rerun_readiness_payload(
            days=int(options.get("days") or 30),
            category=str(options.get("category") or "all"),
            include_ga=bool(options.get("include_ga")),
            candidate_model_path=str(
                options.get("candidate_model_path")
                or "models/roadmap_next_step_v5_historical_anchor_targeted_v1/model.pkl"
            ),
        )
        markdown = render_v5_db_rerun_readiness_markdown(payload)
        out_format = str(options.get("format") or "both").strip().lower()
        out_stem.parent.mkdir(parents=True, exist_ok=True)

        if out_format in {"json", "both"}:
            out_stem.with_suffix(".json").write_text(
                json.dumps(payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        if out_format in {"md", "both"}:
            out_stem.with_suffix(".md").write_text(markdown, encoding="utf-8")

        self.stdout.write(self.style.SUCCESS(f"DB rerun readiness report written to `{out_stem}`"))
