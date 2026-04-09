from __future__ import annotations

import json
from pathlib import Path

from django.core.management.base import BaseCommand

from roadmap_app.nextstep_haircare_shampoo_truth_design import (
    DEFAULT_REPORT_STEM,
    build_nextstep_haircare_shampoo_truth_design_payload,
    render_nextstep_haircare_shampoo_truth_design_markdown,
)


class Command(BaseCommand):
    help = "Build a read-only shampoo truth-design report on immutable historical anchors."

    def add_arguments(self, parser):
        parser.add_argument("--model-path", required=True)
        parser.add_argument("--reference-model-path", default="")
        parser.add_argument("--days", type=int, default=30)
        parser.add_argument("--include-ga", action="store_true", default=False)
        parser.add_argument("--format", choices=["md", "json", "both"], default="both")
        parser.add_argument("--out", default=str(DEFAULT_REPORT_STEM))

    def handle(self, *args, **options):
        out_stem = Path(str(options.get("out") or DEFAULT_REPORT_STEM)).expanduser().resolve()
        out_stem.parent.mkdir(parents=True, exist_ok=True)

        payload = build_nextstep_haircare_shampoo_truth_design_payload(
            model_path=str(options.get("model_path") or ""),
            reference_model_path=str(options.get("reference_model_path") or ""),
            days=int(options.get("days") or 30),
            include_ga=bool(options.get("include_ga")),
        )
        markdown = render_nextstep_haircare_shampoo_truth_design_markdown(payload)

        fmt = str(options.get("format") or "both")
        if fmt in {"json", "both"}:
            json_path = out_stem.with_suffix(".json")
            json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            self.stdout.write(f"Shampoo truth-design JSON written to `{json_path}`")
        if fmt in {"md", "both"}:
            md_path = out_stem.with_suffix(".md")
            md_path.write_text(markdown, encoding="utf-8")
            self.stdout.write(f"Shampoo truth-design markdown written to `{md_path}`")
