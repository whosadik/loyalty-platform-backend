from __future__ import annotations

import json
import random
import sys
from pathlib import Path

try:
    import pandas as pd
except Exception:  # pragma: no cover
    pd = None

from django.core.management.base import BaseCommand, CommandError

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ml.training.roadmap_initial_planner_common import (  # noqa: E402
    ACTION_SPACE_BY_CATEGORY,
    longest_common_prefix_rate,
    resolve_path,
    sequence_exact_match,
)
from roadmap_app.ml_initial_planner import rollout_initial_plan  # noqa: E402


class Command(BaseCommand):
    help = "Offline shadow report for trained initial planner models against teacher dataset test anchors."

    def add_arguments(self, parser):
        parser.add_argument("--dataset-dir", type=str, default="tmp/roadmap_teacher_v1")
        parser.add_argument("--model-root", type=str, default="models/roadmap_initial_planner")
        parser.add_argument("--split", type=str, default="test")
        parser.add_argument("--sample-per-category", type=int, default=25)
        parser.add_argument("--seed", type=int, default=42)
        parser.add_argument("--categories", type=str, default="")

    def handle(self, *args, **options):
        if pd is None:
            raise CommandError("pandas is required")

        dataset_dir = resolve_path(str(options["dataset_dir"]))
        model_root = resolve_path(str(options["model_root"]))
        split = str(options["split"] or "test").strip()
        sample_per_category = max(1, int(options["sample_per_category"]))
        seed = int(options["seed"])
        categories = [
            str(item or "").strip().lower()
            for item in (str(options["categories"]).split(",") if str(options["categories"]).strip() else ACTION_SPACE_BY_CATEGORY.keys())
            if str(item or "").strip()
        ]

        sequence_df = pd.read_parquet(dataset_dir / "sequence_dataset.parquet")
        rng = random.Random(seed)
        lines = [
            "# Roadmap Initial Planner Shadow Report",
            "",
            f"- dataset_dir: `{dataset_dir}`",
            f"- model_root: `{model_root}`",
            f"- split: `{split}`",
            "",
        ]
        for category in categories:
            frame = sequence_df[
                (sequence_df["category"].astype(str).str.lower() == category)
                & (sequence_df["split"].astype(str) == split)
            ].copy()
            if frame.empty:
                lines.append(f"## {category}")
                lines.append("- no rows")
                lines.append("")
                continue
            rows = list(frame.to_dict(orient="records"))
            rows.sort(key=lambda row: int(row.get("planning_id") or 0))
            if len(rows) > sample_per_category:
                rows = rng.sample(rows, sample_per_category)
                rows.sort(key=lambda row: int(row.get("planning_id") or 0))
            exact = []
            prefix_rates = []
            examples = []
            for row in rows:
                predicted = rollout_initial_plan(category, row, model_root=model_root)
                target = json.loads(str(row.get("target_sequence_json") or "[]"))
                exact.append(sequence_exact_match(predicted, target))
                prefix_rates.append(longest_common_prefix_rate(predicted, target))
                if len(examples) < 3:
                    examples.append(
                        f"- planning_id={row.get('planning_id')}: target={target} predicted={predicted}"
                    )
            lines.append(f"## {category}")
            lines.append(f"- sampled_rows: **{len(rows)}**")
            lines.append(f"- exact_full_match_rate: **{float(sum(exact) / max(1, len(exact))):.4f}**")
            lines.append(f"- prefix_match_rate: **{float(sum(prefix_rates) / max(1, len(prefix_rates))):.4f}**")
            lines.extend(examples)
            lines.append("")

        self.stdout.write("\n".join(lines).rstrip() + "\n")
