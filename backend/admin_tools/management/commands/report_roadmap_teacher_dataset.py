from __future__ import annotations

import json
from pathlib import Path

from django.core.management.base import BaseCommand, CommandError


def _resolve_dir(raw_path: str) -> Path:
    candidate = Path(str(raw_path)).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    cwd_path = (Path.cwd() / candidate).resolve()
    if cwd_path.exists():
        return cwd_path
    return (Path(__file__).resolve().parents[4] / candidate).resolve()


class Command(BaseCommand):
    help = "Report coverage and readiness for an already-built roadmap teacher dataset."

    def add_arguments(self, parser):
        parser.add_argument("--dataset-dir", type=str, default="data/ml/roadmap_teacher_v1")

    def handle(self, *args, **options):
        dataset_dir = _resolve_dir(str(options["dataset_dir"]))
        metadata_path = dataset_dir / "metadata.json"
        if not metadata_path.exists():
            raise CommandError(f"metadata.json not found in {dataset_dir}")

        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        readiness = metadata.get("readiness") or {}
        edge_exclusions = metadata.get("edge_exclusions") or {}
        lines = [
            "# Roadmap Teacher Dataset Report",
            "",
            f"- planning_examples_total: **{metadata.get('planning_examples_total', 0)}**",
            f"- stepwise_rows_total: **{metadata.get('stepwise_rows_total', 0)}**",
            f"- users_total: **{metadata.get('users_total', 0)}**",
            f"- meaningful_non_trivial_length_share: **{metadata.get('meaningful_non_trivial_length_share', 0.0)}**",
            "",
            "## By Category",
        ]
        for name, value in sorted((metadata.get("planning_examples_by_category") or {}).items()):
            lines.append(f"- {name}: **{value}**")
        lines.extend(["", "## First Anchor Type"])
        for name, value in sorted((metadata.get("first_anchor_type_distribution") or {}).items()):
            lines.append(f"- {name}: **{value}**")
        lines.extend(["", "## Target Lengths"])
        for name, value in sorted((metadata.get("target_length_distribution") or {}).items(), key=lambda row: int(row[0])):
            lines.append(f"- {name}: **{value}**")
        lines.extend(["", "## Fragrance Slots"])
        for name, value in sorted((metadata.get("fragrance_slot_distribution") or {}).items()):
            lines.append(f"- {name}: **{value}**")
        lines.extend(["", "## Splits"])
        lines.append(f"- counts: **{json.dumps(metadata.get('split_counts') or {}, ensure_ascii=False)}**")
        lines.append(f"- user_overlap_counts: **{json.dumps(metadata.get('split_user_overlap_counts') or {}, ensure_ascii=False)}**")
        lines.extend(["", "## Edge Exclusions"])
        for name, value in sorted(edge_exclusions.items()):
            lines.append(f"- {name}: **{value}**")
        lines.extend(["", "## Readiness"])
        for name, payload in sorted(readiness.items()):
            lines.append(f"- {name}: **{payload.get('status', 'unknown')}** - {payload.get('why', '')}")

        summary_path = dataset_dir / "teacher_report.md"
        summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        self.stdout.write("\n".join(lines))
        self.stdout.write(f"[report_roadmap_teacher_dataset] report={summary_path}")
