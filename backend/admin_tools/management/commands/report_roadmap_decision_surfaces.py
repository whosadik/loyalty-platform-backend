from __future__ import annotations

import json
from pathlib import Path

from django.core.management.base import BaseCommand, CommandError

from admin_tools.roadmap_planner_transitions import (
    build_transition_decision_records,
    load_transition_source_data,
    readiness_assessment,
    summarize_decision_surfaces,
)


def _resolve_out_dir(raw_path: str) -> Path:
    candidate = Path(str(raw_path)).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    cwd_path = (Path.cwd() / candidate).resolve()
    if cwd_path.parent.exists():
        return cwd_path
    return (Path(__file__).resolve().parents[4] / candidate).resolve()


class Command(BaseCommand):
    help = "Report available roadmap decision surfaces for planner initial and continuation datasets."

    def add_arguments(self, parser):
        parser.add_argument("--days", type=int, default=180)
        parser.add_argument("--include-ga", action="store_true", default=False)
        parser.add_argument("--label-window-days", type=int, default=7)
        parser.add_argument("--out-dir", type=str, default="data/reports/roadmap_decision_surfaces")

    def handle(self, *args, **options):
        days = int(options["days"])
        include_ga = bool(options["include_ga"])
        label_window_days = int(options["label_window_days"])
        out_dir = _resolve_out_dir(str(options["out_dir"]))

        if days <= 0:
            raise CommandError("--days must be > 0")
        if label_window_days <= 0:
            raise CommandError("--label-window-days must be > 0")

        source_data = load_transition_source_data(
            days=days,
            include_ga=include_ga,
            label_window_days=label_window_days,
        )
        if not source_data.get("refresh_rows"):
            raise CommandError("No PLAN_REFRESHED events for selected window.")

        bundle = build_transition_decision_records(
            source_data,
            label_window_days=label_window_days,
            mode="combined",
        )
        summary = summarize_decision_surfaces(source_data, bundle)
        readiness = readiness_assessment(list(bundle.get("decision_records") or []))

        out_dir.mkdir(parents=True, exist_ok=True)
        report_payload = {
            "window_days": int(days),
            "label_window_days": int(label_window_days),
            "include_ga": bool(include_ga),
            **summary,
            "readiness": readiness,
        }
        report_path = out_dir / "report.json"
        report_path.write_text(json.dumps(report_payload, ensure_ascii=False, indent=2), encoding="utf-8")

        lines = [
            "# Roadmap Decision Surfaces",
            "",
            f"- raw_plan_refreshed_events: **{summary.get('raw_plan_refreshed_events', 0)}**",
            f"- excluded_noisy_decision_points_count: **{summary.get('excluded_noisy_decision_points_count', 0)}**",
            f"- excluded_legacy_bad_fragrance_completions_count: **{summary.get('excluded_legacy_bad_fragrance_completions_count', 0)}**",
            "",
            "## Surface Types",
        ]
        for decision_type, payload in sorted((summary.get("surface_types") or {}).items()):
            lines.append(
                f"- {decision_type}: raw={payload.get('raw_count', 0)}, trusted={payload.get('trusted_count', 0)}, "
                f"trusted_users={payload.get('trusted_users', 0)}, non_stop_positive_share={payload.get('non_stop_positive_share', 0.0)}, "
                f"fragrance_share={payload.get('fragrance_share', 0.0)}, step_advance_count={payload.get('step_advance_count', 0)}"
            )
        lines.extend(["", "## Dataset Slices"])
        for slice_name, payload in sorted(({
            "initial_only": summary.get("initial_only") or {},
            "continuation_only": summary.get("continuation_only") or {},
            "combined": summary.get("combined") or {},
        }).items()):
            lines.append(
                f"- {slice_name}: trusted={payload.get('trusted_decisions_total', 0)}, positives={payload.get('positives_excluding_stop', 0)}, "
                f"stop_rate={payload.get('stop_rate', 0.0)}, fragrance_positives={payload.get('fragrance_trusted_positives_count', 0)}"
            )
        lines.extend(["", "## Readiness"])
        for name, payload in sorted(readiness.items()):
            lines.append(f"- {name}: **{payload.get('status', 'unknown')}** - {payload.get('why', '')}")
        summary_path = out_dir / "summary.md"
        summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

        self.stdout.write("\n".join(lines))
        self.stdout.write(f"[report_roadmap_decision_surfaces] report={report_path}")
        self.stdout.write(f"[report_roadmap_decision_surfaces] summary={summary_path}")
