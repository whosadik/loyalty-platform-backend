from __future__ import annotations

import json

from django.core.management.base import BaseCommand, CommandError

from admin_tools.roadmap_continuation_dataset import build_continuation_bundle, resolve_repo_path, selected_categories


def _markdown(report: dict) -> str:
    readiness = dict(report.get("readiness") or {})
    lines = [
        "# Roadmap Continuation Readiness",
        "",
        f"- trusted continuation decision points total: **{report.get('trusted_continuation_decision_points_total', 0)}**",
        f"- non-stop positives total: **{report.get('non_stop_positives_total', 0)}**",
        f"- stop rate: **{report.get('stop_rate', 0.0)}**",
        f"- users: **{report.get('users_total', 0)}**",
        "",
        "## Positives By Category",
    ]
    for category, count in sorted((report.get("positives_by_category") or {}).items()):
        lines.append(f"- {category}: **{count}**")
    lines.extend(["", "## Positives By Label"])
    for category, payload in sorted((report.get("positives_by_label_within_category") or {}).items()):
        summary = ", ".join(f"{label}={count}" for label, count in sorted(payload.items()))
        lines.append(f"- {category}: {summary}")
    lines.extend(["", "## Time / Outcome Shape"])
    lines.append(f"- time_distribution_by_month: `{report.get('time_distribution_by_month')}`")
    lines.append(f"- skipped_vs_completed_share: `{report.get('skipped_vs_completed_share')}`")
    lines.append(f"- suffix_length_distribution: `{report.get('suffix_length_distribution')}`")
    lines.append(f"- fragrance_slot_label_distribution: `{report.get('fragrance_slot_label_distribution')}`")
    lines.extend(["", "## Readiness Verdict"])
    lines.append(f"- usable_for_continuation_training: **{str(readiness.get('usable_for_continuation_training')).lower()}**")
    lines.append(f"- usable_for_continuation_shadow: **{str(readiness.get('usable_for_continuation_shadow')).lower()}**")
    lines.append(f"- usable_for_continuation_runtime_candidate: **{str(readiness.get('usable_for_continuation_runtime_candidate')).lower()}**")
    lines.append(f"- recommended_categories_for_training: `{readiness.get('recommended_categories_for_training')}`")
    lines.append(f"- blocked_categories: `{readiness.get('blocked_categories')}`")
    lines.extend(["", "## Category Usability"])
    for category, payload in sorted((readiness.get("categories") or {}).items()):
        lines.append(
            f"- {category}: **{payload.get('status', 'unknown')}** - {payload.get('why', '')} "
            f"(positives={payload.get('non_stop_positives_total', 0)}, missing={payload.get('missing_actions', [])})"
        )
    return "\n".join(lines).rstrip() + "\n"


class Command(BaseCommand):
    help = "Audit trusted continuation/replanning decision points on the current seeded DB."

    def add_arguments(self, parser):
        parser.add_argument("--days", type=int, default=365)
        parser.add_argument("--label-window-days", type=int, default=3)
        parser.add_argument("--include-ga", action="store_true", default=False)
        parser.add_argument("--seed", type=int, default=42)
        parser.add_argument("--categories", type=str, default="haircare,skincare,fragrance")
        parser.add_argument("--output-md", type=str, default="reports/roadmap_continuation_readiness.md")
        parser.add_argument("--output-json", type=str, default="reports/roadmap_continuation_readiness.json")

    def handle(self, *args, **options):
        categories = selected_categories(str(options["categories"]))
        output_md = resolve_repo_path(str(options["output_md"]))
        output_json = resolve_repo_path(str(options["output_json"]))
        output_md.parent.mkdir(parents=True, exist_ok=True)
        output_json.parent.mkdir(parents=True, exist_ok=True)

        try:
            bundle = build_continuation_bundle(
                days=int(options["days"]),
                label_window_days=int(options["label_window_days"]),
                include_ga=bool(options["include_ga"]),
                seed=int(options["seed"]),
                categories=categories,
            )
        except Exception as exc:
            raise CommandError(str(exc)) from exc

        report = dict(bundle["report"])
        output_json.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        output_md.write_text(_markdown(report), encoding="utf-8")
        self.stdout.write(f"[report_roadmap_continuation_readiness] json={output_json}")
        self.stdout.write(f"[report_roadmap_continuation_readiness] md={output_md}")

