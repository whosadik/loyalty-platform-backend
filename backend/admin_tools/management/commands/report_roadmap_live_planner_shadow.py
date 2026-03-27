from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pandas as pd
from django.core.management.base import BaseCommand

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ml.training.roadmap_live_planner_common import (  # noqa: E402
    ALLOWED_CATEGORIES,
    STOP_TOKEN,
    build_live_decision_dataframe,
    load_live_dataset_bundle,
    parse_runtime_plan_tokens,
    resolve_path,
    selected_categories,
)
from roadmap_app.ml_live_planner import predict_live_next_action, rollout_live_plan  # noqa: E402


def _selected_categories(raw: str) -> list[str]:
    return selected_categories(raw, allowed=ALLOWED_CATEGORIES)


def _first_step(chain: list[str]) -> str:
    return str(chain[0]) if chain else STOP_TOKEN


def _prefix_rate(left: list[str], right: list[str]) -> float:
    target_len = max(1, len(right))
    matched = 0
    for lval, rval in zip(left, right):
        if str(lval) != str(rval):
            break
        matched += 1
    return float(matched / target_len)


def _diff_reason(left: list[str], right: list[str], *, category: str) -> str:
    if list(left) == list(right):
        return "exact"
    if _first_step(left) != _first_step(right):
        return "first_step_diff"
    if category == "fragrance" and set(left) == set(right):
        return "fragrance_slot_order_diff"
    shorter = min(len(left), len(right))
    if list(left[:shorter]) == list(right[:shorter]):
        if category == "skincare":
            left_tail = left[shorter:]
            right_tail = right[shorter:]
            if left_tail == ["mask"] or right_tail == ["mask"]:
                return "skincare_mask_vs_stop_tail"
        return "tail_length_diff"
    return "mid_plan_diff"


def _sample_rows(frame: pd.DataFrame, *, sample_size: int) -> pd.DataFrame:
    if frame.empty or len(frame) <= sample_size:
        return frame.copy()
    frame = frame.sort_values(["t0_utc", "decision_id"]).reset_index(drop=True)
    return frame.head(sample_size).copy()


def _initial_shadow_for_category(
    *,
    frame: pd.DataFrame,
    category: str,
    model_root: Path,
    sample_size: int,
) -> dict[str, Any]:
    sample = _sample_rows(frame, sample_size=sample_size)
    cases: list[dict[str, Any]] = []
    reason_counts: Counter[str] = Counter()
    first_step_hits = 0
    exact_hits = 0
    prefix_rates: list[float] = []
    for row in sample.itertuples(index=False):
        runtime_chain = parse_runtime_plan_tokens(getattr(row, "plan_product_types", ""))
        ml_chain = rollout_live_plan(category, row._asdict(), model_root=model_root)
        reason = _diff_reason(ml_chain, runtime_chain, category=category)
        reason_counts[reason] += 1
        exact_hits += int(ml_chain == runtime_chain)
        first_step_hits += int(_first_step(ml_chain) == _first_step(runtime_chain))
        prefix_rates.append(_prefix_rate(ml_chain, runtime_chain))
        cases.append(
            {
                "decision_id": int(getattr(row, "decision_id")),
                "user_id": int(getattr(row, "user_id")),
                "label": str(getattr(row, "label")),
                "ml_chain": ml_chain,
                "runtime_chain": runtime_chain,
                "reason": reason,
            }
        )
    return {
        "anchors": int(len(sample)),
        "exact_match_rate": float(exact_hits / max(1, len(sample))),
        "first_step_match_rate": float(first_step_hits / max(1, len(sample))),
        "prefix_match_rate": float(sum(prefix_rates) / max(1, len(prefix_rates))) if prefix_rates else 0.0,
        "top_divergence_reason": reason_counts.most_common(1)[0][0] if reason_counts else "exact",
        "divergence_reasons": dict(reason_counts),
        "sample_cases": cases[:sample_size],
    }


def _transition_shadow_for_category(
    *,
    frame: pd.DataFrame,
    category: str,
    model_root: Path,
    sample_size: int,
) -> dict[str, Any]:
    sample = _sample_rows(frame, sample_size=sample_size)
    cases: list[dict[str, Any]] = []
    reason_counts: Counter[str] = Counter()
    exact_hits = 0
    stop_hits = 0
    continue_hits = 0
    stop_total = 0
    continue_total = 0
    for row in sample.itertuples(index=False):
        runtime_action = str(getattr(row, "current_next_product_type", "") or "").strip().lower() or STOP_TOKEN
        ranked = predict_live_next_action(category, row._asdict(), model_root=model_root)
        ml_action = str((ranked[0] or {}).get("action") or STOP_TOKEN).strip().lower() if ranked else STOP_TOKEN
        if runtime_action == STOP_TOKEN:
            stop_total += 1
            stop_hits += int(ml_action == runtime_action)
        else:
            continue_total += 1
            continue_hits += int(ml_action == runtime_action)
        exact_hits += int(ml_action == runtime_action)
        if ml_action == runtime_action:
            reason = "exact"
        elif ml_action == STOP_TOKEN and runtime_action != STOP_TOKEN:
            reason = "ml_stop_runtime_continue"
        elif ml_action != STOP_TOKEN and runtime_action == STOP_TOKEN:
            reason = "ml_continue_runtime_stop"
        else:
            reason = "action_diff"
        reason_counts[reason] += 1
        cases.append(
            {
                "decision_id": int(getattr(row, "decision_id")),
                "user_id": int(getattr(row, "user_id")),
                "decision_type": str(getattr(row, "decision_type")),
                "label": str(getattr(row, "label")),
                "ml_action": ml_action,
                "runtime_action": runtime_action,
                "reason": reason,
            }
        )
    return {
        "decision_points": int(len(sample)),
        "exact_match_rate": float(exact_hits / max(1, len(sample))),
        "stop_match_rate": float(stop_hits / max(1, stop_total)) if stop_total else 0.0,
        "continue_match_rate": float(continue_hits / max(1, continue_total)) if continue_total else 0.0,
        "top_divergence_reason": reason_counts.most_common(1)[0][0] if reason_counts else "exact",
        "divergence_reasons": dict(reason_counts),
        "sample_cases": cases[:sample_size],
    }


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Roadmap Live Planner Shadow Report",
        "",
        f"- initial_dataset_dir: `{report['initial_dataset_dir']}`",
        f"- transitions_dataset_dir: `{report['transitions_dataset_dir']}`",
        f"- initial_model_root: `{report['initial_model_root']}`",
        f"- transition_model_root: `{report['transition_model_root']}`",
        f"- split_scheme: `{report['split_scheme']}`",
        "",
        "## Initial vs Runtime",
    ]
    for category, payload in sorted((report.get("initial_shadow") or {}).items()):
        lines.extend(
            [
                f"### {category}",
                f"- anchors: `{payload['anchors']}`",
                f"- exact_match_rate: `{payload['exact_match_rate']:.4f}`",
                f"- first_step_match_rate: `{payload['first_step_match_rate']:.4f}`",
                f"- prefix_match_rate: `{payload['prefix_match_rate']:.4f}`",
                f"- top_divergence_reason: `{payload['top_divergence_reason']}`",
            ]
        )
    lines.extend(["", "## Transition vs Runtime"])
    for category, payload in sorted((report.get("transition_shadow") or {}).items()):
        lines.extend(
            [
                f"### {category}",
                f"- decision_points: `{payload['decision_points']}`",
                f"- exact_match_rate: `{payload['exact_match_rate']:.4f}`",
                f"- stop_match_rate: `{payload['stop_match_rate']:.4f}`",
                f"- continue_match_rate: `{payload['continue_match_rate']:.4f}`",
                f"- top_divergence_reason: `{payload['top_divergence_reason']}`",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


class Command(BaseCommand):
    help = "Shadow-only comparison of live-trained planner models against runtime rules on frozen live datasets."

    def add_arguments(self, parser):
        parser.add_argument("--initial-dir", type=str, default="tmp/roadmap_planner_live_initial_v1")
        parser.add_argument("--transitions-dir", type=str, default="tmp/roadmap_planner_live_transitions_v1")
        parser.add_argument("--initial-model-root", type=str, default="models/roadmap_live_initial_planner_v1/time")
        parser.add_argument("--transition-model-root", type=str, default="models/roadmap_live_transition_planner_v1/time")
        parser.add_argument("--split-scheme", type=str, default="time")
        parser.add_argument("--categories", type=str, default="haircare,skincare,fragrance")
        parser.add_argument("--sample-per-category", type=int, default=20)
        parser.add_argument("--seed", type=int, default=42)
        parser.add_argument("--report-md", type=str, default="reports/roadmap_live_planner_shadow.md")
        parser.add_argument("--report-json", type=str, default="reports/roadmap_live_planner_shadow.json")

    def handle(self, *args, **options):
        initial_dir = resolve_path(str(options["initial_dir"]))
        transitions_dir = resolve_path(str(options["transitions_dir"]))
        initial_model_root = resolve_path(str(options["initial_model_root"]))
        transition_model_root = resolve_path(str(options["transition_model_root"]))
        split_scheme = str(options["split_scheme"] or "time").strip().lower()
        categories = _selected_categories(str(options["categories"]))
        sample_size = max(1, int(options["sample_per_category"]))
        seed = int(options["seed"])
        report_md_path = resolve_path(str(options["report_md"]))
        report_json_path = resolve_path(str(options["report_json"]))
        report_md_path.parent.mkdir(parents=True, exist_ok=True)
        report_json_path.parent.mkdir(parents=True, exist_ok=True)

        initial_df, _initial_md, _ = load_live_dataset_bundle(initial_dir)
        transitions_df, _transition_md, _ = load_live_dataset_bundle(transitions_dir)

        report: dict[str, Any] = {
            "initial_dataset_dir": str(initial_dir),
            "transitions_dataset_dir": str(transitions_dir),
            "initial_model_root": str(initial_model_root),
            "transition_model_root": str(transition_model_root),
            "split_scheme": split_scheme,
            "initial_shadow": {},
            "transition_shadow": {},
        }

        for category in categories:
            initial_frame = build_live_decision_dataframe(
                dataset_df=initial_df,
                category=category,
                split_scheme=split_scheme,
                seed=seed,
                continuation_only=False,
            )
            initial_frame = initial_frame[initial_frame["eval_split"].astype(str) == "test"].copy()
            report["initial_shadow"][category] = _initial_shadow_for_category(
                frame=initial_frame,
                category=category,
                model_root=initial_model_root,
                sample_size=sample_size,
            )

            transition_frame = build_live_decision_dataframe(
                dataset_df=transitions_df,
                category=category,
                split_scheme=split_scheme,
                seed=seed,
                continuation_only=True,
            )
            transition_frame = transition_frame[transition_frame["eval_split"].astype(str) == "test"].copy()
            report["transition_shadow"][category] = _transition_shadow_for_category(
                frame=transition_frame,
                category=category,
                model_root=transition_model_root,
                sample_size=sample_size,
            )

        report_json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        report_md_path.write_text(_markdown(report), encoding="utf-8")
        self.stdout.write(f"[report_roadmap_live_planner_shadow] json={report_json_path}")
        self.stdout.write(f"[report_roadmap_live_planner_shadow] md={report_md_path}")
