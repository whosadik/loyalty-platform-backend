from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd
from django.core.management.base import BaseCommand

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ml.training.roadmap_continuation_planner_common import (  # noqa: E402
    CONTINUATION_DECISION_TYPES,
    continuation_categories,
    resolve_path,
    selected_split_schemes,
    suffix_targets_from_decisions,
)
from ml.training.roadmap_live_planner_common import load_live_dataset_bundle  # noqa: E402
from roadmap_app.ml_continuation_planner import (  # noqa: E402
    predict_continuation_next_action,
    rollout_continuation_suffix,
)
from roadmap_app.services import _runtime_continuation_action_from_signal_state  # noqa: E402

STOP_TOKEN = "__stop__"
NONE_TOKEN = "__none__"


def _runtime_action_from_row(row: dict[str, Any]) -> str:
    token = str(row.get("current_next_product_type") or "").strip().lower()
    if token in {"", NONE_TOKEN}:
        return STOP_TOKEN
    plan_tokens = [
        str(item or "").strip().lower()
        for item in str(row.get("plan_product_types") or "").split("|")
        if str(item or "").strip()
    ]
    patched = _runtime_continuation_action_from_signal_state(
        category=str(row.get("category") or "").strip().lower(),
        trigger=str(row.get("decision_type") or "").strip().lower(),
        current_next_product_type=token,
        plan_product_types=plan_tokens,
        purchased_types=set(),
        owned_types=set(),
        profile_skin_type=str(row.get("profile_skin_type") or ""),
        profile_goals_count=int(row.get("profile_goals_count") or 0),
        profile_avoid_flags_count=int(row.get("profile_avoid_flags_count") or 0),
        profile_hair_type=str(row.get("profile_hair_type") or ""),
        profile_scalp_type=str(row.get("profile_scalp_type") or ""),
        profile_hair_thickness=str(row.get("profile_hair_thickness") or ""),
        profile_hair_concerns_count=int(row.get("profile_hair_concerns_count") or 0),
        profile_has_scalp_objective=bool(row.get("profile_has_scalp_objective")),
        anchor_actives_count=int(row.get("anchor_actives_count") or 0),
        anchor_concerns_count=int(row.get("anchor_concerns_count") or 0),
        anchor_has_scalp_focus=bool(row.get("anchor_has_scalp_focus")),
    )
    return str((patched or {}).get("action") or STOP_TOKEN).strip().lower() or STOP_TOKEN


def _sample_rows(frame: pd.DataFrame, *, sample_size: int) -> pd.DataFrame:
    if frame.empty or len(frame) <= sample_size:
        return frame.copy()
    frame = frame.sort_values(["t0_utc", "decision_id"]).reset_index(drop=True)
    return frame.head(sample_size).copy()


def _suffix_diff_reason(predicted: list[str], target: list[str]) -> str:
    if list(predicted) == list(target):
        return "exact"
    if (predicted[:1] or [STOP_TOKEN])[0] != (target[:1] or [STOP_TOKEN])[0]:
        return "first_action_diff"
    shorter = min(len(predicted), len(target))
    if list(predicted[:shorter]) == list(target[:shorter]):
        return "tail_length_diff"
    return "mid_suffix_diff"


def _next_action_reason(*, ml_action: str, runtime_action: str) -> str:
    if ml_action == runtime_action:
        return "exact"
    if ml_action == STOP_TOKEN and runtime_action != STOP_TOKEN:
        return "ml_stop_runtime_continue"
    if ml_action != STOP_TOKEN and runtime_action == STOP_TOKEN:
        return "ml_continue_runtime_stop"
    return "action_diff"


def _category_shadow(
    *,
    frame: pd.DataFrame,
    category: str,
    model_root: Path,
    sample_size: int,
    suffix_targets: dict[int, list[str]],
) -> dict[str, Any]:
    sample = _sample_rows(frame, sample_size=sample_size)
    next_reason_counts: Counter[str] = Counter()
    suffix_reason_counts: Counter[str] = Counter()
    next_hits = 0
    stop_hits = 0
    continue_hits = 0
    stop_total = 0
    continue_total = 0
    suffix_exact_hits = 0
    suffix_prefix_rates: list[float] = []
    cases: list[dict[str, Any]] = []

    for row in frame.itertuples(index=False):
        row_dict = row._asdict()
        runtime_action = _runtime_action_from_row(row_dict)
        ranked = predict_continuation_next_action(category, row_dict, model_root=model_root)
        ml_action = str((ranked[0] or {}).get("action") or STOP_TOKEN).strip().lower() if ranked else STOP_TOKEN
        next_reason = _next_action_reason(ml_action=ml_action, runtime_action=runtime_action)
        next_reason_counts[next_reason] += 1
        next_hits += int(ml_action == runtime_action)
        if runtime_action == STOP_TOKEN:
            stop_total += 1
            stop_hits += int(ml_action == runtime_action)
        else:
            continue_total += 1
            continue_hits += int(ml_action == runtime_action)

        target_suffix = list(suffix_targets.get(int(getattr(row, "decision_id")), []))
        predicted_suffix = rollout_continuation_suffix(category, row_dict, model_root=model_root)
        suffix_reason = _suffix_diff_reason(predicted_suffix, target_suffix)
        suffix_reason_counts[suffix_reason] += 1
        suffix_exact_hits += int(predicted_suffix == target_suffix)
        denom = max(1, len(target_suffix))
        match = 0
        for left, right in zip(predicted_suffix, target_suffix):
            if str(left) != str(right):
                break
            match += 1
        suffix_prefix_rates.append(float(match / denom))

    for row in sample.itertuples(index=False):
        row_dict = row._asdict()
        runtime_action = _runtime_action_from_row(row_dict)
        ranked = predict_continuation_next_action(category, row_dict, model_root=model_root)
        ml_action = str((ranked[0] or {}).get("action") or STOP_TOKEN).strip().lower() if ranked else STOP_TOKEN
        target_suffix = list(suffix_targets.get(int(getattr(row, "decision_id")), []))
        predicted_suffix = rollout_continuation_suffix(category, row_dict, model_root=model_root)
        next_reason = _next_action_reason(ml_action=ml_action, runtime_action=runtime_action)
        suffix_reason = _suffix_diff_reason(predicted_suffix, target_suffix)
        cases.append(
            {
                "decision_id": int(getattr(row, "decision_id")),
                "user_id": int(getattr(row, "user_id")),
                "decision_type": str(getattr(row, "decision_type")),
                "label": str(getattr(row, "label")),
                "runtime_action": runtime_action,
                "ml_action": ml_action,
                "runtime_reason": next_reason,
                "target_suffix": target_suffix,
                "ml_suffix": predicted_suffix,
                "suffix_reason": suffix_reason,
            }
        )
    return {
        "decision_points": int(len(frame)),
        "next_action_exact_match_rate": float(next_hits / max(1, len(frame))),
        "next_action_stop_match_rate": float(stop_hits / max(1, stop_total)) if stop_total else 0.0,
        "next_action_continue_match_rate": float(continue_hits / max(1, continue_total)) if continue_total else 0.0,
        "suffix_exact_match_rate": float(suffix_exact_hits / max(1, len(frame))),
        "suffix_prefix_match_rate": float(sum(suffix_prefix_rates) / max(1, len(suffix_prefix_rates))) if suffix_prefix_rates else 0.0,
        "top_runtime_disagreement_reason": next_reason_counts.most_common(1)[0][0] if next_reason_counts else "exact",
        "top_suffix_disagreement_reason": suffix_reason_counts.most_common(1)[0][0] if suffix_reason_counts else "exact",
        "runtime_disagreement_reasons": dict(next_reason_counts),
        "suffix_disagreement_reasons": dict(suffix_reason_counts),
        "sample_cases": cases[:sample_size],
    }


def _verdict(report: dict[str, Any]) -> dict[str, Any]:
    by_category: dict[str, str] = {}
    runtime_candidate: dict[str, str] = {}
    blocked: dict[str, str] = {}
    for category in report.get("categories") or {}:
        schemes = report["categories"][category]
        time_metrics = schemes.get("time") or {}
        user_metrics = schemes.get("user") or {}
        time_exact = float(time_metrics.get("next_action_exact_match_rate") or 0.0)
        user_exact = float(user_metrics.get("next_action_exact_match_rate") or 0.0)
        time_continue = float(time_metrics.get("next_action_continue_match_rate") or 0.0)
        user_continue = float(user_metrics.get("next_action_continue_match_rate") or 0.0)
        if min(time_exact, user_exact) >= 0.60 and min(time_continue, user_continue) >= 0.55:
            by_category[category] = "yes"
        elif max(time_exact, user_exact) >= 0.50 and max(time_continue, user_continue) >= 0.45:
            by_category[category] = "borderline"
        else:
            by_category[category] = "no"

        if by_category[category] == "yes" and min(time_metrics.get("suffix_exact_match_rate") or 0.0, user_metrics.get("suffix_exact_match_rate") or 0.0) >= 0.45:
            runtime_candidate[category] = "yes"
        else:
            runtime_candidate[category] = "no"

        if by_category[category] == "no":
            blocked[category] = "Shadow agreement with current runtime continuation behavior is too weak."
        elif runtime_candidate[category] == "no":
            blocked[category] = "Next-action shadow is acceptable, but suffix agreement is still too weak for runtime candidacy."

    return {
        "shadow_ready_by_category": by_category,
        "runtime_candidate_by_category": runtime_candidate,
        "blocked_by_category": blocked,
    }


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Roadmap Continuation Shadow Diff",
        "",
        f"- dataset_dir: `{report['dataset_dir']}`",
        f"- model_root: `{report['model_root']}`",
        f"- split_schemes: `{report['split_schemes']}`",
        "",
        "## Teacher / Bootstrap Reference",
        f"- available: `{report['teacher_bootstrap_reference']['available']}`",
        f"- reason: `{report['teacher_bootstrap_reference']['reason']}`",
        "",
    ]
    for category, payload in sorted((report.get("categories") or {}).items()):
        lines.append(f"## {category}")
        for scheme, metrics in sorted(payload.items()):
            lines.extend(
                [
                    f"### {scheme}",
                    f"- decision_points: `{metrics['decision_points']}`",
                    f"- next_action_exact_match_rate: `{metrics['next_action_exact_match_rate']:.4f}`",
                    f"- next_action_stop_match_rate: `{metrics['next_action_stop_match_rate']:.4f}`",
                    f"- next_action_continue_match_rate: `{metrics['next_action_continue_match_rate']:.4f}`",
                    f"- suffix_exact_match_rate: `{metrics['suffix_exact_match_rate']:.4f}`",
                    f"- suffix_prefix_match_rate: `{metrics['suffix_prefix_match_rate']:.4f}`",
                    f"- top_runtime_disagreement_reason: `{metrics['top_runtime_disagreement_reason']}`",
                    f"- top_suffix_disagreement_reason: `{metrics['top_suffix_disagreement_reason']}`",
                ]
            )
        lines.append("")
    lines.extend(["## Verdict"])
    verdict = dict(report.get("verdict") or {})
    lines.append(f"- shadow_ready_by_category: `{verdict.get('shadow_ready_by_category')}`")
    lines.append(f"- runtime_candidate_by_category: `{verdict.get('runtime_candidate_by_category')}`")
    lines.append(f"- blocked_by_category: `{verdict.get('blocked_by_category')}`")
    return "\n".join(lines).rstrip() + "\n"


class Command(BaseCommand):
    help = "Shadow-only comparison of continuation planner models against current runtime continuation behavior."

    def add_arguments(self, parser):
        parser.add_argument("--data-dir", type=str, default="tmp/roadmap_continuation_dataset_v1")
        parser.add_argument("--model-root", type=str, default="models/roadmap_continuation_planner")
        parser.add_argument("--categories", type=str, default="haircare,skincare,fragrance")
        parser.add_argument("--split-schemes", type=str, default="time,user")
        parser.add_argument("--sample-per-category", type=int, default=40)
        parser.add_argument("--seed", type=int, default=42)
        parser.add_argument("--report-md", type=str, default="reports/roadmap_continuation_shadow_diff.md")
        parser.add_argument("--report-json", type=str, default="reports/roadmap_continuation_shadow_diff.json")

    def handle(self, *args, **options):
        data_dir = resolve_path(str(options["data_dir"]))
        model_root = resolve_path(str(options["model_root"]))
        categories = continuation_categories(str(options["categories"]))
        split_schemes = selected_split_schemes(str(options["split_schemes"]))
        sample_size = max(1, int(options["sample_per_category"]))
        seed = int(options["seed"])
        report_md_path = resolve_path(str(options["report_md"]))
        report_json_path = resolve_path(str(options["report_json"]))
        report_md_path.parent.mkdir(parents=True, exist_ok=True)
        report_json_path.parent.mkdir(parents=True, exist_ok=True)

        dataset_df, _metadata, _splits = load_live_dataset_bundle(data_dir)
        report: dict[str, Any] = {
            "dataset_dir": str(data_dir),
            "model_root": str(model_root),
            "split_schemes": split_schemes,
            "teacher_bootstrap_reference": {
                "available": False,
                "reason": "No distinct continuation teacher/bootstrap artifact exists; current runtime continuation behavior is the only rule reference.",
            },
            "categories": {},
        }

        for category in categories:
            report["categories"][category] = {}
            suffix_targets = suffix_targets_from_decisions(dataset_df, category=category)
            for scheme in split_schemes:
                category_df = dataset_df[
                    (dataset_df["category"].astype(str).str.lower() == category)
                    & (pd.to_numeric(dataset_df["y"], errors="coerce").fillna(0).astype(int) == 1)
                ].copy()
                if "decision_type" in category_df.columns:
                    category_df = category_df[category_df["decision_type"].astype(str).str.lower().isin(CONTINUATION_DECISION_TYPES)].copy()
                if category_df.empty:
                    report["categories"][category][scheme] = {"decision_points": 0}
                    continue
                if "t0_utc" in category_df.columns:
                    category_df["t0_utc"] = pd.to_datetime(category_df["t0_utc"], utc=True, format="mixed")
                from ml.training.roadmap_live_planner_common import apply_split_scheme  # noqa: E402

                category_df = apply_split_scheme(category_df, split_scheme=scheme, seed=seed)
                test_frame = category_df[category_df["eval_split"].astype(str) == "test"].copy()
                scheme_root = model_root / scheme
                report["categories"][category][scheme] = _category_shadow(
                    frame=test_frame,
                    category=category,
                    model_root=scheme_root,
                    sample_size=sample_size,
                    suffix_targets=suffix_targets,
                )

        report["verdict"] = _verdict(report)
        report_json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        report_md_path.write_text(_markdown(report), encoding="utf-8")
        self.stdout.write(f"[report_roadmap_continuation_shadow_diff] json={report_json_path}")
        self.stdout.write(f"[report_roadmap_continuation_shadow_diff] md={report_md_path}")
