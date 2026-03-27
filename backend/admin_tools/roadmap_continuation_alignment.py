from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from admin_tools.roadmap_continuation_dataset import build_continuation_bundle
from ml.training.roadmap_continuation_planner_common import (
    CONTINUATION_DECISION_TYPES,
    build_continuation_decision_dataframe,
    continuation_categories,
    selected_split_schemes,
    suffix_targets_from_decisions,
)
from ml.training.roadmap_live_planner_common import load_live_dataset_bundle, resolve_path
from roadmap_app.ml_continuation_planner import predict_continuation_next_action
from roadmap_app.services import _runtime_continuation_action_from_signal_state

STOP_TOKEN = "__stop__"
NONE_TOKEN = "__none__"
DEFAULT_CATEGORIES = ("haircare", "skincare", "fragrance")
LOW_CONFIDENCE_THRESHOLD = 0.55
HIGH_CONFIDENCE_THRESHOLD = 0.80


def runtime_action_from_row(row: dict[str, Any]) -> str:
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
    patched_action = str((patched or {}).get("action") or STOP_TOKEN).strip().lower() or STOP_TOKEN
    return patched_action


def confidence_bucket(score: float) -> str:
    value = float(score or 0.0)
    if value < LOW_CONFIDENCE_THRESHOLD:
        return "low"
    if value < HIGH_CONFIDENCE_THRESHOLD:
        return "medium"
    return "high"


def suffix_length_bucket(length: int) -> str:
    value = int(length or 0)
    if value <= 0:
        return "0"
    if value == 1:
        return "1"
    if value == 2:
        return "2"
    return "3+"


def primary_reason_code(*, category: str, ml_action: str, runtime_action: str, top_probability: float) -> str:
    category = str(category or "").strip().lower()
    ml_action = str(ml_action or "").strip().lower() or STOP_TOKEN
    runtime_action = str(runtime_action or "").strip().lower() or STOP_TOKEN
    if ml_action == runtime_action:
        return "exact"
    if ml_action == STOP_TOKEN and runtime_action != STOP_TOKEN:
        return "ml_stop_runtime_continue"
    if ml_action != STOP_TOKEN and runtime_action == STOP_TOKEN:
        return "ml_continue_runtime_stop"
    if category == "fragrance" and ml_action != STOP_TOKEN and runtime_action != STOP_TOKEN:
        return "fragrance_slot_conflicts"
    if float(top_probability or 0.0) < LOW_CONFIDENCE_THRESHOLD:
        return "low_confidence_ml_disagreement"
    return "both_continue_but_different_action"


def suspected_label_noise_reason(row: dict[str, Any]) -> str:
    truth = str(row.get("truth_label") or STOP_TOKEN).strip().lower() or STOP_TOKEN
    runtime_action = str(row.get("runtime_action") or STOP_TOKEN).strip().lower() or STOP_TOKEN
    ml_action = str(row.get("ml_action") or STOP_TOKEN).strip().lower() or STOP_TOKEN
    label_source = str(row.get("label_source") or "").strip().lower()
    if truth == STOP_TOKEN and runtime_action != STOP_TOKEN and ml_action != STOP_TOKEN:
        return "stop_label_but_both_continue"
    if truth != STOP_TOKEN and runtime_action == STOP_TOKEN and ml_action == STOP_TOKEN:
        return "continue_label_but_both_stop"
    if label_source in {"stop_no_progress", "terminal_after_outcome_stop"} and runtime_action != STOP_TOKEN:
        return "stop_window_conflict"
    return ""


def state_snapshot_from_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "current_next_step": str(row.get("current_next_product_type") or STOP_TOKEN),
        "plan_product_types": str(row.get("plan_product_types") or ""),
        "steps_total": int(row.get("steps_total") or 0),
        "completed_steps_count": int(row.get("completed_steps_count") or 0),
        "skipped_steps_count": int(row.get("skipped_steps_count") or 0),
        "missing_steps_count": int(row.get("missing_steps_count") or 0),
        "recommended_steps_count": int(row.get("recommended_steps_count") or 0),
        "remaining_actionable_steps_count": int(row.get("remaining_actionable_steps_count") or 0),
        "remaining_depth_in_plan": int(row.get("remaining_depth_in_plan") or 0),
        "next_step_index_current": int(row.get("next_step_index_current") or 0),
        "last_product_types": [
            str(row.get("last1_product_type") or NONE_TOKEN),
            str(row.get("last2_product_type") or NONE_TOKEN),
            str(row.get("last3_product_type") or NONE_TOKEN),
        ],
        "days_since_last_purchase_in_category": int(row.get("days_since_last_purchase_in_category") or 0),
        "prior_category_purchase_total": int(row.get("prior_category_purchase_total") or 0),
    }


def prefix_state_from_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "completed_steps_count": int(row.get("completed_steps_count") or 0),
        "skipped_steps_count": int(row.get("skipped_steps_count") or 0),
        "steps_completed_in_episode_count": int(row.get("steps_completed_in_episode_count") or 0),
        "steps_skipped_in_episode_count": int(row.get("steps_skipped_in_episode_count") or 0),
    }


def _decision_frame_from_bundle(bundle: dict[str, Any]) -> pd.DataFrame:
    records = [dict(row) for row in (bundle.get("decision_records") or [])]
    if not records:
        return pd.DataFrame()
    frame = pd.DataFrame(records)
    frame["y"] = 1
    if "t0_utc" in frame.columns:
        frame["t0_utc"] = pd.to_datetime(frame["t0_utc"], utc=True, format="mixed")
    return frame


def evaluate_decision_frame(
    *,
    decisions_df: pd.DataFrame,
    category: str,
    model_root: Path,
    split_scheme: str,
    eval_split: str | None,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if decisions_df.empty:
        return pd.DataFrame()
    category = str(category or "").strip().lower()
    suffix_targets = suffix_targets_from_decisions(decisions_df, category=category)
    working = decisions_df[decisions_df["category"].astype(str).str.lower() == category].copy()
    if "decision_type" in working.columns:
        working = working[working["decision_type"].astype(str).str.lower().isin(CONTINUATION_DECISION_TYPES)].copy()
    if eval_split:
        working = working[working["eval_split"].astype(str) == str(eval_split)].copy()
    if working.empty:
        return pd.DataFrame()

    for row in working.itertuples(index=False):
        row_dict = row._asdict()
        runtime_action = runtime_action_from_row(row_dict)
        ranked = predict_continuation_next_action(category, row_dict, model_root=model_root)
        top_predictions = [
            {
                "action": str(item.get("action") or STOP_TOKEN).strip().lower() or STOP_TOKEN,
                "prob": round(float(item.get("prob") or 0.0), 6),
            }
            for item in ranked[:3]
        ]
        ml_action = str((top_predictions[0] or {}).get("action") or STOP_TOKEN)
        ml_top_probability = float((top_predictions[0] or {}).get("prob") or 0.0)
        truth_label = str(getattr(row, "label") or STOP_TOKEN).strip().lower() or STOP_TOKEN
        reason_code = primary_reason_code(
            category=category,
            ml_action=ml_action,
            runtime_action=runtime_action,
            top_probability=ml_top_probability,
        )
        suffix_target = list(suffix_targets.get(int(getattr(row, "decision_id")), []))
        suffix_len = len(suffix_target)
        payload = {
            "decision_id": int(getattr(row, "decision_id")),
            "episode_id": int(getattr(row, "episode_id")),
            "plan_id": int(getattr(row, "plan_id") or 0),
            "user_id": int(getattr(row, "user_id")),
            "category": category,
            "split_scheme": str(split_scheme),
            "decision_type": str(getattr(row, "decision_type")),
            "decision_time": str(getattr(row, "t0_utc")),
            "truth_label": truth_label,
            "label_source": str(getattr(row, "label_source", "") or ""),
            "runtime_action": runtime_action,
            "ml_action": ml_action,
            "ml_top_probability": round(ml_top_probability, 6),
            "ml_confidence_bucket": confidence_bucket(ml_top_probability),
            "suffix_length": int(suffix_len),
            "suffix_length_bucket": suffix_length_bucket(suffix_len),
            "reason_code": reason_code,
            "is_disagreement": int(ml_action != runtime_action),
            "low_confidence_ml_disagreement": int(
                ml_action != runtime_action and ml_top_probability < LOW_CONFIDENCE_THRESHOLD
            ),
            "fragrance_slot_conflict": int(reason_code == "fragrance_slot_conflicts"),
            "ml_matches_truth": int(ml_action == truth_label),
            "runtime_matches_truth": int(runtime_action == truth_label),
            "suspected_label_noise_reason": "",
            "top_predictions": top_predictions,
            "current_next_step": str(getattr(row, "current_next_product_type", "") or ""),
            "state_snapshot": state_snapshot_from_row(row_dict),
            "prefix_state": prefix_state_from_row(row_dict),
        }
        payload["suspected_label_noise_reason"] = suspected_label_noise_reason(payload)
        rows.append(payload)
    return pd.DataFrame(rows)


def load_eval_rows_from_dataset(
    *,
    data_dir: str | Path,
    model_root: str | Path,
    categories: list[str],
    split_schemes: list[str],
    seed: int,
    eval_split: str = "test",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    dataset_df, metadata, _splits = load_live_dataset_bundle(resolve_path(str(data_dir)))
    frames: list[pd.DataFrame] = []
    model_root_path = resolve_path(str(model_root))
    for scheme in split_schemes:
        for category in categories:
            decisions_df = build_continuation_decision_dataframe(
                dataset_df=dataset_df,
                category=category,
                split_scheme=scheme,
                seed=int(seed),
            )
            if decisions_df.empty:
                continue
            frames.append(
                evaluate_decision_frame(
                    decisions_df=decisions_df,
                    category=category,
                    model_root=model_root_path / scheme,
                    split_scheme=scheme,
                    eval_split=eval_split,
                )
            )
    out = pd.concat([frame for frame in frames if not frame.empty], ignore_index=True) if frames else pd.DataFrame()
    return out, metadata


def _count_rate(series: pd.Series) -> float:
    if series.empty:
        return 0.0
    return round(float(series.mean()), 6)


def aggregate_alignment(rows: pd.DataFrame, *, group_fields: list[str]) -> list[dict[str, Any]]:
    if rows.empty:
        return []
    out: list[dict[str, Any]] = []
    grouped = rows.groupby(group_fields, dropna=False, sort=True)
    for group_key, frame in grouped:
        values = group_key if isinstance(group_key, tuple) else (group_key,)
        payload = {field: value for field, value in zip(group_fields, values)}
        payload.update(
            {
                "count": int(len(frame)),
                "runtime_accuracy": _count_rate(frame["runtime_matches_truth"]),
                "ml_accuracy": _count_rate(frame["ml_matches_truth"]),
                "ml_minus_runtime_accuracy": round(
                    float(frame["ml_matches_truth"].mean() - frame["runtime_matches_truth"].mean()),
                    6,
                ),
                "runtime_right_ml_wrong": int(((frame["runtime_matches_truth"] == 1) & (frame["ml_matches_truth"] == 0)).sum()),
                "ml_right_runtime_wrong": int(((frame["runtime_matches_truth"] == 0) & (frame["ml_matches_truth"] == 1)).sum()),
                "both_right": int(((frame["runtime_matches_truth"] == 1) & (frame["ml_matches_truth"] == 1)).sum()),
                "both_wrong": int(((frame["runtime_matches_truth"] == 0) & (frame["ml_matches_truth"] == 0)).sum()),
                "suspected_label_noise_count": int(frame["suspected_label_noise_reason"].astype(str).ne("").sum()),
            }
        )
        out.append(payload)
    out.sort(key=lambda item: tuple(str(item.get(field)) for field in group_fields))
    return out


def disagreement_summary(rows: pd.DataFrame) -> dict[str, Any]:
    if rows.empty:
        return {"rows_total": 0, "disagreements_total": 0, "reason_counts": {}}
    disagreements = rows[rows["is_disagreement"].astype(int) == 1].copy()
    low_conf = disagreements["low_confidence_ml_disagreement"] if "low_confidence_ml_disagreement" in disagreements.columns else pd.Series([0] * len(disagreements))
    fragrance_conflicts = disagreements["fragrance_slot_conflict"] if "fragrance_slot_conflict" in disagreements.columns else pd.Series([0] * len(disagreements))
    return {
        "rows_total": int(len(rows)),
        "disagreements_total": int(len(disagreements)),
        "reason_counts": {
            "ml_stop_runtime_continue": int((disagreements["reason_code"] == "ml_stop_runtime_continue").sum()),
            "ml_continue_runtime_stop": int((disagreements["reason_code"] == "ml_continue_runtime_stop").sum()),
            "both_continue_but_different_action": int((disagreements["reason_code"] == "both_continue_but_different_action").sum()),
            "low_confidence_ml_disagreement": int(low_conf.sum()),
            "fragrance_slot_conflicts": int(fragrance_conflicts.sum()),
        },
    }


def sample_disagreement_cases(rows: pd.DataFrame, *, per_category: int) -> list[dict[str, Any]]:
    if rows.empty:
        return []
    disagreements = rows[rows["is_disagreement"].astype(int) == 1].copy()
    if disagreements.empty:
        return []
    if "ml_top_probability" not in disagreements.columns:
        disagreements["ml_top_probability"] = 0.0
    if "decision_time" not in disagreements.columns:
        disagreements["decision_time"] = ""
    disagreements = disagreements.sort_values(
        ["category", "split_scheme", "ml_top_probability", "decision_time"],
        ascending=[True, True, False, True],
    )
    samples: list[dict[str, Any]] = []
    for (scheme, category), frame in disagreements.groupby(["split_scheme", "category"], sort=True):
        for row in frame.head(max(1, int(per_category))).to_dict(orient="records"):
            top_predictions = list(row.get("top_predictions") or row.get("ml_top_predictions") or [])
            samples.append(
                {
                    "split_scheme": str(scheme),
                    "category": str(category),
                    "decision_id": int(row["decision_id"]),
                    "user_id": int(row["user_id"]),
                    "decision_time": str(row["decision_time"]),
                    "decision_type": str(row["decision_type"]),
                    "current_next_step": str(row["current_next_step"]),
                    "prefix_state": dict(row["prefix_state"]),
                    "state_snapshot": dict(row["state_snapshot"]),
                    "runtime_action": str(row["runtime_action"]),
                    "truth_label": str(row["truth_label"]),
                    "ml_top_predictions": top_predictions,
                    "reason_code": str(row["reason_code"]),
                    "low_confidence_ml_disagreement": bool(row["low_confidence_ml_disagreement"]),
                    "suspected_label_noise_reason": str(row["suspected_label_noise_reason"]),
                }
            )
    return samples


def top_alignment_examples(rows: pd.DataFrame, *, sample_per_category: int) -> dict[str, list[dict[str, Any]]]:
    if rows.empty:
        return {"runtime_right_ml_wrong": [], "ml_right_runtime_wrong": [], "suspected_label_noise": []}
    out: dict[str, list[dict[str, Any]]] = {
        "runtime_right_ml_wrong": [],
        "ml_right_runtime_wrong": [],
        "suspected_label_noise": [],
    }
    runtime_right = rows[(rows["runtime_matches_truth"] == 1) & (rows["ml_matches_truth"] == 0)].copy()
    ml_right = rows[(rows["runtime_matches_truth"] == 0) & (rows["ml_matches_truth"] == 1)].copy()
    noisy = rows[rows["suspected_label_noise_reason"].astype(str).ne("")].copy()
    for key, frame in {
        "runtime_right_ml_wrong": runtime_right,
        "ml_right_runtime_wrong": ml_right,
        "suspected_label_noise": noisy,
    }.items():
        if frame.empty:
            continue
        if "ml_top_probability" not in frame.columns:
            frame["ml_top_probability"] = 0.0
        frame = frame.sort_values(["category", "split_scheme", "ml_top_probability"], ascending=[True, True, False])
        for (_scheme, _category), chunk in frame.groupby(["split_scheme", "category"], sort=True):
            out[key].extend(chunk.head(max(1, int(sample_per_category))).to_dict(orient="records"))
    return out


def window_sensitivity_report(
    *,
    categories: list[str],
    model_root: str | Path,
    model_split_scheme: str,
    seed: int,
    window_days_list: list[int],
    days: int = 365,
    include_ga: bool = False,
) -> dict[str, Any]:
    report: dict[str, Any] = {}
    model_root_path = resolve_path(str(model_root)) / str(model_split_scheme or "user").strip().lower()
    for window in window_days_list:
        bundle = build_continuation_bundle(
            days=int(days),
            label_window_days=int(window),
            include_ga=bool(include_ga),
            seed=int(seed),
            categories=categories,
        )
        decisions_df = _decision_frame_from_bundle(bundle)
        window_frames: list[pd.DataFrame] = []
        for category in categories:
            category_df = evaluate_decision_frame(
                decisions_df=decisions_df,
                category=category,
                model_root=model_root_path,
                split_scheme=f"sensitivity_{window}",
                eval_split=None,
            )
            if not category_df.empty:
                window_frames.append(category_df)
        eval_rows = pd.concat(window_frames, ignore_index=True) if window_frames else pd.DataFrame()
        report[str(window)] = {
            "trusted_decisions_total": int(bundle["metadata"]["decision_points_total"]),
            "non_stop_positives_total": int(sum(int(v) for v in (bundle["metadata"]["positives_by_label"] or {}).values())),
            "stop_rate": float(bundle["metadata"]["stop_label_rate"]),
            "positives_by_category": dict(bundle["metadata"]["positives_by_category"]),
            "fragrance_slot_label_distribution": dict(bundle["metadata"]["fragrance_slot_label_distribution"]),
            "runtime_accuracy_overall": _count_rate(eval_rows["runtime_matches_truth"]) if not eval_rows.empty else 0.0,
            "ml_accuracy_overall": _count_rate(eval_rows["ml_matches_truth"]) if not eval_rows.empty else 0.0,
            "runtime_continue_when_truth_stop_rate": _count_rate(
                ((eval_rows["truth_label"] == STOP_TOKEN) & (eval_rows["runtime_action"] != STOP_TOKEN)).astype(int)
            )
            if not eval_rows.empty
            else 0.0,
            "by_category": aggregate_alignment(eval_rows, group_fields=["category"]),
            "post_skipped_non_stop_rate": _count_rate(
                (
                    (eval_rows["decision_type"].astype(str) == "post_skipped")
                    & (eval_rows["truth_label"].astype(str) != STOP_TOKEN)
                ).astype(int)
            )
            if not eval_rows.empty
            else 0.0,
        }
    return report
