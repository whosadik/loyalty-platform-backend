from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np
import pandas as pd

try:
    from .roadmap_initial_planner_common import longest_common_prefix_rate, sequence_exact_match
    from .roadmap_live_planner_common import (
        ALLOWED_CATEGORIES,
        STOP_TOKEN,
        accuracy,
        apply_split_scheme,
        ensure_dependencies,
        load_live_dataset_bundle,
        per_label_stats,
        predict_bundle_probabilities,
        recall_at_k,
        resolve_estimator_name,
        resolve_path,
        selected_categories,
        selected_split_schemes,
        split_frames,
        split_user_overlap,
        train_live_category_bundle,
        write_dataset_manifest,
        load_model_artifact,
        repo_root,
    )
except ImportError:  # pragma: no cover
    from roadmap_initial_planner_common import longest_common_prefix_rate, sequence_exact_match
    from roadmap_live_planner_common import (
        ALLOWED_CATEGORIES,
        STOP_TOKEN,
        accuracy,
        apply_split_scheme,
        ensure_dependencies,
        load_live_dataset_bundle,
        per_label_stats,
        predict_bundle_probabilities,
        recall_at_k,
        resolve_estimator_name,
        resolve_path,
        selected_categories,
        selected_split_schemes,
        split_frames,
        split_user_overlap,
        train_live_category_bundle,
        write_dataset_manifest,
        load_model_artifact,
        repo_root,
    )

CONTINUATION_DECISION_TYPES = {"post_completed", "post_skipped", "other_trusted_transition"}
DEFAULT_CATEGORIES = ("haircare", "skincare", "fragrance")


def _fallback_top_confusion_pairs(matrix: dict[str, dict[str, int]], *, limit: int = 5) -> list[dict[str, Any]]:
    pairs: list[dict[str, Any]] = []
    for truth, row in matrix.items():
        for pred, count in row.items():
            if str(truth) == str(pred) or int(count) <= 0:
                continue
            pairs.append({"truth": str(truth), "pred": str(pred), "count": int(count)})
    pairs.sort(key=lambda item: (-int(item["count"]), str(item["truth"]), str(item["pred"])))
    return pairs[: max(1, int(limit))]


def continuation_categories(raw: str) -> list[str]:
    return [category for category in selected_categories(raw, allowed=ALLOWED_CATEGORIES) if category in DEFAULT_CATEGORIES]


def build_continuation_decision_dataframe(
    *,
    dataset_df: pd.DataFrame,
    category: str,
    split_scheme: str,
    seed: int,
) -> pd.DataFrame:
    category = str(category or "").strip().lower()
    df = dataset_df[
        (dataset_df["category"].astype(str).str.lower() == category)
        & (pd.to_numeric(dataset_df["y"], errors="coerce").fillna(0).astype(int) == 1)
    ].copy()
    if "decision_type" in df.columns:
        df = df[df["decision_type"].astype(str).str.lower().isin(CONTINUATION_DECISION_TYPES)].copy()
    if df.empty:
        return df
    df["label"] = df["label"].astype(str).str.strip().str.lower()
    df = apply_split_scheme(df, split_scheme=split_scheme, seed=seed)
    sort_keys = [name for name in ("t0_utc", "decision_id") if name in df.columns]
    return df.sort_values(sort_keys).reset_index(drop=True)


def macro_f1_and_bal_acc(stats: dict[str, Any]) -> tuple[float, float]:
    f1_values: list[float] = []
    recalls: list[float] = []
    for payload in stats.values():
        support = int(payload.get("support") or 0)
        if support <= 0:
            continue
        precision = float(payload.get("precision") or 0.0)
        recall = float(payload.get("recall") or 0.0)
        denom = precision + recall
        f1_values.append(0.0 if denom <= 0 else (2.0 * precision * recall / denom))
        recalls.append(recall)
    macro_f1 = float(sum(f1_values) / max(1, len(f1_values)))
    bal_acc = float(sum(recalls) / max(1, len(recalls)))
    return round(macro_f1, 6), round(bal_acc, 6)


def decision_metrics(frame: pd.DataFrame, predictions: list[str], prob_df: pd.DataFrame, labels: list[str]) -> dict[str, Any]:
    y_true = [str(item) for item in frame["label"].astype(str).tolist()]
    stop_true = [label for label in y_true if label == STOP_TOKEN]
    stop_pred = [pred for label, pred in zip(y_true, predictions) if label == STOP_TOKEN]
    cont_true = [label for label in y_true if label != STOP_TOKEN]
    cont_pred = [pred for label, pred in zip(y_true, predictions) if label != STOP_TOKEN]
    stats = per_label_stats(y_true, predictions, labels)
    macro_f1, balanced_acc = macro_f1_and_bal_acc(stats)
    confusion = _fallback_top_confusion_pairs(
        {
            truth: {pred: sum(int(t == truth and p == pred) for t, p in zip(y_true, predictions)) for pred in labels}
            for truth in labels
        }
    )
    return {
        "rows": int(len(frame)),
        "acc_at_1": round(accuracy(y_true, predictions), 6),
        "recall_at_3": round(recall_at_k(prob_df, y_true, 3), 6),
        "stop_acc": round(accuracy(stop_true, stop_pred), 6) if stop_true else 0.0,
        "macro_f1": float(macro_f1),
        "balanced_accuracy": float(balanced_acc),
        "continuation_non_stop_accuracy": round(accuracy(cont_true, cont_pred), 6) if cont_true else 0.0,
        "per_label": stats,
        "confusion_top_pairs": confusion,
    }


def suffix_targets_from_decisions(decisions_df: pd.DataFrame, *, category: str) -> dict[int, list[str]]:
    df = decisions_df[
        (decisions_df["category"].astype(str).str.lower() == str(category or "").strip().lower())
        & (pd.to_numeric(decisions_df["y"], errors="coerce").fillna(0).astype(int) == 1)
    ].copy()
    if df.empty:
        return {}
    if "decision_type" in df.columns:
        df = df[df["decision_type"].astype(str).str.lower().isin(CONTINUATION_DECISION_TYPES)].copy()
    if "t0_utc" in df.columns:
        df["t0_utc"] = pd.to_datetime(df["t0_utc"], utc=True, format="mixed")
    df = df.sort_values(["episode_id", "t0_utc", "decision_id"]).reset_index(drop=True)
    decision_suffix_targets: dict[int, list[str]] = {}
    for _episode_id, rows in df.groupby("episode_id", sort=False):
        ordered = rows.to_dict(orient="records")
        for idx, row in enumerate(ordered):
            suffix = [str(item["label"]) for item in ordered[idx:] if str(item["label"]) != STOP_TOKEN]
            decision_suffix_targets[int(row["decision_id"])] = suffix
    return decision_suffix_targets


def first_error_position(predicted: list[str], target: list[str]) -> int:
    width = max(len(predicted), len(target))
    for idx in range(width):
        pred = predicted[idx] if idx < len(predicted) else STOP_TOKEN
        truth = target[idx] if idx < len(target) else STOP_TOKEN
        if str(pred) != str(truth):
            return idx + 1
    return 0


def suffix_metrics(
    frame: pd.DataFrame,
    *,
    category: str,
    rollout_fn,
    model_root,
    decision_suffix_targets: dict[int, list[str]],
) -> dict[str, Any]:
    if frame.empty:
        return {
            "rows": 0,
            "exact_full_suffix_match": 0.0,
            "prefix_match_rate": 0.0,
            "suffix_length_mae": 0.0,
            "first_error_position_mean": 0.0,
            "first_error_position_distribution": {},
        }
    exact: list[int] = []
    prefix: list[float] = []
    length_mae: list[float] = []
    first_error_positions: list[int] = []
    per_position_hits: dict[int, list[int]] = defaultdict(list)
    for row in frame.itertuples(index=False):
        target = list(decision_suffix_targets.get(int(getattr(row, "decision_id")), []))
        predicted = rollout_fn(category, row._asdict(), model_root=model_root)
        exact.append(sequence_exact_match(predicted, target))
        prefix.append(longest_common_prefix_rate(predicted, target))
        length_mae.append(abs(len(predicted) - len(target)))
        first_error_positions.append(first_error_position(predicted, target))
        width = max(len(predicted), len(target))
        for idx in range(width):
            pred = predicted[idx] if idx < len(predicted) else STOP_TOKEN
            truth = target[idx] if idx < len(target) else STOP_TOKEN
            per_position_hits[idx + 1].append(int(str(pred) == str(truth)))
    distribution = {
        str(position): int(count)
        for position, count in sorted(
            defaultdict(int, {pos: first_error_positions.count(pos) for pos in set(first_error_positions)}).items(),
            key=lambda item: int(item[0]),
        )
    }
    return {
        "rows": int(len(frame)),
        "exact_full_suffix_match": round(float(np.mean(exact)), 6),
        "prefix_match_rate": round(float(np.mean(prefix)), 6),
        "suffix_length_mae": round(float(np.mean(length_mae)), 6),
        "first_error_position_mean": round(float(np.mean(first_error_positions)), 6),
        "first_error_position_distribution": distribution,
        "per_position_accuracy": {
            str(position): round(float(np.mean(values)), 6)
            for position, values in sorted(per_position_hits.items())
        },
    }
