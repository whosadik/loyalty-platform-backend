from __future__ import annotations

import json
import math
import sys
from bisect import bisect_right
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None

try:
    import pandas as pd
except Exception:  # pragma: no cover
    pd = None

from django.contrib.auth import get_user_model

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from admin_tools.demo_history_seed import DEMO_USER_PREFIX, build_user_purchase_plan, generate_demo_user_specs, get_demo_users  # noqa: E402
from admin_tools.roadmap_teacher import _estimate_current_owned_tokens_before_anchor, build_teacher_policy  # noqa: E402
from ml.training.roadmap_initial_planner_common import STOP_TOKEN, build_model_pipeline, longest_common_prefix_rate, resolve_estimator_name, sequence_exact_match  # noqa: E402
from ml.training.roadmap_live_planner_common import (  # noqa: E402
    ALLOWED_CATEGORIES,
    accuracy,
    apply_split_scheme,
    build_live_decision_dataframe,
    confusion_matrix,
    ensure_dependencies,
    episode_targets_from_transitions,
    feature_spec_for_live_dataframe,
    load_live_dataset_bundle,
    load_model_artifact,
    per_label_stats,
    predict_bundle_probabilities,
    repo_root,
    split_frames,
)
from roadmap_app.content_features import profile_signature  # noqa: E402
from roadmap_app.ml_live_planner import rollout_live_plan  # noqa: E402
from transactions.models import TransactionItem  # noqa: E402
from users_app.models import CustomerProfile  # noqa: E402

User = get_user_model()

DEFAULT_CATEGORIES = ("haircare", "skincare", "fragrance")
DEFAULT_SPLIT_SCHEMES = ("time", "user")
DEFAULT_NUMERIC_DRIFT_FEATURES = (
    "steps_total",
    "next_step_index_current",
    "days_since_last_purchase_in_category",
    "prior_category_purchase_total",
    "prior_category_distinct_token_count",
    "tx_count_90d_category",
    "tx_amount_90d_category",
    "completed_steps_count",
    "recommended_steps_count",
    "fragrance_slot_coverage_count",
)
DEFAULT_CATEGORICAL_DRIFT_FEATURES = (
    "refresh_caller",
    "anchor_product_type",
    "current_next_product_type",
    "last1_product_type",
    "profile_skin_type",
    "profile_hair_type",
    "profile_scalp_type",
    "profile_fragrance_intensity_pref",
)
FSM_SHORTCUT_COLUMNS = (
    "current_next_product_type",
    "plan_product_types",
    "refresh_caller",
    "anchor_product_type",
    "last1_product_type",
)
PROFILE_PREFIX = "profile_"
PRIOR_STATE_COLUMNS = {
    "favorite_brand_in_category",
    "last1_product_type",
    "last2_product_type",
    "last3_product_type",
    "last4_product_type",
    "last5_product_type",
    "last1_category",
    "last2_category",
    "last3_category",
    "last4_category",
    "last5_category",
    "days_since_last_purchase_in_category",
    "prior_category_purchase_total",
    "prior_category_distinct_token_count",
    "fragrance_slot_coverage_count",
    "tx_count_90d_category",
    "tx_amount_90d_category",
}
CATALOG_COVERAGE_PREFIXES = ("anchor_",)
SCENARIO_HINT_PREFIXES = ("scenario_", "cohort_")


@dataclass(frozen=True)
class TeacherBaselineContext:
    profile_map: dict[int, Any]
    item_rows_by_key: dict[tuple[int, str], list[dict[str, Any]]]
    item_positions_by_key: dict[tuple[int, str], list[tuple[Any, int, int]]]


def _resolve_repo_path(raw_path: str | Path) -> Path:
    candidate = Path(str(raw_path)).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    return (REPO_ROOT / candidate).resolve()


def selected_categories(raw: str | None) -> list[str]:
    if not str(raw or "").strip():
        return list(DEFAULT_CATEGORIES)
    out: list[str] = []
    for item in str(raw).split(","):
        token = str(item or "").strip().lower()
        if token in ALLOWED_CATEGORIES and token not in out and token != "makeup":
            out.append(token)
    return out or list(DEFAULT_CATEGORIES)


def selected_split_schemes(raw: str | None) -> list[str]:
    if not str(raw or "").strip():
        return list(DEFAULT_SPLIT_SCHEMES)
    out: list[str] = []
    for item in str(raw).split(","):
        token = str(item or "").strip().lower()
        if token in DEFAULT_SPLIT_SCHEMES and token not in out:
            out.append(token)
    return out or list(DEFAULT_SPLIT_SCHEMES)


def top_confusion_pairs(matrix: dict[str, dict[str, int]], *, limit: int = 5) -> list[dict[str, Any]]:
    pairs: list[dict[str, Any]] = []
    for truth, row in matrix.items():
        for pred, count in row.items():
            if str(truth) == str(pred) or int(count) <= 0:
                continue
            pairs.append({"truth": str(truth), "pred": str(pred), "count": int(count)})
    pairs.sort(key=lambda item: (-int(item["count"]), str(item["truth"]), str(item["pred"])))
    return pairs[: max(1, int(limit))]


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


def distribution_tv_distance(left: dict[str, int], right: dict[str, int]) -> float:
    labels = sorted(set(left) | set(right))
    left_total = float(sum(int(value) for value in left.values()) or 1)
    right_total = float(sum(int(value) for value in right.values()) or 1)
    distance = 0.0
    for label in labels:
        distance += abs(float(int(left.get(label, 0)) / left_total) - float(int(right.get(label, 0)) / right_total))
    return round(distance / 2.0, 6)


def ranked_labels_from_order(primary: str, base_order: list[str], labels: list[str], *, k: int = 3) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for label in [primary] + list(base_order):
        token = str(label or "").strip().lower()
        if token not in labels or token in seen:
            continue
        seen.add(token)
        out.append(token)
        if len(out) >= max(1, int(k)):
            break
    for label in labels:
        if label in seen:
            continue
        out.append(label)
        if len(out) >= max(1, int(k)):
            break
    return out


def _topk_recall_from_ranked(y_true: list[str], ranked: list[list[str]], *, k: int = 3) -> float:
    if not y_true:
        return 0.0
    hits = 0
    width = max(1, int(k))
    for truth, labels in zip(y_true, ranked):
        hits += int(str(truth) in {str(item) for item in labels[:width]})
    return round(float(hits / max(1, len(y_true))), 6)


def classification_summary(
    *,
    y_true: list[str],
    y_pred: list[str],
    labels: list[str],
    ranked_predictions: list[list[str]] | None = None,
) -> dict[str, Any]:
    stats = per_label_stats(y_true, y_pred, labels)
    matrix = confusion_matrix(y_true, y_pred, labels)
    macro_f1, balanced_acc = macro_f1_and_bal_acc(stats)
    stop_idx = [idx for idx, label in enumerate(y_true) if str(label) == STOP_TOKEN]
    stop_true = [y_true[idx] for idx in stop_idx]
    stop_pred = [y_pred[idx] for idx in stop_idx]
    return {
        "rows": int(len(y_true)),
        "acc_at_1": round(accuracy(y_true, y_pred), 6),
        "recall_at_3": _topk_recall_from_ranked(y_true, ranked_predictions or [[pred] for pred in y_pred], k=3),
        "macro_f1": macro_f1,
        "balanced_accuracy": balanced_acc,
        "stop_accuracy": round(accuracy(stop_true, stop_pred), 6) if stop_true else 0.0,
        "per_label": stats,
        "confusion_top_pairs": top_confusion_pairs(matrix),
    }


def _series_to_counts(series: "pd.Series") -> dict[str, int]:
    if series.empty:
        return {}
    return {str(key): int(value) for key, value in sorted(series.astype(str).value_counts().to_dict().items())}


def load_initial_first_step_rows(
    *,
    data_dir: str | Path,
    categories: list[str],
    split_scheme: str,
    seed: int,
) -> tuple["pd.DataFrame", dict[str, Any]]:
    ensure_dependencies()
    dataset_df, metadata, _ = load_live_dataset_bundle(data_dir)
    frames: list["pd.DataFrame"] = []
    for category in categories:
        frame = build_live_decision_dataframe(
            dataset_df=dataset_df,
            category=category,
            split_scheme=split_scheme,
            seed=seed,
            continuation_only=False,
        )
        if not frame.empty:
            frames.append(frame)
    if not frames:
        return pd.DataFrame(), metadata
    combined = pd.concat(frames, ignore_index=True)
    combined["label"] = combined["label"].astype(str).str.strip().str.lower()
    combined["category"] = combined["category"].astype(str).str.strip().str.lower()
    return combined.sort_values(["t0_utc", "decision_id"]).reset_index(drop=True), metadata


def first_step_dataset_view(
    *,
    data_dir: str | Path,
    categories: list[str],
    split_schemes: list[str],
    seed: int,
) -> dict[str, Any]:
    ensure_dependencies()
    report: dict[str, Any] = {
        "dataset_dir": str(_resolve_repo_path(data_dir)),
        "categories": list(categories),
        "split_schemes": {},
    }
    for scheme in split_schemes:
        decisions_df, metadata = load_initial_first_step_rows(
            data_dir=data_dir,
            categories=categories,
            split_scheme=scheme,
            seed=seed,
        )
        if decisions_df.empty:
            report["split_schemes"][scheme] = {"rows_total": 0, "decision_points_total": 0, "categories": {}}
            continue
        scheme_payload: dict[str, Any] = {
            "rows_total": int(len(decisions_df)),
            "decision_points_total": int(decisions_df["decision_id"].nunique()),
            "users_total": int(decisions_df["user_id"].nunique()),
            "stop_share": round(float((decisions_df["label"] == STOP_TOKEN).mean()), 6),
            "split_counts": {str(key): int(value) for key, value in sorted(decisions_df["eval_split"].value_counts().to_dict().items())},
            "split_user_counts": {
                split_name: int(decisions_df.loc[decisions_df["eval_split"] == split_name, "user_id"].nunique())
                for split_name in ("train", "val", "test")
            },
            "categories": {},
        }
        for category in categories:
            category_df = decisions_df[decisions_df["category"] == category].copy()
            labels = _series_to_counts(category_df["label"])
            non_stop_labels = {key: value for key, value in labels.items() if key != STOP_TOKEN}
            scheme_payload["categories"][category] = {
                "rows_total": int(len(category_df)),
                "users_total": int(category_df["user_id"].nunique()),
                "stop_share": round(float((category_df["label"] == STOP_TOKEN).mean()), 6) if not category_df.empty else 0.0,
                "positive_labels": non_stop_labels,
                "class_balance": {
                    key: {
                        "count": int(value),
                        "share": round(float(value) / float(len(category_df) or 1), 6),
                    }
                    for key, value in labels.items()
                },
                "label_support_by_split": {
                    split_name: _series_to_counts(category_df.loc[category_df["eval_split"] == split_name, "label"])
                    for split_name in ("train", "val", "test")
                },
            }
        report["split_schemes"][scheme] = scheme_payload
        report["candidate_space"] = dict(metadata.get("candidate_types_by_category") or {})
    return report


def _numeric_drift(train_series: "pd.Series", eval_series: "pd.Series") -> float:
    train_values = pd.to_numeric(train_series, errors="coerce").dropna()
    eval_values = pd.to_numeric(eval_series, errors="coerce").dropna()
    if train_values.empty and eval_values.empty:
        return 0.0
    train_mean = float(train_values.mean()) if not train_values.empty else 0.0
    eval_mean = float(eval_values.mean()) if not eval_values.empty else 0.0
    train_std = float(train_values.std(ddof=0)) if len(train_values) > 1 else 0.0
    if train_std <= 1e-9:
        return round(abs(eval_mean - train_mean), 6)
    return round(abs(eval_mean - train_mean) / train_std, 6)


def _categorical_drift(train_series: "pd.Series", eval_series: "pd.Series") -> float:
    train_counts = _series_to_counts(train_series.fillna("__none__"))
    eval_counts = _series_to_counts(eval_series.fillna("__none__"))
    return distribution_tv_distance(train_counts, eval_counts)


def _top_feature_drift(
    *,
    category_df: "pd.DataFrame",
    feature_columns: list[str],
    split_name: str,
    limit: int = 10,
) -> list[dict[str, Any]]:
    train_df = category_df[category_df["eval_split"] == "train"].copy()
    eval_df = category_df[category_df["eval_split"] == split_name].copy()
    out: list[dict[str, Any]] = []
    for feature in feature_columns:
        if feature not in category_df.columns:
            continue
        series = category_df[feature]
        if pd.api.types.is_numeric_dtype(series.dtype) or pd.api.types.is_bool_dtype(series.dtype):
            score = _numeric_drift(train_df[feature], eval_df[feature])
            feature_type = "numeric"
        else:
            score = _categorical_drift(train_df[feature], eval_df[feature])
            feature_type = "categorical"
        out.append({"feature": feature, "drift_score": float(score), "feature_type": feature_type})
    out.sort(key=lambda item: (-float(item["drift_score"]), str(item["feature"])))
    return out[: max(1, int(limit))]


def _label_drift_summary(category_df: "pd.DataFrame") -> dict[str, Any]:
    train_counts = _series_to_counts(category_df.loc[category_df["eval_split"] == "train", "label"])
    val_counts = _series_to_counts(category_df.loc[category_df["eval_split"] == "val", "label"])
    test_counts = _series_to_counts(category_df.loc[category_df["eval_split"] == "test", "label"])
    return {
        "train_distribution": train_counts,
        "val_distribution": val_counts,
        "test_distribution": test_counts,
        "train_val_tv": distribution_tv_distance(train_counts, val_counts),
        "train_test_tv": distribution_tv_distance(train_counts, test_counts),
    }


def _month_token(ts: Any) -> str:
    if not isinstance(ts, pd.Timestamp):
        ts = pd.Timestamp(ts)
    return ts.tz_convert("UTC").strftime("%Y-%m")


def _anchor_month_distribution(decisions_df: "pd.DataFrame") -> dict[str, Any]:
    out: dict[str, Any] = {}
    for (split_name, category), frame in decisions_df.groupby(["eval_split", "category"]):
        months = frame["t0_utc"].map(_month_token).value_counts().sort_index().to_dict()
        out[f"{split_name}:{category}"] = {str(key): int(value) for key, value in months.items()}
    return out


def _action_distribution_over_time(decisions_df: "pd.DataFrame") -> dict[str, Any]:
    out: dict[str, Any] = {}
    non_stop = decisions_df[decisions_df["label"] != STOP_TOKEN].copy()
    for category, frame in non_stop.groupby("category"):
        month_rows: dict[str, dict[str, int]] = {}
        frame["_month"] = frame["t0_utc"].map(_month_token)
        for month, month_df in frame.groupby("_month"):
            month_rows[str(month)] = _series_to_counts(month_df["label"])
        out[str(category)] = month_rows
    return out


def _late_only_labels(decisions_df: "pd.DataFrame") -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    for category, frame in decisions_df.groupby("category"):
        findings: list[dict[str, Any]] = []
        for label, label_df in frame.groupby("label"):
            split_support = {
                split_name: int(label_df.loc[label_df["eval_split"] == split_name].shape[0])
                for split_name in ("train", "val", "test")
            }
            if sum(split_support.values()) <= 0:
                continue
            first_month = _month_token(label_df["t0_utc"].min())
            last_month = _month_token(label_df["t0_utc"].max())
            late_only = split_support["train"] == 0 and (split_support["val"] > 0 or split_support["test"] > 0)
            early_only = split_support["train"] > 0 and split_support["val"] == 0 and split_support["test"] == 0
            if not (late_only or early_only):
                continue
            findings.append(
                {
                    "label": str(label),
                    "train_support": split_support["train"],
                    "val_support": split_support["val"],
                    "test_support": split_support["test"],
                    "first_month": first_month,
                    "last_month": last_month,
                    "pattern": "late_only" if late_only else "early_only",
                }
            )
        out[str(category)] = sorted(findings, key=lambda item: (item["pattern"], item["label"]))
    return out


def _scenario_family_timeline(*, seed: int, users: int, max_transactions_per_user: int, prefix: str) -> dict[str, Any]:
    demo_users = get_demo_users(prefix=prefix, seed=seed, limit=users)
    if not demo_users:
        return {"months": {}, "note": "no demo users found"}
    specs = generate_demo_user_specs(total_users=users, seed=seed, prefix=prefix)
    spec_by_username = {spec.username: spec for spec in specs}
    actual_rows = list(User.objects.filter(id__in=[user.id for user in demo_users]).values_list("id", "username"))
    actual_users = {int(user_id): str(username) for user_id, username in actual_rows}
    transactions = list(
        TransactionItem.objects.filter(transaction__user_id__in=actual_users)
        .order_by("transaction__user_id", "transaction__created_at", "transaction_id", "id")
        .values("transaction__user_id", "transaction__created_at", "transaction__pricing_meta", "product__category")
    )
    rows_by_user: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in transactions:
        rows_by_user[int(row["transaction__user_id"])].append(dict(row))
    months: dict[str, Counter[str]] = defaultdict(Counter)
    details: dict[str, Counter[str]] = defaultdict(Counter)
    for user_id, rows in rows_by_user.items():
        username = actual_users.get(int(user_id))
        if not username:
            continue
        spec = spec_by_username.get(username)
        planned = build_user_purchase_plan(spec, max_transactions=max_transactions_per_user)["planned_transactions"] if spec else []
        for index, row in enumerate(rows):
            created_at = pd.Timestamp(row["transaction__created_at"])
            if created_at.tzinfo is None:
                created_at = created_at.tz_localize("UTC")
            else:
                created_at = created_at.tz_convert("UTC")
            month = _month_token(created_at)
            role = ""
            cohort = ""
            if index < len(planned):
                role = str(planned[index].get("scenario_role") or planned[index].get("mode") or "planned")
                cohort = str(spec.cohort if spec else "planned")
            else:
                pricing_meta = row.get("transaction__pricing_meta") or {}
                scenario = pricing_meta.get("scenario") or {}
                role = str(scenario.get("scenario_role") or scenario.get("required_pair") or pricing_meta.get("mode") or "extra")
                cohort = str(scenario.get("cohort") or "extra")
            family = role or "unknown"
            months[month][family] += 1
            details[month][f"{cohort}:{family}"] += 1
    return {
        "months": {
            month: {str(key): int(value) for key, value in sorted(counter.items())}
            for month, counter in sorted(months.items())
        },
        "cohort_role_months": {
            month: {str(key): int(value) for key, value in sorted(counter.items())}
            for month, counter in sorted(details.items())
        },
    }


def build_live_initial_diagnostics_report(
    *,
    data_dir: str | Path,
    categories: list[str],
    split_schemes: list[str],
    seed: int,
    scenario_seed: int,
    scenario_users: int,
    scenario_max_transactions_per_user: int,
    scenario_prefix: str = DEMO_USER_PREFIX,
    runtime_shadow_time_report: str | Path | None = None,
    runtime_shadow_user_report: str | Path | None = None,
) -> dict[str, Any]:
    ensure_dependencies()
    data_dir_path = _resolve_repo_path(data_dir)
    time_df, metadata = load_initial_first_step_rows(
        data_dir=data_dir_path,
        categories=categories,
        split_scheme="time",
        seed=seed,
    )
    feature_columns, _categorical_features, _numeric_features = feature_spec_for_live_dataframe(time_df)
    diagnostics: dict[str, Any] = {
        "dataset_view": first_step_dataset_view(
            data_dir=data_dir_path,
            categories=categories,
            split_schemes=split_schemes,
            seed=seed,
        ),
        "time_split_gap": {
            "metadata_version": str(metadata.get("version") or ""),
            "anchor_month_distribution": _anchor_month_distribution(time_df),
            "action_distribution_over_time": _action_distribution_over_time(time_df),
            "late_or_early_only_labels": _late_only_labels(time_df),
            "scenario_family_over_time": _scenario_family_timeline(
                seed=scenario_seed,
                users=scenario_users,
                max_transactions_per_user=scenario_max_transactions_per_user,
                prefix=scenario_prefix,
            ),
            "feature_drift": {},
            "label_drift": {},
        },
    }
    chosen_drift_features = [
        feature
        for feature in list(DEFAULT_NUMERIC_DRIFT_FEATURES) + list(DEFAULT_CATEGORICAL_DRIFT_FEATURES)
        if feature in set(feature_columns)
    ]
    if not chosen_drift_features:
        chosen_drift_features = list(feature_columns[:10])
    for category in categories:
        category_df = time_df[time_df["category"] == category].copy()
        diagnostics["time_split_gap"]["feature_drift"][category] = {
            "train_vs_val": _top_feature_drift(category_df=category_df, feature_columns=chosen_drift_features, split_name="val"),
            "train_vs_test": _top_feature_drift(category_df=category_df, feature_columns=chosen_drift_features, split_name="test"),
        }
        diagnostics["time_split_gap"]["label_drift"][category] = _label_drift_summary(category_df)
    for name, raw_path in {
        "runtime_shadow_time_reference": runtime_shadow_time_report,
        "runtime_shadow_user_reference": runtime_shadow_user_report,
    }.items():
        if raw_path:
            path = _resolve_repo_path(raw_path)
            if path.exists():
                diagnostics[name] = json.loads(path.read_text(encoding="utf-8"))
    diagnostics["notes"] = [
        "first-step view uses the current live initial dataset positives only; one row equals one initial decision point",
        "time-gap diagnostics are computed on the time split because that is where the performance collapse was observed",
        "scenario family over time is reconstructed from deterministic seeding plans plus stored transaction metadata; live-tail checkout transactions inherit planned role by ordinal alignment",
    ]
    return diagnostics


def _majority_order(train_df: "pd.DataFrame", labels: list[str]) -> list[str]:
    counts = Counter(str(item) for item in train_df["label"].astype(str).tolist())
    ordered = [label for label, _count in sorted(counts.items(), key=lambda item: (-item[1], item[0])) if label in labels]
    for label in labels:
        if label not in ordered:
            ordered.append(label)
    return ordered


def _signature_map(train_df: "pd.DataFrame", keys: list[str]) -> dict[tuple[Any, ...], list[str]]:
    grouped: dict[tuple[Any, ...], Counter[str]] = defaultdict(Counter)
    for row in train_df[keys + ["label"]].itertuples(index=False):
        key = tuple(getattr(row, column) for column in keys)
        grouped[key][str(getattr(row, "label"))] += 1
    return {
        key: [label for label, _value in sorted(counter.items(), key=lambda item: (-item[1], item[0]))]
        for key, counter in grouped.items()
    }


def _evaluate_order_baseline(
    *,
    eval_df: "pd.DataFrame",
    labels: list[str],
    order_lookup: dict[tuple[Any, ...], list[str]] | None,
    key_columns: list[str],
    fallback_order: list[str],
) -> dict[str, Any]:
    y_true = [str(item) for item in eval_df["label"].astype(str).tolist()]
    y_pred: list[str] = []
    ranked: list[list[str]] = []
    covered = 0
    for row in eval_df.itertuples(index=False):
        order = fallback_order
        if order_lookup is not None:
            key = tuple(getattr(row, column) for column in key_columns)
            found = order_lookup.get(key)
            if found:
                covered += 1
                order = [item for item in found if item in labels] + [item for item in fallback_order if item not in found]
        top_label = str(order[0] if order else STOP_TOKEN)
        y_pred.append(top_label)
        ranked.append(ranked_labels_from_order(top_label, order, labels, k=3))
    summary = classification_summary(y_true=y_true, y_pred=y_pred, labels=labels, ranked_predictions=ranked)
    summary["coverage"] = round(float(covered / max(1, len(eval_df))), 6) if order_lookup is not None else 1.0
    return summary


def _prepare_teacher_context(decisions_df: "pd.DataFrame") -> TeacherBaselineContext:
    user_ids = sorted({int(value) for value in decisions_df["user_id"].tolist()})
    categories = sorted({str(value) for value in decisions_df["category"].tolist()})
    profiles = CustomerProfile.objects.filter(user_id__in=user_ids)
    profile_map = {int(profile.user_id): profile for profile in profiles}
    item_rows = list(
        TransactionItem.objects.filter(transaction__user_id__in=user_ids, product__category__in=categories)
        .order_by("transaction__user_id", "product__category", "transaction__created_at", "transaction_id", "id")
        .values(
            "transaction__user_id",
            "transaction__created_at",
            "transaction_id",
            "id",
            "product__category",
            "product_id",
            "product__product_type",
            "product__brand",
            "product__price",
            "product__concerns",
            "product__actives",
            "product__flags",
            "product__supported_skin_types",
            "product__attrs",
            "product__ingredients_inci",
            "product__raw_meta",
        )
    )
    rows_by_key: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    positions_by_key: dict[tuple[int, str], list[tuple[Any, int, int]]] = defaultdict(list)
    for row in item_rows:
        key = (int(row["transaction__user_id"]), str(row["product__category"] or "").strip().lower())
        created_at = pd.Timestamp(row["transaction__created_at"]).tz_convert("UTC")
        rows_by_key[key].append(
            {
                "position": (created_at, int(row["transaction_id"]), int(row["id"])),
                "ts": created_at,
                "history_token": str(row["product__product_type"] or "").strip().lower(),
                "seed_product": {
                    "id": int(row["product_id"]),
                    "category": str(row["product__category"] or "").strip().lower(),
                    "product_type": str(row["product__product_type"] or "").strip().lower(),
                    "brand": str(row["product__brand"] or "").strip().lower(),
                    "price": row["product__price"],
                    "concerns": row.get("product__concerns") or [],
                    "actives": row.get("product__actives") or [],
                    "flags": row.get("product__flags") or [],
                    "supported_skin_types": row.get("product__supported_skin_types") or [],
                    "attrs": row.get("product__attrs") or {},
                    "ingredients_inci": row.get("product__ingredients_inci") or "",
                    "raw_meta": row.get("product__raw_meta") or {},
                },
            }
        )
        positions_by_key[key].append((created_at, int(row["transaction_id"]), int(row["id"])))
    return TeacherBaselineContext(profile_map=profile_map, item_rows_by_key=rows_by_key, item_positions_by_key=positions_by_key)


def _teacher_first_step_for_row(row: Any, *, context: TeacherBaselineContext) -> tuple[str, bool]:
    user_id = int(getattr(row, "user_id"))
    category = str(getattr(row, "category") or "").strip().lower()
    t0_utc = pd.Timestamp(getattr(row, "t0_utc")).tz_convert("UTC")
    key = (user_id, category)
    positions = context.item_positions_by_key.get(key) or []
    if not positions:
        return STOP_TOKEN, False
    pivot = bisect_right(positions, (t0_utc, math.inf, math.inf)) - 1
    if pivot < 0:
        return STOP_TOKEN, False
    anchor_row = context.item_rows_by_key[key][pivot]
    prior_items = context.item_rows_by_key[key][:pivot]
    profile = context.profile_map.get(user_id)
    teacher = build_teacher_policy(
        category=category,
        seed_product=dict(anchor_row.get("seed_product") or {}),
        profile_sig=profile_signature(profile),
        prior_state={
            "current_owned_tokens_before_anchor": _estimate_current_owned_tokens_before_anchor(
                category=category,
                prior_items=prior_items,
                anchor_ts=t0_utc,
            )
        },
    )
    sequence = list(teacher.get("sequence") or [])
    if not sequence:
        return STOP_TOKEN, True
    return str(sequence[0]), True


def _evaluate_teacher_baseline(
    *,
    eval_df: "pd.DataFrame",
    labels: list[str],
    fallback_order: list[str],
    context: TeacherBaselineContext,
) -> dict[str, Any]:
    y_true = [str(item) for item in eval_df["label"].astype(str).tolist()]
    y_pred: list[str] = []
    ranked: list[list[str]] = []
    covered = 0
    for row in eval_df.itertuples(index=False):
        predicted, ok = _teacher_first_step_for_row(row, context=context)
        covered += int(ok)
        predicted = predicted if predicted in labels else STOP_TOKEN
        y_pred.append(predicted)
        ranked.append(ranked_labels_from_order(predicted, fallback_order, labels, k=3))
    summary = classification_summary(y_true=y_true, y_pred=y_pred, labels=labels, ranked_predictions=ranked)
    summary["coverage"] = round(float(covered / max(1, len(eval_df))), 6)
    return summary


def _evaluate_live_model_baseline(
    *,
    eval_df: "pd.DataFrame",
    model_root: Path,
    category: str,
) -> dict[str, Any]:
    bundle = load_model_artifact(category, model_root=model_root)
    labels = [str(item) for item in list(bundle.get("action_space") or [])]
    prob_df = predict_bundle_probabilities(bundle, eval_df)
    y_true = [str(item) for item in eval_df["label"].astype(str).tolist()]
    y_pred = prob_df.idxmax(axis=1).astype(str).tolist() if not prob_df.empty else []
    ranked = []
    if not prob_df.empty:
        for _idx, row in prob_df.iterrows():
            ranked.append([str(label) for label in list(row.sort_values(ascending=False).index[:3])])
    summary = classification_summary(y_true=y_true, y_pred=y_pred, labels=labels, ranked_predictions=ranked)
    summary["coverage"] = 1.0
    return summary


def _feature_groups(feature_columns: list[str]) -> dict[str, list[str]]:
    profile = sorted([column for column in feature_columns if column.startswith(PROFILE_PREFIX)])
    prior_state = sorted([column for column in feature_columns if column in PRIOR_STATE_COLUMNS])
    catalog = sorted([column for column in feature_columns if any(column.startswith(prefix) for prefix in CATALOG_COVERAGE_PREFIXES)])
    scenario = sorted(
        [
            column
            for column in feature_columns
            if any(column.startswith(prefix) for prefix in SCENARIO_HINT_PREFIXES) or "scenario" in column or "cohort" in column
        ]
    )
    return {
        "full_features": [],
        "without_profile_features": profile,
        "without_prior_state_history": prior_state,
        "without_catalog_coverage_features": catalog,
        "without_seeded_scenario_family_indicators": scenario,
    }


def _train_ablation_bundle(
    *,
    decisions_df: "pd.DataFrame",
    excluded_features: list[str],
    estimator_name: str,
    seed: int,
) -> dict[str, Any]:
    feature_columns, categorical_features, numeric_features = feature_spec_for_live_dataframe(decisions_df)
    included_features = [column for column in feature_columns if column not in set(excluded_features)]
    categorical = [column for column in categorical_features if column in included_features]
    numeric = [column for column in numeric_features if column in included_features]
    split_map = split_frames(decisions_df)
    train_df = split_map["train"].sort_values(["t0_utc", "decision_id"]).reset_index(drop=True)
    labels = train_df["label"].astype(str).tolist()
    unique_labels = sorted(set(labels))
    if len(unique_labels) <= 1:
        top = unique_labels[0] if unique_labels else STOP_TOKEN

        class ConstantModel:
            classes_ = np.asarray([top], dtype=object)

            def predict_proba(self, X):
                return np.ones((len(X), 1), dtype=float)

        return {
            "model": ConstantModel(),
            "feature_columns": included_features,
            "categorical_features": categorical,
            "numeric_features": numeric,
            "action_space": sorted(set(labels) | {STOP_TOKEN}),
        }
    model = build_model_pipeline(
        estimator_name=resolve_estimator_name(estimator_name),
        n_classes=len(unique_labels),
        seed=seed,
        categorical_features=categorical,
        numeric_features=numeric,
    )
    model.fit(train_df[included_features], train_df["label"])
    return {
        "model": model,
        "feature_columns": included_features,
        "categorical_features": categorical,
        "numeric_features": numeric,
        "action_space": sorted(set(labels) | {STOP_TOKEN}),
    }


def _evaluate_bundle_on_frame(bundle: dict[str, Any], frame: "pd.DataFrame") -> dict[str, Any]:
    labels = [str(item) for item in list(bundle.get("action_space") or [])]
    if frame.empty:
        return classification_summary(y_true=[], y_pred=[], labels=labels, ranked_predictions=[])
    prob_df = predict_bundle_probabilities(bundle, frame)
    y_true = [str(item) for item in frame["label"].astype(str).tolist()]
    y_pred = prob_df.idxmax(axis=1).astype(str).tolist() if not prob_df.empty else []
    ranked = []
    if not prob_df.empty:
        for _idx, row in prob_df.iterrows():
            ranked.append([str(label) for label in list(row.sort_values(ascending=False).index[:3])])
    return classification_summary(y_true=y_true, y_pred=y_pred, labels=labels, ranked_predictions=ranked)


def _sequence_metrics_for_live_initial(
    *,
    frame: "pd.DataFrame",
    category: str,
    model_root: Path,
    episode_targets: dict[int, list[str]],
) -> dict[str, Any]:
    if frame.empty:
        return {"rows": 0, "exact_full_plan_match": 0.0, "prefix_match_rate": 0.0, "length_mae": 0.0}
    exact: list[int] = []
    prefix: list[float] = []
    length_mae: list[float] = []
    for row in frame.itertuples(index=False):
        target = list(episode_targets.get(int(getattr(row, "episode_id")), []))
        predicted = rollout_live_plan(category, row._asdict(), model_root=model_root)
        exact.append(sequence_exact_match(predicted, target))
        prefix.append(longest_common_prefix_rate(predicted, target))
        length_mae.append(abs(len(predicted) - len(target)))
    return {
        "rows": int(len(frame)),
        "exact_full_plan_match": round(float(np.mean(exact)), 6) if exact else 0.0,
        "prefix_match_rate": round(float(np.mean(prefix)), 6) if prefix else 0.0,
        "length_mae": round(float(np.mean(length_mae)), 6) if length_mae else 0.0,
    }


def build_live_initial_baseline_compare_report(
    *,
    data_dir: str | Path,
    transitions_dir: str | Path,
    model_root: str | Path,
    categories: list[str],
    split_schemes: list[str],
    seed: int,
    estimator_name: str = "lightgbm",
    runtime_shadow_time_report: str | Path | None = None,
    runtime_shadow_user_report: str | Path | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    ensure_dependencies()
    data_dir_path = _resolve_repo_path(data_dir)
    transitions_dir_path = _resolve_repo_path(transitions_dir)
    transitions_df, _transitions_meta, _ = load_live_dataset_bundle(transitions_dir_path)
    model_root_path = _resolve_repo_path(model_root)
    report: dict[str, Any] = {
        "dataset_dir": str(data_dir_path),
        "transitions_dir": str(transitions_dir_path),
        "model_root": str(model_root_path.resolve()),
        "categories": {},
    }
    ablation_report: dict[str, Any] = {
        "dataset_dir": report["dataset_dir"],
        "model_root": report["model_root"],
        "split_schemes": {},
    }
    shadow_refs: dict[str, Any] = {}
    for key, raw_path in {"time": runtime_shadow_time_report, "user": runtime_shadow_user_report}.items():
        if raw_path:
            path = _resolve_repo_path(raw_path)
            if path.exists():
                shadow_refs[key] = json.loads(path.read_text(encoding="utf-8"))

    for scheme in split_schemes:
        decisions_df, _meta = load_initial_first_step_rows(
            data_dir=data_dir_path,
            categories=categories,
            split_scheme=scheme,
            seed=seed,
        )
        teacher_context = _prepare_teacher_context(decisions_df)
        scheme_root = model_root_path / scheme
        ablation_report["split_schemes"][scheme] = {"categories": {}}
        for category in categories:
            category_df = decisions_df[decisions_df["category"] == category].copy()
            if category_df.empty:
                continue
            split_map = split_frames(category_df)
            train_df = split_map["train"]
            labels = list(load_model_artifact(category, model_root=scheme_root).get("action_space") or [])
            fallback_order = _majority_order(train_df, labels)
            signature_cols = [column for column in FSM_SHORTCUT_COLUMNS if column in category_df.columns]
            signature_lookup = _signature_map(train_df, signature_cols) if signature_cols else {}
            category_payload: dict[str, Any] = {"rows": {name: int(len(frame)) for name, frame in split_map.items()}, "baselines": {}, "rollout_vs_first_step": {}}
            for split_name in ("val", "test"):
                frame = split_map[split_name].sort_values(["t0_utc", "decision_id"]).reset_index(drop=True)
                category_payload["baselines"].setdefault(split_name, {})
                category_payload["baselines"][split_name]["most_popular_next_action_by_category"] = _evaluate_order_baseline(
                    eval_df=frame, labels=labels, order_lookup=None, key_columns=[], fallback_order=fallback_order
                )
                category_payload["baselines"][split_name]["fsm_signature_shortcut"] = _evaluate_order_baseline(
                    eval_df=frame, labels=labels, order_lookup=signature_lookup, key_columns=signature_cols, fallback_order=fallback_order
                )
                category_payload["baselines"][split_name]["teacher_policy_first_step"] = _evaluate_teacher_baseline(
                    eval_df=frame, labels=labels, fallback_order=fallback_order, context=teacher_context
                )
                category_payload["baselines"][split_name]["current_live_initial_model"] = _evaluate_live_model_baseline(
                    eval_df=frame, model_root=scheme_root, category=category
                )
            episode_targets, _decision_suffix = episode_targets_from_transitions(transitions_df, category=category)
            for split_name in ("val", "test"):
                frame = split_map[split_name].sort_values(["t0_utc", "decision_id"]).reset_index(drop=True)
                first_step = category_payload["baselines"][split_name]["current_live_initial_model"]
                full_plan = _sequence_metrics_for_live_initial(frame=frame, category=category, model_root=scheme_root, episode_targets=episode_targets)
                category_payload["rollout_vs_first_step"][split_name] = {
                    "first_step": {
                        "acc_at_1": float(first_step["acc_at_1"]),
                        "recall_at_3": float(first_step["recall_at_3"]),
                        "macro_f1": float(first_step["macro_f1"]),
                        "balanced_accuracy": float(first_step["balanced_accuracy"]),
                        "stop_accuracy": float(first_step["stop_accuracy"]),
                    },
                    "full_rollout": full_plan,
                }
            report["categories"].setdefault(scheme, {})[category] = category_payload

            feature_columns, _categorical, _numeric = feature_spec_for_live_dataframe(category_df)
            groups = _feature_groups(feature_columns)
            ablation_payload: dict[str, Any] = {"feature_groups": {key: list(value) for key, value in groups.items()}, "test_metrics": {}}
            for name, excluded_features in groups.items():
                if name != "full_features" and not excluded_features:
                    ablation_payload["test_metrics"][name] = {"status": "skipped", "reason": "feature_group_not_present"}
                    continue
                bundle = _train_ablation_bundle(decisions_df=category_df, excluded_features=excluded_features, estimator_name=estimator_name, seed=seed)
                metrics = _evaluate_bundle_on_frame(bundle, split_map["test"].sort_values(["t0_utc", "decision_id"]).reset_index(drop=True))
                ablation_payload["test_metrics"][name] = metrics
            ablation_report["split_schemes"][scheme]["categories"][category] = ablation_payload

    report["shadow_reference"] = shadow_refs
    report["notes"] = [
        "most_popular_next_action_by_category is a pure train-majority baseline per category and split",
        "fsm_signature_shortcut uses current_next_product_type + plan_product_types + refresh_caller + anchor_product_type + last1_product_type as a trivial runtime-state signature baseline",
        "teacher_policy_first_step reconstructs the closest prior purchase anchor per decision and runs build_teacher_policy() offline; coverage is reported explicitly",
        "rollout_vs_first_step isolates whether the current live initial model knows the first step separately from whether it can roll out a full plan",
    ]
    return report, ablation_report


def diagnostics_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Roadmap Live Initial Diagnostics",
        "",
        f"- dataset_dir: `{report['dataset_view']['dataset_dir']}`",
        "",
        "## First-Step Dataset View",
    ]
    for scheme, payload in sorted((report["dataset_view"].get("split_schemes") or {}).items()):
        lines.extend(
            [
                f"### {scheme}",
                f"- rows_total: **{payload['rows_total']}**",
                f"- decision_points_total: **{payload['decision_points_total']}**",
                f"- users_total: **{payload['users_total']}**",
                f"- stop_share: **{payload['stop_share']:.6f}**",
            ]
        )
        for category, category_payload in sorted((payload.get("categories") or {}).items()):
            labels = ", ".join(f"{key}={value}" for key, value in sorted((category_payload.get("positive_labels") or {}).items()))
            lines.append(f"- {category}: stop_share=`{category_payload['stop_share']:.4f}`, positives=`{labels or 'none'}`")
        lines.append("")

    time_gap = report.get("time_split_gap") or {}
    lines.extend(["## Time Split Gap", ""])
    for category, drift in sorted((time_gap.get("label_drift") or {}).items()):
        lines.append(f"- {category}: train_test_label_tv=`{drift['train_test_tv']:.4f}`, train_val_label_tv=`{drift['train_val_tv']:.4f}`")
    lines.extend(["", "## Late/Early Label Flags"])
    for category, findings in sorted((time_gap.get("late_or_early_only_labels") or {}).items()):
        if not findings:
            lines.append(f"- {category}: none")
            continue
        summary = ", ".join(f"{item['label']}({item['pattern']})" for item in findings)
        lines.append(f"- {category}: {summary}")
    lines.extend(["", "## Scenario Family Over Time"])
    for month, payload in sorted(((time_gap.get("scenario_family_over_time") or {}).get("months") or {}).items()):
        summary = ", ".join(f"{key}={value}" for key, value in sorted(payload.items()))
        lines.append(f"- {month}: {summary}")
    if report.get("runtime_shadow_user_reference"):
        lines.extend(["", "## Existing User-Shadow Reference"])
        for category, payload in sorted(((report["runtime_shadow_user_reference"].get("initial_shadow") or {}).items())):
            lines.append(
                f"- {category}: exact=`{float(payload.get('exact_match_rate') or 0.0):.4f}`, first_step=`{float(payload.get('first_step_match_rate') or 0.0):.4f}`, reason=`{payload.get('top_divergence_reason')}`"
            )
    return "\n".join(lines).rstrip() + "\n"


def baseline_compare_markdown(report: dict[str, Any], ablation_report: dict[str, Any]) -> tuple[str, str]:
    lines = [
        "# Roadmap Live Initial Baseline Comparison",
        "",
        f"- dataset_dir: `{report['dataset_dir']}`",
        f"- model_root: `{report['model_root']}`",
        "",
    ]
    for scheme, categories in sorted((report.get("categories") or {}).items()):
        lines.append(f"## {scheme}")
        lines.append("")
        for category, payload in sorted(categories.items()):
            test_baselines = payload["baselines"]["test"]
            lines.append(f"### {category}")
            for name, metrics in test_baselines.items():
                lines.append(
                    f"- {name}: acc@1=`{metrics['acc_at_1']:.4f}`, recall@3=`{metrics['recall_at_3']:.4f}`, macro_f1=`{metrics['macro_f1']:.4f}`, balanced_acc=`{metrics['balanced_accuracy']:.4f}`, stop_acc=`{metrics['stop_accuracy']:.4f}`"
                )
            rollout = payload["rollout_vs_first_step"]["test"]
            lines.append(
                f"- current_model first_step vs full_rollout: acc@1=`{rollout['first_step']['acc_at_1']:.4f}`, exact_full=`{rollout['full_rollout']['exact_full_plan_match']:.4f}`, prefix=`{rollout['full_rollout']['prefix_match_rate']:.4f}`, length_mae=`{rollout['full_rollout']['length_mae']:.4f}`"
            )
            lines.append("")
    ablation_lines = [
        "# Roadmap Live Initial Ablation Report",
        "",
        f"- dataset_dir: `{ablation_report['dataset_dir']}`",
        f"- model_root: `{ablation_report['model_root']}`",
        "",
    ]
    for scheme, scheme_payload in sorted((ablation_report.get("split_schemes") or {}).items()):
        ablation_lines.append(f"## {scheme}")
        ablation_lines.append("")
        for category, payload in sorted((scheme_payload.get("categories") or {}).items()):
            ablation_lines.append(f"### {category}")
            for name, metrics in sorted((payload.get("test_metrics") or {}).items()):
                if metrics.get("status") == "skipped":
                    ablation_lines.append(f"- {name}: skipped (`{metrics['reason']}`)")
                    continue
                ablation_lines.append(
                    f"- {name}: acc@1=`{metrics['acc_at_1']:.4f}`, macro_f1=`{metrics['macro_f1']:.4f}`, balanced_acc=`{metrics['balanced_accuracy']:.4f}`, stop_acc=`{metrics['stop_accuracy']:.4f}`"
                )
            ablation_lines.append("")
    return "\n".join(lines).rstrip() + "\n", "\n".join(ablation_lines).rstrip() + "\n"
