from __future__ import annotations

from collections import defaultdict
from datetime import timedelta
from functools import lru_cache
import json
from pathlib import Path
from typing import Any

try:
    import joblib
except Exception:  # pragma: no cover
    joblib = None

try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None

try:
    import pandas as pd
except Exception:  # pragma: no cover
    pd = None

from django.conf import settings
from django.utils import timezone

from catalog.models import Product
from roadmap_app.content_features import (
    build_base_content_features,
    build_candidate_catalog_summaries,
    build_candidate_content_features,
    product_signature,
    profile_signature,
)
from transactions.models import TransactionItem
from users_app.models import CustomerProfile

STOP_TOKEN = "__stop__"
NONE_TOKEN = "__none__"
PLANNER_SOURCES = {"ml_planner", "planner_fallback", "state_prefix"}

DEFAULT_CANDIDATE_TYPES: dict[str, list[str]] = {
    "haircare": ["shampoo", "conditioner", "hair_mask", "hair_oil", "scalp_serum", "leave_in"],
    "skincare": ["cleanser", "serum", "moisturizer", "spf", "toner", "mask", "eye_cream", "essence"],
    "makeup": ["foundation", "mascara", "blush", "lipstick", "eyeshadow", "primer", "setting_spray"],
    "fragrance": ["warm_day", "warm_evening", "cold_day", "cold_evening"],
}


def _unique(values: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        token = str(value or "").strip().lower()
        if not token or token in seen:
            continue
        seen.add(token)
        out.append(token)
    return out


def planner_runtime_mode() -> str:
    raw = str(getattr(settings, "ROADMAP_PLANNER_V1_MODE", "off") or "off").strip().lower()
    if raw in {"off", "shadow", "serve"}:
        return raw
    return "off"


def planner_enabled_for_category(category: str) -> bool:
    if planner_runtime_mode() == "off":
        return False
    enabled_categories = {
        str(item or "").strip().lower()
        for item in (getattr(settings, "ROADMAP_PLANNER_V1_ENABLED_CATEGORIES", []) or [])
    }
    if not enabled_categories:
        return True
    return str(category or "").strip().lower() in enabled_categories


def planner_model_path() -> str:
    return str(getattr(settings, "ROADMAP_PLANNER_V1_MODEL_PATH", "") or "").strip()


def _artifact_dir_for_model_path(model_path: Path) -> Path:
    return model_path.parent if model_path.suffix else model_path


def _artifact_metadata_path_for_model_path(model_path: Path) -> Path:
    return (_artifact_dir_for_model_path(model_path) / "metadata.json").expanduser()


def _artifact_eval_report_path_for_model_path(model_path: Path) -> Path:
    return (_artifact_dir_for_model_path(model_path) / "eval_report.json").expanduser()


@lru_cache(maxsize=8)
def _load_planner_artifact(model_path: str) -> dict[str, Any] | None:
    if not model_path or joblib is None:
        return None
    try:
        path = Path(model_path).expanduser().resolve()
    except Exception:
        return None
    if not path.exists():
        return None
    try:
        artifact = joblib.load(path)
    except Exception:
        return None
    return artifact if isinstance(artifact, dict) else None


@lru_cache(maxsize=8)
def _load_metadata_cached(path_str: str, mtime_ns: int) -> dict[str, Any] | None:
    del mtime_ns
    path = Path(path_str)
    if not path.exists() or not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _load_planner_metadata(model_path: str | Path | None) -> dict[str, Any] | None:
    raw = str(model_path or "").strip()
    if not raw:
        return None
    path = _artifact_metadata_path_for_model_path(Path(raw).expanduser())
    if not path.exists() or not path.is_file():
        return None
    try:
        return _load_metadata_cached(str(path.resolve()), int(path.stat().st_mtime_ns))
    except Exception:
        return None


def planner_model_artifact_summary(model_path: str | Path | None = None) -> dict[str, Any]:
    raw = str(model_path or planner_model_path()).strip()
    if not raw:
        return {
            "model_path": "",
            "artifact_dir": "",
            "exists": False,
            "metadata_path": "",
            "metadata_exists": False,
            "eval_report_path": "",
            "eval_report_exists": False,
            "model_version": "",
            "selected_feature_set": "",
            "trained_at_utc": "",
            "task": "",
            "estimator": "",
            "metrics_test": {},
            "planner_guard": {},
        }

    path = Path(raw).expanduser()
    artifact_dir = _artifact_dir_for_model_path(path)
    metadata_path = _artifact_metadata_path_for_model_path(path)
    eval_report_path = _artifact_eval_report_path_for_model_path(path)
    metadata = _load_planner_metadata(path)
    artifact = _load_planner_artifact(str(path))
    metrics_test = metadata.get("metrics_test") if isinstance(metadata, dict) else None
    planner_guard = metadata.get("planner_guard") if isinstance(metadata, dict) else None
    path_version = path.stem if path.suffix else path.name
    return {
        "model_path": str(path),
        "artifact_dir": str(artifact_dir),
        "exists": bool(path.exists() and path.is_file()),
        "metadata_path": str(metadata_path),
        "metadata_exists": bool(metadata_path.exists() and metadata_path.is_file()),
        "eval_report_path": str(eval_report_path),
        "eval_report_exists": bool(eval_report_path.exists() and eval_report_path.is_file()),
        "model_version": str(
            ((metadata or {}).get("model_version") if isinstance(metadata, dict) else None)
            or ((artifact or {}).get("model_version") if isinstance(artifact, dict) else None)
            or path_version
            or ""
        ),
        "selected_feature_set": str(
            ((metadata or {}).get("selected_feature_set") if isinstance(metadata, dict) else None)
            or ((artifact or {}).get("selected_feature_set") if isinstance(artifact, dict) else None)
            or ""
        ),
        "trained_at_utc": str(
            ((metadata or {}).get("trained_at_utc") if isinstance(metadata, dict) else None)
            or ""
        ),
        "task": str(
            ((metadata or {}).get("task") if isinstance(metadata, dict) else None)
            or ((artifact or {}).get("task") if isinstance(artifact, dict) else None)
            or ""
        ),
        "estimator": str(((metadata or {}).get("estimator") if isinstance(metadata, dict) else None) or ""),
        "metrics_test": metrics_test if isinstance(metrics_test, dict) else {},
        "planner_guard": planner_guard if isinstance(planner_guard, dict) else {},
    }


def _candidate_types_for_category(artifact: dict[str, Any], category: str) -> list[str]:
    by_category = artifact.get("candidate_types_by_category") or {}
    values = by_category.get(category) or DEFAULT_CANDIDATE_TYPES.get(category) or []
    return [x for x in _unique([str(item) for item in values]) if x != STOP_TOKEN]


def _candidate_popularity_prior(artifact: dict[str, Any], *, category: str, candidate_type: str) -> float:
    priors = artifact.get("candidate_popularity_priors") or {}
    category_map = priors.get(category) if isinstance(priors, dict) else None
    if isinstance(category_map, dict):
        try:
            return float(category_map.get(candidate_type, 0.0) or 0.0)
        except Exception:
            return 0.0
    return 0.0


def _history_rows(user, *, now_utc) -> list[dict[str, Any]]:
    return list(
        TransactionItem.objects.filter(transaction__user=user, transaction__created_at__lte=now_utc)
        .order_by("-transaction__created_at", "-transaction__id", "-id")
        .values(
            "transaction__id",
            "transaction__created_at",
            "transaction__total_amount",
            "product__category",
            "product__product_type",
            "product__concerns",
            "product__actives",
            "product__flags",
            "product__supported_skin_types",
            "product__attrs",
            "product__ingredients_inci",
            "product__raw_meta",
        )
    )


def _history_features(history_rows: list[dict[str, Any]], *, category: str, now_utc) -> dict[str, Any]:
    last_rows = history_rows[:5]
    out: dict[str, Any] = {}
    for idx in range(5):
        if idx < len(last_rows):
            out[f"last{idx + 1}_product_type"] = str(last_rows[idx].get("product__product_type") or NONE_TOKEN).strip().lower() or NONE_TOKEN
            out[f"last{idx + 1}_category"] = str(last_rows[idx].get("product__category") or NONE_TOKEN).strip().lower() or NONE_TOKEN
        else:
            out[f"last{idx + 1}_product_type"] = NONE_TOKEN
            out[f"last{idx + 1}_category"] = NONE_TOKEN

    category_rows = [
        row for row in history_rows if str(row.get("product__category") or "").strip().lower() == category
    ]
    if category_rows:
        last_category_ts = category_rows[0]["transaction__created_at"]
        out["days_since_last_purchase_in_category"] = int((now_utc - last_category_ts).days)
    else:
        out["days_since_last_purchase_in_category"] = -1

    window_start = now_utc - timedelta(days=90)
    category_90d = [
        row for row in category_rows if row["transaction__created_at"] >= window_start
    ]
    tx_totals: dict[int, float] = {}
    for row in category_90d:
        tx_id = int(row["transaction__id"])
        if tx_id not in tx_totals:
            try:
                tx_totals[tx_id] = float(row.get("transaction__total_amount") or 0.0)
            except Exception:
                tx_totals[tx_id] = 0.0
    out["tx_count_90d_category"] = int(len(tx_totals))
    out["tx_amount_90d_category"] = round(float(sum(tx_totals.values())), 6)
    return out


def _candidate_history_features(
    history_rows: list[dict[str, Any]],
    *,
    category: str,
    candidate_type: str,
    last_product_types: list[str],
    now_utc,
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    out["candidate_matches_last1"] = int(bool(last_product_types) and candidate_type == last_product_types[0])
    out["candidate_matches_last3_any"] = int(candidate_type in last_product_types[:3])
    out["candidate_seen_count_last5"] = int(sum(1 for item in last_product_types if item == candidate_type))

    window_start = now_utc - timedelta(days=90)
    matching_rows = [
        row
        for row in history_rows
        if str(row.get("product__category") or "").strip().lower() == category
        and str(row.get("product__product_type") or "").strip().lower() == candidate_type
        and row["transaction__created_at"] >= window_start
    ]
    out["candidate_seen_90d_count_in_category"] = int(len(matching_rows))
    if matching_rows:
        out["candidate_days_since_last_seen_in_category"] = int((now_utc - matching_rows[0]["transaction__created_at"]).days)
    else:
        out["candidate_days_since_last_seen_in_category"] = -1
    return out


def _empty_runtime_result(*, category: str, reason: str, decision: str = "disabled") -> dict[str, Any]:
    return {
        "category": category,
        "decision": decision,
        "fallback_reason": reason if decision == "fallback" else None,
        "disabled_reason": reason if decision == "disabled" else None,
        "chain": [],
        "source_by_type": {},
        "trace": [],
        "model_path": planner_model_path(),
        "model_version": None,
        "selected_feature_set": None,
    }


def _prepare_runtime_features(
    artifact: dict[str, Any],
    rows: list[dict[str, Any]],
) -> "pd.DataFrame":
    feature_columns = [str(col) for col in (artifact.get("feature_columns") or []) if str(col)]
    categorical_features = {
        str(col) for col in (artifact.get("categorical_features") or []) if str(col)
    }
    numeric_features = {
        str(col) for col in (artifact.get("numeric_features") or []) if str(col)
    }
    if not feature_columns:
        raise ValueError("Planner artifact has no feature columns")
    frame = pd.DataFrame(rows)
    model_type = str(artifact.get("model_type") or "").strip().lower()
    for column in feature_columns:
        if column not in frame.columns:
            frame[column] = 0.0 if column in numeric_features else NONE_TOKEN
    for column in feature_columns:
        if column in categorical_features:
            frame[column] = frame[column].fillna(NONE_TOKEN).astype(str)
            if model_type == "lightgbm_ranker":
                frame[column] = frame[column].astype("category")
        elif column in numeric_features:
            frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(0.0)
    return frame[feature_columns]


def _predict_scores(artifact: dict[str, Any], features: "pd.DataFrame") -> "np.ndarray":
    model = artifact.get("model")
    if model is None:
        raise ValueError("Planner artifact is missing model")
    X = features
    preprocessor = artifact.get("preprocessor")
    if preprocessor is not None:
        X = preprocessor.transform(features)
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)
        if getattr(probs, "ndim", 1) == 2 and probs.shape[1] >= 2:
            return np.asarray(probs[:, 1], dtype=float)
        return np.asarray(probs, dtype=float).reshape(-1)
    if hasattr(model, "decision_function"):
        return np.asarray(model.decision_function(X), dtype=float).reshape(-1)
    return np.asarray(model.predict(X), dtype=float).reshape(-1)


def generate_planner_chain(
    *,
    user,
    category: str,
    candidate_types: list[str],
    purchased_types: list[str],
    owned_types_ordered: list[str],
    min_steps: int,
    max_steps: int,
    refresh_caller: str,
) -> dict[str, Any]:
    category = str(category or "").strip().lower()
    if pd is None or np is None:
        return _empty_runtime_result(category=category, reason="planner_dependencies_missing", decision="disabled")
    if not planner_enabled_for_category(category):
        return _empty_runtime_result(category=category, reason="planner_category_disabled", decision="disabled")

    model_path = planner_model_path()
    artifact = _load_planner_artifact(model_path)
    if not artifact:
        return _empty_runtime_result(category=category, reason="planner_model_missing", decision="disabled")

    category_candidates = _candidate_types_for_category(artifact, category)
    if candidate_types:
        category_candidates = _unique(category_candidates + candidate_types)
    category_candidates = [item for item in category_candidates if item != STOP_TOKEN]
    if not category_candidates:
        return _empty_runtime_result(category=category, reason="planner_no_candidates", decision="disabled")

    purchased_prefix = [item for item in _unique(purchased_types) if item in category_candidates]
    owned_only = [
        item
        for item in _unique(owned_types_ordered)
        if item in category_candidates and item not in set(purchased_prefix)
    ]
    remaining = [
        item
        for item in category_candidates
        if item not in set(purchased_prefix) and item not in set(owned_only)
    ]

    now_utc = timezone.now()
    history_rows = _history_rows(user, now_utc=now_utc)
    profile_row = CustomerProfile.objects.filter(user=user).first()
    profile_sig = profile_signature(profile_row)
    history = _history_features(history_rows, category=category, now_utc=now_utc)
    last_product_types = [str(history[f"last{i}_product_type"]) for i in range(1, 6)]
    anchor_row = next(
        (
            row
            for row in history_rows
            if str(row.get("product__category") or "").strip().lower() == category
        ),
        None,
    )
    anchor_sig = product_signature(
        {
            "category": anchor_row.get("product__category") if isinstance(anchor_row, dict) else "",
            "product_type": anchor_row.get("product__product_type") if isinstance(anchor_row, dict) else "",
            "concerns": (
                anchor_row.get("product__concerns")
                if isinstance(anchor_row, dict) and isinstance(anchor_row.get("product__concerns"), list)
                else []
            ),
            "actives": (
                anchor_row.get("product__actives")
                if isinstance(anchor_row, dict) and isinstance(anchor_row.get("product__actives"), list)
                else []
            ),
            "flags": (
                anchor_row.get("product__flags")
                if isinstance(anchor_row, dict) and isinstance(anchor_row.get("product__flags"), list)
                else []
            ),
            "supported_skin_types": (
                anchor_row.get("product__supported_skin_types")
                if isinstance(anchor_row, dict) and isinstance(anchor_row.get("product__supported_skin_types"), list)
                else []
            ),
            "attrs": anchor_row.get("product__attrs") if isinstance(anchor_row, dict) else {},
            "ingredients_inci": anchor_row.get("product__ingredients_inci") if isinstance(anchor_row, dict) else "",
            "raw_meta": anchor_row.get("product__raw_meta") if isinstance(anchor_row, dict) else {},
        }
    )
    base_content = build_base_content_features(profile_sig, anchor_sig)
    candidate_catalog_summaries = build_candidate_catalog_summaries(
        list(
            Product.objects.filter(category=category, product_type__in=category_candidates).values(
                "category",
                "product_type",
                "concerns",
                "actives",
                "flags",
                "supported_skin_types",
                "attrs",
                "ingredients_inci",
                "raw_meta",
            )
        )
    )
    trace: list[dict[str, Any]] = []
    planned_actionable: list[str] = []

    while remaining and len(_unique(purchased_prefix + owned_only + planned_actionable)) < int(max_steps):
        completed_set = set(purchased_prefix) | set(planned_actionable)
        owned_set = set(owned_only)
        current_order = _unique(purchased_prefix + owned_only + planned_actionable + remaining)
        current_missing = [item for item in current_order if item not in completed_set and item not in owned_set]
        current_next = current_missing[0] if current_missing else NONE_TOKEN

        row_payloads: list[dict[str, Any]] = []
        scoring_candidates = list(current_missing) + [STOP_TOKEN]
        for candidate in scoring_candidates:
            position = current_order.index(candidate) + 1 if candidate in current_order else -1
            status_completed = int(candidate in completed_set)
            status_owned = int(candidate in owned_set)
            status_missing = int(candidate in current_missing)
            row = {
                "category": category,
                "candidate_type": candidate,
                "matched_by": NONE_TOKEN,
                "refresh_caller": str(refresh_caller or NONE_TOKEN),
                "current_ml_decision": "planner_v1",
                "current_rollout_mode": planner_runtime_mode(),
                "current_next_product_type": current_next,
                "steps_total": int(len(current_order)),
                "missing_steps_count": int(len(current_missing)),
                "recommended_steps_count": 0,
                "owned_steps_count": int(len(owned_only)),
                "completed_steps_count": int(len(completed_set)),
                "skipped_steps_count": 0,
                "next_step_index_current": int(current_order.index(current_next) + 1) if current_next in current_order else -1,
                "days_since_last_purchase_in_category": int(history["days_since_last_purchase_in_category"]),
                "tx_count_90d_category": int(history["tx_count_90d_category"]),
                "tx_amount_90d_category": float(history["tx_amount_90d_category"]),
                "candidate_in_generated_plan": int(candidate in current_order),
                "candidate_position_in_generated_plan": int(position),
                "candidate_is_current_next_step": int(candidate == current_next),
                "candidate_has_recommendation_in_plan": 0,
                "candidate_current_missing": int(status_missing),
                "candidate_current_recommended": 0,
                "candidate_current_owned": int(status_owned),
                "candidate_current_completed": int(status_completed),
                "candidate_current_skipped": 0,
                "candidate_popularity_in_train": _candidate_popularity_prior(
                    artifact,
                    category=category,
                    candidate_type=candidate,
                ),
                "candidate_is_stop": int(candidate == STOP_TOKEN),
            }
            row.update(history)
            row.update(base_content)
            row.update(
                _candidate_history_features(
                    history_rows,
                    category=category,
                    candidate_type=candidate,
                    last_product_types=last_product_types,
                    now_utc=now_utc,
                )
            )
            row.update(
                build_candidate_content_features(
                    candidate_catalog_summaries.get((category, candidate)),
                    profile_sig,
                    anchor_sig,
                )
            )
            row_payloads.append(row)

        try:
            features = _prepare_runtime_features(artifact, row_payloads)
            scores = _predict_scores(artifact, features)
        except Exception:
            return _empty_runtime_result(category=category, reason="planner_inference_failed", decision="fallback")

        ranked = sorted(
            zip(row_payloads, scores.tolist()),
            key=lambda pair: float(pair[1]),
            reverse=True,
        )
        selected = None
        selected_score = None
        for row, score in ranked:
            candidate = str(row.get("candidate_type") or "")
            if candidate == STOP_TOKEN:
                current_length = len(_unique(purchased_prefix + owned_only + planned_actionable))
                if current_length >= int(min_steps):
                    selected = STOP_TOKEN
                    selected_score = float(score)
                    break
                continue
            selected = candidate
            selected_score = float(score)
            break

        if selected in {None, STOP_TOKEN}:
            trace.append({"selected": STOP_TOKEN, "score": float(selected_score or 0.0)})
            break

        planned_actionable.append(selected)
        remaining = [item for item in remaining if item != selected]
        trace.append({"selected": selected, "score": float(selected_score or 0.0)})

    final_chain = _unique(purchased_prefix + owned_only + planned_actionable)
    if len(final_chain) < int(min_steps):
        for item in remaining:
            if item in final_chain:
                continue
            final_chain.append(item)
            if len(final_chain) >= int(min_steps):
                break
    final_chain = final_chain[: int(max_steps)]

    if not final_chain:
        return _empty_runtime_result(category=category, reason="planner_empty_chain", decision="fallback")

    source_by_type: dict[str, dict[str, Any]] = {}
    for item in purchased_prefix + owned_only:
        source_by_type[item] = {"source": "state_prefix", "score": None}
    score_by_type = {
        str(item.get("selected") or ""): float(item.get("score") or 0.0)
        for item in trace
        if str(item.get("selected") or "") not in {"", STOP_TOKEN}
    }
    for item in final_chain:
        if item in source_by_type:
            continue
        if item in score_by_type:
            source_by_type[item] = {"source": "ml_planner", "score": float(score_by_type[item])}
        else:
            source_by_type[item] = {"source": "planner_fallback", "score": None}

    return {
        "category": category,
        "decision": "model_used",
        "fallback_reason": None,
        "disabled_reason": None,
        "chain": final_chain,
        "source_by_type": source_by_type,
        "trace": trace,
        "model_path": model_path,
        "model_version": str(artifact.get("model_version") or ""),
        "selected_feature_set": str(artifact.get("selected_feature_set") or ""),
    }
