from __future__ import annotations

import json
from datetime import timedelta, timezone as dt_timezone
from functools import lru_cache
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

from django.conf import settings
from django.utils import timezone

from catalog.models import Product
from roadmap_app.ml_artifact_proof import (
    PROOF_FILE_EVAL,
    PROOF_FILE_METADATA,
    PROOF_FILE_SHADOW,
    PROOF_FILE_UPLIFT_30D,
    PROOF_FILE_UPLIFT_7D,
    artifact_dir_for_model_path,
    artifact_file_path,
    load_json_file,
    proof_bundle_status,
)
from roadmap_app.content_features import (
    build_base_content_features,
    build_candidate_catalog_summaries,
    build_candidate_content_features,
    build_chain_transition_features,
    effective_nextstep_rules_chain,
    build_nextstep_plan_state_features,
    product_signature,
    profile_signature,
)
from roadmap_app.fragrance_slots import SLOTS, slot_of_fragrance
from transactions.models import TransactionItem
from users_app.models import CustomerProfile


def _model_path() -> Path:
    if bool(getattr(settings, "ROADMAP_NEXTSTEP_V4_ENABLED", False)):
        raw = str(getattr(settings, "ROADMAP_NEXTSTEP_V4_MODEL_PATH", "") or "").strip()
        if not raw:
            raw = str(getattr(settings, "ROADMAP_NEXTSTEP_V3_MODEL_PATH", "") or "").strip()
        if not raw:
            raw = str(getattr(settings, "ROADMAP_NEXTSTEP_MODEL_PATH", "") or "")
        return Path(raw).expanduser()

    if bool(getattr(settings, "ROADMAP_NEXTSTEP_V3_ENABLED", False)):
        raw = str(getattr(settings, "ROADMAP_NEXTSTEP_V3_MODEL_PATH", "") or "").strip()
        if not raw:
            raw = str(getattr(settings, "ROADMAP_NEXTSTEP_MODEL_PATH", "") or "")
        return Path(raw).expanduser()

    return Path(str(getattr(settings, "ROADMAP_NEXTSTEP_MODEL_PATH", "") or "")).expanduser()


def _model_dir() -> Path:
    path = _model_path()
    return path.parent if path.suffix else path


def _artifact_dir_for_model_path(model_path: Path) -> Path:
    return artifact_dir_for_model_path(model_path)


def _artifact_metadata_path_for_model_path(model_path: Path) -> Path:
    return artifact_file_path(model_path, PROOF_FILE_METADATA)


def _artifact_eval_report_path_for_model_path(model_path: Path) -> Path:
    return artifact_file_path(model_path, PROOF_FILE_EVAL)


def _artifact_shadow_report_path_for_model_path(model_path: Path) -> Path:
    return artifact_file_path(model_path, PROOF_FILE_SHADOW)


def _artifact_metadata_path() -> Path:
    return (_model_dir() / "metadata.json").expanduser()


def _artifact_eval_report_path() -> Path:
    return (_model_dir() / "eval_report.json").expanduser()


def _artifact_uplift_report_path(window: str) -> Path:
    suffix = PROOF_FILE_UPLIFT_30D if str(window or "").strip() == "30d" else PROOF_FILE_UPLIFT_7D
    return artifact_file_path(_model_path(), suffix)


def v4_runtime_eval_report_path() -> Path:
    return _artifact_eval_report_path()


def v4_runtime_uplift_report_path(window: str) -> Path:
    return _artifact_uplift_report_path(window)


@lru_cache(maxsize=4)
def _load_model_cached(path_str: str, mtime_ns: int) -> Any | None:
    del mtime_ns
    path = Path(path_str)
    if not path.exists() or not path.is_file():
        return None
    try:
        import joblib

        return joblib.load(path_str)
    except Exception:
        pass
    try:
        import pickle

        with path.open("rb") as f:
            return pickle.load(f)
    except Exception:
        return None


def _load_model() -> Any | None:
    path = _model_path()
    if not path.exists() or not path.is_file():
        return None
    try:
        return _load_model_cached(str(path.resolve()), int(path.stat().st_mtime_ns))
    except Exception:
        return None


def _load_model_for_path(model_path: str | Path | None) -> Any | None:
    raw = str(model_path or "").strip()
    if not raw:
        return None
    path = Path(raw).expanduser()
    if not path.exists() or not path.is_file():
        return None
    try:
        return _load_model_cached(str(path.resolve()), int(path.stat().st_mtime_ns))
    except Exception:
        return None


@lru_cache(maxsize=4)
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


def _load_model_metadata() -> dict[str, Any] | None:
    path = _artifact_metadata_path()
    if not path.exists() or not path.is_file():
        return None
    try:
        return _load_metadata_cached(str(path.resolve()), int(path.stat().st_mtime_ns))
    except Exception:
        return None


def _load_model_metadata_for_path(model_path: str | Path | None) -> dict[str, Any] | None:
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


@lru_cache(maxsize=4)
def _load_eval_report_cached(path_str: str, mtime_ns: int) -> dict[str, Any] | None:
    del mtime_ns
    path = Path(path_str)
    if not path.exists() or not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _eval_report_from_metadata(metadata: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(metadata, dict):
        return None
    metrics_test = metadata.get("metrics_test")
    dataset_baselines = metadata.get("dataset_baselines")
    if not isinstance(metrics_test, dict) or not isinstance(dataset_baselines, dict):
        return None
    report: dict[str, Any] = {
        "metrics_test": metrics_test,
        "dataset_baselines": dataset_baselines,
    }
    for key in [
        "trained_at_utc",
        "model_version",
        "estimator",
        "selected_feature_set",
        "dataset_path",
        "train_rows",
        "val_rows",
        "test_rows",
        "train_rows_fit",
        "train_rows_sampled",
        "runtime_guard",
        "baseline_comparison",
    ]:
        if key in metadata:
            report[key] = metadata.get(key)
    return report


def _load_eval_report_for_path(model_path: str | Path | None) -> dict[str, Any] | None:
    path = _artifact_eval_report_path_for_model_path(Path(str(model_path or "").strip()).expanduser())
    return load_json_file(path)


def nextstep_model_artifact_summary(model_path: str | Path | None = None) -> dict[str, Any]:
    raw = str(model_path or _model_path()).strip()
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
            "runtime_guard": {},
        }

    path = Path(raw).expanduser()
    artifact_dir = _artifact_dir_for_model_path(path)
    metadata_path = _artifact_metadata_path_for_model_path(path)
    eval_report_path = _artifact_eval_report_path_for_model_path(path)
    metadata = _load_model_metadata_for_path(path)
    artifact = _load_model_for_path(path)
    artifact_dict = artifact if isinstance(artifact, dict) else {}
    metrics_test = metadata.get("metrics_test") if isinstance(metadata, dict) else None
    runtime_guard = metadata.get("runtime_guard") if isinstance(metadata, dict) else None
    path_version = path.stem if path.suffix else path.name
    summary: dict[str, Any] = {
        "model_path": str(path),
        "artifact_dir": str(artifact_dir),
        "exists": bool(path.exists() and path.is_file()),
        "metadata_path": str(metadata_path),
        "metadata_exists": bool(metadata_path.exists() and metadata_path.is_file()),
        "eval_report_path": str(eval_report_path),
        "eval_report_exists": bool(eval_report_path.exists() and eval_report_path.is_file()),
        "model_version": str(
            ((metadata or {}).get("model_version") if isinstance(metadata, dict) else None)
            or artifact_dict.get("model_version")
            or path_version
            or ""
        ),
        "selected_feature_set": str(
            ((metadata or {}).get("selected_feature_set") if isinstance(metadata, dict) else None)
            or artifact_dict.get("selected_feature_set")
            or ""
        ),
        "trained_at_utc": str(
            ((metadata or {}).get("trained_at_utc") if isinstance(metadata, dict) else None)
            or artifact_dict.get("trained_at_utc")
            or ""
        ),
        "task": str(
            ((metadata or {}).get("task") if isinstance(metadata, dict) else None)
            or artifact_dict.get("task")
            or ""
        ),
        "estimator": str(((metadata or {}).get("estimator") if isinstance(metadata, dict) else None) or ""),
        "metrics_test": metrics_test if isinstance(metrics_test, dict) else {},
        "runtime_guard": runtime_guard if isinstance(runtime_guard, dict) else {},
    }
    return summary


@lru_cache(maxsize=4)
def _load_uplift_report_cached(path_str: str, mtime_ns: int) -> dict[str, Any] | None:
    del mtime_ns
    path = Path(path_str)
    if not path.exists() or not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _load_uplift_report_from_path(path: Path) -> dict[str, Any] | None:
    if not path.exists() or not path.is_file():
        return None
    try:
        return _load_uplift_report_cached(str(path.resolve()), int(path.stat().st_mtime_ns))
    except Exception:
        return None


def _artifact_uplift_report_path_for_model_path(model_path: str | Path | None, window: str) -> Path:
    suffix = PROOF_FILE_UPLIFT_30D if str(window or "").strip() == "30d" else PROOF_FILE_UPLIFT_7D
    return artifact_file_path(model_path, suffix)


def _load_uplift_report_bundle_for_model_path(
    model_path: str | Path | None,
    window: str,
) -> tuple[dict[str, Any] | None, str]:
    path = _artifact_uplift_report_path_for_model_path(model_path, window)
    return _load_uplift_report_from_path(path), str(path)


def nextstep_proof_bundle_status(
    model_path: str | Path | None = None,
    *,
    require_shadow_report: bool = False,
    require_uplift_reports: bool = False,
) -> dict[str, Any]:
    raw = str(model_path or _model_path()).strip()
    if not raw:
        return {
            "model_path": "",
            "artifact_dir": "",
            "required_files": [],
            "optional_files": [],
            "files": {},
            "required_complete": False,
            "missing_required": ["model.pkl"],
            "invalid_required": [],
            "stale_required": [],
            "reason": "missing_model_path",
        }
    metadata = _load_model_metadata_for_path(raw)
    expected_model_version = str((metadata or {}).get("model_version") or "").strip() or None
    required_files = [PROOF_FILE_METADATA, PROOF_FILE_EVAL]
    if require_shadow_report:
        required_files.append(PROOF_FILE_SHADOW)
    if require_uplift_reports:
        required_files.extend([PROOF_FILE_UPLIFT_7D, PROOF_FILE_UPLIFT_30D])
    return proof_bundle_status(
        model_path=raw,
        required_files=required_files,
        expected_model_version=expected_model_version,
    )


def _to_float(v: Any) -> float:
    try:
        return float(v)
    except Exception:
        return 0.0


def _safe_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _safe_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _ordered_unique_tokens(values: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        token = str(value or "").strip().lower()
        if not token or token in seen:
            continue
        seen.add(token)
        out.append(token)
    return out


def _normalized_category_set(value: Any) -> set[str]:
    out: set[str] = set()
    if isinstance(value, str):
        for token in value.split(","):
            token_norm = str(token or "").strip().lower()
            if token_norm:
                out.add(token_norm)
        return out
    if isinstance(value, (list, tuple, set)):
        for token in value:
            token_norm = str(token or "").strip().lower()
            if token_norm:
                out.add(token_norm)
    return out


def _report_metric(report: dict[str, Any], metric: str) -> tuple[float | None, float | None]:
    metric = str(metric or "ndcg_at_5").strip()
    metrics_test = _safe_dict(report.get("metrics_test"))
    baselines = _safe_dict(_safe_dict(_safe_dict(report.get("dataset_baselines")).get("splits")).get("test"))
    popularity = _safe_dict(baselines.get("popularity"))
    model_value = metrics_test.get(metric)
    baseline_value = popularity.get(metric)
    try:
        model_f = float(model_value)
    except Exception:
        model_f = None
    try:
        baseline_f = float(baseline_value)
    except Exception:
        baseline_f = None
    return model_f, baseline_f


def v4_min_lift_guard_status(model_path: str | Path | None = None) -> dict[str, Any]:
    enabled = bool(getattr(settings, "ROADMAP_NEXTSTEP_V4_MIN_LIFT_GUARD_ENABLED", True))
    metric = str(getattr(settings, "ROADMAP_NEXTSTEP_V4_MIN_LIFT_METRIC", "ndcg_at_5") or "ndcg_at_5").strip()
    required_delta = float(getattr(settings, "ROADMAP_NEXTSTEP_V4_MIN_LIFT_DELTA", 0.01))
    resolved_model_path = str(model_path or _model_path()).strip()
    proof = nextstep_proof_bundle_status(resolved_model_path)
    eval_path = str(_artifact_eval_report_path_for_model_path(Path(resolved_model_path).expanduser())) if resolved_model_path else ""
    out: dict[str, Any] = {
        "enabled": enabled,
        "metric": metric,
        "required_delta": required_delta,
        "passed": True,
        "reason": "guard_disabled" if not enabled else "ok",
        "model_path": resolved_model_path,
        "eval_path": eval_path,
        "proof_bundle": proof,
    }
    if not enabled:
        return out

    if not resolved_model_path:
        out["passed"] = False
        out["reason"] = "missing_model_path"
        return out
    if not bool(proof.get("required_complete")):
        out["passed"] = False
        out["reason"] = str(proof.get("reason") or "missing_eval_report")
        return out
    report = _load_eval_report_for_path(resolved_model_path)
    if not report:
        out["passed"] = False
        out["reason"] = "missing_eval_report"
        return out

    model_value, baseline_value = _report_metric(report, metric)
    out["model_value"] = model_value
    out["baseline_value"] = baseline_value
    if model_value is None or baseline_value is None:
        out["passed"] = False
        out["reason"] = "missing_eval_metric"
        return out

    delta = float(model_value - baseline_value)
    out["delta"] = delta
    if delta < required_delta:
        out["passed"] = False
        out["reason"] = "insufficient_lift"
    return out


def v4_category_rollout_status(category: str) -> dict[str, Any]:
    category_norm = str(category or "").strip().lower()
    global_enabled = bool(getattr(settings, "ROADMAP_NEXTSTEP_V4_ENABLED", False))
    enabled_categories = _normalized_category_set(
        getattr(settings, "ROADMAP_NEXTSTEP_V4_ENABLED_CATEGORIES", [])
    )
    disabled_categories = _normalized_category_set(
        getattr(settings, "ROADMAP_NEXTSTEP_V4_DISABLED_CATEGORIES", [])
    )
    out: dict[str, Any] = {
        "category": category_norm,
        "global_enabled": global_enabled,
        "enabled_categories": sorted(enabled_categories),
        "disabled_categories": sorted(disabled_categories),
        "passed": True,
        "reason": "passed",
    }
    if not global_enabled:
        out["passed"] = False
        out["reason"] = "ml_disabled"
        return out
    if category_norm in disabled_categories:
        out["passed"] = False
        out["reason"] = "category_disabled"
        return out
    if enabled_categories and category_norm not in enabled_categories:
        out["passed"] = False
        out["reason"] = "category_disabled"
        return out
    return out


def _category_guard_thresholds_snapshot() -> dict[str, Any]:
    min_plans = int(getattr(settings, "ROADMAP_NEXTSTEP_V4_CATEGORY_MIN_PLANS", 100))
    min_step_completion_lift = float(
        getattr(settings, "ROADMAP_NEXTSTEP_V4_MIN_STEP_COMPLETION_LIFT", 0.01)
    )
    min_offer_redeem_lift = float(
        getattr(settings, "ROADMAP_NEXTSTEP_V4_MIN_OFFER_REDEEM_LIFT", 0.005)
    )
    max_negative_step_ctr_lift_soft = float(
        getattr(
            settings,
            "ROADMAP_NEXTSTEP_V4_MAX_NEGATIVE_STEP_CTR_LIFT_SOFT",
            getattr(settings, "ROADMAP_NEXTSTEP_V4_MAX_NEGATIVE_STEP_CTR_LIFT", -0.02),
        )
    )
    max_negative_offer_ctr_lift_soft = float(
        getattr(
            settings,
            "ROADMAP_NEXTSTEP_V4_MAX_NEGATIVE_OFFER_CTR_LIFT_SOFT",
            getattr(settings, "ROADMAP_NEXTSTEP_V4_MAX_NEGATIVE_OFFER_CTR_LIFT", -0.03),
        )
    )
    max_negative_step_ctr_lift_strict = float(
        getattr(settings, "ROADMAP_NEXTSTEP_V4_MAX_NEGATIVE_STEP_CTR_LIFT", -0.01)
    )
    max_negative_offer_ctr_lift_strict = float(
        getattr(settings, "ROADMAP_NEXTSTEP_V4_MAX_NEGATIVE_OFFER_CTR_LIFT", -0.01)
    )
    allow_primary_win_despite_soft_ctr_drop = bool(
        getattr(settings, "ROADMAP_NEXTSTEP_V4_ALLOW_PRIMARY_WIN_DESPITE_SOFT_CTR_DROP", True)
    )
    secondary_step_threshold = (
        max_negative_step_ctr_lift_soft
        if allow_primary_win_despite_soft_ctr_drop
        else max_negative_step_ctr_lift_strict
    )
    secondary_offer_threshold = (
        max_negative_offer_ctr_lift_soft
        if allow_primary_win_despite_soft_ctr_drop
        else max_negative_offer_ctr_lift_strict
    )
    return {
        "min_plans": min_plans,
        "min_step_completion_lift": min_step_completion_lift,
        "min_offer_redeem_lift": min_offer_redeem_lift,
        "max_negative_step_ctr_lift_soft": max_negative_step_ctr_lift_soft,
        "max_negative_offer_ctr_lift_soft": max_negative_offer_ctr_lift_soft,
        "max_negative_step_ctr_lift_strict": max_negative_step_ctr_lift_strict,
        "max_negative_offer_ctr_lift_strict": max_negative_offer_ctr_lift_strict,
        "allow_primary_win_despite_soft_ctr_drop": allow_primary_win_despite_soft_ctr_drop,
        "secondary_step_threshold": secondary_step_threshold,
        "secondary_offer_threshold": secondary_offer_threshold,
    }


def _report_metric_abs_lift(report: dict[str, Any], category: str, metric: str) -> float | None:
    value = (
        _safe_dict(_safe_dict(_safe_dict(report.get("uplift")).get("by_category")).get(category))
        .get("step_funnel" if metric.startswith("step_") else "offer_funnel", {})
    )
    payload = _safe_dict(_safe_dict(value).get(metric))
    raw = payload.get("abs_lift")
    try:
        return float(raw)
    except Exception:
        return None


def v4_category_uplift_guard_status_from_report(
    category: str,
    report: dict[str, Any] | None,
    *,
    report_path: str | None = None,
) -> dict[str, Any]:
    category_norm = str(category or "").strip().lower()
    thresholds = _category_guard_thresholds_snapshot()
    report_path_final = str(report_path or "")

    out: dict[str, Any] = {
        "passed": False,
        "primary_passed": False,
        "secondary_passed": False,
        "reason": "insufficient_sample",
        "source_report_path": report_path_final,
        "cohort_mode": "fresh",
        "control": "non_model",
        "category": category_norm,
        "sample_size_model": None,
        "sample_size_control": None,
        "step_completion_abs_lift": None,
        "offer_redeem_abs_lift": None,
        "step_ctr_abs_lift": None,
        "offer_ctr_abs_lift": None,
        "thresholds": thresholds,
        "min_plans": int(thresholds["min_plans"]),
        "min_step_completion_lift": float(thresholds["min_step_completion_lift"]),
        "min_offer_redeem_lift": float(thresholds["min_offer_redeem_lift"]),
        "max_negative_step_ctr_lift_soft": float(thresholds["max_negative_step_ctr_lift_soft"]),
        "max_negative_offer_ctr_lift_soft": float(thresholds["max_negative_offer_ctr_lift_soft"]),
        "allow_primary_win_despite_soft_ctr_drop": bool(
            thresholds["allow_primary_win_despite_soft_ctr_drop"]
        ),
    }

    if not report:
        return out

    params = _safe_dict(report.get("params"))
    cohort_mode = str(params.get("cohort_mode") or "").strip().lower()
    control = str(params.get("control") or "").strip().lower()
    if cohort_mode:
        out["cohort_mode"] = cohort_mode
    if control:
        out["control"] = control
    if out["cohort_mode"] != "fresh" or out["control"] != "non_model":
        out["reason"] = "primary_metrics_not_met"
        return out

    by_category = _safe_dict(_safe_dict(report.get("breakdowns")).get("by_category"))
    cat_block = _safe_dict(by_category.get(category_norm))
    model_block = _safe_dict(cat_block.get("model_used"))
    control_block = _safe_dict(cat_block.get("control"))

    model_plans = int(model_block.get("plans_total", 0) or 0)
    control_plans = int(control_block.get("plans_total", 0) or 0)
    out["sample_size_model"] = model_plans
    out["sample_size_control"] = control_plans

    out["step_completion_abs_lift"] = _report_metric_abs_lift(
        report, category_norm, "step_completion_rate"
    )
    out["offer_redeem_abs_lift"] = _report_metric_abs_lift(
        report, category_norm, "offer_redeem_rate"
    )
    out["step_ctr_abs_lift"] = _report_metric_abs_lift(report, category_norm, "step_ctr")
    out["offer_ctr_abs_lift"] = _report_metric_abs_lift(report, category_norm, "offer_ctr")

    min_plans = int(thresholds["min_plans"])
    if model_plans < min_plans or control_plans < min_plans:
        if model_plans > 0 and control_plans > 0:
            out["reason"] = "sample_too_small_but_nonzero_control"
        elif model_plans > 0 and control_plans <= 0:
            out["reason"] = "missing_control_sample"
        elif control_plans > 0 and model_plans <= 0:
            out["reason"] = "missing_model_sample"
        else:
            out["reason"] = "insufficient_sample"
        return out

    min_step_completion_lift = float(thresholds["min_step_completion_lift"])
    min_offer_redeem_lift = float(thresholds["min_offer_redeem_lift"])
    step_completion_lift = out["step_completion_abs_lift"]
    offer_redeem_lift = out["offer_redeem_abs_lift"]
    primary_passed = bool(
        (step_completion_lift is not None and float(step_completion_lift) >= min_step_completion_lift)
        or (offer_redeem_lift is not None and float(offer_redeem_lift) >= min_offer_redeem_lift)
    )
    out["primary_passed"] = primary_passed
    if not primary_passed:
        out["reason"] = "low_uplift"
        return out

    secondary_step_threshold = float(thresholds["secondary_step_threshold"])
    secondary_offer_threshold = float(thresholds["secondary_offer_threshold"])
    step_ctr_lift = out["step_ctr_abs_lift"]
    offer_ctr_lift = out["offer_ctr_abs_lift"]
    step_severe = step_ctr_lift is not None and float(step_ctr_lift) < secondary_step_threshold
    offer_severe = offer_ctr_lift is not None and float(offer_ctr_lift) < secondary_offer_threshold
    secondary_passed = not bool(step_severe or offer_severe)
    out["secondary_passed"] = secondary_passed
    if not secondary_passed:
        out["reason"] = (
            "severe_negative_offer_ctr_lift"
            if offer_severe
            else "severe_negative_step_ctr_lift"
        )
        return out

    out["passed"] = True
    out["reason"] = "passed"
    return out


def v4_category_uplift_guard_status(
    category: str,
    *,
    model_path: str | Path | None = None,
    window: str = "30d",
) -> dict[str, Any]:
    report, report_path = _load_uplift_report_bundle_for_model_path(model_path or _model_path(), window)
    return v4_category_uplift_guard_status_from_report(
        category,
        report,
        report_path=report_path,
    )


def _guard_recommendation(rollout: dict[str, Any], guard: dict[str, Any]) -> str:
    if not bool(rollout.get("passed")):
        return "DISABLE"
    return "ENABLE" if bool(guard.get("passed")) else "HOLD"


def v4_category_staged_rollout_status_from_reports(
    category: str,
    *,
    report_7d: dict[str, Any] | None,
    report_30d: dict[str, Any] | None,
    report_path_7d: str | None = None,
    report_path_30d: str | None = None,
) -> dict[str, Any]:
    category_norm = str(category or "").strip().lower()
    rollout = v4_category_rollout_status(category_norm)
    guard_7d = v4_category_uplift_guard_status_from_report(
        category_norm,
        report_7d,
        report_path=report_path_7d,
    )
    guard_30d = v4_category_uplift_guard_status_from_report(
        category_norm,
        report_30d,
        report_path=report_path_30d,
    )

    recommendation_7d = _guard_recommendation(rollout, guard_7d)
    recommendation_30d = _guard_recommendation(rollout, guard_30d)

    final_status = "ENABLE"
    reason = "passed"
    hold_reason = None
    stability_gate_failures: list[str] = []

    if not bool(rollout.get("passed")):
        final_status = "DISABLE"
        reason = str(rollout.get("reason") or "category_disabled")
    else:
        if not bool(guard_7d.get("passed")):
            stability_gate_failures.append(f"7d:{str(guard_7d.get('reason') or 'guard_failed')}")
        if not bool(guard_30d.get("passed")):
            stability_gate_failures.append(f"30d:{str(guard_30d.get('reason') or 'guard_failed')}")
        if stability_gate_failures:
            final_status = "HOLD"
            if bool(guard_30d.get("passed")) and not bool(guard_7d.get("passed")):
                reason = "7d_unstable"
                hold_reason = str(guard_7d.get("reason") or "7d_unstable")
            elif not bool(guard_30d.get("passed")):
                reason = str(guard_30d.get("reason") or "30d_guard_failed")
                hold_reason = reason
            else:
                reason = str(guard_7d.get("reason") or "7d_guard_failed")
                hold_reason = reason

    out: dict[str, Any] = {
        "category": category_norm,
        "passed": final_status == "ENABLE",
        "final_status": final_status,
        "current_decision": final_status,
        "reason": reason,
        "hold_reason": hold_reason,
        "recommendation_7d": recommendation_7d,
        "recommendation_30d": recommendation_30d,
        "rollout": rollout,
        "guard_7d": guard_7d,
        "guard_30d": guard_30d,
        "stability_gate_failures": stability_gate_failures,
        "source_report_path_7d": str(report_path_7d or ""),
        "source_report_path_30d": str(report_path_30d or ""),
    }
    return out


def v4_category_staged_rollout_status(
    category: str,
    *,
    model_path: str | Path | None = None,
) -> dict[str, Any]:
    resolved_model_path = str(model_path or _model_path()).strip()
    report_7d, report_path_7d = _load_uplift_report_bundle_for_model_path(resolved_model_path, "7d")
    report_30d, report_path_30d = _load_uplift_report_bundle_for_model_path(resolved_model_path, "30d")
    return v4_category_staged_rollout_status_from_reports(
        category,
        report_7d=report_7d,
        report_30d=report_30d,
        report_path_7d=report_path_7d,
        report_path_30d=report_path_30d,
    )


def _normalize_predictions(raw: Any) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if raw is None:
        return out

    if isinstance(raw, dict):
        for key, value in raw.items():
            candidate = str(key or "").strip()
            if not candidate:
                continue
            score = _to_float(value)
            out.append({"candidate_type": candidate, "product_type": candidate, "score": score})
    elif isinstance(raw, (list, tuple)):
        for idx, item in enumerate(raw):
            if isinstance(item, dict):
                candidate = str(item.get("candidate_type") or item.get("product_type") or "").strip()
                if not candidate:
                    continue
                score = _to_float(
                    item.get("score", item.get("prob", item.get("confidence", 0.0)))
                )
                normalized_row = {"candidate_type": candidate, "product_type": candidate, "score": score}
                if "raw_score" in item:
                    normalized_row["raw_score"] = _to_float(item.get("raw_score"))
                if "runtime_bias" in item:
                    normalized_row["runtime_bias"] = _to_float(item.get("runtime_bias"))
                if isinstance(item.get("runtime_policy_biases"), dict):
                    normalized_row["runtime_policy_biases"] = {
                        str(k): _to_float(v)
                        for k, v in item["runtime_policy_biases"].items()
                        if str(k).strip()
                    }
                if isinstance(item.get("runtime_policies"), list):
                    normalized_row["runtime_policies"] = [
                        str(x).strip() for x in item["runtime_policies"] if str(x).strip()
                    ]
                if "model_type" in item:
                    normalized_row["model_type"] = str(item.get("model_type") or "").strip()
                out.append(normalized_row)
                continue

            if isinstance(item, (list, tuple)) and len(item) >= 2:
                candidate = str(item[0] or "").strip()
                if not candidate:
                    continue
                out.append(
                    {
                        "candidate_type": candidate,
                        "product_type": candidate,
                        "score": _to_float(item[1]),
                    }
                )
                continue

            candidate = str(item or "").strip()
            if candidate:
                out.append(
                    {
                        "candidate_type": candidate,
                        "product_type": candidate,
                        "score": max(0.0, 1.0 - (idx * 0.1)),
                    }
                )

    out.sort(key=lambda row: float(row.get("score", 0.0)), reverse=True)
    dedup: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in out:
        candidate = str(row.get("candidate_type") or row.get("product_type") or "").strip()
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        normalized_row = {
            "candidate_type": candidate,
            "product_type": candidate,
            "score": float(row.get("score", 0.0)),
        }
        if "raw_score" in row:
            normalized_row["raw_score"] = _to_float(row.get("raw_score"))
        if "runtime_bias" in row:
            normalized_row["runtime_bias"] = _to_float(row.get("runtime_bias"))
        if isinstance(row.get("runtime_policy_biases"), dict):
            normalized_row["runtime_policy_biases"] = {
                str(k): _to_float(v)
                for k, v in row["runtime_policy_biases"].items()
                if str(k).strip()
            }
        if isinstance(row.get("runtime_policies"), list):
            normalized_row["runtime_policies"] = [
                str(x).strip() for x in row["runtime_policies"] if str(x).strip()
            ]
        if "model_type" in row:
            normalized_row["model_type"] = str(row.get("model_type") or "").strip()
        dedup.append(normalized_row)
    return dedup


def blend_prediction_rows(
    base: Any,
    overlay: Any,
    *,
    overlay_weight: float = 0.25,
    overlay_label: str = "teacher",
) -> list[dict[str, Any]]:
    base_rows = _normalize_predictions(base)
    overlay_rows = _normalize_predictions(overlay)
    weight = max(0.0, float(overlay_weight))
    if not base_rows:
        return overlay_rows
    if not overlay_rows or weight <= 0.0:
        return base_rows

    base_by_candidate = {
        str(row.get("candidate_type") or "").strip().lower(): row for row in base_rows if str(row.get("candidate_type") or "").strip()
    }
    overlay_by_candidate = {
        str(row.get("candidate_type") or "").strip().lower(): row
        for row in overlay_rows
        if str(row.get("candidate_type") or "").strip()
    }
    ordered_candidates = list(base_by_candidate.keys()) + [
        candidate for candidate in overlay_by_candidate.keys() if candidate not in base_by_candidate
    ]

    blended_rows: list[dict[str, Any]] = []
    blended_total = 0.0
    for candidate in ordered_candidates:
        base_row = dict(base_by_candidate.get(candidate) or {})
        overlay_row = dict(overlay_by_candidate.get(candidate) or {})
        base_score = max(0.0, _to_float(base_row.get("score", 0.0)))
        overlay_score = max(0.0, _to_float(overlay_row.get("score", 0.0)))
        blended_score = base_score + (weight * overlay_score)
        blended_total += blended_score

        row = dict(base_row or overlay_row)
        row["candidate_type"] = candidate
        row["product_type"] = candidate
        row["score"] = blended_score
        row["raw_score"] = blended_score
        row["blend_components"] = {
            "base_score": base_score,
            f"{overlay_label}_score": overlay_score,
            f"{overlay_label}_weight": weight,
        }
        row["blend_mode"] = "weighted_probability_sum"
        blended_rows.append(row)

    if blended_total > 0:
        for row in blended_rows:
            row["score"] = float(_to_float(row.get("score")) / blended_total)

    blended_rows.sort(key=lambda row: float(row.get("score", 0.0)), reverse=True)
    return blended_rows


def _group_softmax(values: list[float], temperature: float) -> list[float]:
    if np is None or not values:
        total = float(sum(max(0.0, x) for x in values) or 1.0)
        return [float(max(0.0, x) / total) for x in values]

    t = max(0.05, float(temperature))
    arr = np.asarray(values, dtype=float) / t
    arr = arr - float(arr.max())
    exp = np.exp(arr)
    den = float(exp.sum())
    if den <= 0:
        den = 1.0
    probs = exp / den
    return [float(x) for x in probs.tolist()]


def _apply_runtime_progression_bias(
    *,
    category: str,
    rows: list[dict[str, Any]],
    score_list: list[float],
    days_since_last_purchase: int,
    has_context_anchor: bool,
) -> tuple[list[float], list[float]]:
    if not bool(getattr(settings, "ROADMAP_NEXTSTEP_V4_HAIRCARE_RUNTIME_BIAS_ENABLED", False)):
        return score_list, [0.0 for _ in score_list]
    if str(category or "").strip().lower() != "haircare":
        return score_list, [0.0 for _ in score_list]
    if not rows or len(rows) != len(score_list):
        return score_list, [0.0 for _ in score_list]

    effective_days = 0 if has_context_anchor else int(days_since_last_purchase)
    if effective_days < 0 or effective_days > 3:
        return score_list, [0.0 for _ in score_list]

    freshness = 1.0 if has_context_anchor else max(0.0, min(1.0, (3.0 - min(float(effective_days), 3.0)) / 3.0))
    if freshness <= 0.0:
        return score_list, [0.0 for _ in score_list]

    adjusted_scores: list[float] = []
    biases: list[float] = []
    for row, base_score in zip(rows, score_list):
        bias = 0.0
        if int(row.get("candidate_is_immediate_followup_to_anchor", 0) or 0) == 1:
            bias += 0.14 * freshness

        if int(row.get("candidate_is_same_as_anchor", 0) or 0) == 1:
            bias -= 0.14 * freshness
        elif int(row.get("candidate_is_before_anchor", 0) or 0) == 1:
            bias -= 0.08 * freshness

        adjusted_scores.append(float(base_score) + float(bias))
        biases.append(float(round(bias, 6)))
    return adjusted_scores, biases


def _apply_runtime_scalp_planned_target_rerank(
    *,
    category: str,
    rows: list[dict[str, Any]],
    score_list: list[float],
) -> tuple[list[float], list[float]]:
    if not bool(getattr(settings, "ROADMAP_NEXTSTEP_V4_HAIRCARE_SCALP_RERANK_ENABLED", False)):
        return score_list, [0.0 for _ in score_list]
    if str(category or "").strip().lower() != "haircare":
        return score_list, [0.0 for _ in score_list]
    if not rows or len(rows) != len(score_list):
        return score_list, [0.0 for _ in score_list]

    head = rows[0]
    planned_target = str(head.get("planned_target_product_type") or "").strip().lower()
    if planned_target != "scalp_serum":
        return score_list, [0.0 for _ in score_list]

    profile_has_scalp_objective = int(head.get("profile_has_scalp_objective", 0) or 0)
    anchor_has_scalp_focus = int(head.get("anchor_has_scalp_focus", 0) or 0)
    anchor_product_type = str(head.get("anchor_product_type") or "").strip().lower()
    if not profile_has_scalp_objective:
        return score_list, [0.0 for _ in score_list]
    if not anchor_has_scalp_focus and anchor_product_type != "shampoo":
        return score_list, [0.0 for _ in score_list]

    adjusted_scores: list[float] = []
    biases: list[float] = []
    for row, base_score in zip(rows, score_list):
        candidate = str(row.get("candidate_type") or row.get("product_type") or "").strip().lower()
        bias = 0.0
        matches_target = int(row.get("candidate_matches_planned_target", 0) or 0) == 1
        is_scalp_specialty = int(row.get("candidate_is_scalp_specialty", 0) or 0) == 1
        scalp_objective_match = _to_float(row.get("candidate_profile_scalp_objective_match_rate"))
        scalp_focus = max(
            _to_float(row.get("candidate_scalp_concern_focus_rate")),
            _to_float(row.get("candidate_scalp_active_focus_rate")),
        )

        if candidate == "scalp_serum" and matches_target and is_scalp_specialty:
            bias += 0.85
            bias += 0.25 * min(1.0, scalp_objective_match)
            bias += 0.15 * min(1.0, scalp_focus)
        elif candidate in {"conditioner", "hair_mask", "hair_oil", "leave_in"} and not is_scalp_specialty:
            bias -= 0.18
        elif candidate == "shampoo":
            bias -= 0.08

        adjusted_scores.append(float(base_score) + float(bias))
        biases.append(float(round(bias, 6)))
    return adjusted_scores, biases


def _apply_runtime_leavein_planned_target_rerank(
    *,
    category: str,
    rows: list[dict[str, Any]],
    score_list: list[float],
) -> tuple[list[float], list[float]]:
    if not bool(getattr(settings, "ROADMAP_NEXTSTEP_V4_HAIRCARE_LEAVEIN_RERANK_ENABLED", False)):
        return score_list, [0.0 for _ in score_list]
    if str(category or "").strip().lower() != "haircare":
        return score_list, [0.0 for _ in score_list]
    if not rows or len(rows) != len(score_list):
        return score_list, [0.0 for _ in score_list]

    head = rows[0]
    planned_target = str(head.get("planned_target_product_type") or "").strip().lower()
    if planned_target != "leave_in":
        return score_list, [0.0 for _ in score_list]

    profile_hair_type = str(head.get("profile_hair_type") or "").strip().lower()
    anchor_product_type = str(head.get("anchor_product_type") or "").strip().lower()
    if profile_hair_type not in {"curly", "wavy", "coily"} and anchor_product_type not in {"hair_mask", "conditioner"}:
        return score_list, [0.0 for _ in score_list]

    adjusted_scores: list[float] = []
    biases: list[float] = []
    for row, base_score in zip(rows, score_list):
        candidate = str(row.get("candidate_type") or row.get("product_type") or "").strip().lower()
        bias = 0.0
        matches_target = int(row.get("candidate_matches_planned_target", 0) or 0) == 1
        goal_match = _to_float(row.get("candidate_profile_goal_match_rate"))
        concern_match = _to_float(row.get("candidate_profile_hair_concern_match_rate"))
        hair_type_match = _to_float(row.get("candidate_profile_hair_type_match_rate"))
        thickness_match = _to_float(row.get("candidate_profile_hair_thickness_match_rate"))
        anchor_shared = max(
            _to_float(row.get("candidate_anchor_shared_concern_rate")),
            _to_float(row.get("candidate_anchor_shared_active_rate")),
        )

        if candidate == "leave_in" and matches_target:
            bias += 1.05
            bias += 0.20 * min(1.0, goal_match)
            bias += 0.18 * min(1.0, concern_match)
            bias += 0.12 * min(1.0, hair_type_match)
            bias += 0.08 * min(1.0, thickness_match)
            if anchor_product_type in {"hair_mask", "conditioner"}:
                bias += 0.12
            bias += 0.08 * min(1.0, anchor_shared)
        elif candidate == "hair_oil":
            bias -= 0.22
        elif candidate in {"conditioner", "hair_mask", "shampoo"}:
            bias -= 0.06

        adjusted_scores.append(float(base_score) + float(bias))
        biases.append(float(round(bias, 6)))
    return adjusted_scores, biases


def _apply_runtime_corechain_teacher_rerank(
    *,
    category: str,
    rows_out: list[dict[str, Any]],
    now_utc,
    items: list[dict[str, Any]] | None,
    profile: Any | None,
    context_products: list[dict[str, Any]] | None,
    catalog_products: list[dict[str, Any]] | None,
    planned_target_product_type: str | None,
    planned_target_step_index: int | None,
    candidate_types: list[str] | None,
) -> list[dict[str, Any]]:
    if not bool(getattr(settings, "ROADMAP_NEXTSTEP_V4_HAIRCARE_CORECHAIN_TEACHER_RERANK_ENABLED", False)):
        return rows_out
    if str(category or "").strip().lower() != "haircare":
        return rows_out
    if not rows_out:
        return rows_out

    planned_target = str(planned_target_product_type or "").strip().lower()
    allowed_targets = {
        str(item or "").strip().lower()
        for item in getattr(settings, "ROADMAP_NEXTSTEP_V4_HAIRCARE_CORECHAIN_TEACHER_TARGETS", []) or []
        if str(item or "").strip()
    }
    if planned_target not in allowed_targets:
        return rows_out

    teacher_path_raw = str(
        getattr(settings, "ROADMAP_NEXTSTEP_V4_HAIRCARE_CORECHAIN_TEACHER_MODEL_PATH", "") or ""
    ).strip()
    teacher_weight = max(
        0.0,
        _to_float(getattr(settings, "ROADMAP_NEXTSTEP_V4_HAIRCARE_CORECHAIN_TEACHER_WEIGHT", 0.75)),
    )
    if not teacher_path_raw or teacher_weight <= 0.0:
        return rows_out

    teacher_artifact = _load_model_for_path(teacher_path_raw)
    if not isinstance(teacher_artifact, dict):
        return rows_out
    if str(teacher_artifact.get("task") or "").strip() != "roadmap_nextstep_v4_ranking":
        return rows_out

    teacher_rows = _predict_with_v4_artifact_from_sources(
        artifact=teacher_artifact,
        category=category,
        now_utc=now_utc,
        items=items,
        profile=profile,
        context_products=context_products,
        catalog_products=catalog_products,
        planned_target_product_type=planned_target_product_type,
        planned_target_step_index=planned_target_step_index,
        candidate_types=candidate_types,
        allow_teacher_rerank=False,
    )
    if not teacher_rows:
        return rows_out

    blended_rows = blend_prediction_rows(
        rows_out,
        teacher_rows,
        overlay_weight=teacher_weight,
        overlay_label="teacher",
    )
    if not blended_rows:
        return rows_out

    policy_name = "haircare_corechain_teacher_rerank"
    for row in blended_rows:
        components = row.get("blend_components") if isinstance(row.get("blend_components"), dict) else {}
        base_score = _to_float(components.get("base_score"))
        blended_score = _to_float(row.get("score"))
        policy_bias = float(round(blended_score - base_score, 6))
        if abs(policy_bias) <= 1e-9:
            continue
        runtime_policy_biases = {
            str(k): _to_float(v)
            for k, v in (row.get("runtime_policy_biases") or {}).items()
            if str(k).strip()
        }
        runtime_policies = [str(x).strip() for x in (row.get("runtime_policies") or []) if str(x).strip()]
        runtime_policy_biases[policy_name] = policy_bias
        if policy_name not in runtime_policies:
            runtime_policies.append(policy_name)
        row["runtime_policy_biases"] = runtime_policy_biases
        row["runtime_policies"] = runtime_policies
        row["runtime_bias"] = float(round(_to_float(row.get("runtime_bias")) + policy_bias, 6))
    return blended_rows


def _apply_runtime_scalp_teacher_rerank(
    *,
    category: str,
    rows_out: list[dict[str, Any]],
    now_utc,
    items: list[dict[str, Any]] | None,
    profile: Any | None,
    context_products: list[dict[str, Any]] | None,
    catalog_products: list[dict[str, Any]] | None,
    planned_target_product_type: str | None,
    planned_target_step_index: int | None,
    candidate_types: list[str] | None,
) -> list[dict[str, Any]]:
    if not bool(getattr(settings, "ROADMAP_NEXTSTEP_V4_HAIRCARE_SCALP_TEACHER_RERANK_ENABLED", False)):
        return rows_out
    if str(category or "").strip().lower() != "haircare":
        return rows_out
    if not rows_out:
        return rows_out

    planned_target = str(planned_target_product_type or "").strip().lower()
    allowed_targets = {
        str(item or "").strip().lower()
        for item in getattr(settings, "ROADMAP_NEXTSTEP_V4_HAIRCARE_SCALP_TEACHER_TARGETS", []) or []
        if str(item or "").strip()
    }
    if planned_target not in allowed_targets:
        return rows_out

    teacher_path_raw = str(
        getattr(settings, "ROADMAP_NEXTSTEP_V4_HAIRCARE_SCALP_TEACHER_MODEL_PATH", "") or ""
    ).strip()
    teacher_weight = max(
        0.0,
        _to_float(getattr(settings, "ROADMAP_NEXTSTEP_V4_HAIRCARE_SCALP_TEACHER_WEIGHT", 0.75)),
    )
    if not teacher_path_raw or teacher_weight <= 0.0:
        return rows_out

    teacher_artifact = _load_model_for_path(teacher_path_raw)
    if not isinstance(teacher_artifact, dict):
        return rows_out
    if str(teacher_artifact.get("task") or "").strip() != "roadmap_nextstep_v4_ranking":
        return rows_out

    teacher_rows = _predict_with_v4_artifact_from_sources(
        artifact=teacher_artifact,
        category=category,
        now_utc=now_utc,
        items=items,
        profile=profile,
        context_products=context_products,
        catalog_products=catalog_products,
        planned_target_product_type=planned_target_product_type,
        planned_target_step_index=planned_target_step_index,
        candidate_types=candidate_types,
        allow_teacher_rerank=False,
    )
    if not teacher_rows:
        return rows_out

    blended_rows = blend_prediction_rows(
        rows_out,
        teacher_rows,
        overlay_weight=teacher_weight,
        overlay_label="teacher",
    )
    if not blended_rows:
        return rows_out

    policy_name = "haircare_scalp_teacher_rerank"
    for row in blended_rows:
        components = row.get("blend_components") if isinstance(row.get("blend_components"), dict) else {}
        base_score = _to_float(components.get("base_score"))
        blended_score = _to_float(row.get("score"))
        policy_bias = float(round(blended_score - base_score, 6))
        if abs(policy_bias) <= 1e-9:
            continue
        runtime_policy_biases = {
            str(k): _to_float(v)
            for k, v in (row.get("runtime_policy_biases") or {}).items()
            if str(k).strip()
        }
        runtime_policies = [str(x).strip() for x in (row.get("runtime_policies") or []) if str(x).strip()]
        runtime_policy_biases[policy_name] = policy_bias
        if policy_name not in runtime_policies:
            runtime_policies.append(policy_name)
        row["runtime_policy_biases"] = runtime_policy_biases
        row["runtime_policies"] = runtime_policies
        row["runtime_bias"] = float(round(_to_float(row.get("runtime_bias")) + policy_bias, 6))
    return blended_rows


def _build_v4_feature_frame_from_sources(
    *,
    artifact: dict[str, Any],
    category: str,
    now_utc,
    items: list[dict[str, Any]] | None,
    profile: Any | None,
    context_products: list[dict[str, Any]] | None,
    catalog_products: list[dict[str, Any]] | None,
    planned_target_product_type: str | None,
    planned_target_step_index: int | None,
    candidate_types: list[str] | None,
) -> tuple[Any, list[str], list[str], list[str]]:
    if pd is None:
        return None, [], [], []

    feature_columns = [str(x) for x in (artifact.get("feature_columns") or []) if str(x)]
    if not feature_columns:
        return None, [], [], []

    categorical_features = [
        str(x) for x in (artifact.get("categorical_features") or []) if str(x)
    ]
    numeric_features = [str(x) for x in (artifact.get("numeric_features") or []) if str(x)]

    category = str(category or "").strip().lower()
    if not category:
        return None, [], [], []

    candidates = [str(x).strip().lower() for x in (candidate_types or []) if str(x).strip()]
    if not candidates:
        candidates = [
            str(x).strip().lower()
            for x in ((artifact.get("candidate_types_by_category") or {}).get(category) or [])
            if str(x).strip()
        ]
    candidates = list(dict.fromkeys(candidates))
    if not candidates:
        return None, [], [], []

    now_utc = now_utc.astimezone(dt_timezone.utc)
    since_90d = now_utc - timedelta(days=90)

    normalized_items: list[dict[str, Any]] = []
    for row in items or []:
        item_category = str(row.get("category") or "").strip().lower()
        item_type = str(row.get("product_type") or "").strip().lower()
        if not item_category or not item_type:
            continue
        slot_value = str(row.get("slot") or "").strip().lower()
        if item_category == "fragrance" and not slot_value:
            slot_value = slot_of_fragrance(
                _safe_dict(row.get("attrs")),
                raw_meta=_safe_dict(row.get("raw_meta")),
            )
        ts = row.get("ts")
        if ts is None:
            continue
        normalized_items.append(
            {
                "ts": ts.astimezone(dt_timezone.utc),
                "tx_id": int(row.get("tx_id") or 0),
                "tx_total": _to_float(row.get("tx_total")),
                "category": item_category,
                "product_type": item_type,
                "concerns": row.get("concerns") if isinstance(row.get("concerns"), list) else [],
                "actives": row.get("actives") if isinstance(row.get("actives"), list) else [],
                "flags": row.get("flags") if isinstance(row.get("flags"), list) else [],
                "supported_skin_types": (
                    row.get("supported_skin_types")
                    if isinstance(row.get("supported_skin_types"), list)
                    else []
                ),
                "attrs": _safe_dict(row.get("attrs")),
                "ingredients_inci": str(row.get("ingredients_inci") or ""),
                "raw_meta": _safe_dict(row.get("raw_meta")),
                "quantity": max(1, int(row.get("quantity") or 0)),
                "slot": slot_value,
            }
        )
    normalized_items.sort(key=lambda row: (row["ts"], int(row.get("tx_id") or 0)))

    profile_sig = profile_signature(profile)

    normalized_context_products: list[dict[str, Any]] = []
    for row in context_products or []:
        item_category = str(row.get("category") or "").strip().lower()
        item_type = str(row.get("product_type") or "").strip().lower()
        if item_category != category or not item_type:
            continue
        slot_value = str(row.get("slot") or "").strip().lower()
        if item_category == "fragrance" and not slot_value:
            slot_value = slot_of_fragrance(
                _safe_dict(row.get("attrs")),
                raw_meta=_safe_dict(row.get("raw_meta")),
            )
        normalized_context_products.append(
            {
                "category": item_category,
                "product_type": item_type,
                "concerns": row.get("concerns") if isinstance(row.get("concerns"), list) else [],
                "actives": row.get("actives") if isinstance(row.get("actives"), list) else [],
                "flags": row.get("flags") if isinstance(row.get("flags"), list) else [],
                "supported_skin_types": (
                    row.get("supported_skin_types")
                    if isinstance(row.get("supported_skin_types"), list)
                    else []
                ),
                "attrs": _safe_dict(row.get("attrs")),
                "ingredients_inci": str(row.get("ingredients_inci") or ""),
                "raw_meta": _safe_dict(row.get("raw_meta")),
                "slot": slot_value,
            }
        )

    last_types: list[str] = []
    last_categories: list[str] = []
    slot_counter: dict[str, int] = {slot: 0 for slot in SLOTS}
    owned_counts_all: dict[tuple[str, str], int] = {}
    candidate_owned_counter: dict[str, int] = {}
    candidate_seen_90d_counter: dict[str, int] = {}
    candidate_last_seen_at: dict[str, Any] = {}
    anchor_item: dict[str, Any] | None = None

    last_ts_in_category = None
    tx_ids_90d: set[int] = set()
    tx_amount_90d: dict[int, float] = {}

    for row in reversed(normalized_items):
        item_category = str(row["category"])
        item_type = str(row["product_type"])
        qty = int(row["quantity"])
        ts = row["ts"]

        if item_type and len(last_types) < 5:
            last_types.append(item_type)
        if item_category and len(last_categories) < 5:
            last_categories.append(item_category)

        if item_category and item_type:
            key = (item_category, item_type)
            owned_counts_all[key] = int(owned_counts_all.get(key, 0) + qty)

        if item_category == category:
            candidate_key = str(row.get("slot") or "") if category == "fragrance" else item_type
            if candidate_key:
                candidate_owned_counter[candidate_key] = int(
                    candidate_owned_counter.get(candidate_key, 0) + qty
                )
                if candidate_key not in candidate_last_seen_at:
                    candidate_last_seen_at[candidate_key] = ts
            if last_ts_in_category is None:
                last_ts_in_category = ts
                anchor_item = row
            if ts >= since_90d:
                tx_id = int(row["tx_id"])
                tx_ids_90d.add(tx_id)
                tx_amount_90d[tx_id] = float(row["tx_total"])
                if candidate_key:
                    candidate_seen_90d_counter[candidate_key] = int(
                        candidate_seen_90d_counter.get(candidate_key, 0) + qty
                    )

        if item_category == "fragrance":
            slot_value = str(row.get("slot") or "")
            if slot_value in slot_counter:
                slot_counter[slot_value] += qty

    days_since_last_purchase = -1
    if last_ts_in_category is not None:
        days_since_last_purchase = int((now_utc.date() - last_ts_in_category.date()).days)

    planned_target_type_norm = str(planned_target_product_type or "").strip().lower()
    planned_target_index_int = int(planned_target_step_index or 0)

    base: dict[str, Any] = {
        "category": category,
        "month_of_year": int(now_utc.month),
        "day_of_week": int(now_utc.weekday()),
        "days_since_last_purchase_in_category": int(days_since_last_purchase),
        "tx_count_90d_category": int(len(tx_ids_90d)),
        "tx_amount_90d_category": float(round(sum(tx_amount_90d.values()), 4)),
        "owned_slot_warm_day": int(slot_counter.get("warm_day", 0)),
        "owned_slot_warm_evening": int(slot_counter.get("warm_evening", 0)),
        "owned_slot_cold_day": int(slot_counter.get("cold_day", 0)),
        "owned_slot_cold_evening": int(slot_counter.get("cold_evening", 0)),
    }
    for idx in range(5):
        base[f"last{idx + 1}_product_type"] = (
            str(last_types[idx]) if idx < len(last_types) else "__none__"
        )
        base[f"last{idx + 1}_category"] = (
            str(last_categories[idx]) if idx < len(last_categories) else "__none__"
        )

    owned_feature_columns = [str(x) for x in (artifact.get("owned_feature_columns") or []) if str(x)]
    for col in owned_feature_columns:
        base[col] = 0

    owned_feature_map = artifact.get("owned_feature_map") or {}
    if isinstance(owned_feature_map, dict):
        for col, raw in owned_feature_map.items():
            col_name = str(col or "").strip()
            if not col_name:
                continue
            raw_info = raw if isinstance(raw, dict) else {}
            cat = str(raw_info.get("category") or "").strip().lower()
            ptype = str(raw_info.get("product_type") or "").strip().lower()
            if not cat or not ptype:
                continue
            base[col_name] = int(owned_counts_all.get((cat, ptype), 0))

    anchor_source_item = normalized_context_products[0] if normalized_context_products else anchor_item
    rules_chain = effective_nextstep_rules_chain(
        category=category,
        rules_chain=[
            str(x).strip().lower()
            for x in ((artifact.get("rules_chain_by_category") or {}).get(category) or [])
            if str(x)
        ],
        planned_target_product_type=planned_target_type_norm,
        profile_sig=profile_sig,
        anchor_product_type=anchor_source_item.get("product_type") if isinstance(anchor_source_item, dict) else "",
    )
    pos_by_candidate = {token: idx for idx, token in enumerate(rules_chain)}
    pop_map_raw = (artifact.get("candidate_popularity_in_train_by_category") or {}).get(category) or {}
    pop_map: dict[str, float] = {str(k).strip().lower(): _to_float(v) for k, v in pop_map_raw.items()}
    candidate_set = set(candidates)
    candidate_catalog_summaries = build_candidate_catalog_summaries(
        [
            row
            for row in (catalog_products or [])
            if str(row.get("category") or "").strip().lower() == category
            and str(row.get("product_type") or "").strip().lower() in candidate_set
        ]
    )
    anchor_sig = product_signature(anchor_source_item)

    context_candidate_tokens = [
        str(row.get("slot") or "") if category == "fragrance" else str(row.get("product_type") or "")
        for row in normalized_context_products
        if str((row.get("slot") or "") if category == "fragrance" else (row.get("product_type") or "")).strip()
    ]
    history_candidate_tokens = [
        token
        for token in (
            [str(row.get("slot") or "") for row in reversed(normalized_items) if str(row["category"]) == "fragrance"]
            if category == "fragrance"
            else [str(row["product_type"]) for row in reversed(normalized_items) if str(row["category"]) == category]
        )
        if token
    ]
    recent_candidate_tokens = _ordered_unique_tokens(context_candidate_tokens + history_candidate_tokens)[:5]
    anchor_chain_token = (
        recent_candidate_tokens[0]
        if recent_candidate_tokens
        else str(anchor_sig.get("product_type") or "").strip().lower()
    )
    last1_chain_token = recent_candidate_tokens[0] if recent_candidate_tokens else ""
    last2_chain_token = recent_candidate_tokens[1] if len(recent_candidate_tokens) > 1 else ""
    recent_candidate_set = set(recent_candidate_tokens[:3])

    rows: list[dict[str, Any]] = []
    for candidate in candidates:
        seen_count_last5 = int(sum(1 for token in recent_candidate_tokens if token == candidate))
        row = dict(base)
        row.update(build_base_content_features(profile_sig, anchor_sig))
        row["candidate_type"] = candidate
        row["candidate_is_fragrance_slot"] = int(candidate in SLOTS)
        row["candidate_position_in_chain"] = int(pos_by_candidate.get(candidate, -1))
        row["candidate_popularity_in_train"] = float(pop_map.get(candidate, 0.0))
        row["candidate_matches_last1"] = int(bool(recent_candidate_tokens and recent_candidate_tokens[0] == candidate))
        row["candidate_matches_last3_any"] = int(candidate in recent_candidate_set)
        row["candidate_seen_count_last5"] = int(seen_count_last5)
        row["candidate_owned_count_in_category"] = int(candidate_owned_counter.get(candidate, 0))
        row["candidate_seen_90d_count_in_category"] = int(candidate_seen_90d_counter.get(candidate, 0))
        row["candidate_days_since_last_seen_in_category"] = int(
            (now_utc.date() - candidate_last_seen_at[candidate].date()).days
        ) if candidate in candidate_last_seen_at else -1
        row.update(
            build_candidate_content_features(
                candidate_catalog_summaries.get((category, candidate)),
                profile_sig,
                anchor_sig,
                candidate_type=candidate,
            )
        )
        row.update(
            build_nextstep_plan_state_features(
                rules_chain=rules_chain,
                candidate_type=candidate,
                planned_target_product_type=planned_target_type_norm,
                planned_target_step_index=planned_target_index_int,
            )
        )
        row.update(
            build_chain_transition_features(
                rules_chain=rules_chain,
                candidate_type=candidate,
                anchor_product_type=anchor_chain_token,
                last1_product_type=last1_chain_token,
                last2_product_type=last2_chain_token,
            )
        )
        for col in feature_columns:
            if col in row:
                continue
            if col in numeric_features:
                row[col] = 0.0
            else:
                row[col] = "__none__"
        rows.append(row)

    if not rows:
        return None, feature_columns, categorical_features, numeric_features

    frame = pd.DataFrame(rows)
    for col in categorical_features:
        if col in frame.columns:
            frame[col] = frame[col].fillna("__none__").astype(str)
    for col in numeric_features:
        if col in frame.columns:
            frame[col] = pd.to_numeric(frame[col], errors="coerce").fillna(0.0)
    return frame, feature_columns, categorical_features, numeric_features


def _predict_with_v4_artifact_from_sources(
    *,
    artifact: dict[str, Any],
    category: str,
    now_utc,
    items: list[dict[str, Any]] | None,
    profile: Any | None,
    context_products: list[dict[str, Any]] | None,
    catalog_products: list[dict[str, Any]] | None,
    planned_target_product_type: str | None,
    planned_target_step_index: int | None,
    candidate_types: list[str] | None,
    allow_teacher_rerank: bool = True,
) -> list[dict[str, Any]]:
    if pd is None:
        return []

    model = artifact.get("model")
    if model is None:
        return []

    feature_columns = [str(x) for x in (artifact.get("feature_columns") or []) if str(x)]
    if not feature_columns:
        return []

    categorical_features = [
        str(x) for x in (artifact.get("categorical_features") or []) if str(x)
    ]
    numeric_features = [str(x) for x in (artifact.get("numeric_features") or []) if str(x)]

    category = str(category or "").strip().lower()
    if not category:
        return []

    candidates = [str(x).strip().lower() for x in (candidate_types or []) if str(x).strip()]
    if not candidates:
        candidates = [
            str(x).strip().lower()
            for x in ((artifact.get("candidate_types_by_category") or {}).get(category) or [])
            if str(x).strip()
        ]
    candidates = list(dict.fromkeys(candidates))
    if not candidates:
        return []

    now_utc = now_utc.astimezone(dt_timezone.utc)
    since_90d = now_utc - timedelta(days=90)

    normalized_items: list[dict[str, Any]] = []
    for row in items or []:
        item_category = str(row.get("category") or "").strip().lower()
        item_type = str(row.get("product_type") or "").strip().lower()
        if not item_category or not item_type:
            continue
        slot_value = str(row.get("slot") or "").strip().lower()
        if item_category == "fragrance" and not slot_value:
            slot_value = slot_of_fragrance(
                _safe_dict(row.get("attrs")),
                raw_meta=_safe_dict(row.get("raw_meta")),
            )
        ts = row.get("ts")
        if ts is None:
            continue
        normalized_items.append(
            {
                "ts": ts.astimezone(dt_timezone.utc),
                "tx_id": int(row.get("tx_id") or 0),
                "tx_total": _to_float(row.get("tx_total")),
                "category": item_category,
                "product_type": item_type,
                "concerns": row.get("concerns") if isinstance(row.get("concerns"), list) else [],
                "actives": row.get("actives") if isinstance(row.get("actives"), list) else [],
                "flags": row.get("flags") if isinstance(row.get("flags"), list) else [],
                "supported_skin_types": (
                    row.get("supported_skin_types")
                    if isinstance(row.get("supported_skin_types"), list)
                    else []
                ),
                "attrs": _safe_dict(row.get("attrs")),
                "ingredients_inci": str(row.get("ingredients_inci") or ""),
                "raw_meta": _safe_dict(row.get("raw_meta")),
                "quantity": max(1, int(row.get("quantity") or 0)),
                "slot": slot_value,
            }
        )
    normalized_items.sort(key=lambda row: (row["ts"], int(row.get("tx_id") or 0)))

    profile_sig = profile_signature(profile)

    normalized_context_products: list[dict[str, Any]] = []
    for row in context_products or []:
        item_category = str(row.get("category") or "").strip().lower()
        item_type = str(row.get("product_type") or "").strip().lower()
        if item_category != category or not item_type:
            continue
        slot_value = str(row.get("slot") or "").strip().lower()
        if item_category == "fragrance" and not slot_value:
            slot_value = slot_of_fragrance(
                _safe_dict(row.get("attrs")),
                raw_meta=_safe_dict(row.get("raw_meta")),
            )
        normalized_context_products.append(
            {
                "category": item_category,
                "product_type": item_type,
                "concerns": row.get("concerns") if isinstance(row.get("concerns"), list) else [],
                "actives": row.get("actives") if isinstance(row.get("actives"), list) else [],
                "flags": row.get("flags") if isinstance(row.get("flags"), list) else [],
                "supported_skin_types": (
                    row.get("supported_skin_types")
                    if isinstance(row.get("supported_skin_types"), list)
                    else []
                ),
                "attrs": _safe_dict(row.get("attrs")),
                "ingredients_inci": str(row.get("ingredients_inci") or ""),
                "raw_meta": _safe_dict(row.get("raw_meta")),
                "slot": slot_value,
            }
        )

    last_types: list[str] = []
    last_categories: list[str] = []
    slot_counter: dict[str, int] = {slot: 0 for slot in SLOTS}
    owned_counts_all: dict[tuple[str, str], int] = {}
    candidate_owned_counter: dict[str, int] = {}
    candidate_seen_90d_counter: dict[str, int] = {}
    candidate_last_seen_at: dict[str, Any] = {}
    anchor_item: dict[str, Any] | None = None

    last_ts_in_category = None
    tx_ids_90d: set[int] = set()
    tx_amount_90d: dict[int, float] = {}

    for row in reversed(normalized_items):
        item_category = str(row["category"])
        item_type = str(row["product_type"])
        qty = int(row["quantity"])
        ts = row["ts"]

        if item_type and len(last_types) < 5:
            last_types.append(item_type)
        if item_category and len(last_categories) < 5:
            last_categories.append(item_category)

        if item_category and item_type:
            key = (item_category, item_type)
            owned_counts_all[key] = int(owned_counts_all.get(key, 0) + qty)

        if item_category == category:
            candidate_key = str(row.get("slot") or "") if category == "fragrance" else item_type
            if candidate_key:
                candidate_owned_counter[candidate_key] = int(
                    candidate_owned_counter.get(candidate_key, 0) + qty
                )
                if candidate_key not in candidate_last_seen_at:
                    candidate_last_seen_at[candidate_key] = ts
            if last_ts_in_category is None:
                last_ts_in_category = ts
                anchor_item = row
            if ts >= since_90d:
                tx_id = int(row["tx_id"])
                tx_ids_90d.add(tx_id)
                tx_amount_90d[tx_id] = float(row["tx_total"])
                if candidate_key:
                    candidate_seen_90d_counter[candidate_key] = int(
                        candidate_seen_90d_counter.get(candidate_key, 0) + qty
                    )

        if item_category == "fragrance":
            slot_value = str(row.get("slot") or "")
            if slot_value in slot_counter:
                slot_counter[slot_value] += qty

    days_since_last_purchase = -1
    if last_ts_in_category is not None:
        days_since_last_purchase = int((now_utc.date() - last_ts_in_category.date()).days)

    has_context_anchor = bool(normalized_context_products)
    planned_target_type_norm = str(planned_target_product_type or "").strip().lower()
    planned_target_index_int = int(planned_target_step_index or 0)

    base: dict[str, Any] = {
        "category": category,
        "month_of_year": int(now_utc.month),
        "day_of_week": int(now_utc.weekday()),
        "days_since_last_purchase_in_category": int(days_since_last_purchase),
        "tx_count_90d_category": int(len(tx_ids_90d)),
        "tx_amount_90d_category": float(round(sum(tx_amount_90d.values()), 4)),
        "owned_slot_warm_day": int(slot_counter.get("warm_day", 0)),
        "owned_slot_warm_evening": int(slot_counter.get("warm_evening", 0)),
        "owned_slot_cold_day": int(slot_counter.get("cold_day", 0)),
        "owned_slot_cold_evening": int(slot_counter.get("cold_evening", 0)),
    }
    for idx in range(5):
        base[f"last{idx + 1}_product_type"] = (
            str(last_types[idx]) if idx < len(last_types) else "__none__"
        )
        base[f"last{idx + 1}_category"] = (
            str(last_categories[idx]) if idx < len(last_categories) else "__none__"
        )

    owned_feature_columns = [str(x) for x in (artifact.get("owned_feature_columns") or []) if str(x)]
    for col in owned_feature_columns:
        base[col] = 0

    owned_feature_map = artifact.get("owned_feature_map") or {}
    if isinstance(owned_feature_map, dict):
        for col, raw in owned_feature_map.items():
            col_name = str(col or "").strip()
            if not col_name:
                continue
            raw_info = raw if isinstance(raw, dict) else {}
            cat = str(raw_info.get("category") or "").strip().lower()
            ptype = str(raw_info.get("product_type") or "").strip().lower()
            if not cat or not ptype:
                continue
            base[col_name] = int(owned_counts_all.get((cat, ptype), 0))

    anchor_source_item = normalized_context_products[0] if normalized_context_products else anchor_item
    rules_chain = effective_nextstep_rules_chain(
        category=category,
        rules_chain=[
            str(x).strip().lower()
            for x in ((artifact.get("rules_chain_by_category") or {}).get(category) or [])
            if str(x)
        ],
        planned_target_product_type=planned_target_type_norm,
        profile_sig=profile_sig,
        anchor_product_type=anchor_source_item.get("product_type") if isinstance(anchor_source_item, dict) else "",
    )
    pos_by_candidate = {token: idx for idx, token in enumerate(rules_chain)}
    pop_map_raw = (artifact.get("candidate_popularity_in_train_by_category") or {}).get(category) or {}
    pop_map: dict[str, float] = {str(k).strip().lower(): _to_float(v) for k, v in pop_map_raw.items()}
    candidate_set = set(candidates)
    candidate_catalog_summaries = build_candidate_catalog_summaries(
        [
            row
            for row in (catalog_products or [])
            if str(row.get("category") or "").strip().lower() == category
            and str(row.get("product_type") or "").strip().lower() in candidate_set
        ]
    )
    anchor_sig = product_signature(anchor_source_item)

    context_candidate_tokens = [
        str(row.get("slot") or "") if category == "fragrance" else str(row.get("product_type") or "")
        for row in normalized_context_products
        if str((row.get("slot") or "") if category == "fragrance" else (row.get("product_type") or "")).strip()
    ]
    history_candidate_tokens = [
        token
        for token in (
            [str(row.get("slot") or "") for row in reversed(normalized_items) if str(row["category"]) == "fragrance"]
            if category == "fragrance"
            else [str(row["product_type"]) for row in reversed(normalized_items) if str(row["category"]) == category]
        )
        if token
    ]
    recent_candidate_tokens = _ordered_unique_tokens(context_candidate_tokens + history_candidate_tokens)[:5]
    anchor_chain_token = (
        recent_candidate_tokens[0]
        if recent_candidate_tokens
        else str(anchor_sig.get("product_type") or "").strip().lower()
    )
    last1_chain_token = recent_candidate_tokens[0] if recent_candidate_tokens else ""
    last2_chain_token = recent_candidate_tokens[1] if len(recent_candidate_tokens) > 1 else ""
    recent_candidate_set = set(recent_candidate_tokens[:3])

    rows: list[dict[str, Any]] = []
    for candidate in candidates:
        seen_count_last5 = int(sum(1 for token in recent_candidate_tokens if token == candidate))
        row = dict(base)
        row.update(build_base_content_features(profile_sig, anchor_sig))
        row["candidate_type"] = candidate
        row["candidate_is_fragrance_slot"] = int(candidate in SLOTS)
        row["candidate_position_in_chain"] = int(pos_by_candidate.get(candidate, -1))
        row["candidate_popularity_in_train"] = float(pop_map.get(candidate, 0.0))
        row["candidate_matches_last1"] = int(bool(recent_candidate_tokens and recent_candidate_tokens[0] == candidate))
        row["candidate_matches_last3_any"] = int(candidate in recent_candidate_set)
        row["candidate_seen_count_last5"] = int(seen_count_last5)
        row["candidate_owned_count_in_category"] = int(candidate_owned_counter.get(candidate, 0))
        row["candidate_seen_90d_count_in_category"] = int(candidate_seen_90d_counter.get(candidate, 0))
        row["candidate_days_since_last_seen_in_category"] = int(
            (now_utc.date() - candidate_last_seen_at[candidate].date()).days
        ) if candidate in candidate_last_seen_at else -1
        row.update(
            build_candidate_content_features(
                candidate_catalog_summaries.get((category, candidate)),
                profile_sig,
                anchor_sig,
                candidate_type=candidate,
            )
        )
        row.update(
            build_nextstep_plan_state_features(
                rules_chain=rules_chain,
                candidate_type=candidate,
                planned_target_product_type=planned_target_type_norm,
                planned_target_step_index=planned_target_index_int,
            )
        )
        row.update(
            build_chain_transition_features(
                rules_chain=rules_chain,
                candidate_type=candidate,
                anchor_product_type=anchor_chain_token,
                last1_product_type=last1_chain_token,
                last2_product_type=last2_chain_token,
            )
        )
        for col in feature_columns:
            if col in row:
                continue
            if col in numeric_features:
                row[col] = 0.0
            else:
                row[col] = "__none__"
        rows.append(row)

    if not rows:
        return []

    frame = pd.DataFrame(rows)
    X = frame[feature_columns].copy()

    for col in categorical_features:
        if col in X.columns:
            X[col] = X[col].fillna("__none__").astype(str)
    for col in numeric_features:
        if col in X.columns:
            X[col] = pd.to_numeric(X[col], errors="coerce").fillna(0.0)

    preprocessor = artifact.get("preprocessor")
    model_type = str(artifact.get("model_type") or "").strip().lower()

    try:
        if preprocessor is not None:
            X_model = preprocessor.transform(X)
            if hasattr(model, "decision_function"):
                raw_scores = model.decision_function(X_model)
            elif hasattr(model, "predict_proba"):
                raw_scores = model.predict_proba(X_model)
            else:
                raw_scores = model.predict(X_model)
        else:
            if model_type.startswith("lightgbm"):
                for col in categorical_features:
                    if col in X.columns:
                        X[col] = X[col].astype("category")
            raw_scores = model.predict(X)
    except Exception:
        return []

    if np is not None:
        score_arr = np.asarray(raw_scores)
        if score_arr.ndim > 1:
            if score_arr.shape[1] >= 2:
                score_list = [float(x) for x in score_arr[:, -1].tolist()]
            else:
                score_list = [float(x) for x in score_arr.reshape(-1).tolist()]
        else:
            score_list = [float(x) for x in score_arr.tolist()]
    else:
        score_list = []
        for raw_score in list(raw_scores):
            if isinstance(raw_score, (list, tuple)):
                score_list.append(float(raw_score[-1]))
            else:
                score_list.append(float(raw_score))

    score_list, progression_biases = _apply_runtime_progression_bias(
        category=category,
        rows=rows,
        score_list=score_list,
        days_since_last_purchase=days_since_last_purchase,
        has_context_anchor=has_context_anchor,
    )
    score_list, scalp_biases = _apply_runtime_scalp_planned_target_rerank(
        category=category,
        rows=rows,
        score_list=score_list,
    )
    score_list, leavein_biases = _apply_runtime_leavein_planned_target_rerank(
        category=category,
        rows=rows,
        score_list=score_list,
    )
    policy_bias_components: dict[str, list[float]] = {
        "haircare_progression_bias": progression_biases,
        "haircare_scalp_rerank": scalp_biases,
        "haircare_leavein_rerank": leavein_biases,
    }
    runtime_biases = [
        float(
            round(
                float(progression_biases[idx])
                + float(scalp_biases[idx])
                + float(leavein_biases[idx]),
                6,
            )
        )
        for idx in range(len(score_list))
    ]
    temperature = _to_float(artifact.get("temperature", 1.0)) or 1.0
    prob_list = _group_softmax(score_list, temperature)

    rows_out: list[dict[str, Any]] = []
    for idx, candidate in enumerate(candidates):
        runtime_policy_biases: dict[str, float] = {}
        runtime_policies: list[str] = []
        for policy_name, component_values in policy_bias_components.items():
            if idx >= len(component_values):
                continue
            component_bias = float(round(float(component_values[idx]), 6))
            if abs(component_bias) <= 1e-9:
                continue
            runtime_policy_biases[str(policy_name)] = component_bias
            runtime_policies.append(str(policy_name))
        rows_out.append(
            {
                "candidate_type": candidate,
                "product_type": candidate,
                "score": float(prob_list[idx]),
                "raw_score": float(score_list[idx]),
                "runtime_bias": float(runtime_biases[idx]) if idx < len(runtime_biases) else 0.0,
                "runtime_policy_biases": runtime_policy_biases,
                "runtime_policies": runtime_policies,
                "model_type": model_type,
            }
        )

    rows_out.sort(key=lambda row: float(row.get("score", 0.0)), reverse=True)
    if allow_teacher_rerank:
        rows_out = _apply_runtime_corechain_teacher_rerank(
            category=category,
            rows_out=rows_out,
            now_utc=now_utc,
            items=normalized_items,
            profile=profile,
            context_products=normalized_context_products,
            catalog_products=catalog_products,
            planned_target_product_type=planned_target_product_type,
            planned_target_step_index=planned_target_step_index,
            candidate_types=candidates,
        )
        rows_out = _apply_runtime_scalp_teacher_rerank(
            category=category,
            rows_out=rows_out,
            now_utc=now_utc,
            items=normalized_items,
            profile=profile,
            context_products=normalized_context_products,
            catalog_products=catalog_products,
            planned_target_product_type=planned_target_product_type,
            planned_target_step_index=planned_target_step_index,
            candidate_types=candidates,
        )
    return rows_out


def _predict_with_v4_artifact(
    *,
    artifact: dict[str, Any],
    user_id: int,
    category: str,
    context_product_ids: list[int] | None,
    planned_target_product_type: str | None,
    planned_target_step_index: int | None,
    candidate_types: list[str] | None,
) -> list[dict[str, Any]]:
    category = str(category or "").strip().lower()
    if not category:
        return []

    now_utc = timezone.now().astimezone(dt_timezone.utc)
    tx_rows = list(
        TransactionItem.objects.filter(
            transaction__user_id=int(user_id),
            transaction__created_at__lte=now_utc,
        )
        .values(
            "id",
            "transaction__id",
            "transaction__created_at",
            "transaction__total_amount",
            "product__category",
            "product__product_type",
            "quantity",
            "product__concerns",
            "product__actives",
            "product__flags",
            "product__supported_skin_types",
            "product__attrs",
            "product__ingredients_inci",
            "product__raw_meta",
        )
        .order_by("transaction__created_at", "transaction__id", "id")
    )

    profile_row = CustomerProfile.objects.filter(user_id=int(user_id)).first()
    context_rows = list(
        Product.objects.filter(id__in=[int(x) for x in (context_product_ids or []) if str(x).strip()])
        .values(
            "id",
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
    context_by_id = {int(row["id"]): row for row in context_rows}
    ordered_context_products: list[dict[str, Any]] = []
    for raw_id in context_product_ids or []:
        try:
            row = context_by_id.get(int(raw_id))
        except Exception:
            row = None
        if row:
            ordered_context_products.append(dict(row))

    items: list[dict[str, Any]] = []
    for row in tx_rows:
        item_category = str(row.get("product__category") or "").strip().lower()
        item_type = str(row.get("product__product_type") or "").strip().lower()
        slot_value = ""
        if item_category == "fragrance":
            slot_value = slot_of_fragrance(
                _safe_dict(row.get("product__attrs")),
                raw_meta=_safe_dict(row.get("product__raw_meta")),
            )
        items.append(
            {
                "ts": row["transaction__created_at"].astimezone(dt_timezone.utc),
                "tx_id": int(row["transaction__id"]),
                "tx_total": _to_float(row.get("transaction__total_amount")),
                "category": item_category,
                "product_type": item_type,
                "concerns": row.get("product__concerns") if isinstance(row.get("product__concerns"), list) else [],
                "actives": row.get("product__actives") if isinstance(row.get("product__actives"), list) else [],
                "flags": row.get("product__flags") if isinstance(row.get("product__flags"), list) else [],
                "supported_skin_types": (
                    row.get("product__supported_skin_types")
                    if isinstance(row.get("product__supported_skin_types"), list)
                    else []
                ),
                "attrs": _safe_dict(row.get("product__attrs")),
                "ingredients_inci": str(row.get("product__ingredients_inci") or ""),
                "raw_meta": _safe_dict(row.get("product__raw_meta")),
                "quantity": max(1, int(row.get("quantity") or 0)),
                "slot": slot_value,
            }
        )
    catalog_rows = list(
        Product.objects.filter(category=category).values(
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
    return _predict_with_v4_artifact_from_sources(
        artifact=artifact,
        category=category,
        now_utc=now_utc,
        items=items,
        profile=profile_row,
        context_products=ordered_context_products,
        catalog_products=catalog_rows,
        planned_target_product_type=planned_target_product_type,
        planned_target_step_index=planned_target_step_index,
        candidate_types=candidate_types,
    )


def predict_next_product_types(
    user,
    context_product_ids: list[int],
    category: str,
    planned_target_product_type: str | None = None,
    planned_target_step_index: int | None = None,
    candidate_types: list[str] | None = None,
) -> list[dict[str, Any]]:
    return predict_next_product_types_for_model_path(
        None,
        user=user,
        context_product_ids=context_product_ids,
        category=category,
        planned_target_product_type=planned_target_product_type,
        planned_target_step_index=planned_target_step_index,
        candidate_types=candidate_types,
    )


def predict_next_product_types_for_model_path(
    model_path: str | Path | None,
    user,
    context_product_ids: list[int],
    category: str,
    planned_target_product_type: str | None = None,
    planned_target_step_index: int | None = None,
    candidate_types: list[str] | None = None,
) -> list[dict[str, Any]]:
    """
    Return ranked product_type/candidate_type predictions for next roadmap step.
    If model artifact is absent or incompatible, returns [].
    """
    model = _load_model() if model_path in (None, "") else _load_model_for_path(model_path)
    if model is None:
        return []

    user_id = int(getattr(user, "id", user) or 0)
    context_ids = [int(x) for x in (context_product_ids or []) if str(x).strip()]
    category = str(category or "").strip().lower()

    if isinstance(model, dict) and str(model.get("task") or "") == "roadmap_nextstep_v4_ranking":
        raw_rows = _predict_with_v4_artifact(
            artifact=model,
            user_id=user_id,
            category=category,
            context_product_ids=context_ids,
            planned_target_product_type=planned_target_product_type,
            planned_target_step_index=planned_target_step_index,
            candidate_types=candidate_types,
        )
        return _normalize_predictions(raw_rows)

    raw = None
    try:
        if hasattr(model, "predict_next_product_types"):
            try:
                raw = model.predict_next_product_types(
                    user_id=user_id,
                    context_product_ids=context_ids,
                    category=category,
                    candidate_types=candidate_types,
                )
            except TypeError:
                raw = model.predict_next_product_types(
                    user_id=user_id,
                    context_product_ids=context_ids,
                    category=category,
                )
        elif callable(model):
            raw = model(
                user_id=user_id,
                context_product_ids=context_ids,
                category=category,
                candidate_types=candidate_types,
            )
        elif hasattr(model, "predict"):
            payload = {
                "user_id": user_id,
                "context_product_ids": context_ids,
                "category": category,
                "candidate_types": candidate_types or [],
            }
            raw = model.predict(payload)
    except Exception:
        return []

    return _normalize_predictions(raw)
