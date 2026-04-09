from __future__ import annotations

from pathlib import Path
from typing import Any


SHADOW_EVIDENCE_KEY = "shadow_evidence"
SHADOW_EVIDENCE_SOURCE = "shadow_replay_v1"
CONTROL_EVIDENCE_KEY = "baseline_control_evidence"
CONTROL_EVIDENCE_SOURCE = "baseline_control_replay_v1"
HISTORICAL_SHADOW_EVIDENCE_KEY = "historical_shadow_evidence"
HISTORICAL_SHADOW_EVIDENCE_SOURCE = "historical_anchor_replay_v1"
HISTORICAL_CONTROL_EVIDENCE_KEY = "historical_control_evidence"
HISTORICAL_CONTROL_EVIDENCE_SOURCE = "historical_anchor_control_replay_v1"
_VOLATILE_KEYS = {"evidence_generated_at"}


def _safe_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _safe_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def normalized_model_path(value: Any) -> str:
    raw = str(value or "").strip()
    if not raw:
        return ""
    try:
        return str(Path(raw).expanduser().resolve())
    except Exception:
        return str(Path(raw).expanduser())


def top_prediction_row(rows: Any) -> dict[str, Any] | None:
    best_row: dict[str, Any] | None = None
    best_score: float | None = None
    for row in _safe_list(rows):
        if not isinstance(row, dict):
            continue
        product_type = str(row.get("product_type") or row.get("candidate_type") or "").strip().lower()
        if not product_type:
            continue
        try:
            score = float(row.get("score", 0.0) or 0.0)
        except Exception:
            score = 0.0
        if best_row is None or best_score is None or score > best_score:
            best_row = row
            best_score = score
    return dict(best_row) if isinstance(best_row, dict) else None


def runtime_policy_payload(predictions: Any) -> tuple[list[str], dict[str, float]]:
    policy_names: set[str] = set()
    max_abs_bias: dict[str, float] = {}
    for row in _safe_list(predictions):
        if not isinstance(row, dict):
            continue
        for raw_policy in row.get("runtime_policies") or []:
            policy_name = str(raw_policy or "").strip()
            if policy_name:
                policy_names.add(policy_name)
        raw_biases = row.get("runtime_policy_biases")
        if not isinstance(raw_biases, dict):
            continue
        for raw_policy_name, raw_bias_value in raw_biases.items():
            policy_name = str(raw_policy_name or "").strip()
            if not policy_name:
                continue
            try:
                bias_value = abs(float(raw_bias_value or 0.0))
            except Exception:
                continue
            prev_value = float(max_abs_bias.get(policy_name, 0.0))
            if bias_value > prev_value:
                max_abs_bias[policy_name] = float(round(bias_value, 6))
    return sorted(policy_names), {
        str(k): float(v) for k, v in sorted(max_abs_bias.items(), key=lambda kv: kv[0])
    }


def build_shadow_evidence_payload(
    *,
    model_path: str,
    model_version: str,
    selected_feature_set: str,
    plan_id: int,
    category: str,
    plan_updated_at: str,
    evidence_generated_at: str,
    threshold: float,
    candidate_types: list[str],
    planned_target_product_type: str,
    planned_target_step_index: int,
    prediction_reason: str,
    predictions: list[dict[str, Any]],
    was_model_considered: bool = True,
) -> dict[str, Any]:
    predictions_trimmed = [dict(row) for row in _safe_list(predictions)[:10] if isinstance(row, dict)]
    runtime_policies, runtime_policy_max_abs_bias = runtime_policy_payload(predictions_trimmed)
    top_row = top_prediction_row(predictions_trimmed)
    top_product_type = str(
        _safe_dict(top_row).get("product_type") or _safe_dict(top_row).get("candidate_type") or ""
    ).strip().lower()
    top_score: float | None = None
    if top_row is not None:
        try:
            top_score = float(top_row.get("score", 0.0) or 0.0)
        except Exception:
            top_score = 0.0

    threshold_passed = bool(top_row is not None and top_score is not None and float(top_score) >= float(threshold))
    was_model_selected = bool(was_model_considered and threshold_passed and top_product_type)
    comparable_decision = "model_used" if was_model_selected else ("fallback" if was_model_considered else "disabled")
    comparable_reason = (
        "selected_top1"
        if was_model_selected
        else (str(prediction_reason or "").strip() or ("low_confidence" if top_row is not None else "no_predictions_or_model_unavailable"))
    )

    return {
        "enabled": bool(was_model_considered),
        "reason": str(prediction_reason or "").strip() or ("ok" if predictions_trimmed else "no_predictions_or_model_unavailable"),
        "evidence_source": SHADOW_EVIDENCE_SOURCE,
        "evidence_generated_at": str(evidence_generated_at or "").strip(),
        "model_path": normalized_model_path(model_path),
        "model_version": str(model_version or "").strip(),
        "selected_feature_set": str(selected_feature_set or "").strip(),
        "plan_id": int(plan_id),
        "category": str(category or "").strip().lower(),
        "plan_updated_at": str(plan_updated_at or "").strip(),
        "candidate_types": [str(item or "").strip().lower() for item in candidate_types if str(item or "").strip()],
        "prediction_count": int(len(predictions_trimmed)),
        "planned_target_product_type": str(planned_target_product_type or "").strip().lower(),
        "planned_target_step_index": int(planned_target_step_index or 0),
        "threshold": float(threshold),
        "top1_product_type": top_product_type,
        "top1_score": None if top_score is None else float(round(top_score, 6)),
        "threshold_passed": bool(threshold_passed),
        "was_model_considered": bool(was_model_considered),
        "was_model_selected": bool(was_model_selected),
        "comparable_decision": comparable_decision,
        "comparable_reason": comparable_reason,
        "runtime_policies": runtime_policies,
        "runtime_policy_max_abs_bias": runtime_policy_max_abs_bias,
        "predictions": predictions_trimmed,
    }


def build_control_evidence_payload(
    *,
    model_path: str,
    plan_id: int,
    category: str,
    plan_updated_at: str,
    evidence_generated_at: str,
    candidate_types: list[str],
    selected_product_type: str,
    selected_step_index: int,
    selected_step_status: str,
    planned_target_product_type: str,
    planned_target_step_index: int,
    baseline_source: str,
    was_control_available: bool,
    comparable_reason: str,
) -> dict[str, Any]:
    selected_product_type_norm = str(selected_product_type or "").strip().lower()
    return {
        "evidence_source": CONTROL_EVIDENCE_SOURCE,
        "evidence_generated_at": str(evidence_generated_at or "").strip(),
        "model_path": normalized_model_path(model_path),
        "plan_id": int(plan_id),
        "category": str(category or "").strip().lower(),
        "plan_updated_at": str(plan_updated_at or "").strip(),
        "candidate_types": [str(item or "").strip().lower() for item in candidate_types if str(item or "").strip()],
        "baseline_source": str(baseline_source or "").strip() or "current_rule_plan",
        "selected_product_type": selected_product_type_norm,
        "selected_step_index": int(selected_step_index or 0),
        "selected_step_status": str(selected_step_status or "").strip().lower(),
        "planned_target_product_type": str(planned_target_product_type or "").strip().lower(),
        "planned_target_step_index": int(planned_target_step_index or 0),
        "was_control_available": bool(was_control_available),
        "was_control_selected": bool(was_control_available and selected_product_type_norm),
        "comparable_decision": "control_used" if bool(was_control_available and selected_product_type_norm) else "disabled",
        "comparable_reason": str(comparable_reason or "").strip()
        or ("selected_current_plan_next_step" if bool(was_control_available and selected_product_type_norm) else "no_actionable_step"),
    }


def build_historical_shadow_evidence_payload(
    *,
    anchor_key: str,
    anchor_event_id: int,
    anchor_created_at: str,
    anchor_source: str,
    reconstruction_reason: str,
    reconstructed_candidate_types: list[str],
    current_snapshot_exclusion_reason: str = "",
    **kwargs,
) -> dict[str, Any]:
    payload = build_shadow_evidence_payload(**kwargs)
    payload["evidence_source"] = HISTORICAL_SHADOW_EVIDENCE_SOURCE
    payload["anchor_key"] = str(anchor_key or "").strip()
    payload["anchor_event_id"] = int(anchor_event_id or 0)
    payload["anchor_created_at"] = str(anchor_created_at or "").strip()
    payload["anchor_source"] = str(anchor_source or "").strip() or "plan_refreshed"
    payload["replay_mode"] = "historical_anchors"
    payload["reconstruction_reason"] = str(reconstruction_reason or "").strip()
    payload["reconstructed_candidate_types"] = [
        str(item or "").strip().lower()
        for item in reconstructed_candidate_types
        if str(item or "").strip()
    ]
    payload["current_snapshot_exclusion_reason"] = str(current_snapshot_exclusion_reason or "").strip()
    return payload


def build_historical_control_evidence_payload(
    *,
    anchor_key: str,
    anchor_event_id: int,
    anchor_created_at: str,
    anchor_source: str,
    reconstruction_reason: str,
    current_snapshot_exclusion_reason: str = "",
    **kwargs,
) -> dict[str, Any]:
    payload = build_control_evidence_payload(**kwargs)
    payload["evidence_source"] = HISTORICAL_CONTROL_EVIDENCE_SOURCE
    payload["anchor_key"] = str(anchor_key or "").strip()
    payload["anchor_event_id"] = int(anchor_event_id or 0)
    payload["anchor_created_at"] = str(anchor_created_at or "").strip()
    payload["anchor_source"] = str(anchor_source or "").strip() or "plan_refreshed"
    payload["replay_mode"] = "historical_anchors"
    payload["reconstruction_reason"] = str(reconstruction_reason or "").strip()
    payload["current_snapshot_exclusion_reason"] = str(current_snapshot_exclusion_reason or "").strip()
    return payload


def shadow_evidence_map(meta: dict[str, Any] | None) -> dict[str, Any]:
    ml = _safe_dict(_safe_dict(meta).get("ml"))
    evidence = ml.get(SHADOW_EVIDENCE_KEY)
    return dict(evidence) if isinstance(evidence, dict) else {}


def get_shadow_evidence_for_model_path(
    meta: dict[str, Any] | None,
    model_path: str | None,
) -> dict[str, Any] | None:
    normalized_path = normalized_model_path(model_path)
    if not normalized_path:
        return None

    evidence = shadow_evidence_map(meta)
    candidate = _safe_dict(evidence.get(normalized_path))
    if candidate:
        return candidate

    ml = _safe_dict(_safe_dict(meta).get("ml"))
    legacy_shadow = _safe_dict(ml.get("shadow"))
    if normalized_model_path(legacy_shadow.get("model_path")) == normalized_path:
        return legacy_shadow
    return None


def control_evidence_map(meta: dict[str, Any] | None) -> dict[str, Any]:
    ml = _safe_dict(_safe_dict(meta).get("ml"))
    evidence = ml.get(CONTROL_EVIDENCE_KEY)
    return dict(evidence) if isinstance(evidence, dict) else {}


def get_control_evidence_for_model_path(
    meta: dict[str, Any] | None,
    model_path: str | None,
) -> dict[str, Any] | None:
    normalized_path = normalized_model_path(model_path)
    if not normalized_path:
        return None

    evidence = control_evidence_map(meta)
    candidate = _safe_dict(evidence.get(normalized_path))
    if candidate:
        return candidate
    return None


def historical_shadow_evidence_map(meta: dict[str, Any] | None) -> dict[str, Any]:
    ml = _safe_dict(_safe_dict(meta).get("ml"))
    evidence = ml.get(HISTORICAL_SHADOW_EVIDENCE_KEY)
    return dict(evidence) if isinstance(evidence, dict) else {}


def get_historical_shadow_evidence_for_model_path(
    meta: dict[str, Any] | None,
    model_path: str | None,
) -> dict[str, Any]:
    normalized_path = normalized_model_path(model_path)
    if not normalized_path:
        return {}
    evidence = historical_shadow_evidence_map(meta)
    candidate = evidence.get(normalized_path)
    return dict(candidate) if isinstance(candidate, dict) else {}


def historical_control_evidence_map(meta: dict[str, Any] | None) -> dict[str, Any]:
    ml = _safe_dict(_safe_dict(meta).get("ml"))
    evidence = ml.get(HISTORICAL_CONTROL_EVIDENCE_KEY)
    return dict(evidence) if isinstance(evidence, dict) else {}


def get_historical_control_evidence_for_model_path(
    meta: dict[str, Any] | None,
    model_path: str | None,
) -> dict[str, Any]:
    normalized_path = normalized_model_path(model_path)
    if not normalized_path:
        return {}
    evidence = historical_control_evidence_map(meta)
    candidate = evidence.get(normalized_path)
    return dict(candidate) if isinstance(candidate, dict) else {}


def _evidence_without_volatile(payload: dict[str, Any]) -> dict[str, Any]:
    return {str(k): v for k, v in payload.items() if str(k) not in _VOLATILE_KEYS}


def _merge_evidence_map(
    updated_ml: dict[str, Any],
    *,
    key: str,
    model_path: str,
    evidence_payload: dict[str, Any],
) -> None:
    raw_map = updated_ml.get(key)
    evidence_map = dict(raw_map) if isinstance(raw_map, dict) else {}
    existing_payload = _safe_dict(evidence_map.get(model_path))
    final_payload = dict(evidence_payload)
    if existing_payload and _evidence_without_volatile(existing_payload) == _evidence_without_volatile(final_payload):
        final_payload = existing_payload
    evidence_map[model_path] = final_payload
    updated_ml[key] = evidence_map


def merge_shadow_evidence_into_meta(
    meta: dict[str, Any] | None,
    evidence_payload: dict[str, Any],
    *,
    set_legacy_shadow: bool = False,
) -> dict[str, Any]:
    original_meta = dict(meta) if isinstance(meta, dict) else {}
    updated_meta = dict(original_meta)
    updated_ml = dict(_safe_dict(updated_meta.get("ml")))
    model_path = normalized_model_path(evidence_payload.get("model_path"))
    if not model_path:
        updated_meta["ml"] = updated_ml
        return updated_meta

    _merge_evidence_map(
        updated_ml,
        key=SHADOW_EVIDENCE_KEY,
        model_path=model_path,
        evidence_payload=evidence_payload,
    )

    if set_legacy_shadow:
        existing_shadow = _safe_dict(updated_ml.get("shadow"))
        final_payload = _safe_dict(_safe_dict(updated_ml.get(SHADOW_EVIDENCE_KEY)).get(model_path))
        legacy_payload = dict(final_payload)
        if existing_shadow and _evidence_without_volatile(existing_shadow) == _evidence_without_volatile(legacy_payload):
            legacy_payload = existing_shadow
        updated_ml["shadow"] = legacy_payload

    updated_meta["ml"] = updated_ml
    return updated_meta


def merge_control_evidence_into_meta(
    meta: dict[str, Any] | None,
    evidence_payload: dict[str, Any],
) -> dict[str, Any]:
    original_meta = dict(meta) if isinstance(meta, dict) else {}
    updated_meta = dict(original_meta)
    updated_ml = dict(_safe_dict(updated_meta.get("ml")))
    model_path = normalized_model_path(evidence_payload.get("model_path"))
    if not model_path:
        updated_meta["ml"] = updated_ml
        return updated_meta

    _merge_evidence_map(
        updated_ml,
        key=CONTROL_EVIDENCE_KEY,
        model_path=model_path,
        evidence_payload=evidence_payload,
    )
    updated_meta["ml"] = updated_ml
    return updated_meta


def _merge_historical_evidence_map(
    updated_ml: dict[str, Any],
    *,
    key: str,
    model_path: str,
    anchor_key: str,
    evidence_payload: dict[str, Any],
) -> None:
    raw_map = updated_ml.get(key)
    evidence_map = dict(raw_map) if isinstance(raw_map, dict) else {}
    model_map = evidence_map.get(model_path)
    anchor_map = dict(model_map) if isinstance(model_map, dict) else {}
    existing_payload = _safe_dict(anchor_map.get(anchor_key))
    final_payload = dict(evidence_payload)
    if existing_payload and _evidence_without_volatile(existing_payload) == _evidence_without_volatile(final_payload):
        final_payload = existing_payload
    anchor_map[anchor_key] = final_payload
    evidence_map[model_path] = anchor_map
    updated_ml[key] = evidence_map


def merge_historical_shadow_evidence_into_meta(
    meta: dict[str, Any] | None,
    evidence_payload: dict[str, Any],
) -> dict[str, Any]:
    original_meta = dict(meta) if isinstance(meta, dict) else {}
    updated_meta = dict(original_meta)
    updated_ml = dict(_safe_dict(updated_meta.get("ml")))
    model_path = normalized_model_path(evidence_payload.get("model_path"))
    anchor_key = str(evidence_payload.get("anchor_key") or "").strip()
    if not model_path or not anchor_key:
        updated_meta["ml"] = updated_ml
        return updated_meta

    _merge_historical_evidence_map(
        updated_ml,
        key=HISTORICAL_SHADOW_EVIDENCE_KEY,
        model_path=model_path,
        anchor_key=anchor_key,
        evidence_payload=evidence_payload,
    )
    updated_meta["ml"] = updated_ml
    return updated_meta


def merge_historical_control_evidence_into_meta(
    meta: dict[str, Any] | None,
    evidence_payload: dict[str, Any],
) -> dict[str, Any]:
    original_meta = dict(meta) if isinstance(meta, dict) else {}
    updated_meta = dict(original_meta)
    updated_ml = dict(_safe_dict(updated_meta.get("ml")))
    model_path = normalized_model_path(evidence_payload.get("model_path"))
    anchor_key = str(evidence_payload.get("anchor_key") or "").strip()
    if not model_path or not anchor_key:
        updated_meta["ml"] = updated_ml
        return updated_meta

    _merge_historical_evidence_map(
        updated_ml,
        key=HISTORICAL_CONTROL_EVIDENCE_KEY,
        model_path=model_path,
        anchor_key=anchor_key,
        evidence_payload=evidence_payload,
    )
    updated_meta["ml"] = updated_ml
    return updated_meta
