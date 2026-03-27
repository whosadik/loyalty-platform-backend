from __future__ import annotations

from typing import Any

from django.conf import settings

ROADMAP_PICKED_VIA_RULES = "rules"
ROADMAP_PICKED_VIA_TEACHER = "teacher"
ROADMAP_PICKED_VIA_RUNTIME_CONTINUATION = "runtime_continuation_rules"

ROADMAP_SOURCE_MARKERS = {
    "picked via rules",
    "picked via teacher",
    "picked via runtime_continuation_rules",
    "picked via ml planner",
    "picked via ml next_step",
    "picked via planner fallback",
    "picked via user state",
}

ROADMAP_CONTINUATION_REASON_MARKERS = {
    "continued_due_to_core_gap",
    "continued_due_to_profile_need",
    "continued_due_to_owned_gap",
    "stopped_due_to_weak_tail_signal",
    "stopped_after_optional_tail",
}

ROADMAP_FROZEN_ARCHITECTURE: dict[str, Any] = {
    "initial_runtime_source": "teacher_rules",
    "continuation_runtime_source": "runtime_continuation_rules",
    "initial_live_ml_status": "experimental_off",
    "continuation_live_ml_status": "experimental_off",
    "initial_live_ml_runtime_candidate": False,
    "continuation_live_ml_runtime_candidate": False,
    "initial_runtime_categories": ["haircare", "skincare", "makeup", "fragrance"],
    "continuation_runtime_categories": ["haircare", "skincare", "makeup", "fragrance"],
    "continuation_ml_blocked_categories": ["fragrance", "makeup"],
}


def roadmap_runtime_ml_flags() -> dict[str, Any]:
    runtime_freeze_ml = bool(getattr(settings, "ROADMAP_RUNTIME_FREEZE_ML", True))
    planner_mode = str(getattr(settings, "ROADMAP_PLANNER_V1_MODE", "off") or "off").strip().lower()
    planner_categories = [
        str(item).strip().lower()
        for item in (getattr(settings, "ROADMAP_PLANNER_V1_ENABLED_CATEGORIES", []) or [])
        if str(item).strip()
    ]
    nextstep_v3_enabled = bool(getattr(settings, "ROADMAP_NEXTSTEP_V3_ENABLED", False))
    nextstep_v4_enabled = bool(getattr(settings, "ROADMAP_NEXTSTEP_V4_ENABLED", False))
    nextstep_v4_categories = [
        str(item).strip().lower()
        for item in (getattr(settings, "ROADMAP_NEXTSTEP_V4_ENABLED_CATEGORIES", []) or [])
        if str(item).strip()
    ]
    nextstep_v4_disabled_categories = [
        str(item).strip().lower()
        for item in (getattr(settings, "ROADMAP_NEXTSTEP_V4_DISABLED_CATEGORIES", []) or [])
        if str(item).strip()
    ]
    effective_planner_mode = "off" if runtime_freeze_ml else planner_mode
    effective_nextstep_v3_enabled = False if runtime_freeze_ml else nextstep_v3_enabled
    effective_nextstep_v4_enabled = False if runtime_freeze_ml else nextstep_v4_enabled
    return {
        "runtime_freeze_ml": runtime_freeze_ml,
        "planner_v1_mode": planner_mode,
        "effective_planner_v1_mode": effective_planner_mode,
        "planner_v1_enabled_categories": planner_categories,
        "nextstep_v3_enabled": nextstep_v3_enabled,
        "effective_nextstep_v3_enabled": effective_nextstep_v3_enabled,
        "nextstep_v4_enabled": nextstep_v4_enabled,
        "effective_nextstep_v4_enabled": effective_nextstep_v4_enabled,
        "nextstep_v4_enabled_categories": nextstep_v4_categories,
        "nextstep_v4_disabled_categories": nextstep_v4_disabled_categories,
        "rule_only_expected": (
            runtime_freeze_ml
            or (
                effective_planner_mode == "off"
                and not effective_nextstep_v3_enabled
                and not effective_nextstep_v4_enabled
            )
        ),
    }


def _normalized_why(why: list[Any] | None) -> list[str]:
    out: list[str] = []
    for item in (why or []):
        token = str(item or "").strip()
        if token:
            out.append(token)
    return out


def derive_roadmap_picked_via(*, why: list[Any] | None, plan_meta: dict[str, Any] | None = None) -> str:
    why_lower = {item.lower() for item in _normalized_why(why)}
    if why_lower & ROADMAP_CONTINUATION_REASON_MARKERS:
        return ROADMAP_PICKED_VIA_RUNTIME_CONTINUATION
    if "picked via user state" in why_lower or "picked via planner fallback" in why_lower:
        return ROADMAP_PICKED_VIA_TEACHER
    if "picked via ml planner" in why_lower or "picked via ml next_step" in why_lower:
        return ROADMAP_PICKED_VIA_TEACHER
    return ROADMAP_PICKED_VIA_RULES


def sanitize_roadmap_why(
    *,
    why: list[Any] | None,
    picked_via: str,
) -> list[str]:
    raw = _normalized_why(why)
    source_marker = {
        ROADMAP_PICKED_VIA_RULES: "picked via rules",
        ROADMAP_PICKED_VIA_TEACHER: "picked via teacher",
        ROADMAP_PICKED_VIA_RUNTIME_CONTINUATION: "picked via runtime_continuation_rules",
    }.get(str(picked_via), "picked via rules")
    out: list[str] = [source_marker]
    seen = {source_marker}
    for item in raw:
        normalized = item.lower()
        if normalized in ROADMAP_SOURCE_MARKERS:
            continue
        if item not in seen:
            out.append(item)
            seen.add(item)
    return out


def roadmap_step_explainability(
    *,
    why: list[Any] | None,
    plan_meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    picked_via = derive_roadmap_picked_via(why=why, plan_meta=plan_meta)
    continuation_markers = [
        item
        for item in _normalized_why(why)
        if str(item or "").strip().lower() in ROADMAP_CONTINUATION_REASON_MARKERS
    ]
    continuation_reason = None
    if continuation_markers:
        plan_continuation = (plan_meta or {}).get("continuation") if isinstance(plan_meta, dict) else {}
        if isinstance(plan_continuation, dict):
            continuation_reason = str(plan_continuation.get("reason") or "").strip() or None
        if not continuation_reason:
            continuation_reason = str(continuation_markers[-1] or "").strip() or None
    return {
        "picked_via": picked_via,
        "decision_source": picked_via,
        "why": sanitize_roadmap_why(why=why, picked_via=picked_via),
        "continuation_reason": continuation_reason,
        "continuation_markers": continuation_markers,
    }
