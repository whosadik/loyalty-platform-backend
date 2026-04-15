from __future__ import annotations

from typing import Any

from roadmap_app.historical_anchor_replay import build_historical_continuation_anchor_records
from roadmap_app.models import RoadmapPlan
from roadmap_app.nextstep_historical_anchor_dataset import completion_events_by_step


def _safe_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _safe_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _int_keyed_dict(value: Any) -> dict[int, Any]:
    out: dict[int, Any] = {}
    for raw_key, raw_value in _safe_dict(value).items():
        try:
            key = int(raw_key)
        except Exception:
            continue
        out[key] = raw_value
    return out


def build_historical_anchor_read_context(
    *,
    since,
    until,
    category: str = "all",
    include_ga: bool = False,
) -> dict[str, Any]:
    category_norm = str(category or "all").strip().lower() or "all"
    anchors = build_historical_continuation_anchor_records(
        since=since,
        until=until,
        category=category_norm,
        include_ga=include_ga,
    )
    plan_ids = {
        int(anchor.get("plan_id") or 0)
        for anchor in anchors
        if int(anchor.get("plan_id") or 0) > 0
    }
    meta_by_plan = {
        int(row["id"]): _safe_dict(row.get("meta"))
        for row in RoadmapPlan.objects.filter(id__in=plan_ids).values("id", "meta")
    }
    all_generated_step_ids = {
        int(step_id)
        for anchor in anchors
        for step_id in _safe_list(anchor.get("generated_step_ids"))
        if str(step_id or "").strip()
    }
    return {
        "since": since,
        "until": until,
        "category": category_norm,
        "include_ga": bool(include_ga),
        "anchors": anchors,
        "meta_by_plan": meta_by_plan,
        "completions_by_step": completion_events_by_step(
            since=since,
            until=until,
            step_ids=all_generated_step_ids,
        ),
        "read_only": True,
    }


def resolve_historical_anchor_read_context(
    *,
    since,
    until,
    category: str = "all",
    include_ga: bool = False,
    historical_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if historical_context is None:
        return build_historical_anchor_read_context(
            since=since,
            until=until,
            category=category,
            include_ga=include_ga,
        )

    category_norm = str(category or "all").strip().lower() or "all"
    provided_category = str(historical_context.get("category") or category_norm).strip().lower() or "all"
    if provided_category != category_norm:
        raise ValueError(
            f"historical_context category mismatch: expected {category_norm}, got {provided_category}"
        )

    provided_include_ga = bool(historical_context.get("include_ga"))
    if provided_include_ga != bool(include_ga):
        raise ValueError(
            f"historical_context include_ga mismatch: expected {bool(include_ga)}, got {provided_include_ga}"
        )

    meta_by_plan = {
        plan_id: _safe_dict(meta)
        for plan_id, meta in _int_keyed_dict(historical_context.get("meta_by_plan")).items()
    }
    completions_by_step = {
        step_id: _safe_list(rows)
        for step_id, rows in _int_keyed_dict(historical_context.get("completions_by_step")).items()
    }
    return {
        "since": historical_context.get("since", since),
        "until": historical_context.get("until", until),
        "category": provided_category,
        "include_ga": provided_include_ga,
        "anchors": _safe_list(historical_context.get("anchors")),
        "meta_by_plan": meta_by_plan,
        "completions_by_step": completions_by_step,
        "read_only": bool(historical_context.get("read_only", True)),
    }
