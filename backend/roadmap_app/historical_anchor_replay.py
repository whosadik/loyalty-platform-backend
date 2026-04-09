from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from typing import Any

from roadmap_app.models import RoadmapEvent


def _safe_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _safe_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _to_int(value: Any) -> int | None:
    try:
        if value is None:
            return None
        return int(value)
    except Exception:
        return None


def _event_key(created_at: Any, event_id: int | None) -> tuple[Any, int]:
    return created_at, int(event_id or 0)


def _first_event_in_range(
    events: list[dict[str, Any]],
    *,
    start_key: tuple[Any, int],
    end_key: tuple[Any, int] | None,
) -> dict[str, Any] | None:
    for row in events:
        key = _event_key(row.get("created_at"), _to_int(row.get("event_id")))
        if key < start_key:
            continue
        if end_key is not None and key >= end_key:
            break
        return row
    return None


def _any_event_in_range(
    events: list[dict[str, Any]],
    *,
    start_key: tuple[Any, int],
    end_key: tuple[Any, int] | None,
) -> bool:
    return _first_event_in_range(events, start_key=start_key, end_key=end_key) is not None


def _source_from_expose_context(ctx: dict[str, Any]) -> str:
    sources = _safe_list(ctx.get("sources"))
    normalized = {str(x).strip().lower() for x in sources if str(x).strip()}
    if "offers" in normalized:
        return "offers"
    if "roadmap_api" in normalized:
        return "roadmap_api"
    if ctx.get("offer_assignment_id") not in (None, ""):
        return "offers"
    return "roadmap_api"


def _candidate_types_for_rows(rows: list[dict[str, Any]]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for row in sorted(
        rows,
        key=lambda item: (
            int(item.get("step_index") or 0),
            item.get("created_at"),
            int(item.get("event_id") or 0),
        ),
    ):
        product_type = str(row.get("product_type") or "").strip().lower()
        if not product_type or product_type in seen:
            continue
        seen.add(product_type)
        out.append(product_type)
    return out


def build_historical_continuation_anchor_records(
    *,
    since,
    until,
    category: str = "all",
    include_ga: bool = False,
) -> list[dict[str, Any]]:
    event_types = [
        RoadmapEvent.Type.PLAN_REFRESHED,
        RoadmapEvent.Type.STEP_GENERATED,
        RoadmapEvent.Type.STEP_EXPOSED,
        RoadmapEvent.Type.STEP_CLICKED,
        RoadmapEvent.Type.STEP_COMPLETED,
        RoadmapEvent.Type.STEP_SKIPPED,
    ]
    qs = RoadmapEvent.objects.filter(
        created_at__gte=since,
        created_at__lte=until,
        event_type__in=event_types,
        plan_id__isnull=False,
    ).order_by("plan_id", "created_at", "id")
    if not include_ga:
        qs = qs.exclude(user__username__startswith="ga_")

    rows = list(
        qs.values(
            "id",
            "user_id",
            "plan_id",
            "step_id",
            "event_type",
            "created_at",
            "context",
        )
    )

    refreshes_by_plan: dict[int, list[dict[str, Any]]] = defaultdict(list)
    generated_by_plan: dict[int, list[dict[str, Any]]] = defaultdict(list)
    generated_by_plan_step: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    exposes_by_step: dict[int, list[dict[str, Any]]] = defaultdict(list)
    clicks_by_step: dict[int, list[dict[str, Any]]] = defaultdict(list)
    completions_by_step: dict[int, list[dict[str, Any]]] = defaultdict(list)
    skips_by_step: dict[int, list[dict[str, Any]]] = defaultdict(list)

    category_filter = str(category or "all").strip().lower()

    for row in rows:
        ctx = _safe_dict(row.get("context"))
        event_type = str(row.get("event_type") or "")
        event_id = int(row.get("id") or 0)
        plan_id = _to_int(row.get("plan_id"))
        step_id = _to_int(row.get("step_id")) or _to_int(ctx.get("step_id"))
        user_id = _to_int(row.get("user_id"))
        created_at = row.get("created_at")
        if plan_id is None or user_id is None or created_at is None:
            continue

        if event_type == RoadmapEvent.Type.PLAN_REFRESHED:
            event_category = str(ctx.get("category") or "").strip().lower() or "__unknown__"
            if category_filter != "all" and event_category != category_filter:
                continue
            raw_next_step_id = _to_int(ctx.get("next_step_id"))
            raw_next_step_index = _to_int(ctx.get("next_step_index"))
            raw_next_product_type = str(ctx.get("next_product_type") or "").strip().lower()
            refreshes_by_plan[int(plan_id)].append(
                {
                    "anchor_key": f"plan_refresh:{event_id}",
                    "event_id": event_id,
                    "plan_id": int(plan_id),
                    "user_id": int(user_id),
                    "created_at": created_at,
                    "category": event_category,
                    "plan_source": str(ctx.get("source") or "roadmap_v1"),
                    "refresh_caller": str(ctx.get("refresh_caller") or "").strip().lower(),
                    "next_step_id": raw_next_step_id,
                    "next_step_index": raw_next_step_index,
                    "next_product_type": raw_next_product_type,
                    "anchor_next_step_id": raw_next_step_id,
                    "anchor_next_step_index": raw_next_step_index,
                    "anchor_next_product_type": raw_next_product_type,
                    "ml_decision": str(_safe_dict(ctx.get("ml")).get("decision") or "").strip().lower(),
                }
            )
            continue

        if step_id is None:
            continue

        if event_type == RoadmapEvent.Type.STEP_GENERATED:
            generated = {
                "event_id": event_id,
                "plan_id": int(plan_id),
                "user_id": int(user_id),
                "step_id": int(step_id),
                "created_at": created_at,
                "category": str(ctx.get("category") or "").strip().lower() or "__unknown__",
                "step_index": _to_int(ctx.get("step_index")),
                "product_type": str(ctx.get("product_type") or "").strip().lower(),
                "status": str(ctx.get("status") or "").strip().lower(),
                "recommended_product_id": _to_int(ctx.get("recommended_product_id")),
                "has_recommendation": bool(ctx.get("has_recommendation")),
            }
            generated_by_plan[int(plan_id)].append(generated)
            generated_by_plan_step[(int(plan_id), int(step_id))].append(generated)
            continue

        row_payload = {
            "event_id": event_id,
            "created_at": created_at,
            "context": ctx,
            "product_type": str(ctx.get("product_type") or "").strip().lower(),
            "matched_by": str(ctx.get("matched_by") or "").strip().lower(),
        }
        if event_type == RoadmapEvent.Type.STEP_EXPOSED:
            row_payload["source"] = _source_from_expose_context(ctx)
            exposes_by_step[int(step_id)].append(row_payload)
        elif event_type == RoadmapEvent.Type.STEP_CLICKED:
            clicks_by_step[int(step_id)].append(row_payload)
        elif event_type == RoadmapEvent.Type.STEP_COMPLETED:
            completions_by_step[int(step_id)].append(row_payload)
        elif event_type == RoadmapEvent.Type.STEP_SKIPPED:
            skips_by_step[int(step_id)].append(row_payload)

    for items in generated_by_plan.values():
        items.sort(key=lambda item: _event_key(item.get("created_at"), int(item.get("event_id") or 0)))
    for items in generated_by_plan_step.values():
        items.sort(key=lambda item: _event_key(item.get("created_at"), int(item.get("event_id") or 0)))
    for items in refreshes_by_plan.values():
        items.sort(key=lambda item: _event_key(item.get("created_at"), int(item.get("event_id") or 0)))
    for mapping in (exposes_by_step, clicks_by_step, completions_by_step, skips_by_step):
        for items in mapping.values():
            items.sort(key=lambda item: _event_key(item.get("created_at"), int(item.get("event_id") or 0)))

    anchors: list[dict[str, Any]] = []
    for plan_id, refreshes in refreshes_by_plan.items():
        generated_rows = generated_by_plan.get(int(plan_id), [])
        for idx, refresh in enumerate(refreshes):
            refresh_key = _event_key(refresh.get("created_at"), int(refresh.get("event_id") or 0))
            next_refresh = refreshes[idx + 1] if idx + 1 < len(refreshes) else None
            next_refresh_key = (
                _event_key(next_refresh.get("created_at"), int(next_refresh.get("event_id") or 0))
                if next_refresh is not None
                else None
            )
            generated_in_window = [
                row
                for row in generated_rows
                if _event_key(row.get("created_at"), int(row.get("event_id") or 0)) >= refresh_key
                and (next_refresh_key is None or _event_key(row.get("created_at"), int(row.get("event_id") or 0)) < next_refresh_key)
            ]
            candidate_types = _candidate_types_for_rows(generated_in_window)
            next_step_id = _to_int(refresh.get("next_step_id"))
            next_step_index = _to_int(refresh.get("next_step_index"))
            next_product_type = str(refresh.get("next_product_type") or "").strip().lower()
            anchor_next_step_id = _to_int(refresh.get("anchor_next_step_id"))
            anchor_next_step_index = _to_int(refresh.get("anchor_next_step_index"))
            anchor_next_product_type = str(refresh.get("anchor_next_product_type") or "").strip().lower()
            generated_candidates = [
                {
                    "event_id": int(row.get("event_id") or 0),
                    "step_id": int(row.get("step_id") or 0),
                    "step_index": _to_int(row.get("step_index")),
                    "product_type": str(row.get("product_type") or "").strip().lower(),
                    "created_at": row.get("created_at"),
                    "recommended_product_id": _to_int(row.get("recommended_product_id")),
                    "has_recommendation": bool(row.get("has_recommendation")),
                    "is_generated": True,
                }
                for row in generated_in_window
                if _to_int(row.get("step_id")) is not None
            ]
            anchor_has_actionable_step = bool(
                anchor_next_step_id is not None
                or anchor_next_step_index is not None
                or anchor_next_product_type
            )

            next_generated: dict[str, Any] | None = None
            if next_step_id is not None:
                next_generated = _first_event_in_range(
                    generated_by_plan_step.get((int(plan_id), int(next_step_id)), []),
                    start_key=refresh_key,
                    end_key=next_refresh_key,
                )
            if next_generated is None and next_step_index is not None:
                next_generated = next(
                    (
                        row
                        for row in generated_in_window
                        if _to_int(row.get("step_index")) == int(next_step_index)
                    ),
                    None,
                )
            if next_generated is None and next_product_type:
                next_generated = next(
                    (
                        row
                        for row in generated_in_window
                        if str(row.get("product_type") or "").strip().lower() == next_product_type
                    ),
                    None,
                )

            resolved_step_id = _to_int((next_generated or {}).get("step_id")) or next_step_id
            resolved_step_index = _to_int((next_generated or {}).get("step_index")) or next_step_index
            resolved_product_type = (
                str((next_generated or {}).get("product_type") or "").strip().lower()
                or next_product_type
            )

            reconstruction_reason = ""
            if not candidate_types:
                reconstruction_reason = "missing_generated_steps_in_refresh_window"
            elif resolved_step_id is None:
                reconstruction_reason = "missing_next_step_id_for_outcome_window"
            elif not resolved_product_type:
                reconstruction_reason = "missing_next_step_product_type"

            outcome_start_key = (
                _event_key((next_generated or {}).get("created_at"), _to_int((next_generated or {}).get("event_id")))
                if next_generated is not None
                else refresh_key
            )
            first_exposed = (
                _first_event_in_range(
                    exposes_by_step.get(int(resolved_step_id or 0), []),
                    start_key=outcome_start_key,
                    end_key=next_refresh_key,
                )
                if resolved_step_id is not None
                else None
            )
            first_exposed_key = (
                _event_key(first_exposed.get("created_at"), _to_int(first_exposed.get("event_id")))
                if first_exposed is not None
                else None
            )
            first_completed = (
                _first_event_in_range(
                    completions_by_step.get(int(resolved_step_id or 0), []),
                    start_key=outcome_start_key,
                    end_key=next_refresh_key,
                )
                if resolved_step_id is not None
                else None
            )
            has_clicked_after_exposure = bool(
                resolved_step_id is not None
                and first_exposed_key is not None
                and _any_event_in_range(
                    clicks_by_step.get(int(resolved_step_id), []),
                    start_key=first_exposed_key,
                    end_key=next_refresh_key,
                )
            )
            has_completed_after_generated = bool(
                resolved_step_id is not None
                and _any_event_in_range(
                    completions_by_step.get(int(resolved_step_id), []),
                    start_key=outcome_start_key,
                    end_key=next_refresh_key,
                )
            )
            has_completed_after_exposure = bool(
                resolved_step_id is not None
                and first_exposed_key is not None
                and _any_event_in_range(
                    completions_by_step.get(int(resolved_step_id), []),
                    start_key=first_exposed_key,
                    end_key=next_refresh_key,
                )
            )
            has_skipped_after_generated = bool(
                resolved_step_id is not None
                and _any_event_in_range(
                    skips_by_step.get(int(resolved_step_id), []),
                    start_key=outcome_start_key,
                    end_key=next_refresh_key,
                )
            )

            anchors.append(
                {
                    "anchor_key": str(refresh.get("anchor_key") or ""),
                    "anchor_event_id": int(refresh.get("event_id") or 0),
                    "anchor_created_at": refresh.get("created_at"),
                    "next_refresh_at": next_refresh.get("created_at") if next_refresh is not None else None,
                    "plan_id": int(plan_id),
                    "user_id": int(refresh.get("user_id") or 0),
                    "category": str(refresh.get("category") or "__unknown__"),
                    "plan_source": str(refresh.get("plan_source") or "roadmap_v1"),
                    "refresh_caller": str(refresh.get("refresh_caller") or "").strip().lower(),
                    "ml_decision": str(refresh.get("ml_decision") or "").strip().lower(),
                    "anchor_next_step_id": anchor_next_step_id,
                    "anchor_next_step_index": anchor_next_step_index,
                    "anchor_next_product_type": anchor_next_product_type,
                    "anchor_has_actionable_step": bool(anchor_has_actionable_step),
                    "next_step_id": resolved_step_id,
                    "next_step_index": resolved_step_index,
                    "planned_target_product_type": resolved_product_type,
                    "planned_target_step_index": int(resolved_step_index or 0),
                    "candidate_types": candidate_types,
                    "generated_candidates": generated_candidates,
                    "generated_step_ids": [
                        int(row.get("step_id") or 0)
                        for row in generated_in_window
                        if _to_int(row.get("step_id")) is not None
                    ],
                    "generated_rows_total": int(len(generated_in_window)),
                    "generated_next_step_event_id": _to_int((next_generated or {}).get("event_id")),
                    "generated_next_step_at": (next_generated or {}).get("created_at"),
                    "generated_next_step_has_recommendation": bool((next_generated or {}).get("has_recommendation")),
                    "reconstruction_reason": reconstruction_reason,
                    "has_exposed": bool(first_exposed is not None),
                    "first_expose_source": str((first_exposed or {}).get("source") or "__not_exposed__"),
                    "has_clicked_after_exposure": bool(has_clicked_after_exposure),
                    "has_completed_after_generated": bool(has_completed_after_generated),
                    "has_completed_after_exposure": bool(has_completed_after_exposure),
                    "has_skipped_after_generated": bool(has_skipped_after_generated),
                    "completed_product_type": str((first_completed or {}).get("product_type") or "").strip().lower(),
                    "completed_matched_by": str((first_completed or {}).get("matched_by") or "").strip().lower(),
                }
            )

    anchors.sort(
        key=lambda row: (
            row.get("anchor_created_at"),
            int(row.get("anchor_event_id") or 0),
        )
    )
    return anchors
