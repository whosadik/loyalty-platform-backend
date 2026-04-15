from __future__ import annotations

from typing import Any

from roadmap_app.models import RoadmapEvent


TRAIN_EXCLUSION_REASONS = {
    "no_actionable_step",
    "missing_next_step_id",
    "no_completed_generated_candidate",
    "ambiguous_outcome_window",
    "incomplete_refresh_window",
}


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


def _event_key(created_at: Any, event_id: Any) -> tuple[Any, int]:
    return created_at, int(_to_int(event_id) or 0)


def completion_events_by_step(*, since, until, step_ids: set[int]) -> dict[int, list[dict[str, Any]]]:
    if not step_ids:
        return {}
    out: dict[int, list[dict[str, Any]]] = {}
    for row in (
        RoadmapEvent.objects.filter(
            event_type=RoadmapEvent.Type.STEP_COMPLETED,
            step_id__in=step_ids,
            created_at__gte=since,
            created_at__lte=until,
        )
        .order_by("step_id", "created_at", "id")
        .values("id", "step_id", "created_at", "context")
        .iterator(chunk_size=2000)
    ):
        step_id = _to_int(row.get("step_id"))
        if step_id is None:
            continue
        out.setdefault(int(step_id), []).append(row)
    return out


def generated_candidates_by_product_type(anchor: dict[str, Any]) -> list[dict[str, Any]]:
    candidates = []
    seen: set[str] = set()
    for row in sorted(
        _safe_list(anchor.get("generated_candidates")),
        key=lambda item: (
            item.get("created_at"),
            int(_to_int(item.get("event_id")) or 0),
            int(_to_int(item.get("step_index")) or 0),
            int(_to_int(item.get("step_id")) or 0),
        ),
    ):
        product_type = str(row.get("product_type") or "").strip().lower()
        if not product_type or product_type in seen:
            continue
        seen.add(product_type)
        candidates.append(
            {
                "event_id": int(_to_int(row.get("event_id")) or 0),
                "step_id": int(_to_int(row.get("step_id")) or 0),
                "step_index": _to_int(row.get("step_index")),
                "product_type": product_type,
                "created_at": row.get("created_at"),
                "recommended_product_id": _to_int(row.get("recommended_product_id")),
                "has_recommendation": bool(row.get("has_recommendation")),
                "is_generated": bool(row.get("is_generated", True)),
            }
        )
    return candidates


def resolve_first_completed_generated_candidate(
    anchor: dict[str, Any],
    *,
    completions_by_step: dict[int, list[dict[str, Any]]],
) -> dict[str, Any]:
    start_key = _event_key(anchor.get("anchor_created_at"), anchor.get("anchor_event_id"))
    end_key = (
        _event_key(anchor.get("next_refresh_at"), 10**18)
        if anchor.get("next_refresh_at") is not None
        else None
    )
    hits: list[dict[str, Any]] = []
    truth_types: list[str] = []
    seen_types: set[str] = set()

    for candidate in _safe_list(anchor.get("generated_candidates")):
        step_id = _to_int(candidate.get("step_id"))
        if step_id is None:
            continue
        first_completion_for_step: dict[str, Any] | None = None
        for row in completions_by_step.get(int(step_id), []):
            row_key = _event_key(row.get("created_at"), row.get("id"))
            if row_key < start_key:
                continue
            if end_key is not None and row_key >= end_key:
                break
            first_completion_for_step = row
            break
        if first_completion_for_step is None:
            continue
        ctx = _safe_dict(first_completion_for_step.get("context"))
        truth_product_type = (
            str(ctx.get("product_type") or candidate.get("product_type") or "").strip().lower()
        )
        if not truth_product_type:
            continue
        if truth_product_type not in seen_types:
            seen_types.add(truth_product_type)
            truth_types.append(truth_product_type)
        hits.append(
            {
                "step_id": int(step_id),
                "step_index": _to_int(candidate.get("step_index")),
                "product_type": truth_product_type,
                "matched_by": str(ctx.get("matched_by") or "").strip().lower(),
                "created_at": first_completion_for_step.get("created_at"),
                "event_id": int(_to_int(first_completion_for_step.get("id")) or 0),
            }
        )

    if not hits:
        return {
            "resolved": False,
            "reason": "no_completed_generated_candidate",
            "truth_selected_candidate_step_id": None,
            "truth_selected_product_type": "",
            "truth_matched_by": "",
            "truth_product_types_in_window": [],
            "truth_is_resolved": False,
            "ambiguous_multiple_completed_types": False,
        }

    first_key = min(_event_key(hit.get("created_at"), hit.get("event_id")) for hit in hits)
    earliest_hits = [
        hit for hit in hits if _event_key(hit.get("created_at"), hit.get("event_id")) == first_key
    ]
    ambiguous = len({int(hit.get("step_id") or 0) for hit in earliest_hits}) > 1
    if ambiguous:
        return {
            "resolved": False,
            "reason": "ambiguous_outcome_window",
            "truth_selected_candidate_step_id": None,
            "truth_selected_product_type": "",
            "truth_matched_by": "",
            "truth_product_types_in_window": truth_types,
            "truth_is_resolved": False,
            "ambiguous_multiple_completed_types": True,
        }

    first_hit = sorted(
        earliest_hits,
        key=lambda item: (
            item.get("created_at"),
            int(item.get("event_id") or 0),
            int(item.get("step_id") or 0),
        ),
    )[0]
    return {
        "resolved": True,
        "reason": "ok",
        "truth_selected_candidate_step_id": int(first_hit.get("step_id") or 0),
        "truth_selected_product_type": str(first_hit.get("product_type") or ""),
        "truth_matched_by": str(first_hit.get("matched_by") or ""),
        "truth_product_types_in_window": truth_types,
        "truth_is_resolved": True,
        "ambiguous_multiple_completed_types": len(truth_types) > 1,
    }


def classify_train_exclusion_reason(anchor: dict[str, Any], truth: dict[str, Any]) -> str:
    if not bool(anchor.get("anchor_has_actionable_step")):
        return "no_actionable_step"
    anchor_next_step_id = _to_int(anchor.get("anchor_next_step_id"))
    if anchor_next_step_id is None or int(anchor_next_step_id) <= 0:
        return "missing_next_step_id"
    if anchor.get("next_refresh_at") is None:
        return "incomplete_refresh_window"

    reconstruction_reason = str(anchor.get("reconstruction_reason") or "").strip().lower()
    if reconstruction_reason == "missing_generated_steps_in_refresh_window":
        return "incomplete_refresh_window"
    if reconstruction_reason and reconstruction_reason != "ok":
        return f"other:{reconstruction_reason}"

    truth_reason = str(truth.get("reason") or "").strip().lower()
    if bool(truth.get("resolved")):
        return ""
    if truth_reason in TRAIN_EXCLUSION_REASONS:
        return truth_reason
    if truth_reason:
        return f"other:{truth_reason}"
    return "other:unknown_unresolved_anchor"


def bucket_flags_for_row(
    *,
    category: str,
    truth_product_type: str,
    candidate_product_type: str,
) -> dict[str, int]:
    category_norm = str(category or "").strip().lower()
    truth_norm = str(truth_product_type or "").strip().lower()
    candidate_norm = str(candidate_product_type or "").strip().lower()
    return {
        "bucket_skincare_mask": int(
            category_norm == "skincare" and truth_norm == "mask" and candidate_norm == "mask"
        ),
        "bucket_skincare_toner": int(
            category_norm == "skincare" and truth_norm == "toner" and candidate_norm == "toner"
        ),
        "bucket_skincare_eye_cream": int(
            category_norm == "skincare" and truth_norm == "eye_cream" and candidate_norm == "eye_cream"
        ),
        "bucket_haircare_shampoo": int(
            category_norm == "haircare" and truth_norm == "shampoo" and candidate_norm == "shampoo"
        ),
        "bucket_haircare_shampoo_to_conditioner": int(
            category_norm == "haircare" and truth_norm == "shampoo" and candidate_norm == "conditioner"
        ),
        "protected_haircare_hair_mask": int(
            category_norm == "haircare" and truth_norm == "hair_mask" and candidate_norm == "hair_mask"
        ),
        "protected_haircare_hair_oil": int(
            category_norm == "haircare" and truth_norm == "hair_oil" and candidate_norm == "hair_oil"
        ),
        "protected_skincare_essence": int(
            category_norm == "skincare" and truth_norm == "essence" and candidate_norm == "essence"
        ),
        "analysis_fragrance_cold_evening": int(
            category_norm == "fragrance" and truth_norm == "cold_evening" and candidate_norm == "cold_evening"
        ),
    }
