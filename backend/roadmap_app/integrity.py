from __future__ import annotations

from collections import Counter
from datetime import timedelta
from typing import Any

from django.db.models import Prefetch
from django.utils import timezone

from catalog.models import Product
from roadmap_app.fragrance_slots import SLOTS as FRAGRANCE_SLOTS, slot_of_fragrance
from roadmap_app.models import RoadmapEvent, RoadmapPlan, RoadmapStep


_ACTIONABLE_STATUSES = {
    RoadmapStep.Status.MISSING,
    RoadmapStep.Status.RECOMMENDED,
}


def _to_int(value: Any) -> int | None:
    try:
        if value is None:
            return None
        return int(value)
    except Exception:
        return None


def _slot_from_product_like(product_like: Any) -> str | None:
    if isinstance(product_like, Product):
        attrs = getattr(product_like, "attrs", None) or {}
        raw_meta = getattr(product_like, "raw_meta", None) or {}
    elif isinstance(product_like, dict):
        attrs = product_like.get("attrs") or {}
        raw_meta = product_like.get("raw_meta") or {}
    else:
        return None
    slot = slot_of_fragrance(attrs, raw_meta=raw_meta)
    return str(slot or "").strip().lower() or None


def _completion_product_id_from_context(context: dict[str, Any]) -> int | None:
    match_meta = context.get("match_meta") if isinstance(context.get("match_meta"), dict) else {}
    return (
        _to_int(match_meta.get("purchased_product_id"))
        or _to_int(context.get("purchased_product_id"))
        or _to_int(match_meta.get("recommended_product_id"))
        or _to_int(context.get("recommended_product_id"))
    )


def is_fragrance_slot_mismatch_step(step: RoadmapStep) -> bool:
    expected_slot = str(getattr(step, "product_type", "") or "").strip().lower()
    if expected_slot not in FRAGRANCE_SLOTS:
        return False
    if not getattr(step, "recommended_product_id", None):
        return False
    recommended_product = getattr(step, "recommended_product", None)
    if recommended_product is None:
        recommended_product = (
            Product.objects.filter(id=step.recommended_product_id)
            .values("attrs", "raw_meta")
            .first()
        )
    return _slot_from_product_like(recommended_product) != expected_slot


def collect_mismatched_fragrance_step_ids(
    *,
    plan_ids: list[int] | None = None,
    active_only: bool = True,
    chunk_size: int = 500,
) -> dict[int, list[int]]:
    qs = (
        RoadmapStep.objects.filter(
            plan__category=RoadmapPlan.Category.FRAGRANCE,
            recommended_product_id__isnull=False,
            product_type__in=FRAGRANCE_SLOTS,
        )
        .select_related("recommended_product")
        .order_by("plan_id", "step_index", "id")
    )
    if active_only:
        qs = qs.filter(plan__is_active=True)
    if plan_ids:
        qs = qs.filter(plan_id__in=list(plan_ids))

    mismatched: dict[int, list[int]] = {}
    for step in qs.iterator(chunk_size=max(1, int(chunk_size))):
        if not is_fragrance_slot_mismatch_step(step):
            continue
        mismatched.setdefault(int(step.plan_id), []).append(int(step.id))
    return mismatched


def active_fragrance_runtime_integrity_counts(
    *,
    plan_ids: list[int] | None = None,
) -> dict[str, int]:
    plan_qs = RoadmapPlan.objects.filter(
        category=RoadmapPlan.Category.FRAGRANCE,
        is_active=True,
    ).order_by("id")
    if plan_ids:
        plan_qs = plan_qs.filter(id__in=list(plan_ids))

    plan_qs = plan_qs.prefetch_related(
        Prefetch(
            "steps",
            queryset=RoadmapStep.objects.select_related("recommended_product").order_by("step_index", "id"),
            to_attr="prefetched_steps",
        )
    )

    active_total = 0
    active_with_recommended = 0
    active_slot_mismatch_count = 0
    for plan in plan_qs:
        steps = list(getattr(plan, "prefetched_steps", []) or [])
        next_step = next((step for step in steps if step.status in _ACTIONABLE_STATUSES), None)
        if next_step is None:
            continue
        active_total += 1
        if not next_step.recommended_product_id:
            continue
        active_with_recommended += 1
        if is_fragrance_slot_mismatch_step(next_step):
            active_slot_mismatch_count += 1

    return {
        "active_fragrance_next_steps_total": int(active_total),
        "active_fragrance_next_steps_with_recommended_product": int(active_with_recommended),
        "active_fragrance_slot_mismatch_count": int(active_slot_mismatch_count),
    }


def legacy_bad_fragrance_completion_details(*, recent_days: int = 30) -> dict[str, Any]:
    rows = list(
        RoadmapEvent.objects.filter(
            event_type=RoadmapEvent.Type.STEP_COMPLETED,
            step__plan__category=RoadmapPlan.Category.FRAGRANCE,
            context__matched_by="recommended_product_id",
        ).values(
            "id",
            "user_id",
            "created_at",
            "step__plan_id",
            "step__product_type",
            "context",
        )
    )

    product_ids: set[int] = set()
    for row in rows:
        context = row.get("context") or {}
        pid = _completion_product_id_from_context(context)
        if pid:
            product_ids.add(pid)

    product_slots_by_id = {
        int(row["id"]): _slot_from_product_like(row)
        for row in Product.objects.filter(id__in=list(product_ids)).values("id", "attrs", "raw_meta")
    }

    cutoff = timezone.now() - timedelta(days=max(0, int(recent_days)))
    bad_total = 0
    bad_recent = 0
    affected_users: set[int] = set()
    affected_plans: set[int] = set()
    step_state_drift_total = 0
    step_state_drift_recent = 0
    unresolved_missing_event_product_type = 0
    mismatch_pairs: Counter[str] = Counter()

    for row in rows:
        context = row.get("context") or {}
        expected_slot = str(context.get("product_type") or "").strip().lower()
        current_step_slot = str(row.get("step__product_type") or "").strip().lower()
        if expected_slot and current_step_slot and expected_slot != current_step_slot:
            step_state_drift_total += 1
            if row.get("created_at") and row["created_at"] >= cutoff:
                step_state_drift_recent += 1
        if expected_slot not in FRAGRANCE_SLOTS:
            unresolved_missing_event_product_type += 1
            continue
        pid = _completion_product_id_from_context(context)
        if pid is None:
            continue
        actual_slot = str(product_slots_by_id.get(pid) or "").strip().lower()
        if actual_slot == expected_slot:
            continue
        bad_total += 1
        mismatch_pairs[f"{expected_slot}->{actual_slot or '__none__'}"] += 1
        if row.get("created_at") and row["created_at"] >= cutoff:
            bad_recent += 1
        user_id = _to_int(row.get("user_id"))
        plan_id = _to_int(row.get("step__plan_id"))
        if user_id:
            affected_users.add(user_id)
        if plan_id:
            affected_plans.add(plan_id)

    if bad_total > 0:
        legacy_bucket = "true_bad_exact_match"
    elif step_state_drift_total > 0:
        legacy_bucket = "historical_step_state_drift"
    else:
        legacy_bucket = "clean"

    return {
        "legacy_bucket": legacy_bucket,
        "bad_fragrance_completed_exact_match_count": int(bad_total),
        f"bad_fragrance_completed_exact_match_recent_{int(recent_days)}d": int(bad_recent),
        "affected_users_count": int(len(affected_users)),
        "affected_plans_count": int(len(affected_plans)),
        "step_state_drift_count": int(step_state_drift_total),
        f"step_state_drift_recent_{int(recent_days)}d": int(step_state_drift_recent),
        "unresolved_missing_event_product_type_count": int(unresolved_missing_event_product_type),
        "mismatch_pairs": dict(sorted(mismatch_pairs.items(), key=lambda item: item[0])),
    }


def legacy_bad_fragrance_completion_counts(*, recent_days: int = 30) -> dict[str, int]:
    details = legacy_bad_fragrance_completion_details(recent_days=recent_days)
    return {
        "bad_fragrance_completed_exact_match_count": int(details["bad_fragrance_completed_exact_match_count"]),
        f"bad_fragrance_completed_exact_match_recent_{int(recent_days)}d": int(
            details[f"bad_fragrance_completed_exact_match_recent_{int(recent_days)}d"]
        ),
        "affected_users_count": int(details["affected_users_count"]),
        "affected_plans_count": int(details["affected_plans_count"]),
    }
