from __future__ import annotations

from datetime import datetime, time, timedelta, timezone as dt_timezone
from typing import Any

from roadmap_app.models import RoadmapEvent, RoadmapPlan, RoadmapStep
from django.utils import timezone


def build_step_event_context(
    *,
    category: str | None,
    step: RoadmapStep | None,
    offer_assignment_id: int | None = None,
    transaction_id: int | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    ctx: dict[str, Any] = {
        "category": category or (step.plan.category if step and step.plan_id else None),
        "step_index": int(step.step_index) if step else None,
        "product_type": step.product_type if step else None,
        "recommended_product_id": int(step.recommended_product_id) if step and step.recommended_product_id else None,
        "offer_assignment_id": int(offer_assignment_id) if offer_assignment_id else None,
        "transaction_id": int(transaction_id) if transaction_id else None,
    }
    if extra:
        ctx.update(extra)
    return ctx


def record_roadmap_event(
    *,
    user,
    event_type: str,
    plan: RoadmapPlan | None = None,
    step: RoadmapStep | None = None,
    request_id: str | None = None,
    context: dict[str, Any] | None = None,
) -> RoadmapEvent:
    return RoadmapEvent.objects.create(
        user=user,
        plan=plan,
        step=step,
        event_type=event_type,
        request_id=request_id,
        context=context or {},
    )


def _utc_day_bounds(now: datetime | None = None) -> tuple[datetime, datetime]:
    now = now or timezone.now()
    day = now.astimezone(dt_timezone.utc).date()
    start = datetime.combine(day, time.min, tzinfo=dt_timezone.utc)
    end = start + timedelta(days=1)
    return start, end


def has_step_exposed_today(*, user, step: RoadmapStep) -> bool:
    return get_step_exposed_today_event(user=user, step=step) is not None


def get_step_exposed_today_event(*, user, step: RoadmapStep) -> RoadmapEvent | None:
    start, end = _utc_day_bounds()
    return (
        RoadmapEvent.objects.filter(
            user=user,
            step=step,
            event_type=RoadmapEvent.Type.STEP_EXPOSED,
            created_at__gte=start,
            created_at__lt=end,
        )
        .order_by("created_at", "id")
        .first()
    )


def record_step_exposed_dedup(
    *,
    user,
    plan: RoadmapPlan,
    step: RoadmapStep,
    request_id: str | None = None,
    offer_assignment_id: int | None = None,
) -> tuple[RoadmapEvent | None, bool]:
    source = "offers" if offer_assignment_id else "roadmap_api"
    existing = get_step_exposed_today_event(user=user, step=step)
    if existing:
        existing_ctx = existing.context if isinstance(existing.context, dict) else {}
        new_ctx = dict(existing_ctx)

        sources_raw = new_ctx.get("sources")
        if isinstance(sources_raw, list):
            sources = [str(x) for x in sources_raw if str(x).strip()]
        else:
            # Backfill source for old events that were created before sources support.
            inferred = "offers" if new_ctx.get("offer_assignment_id") not in (None, "") else "roadmap_api"
            sources = [inferred]
        if source not in sources:
            sources.append(source)
        new_ctx["sources"] = sources

        if offer_assignment_id and new_ctx.get("offer_assignment_id") in (None, ""):
            new_ctx["offer_assignment_id"] = int(offer_assignment_id)

        if new_ctx != existing_ctx:
            RoadmapEvent.objects.filter(id=existing.id).update(context=new_ctx)
            existing.context = new_ctx
        return existing, False

    base_ctx = build_step_event_context(
        category=plan.category,
        step=step,
        offer_assignment_id=offer_assignment_id,
    )
    base_ctx["sources"] = [source]
    ev = record_roadmap_event(
        user=user,
        event_type=RoadmapEvent.Type.STEP_EXPOSED,
        plan=plan,
        step=step,
        request_id=request_id,
        context=base_ctx,
    )
    return ev, True


def record_exposed_from_offer_assignment(*, assignment, request_id: str | None = None) -> tuple[RoadmapEvent | None, bool]:
    if not assignment:
        return None, False

    target = assignment.target if isinstance(assignment.target, dict) else {}
    picked_via = str(target.get("picked_via") or "").strip()
    if not picked_via.startswith("roadmap_shortcut"):
        return None, False

    reason = assignment.reason if isinstance(assignment.reason, dict) else {}
    roadmap_reason = reason.get("roadmap") if isinstance(reason.get("roadmap"), dict) else {}

    plan = None
    plan_id_raw = roadmap_reason.get("plan_id")
    try:
        plan_id = int(plan_id_raw) if plan_id_raw is not None else None
    except Exception:
        plan_id = None

    if plan_id:
        plan = (
            RoadmapPlan.objects.filter(id=plan_id, user=assignment.user, is_active=True)
            .prefetch_related("steps")
            .first()
        )

    if not plan:
        category = str(target.get("category") or roadmap_reason.get("category") or "").strip()
        if not category:
            return None, False
        from roadmap_app.services import get_active_plan

        plan = get_active_plan(assignment.user, category=category)
        if not plan:
            return None, False

    next_step = (
        plan.steps.filter(status__in=[RoadmapStep.Status.MISSING, RoadmapStep.Status.RECOMMENDED])
        .order_by("step_index")
        .first()
    )
    if not next_step:
        return None, False

    return record_step_exposed_dedup(
        user=assignment.user,
        plan=plan,
        step=next_step,
        request_id=request_id,
        offer_assignment_id=assignment.id,
    )
