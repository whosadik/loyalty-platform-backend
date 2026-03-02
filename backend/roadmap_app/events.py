from __future__ import annotations

from typing import Any

from roadmap_app.models import RoadmapEvent, RoadmapPlan, RoadmapStep


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
