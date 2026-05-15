"""Step-completion loyalty bonuses for roadmap steps.

Each roadmap step has a per-type ``points`` value defined in
:mod:`roadmap_app.step_presentation`. When a step transitions to a
"done" state (COMPLETED via purchase / patch, or OWNED via the user
already owning the product type), we award those points to the user's
loyalty account.

Idempotency is enforced via a ``LoyaltyLedgerEntry.reference`` of the
form ``roadmap_step:{step_id}`` — the same step never credits twice
across the lifetime of a plan.
"""
from __future__ import annotations

from typing import Iterable

from django.db import transaction
from django.utils import timezone

from loyalty.models import LoyaltyAccount, LoyaltyLedgerEntry, Tier
from loyalty.points import DEFAULT_POINTS_RATE
from roadmap_app.models import RoadmapPlan, RoadmapStep
from roadmap_app.step_presentation import get_roadmap_step_meta


DONE_STATUSES = {RoadmapStep.Status.COMPLETED, RoadmapStep.Status.OWNED}


def step_bonus_points(product_type: str | None) -> int:
    meta = get_roadmap_step_meta(product_type, language="ru")
    try:
        return max(0, int(meta.get("points") or 0))
    except (TypeError, ValueError):
        return 0


def _reference_for_step(step_id: int) -> str:
    return f"roadmap_step:{int(step_id)}"


def _ensure_account(user) -> LoyaltyAccount:
    acc, _ = LoyaltyAccount.objects.get_or_create(user=user)
    if acc.tier_id is None:
        bronze, _ = Tier.objects.get_or_create(
            name="Bronze",
            defaults={"threshold_spend_90d": 0, "points_rate": DEFAULT_POINTS_RATE},
        )
        acc.tier = bronze
        acc.save(update_fields=["tier"])
    return acc


def maybe_award_step_completion_bonus(user, step: RoadmapStep) -> dict:
    """Credit the per-step bonus to *user* for *step* if not yet credited.

    Returns dict with keys: ok, awarded, points_added.
    Safe to call repeatedly — the ledger reference enforces idempotency.
    """
    if step is None:
        return {"ok": False, "awarded": False, "points_added": 0}
    if step.status not in DONE_STATUSES:
        return {"ok": True, "awarded": False, "points_added": 0}

    bonus = step_bonus_points(step.product_type)
    if bonus <= 0:
        return {"ok": True, "awarded": False, "points_added": 0}

    ref = _reference_for_step(step.id)

    with transaction.atomic():
        acc = _ensure_account(user)
        acc = LoyaltyAccount.objects.select_for_update().get(id=acc.id)

        if LoyaltyLedgerEntry.objects.filter(account=acc, reference=ref).exists():
            return {"ok": True, "awarded": False, "points_added": 0}

        LoyaltyLedgerEntry.objects.create(
            account=acc,
            entry_type=LoyaltyLedgerEntry.Type.EARN,
            points_delta=bonus,
            reference=ref,
            meta={
                "reason": "roadmap_step_completion",
                "plan_id": int(step.plan_id) if step.plan_id else None,
                "step_id": int(step.id),
                "step_index": int(step.step_index),
                "product_type": str(step.product_type or ""),
                "status": str(step.status),
                "awarded_at": timezone.now().isoformat(),
            },
        )
        acc.points_balance += bonus
        acc.save(update_fields=["points_balance"])

    return {"ok": True, "awarded": True, "points_added": bonus}


def award_completed_steps_for_plan(user, plan: RoadmapPlan | None) -> dict:
    """Walk the plan's steps and credit any not-yet-credited done step.

    Used after refresh_roadmap to catch steps that became COMPLETED/OWNED
    as a side effect of a purchase (or initial plan generation when the
    user already owned products).
    """
    if plan is None:
        return {"ok": False, "awarded_count": 0, "points_added": 0}

    awarded_count = 0
    total_points = 0
    steps: Iterable[RoadmapStep] = plan.steps.filter(status__in=list(DONE_STATUSES)).order_by("step_index")
    for step in steps:
        result = maybe_award_step_completion_bonus(user, step)
        if result.get("awarded"):
            awarded_count += 1
            total_points += int(result.get("points_added") or 0)

    return {"ok": True, "awarded_count": awarded_count, "points_added": total_points}
