from __future__ import annotations

from django.conf import settings
from django.db import transaction
from django.utils import timezone

from loyalty.models import LoyaltyAccount, LoyaltyLedgerEntry, Tier
from users_app.models import CustomerProfile


def _ensure_account(user) -> LoyaltyAccount:
    acc, _ = LoyaltyAccount.objects.get_or_create(user=user)
    if acc.tier_id is None:
        bronze, _ = Tier.objects.get_or_create(
            name="Bronze",
            defaults={"threshold_spend_90d": 0, "points_rate": 1.0},
        )
        acc.tier = bronze
        acc.save(update_fields=["tier"])
    return acc


def _is_profile_complete(cp: CustomerProfile) -> bool:
    if not cp.skin_type:
        return False
    if not (cp.goals or []):
        return False
    if cp.budget is None or cp.budget == "":
        return False
    return True


def maybe_award_profile_completion_bonus(user, cp: CustomerProfile) -> dict:
    """
    Returns dict with:
      ok, awarded(bool), points_added(int), completed(bool)
    """
    now = timezone.now()
    completed = _is_profile_complete(cp)

    if completed and cp.profile_completed_at is None:
        cp.profile_completed_at = now
        cp.save(update_fields=["profile_completed_at"])

    if cp.profile_completion_rewarded_at is not None:
        return {"ok": True, "awarded": False, "points_added": 0, "completed": completed}

    if not completed:
        return {"ok": True, "awarded": False, "points_added": 0, "completed": False}

    bonus = int(getattr(settings, "PROFILE_COMPLETION_BONUS_POINTS", 50))
    ref = f"profile_completion:user:{user.id}"

    with transaction.atomic():
        acc = _ensure_account(user)
        acc = LoyaltyAccount.objects.select_for_update().get(id=acc.id)

        if LoyaltyLedgerEntry.objects.filter(account=acc, reference=ref).exists():
            if cp.profile_completion_rewarded_at is None:
                cp.profile_completion_rewarded_at = now
                cp.save(update_fields=["profile_completion_rewarded_at"])
            return {"ok": True, "awarded": False, "points_added": 0, "completed": True}

        LoyaltyLedgerEntry.objects.create(
            account=acc,
            entry_type=LoyaltyLedgerEntry.Type.EARN,
            points_delta=bonus,
            reference=ref,
            meta={"reason": "profile_completion_bonus"},
        )
        acc.points_balance += bonus
        acc.save(update_fields=["points_balance"])

        cp.profile_completion_rewarded_at = now
        if cp.profile_completed_at is None:
            cp.profile_completed_at = now
        cp.save(update_fields=["profile_completion_rewarded_at", "profile_completed_at"])

    return {"ok": True, "awarded": True, "points_added": bonus, "completed": True}
