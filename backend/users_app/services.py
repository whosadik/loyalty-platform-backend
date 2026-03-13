from __future__ import annotations

from datetime import timedelta

from django.conf import settings
from django.db import transaction
from django.db.models import Count, Max, Sum
from django.utils import timezone

from loyalty.models import LoyaltyAccount, LoyaltyLedgerEntry, Tier
from loyalty.points import DEFAULT_POINTS_RATE
from transactions.models import Transaction
from transactions.models import TransactionItem
from users_app.models import CustomerProfile


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


def _is_profile_complete(cp: CustomerProfile) -> bool:
    if not cp.skin_type:
        return False
    if not (cp.goals or []):
        return False
    if cp.budget is None or cp.budget == "":
        return False
    return True


def is_profile_complete(cp: CustomerProfile) -> bool:
    return _is_profile_complete(cp)


def favorite_category_window_days() -> int:
    return int(getattr(settings, "FAVORITE_CATEGORY_WINDOW_DAYS", 90))


def favorite_category_snapshot(user, now=None, *, max_signals: int = 5) -> dict:
    """
    Returns favorite category with explainability signals over the configured window.
    """
    now = now or timezone.now()
    window_days = favorite_category_window_days()
    since = now - timedelta(days=window_days)

    rows = list(
        TransactionItem.objects.filter(
            transaction__user=user,
            transaction__created_at__gte=since,
        )
        .values("product__category")
        .annotate(
            total_qty=Sum("quantity"),
            line_count=Count("id"),
            last_at=Max("transaction__created_at"),
        )
        .order_by("-total_qty", "-line_count", "-last_at", "product__category")
    )
    window_items_qs = TransactionItem.objects.filter(
        transaction__user=user,
        transaction__created_at__gte=since,
    )
    transactions_qs = Transaction.objects.filter(
        user=user,
        created_at__gte=since,
    )

    signals = [
        {
            "category": r["product__category"],
            "total_qty": int(r["total_qty"] or 0),
            "line_count": int(r["line_count"] or 0),
            "last_at": r["last_at"].isoformat() if r["last_at"] else None,
        }
        for r in rows[:max_signals]
    ]

    top = rows[0] if rows else None
    # products_bought is a quantity-based aggregate across transaction items in the window.
    products_bought = int(window_items_qs.aggregate(total=Sum("quantity"))["total"] or 0)
    total_spent = transactions_qs.aggregate(total=Sum("total_amount"))["total"]
    currency = (
        window_items_qs.exclude(product__currency="")
        .values_list("product__currency", flat=True)
        .first()
    )
    return {
        "favorite_category": top["product__category"] if top else None,
        "window_days": int(window_days),
        "products_bought": products_bought,
        "total_spent": str(total_spent) if total_spent is not None else "0",
        "currency": currency or None,
        "history_items_considered": int(sum(int(r["line_count"] or 0) for r in rows)),
        "window_start": since.isoformat(),
        "window_end": now.isoformat(),
        "picked_by": ["total_qty", "line_count", "last_at", "category"],
        "signals": signals,
    }


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
