from __future__ import annotations

from decimal import Decimal
from datetime import datetime, timedelta

from django.db import transaction as db_tx
from django.utils import timezone
from django.db.models import Sum

from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated

from checkout_app.serializers import CheckoutRequestSerializer

from transactions.models import Transaction, TransactionItem, OwnedProduct
from offers.models import OfferAssignment
from loyalty.models import LoyaltyAccount, LoyaltyLedgerEntry, Tier
from catalog.models import Product
from offers.services import get_or_assign_next_offer


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


def _recalculate_tier(user, now) -> LoyaltyAccount:
    since = now - timedelta(days=90)
    spend = (
        Transaction.objects.filter(user=user, created_at__gte=since)
        .aggregate(s=Sum("total_amount"))["s"]
        or Decimal("0")
    )

    tiers = list(Tier.objects.all().values("id", "threshold_spend_90d"))
    tiers.sort(key=lambda t: Decimal(str(t["threshold_spend_90d"])))

    chosen_id = tiers[0]["id"] if tiers else None
    for t in tiers:
        if spend >= Decimal(str(t["threshold_spend_90d"])):
            chosen_id = t["id"]

    acc = _ensure_account(user)
    if chosen_id and acc.tier_id != chosen_id:
        acc.tier_id = chosen_id
        acc.save(update_fields=["tier"])
    return acc


def _eligible_total(items: list[TransactionItem], target: dict) -> Decimal:
    scope = (target or {}).get("scope", "cart")
    value = (target or {}).get("value")
    cat = (target or {}).get("category")

    def ok(it: TransactionItem) -> bool:
        p = it.product
        if scope == "cart":
            return True
        if scope == "product_id":
            return int(p.id) == int(value)
        if scope == "category":
            return p.category == value
        if scope == "product_type":
            if cat and p.category != cat:
                return False
            if value:
                return p.product_type == value
            return True
        return True

    total = Decimal("0")
    for it in items:
        if ok(it):
            total += (it.unit_price * it.quantity)
    return total


class CheckoutView(APIView):
    permission_classes = [IsAuthenticated]

    def post(self, request):
        req = CheckoutRequestSerializer(data=request.data)
        req.is_valid(raise_exception=True)
        data = req.validated_data

        now = timezone.now()

        with db_tx.atomic():
            # lock loyalty account for safe redeem/earn
            account = _ensure_account(request.user)
            account = LoyaltyAccount.objects.select_for_update().get(id=account.id)

            # 1) create transaction + items
            txn = Transaction.objects.create(user=request.user, channel=data.get("channel", "offline"))

            total = Decimal("0")
            created_items: list[TransactionItem] = []

            for it in data["items"]:
                product_id = it["product"]
                qty = int(it["quantity"])
                unit_price = Decimal(str(it["unit_price"]))

                # ensure product exists
                prod = Product.objects.select_for_update().get(id=product_id)

                ti = TransactionItem.objects.create(
                    transaction=txn,
                    product=prod,
                    quantity=qty,
                    unit_price=unit_price,
                )
                created_items.append(ti)
                total += unit_price * qty

                # owned update (qty/active/last_acquired)
                owned, _ = OwnedProduct.objects.get_or_create(user=request.user, product=prod)
                owned.quantity_total = int(owned.quantity_total or 0) + qty
                owned.is_active = True
                owned.last_acquired_at = now
                owned.save(update_fields=["quantity_total", "is_active", "last_acquired_at"])

            txn.total_amount = total
            txn.save(update_fields=["total_amount"])

            # 2) recalc tier (based on spend last 90d, includes this txn after save)
            account = _recalculate_tier(request.user, now)
            account = LoyaltyAccount.objects.select_for_update().get(id=account.id)
            points_rate = Decimal(str(account.tier.points_rate if account.tier else 1.0))

            # 3) optional offer apply (discount / points_multiplier)
            discount_amount = Decimal("0")
            multiplier = Decimal("1")
            eligible_total = Decimal("0")
            applied_assignment_id = None
            applied_target = None

            apply_assignment_id = data.get("apply_assignment_id")
            if apply_assignment_id is not None:
                assignment = (
                    OfferAssignment.objects.select_for_update()
                    .select_related("offer")
                    .get(id=apply_assignment_id, user=request.user)
                )

                if assignment.is_redeemed:
                    return Response({"ok": False, "message": "Offer already redeemed"}, status=400)

                if assignment.expires_at and assignment.expires_at <= now:
                    return Response({"ok": False, "message": "Offer expired"}, status=400)

                applied_target = assignment.target or {"scope": "cart"}
                eligible_total = _eligible_total(created_items, applied_target)

                if applied_target.get("scope") != "cart" and eligible_total <= 0:
                    return Response({"ok": False, "message": "No eligible items for this offer in transaction"}, status=400)

                if assignment.offer.offer_type == "discount":
                    percent = Decimal(str(assignment.offer.value))
                    discount_amount = (eligible_total * (percent / Decimal("100"))).quantize(Decimal("0.01"))

                elif assignment.offer.offer_type == "points_multiplier":
                    multiplier = Decimal(str(assignment.offer.value))

                assignment.is_redeemed = True
                assignment.save(update_fields=["is_redeemed"])
                applied_assignment_id = assignment.id

            gross_total = total
            net_total = gross_total - discount_amount
            if net_total < 0:
                net_total = Decimal("0")

            # 4) optional redeem points (spend points)
            points_redeemed = 0
            redeem_points = data.get("redeem_points")
            if redeem_points is not None:
                redeem_points = int(redeem_points)
                if account.points_balance < redeem_points:
                    return Response({"ok": False, "message": "Insufficient points"}, status=400)

                LoyaltyLedgerEntry.objects.create(
                    account=account,
                    entry_type=LoyaltyLedgerEntry.Type.REDEEM,
                    points_delta=-redeem_points,
                    reference=f"checkout:txn:{txn.id}",
                    meta={"txn_id": txn.id},
                )
                account.points_balance -= redeem_points
                points_redeemed = redeem_points

            # 5) earn points (tier + multiplier)
            # points начисляем от net_total (после скидки)
            base_points = int(round(float(net_total * points_rate)))

            points_earned = base_points
            if multiplier != Decimal("1"):
                # если multiplier и scope не cart — умножаем только eligible часть
                scope = (applied_target or {}).get("scope", "cart")
                if scope == "cart":
                    points_earned = int(round(float(Decimal(base_points) * multiplier)))
                else:
                    # discount тут нет (offer_type points_multiplier), значит net_total == gross_total
                    eligible_points = int(round(float(eligible_total * points_rate)))
                    rest_total = gross_total - eligible_total
                    if rest_total < 0:
                        rest_total = Decimal("0")
                    rest_points = int(round(float(rest_total * points_rate)))
                    points_earned = rest_points + int(round(float(Decimal(eligible_points) * multiplier)))

            LoyaltyLedgerEntry.objects.create(
                account=account,
                entry_type=LoyaltyLedgerEntry.Type.EARN,
                points_delta=points_earned,
                reference=f"checkout:txn:{txn.id}",
                meta={
                    "txn_id": txn.id,
                    "gross_total": str(gross_total),
                    "discount_amount": str(discount_amount),
                    "net_total": str(net_total),
                    "tier": account.tier.name if account.tier else None,
                    "points_rate": str(points_rate),
                    "base_points": base_points,
                    "multiplier": str(multiplier),
                    "offer_assignment_id": applied_assignment_id,
                    "target": applied_target,
                    "eligible_total": str(eligible_total),
                    "points_redeemed": points_redeemed,
                },
            )

            account.points_balance += points_earned
            account.save(update_fields=["points_balance"])

                    # Auto-assign next offer after successful checkout
            next_assignment = get_or_assign_next_offer(
                user=request.user,
                now=now,
                context_steps=None,  # позже можно передавать missing steps из рутины или post-purchase context
            )

            next_offer_payload = None
            if next_assignment:
                next_offer_payload = {
                    "assignment_id": next_assignment.id,
                    "offer": {
                        "id": next_assignment.offer.id,
                        "name": next_assignment.offer.name,
                        "type": next_assignment.offer.offer_type,
                        "value": str(next_assignment.offer.value),
                        "estimated_cost": str(getattr(next_assignment.offer, "estimated_cost", "")),
                    },
                    "target": next_assignment.target,
                    "reason": next_assignment.reason,
                    "expires_at": getattr(next_assignment, "expires_at", None),
                }

        return Response(
            {
                "ok": True,
                "transaction_id": txn.id,
                "gross_total": str(gross_total),
                "discount_amount": str(discount_amount),
                "net_total": str(net_total),
                "offer_applied": applied_assignment_id is not None,
                "offer_assignment_id": applied_assignment_id,
                "target": applied_target,
                "eligible_total": str(eligible_total),
                "points_redeemed": points_redeemed,
                "points_earned": points_earned,
                "new_balance": account.points_balance,
                "tier": account.tier.name if account.tier else None,
                "next_offer": next_offer_payload,
            }
        )
