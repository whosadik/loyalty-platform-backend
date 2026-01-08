from datetime import datetime, timedelta, timezone

from django.db import transaction as db_tx
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated

from transactions.models import OwnedProduct, Transaction
from loyalty.models import LoyaltyAccount, LoyaltyLedgerEntry, Tier
from .models import Offer, OfferAssignment, CampaignBudget
from .serializers import RedeemOfferRequestSerializer

from ml_logic.next_best_reward import compute_rfm, segment, pick_next_offer
from ml_logic.routine_builder import Profile, build_routine
from users_app.models import CustomerProfile
from catalog.models import Product


def _ensure_loyalty_account(user):
    account, created = LoyaltyAccount.objects.get_or_create(user=user)
    if account.tier_id is None:
        # на случай старых пользователей
        bronze, _ = Tier.objects.get_or_create(name="Bronze", defaults={"threshold_spend_90d": 0, "points_rate": 1.0})
        account.tier = bronze
        account.save(update_fields=["tier"])
    return account


def _recalculate_tier(user, now):
    """
    Пересчёт уровня по тратам за 90 дней.
    MVP: считает total_amount за 90 дней и выбирает максимальный threshold.
    """
    txs = list(Transaction.objects.filter(user=user).values("created_at", "total_amount"))
    rfm = compute_rfm(txs, now)
    spend_90d = rfm.monetary_90d

    tiers = list(Tier.objects.all().values("id", "name", "threshold_spend_90d", "points_rate"))
    tiers.sort(key=lambda t: float(t["threshold_spend_90d"]))

    chosen = tiers[0]
    for t in tiers:
        if spend_90d >= float(t["threshold_spend_90d"]):
            chosen = t

    account = _ensure_loyalty_account(user)
    if account.tier_id != chosen["id"]:
        account.tier_id = chosen["id"]
        account.save(update_fields=["tier"])

    return account


class MeNextOfferView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request):
        user = request.user
        now = datetime.now(timezone.utc)

        txs = list(Transaction.objects.filter(user=user).values("created_at", "total_amount"))
        rfm = compute_rfm(txs, now)
        seg = segment(rfm)

        last = (
            OfferAssignment.objects.filter(user=user)
            .order_by("-assigned_at")
            .values("assigned_at")
            .first()
        )
        last_days_ago = None
        if last:
            last_days_ago = (now - last["assigned_at"]).days

        budget, _ = CampaignBudget.objects.get_or_create(name="default")
        budget_left = float(budget.weekly_limit) - float(budget.weekly_spent)

        offers = list(
            Offer.objects.filter(is_active=True).values(
                "id",
                "is_active",
                "name",
                "offer_type",
                "value",
                "estimated_cost",
                "min_total_spend_90d",
                "cooldown_days",
                "allowed_steps",
            )
        )
        profile_obj, _ = CustomerProfile.objects.get_or_create(user=user)
        profile = Profile(
            skin_type=profile_obj.skin_type,
            goals=profile_obj.goals or [],
            avoid_flags=profile_obj.avoid_flags or [],
            budget=profile_obj.budget,
        )

        products_for_routine = list(
            Product.objects.all().values(
                "id", "name", "brand", "price", "step",
                "actives", "flags", "supported_skin_types",
                "strength", "in_stock",
            )
        )

        owned_ids = list(
            OwnedProduct.objects.filter(user=user, is_active=True).values_list("product_id", flat=True)
        )

        routine = build_routine(
            profile=profile,
            products=products_for_routine,
            top_k=3,
            owned_product_ids=owned_ids,
        )
        owned_steps = set(
            OwnedProduct.objects.filter(user=user, is_active=True)
            .select_related("product")
            .values_list("product__step", flat=True)
        )
        owned_steps_list = list(owned_steps)

        missing_steps = []
        for item in routine["am"] + routine["pm"]:
            if item.get("status") == "missing":
                missing_steps.append(item.get("step"))

        picked = pick_next_offer(
        rfm=rfm,
        segment_name=seg,
        offers=offers,
        last_assignment_days_ago=last_days_ago,
        budget_left=budget_left,
        context_steps=missing_steps or None,
        owned_steps=owned_steps_list,
    )

        if picked is None:
            return Response({"offer": None, "reason": {"segment": seg, "message": "No eligible offers"}})

        with db_tx.atomic():
            offer_obj = Offer.objects.select_for_update().get(id=picked["offer_id"])
            budget_obj = CampaignBudget.objects.select_for_update().get(id=budget.id)

            cost = float(offer_obj.estimated_cost)
            if float(budget_obj.weekly_spent) + cost > float(budget_obj.weekly_limit):
                return Response({"offer": None, "reason": {"segment": seg, "message": "Budget exceeded"}})

            assignment = OfferAssignment.objects.create(
                user=user,
                offer=offer_obj,
                reason=picked["reason"],
            )

            budget_obj.weekly_spent = float(budget_obj.weekly_spent) + cost
            budget_obj.save(update_fields=["weekly_spent"])

        return Response(
            {
                "assignment_id": assignment.id,
                "offer": {
                    "id": offer_obj.id,
                    "name": offer_obj.name,
                    "type": offer_obj.offer_type,
                    "value": str(offer_obj.value),
                    "estimated_cost": str(offer_obj.estimated_cost),
                },
                "reason": picked["reason"],
            }
        )


class RedeemOfferView(APIView):
    """
    Применяем оффер к транзакции и начисляем баллы в ledger.
    MVP: реализуем points_multiplier.
    """
    permission_classes = [IsAuthenticated]

    def post(self, request):
        req = RedeemOfferRequestSerializer(data=request.data)
        req.is_valid(raise_exception=True)

        assignment_id = req.validated_data["assignment_id"]
        transaction_id = req.validated_data["transaction_id"]

        now = datetime.now(timezone.utc)

        with db_tx.atomic():
            assignment = (
                OfferAssignment.objects.select_for_update()
                .select_related("offer")
                .get(id=assignment_id, user=request.user)
            )

            if assignment.is_redeemed:
                return Response({"ok": False, "message": "Offer already redeemed"}, status=400)

            txn = Transaction.objects.select_for_update().get(id=transaction_id, user=request.user)

            # Пересчёт tier перед начислением (MVP)
            account = _recalculate_tier(request.user, now)
            points_rate = float(account.tier.points_rate) if account.tier else 1.0

            base_points = int(round(float(txn.total_amount) * points_rate))

            discount_amount = 0.0
            multiplier = 1.0

            if assignment.offer.offer_type == "points_multiplier":
                multiplier = float(assignment.offer.value)

            earned_points = int(round(base_points * multiplier))

            # discount оффер: считаем скидку в процентах от total_amount (MVP)
            if assignment.offer.offer_type == "discount":
                percent = float(assignment.offer.value)  # например 10 = 10%
                discount_amount = round(float(txn.total_amount) * (percent / 100.0), 2)

            # gift: пока только фиксируем факт в meta (без инвентаря)


            # Пишем ledger
            LoyaltyLedgerEntry.objects.create(
                account=account,
                entry_type=LoyaltyLedgerEntry.Type.EARN,
                points_delta=earned_points,
                reference=f"txn:{txn.id}|offer_assignment:{assignment.id}",
                meta={
                    "txn_total": str(txn.total_amount),
                    "tier": account.tier.name if account.tier else None,
                    "points_rate": points_rate,
                    "base_points": base_points,
                    "multiplier": multiplier,
                    "offer_type": assignment.offer.offer_type,
                    "discount_amount": discount_amount,
                },
            )

            # Обновляем кэш баланса
            account.points_balance = int(account.points_balance) + earned_points
            account.save(update_fields=["points_balance"])

            assignment.is_redeemed = True
            assignment.save(update_fields=["is_redeemed"])

        return Response(
            {
                "ok": True,
                "earned_points": earned_points,
                "new_balance": account.points_balance,
                "tier": account.tier.name if account.tier else None,
                "discount_amount": discount_amount,
            }
        )
