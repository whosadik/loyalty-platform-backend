from datetime import datetime, timedelta, timezone
from collections import defaultdict

from django.db import transaction as db_tx
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated

from catalog.models import Product
from users_app.models import CustomerProfile
from transactions.models import OwnedProduct, Transaction, TransactionItem
from loyalty.models import LoyaltyAccount, LoyaltyLedgerEntry, Tier
from .models import Offer, OfferAssignment, CampaignBudget
from .serializers import RedeemOfferRequestSerializer

from ml_logic.next_best_reward import compute_rfm, segment, pick_next_offer
from ml_logic.routine_builder import Profile, build_routine
from ml_logic.recommender import (
    UserProfile as RecUserProfile,
    recommend as rec_recommend,
    build_cooccurrence,
)



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

def _load_products_for_recs():
    return list(
        Product.objects.all().values(
            "id","name","brand","price",
            "category","product_type",
            "concerns","attrs",
            "actives","flags","supported_skin_types","strength","in_stock"
        )
    )

def _cooccurrence_90d(now):
    since = now - timedelta(days=90)
    items = (
        TransactionItem.objects
        .filter(transaction__created_at__gte=since)
        .values("transaction_id", "product_id")
    )
    txn_map = defaultdict(list)
    for row in items:
        txn_map[row["transaction_id"]].append(row["product_id"])
    return build_cooccurrence(list(txn_map.values()))

def _build_rec_profile(cp: CustomerProfile) -> RecUserProfile:
    return RecUserProfile(
        skin_type=cp.skin_type,
        goals=cp.goals or [],
        avoid_flags=cp.avoid_flags or [],
        budget=cp.budget,
        hair=cp.hair_profile or {},
        makeup=cp.makeup_profile or {},
        fragrance=cp.fragrance_profile or {},
    )

def _pick_target_for_offer(user, offer_obj, now, context_steps: list[str] | None):
    """
    Возвращает dict target для OfferAssignment.
    """
    # 1) если рутина подсказала missing шаги — это самый сильный контекст
    if context_steps:
        if "spf" in context_steps and ("skincare" in (offer_obj.allowed_categories or []) or not offer_obj.allowed_categories):
            return {"scope": "product_type", "value": "spf", "category": "skincare"}

    # 2) если у оффера явно ограничены категории/типы — используем их
    allowed_cats = offer_obj.allowed_categories or []
    allowed_pts = offer_obj.allowed_product_types or []

    # 3) иначе выбираем категорию по профилю (минимально)
    cp, _ = CustomerProfile.objects.get_or_create(user=user)
    prof = _build_rec_profile(cp)

    if not allowed_cats:
        if (prof.fragrance or {}).get("liked_families") or (prof.fragrance or {}).get("liked_notes"):
            allowed_cats = ["fragrance"]
        elif prof.makeup:
            allowed_cats = ["makeup"]
        elif prof.hair:
            allowed_cats = ["haircare"]
        else:
            allowed_cats = ["skincare"]

    category = allowed_cats[0]
    product_type = allowed_pts[0] if allowed_pts else None

    # если scope=cart — таргет не нужен
    if offer_obj.target_scope == "cart":
        return {"scope": "cart"}

    # если scope=category / product_type — можно вернуть без конкретного товара
    if offer_obj.target_scope == "category":
        return {"scope": "category", "value": category}

    if offer_obj.target_scope == "product_type":
        if product_type is None:
            # fallback: любой тип внутри категории
            product_type = None
        return {"scope": "product_type", "value": product_type, "category": category}

    # scope=product_id → выберем top recommendation и зафиксируем конкретный товар
    products = _load_products_for_recs()
    owned_active_ids = list(
        OwnedProduct.objects.filter(user=user, is_active=True).values_list("product_id", flat=True)
    )
    co = _cooccurrence_90d(now)
    context_ids = owned_active_ids[:50]

    recs = rec_recommend(
        prof=prof,
        products=products,
        owned_active_ids=owned_active_ids,
        context_product_ids=context_ids,
        category=category,
        product_type=product_type,
        limit=1,
        co=co,
    )
    if recs:
        p = recs[0]["product"]
        return {"scope": "product_id", "value": p["id"], "category": p["category"], "product_type": p["product_type"]}

    # если рекомендаций нет — деградируем до product_type/category
    if product_type:
        return {"scope": "product_type", "value": product_type, "category": category}
    return {"scope": "category", "value": category}


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
                "allowed_categories",
                "allowed_product_types",
                "target_scope",
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
                "id","name","brand","price",
                "category","product_type","step",
                "actives","flags","supported_skin_types",
                "strength","in_stock",
                "concerns","attrs",
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
            OwnedProduct.objects.filter(user=user, is_active=True, product__category="skincare")
            .select_related("product")
            .values_list("product__product_type", flat=True)
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
            target = _pick_target_for_offer(user, offer_obj, now, missing_steps or None)
            assignment = OfferAssignment.objects.create(
                user=user,
                offer=offer_obj,
                reason=picked["reason"] | {"context_steps": missing_steps or None},
                target=target,
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
                    "target": assignment.target,
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
            items = list(txn.items.select_related("product").all())

            target = assignment.target or {"scope": "cart"}
            scope = target.get("scope", "cart")
            value = target.get("value")

            def is_eligible_item(it):
                p = it.product
                if scope == "cart":
                    return True
                if scope == "product_id":
                    return int(p.id) == int(value)
                if scope == "category":
                    return p.category == value
                if scope == "product_type":
                    # value может быть None → значит любой тип внутри категории
                    cat = target.get("category")
                    if cat and p.category != cat:
                        return False
                    if value:
                        return p.product_type == value
                    return True
                return True

            eligible_total = 0.0
            for it in items:
                if is_eligible_item(it):
                    eligible_total += float(it.unit_price) * int(it.quantity)

            # Пересчёт tier перед начислением (MVP)
            account = _recalculate_tier(request.user, now)
            points_rate = float(account.tier.points_rate) if account.tier else 1.0

            base_points = int(round(float(txn.total_amount) * points_rate))

            discount_amount = 0.0
            multiplier = 1.0

            # базовые поинты по всему чеку
            base_points = int(round(float(txn.total_amount) * points_rate))
            earned_points = base_points

            if assignment.offer.offer_type == "points_multiplier":
                multiplier = float(assignment.offer.value)

                if scope == "cart":
                    earned_points = int(round(base_points * multiplier))
                else:
                    eligible_points = int(round(eligible_total * points_rate))
                    rest_total = max(0.0, float(txn.total_amount) - eligible_total)
                    rest_points = int(round(rest_total * points_rate))
                    earned_points = rest_points + int(round(eligible_points * multiplier))

            elif assignment.offer.offer_type == "discount":
                percent = float(assignment.offer.value)
                discount_amount = round(eligible_total * (percent / 100.0), 2)


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
                    "target": target,
                    "eligible_total": eligible_total,
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

class MeOffersView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request):
        now = datetime.now(timezone.utc)
        qs = OfferAssignment.objects.filter(user=request.user, is_redeemed=False).select_related("offer").order_by("-assigned_at")

        out = []
        for a in qs[:50]:
            if a.expires_at and a.expires_at <= now:
                continue
            out.append(
                {
                    "assignment_id": a.id,
                    "assigned_at": a.assigned_at,
                    "expires_at": a.expires_at,
                    "target": a.target,
                    "reason": a.reason,
                    "offer": {
                        "id": a.offer.id,
                        "name": a.offer.name,
                        "type": a.offer.offer_type,
                        "value": str(a.offer.value),
                    },
                }
            )
        return Response(out)
