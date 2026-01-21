from datetime import datetime, timedelta, timezone as dt_timezone
from collections import defaultdict
from django.utils import timezone as dj_timezone
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
from decimal import Decimal
from .models import OfferAssignment
from .serializers import OfferPreviewRequestSerializer

from checkout_app.pricing import Line, apply_offer_to_totals


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

def _compute_rec_explain(user, prof: RecUserProfile, now, category: str, product_type: str | None, strict: bool = False):
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

    # расширяем только если НЕ strict
    if (not recs) and (not strict) and (product_type is not None):
        recs = rec_recommend(
            prof=prof,
            products=products,
            owned_active_ids=owned_active_ids,
            context_product_ids=context_ids,
            category=category,
            product_type=None,
            limit=1,
            co=co,
        )

    if recs:
        top = recs[0]
        p = top["product"]
        return {
            "example_product_id": p["id"],
            "category": p["category"],
            "product_type": p["product_type"],
            "score": top.get("score"),
            "components": top.get("components", {}),
            "why": (top.get("why") or [])[:6],
        }

    return {"why": ["no recommendation candidate after filters"], "category": category, "product_type": product_type}


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
        top = recs[0]
        p = top["product"]
        return {
            "scope": "product_id",
            "value": p["id"],
            "category": p["category"],
            "product_type": p["product_type"],
            "rec_score": top.get("score"),
            "rec_components": top.get("components", {}),
            "rec_why": (top.get("why") or [])[:6],
        }

    # если рекомендаций нет — деградируем до product_type/category
    if product_type:
        return {"scope": "product_type", "value": product_type, "category": category}
    return {"scope": "category", "value": category}


class MeNextOfferView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request):
        user = request.user
        now = datetime.now(dt_timezone.utc)

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

            reason = picked["reason"] | {"context_steps": missing_steps or None}

            # строим explain всегда (не влияет на redeem логику)
            rec_prof = _build_rec_profile(profile_obj)

            if target.get("scope") == "product_id":
                # если product_id — explain можно взять прямо из target
                reason["rec_explain"] = {
                    "example_product_id": target.get("value"),
                    "category": target.get("category"),
                    "product_type": target.get("product_type"),
                    "score": target.get("rec_score"),
                    "components": target.get("rec_components", {}),
                    "why": target.get("rec_why", []),
                }
            else:
                # category / product_type / cart — делаем explain отдельно
                t_cat = target.get("category") or (target.get("value") if target.get("scope") == "category" else None) or "makeup"
                t_pt = target.get("value") if target.get("scope") == "product_type" else None
                if target.get("scope") == "cart":
                    t_cat = "makeup"  # или любая дефолтная
                    t_pt = None
                scope = target.get("scope")
                reason["rec_explain"] = _compute_rec_explain(
                    user, rec_prof, now, t_cat, t_pt,
                    strict=(scope == "product_type")
                )


            assignment = OfferAssignment.objects.create(
                user=user,
                offer=offer_obj,
                reason=reason,
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
                "reason": assignment.reason,
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

        now = datetime.now(dt_timezone.utc)

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
            
            # Пересчёт tier перед начислением (MVP)
            account = _recalculate_tier(request.user, now)
            points_rate = Decimal(str(account.tier.points_rate if account.tier else 1.0))

            lines = [
                Line(product=it.product, quantity=int(it.quantity), unit_price=Decimal(str(it.unit_price)))
                for it in items
            ]

            calc = apply_offer_to_totals(
                offer_type=assignment.offer.offer_type,
                offer_value=Decimal(str(assignment.offer.value)),
                target=target,
                lines=lines,
                points_rate=points_rate,
            )

            if not calc["ok"]:
                return Response({"ok": False, "message": calc.get("message", "Offer not applicable")}, status=400)

            discount_amount = Decimal(str(calc["discount_amount"]))
            eligible_total = Decimal(str(calc["eligible_total"]))
            base_points = int(calc["base_points"])
            earned_points = int(calc["estimated_points_earned"])
            multiplier = Decimal(str(calc["points_multiplier"]))

            # Пишем ledger
            LoyaltyLedgerEntry.objects.create(
                account=account,
                entry_type=LoyaltyLedgerEntry.Type.EARN,
                points_delta=earned_points,
                reference=f"txn:{txn.id}|offer_assignment:{assignment.id}",
                meta={
                    "txn_total": str(txn.total_amount),
                    "tier": account.tier.name if account.tier else None,
                    "points_rate": str(points_rate),
                    "base_points": base_points,
                    "multiplier": str(multiplier),
                    "offer_type": assignment.offer.offer_type,
                    "discount_amount": str(discount_amount),
                    "target": target,
                    "eligible_total": str(eligible_total),
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
                "discount_amount": str(discount_amount),
            }
        )

class MeOffersView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request):
        now = datetime.now(dt_timezone.utc)
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

class OfferPreviewView(APIView):
    permission_classes = [IsAuthenticated]

    def post(self, request):
        s = OfferPreviewRequestSerializer(data=request.data)
        s.is_valid(raise_exception=True)
        data = s.validated_data

        now = datetime.now(dt_timezone.utc)

        assignment = OfferAssignment.objects.select_related("offer").get(
            id=data["assignment_id"], user=request.user
        )

        if assignment.is_redeemed:
            return Response({"ok": False, "message": "Offer already redeemed"}, status=400)
        if assignment.expires_at and assignment.expires_at <= now:
            return Response({"ok": False, "message": "Offer expired"}, status=400)

        target = assignment.target or {"scope": "cart"}

        items = data["items"]
        product_ids = [it["product"] for it in items]
        products = Product.objects.in_bulk(product_ids)

        lines = []
        for it in items:
            pid = it["product"]
            prod = products.get(pid)
            if not prod:
                return Response({"ok": False, "message": f"Unknown product_id={pid}"}, status=400)

            lines.append(
                Line(
                    product=prod,
                    quantity=int(it["quantity"]),
                    unit_price=Decimal(str(it["unit_price"])),
                )
            )

        account, _ = LoyaltyAccount.objects.get_or_create(user=request.user)
        if account.tier_id is None:
            bronze, _ = Tier.objects.get_or_create(
                name="Bronze",
                defaults={"threshold_spend_90d": 0, "points_rate": 1.0},
            )
            account.tier = bronze
            account.save(update_fields=["tier"])

        points_rate = Decimal(str(account.tier.points_rate if account.tier else 1.0))

        calc = apply_offer_to_totals(
            offer_type=assignment.offer.offer_type,
            offer_value=Decimal(str(assignment.offer.value)),
            target=target,
            lines=lines,
            points_rate=points_rate,
        )

        if not calc["ok"]:
            return Response(calc, status=400)

        return Response(
            {
                "ok": True,
                "assignment_id": assignment.id,
                "offer": {
                    "id": assignment.offer.id,
                    "name": assignment.offer.name,
                    "type": assignment.offer.offer_type,
                    "value": str(assignment.offer.value),
                },
                "target": target,
                **calc,
                "tier": account.tier.name if account.tier else None,
                "points_rate": str(points_rate),
            }
        )
