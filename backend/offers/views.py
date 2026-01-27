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
from offers.services import get_or_assign_next_offer 
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

from drf_spectacular.utils import extend_schema, OpenApiExample, inline_serializer
from drf_spectacular.types import OpenApiTypes
from rest_framework import serializers
from backend.throttles import NextOfferRateThrottle

from audit.logging import log_event
from audit.models import AuditEvent

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
    throttle_classes = [NextOfferRateThrottle]

    @extend_schema(
        tags=["Offers"],
        description="Get active offer assignment or auto-assign next offer under budget/cooldown constraints.",
        responses={
            200: OpenApiTypes.OBJECT,
        },
        examples=[
            OpenApiExample(
                "Next offer (bundle-driven example)",
                response_only=True,
                value={
                    "assignment_id": 4,
                    "offer": {"id": 1, "name": "whosadik", "type": "discount", "value": "2.00", "estimated_cost": "5.00"},
                    "target": {
                        "scope": "product_id",
                        "value": 322,
                        "category": "makeup",
                        "product_type": "eyeshadow",
                        "picked_via": "bundle",
                        "bundle_mode": "cooccurrence",
                        "bundle_why": ["frequently purchased with product_id=330 (count=1)"],
                    },
                    "reason": {
                        "segment": "new_or_rare",
                        "picked_because": "max(score) under eligibility + cooldown + budget constraints",
                        "post_purchase": {"categories": ["makeup"], "product_types": ["foundation"]},
                    },
                },
            ),
        ],
    )
    def get(self, request):
        user = request.user
        now = dj_timezone.now()

        # (опционально) контекст из рутины
        profile_obj, _ = CustomerProfile.objects.get_or_create(user=user)
        prof = Profile(
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
                "strength","in_stock","concerns","attrs",
            )
        )
        owned_ids = list(
            OwnedProduct.objects.filter(user=user, is_active=True).values_list("product_id", flat=True)
        )

        routine = build_routine(profile=prof, products=products_for_routine, top_k=3, owned_product_ids=owned_ids)

        missing_steps = [
            x.get("step")
            for x in (routine["am"] + routine["pm"])
            if x.get("status") == "missing"
        ] or None

        with db_tx.atomic():
            existing = (
                OfferAssignment.objects.filter(user=user, is_redeemed=False)
                .order_by("-assigned_at")
                .first()
            )
            a = get_or_assign_next_offer(user=user, now=now, context_steps=missing_steps, post_ctx=None)

        created_new = bool(a) and (existing is None or a.id != existing.id)
        if created_new:
            t = a.target or {}
            log_event(
                request=request,
                action=AuditEvent.Action.NEXT_OFFER_ASSIGNED,
                entity_type="OfferAssignment",
                entity_id=a.id,
                status_code=200,  # GET вернул 200, но запись создалась
                meta={
                    "picked_via": t.get("picked_via"),
                    "scope": t.get("scope"),
                    "value": t.get("value"),
                    "category": t.get("category"),
                    "product_type": t.get("product_type"),
                    "context_steps": missing_steps,
                },
            )


        if not a:
            return Response({"offer": None, "reason": {"message": "No eligible offers"}})

        return Response({
            "assignment_id": a.id,
            "offer": {
                "id": a.offer.id,
                "name": a.offer.name,
                "type": a.offer.offer_type,
                "value": str(a.offer.value),
                "estimated_cost": str(a.offer.estimated_cost),
            },
            "target": a.target,
            "reason": a.reason,
            "expires_at": a.expires_at,
        })

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
        if assignment.offer.offer_type == "discount":
            return Response({"ok": False, "message": "Use /api/checkout with apply_assignment_id for discount offers"}, status=400)

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
            log_event(
                request=request,
                action=AuditEvent.Action.OFFER_REDEEM,
                entity_type="OfferAssignment",
                entity_id=assignment.id,
                status_code=200,
                meta={
                    "transaction_id": txn.id,
                    "offer_id": assignment.offer_id,
                    "offer_type": assignment.offer.offer_type,
                    "earned_points": earned_points,
                    "tier": account.tier.name if account.tier else None,
                    "target": target,
                    "eligible_total": str(eligible_total),
                    "multiplier": str(multiplier),
                },
            )

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

    @extend_schema(
        tags=["Offers"],
        description="Preview offer effect on provided cart (no DB writes).",
        request=OfferPreviewRequestSerializer,
        responses={
            200: inline_serializer(
                name="OfferPreviewResponse",
                fields={
                    "ok": serializers.BooleanField(),
                    "assignment_id": serializers.IntegerField(),
                    "offer": serializers.DictField(),
                    "target": serializers.DictField(),
                    "gross_total": serializers.CharField(),
                    "eligible_total": serializers.CharField(),
                    "discount_amount": serializers.CharField(),
                    "net_total": serializers.CharField(),
                    "estimated_points_earned": serializers.IntegerField(),
                },
            ),
            400: OpenApiTypes.OBJECT,
        },
        examples=[
            OpenApiExample(
                "Preview discount on target product",
                request_only=True,
                value={
                    "assignment_id": 4,
                    "items": [
                        {"product": 330, "quantity": 1, "unit_price": "12.99"},
                        {"product": 322, "quantity": 1, "unit_price": "12.99"},
                    ],
                },
            ),
            OpenApiExample(
                "Preview response (sample)",
                response_only=True,
                value={
                    "ok": True,
                    "assignment_id": 4,
                    "offer": {"id": 1, "name": "whosadik", "type": "discount", "value": "10.00"},
                    "target": {"scope": "product_id", "value": 330, "category": "makeup", "product_type": "foundation"},
                    "gross_total": "25.98",
                    "eligible_total": "12.99",
                    "discount_amount": "1.30",
                    "net_total": "24.68",
                    "estimated_points_earned": 25,
                },
            ),
        ],
    )
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
                    unit_price=Decimal(str(prod.price)),
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
        
        log_event(
            request=request,
            action=AuditEvent.Action.OFFER_PREVIEW,
            entity_type="OfferAssignment",
            entity_id=assignment.id,
            status_code=200,
            meta={
                "offer_id": assignment.offer_id,
                "offer_type": assignment.offer.offer_type,
                "target": target,
                "gross_total": calc.get("gross_total"),
                "net_total": calc.get("net_total"),
            },
        )

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
