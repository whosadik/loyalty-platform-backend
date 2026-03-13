from __future__ import annotations

from datetime import timedelta
from decimal import Decimal

from django.conf import settings
from django.db import IntegrityError
from django.db import transaction as db_tx
from django.utils import timezone
from drf_spectacular.utils import OpenApiResponse, extend_schema
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework.views import APIView

from audit.logging import log_event
from audit.models import AuditEvent
from gift_cards.models import GiftCard, GiftCardLedgerEntry
from gift_cards.serializers import (
    GiftCardPurchaseRequestSerializer,
    GiftCardPurchaseResponseSerializer,
    GiftCardReceivedItemSerializer,
    GiftCardReceivedListResponseSerializer,
    GiftCardSentItemSerializer,
    GiftCardSentListResponseSerializer,
)
from gift_cards.services import (
    d2,
    generate_gift_card_code,
    gift_card_snapshot,
    refresh_gift_card_status,
    send_gift_card_message,
)
from loyalty.models import LoyaltyAccount, Tier
from loyalty.points import DEFAULT_POINTS_RATE
from transactions.models import Transaction


def _ensure_loyalty_account(user) -> LoyaltyAccount:
    account, _ = LoyaltyAccount.objects.get_or_create(user=user)
    if account.tier_id is None:
        bronze, _ = Tier.objects.get_or_create(
            name="Bronze",
            defaults={"threshold_spend_90d": 0, "points_rate": DEFAULT_POINTS_RATE},
        )
        account.tier = bronze
        account.save(update_fields=["tier"])
    return account


class GiftCardPurchaseView(APIView):
    permission_classes = [IsAuthenticated]

    @extend_schema(
        tags=["Gift Cards"],
        request=GiftCardPurchaseRequestSerializer,
        responses={
            201: GiftCardPurchaseResponseSerializer,
            200: GiftCardPurchaseResponseSerializer,
            400: OpenApiResponse(description="Validation error"),
            409: OpenApiResponse(description="Duplicate idempotency key without stored payload"),
        },
    )
    def post(self, request):
        serializer = GiftCardPurchaseRequestSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        data = serializer.validated_data

        idem = data.get("idempotency_key")
        if idem:
            conflicting = Transaction.objects.filter(user=request.user, idempotency_key=idem).first()
            if conflicting and (conflicting.pricing_meta or {}).get("type") != "gift_card_purchase":
                return Response({"ok": False, "message": "Duplicate idempotency_key"}, status=409)
            prev = Transaction.objects.filter(
                user=request.user,
                idempotency_key=idem,
                pricing_meta__type="gift_card_purchase",
            ).first()
            if prev and prev.pricing_meta:
                return Response({"ok": True, "idempotent_replay": True, **prev.pricing_meta})

        amount = d2(Decimal(data["amount"]))
        account = _ensure_loyalty_account(request.user)
        email_sent = False

        with db_tx.atomic():
            try:
                txn = Transaction.objects.create(
                    user=request.user,
                    channel=data.get("channel", "online"),
                    total_amount=amount,
                    idempotency_key=idem,
                )
            except IntegrityError:
                prev = Transaction.objects.get(user=request.user, idempotency_key=idem)
                if (prev.pricing_meta or {}).get("type") != "gift_card_purchase":
                    return Response({"ok": False, "message": "Duplicate idempotency_key"}, status=409)
                if prev.pricing_meta:
                    return Response({"ok": True, "idempotent_replay": True, **prev.pricing_meta})
                return Response({"ok": False, "message": "Duplicate idempotency_key"}, status=409)

            gift_card = GiftCard.objects.create(
                code=generate_gift_card_code(),
                purchaser=request.user,
                recipient_email=data["recipient_email"],
                message=data.get("message", ""),
                currency="KZT",
                initial_amount=amount,
                remaining_amount=amount,
                status=GiftCard.Status.ACTIVE,
                expires_at=timezone.now() + timedelta(days=getattr(settings, "GIFT_CARD_EXPIRES_IN_DAYS", 365)),
                purchase_transaction=txn,
            )
            GiftCardLedgerEntry.objects.create(
                gift_card=gift_card,
                entry_type=GiftCardLedgerEntry.EntryType.ISSUE,
                amount_delta=amount,
                transaction=txn,
                meta={"source": "gift_card_purchase"},
            )

            payload = {
                "type": "gift_card_purchase",
                "status": "completed",
                "transaction_id": txn.id,
                "gross_total": str(amount),
                "discount_amount": "0.00",
                "net_total": str(amount),
                "offer_applied": False,
                "offer_assignment_id": None,
                "eligible_total": "0.00",
                "points_redeemed": 0,
                "points_earned": 0,
                "new_balance": int(account.points_balance or 0),
                "gift_card": gift_card_snapshot(gift_card),
                "email_sent": False,
            }
            Transaction.objects.filter(id=txn.id).update(pricing_meta=payload)

        try:
            send_gift_card_message(gift_card)
            gift_card.sent_at = timezone.now()
            gift_card.save(update_fields=["sent_at", "updated_at"])
            email_sent = True
        except Exception:
            email_sent = False

        payload["gift_card"] = gift_card_snapshot(gift_card)
        payload["email_sent"] = email_sent
        Transaction.objects.filter(id=txn.id).update(pricing_meta=payload)

        log_event(
            request=request,
            action=AuditEvent.Action.CHECKOUT_CREATED,
            entity_type="GiftCard",
            entity_id=gift_card.id,
            status_code=201,
            meta={
                "transaction_id": txn.id,
                "amount": str(amount),
                "recipient_email": gift_card.recipient_email,
                "email_sent": email_sent,
            },
        )
        return Response({"ok": True, **payload}, status=201)


class MySentGiftCardsView(APIView):
    permission_classes = [IsAuthenticated]

    @extend_schema(
        tags=["Gift Cards"],
        responses={200: GiftCardSentListResponseSerializer},
    )
    def get(self, request):
        qs = GiftCard.objects.filter(purchaser=request.user).order_by("-created_at", "-id")
        return Response({"ok": True, "count": qs.count(), "items": GiftCardSentItemSerializer(qs, many=True).data})


class MyReceivedGiftCardsView(APIView):
    permission_classes = [IsAuthenticated]

    @extend_schema(
        tags=["Gift Cards"],
        responses={200: GiftCardReceivedListResponseSerializer},
    )
    def get(self, request):
        email = (request.user.email or "").strip()
        if not email:
            return Response({"ok": True, "count": 0, "items": []})

        items: list[GiftCard] = []
        qs = GiftCard.objects.filter(recipient_email__iexact=email).exclude(purchaser=request.user).order_by("-created_at", "-id")
        for gift_card in qs:
            items.append(refresh_gift_card_status(gift_card))

        return Response(
            {
                "ok": True,
                "count": len(items),
                "items": GiftCardReceivedItemSerializer(items, many=True).data,
            }
        )
