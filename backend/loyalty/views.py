from datetime import timedelta
from decimal import Decimal

from django.db import transaction as db_tx
from django.db.models import Sum
from django.utils import timezone
from drf_spectacular.utils import OpenApiResponse, extend_schema
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated

from backend.api_serializers import ApiErrorSerializer
from rest_framework.pagination import PageNumberPagination
from transactions.models import Transaction
from .models import LoyaltyAccount, LoyaltyLedgerEntry, Tier
from .points import DEFAULT_POINTS_RATE
from .serializers import (
    LoyaltyLedgerEntrySerializer,
    MeLoyaltyResponseSerializer,
    RedeemPointsRequestSerializer,
    RedeemPointsResponseSerializer,
)


def _ensure_account(user) -> LoyaltyAccount:
    account, _ = LoyaltyAccount.objects.get_or_create(user=user)
    if account.tier_id is None:
        bronze, _ = Tier.objects.get_or_create(
            name="Bronze",
            defaults={"threshold_spend_90d": 0, "points_rate": DEFAULT_POINTS_RATE},
        )
        account.tier = bronze
        account.save(update_fields=["tier"])
    return account


class MeLoyaltyStatusView(APIView):
    permission_classes = [IsAuthenticated]

    @extend_schema(
        tags=["Loyalty"],
        responses={200: MeLoyaltyResponseSerializer},
    )
    def get(self, request):
        account = _ensure_account(request.user)

        since = timezone.now() - timedelta(days=90)
        spend_90d = (
            Transaction.objects.filter(user=request.user, created_at__gte=since)
            .aggregate(s=Sum("total_amount"))["s"]
            or Decimal("0")
        )

        tiers = list(
            Tier.objects.all()
            .values("name", "threshold_spend_90d")
            .order_by("threshold_spend_90d")
        )

        current_threshold = (
            Decimal(str(account.tier.threshold_spend_90d)) if account.tier else Decimal("0")
        )
        next_tier_name = None
        next_tier_threshold = None
        for t in tiers:
            thr = Decimal(str(t["threshold_spend_90d"]))
            if thr > current_threshold:
                next_tier_name = t["name"]
                next_tier_threshold = float(thr)
                break

        return Response(
            {
                "tier": account.tier.name if account.tier else None,
                "points_balance": account.points_balance,
                "spend_90d": float(spend_90d),
                "next_tier": next_tier_name,
                "next_tier_threshold": next_tier_threshold,
            }
        )


class RedeemPointsView(APIView):
    permission_classes = [IsAuthenticated]

    @extend_schema(
        tags=["Loyalty"],
        request=RedeemPointsRequestSerializer,
        responses={
            200: RedeemPointsResponseSerializer,
            400: OpenApiResponse(response=ApiErrorSerializer),
        },
    )
    def post(self, request):
        req = RedeemPointsRequestSerializer(data=request.data)
        req.is_valid(raise_exception=True)

        points = int(req.validated_data["points"])
        reference = req.validated_data.get("reference") or ""

        with db_tx.atomic():
            account = LoyaltyAccount.objects.select_for_update().get(user=request.user)

            if account.points_balance < points:
                return Response(
                    {"ok": False, "message": "Insufficient points"},
                    status=400,
                )

            LoyaltyLedgerEntry.objects.create(
                account=account,
                entry_type=LoyaltyLedgerEntry.Type.REDEEM,
                points_delta=-points,
                reference=reference or "manual_redeem",
                meta={"requested_points": points},
            )

            account.points_balance -= points
            account.save(update_fields=["points_balance"])

        return Response({"ok": True, "new_balance": account.points_balance})


class LoyaltyHistoryPagination(PageNumberPagination):
    page_size = 25
    page_size_query_param = "page_size"
    max_page_size = 100


class MeLoyaltyHistoryView(APIView):
    permission_classes = [IsAuthenticated]
    pagination_class = LoyaltyHistoryPagination

    @extend_schema(
        tags=["Loyalty"],
        responses={200: LoyaltyLedgerEntrySerializer(many=True)},
    )
    def get(self, request):
        qs = (
            LoyaltyLedgerEntry.objects
            .filter(account__user=request.user)
            .order_by("-created_at", "-id")
        )

        entry_type = (request.query_params.get("entry_type") or "").strip().lower()
        valid_types = {
            LoyaltyLedgerEntry.Type.EARN,
            LoyaltyLedgerEntry.Type.REDEEM,
            LoyaltyLedgerEntry.Type.ADJUST,
        }
        if entry_type in valid_types:
            qs = qs.filter(entry_type=entry_type)

        account_balance = (
            LoyaltyAccount.objects.filter(user=request.user)
            .values_list("points_balance", flat=True)
            .first()
        )

        paginator = self.pagination_class()
        page = paginator.paginate_queryset(qs, request, view=self)
        serializer = LoyaltyLedgerEntrySerializer(
            page if page is not None else qs,
            many=True,
            context={"request": request},
        )

        if page is not None:
            paginated = paginator.get_paginated_response(serializer.data)
            paginated.data["ok"] = True
            paginated.data["points_balance"] = int(account_balance or 0)
            return paginated

        return Response(
            {
                "ok": True,
                "points_balance": int(account_balance or 0),
                "results": serializer.data,
            }
        )
