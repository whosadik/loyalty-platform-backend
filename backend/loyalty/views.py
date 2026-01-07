from django.db import transaction as db_tx
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated

from .models import LoyaltyAccount, LoyaltyLedgerEntry, Tier
from .serializers import RedeemPointsRequestSerializer


def _ensure_account(user) -> LoyaltyAccount:
    account, _ = LoyaltyAccount.objects.get_or_create(user=user)
    if account.tier_id is None:
        bronze, _ = Tier.objects.get_or_create(
            name="Bronze",
            defaults={"threshold_spend_90d": 0, "points_rate": 1.0},
        )
        account.tier = bronze
        account.save(update_fields=["tier"])
    return account


class MeLoyaltyStatusView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request):
        account = _ensure_account(request.user)
        return Response(
            {
                "tier": account.tier.name if account.tier else None,
                "points_balance": account.points_balance,
            }
        )


class RedeemPointsView(APIView):
    permission_classes = [IsAuthenticated]

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
