from django.db import connection
from django.core.cache import cache
from django.utils import timezone

from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.permissions import IsAdminUser

from audit.models import AuditEvent
from offers.models import OfferAssignment
from transactions.models import Transaction
from datetime import timedelta
from django.db.models import Sum, Count
from loyalty.models import LoyaltyLedgerEntry
from drf_spectacular.utils import extend_schema, OpenApiParameter

class AdminHealthView(APIView):
    permission_classes = [IsAdminUser]

    @extend_schema(
        tags=["Admin"],
        description="Health check: database + cache + basic counters.",
    )
    def get(self, request):
        # DB check
        db_ok = True
        db_error = None
        try:
            with connection.cursor() as cur:
                cur.execute("SELECT 1;")
                cur.fetchone()
        except Exception as e:
            db_ok = False
            db_error = str(e)

        # Cache check
        cache_ok = True
        cache_error = None
        try:
            key = f"health:{timezone.now().timestamp()}"
            cache.set(key, "ok", timeout=10)
            cache_ok = (cache.get(key) == "ok")
        except Exception as e:
            cache_ok = False
            cache_error = str(e)

        counts = {
            "transactions": Transaction.objects.count(),
            "offer_assignments": OfferAssignment.objects.count(),
            "audit_events": AuditEvent.objects.count(),
        }

        return Response({
            "ok": db_ok and cache_ok,
            "db": {"ok": db_ok, "error": db_error},
            "cache": {"ok": cache_ok, "error": cache_error},
            "counts": counts,
            "server_time": timezone.now().isoformat(),
        })

from offers.admin_metrics import offers_metrics_30d

class AdminOverviewView(APIView):
    permission_classes = [IsAdminUser]

    @extend_schema(
        tags=["Admin"],
        description="Aggregated overview for last 7 and 30 days.",
    )
    def get(self, request):
        now = timezone.now()
        since7 = now - timedelta(days=7)
        since30 = now - timedelta(days=30)

        def txn_block(since):
            qs = Transaction.objects.filter(created_at__gte=since)
            total = qs.count()
            revenue = qs.aggregate(s=Sum("total_amount"))["s"] or 0
            return {"count": total, "revenue": float(revenue)}

        def points_block(since):
            qs = LoyaltyLedgerEntry.objects.filter(created_at__gte=since)
            earned = qs.filter(entry_type="EARN").aggregate(s=Sum("points_delta"))["s"] or 0
            redeemed = qs.filter(entry_type="REDEEM").aggregate(s=Sum("points_delta"))["s"] or 0
            return {"earned": int(earned), "redeemed": int(abs(redeemed))}


        return Response({
            "ok": True,
            "last_7d": {
                "transactions": txn_block(since7),
                "points": points_block(since7),
                "offers": {
                    "assignments": OfferAssignment.objects.filter(assigned_at__gte=since7).count(),
                    "redemptions": OfferAssignment.objects.filter(assigned_at__gte=since7, is_redeemed=True).count(),
                },
            },
            "last_30d": {
                "transactions": txn_block(since30),
                "points": points_block(since30),
                "offers_v3": offers_metrics_30d(),
            },
        })
