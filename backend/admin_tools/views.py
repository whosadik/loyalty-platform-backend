import os
from datetime import timedelta

from django.conf import settings
from django.core.cache import cache
from django.db import connection
from django.db.models import Count, DecimalField, ExpressionWrapper, F, Sum
from django.utils import timezone

from drf_spectacular.types import OpenApiTypes
from drf_spectacular.utils import OpenApiParameter, extend_schema
from rest_framework.permissions import IsAdminUser
from rest_framework.response import Response
from rest_framework import serializers
from rest_framework.views import APIView

from audit.models import AuditEvent
from backend.permissions import HasStaffPermission
from loyalty.models import LoyaltyLedgerEntry
from offers.admin_metrics import offers_events_kpis, offers_promo_efficiency_30d
from offers.models import OfferAssignment, OfferEvent
from recs_analytics.admin_metrics import recs_experiments_metrics, recs_metrics_30d
from recs_analytics.models import RecommendationEvent
from transactions.models import Transaction, TransactionItem


class AdminHealthView(APIView):
    permission_classes = [IsAdminUser]

    @extend_schema(
        tags=["Admin"],
        description="Health check: database + cache + basic counters.",
    )
    def get(self, request):
        db_ok = True
        db_error = None
        try:
            with connection.cursor() as cur:
                cur.execute("SELECT 1;")
                cur.fetchone()
        except Exception as e:
            db_ok = False
            db_error = str(e)

        cache_ok = True
        cache_error = None
        try:
            key = f"health:{timezone.now().timestamp()}"
            cache.set(key, "ok", timeout=10)
            cache_ok = cache.get(key) == "ok"
        except Exception as e:
            cache_ok = False
            cache_error = str(e)

        counts = {
            "transactions": Transaction.objects.count(),
            "offer_assignments": OfferAssignment.objects.count(),
            "offer_events": OfferEvent.objects.count(),
            "audit_events": AuditEvent.objects.count(),
        }

        return Response(
            {
                "ok": db_ok and cache_ok,
                "db": {"ok": db_ok, "error": db_error},
                "cache": {"ok": cache_ok, "error": cache_error},
                "counts": counts,
                "server_time": timezone.now().isoformat(),
            }
        )


def _txn_block(since):
    qs = Transaction.objects.filter(created_at__gte=since)
    count = qs.count()
    revenue = qs.aggregate(s=Sum("total_amount"))["s"] or 0
    unique_buyers = qs.values("user_id").distinct().count()
    aov = (float(revenue) / count) if count else 0.0
    return {
        "count": int(count),
        "revenue_sum": float(revenue),
        "aov": round(aov, 4),
        "unique_buyers": int(unique_buyers),
    }


def _points_block(since):
    qs = LoyaltyLedgerEntry.objects.filter(created_at__gte=since)
    earned = qs.filter(entry_type=LoyaltyLedgerEntry.Type.EARN).aggregate(s=Sum("points_delta"))["s"] or 0
    redeemed = qs.filter(entry_type=LoyaltyLedgerEntry.Type.REDEEM).aggregate(s=Sum("points_delta"))["s"] or 0
    return {"earned": int(earned), "redeemed": int(abs(redeemed))}


def _offers_lifecycle_block(since):
    ev = OfferEvent.objects.filter(created_at__gte=since)
    assigned = ev.filter(event_type=OfferEvent.Type.ASSIGNED).count()
    exposed = ev.filter(event_type=OfferEvent.Type.EXPOSED).count()
    clicked = ev.filter(event_type=OfferEvent.Type.CLICKED).count()
    redeemed = ev.filter(event_type=OfferEvent.Type.REDEEMED).count()
    expired = ev.filter(event_type=OfferEvent.Type.EXPIRED).count()

    ctr = (clicked / exposed) if exposed else 0.0
    redemption_rate = (redeemed / exposed) if exposed else 0.0

    return {
        "assigned": int(assigned),
        "exposed": int(exposed),
        "clicked": int(clicked),
        "redeemed": int(redeemed),
        "expired": int(expired),
        "ctr_clicks_exposed": round(ctr, 4),
        "redemption_rate_exposed": round(redemption_rate, 4),
    }


def _repeat_purchase_block(now):
    def _repeat_rate(days: int):
        since = now - timedelta(days=days)
        per_user = (
            Transaction.objects.filter(created_at__gte=since)
            .values("user_id")
            .annotate(txn_count=Count("id"))
        )
        active_users = per_user.count()
        repeat_users = per_user.filter(txn_count__gte=2).count()
        rate = (repeat_users / active_users) if active_users else 0.0
        return int(active_users), int(repeat_users), round(rate, 4)

    au30, ru30, rr30 = _repeat_rate(30)
    au60, ru60, rr60 = _repeat_rate(60)
    au90, ru90, rr90 = _repeat_rate(90)

    return {
        "repeat_purchase_rate_30d": rr30,
        "repeat_purchase_rate_60d": rr60,
        "repeat_purchase_rate_90d": rr90,
        "active_users_30d": au30,
        "repeat_users_30d": ru30,
        "active_users_60d": au60,
        "repeat_users_60d": ru60,
        "active_users_90d": au90,
        "repeat_users_90d": ru90,
    }


def _recs_block(since):
    qs = RecommendationEvent.objects.filter(created_at__gte=since)
    impressions = qs.filter(action=RecommendationEvent.Action.IMPRESSION).count()
    clicks = qs.filter(action=RecommendationEvent.Action.CLICK).count()
    purchases = qs.filter(action=RecommendationEvent.Action.PURCHASE_ATTRIBUTED).count()
    ctr = (clicks / impressions) if impressions else 0.0
    cr = (purchases / impressions) if impressions else 0.0
    return {
        "impressions": int(impressions),
        "clicks": int(clicks),
        "purchase_attributed": int(purchases),
        "ctr": round(ctr, 4),
        "cr": round(cr, 4),
    }


def _top_categories_30d(*, now):
    since_30 = now - timedelta(days=30)
    prev_since_30 = now - timedelta(days=60)
    price_expr = ExpressionWrapper(
        F("quantity") * F("unit_price"),
        output_field=DecimalField(max_digits=14, decimal_places=2),
    )

    current_rows = (
        TransactionItem.objects.filter(transaction__created_at__gte=since_30)
        .values("product__category")
        .annotate(revenue=Sum(price_expr))
        .order_by("-revenue")
    )
    previous_rows = (
        TransactionItem.objects.filter(
            transaction__created_at__gte=prev_since_30,
            transaction__created_at__lt=since_30,
        )
        .values("product__category")
        .annotate(revenue=Sum(price_expr))
    )
    previous_map = {
        row["product__category"] or "unknown": float(row["revenue"] or 0)
        for row in previous_rows
    }

    out = []
    for row in current_rows[:5]:
        category = row["product__category"] or "unknown"
        current_revenue = float(row["revenue"] or 0)
        previous_revenue = previous_map.get(category, 0.0)
        growth = None
        if previous_revenue > 0:
            growth = round(((current_revenue - previous_revenue) / previous_revenue) * 100, 2)

        out.append(
            {
                "name": category,
                "revenue": round(current_revenue, 2),
                "growth": growth,
            }
        )
    return out


def _overview_alerts_and_actions(*, tx30, recs30, offers30):
    alerts = []
    actions = []

    buyers_30d = int((tx30 or {}).get("unique_buyers") or 0)
    recs_ctr_30d = float((recs30 or {}).get("ctr") or 0.0)
    offer_redemption_30d = float((offers30 or {}).get("redemption_rate_exposed") or 0.0)

    if buyers_30d == 0:
        alerts.append(
            {
                "id": "buyers-zero-30d",
                "level": "error",
                "title": "Нет уникальных покупателей за 30 дней",
                "detail": "Транзакции за окно 30d отсутствуют.",
                "action": {"label": "Проверить метрики", "href": "/admin/metrics"},
            }
        )
        actions.append(
            {
                "id": "buyers-zero-30d-action",
                "priority": "high",
                "title": "Проверить источники транзакций",
                "reason": "В окне 30d нет уникальных покупателей.",
                "href": "/admin/metrics",
            }
        )

    if recs_ctr_30d > 0 and recs_ctr_30d < 0.02:
        alerts.append(
            {
                "id": "recs-ctr-low-30d",
                "level": "warning",
                "title": "Низкий CTR рекомендаций (30d)",
                "detail": f"CTR={recs_ctr_30d:.2%}, проверьте качество выдачи.",
                "action": {"label": "Открыть Recs Experiments", "href": "/admin/experiments"},
            }
        )
        actions.append(
            {
                "id": "recs-ctr-low-30d-action",
                "priority": "medium",
                "title": "Проверить алгоритмы рекомендаций",
                "reason": f"CTR рекомендаций за 30d = {recs_ctr_30d:.2%}.",
                "href": "/admin/experiments",
            }
        )

    if offer_redemption_30d > 0 and offer_redemption_30d < 0.01:
        alerts.append(
            {
                "id": "offers-redemption-low-30d",
                "level": "info",
                "title": "Низкая доля погашений офферов (30d)",
                "detail": f"Redemption={offer_redemption_30d:.2%}, нужна проверка офферов.",
                "action": {"label": "Открыть Campaigns", "href": "/admin/campaigns"},
            }
        )
        actions.append(
            {
                "id": "offers-redemption-low-30d-action",
                "priority": "low",
                "title": "Проверить активные кампании",
                "reason": f"Доля погашений офферов за 30d = {offer_redemption_30d:.2%}.",
                "href": "/admin/campaigns",
            }
        )

    return alerts[:5], actions[:5]


class AdminRecsExperimentsQuerySerializer(serializers.Serializer):
    days = serializers.IntegerField(required=False, min_value=1, max_value=365, default=30)
    experiment_id = serializers.CharField(required=False, allow_blank=True)
    variant = serializers.CharField(required=False, allow_blank=True)


class AdminOverviewView(APIView):
    permission_classes = [HasStaffPermission.with_perm("view_metrics")]

    @extend_schema(
        tags=["Admin"],
        description="Single dashboard payload for defense/demo: txns, offers lifecycle, promo, retention, recs.",
    )
    def get(self, request):
        ttl = int(getattr(settings, "ADMIN_OVERVIEW_CACHE_TTL_SECONDS", 60))
        db_name = connection.settings_dict.get("NAME", "default")
        cache_key = f"admin:overview:v1:{db_name}:{os.getpid()}"
        if ttl > 0:
            cached = cache.get(cache_key)
            if cached is not None:
                return Response(cached)

        now = timezone.now()
        since7 = now - timedelta(days=7)
        since30 = now - timedelta(days=30)
        recs_details = recs_metrics_30d()
        tx_7d = _txn_block(since7)
        tx_30d = _txn_block(since30)
        offers_7d = _offers_lifecycle_block(since7)
        offers_30d = _offers_lifecycle_block(since30)
        recs_7d = _recs_block(since7)
        recs_30d = _recs_block(since30)
        alerts, actions = _overview_alerts_and_actions(tx30=tx_30d, recs30=recs_30d, offers30=offers_30d)

        payload = {
            "ok": True,
            "generated_at": now.isoformat(),
            "transactions": {
                "7d": tx_7d,
                "30d": tx_30d,
            },
            "points": {
                "7d": _points_block(since7),
                "30d": _points_block(since30),
            },
            "offers": {
                "7d": offers_7d,
                "30d": offers_30d,
                "events_kpis": offers_events_kpis(),
                "promo_efficiency_30d": offers_promo_efficiency_30d(),
            },
            "retention": _repeat_purchase_block(now),
            "recs": {
                "7d": recs_7d,
                "30d": recs_30d,
                "details_30d": recs_details,
                "experiments_30d": recs_details.get("by_experiment", {}),
            },
            "alerts": alerts,
            "recommended_actions": actions,
            "top_categories": _top_categories_30d(now=now),
        }
        if ttl > 0:
            cache.set(cache_key, payload, timeout=ttl)
        return Response(payload)


class AdminRecsExperimentsView(APIView):
    permission_classes = [HasStaffPermission.with_perm("view_metrics")]

    @extend_schema(
        tags=["Admin"],
        description="Detailed recommendation experiment KPIs with optional filters.",
        parameters=[
            OpenApiParameter("days", OpenApiTypes.INT, required=False),
            OpenApiParameter("experiment_id", OpenApiTypes.STR, required=False),
            OpenApiParameter("variant", OpenApiTypes.STR, required=False),
        ],
    )
    def get(self, request):
        q = AdminRecsExperimentsQuerySerializer(data=request.query_params)
        q.is_valid(raise_exception=True)
        days = int(q.validated_data.get("days") or 30)
        experiment_id = (q.validated_data.get("experiment_id") or "").strip() or None
        variant = (q.validated_data.get("variant") or "").strip() or None

        ttl = int(getattr(settings, "ADMIN_METRICS_CACHE_TTL_SECONDS", 60))
        db_name = connection.settings_dict.get("NAME", "default")
        cache_key = (
            f"admin:recs:experiments:v1:{db_name}:{os.getpid()}:{days}:{experiment_id or 'all'}:{variant or 'all'}"
        )
        if ttl > 0:
            cached = cache.get(cache_key)
            if cached is not None:
                return Response(cached)

        payload = {"ok": True, **recs_experiments_metrics(days=days, experiment_id=experiment_id, variant=variant)}
        if ttl > 0:
            cache.set(cache_key, payload, timeout=ttl)
        return Response(payload)
