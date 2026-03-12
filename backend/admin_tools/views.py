import os
from datetime import timedelta
from time import perf_counter

from django.conf import settings
from django.core.cache import cache
from django.db import connection
from django.db.models import Count, DecimalField, ExpressionWrapper, F, Max, Sum
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
from offers.models import Offer, OfferAssignment, OfferEvent
from recs_analytics.admin_metrics import recs_experiments_metrics, recs_metrics_30d
from recs_analytics.models import RecommendationEvent
from transactions.models import Transaction, TransactionItem


PROCESS_STARTED_AT = timezone.now()


class AdminHealthView(APIView):
    permission_classes = [IsAdminUser]

    @extend_schema(
        tags=["Admin"],
        description="Health check with live service status, latency and subsystem activity.",
    )
    def get(self, request):
        now = timezone.now()
        services = [
            _db_health_service(now),
            _cache_health_service(now),
            _transactions_health_service(now),
            _offers_health_service(now),
            _audit_health_service(now),
            _recommendations_health_service(now),
        ]

        summary = {
            "healthy_services": sum(1 for service in services if service["status"] == "ok"),
            "degraded_services": sum(1 for service in services if service["status"] == "degraded"),
            "down_services": sum(1 for service in services if service["status"] == "down"),
            "total_services": len(services),
        }
        overall_status = _overall_health_status(services)
        counts = {
            "transactions": int(_service_metric(services, "transactions", "total_count") or 0),
            "offer_assignments": int(_service_metric(services, "offers", "total_assignments") or 0),
            "offer_events": int(_service_metric(services, "offers", "total_events") or 0),
            "audit_events": int(_service_metric(services, "audit", "total_count") or 0),
            "recommendation_events": int(_service_metric(services, "recommendations", "total_count") or 0),
        }

        db_service = _service_by_name(services, "db") or {}
        cache_service = _service_by_name(services, "cache") or {}
        uptime_seconds = max(int((now - PROCESS_STARTED_AT).total_seconds()), 0)

        return Response(
            {
                "ok": overall_status == "ok",
                "overall_status": overall_status,
                "generated_at": now.isoformat(),
                "server_time": now.isoformat(),
                "summary": summary,
                "app": {
                    "pid": os.getpid(),
                    "db_name": connection.settings_dict.get("NAME", "default"),
                    "uptime_seconds": uptime_seconds,
                    "uptime_human": _format_duration(uptime_seconds),
                },
                "db": {
                    "ok": db_service.get("status") == "ok",
                    "error": db_service.get("error"),
                    "latency_ms": db_service.get("latency_ms"),
                },
                "cache": {
                    "ok": cache_service.get("status") == "ok",
                    "error": cache_service.get("error"),
                    "latency_ms": cache_service.get("latency_ms"),
                },
                "counts": counts,
                "services": services,
            }
        )


def _safe_iso(value):
    return value.isoformat() if value else None


def _format_duration(total_seconds):
    seconds = max(int(total_seconds or 0), 0)
    days, rem = divmod(seconds, 86400)
    hours, rem = divmod(rem, 3600)
    minutes, secs = divmod(rem, 60)
    parts = []
    if days:
        parts.append(f"{days}d")
    if hours:
        parts.append(f"{hours}h")
    if minutes:
        parts.append(f"{minutes}m")
    if secs or not parts:
        parts.append(f"{secs}s")
    return " ".join(parts[:3])


def _format_age(now, value):
    if not value:
        return "n/a"
    return f"{_format_duration((now - value).total_seconds())} ago"


def _format_money(amount):
    try:
        return f"{round(float(amount or 0), 2)} KZT"
    except (TypeError, ValueError):
        return "0 KZT"


def _build_service(*, name, label, status, latency_ms, detail, last_check, highlights, metrics, error=None):
    return {
        "name": name,
        "label": label,
        "status": status,
        "latency_ms": latency_ms,
        "detail": detail,
        "last_check": _safe_iso(last_check),
        "highlights": highlights,
        "metrics": metrics,
        "error": error,
    }


def _service_by_name(services, name):
    for service in services:
        if service.get("name") == name:
            return service
    return None


def _service_metric(services, name, key):
    service = _service_by_name(services, name) or {}
    metrics = service.get("metrics") or {}
    return metrics.get(key)


def _overall_health_status(services):
    if any(service.get("status") == "down" for service in services):
        return "down"
    if any(service.get("status") == "degraded" for service in services):
        return "degraded"
    return "ok"


def _db_health_service(now):
    started = perf_counter()
    try:
        with connection.cursor() as cur:
            cur.execute("SELECT 1;")
            cur.fetchone()
        latency_ms = round((perf_counter() - started) * 1000, 2)
        return _build_service(
            name="db",
            label="Database",
            status="ok",
            latency_ms=latency_ms,
            detail="Primary database responds to SELECT 1.",
            last_check=now,
            highlights=[
                {"label": "Latency", "value": f"{latency_ms} ms"},
                {"label": "Engine", "value": connection.vendor},
                {"label": "Database", "value": str(connection.settings_dict.get("NAME", "default"))},
            ],
            metrics={
                "backend": connection.vendor,
                "database": connection.settings_dict.get("NAME", "default"),
            },
        )
    except Exception as exc:
        latency_ms = round((perf_counter() - started) * 1000, 2)
        return _build_service(
            name="db",
            label="Database",
            status="down",
            latency_ms=latency_ms,
            detail="Database health check failed.",
            last_check=now,
            highlights=[
                {"label": "Latency", "value": f"{latency_ms} ms"},
                {"label": "Error", "value": str(exc)},
            ],
            metrics={},
            error=str(exc),
        )


def _cache_health_service(now):
    started = perf_counter()
    try:
        key = f"health:{now.timestamp()}"
        cache.set(key, "ok", timeout=10)
        cache_ok = cache.get(key) == "ok"
        latency_ms = round((perf_counter() - started) * 1000, 2)
        status = "ok" if cache_ok else "down"
        detail = "Cache round-trip completed." if cache_ok else "Cache returned unexpected value."
        error = None if cache_ok else "cache_round_trip_failed"
        return _build_service(
            name="cache",
            label="Cache",
            status=status,
            latency_ms=latency_ms,
            detail=detail,
            last_check=now,
            highlights=[
                {"label": "Latency", "value": f"{latency_ms} ms"},
                {"label": "Backend", "value": str(settings.CACHES.get("default", {}).get("BACKEND", "unknown"))},
                {"label": "Round-trip", "value": "ok" if cache_ok else "failed"},
            ],
            metrics={
                "backend": settings.CACHES.get("default", {}).get("BACKEND", "unknown"),
                "round_trip_ok": cache_ok,
            },
            error=error,
        )
    except Exception as exc:
        latency_ms = round((perf_counter() - started) * 1000, 2)
        return _build_service(
            name="cache",
            label="Cache",
            status="down",
            latency_ms=latency_ms,
            detail="Cache health check failed.",
            last_check=now,
            highlights=[
                {"label": "Latency", "value": f"{latency_ms} ms"},
                {"label": "Error", "value": str(exc)},
            ],
            metrics={},
            error=str(exc),
        )


def _transactions_health_service(now):
    started = perf_counter()
    try:
        since_24h = now - timedelta(hours=24)
        since_7d = now - timedelta(days=7)
        total_count = Transaction.objects.count()
        count_24h = Transaction.objects.filter(created_at__gte=since_24h).count()
        count_7d = Transaction.objects.filter(created_at__gte=since_7d).count()
        revenue_7d = Transaction.objects.filter(created_at__gte=since_7d).aggregate(s=Sum("total_amount"))["s"] or 0
        online_7d = Transaction.objects.filter(created_at__gte=since_7d, channel="online").count()
        offline_7d = Transaction.objects.filter(created_at__gte=since_7d, channel="offline").count()
        last_created_at = Transaction.objects.aggregate(last_created_at=Max("created_at"))["last_created_at"]
        latency_ms = round((perf_counter() - started) * 1000, 2)

        if total_count == 0:
            status = "degraded"
            detail = "No transactions recorded yet."
        elif last_created_at and last_created_at < now - timedelta(days=14):
            status = "degraded"
            detail = f"Last transaction was {_format_age(now, last_created_at)}."
        else:
            status = "ok"
            detail = f"{count_24h} transactions in the last 24h."

        return _build_service(
            name="transactions",
            label="Transactions",
            status=status,
            latency_ms=latency_ms,
            detail=detail,
            last_check=last_created_at or now,
            highlights=[
                {"label": "Total", "value": str(total_count)},
                {"label": "24h", "value": str(count_24h)},
                {"label": "7d revenue", "value": _format_money(revenue_7d)},
            ],
            metrics={
                "total_count": total_count,
                "count_24h": count_24h,
                "count_7d": count_7d,
                "online_7d": online_7d,
                "offline_7d": offline_7d,
                "revenue_7d": float(revenue_7d or 0),
                "last_transaction_at": _safe_iso(last_created_at),
            },
        )
    except Exception as exc:
        latency_ms = round((perf_counter() - started) * 1000, 2)
        return _build_service(
            name="transactions",
            label="Transactions",
            status="down",
            latency_ms=latency_ms,
            detail="Transaction metrics query failed.",
            last_check=now,
            highlights=[
                {"label": "Latency", "value": f"{latency_ms} ms"},
                {"label": "Error", "value": str(exc)},
            ],
            metrics={},
            error=str(exc),
        )


def _offers_health_service(now):
    started = perf_counter()
    try:
        since_24h = now - timedelta(hours=24)
        since_7d = now - timedelta(days=7)
        active_offers = Offer.objects.filter(is_active=True).count()
        active_campaigns = OfferAssignment.objects.filter(is_active=True).values("offer__campaign_id").distinct().count()
        total_assignments = OfferAssignment.objects.count()
        active_assignments = OfferAssignment.objects.filter(is_active=True, is_redeemed=False).count()
        total_events = OfferEvent.objects.count()
        events_24h = OfferEvent.objects.filter(created_at__gte=since_24h).count()
        redeemed_7d = OfferEvent.objects.filter(
            created_at__gte=since_7d,
            event_type=OfferEvent.Type.REDEEMED,
        ).count()
        last_event_at = OfferEvent.objects.aggregate(last_event_at=Max("created_at"))["last_event_at"]
        latency_ms = round((perf_counter() - started) * 1000, 2)

        if active_offers == 0 and total_assignments == 0:
            status = "degraded"
            detail = "No active offers are configured."
        elif total_assignments == 0:
            status = "degraded"
            detail = "Offers exist but nothing has been assigned yet."
        elif last_event_at and last_event_at < now - timedelta(days=14):
            status = "degraded"
            detail = f"Offer activity is stale: last event {_format_age(now, last_event_at)}."
        else:
            status = "ok"
            detail = f"{events_24h} offer events in the last 24h."

        return _build_service(
            name="offers",
            label="Offers",
            status=status,
            latency_ms=latency_ms,
            detail=detail,
            last_check=last_event_at or now,
            highlights=[
                {"label": "Active offers", "value": str(active_offers)},
                {"label": "Assignments", "value": str(total_assignments)},
                {"label": "Redeemed 7d", "value": str(redeemed_7d)},
            ],
            metrics={
                "active_offers": active_offers,
                "active_campaigns": active_campaigns,
                "total_assignments": total_assignments,
                "active_assignments": active_assignments,
                "total_events": total_events,
                "events_24h": events_24h,
                "redeemed_7d": redeemed_7d,
                "last_event_at": _safe_iso(last_event_at),
            },
        )
    except Exception as exc:
        latency_ms = round((perf_counter() - started) * 1000, 2)
        return _build_service(
            name="offers",
            label="Offers",
            status="down",
            latency_ms=latency_ms,
            detail="Offer metrics query failed.",
            last_check=now,
            highlights=[
                {"label": "Latency", "value": f"{latency_ms} ms"},
                {"label": "Error", "value": str(exc)},
            ],
            metrics={},
            error=str(exc),
        )


def _audit_health_service(now):
    started = perf_counter()
    try:
        since_24h = now - timedelta(hours=24)
        total_count = AuditEvent.objects.count()
        events_24h = AuditEvent.objects.filter(created_at__gte=since_24h).count()
        error_events_24h = AuditEvent.objects.filter(created_at__gte=since_24h, status_code__gte=500).count()
        last_event_at = AuditEvent.objects.aggregate(last_event_at=Max("created_at"))["last_event_at"]
        latency_ms = round((perf_counter() - started) * 1000, 2)

        if total_count == 0:
            status = "degraded"
            detail = "Audit log has no events yet."
        elif error_events_24h > 0:
            status = "degraded"
            detail = f"{error_events_24h} audit events with 5xx status in the last 24h."
        elif last_event_at and last_event_at < now - timedelta(days=14):
            status = "degraded"
            detail = f"Audit activity is stale: last event {_format_age(now, last_event_at)}."
        else:
            status = "ok"
            detail = f"{events_24h} audit events in the last 24h."

        return _build_service(
            name="audit",
            label="Audit",
            status=status,
            latency_ms=latency_ms,
            detail=detail,
            last_check=last_event_at or now,
            highlights=[
                {"label": "Total", "value": str(total_count)},
                {"label": "24h", "value": str(events_24h)},
                {"label": "5xx / 24h", "value": str(error_events_24h)},
            ],
            metrics={
                "total_count": total_count,
                "events_24h": events_24h,
                "error_events_24h": error_events_24h,
                "last_event_at": _safe_iso(last_event_at),
            },
        )
    except Exception as exc:
        latency_ms = round((perf_counter() - started) * 1000, 2)
        return _build_service(
            name="audit",
            label="Audit",
            status="down",
            latency_ms=latency_ms,
            detail="Audit metrics query failed.",
            last_check=now,
            highlights=[
                {"label": "Latency", "value": f"{latency_ms} ms"},
                {"label": "Error", "value": str(exc)},
            ],
            metrics={},
            error=str(exc),
        )


def _recommendations_health_service(now):
    started = perf_counter()
    try:
        since_24h = now - timedelta(hours=24)
        total_count = RecommendationEvent.objects.count()
        impressions_24h = RecommendationEvent.objects.filter(
            created_at__gte=since_24h,
            action=RecommendationEvent.Action.IMPRESSION,
        ).count()
        clicks_24h = RecommendationEvent.objects.filter(
            created_at__gte=since_24h,
            action=RecommendationEvent.Action.CLICK,
        ).count()
        purchases_24h = RecommendationEvent.objects.filter(
            created_at__gte=since_24h,
            action=RecommendationEvent.Action.PURCHASE_ATTRIBUTED,
        ).count()
        ctr_24h = round((clicks_24h / impressions_24h) * 100, 2) if impressions_24h else 0.0
        last_event_at = RecommendationEvent.objects.aggregate(last_event_at=Max("created_at"))["last_event_at"]
        latency_ms = round((perf_counter() - started) * 1000, 2)

        if total_count == 0:
            status = "degraded"
            detail = "No recommendation telemetry yet."
        elif last_event_at and last_event_at < now - timedelta(days=14):
            status = "degraded"
            detail = f"Recommendation telemetry is stale: last event {_format_age(now, last_event_at)}."
        else:
            status = "ok"
            detail = f"{impressions_24h} recommendation impressions in the last 24h."

        return _build_service(
            name="recommendations",
            label="Recommendations",
            status=status,
            latency_ms=latency_ms,
            detail=detail,
            last_check=last_event_at or now,
            highlights=[
                {"label": "Total", "value": str(total_count)},
                {"label": "Impr. 24h", "value": str(impressions_24h)},
                {"label": "CTR 24h", "value": f"{ctr_24h}%"},
            ],
            metrics={
                "total_count": total_count,
                "impressions_24h": impressions_24h,
                "clicks_24h": clicks_24h,
                "purchases_24h": purchases_24h,
                "ctr_24h": ctr_24h,
                "last_event_at": _safe_iso(last_event_at),
            },
        )
    except Exception as exc:
        latency_ms = round((perf_counter() - started) * 1000, 2)
        return _build_service(
            name="recommendations",
            label="Recommendations",
            status="down",
            latency_ms=latency_ms,
            detail="Recommendation metrics query failed.",
            last_check=now,
            highlights=[
                {"label": "Latency", "value": f"{latency_ms} ms"},
                {"label": "Error", "value": str(exc)},
            ],
            metrics={},
            error=str(exc),
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


def _overview_trend(*, tx7, tx30, recs7, recs30):
    trend = []
    if tx7 or recs7:
        trend.append(
            {
                "day": "7d",
                "ctr": round(float((recs7 or {}).get("ctr") or 0) * 100, 2),
                "cr": round(float((recs7 or {}).get("cr") or 0) * 100, 2),
                "users": int((tx7 or {}).get("unique_buyers") or 0),
            }
        )
    if tx30 or recs30:
        trend.append(
            {
                "day": "30d",
                "ctr": round(float((recs30 or {}).get("ctr") or 0) * 100, 2),
                "cr": round(float((recs30 or {}).get("cr") or 0) * 100, 2),
                "users": int((tx30 or {}).get("unique_buyers") or 0),
            }
        )
    return trend


def _overview_top_offers(*, events_kpis):
    out = []
    for index, row in enumerate((events_kpis or {}).get("by_campaign_30d") or []):
        out.append(
            {
                "id": str(row.get("campaign_name") or index + 1),
                "name": str(row.get("campaign_name") or f"Campaign {index + 1}"),
                "type": "campaign",
                "cr": float(row.get("redemption_rate_exposed") or 0),
                "exposed": int(row.get("exposed") or 0),
                "clicked": int(row.get("clicked") or 0),
                "redeemed": int(row.get("redeemed") or 0),
            }
        )
    return out[:5]


def _overview_kpis(*, tx7, tx30, recs7, recs30, offers7, offers30, retention):
    return {
        "7d": {
            "ctr": float((recs7 or {}).get("ctr") or 0),
            "cr": float((recs7 or {}).get("cr") or 0),
            "unique_buyers": int((tx7 or {}).get("unique_buyers") or 0),
            "promo_redemption": float((offers7 or {}).get("redemption_rate_exposed") or 0),
        },
        "30d": {
            "ctr": float((recs30 or {}).get("ctr") or 0),
            "cr": float((recs30 or {}).get("cr") or 0),
            "unique_buyers": int((tx30 or {}).get("unique_buyers") or 0),
            "promo_redemption": float((offers30 or {}).get("redemption_rate_exposed") or 0),
        },
        "90d": {
            "active_users": int((retention or {}).get("active_users_90d") or 0),
        },
    }


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
        retention = _repeat_purchase_block(now)
        events_kpis = offers_events_kpis()
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
                "events_kpis": events_kpis,
                "promo_efficiency_30d": offers_promo_efficiency_30d(),
            },
            "retention": retention,
            "recs": {
                "7d": recs_7d,
                "30d": recs_30d,
                "details_30d": recs_details,
                "experiments_30d": recs_details.get("by_experiment", {}),
            },
            "kpis": _overview_kpis(
                tx7=tx_7d,
                tx30=tx_30d,
                recs7=recs_7d,
                recs30=recs_30d,
                offers7=offers_7d,
                offers30=offers_30d,
                retention=retention,
            ),
            "trend": _overview_trend(tx7=tx_7d, tx30=tx_30d, recs7=recs_7d, recs30=recs_30d),
            "top_offers": _overview_top_offers(events_kpis=events_kpis),
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
