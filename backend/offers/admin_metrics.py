from datetime import timedelta
from decimal import Decimal

from django.utils import timezone
from django.db.models import Count, Q, Sum

from offers.models import OfferAssignment, CampaignBudget, OfferEvent
from transactions.models import Transaction

def offers_metrics_30d():
    now = timezone.now()
    since = now - timedelta(days=30)

    qs = OfferAssignment.objects.filter(assigned_at__gte=since)

    total = qs.count()
    redeemed = qs.filter(is_redeemed=True).count()
    redemption_rate = (redeemed / total) if total else 0.0

    # picked_via distribution (from target or reason)
    picked = {"bundle": 0, "post_purchase_rules": 0, "fallback": 0, "unknown": 0}
    bundle_mode = {"cooccurrence": 0, "fallback": 0, "unknown": 0}
    cat_dist = {}

    for a in qs.only("id", "target", "reason"):
        t = a.target or {}
        pv_raw = t.get("picked_via") or (a.reason or {}).get("picked_via") or ""
        pv = pv_raw.lower()
        if pv == "bundle":
            pv = "bundle"
        elif pv.startswith("post_purchase_rules"):
            pv = "post_purchase_rules"
        elif pv:
            pv = "fallback"
        else:
            pv = "unknown"
        picked[pv] += 1

        if t.get("picked_via") == "bundle":
            bm = t.get("bundle_mode") or "unknown"
            if bm not in bundle_mode:
                bm = "unknown"
            bundle_mode[bm] += 1

        cat = t.get("category")
        if cat:
            cat_dist[cat] = cat_dist.get(cat, 0) + 1

    return {
        "assignments_30d": total,
        "redemptions_30d": redeemed,
        "redemption_rate_30d": round(redemption_rate, 4),
        "picked_via_distribution_30d": picked,
        "bundle_mode_distribution_30d": bundle_mode,
        "offer_target_category_distribution_30d": cat_dist,
    }

def campaigns_metrics_30d():
    now = timezone.now()
    since = now - timedelta(days=30)

    # assignments per campaign (через offer__campaign)
    rows = (
        OfferAssignment.objects.filter(assigned_at__gte=since)
        .values("offer__campaign__name")
        .annotate(
            assignments=Count("id"),
            redemptions=Count("id", filter=Q(is_redeemed=True)),
        )
        .order_by("-assignments")
    )

    out = []
    for r in rows:
        name = r["offer__campaign__name"] or "none"
        a = int(r["assignments"] or 0)
        red = int(r["redemptions"] or 0)
        out.append({
            "campaign": name,
            "assignments_30d": a,
            "redemptions_30d": red,
            "redemption_rate": round(red / a, 4) if a else 0.0,
        })

    # current weekly spend/left
    current = []
    for c in CampaignBudget.objects.all().order_by("priority", "name"):
        left = float(Decimal(str(c.weekly_limit)) - Decimal(str(c.weekly_spent)))
        current.append({
            "campaign": c.name,
            "priority": c.priority,
            "is_active": bool(c.is_active),
            "weekly_limit": float(c.weekly_limit),
            "weekly_spent": float(c.weekly_spent),
            "weekly_left": left,
        })

    return {"window_days": 30, "current_week": current, "last_30d": out}


def offers_events_kpis():
    now = timezone.now()

    def counts(days: int):
        since = now - timedelta(days=days)
        exposed = OfferEvent.objects.filter(
            created_at__gte=since,
            event_type=OfferEvent.Type.EXPOSED,
        ).count()
        clicked = OfferEvent.objects.filter(
            created_at__gte=since,
            event_type=OfferEvent.Type.CLICKED,
        ).count()
        redeemed = OfferEvent.objects.filter(
            created_at__gte=since,
            event_type=OfferEvent.Type.REDEEMED,
        ).count()
        redemption_rate = (redeemed / exposed) if exposed else 0.0
        ctr = (clicked / exposed) if exposed else 0.0
        return exposed, clicked, redeemed, redemption_rate, ctr

    e7, c7, r7, red_rate7, ctr7 = counts(7)
    e30, c30, r30, red_rate30, ctr30 = counts(30)

    # campaign breakdown for last 30 days
    since30 = now - timedelta(days=30)
    agg = {}
    rows = (
        OfferEvent.objects.filter(
            created_at__gte=since30,
            event_type__in=[
                OfferEvent.Type.EXPOSED,
                OfferEvent.Type.CLICKED,
                OfferEvent.Type.REDEEMED,
            ],
        )
        .values("campaign_name", "event_type")
        .annotate(cnt=Count("id"))
        .order_by("campaign_name")
    )
    for r in rows:
        name = r["campaign_name"] or "none"
        bucket = agg.setdefault(name, {"campaign_name": name, "exposed": 0, "clicked": 0, "redeemed": 0})
        et = r["event_type"]
        if et == OfferEvent.Type.EXPOSED:
            bucket["exposed"] += int(r["cnt"])
        elif et == OfferEvent.Type.CLICKED:
            bucket["clicked"] += int(r["cnt"])
        elif et == OfferEvent.Type.REDEEMED:
            bucket["redeemed"] += int(r["cnt"])
    by_campaign_30d = []
    for item in agg.values():
        exposed = item["exposed"]
        clicked = item["clicked"]
        redeemed = item["redeemed"]
        by_campaign_30d.append(
            {
                **item,
                "ctr_clicks_exposed": round((clicked / exposed), 4) if exposed else 0.0,
                "redemption_rate_exposed": round((redeemed / exposed), 4) if exposed else 0.0,
            }
        )
    by_campaign_30d.sort(key=lambda x: (-x["exposed"], x["campaign_name"]))

    return {
        "exposed_7d": int(e7),
        "clicked_7d": int(c7),
        "redeemed_7d": int(r7),
        "ctr_clicks_exposed_7d": round(ctr7, 4),
        "redemption_rate_exposed_7d": round(red_rate7, 4),
        "exposed_30d": int(e30),
        "clicked_30d": int(c30),
        "redeemed_30d": int(r30),
        "ctr_clicks_exposed_30d": round(ctr30, 4),
        "redemption_rate_exposed_30d": round(red_rate30, 4),
        "by_campaign_30d": by_campaign_30d,
    }


def offers_promo_efficiency_30d():
    now = timezone.now()
    since = now - timedelta(days=30)

    redeemed_assignment_ids = list(
        OfferEvent.objects.filter(
            created_at__gte=since,
            event_type=OfferEvent.Type.REDEEMED,
        ).values_list("assignment_id", flat=True)
    )
    if not redeemed_assignment_ids:
        return {
            "window_days": 30,
            "redeemed_count": 0,
            "estimated_cost_total": 0.0,
            "redeemed_revenue_total": 0.0,
            "promo_efficiency": 0.0,
            "by_campaign_30d": [],
        }

    assignments = list(
        OfferAssignment.objects.filter(id__in=redeemed_assignment_ids)
        .select_related("offer", "offer__campaign")
        .only("id", "redeemed_transaction_id", "offer__estimated_cost", "offer__campaign__name")
    )

    txn_ids = [a.redeemed_transaction_id for a in assignments if a.redeemed_transaction_id]
    txn_sum = (
        Transaction.objects.filter(id__in=txn_ids).aggregate(s=Sum("total_amount"))["s"]
        or Decimal("0")
    )
    txn_map = {
        tid: total
        for tid, total in Transaction.objects.filter(id__in=txn_ids).values_list("id", "total_amount")
    }

    est_total = sum((Decimal(str(a.offer.estimated_cost or 0)) for a in assignments), Decimal("0"))
    efficiency = (txn_sum / est_total) if est_total > 0 else Decimal("0")

    by_campaign = {}
    for a in assignments:
        campaign_name = getattr(getattr(a.offer, "campaign", None), "name", None) or "none"
        bucket = by_campaign.setdefault(
            campaign_name,
            {
                "campaign_name": campaign_name,
                "redeemed_count": 0,
                "estimated_cost_total": Decimal("0"),
                "redeemed_revenue_total": Decimal("0"),
            },
        )
        bucket["redeemed_count"] += 1
        bucket["estimated_cost_total"] += Decimal(str(a.offer.estimated_cost or 0))
        if a.redeemed_transaction_id:
            bucket["redeemed_revenue_total"] += Decimal(str(txn_map.get(a.redeemed_transaction_id) or 0))

    by_campaign_30d = []
    for item in by_campaign.values():
        cost = item["estimated_cost_total"]
        revenue = item["redeemed_revenue_total"]
        by_campaign_30d.append(
            {
                "campaign_name": item["campaign_name"],
                "redeemed_count": int(item["redeemed_count"]),
                "estimated_cost_total": float(cost),
                "redeemed_revenue_total": float(revenue),
                "promo_efficiency": round(float(revenue / cost), 4) if cost > 0 else 0.0,
            }
        )
    by_campaign_30d.sort(key=lambda x: (-x["redeemed_count"], x["campaign_name"]))

    return {
        "window_days": 30,
        "redeemed_count": int(len(assignments)),
        "estimated_cost_total": float(est_total),
        "redeemed_revenue_total": float(txn_sum),
        "promo_efficiency": round(float(efficiency), 4) if est_total > 0 else 0.0,
        "by_campaign_30d": by_campaign_30d,
    }
