from datetime import timedelta
from decimal import Decimal

from django.utils import timezone
from django.db.models import Count, Q

from offers.models import OfferAssignment, CampaignBudget

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