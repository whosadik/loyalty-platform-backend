from datetime import timedelta
from django.utils import timezone
from django.db.models import Count

from recs_analytics.models import RecommendationEvent


def recs_metrics_30d():
    now = timezone.now()
    since = now - timedelta(days=30)

    qs = RecommendationEvent.objects.filter(created_at__gte=since)

    rows = qs.values("page", "section_key", "action").annotate(c=Count("id"))
    agg = {}
    for r in rows:
        key = f'{r["page"]}:{r["section_key"] or "none"}'
        agg.setdefault(key, {"impression": 0, "click": 0, "add_to_cart": 0, "purchase_attributed": 0})
        agg[key][r["action"]] = int(r["c"])

    # посчитаем CTR/CR
    out = {}
    for k, v in agg.items():
        imp = v["impression"] or 0
        clk = v["click"] or 0
        pur = v["purchase_attributed"] or 0
        out[k] = {
            **v,
            "ctr": round(clk / imp, 4) if imp else 0.0,
            "conversion": round(pur / imp, 4) if imp else 0.0,
        }

    return {"window_days": 30, "by_section": out}
