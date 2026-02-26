from datetime import timedelta

from django.db.models import Count
from django.utils import timezone

from recs_analytics.models import RecommendationEvent


def _with_rates(raw: dict[str, int]) -> dict[str, float | int]:
    imp = int(raw.get("impression", 0) or 0)
    clk = int(raw.get("click", 0) or 0)
    pur = int(raw.get("purchase_attributed", 0) or 0)
    return {
        "impression": imp,
        "click": clk,
        "add_to_cart": int(raw.get("add_to_cart", 0) or 0),
        "purchase_attributed": pur,
        "ctr": round(clk / imp, 4) if imp else 0.0,
        "conversion": round(pur / imp, 4) if imp else 0.0,
    }


def recs_metrics_30d():
    now = timezone.now()
    since = now - timedelta(days=30)
    qs = RecommendationEvent.objects.filter(created_at__gte=since)

    rows_section = qs.values("page", "section_key", "action").annotate(c=Count("id"))
    agg_section: dict[str, dict[str, int]] = {}
    for r in rows_section:
        key = f'{r["page"]}:{r["section_key"] or "none"}'
        agg_section.setdefault(
            key,
            {"impression": 0, "click": 0, "add_to_cart": 0, "purchase_attributed": 0},
        )
        agg_section[key][r["action"]] = int(r["c"])
    by_section = {k: _with_rates(v) for k, v in agg_section.items()}

    rows_algo = qs.values("algo_mode", "action").annotate(c=Count("id"))
    agg_algo: dict[str, dict[str, int]] = {}
    for r in rows_algo:
        algo_key = str((r["algo_mode"] or "unknown")).strip() or "unknown"
        agg_algo.setdefault(
            algo_key,
            {"impression": 0, "click": 0, "add_to_cart": 0, "purchase_attributed": 0},
        )
        agg_algo[algo_key][r["action"]] = int(r["c"])
    by_algo = {k: _with_rates(v) for k, v in agg_algo.items()}

    return {"window_days": 30, "by_section": by_section, "by_algo": by_algo}
