from __future__ import annotations
from dataclasses import dataclass
from datetime import timedelta
from typing import Any

from django.conf import settings
from django.db.models import Count
from django.utils import timezone

from recs_analytics.models import RecommendationEvent
from recs_analytics.effectiveness import global_uplift


@dataclass
class RecStats:
    impressions: int = 0
    clicks: int = 0
    purchases: int = 0

    @property
    def is_fatigued(self) -> bool:
        if self.impressions < settings.RECS_FATIGUE_IMPRESSIONS_MIN:
            return False
        return self.clicks == 0 and self.purchases == 0


def stats_for_products(user, product_ids: list[int], *, now=None) -> dict[int, RecStats]:
    if not product_ids:
        return {}
    now = now or timezone.now()
    since = now - timedelta(days=getattr(settings, "RECS_FATIGUE_WINDOW_DAYS", 14))

    qs = (
        RecommendationEvent.objects.filter(
            user=user,
            created_at__gte=since,
            product_id__in=product_ids,
            action__in=[
                RecommendationEvent.Action.IMPRESSION,
                RecommendationEvent.Action.CLICK,
                RecommendationEvent.Action.PURCHASE_ATTRIBUTED,
            ],
        )
        .values("product_id", "action")
        .annotate(c=Count("id"))
    )

    out: dict[int, RecStats] = {int(pid): RecStats() for pid in product_ids}
    for r in qs:
        pid = int(r["product_id"])
        a = r["action"]
        c = int(r["c"] or 0)
        st = out.setdefault(pid, RecStats())
        if a == RecommendationEvent.Action.IMPRESSION:
            st.impressions += c
        elif a == RecommendationEvent.Action.CLICK:
            st.clicks += c
        elif a == RecommendationEvent.Action.PURCHASE_ATTRIBUTED:
            st.purchases += c
    return out


def adjust_recs(user, recs: list[dict[str, Any]], *, now=None) -> list[dict[str, Any]]:
    """
    Mutates/returns recs with:
      - adjusted_score
      - components.fatigue / components.uplift
      - why additions
    """
    now = now or timezone.now()

    ids = []
    for r in recs:
        p = r.get("product") or {}
        if p.get("id"):
            ids.append(int(p["id"]))

    stmap = stats_for_products(user, ids, now=now)

    out = []
    for r in recs:
        p = r.get("product") or {}
        pid = p.get("id")
        if not pid:
            continue
        pid = int(pid)

        base = float(r.get("score") or 0.0)
        st = stmap.get(pid, RecStats())

        penalty = st.impressions * float(getattr(settings, "RECS_PENALTY_IMP", 0.05))
        uplift = 0.0
        if st.clicks > 0:
            uplift += float(getattr(settings, "RECS_UPLIFT_CLICK", 0.15))
        if st.purchases > 0:
            uplift += float(getattr(settings, "RECS_UPLIFT_PURCHASE", 0.5))

        g_adj, g_st = global_uplift(pid, now=now)
        adjusted = base + uplift - penalty + g_adj

        comps = dict(r.get("components") or {})
        comps["fatigue"] = {
            "window_days": int(getattr(settings, "RECS_FATIGUE_WINDOW_DAYS", 14)),
            "impressions": st.impressions,
            "clicks": st.clicks,
            "purchases": st.purchases,
            "penalty": round(penalty, 4),
            "uplift": round(uplift, 4),
            "fatigued": bool(st.is_fatigued),
        }

        comps["global"] = {
            "window_days": int(getattr(settings, "RECS_GLOBAL_WINDOW_DAYS", 30)),
            "impressions": g_st.impressions,
            "clicks": g_st.clicks,
            "purchases": g_st.purchases,
            "ctr": round(g_st.ctr, 4),
            "cr": round(g_st.cr, 4),
            "adjust": round(g_adj, 4),
        }

        why = list(r.get("why") or [])
        why.append(f"fatigue: imp={st.impressions}, clk={st.clicks}, pur={st.purchases}")
        why.append(f"global: cr={round(g_st.cr,4)} adj={round(g_adj,4)}")  # NEW
        if uplift:
            why.append("uplifted due to engagement")
        if penalty:
            why.append("penalized due to repeated impressions")

        r2 = dict(r)
        r2["components"] = comps
        r2["why"] = why[:8]
        r2["adjusted_score"] = round(adjusted, 4)
        out.append(r2)

    out.sort(key=lambda x: x.get("adjusted_score", -1e9), reverse=True)
    return out
