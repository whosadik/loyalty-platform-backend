from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta

from django.conf import settings
from django.core.cache import cache
from django.db.models import Count
from django.utils import timezone

from recs_analytics.models import RecommendationEvent


@dataclass
class GlobalPerf:
    impressions: int = 0
    clicks: int = 0
    purchases: int = 0

    @property
    def ctr(self) -> float:
        return (self.clicks / self.impressions) if self.impressions else 0.0

    @property
    def cr(self) -> float:
        return (self.purchases / self.impressions) if self.impressions else 0.0
    
def load_category_perf(*, now=None) -> dict[str, GlobalPerf]:
    now = now or timezone.now()
    d = int(getattr(settings, "RECS_GLOBAL_WINDOW_DAYS", 30))
    key = f"recs:global_cat_perf:v1:{d}"
    cached = cache.get(key)
    if cached is not None:
        return cached

    since = now - timedelta(days=d)

    qs = (
        RecommendationEvent.objects.filter(
            created_at__gte=since,
            action__in=[
                RecommendationEvent.Action.IMPRESSION,
                RecommendationEvent.Action.CLICK,
                RecommendationEvent.Action.PURCHASE_ATTRIBUTED,
            ],
        )
        .values("product__category", "action")
        .annotate(c=Count("id"))
    )

    out: dict[str, GlobalPerf] = {}
    for r in qs:
        cat = r["product__category"] or "unknown"
        act = r["action"]
        c = int(r["c"] or 0)
        st = out.setdefault(cat, GlobalPerf())
        if act == RecommendationEvent.Action.IMPRESSION:
            st.impressions += c
        elif act == RecommendationEvent.Action.CLICK:
            st.clicks += c
        elif act == RecommendationEvent.Action.PURCHASE_ATTRIBUTED:
            st.purchases += c

    cache.set(key, out, timeout=600)
    return out


def category_uplift(category: str, *, now=None) -> tuple[float, GlobalPerf]:
    """
    Возвращает adjustment по категории ([-penalty, +uplift], stats)
    """
    now = now or timezone.now()
    perf_map = load_category_perf(now=now)
    st = perf_map.get(category, GlobalPerf())

    min_imp = int(getattr(settings, "RECS_GLOBAL_MIN_IMP", 20))
    if st.impressions < min_imp:
        return 0.0, st

    baseline = float(getattr(settings, "RECS_GLOBAL_BASELINE_CR", 0.02))
    scale = float(getattr(settings, "RECS_GLOBAL_SCALE", 6.0))
    max_up = float(getattr(settings, "RECS_GLOBAL_MAX_UPLIFT", 0.35))
    max_pen = float(getattr(settings, "RECS_GLOBAL_MAX_PENALTY", 0.35))

    delta = (st.cr - baseline) * scale
    if delta >= 0:
        return min(delta, max_up), st
    return max(delta, -max_pen), st

def _key(now) -> str:
    d = int(getattr(settings, "RECS_GLOBAL_WINDOW_DAYS", 30))
    # ключ “на сейчас” → кэшируем на 10 минут, можно сделать почасовой
    return f"recs:global_perf:v1:{d}"


def load_global_perf(*, now=None) -> dict[int, GlobalPerf]:
    """
    Returns map: product_id -> GlobalPerf over window.
    Cached for 10 minutes.
    """
    now = now or timezone.now()
    key = _key(now)
    cached = cache.get(key)
    if cached is not None:
        return cached

    window_days = int(getattr(settings, "RECS_GLOBAL_WINDOW_DAYS", 30))
    since = now - timedelta(days=window_days)

    qs = (
        RecommendationEvent.objects.filter(
            created_at__gte=since,
            action__in=[
                RecommendationEvent.Action.IMPRESSION,
                RecommendationEvent.Action.CLICK,
                RecommendationEvent.Action.PURCHASE_ATTRIBUTED,
            ],
        )
        .values("product_id", "action")
        .annotate(c=Count("id"))
    )

    out: dict[int, GlobalPerf] = {}
    for r in qs:
        pid = int(r["product_id"])
        act = r["action"]
        c = int(r["c"] or 0)

        st = out.setdefault(pid, GlobalPerf())
        if act == RecommendationEvent.Action.IMPRESSION:
            st.impressions += c
        elif act == RecommendationEvent.Action.CLICK:
            st.clicks += c
        elif act == RecommendationEvent.Action.PURCHASE_ATTRIBUTED:
            st.purchases += c

    cache.set(key, out, timeout=600)
    return out


def global_uplift(product_id: int, *, now=None) -> tuple[float, GlobalPerf]:
    """
    Converts global conversion signal into [-penalty, +uplift] adjustment.
    """
    now = now or timezone.now()
    perf_map = load_global_perf(now=now)
    st = perf_map.get(int(product_id), GlobalPerf())

    min_imp = int(getattr(settings, "RECS_GLOBAL_MIN_IMP", 20))
    if st.impressions < min_imp:
        return 0.0, st

    baseline = float(getattr(settings, "RECS_GLOBAL_BASELINE_CR", 0.02))
    scale = float(getattr(settings, "RECS_GLOBAL_SCALE", 6.0))
    max_up = float(getattr(settings, "RECS_GLOBAL_MAX_UPLIFT", 0.35))
    max_pen = float(getattr(settings, "RECS_GLOBAL_MAX_PENALTY", 0.35))

    # deviation from baseline conversion
    delta = (st.cr - baseline) * scale

    if delta >= 0:
        return min(delta, max_up), st
    return max(delta, -max_pen), st
