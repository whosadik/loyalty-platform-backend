from __future__ import annotations

from collections import Counter
from datetime import datetime, timedelta
from typing import Any, Iterable

from django.utils import timezone

from roadmap_app.models import RoadmapMLInvocation

DEFAULT_WINDOW_MINUTES = 1440
DEFAULT_TOP_DIVERGENCES = 5

_INVOCATION_FIELDS = (
    "category",
    "decision",
    "fallback_reason",
    "ml_mode",
    "rollout_selected",
    "active_top_product_type",
    "shadow_top_product_type",
    "planned_target_product_type",
)


def _norm(value: Any) -> str:
    return str(value or "").strip().lower()


def _agreement(pairs: Iterable[tuple[str, str]]) -> dict[str, Any]:
    compared = 0
    matches = 0
    divergences: Counter[tuple[str, str]] = Counter()
    for left, right in pairs:
        left_n = _norm(left)
        right_n = _norm(right)
        if not left_n or not right_n:
            continue
        compared += 1
        if left_n == right_n:
            matches += 1
        else:
            divergences[(left_n, right_n)] += 1
    return {
        "compared": compared,
        "matches": matches,
        "agreement_pct": round((matches / compared * 100.0) if compared else 0.0, 4),
        "_divergences": divergences,
    }


def _top_divergences(
    divergences: Counter[tuple[str, str]],
    *,
    left_label: str,
    right_label: str,
    limit: int,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for (left, right), count in divergences.most_common(limit):
        out.append({left_label: left, right_label: right, "count": int(count)})
    return out


def _aggregate_category(rows: list[dict[str, Any]], *, top_divergences: int) -> dict[str, Any]:
    decision_counts: Counter[str] = Counter()
    fallback_reason_counts: Counter[str] = Counter()
    ml_mode_counts: Counter[str] = Counter()
    rollout_selected = 0

    served_vs_active: list[tuple[str, str]] = []
    served_vs_shadow: list[tuple[str, str]] = []
    active_vs_shadow: list[tuple[str, str]] = []

    for row in rows:
        decision_counts[_norm(row.get("decision")) or "(empty)"] += 1
        fb = _norm(row.get("fallback_reason"))
        if fb:
            fallback_reason_counts[fb] += 1
        mode = _norm(row.get("ml_mode"))
        if mode:
            ml_mode_counts[mode] += 1
        if bool(row.get("rollout_selected")):
            rollout_selected += 1

        served = row.get("planned_target_product_type")
        active = row.get("active_top_product_type")
        shadow = row.get("shadow_top_product_type")
        served_vs_active.append((served, active))
        served_vs_shadow.append((served, shadow))
        active_vs_shadow.append((active, shadow))

    sva = _agreement(served_vs_active)
    svs = _agreement(served_vs_shadow)
    avs = _agreement(active_vs_shadow)

    return {
        "total": len(rows),
        "rollout_selected_count": rollout_selected,
        "decision_counts": dict(decision_counts),
        "fallback_reason_counts": dict(fallback_reason_counts),
        "ml_mode_counts": dict(ml_mode_counts),
        "agreement": {
            "served_vs_active": {k: v for k, v in sva.items() if k != "_divergences"},
            "served_vs_shadow": {k: v for k, v in svs.items() if k != "_divergences"},
            "active_vs_shadow": {k: v for k, v in avs.items() if k != "_divergences"},
        },
        "top_divergences": {
            "served_vs_active": _top_divergences(
                sva["_divergences"], left_label="served", right_label="active", limit=top_divergences
            ),
            "served_vs_shadow": _top_divergences(
                svs["_divergences"], left_label="served", right_label="shadow", limit=top_divergences
            ),
            "active_vs_shadow": _top_divergences(
                avs["_divergences"], left_label="active", right_label="shadow", limit=top_divergences
            ),
        },
    }


def build_control_vs_ml_diff_report(
    *,
    window_minutes: int = DEFAULT_WINDOW_MINUTES,
    categories: list[str] | None = None,
    now: datetime | None = None,
    top_divergences: int = DEFAULT_TOP_DIVERGENCES,
) -> dict[str, Any]:
    now_dt = now or timezone.now()
    cutoff = now_dt - timedelta(minutes=int(window_minutes))
    qs = RoadmapMLInvocation.objects.filter(created_at__gte=cutoff)
    if categories:
        normalized = [_norm(c) for c in categories if _norm(c)]
        qs = qs.filter(category__in=normalized)
    rows = list(qs.values(*_INVOCATION_FIELDS))

    by_cat: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_cat.setdefault(_norm(row.get("category")), []).append(row)

    per_category: dict[str, dict[str, Any]] = {
        cat: _aggregate_category(cat_rows, top_divergences=top_divergences)
        for cat, cat_rows in by_cat.items()
    }

    return {
        "window_minutes": int(window_minutes),
        "cutoff_utc": cutoff.isoformat(),
        "now_utc": now_dt.isoformat(),
        "total_invocations": len(rows),
        "category_filter": [_norm(c) for c in (categories or []) if _norm(c)],
        "per_category": per_category,
    }
