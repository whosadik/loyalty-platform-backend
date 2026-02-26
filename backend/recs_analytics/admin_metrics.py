from __future__ import annotations

from datetime import timedelta
from typing import Any

from django.db.models import Count
from django.utils import timezone

from recs_analytics.experiment import extract_experiment_context
from recs_analytics.models import RecommendationEvent


ACTION_KEYS = {"impression", "click", "add_to_cart", "purchase_attributed"}


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


def _new_stats() -> dict[str, int]:
    return {"impression": 0, "click": 0, "add_to_cart": 0, "purchase_attributed": 0}


def _normalize_algo_mode(raw: str | None) -> str:
    s = str(raw or "").strip().lower()
    if not s:
        return "unknown"
    if s.startswith("reranker"):
        return "reranker"
    if s.startswith("cooc") or s in {"cooccurrence", "fallback", "recommend"}:
        return "cooc"
    if s == "trending":
        return "trending"
    return s


def _aggregate_experiments(
    qs,
    *,
    experiment_id: str | None = None,
    variant: str | None = None,
) -> dict[str, dict[str, dict[str, float | int] | str]]:
    exp_filter = str(experiment_id or "").strip()
    variant_filter = str(variant or "").strip()

    by_experiment_raw: dict[str, dict[str, dict[str, int]]] = {}
    for row in qs.values("action", "context"):
        action = str(row["action"] or "")
        if action not in ACTION_KEYS:
            continue
        exp_ctx = extract_experiment_context(row.get("context") or {})
        exp_id = str(exp_ctx.get("experiment_id") or "").strip()
        if not exp_id:
            continue
        exp_variant = str(exp_ctx.get("experiment_variant") or "unknown").strip() or "unknown"
        if exp_filter and exp_id != exp_filter:
            continue
        if variant_filter and exp_variant != variant_filter:
            continue
        by_experiment_raw.setdefault(exp_id, {})
        by_experiment_raw[exp_id].setdefault(exp_variant, _new_stats())
        by_experiment_raw[exp_id][exp_variant][action] += 1

    by_experiment: dict[str, dict[str, dict[str, float | int] | str]] = {}
    for exp_id, variants in by_experiment_raw.items():
        totals = _new_stats()
        variants_payload: dict[str, dict[str, float | int]] = {}
        for exp_variant, stats in variants.items():
            variants_payload[exp_variant] = _with_rates(stats)
            for action, cnt in stats.items():
                totals[action] += int(cnt or 0)

        by_experiment[exp_id] = {
            "experiment_id": exp_id,
            "totals": _with_rates(totals),
            "variants": variants_payload,
        }
    return by_experiment


def recs_experiments_metrics(
    *,
    days: int = 30,
    experiment_id: str | None = None,
    variant: str | None = None,
) -> dict[str, Any]:
    window_days = max(1, min(365, int(days or 30)))
    now = timezone.now()
    since = now - timedelta(days=window_days)
    qs = RecommendationEvent.objects.filter(created_at__gte=since)

    by_experiment = _aggregate_experiments(
        qs,
        experiment_id=experiment_id,
        variant=variant,
    )

    summary_stats = _new_stats()
    for exp_payload in by_experiment.values():
        totals = exp_payload.get("totals") or {}
        for action in ACTION_KEYS:
            summary_stats[action] += int(totals.get(action, 0) or 0)

    summary = _with_rates(summary_stats)
    summary["experiments_count"] = len(by_experiment)

    return {
        "window_days": window_days,
        "filters": {
            "experiment_id": str(experiment_id or "").strip() or None,
            "variant": str(variant or "").strip() or None,
        },
        "summary": summary,
        "experiments": by_experiment,
    }


def recs_metrics_30d():
    now = timezone.now()
    since = now - timedelta(days=30)
    qs = RecommendationEvent.objects.filter(created_at__gte=since)

    rows_section = qs.values("page", "section_key", "action").annotate(c=Count("id"))
    agg_section: dict[str, dict[str, int]] = {}
    for r in rows_section:
        key = f'{r["page"]}:{r["section_key"] or "none"}'
        agg_section.setdefault(key, _new_stats())
        agg_section[key][r["action"]] = int(r["c"])
    by_section = {k: _with_rates(v) for k, v in agg_section.items()}

    rows_algo = qs.values("algo_mode", "action").annotate(c=Count("id"))
    agg_algo: dict[str, dict[str, int]] = {}
    for r in rows_algo:
        algo_key = _normalize_algo_mode(r["algo_mode"])
        agg_algo.setdefault(algo_key, _new_stats())
        agg_algo[algo_key][r["action"]] = int(r["c"])
    by_algo = {k: _with_rates(v) for k, v in agg_algo.items()}

    by_experiment = _aggregate_experiments(qs)

    return {
        "window_days": 30,
        "by_section": by_section,
        "by_algo": by_algo,
        "by_experiment": by_experiment,
    }
