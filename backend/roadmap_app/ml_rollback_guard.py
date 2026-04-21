from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from typing import Any

from django.utils import timezone

from roadmap_app import runtime_config
from roadmap_app.models import RoadmapMLInvocation

DEFAULT_WINDOW_MINUTES = 15
DEFAULT_MAX_ERROR_RATE_PCT = 5.0
DEFAULT_MAX_FALLBACK_RATE_PCT = 30.0
DEFAULT_MAX_P95_LATENCY_MS = 500.0
DEFAULT_MIN_SAMPLE_SIZE = 50

GUARD_ACTOR = "rollback_guard"
ATTEMPTED_DECISIONS = {"model_used", "fallback"}


@dataclass
class GuardThresholds:
    max_error_rate_pct: float = DEFAULT_MAX_ERROR_RATE_PCT
    max_fallback_rate_pct: float = DEFAULT_MAX_FALLBACK_RATE_PCT
    max_p95_latency_ms: float = DEFAULT_MAX_P95_LATENCY_MS
    min_sample_size: int = DEFAULT_MIN_SAMPLE_SIZE


def _percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    data = sorted(values)
    n = len(data)
    if n == 1:
        return float(data[0])
    idx = int(round((pct / 100.0) * (n - 1)))
    idx = max(0, min(n - 1, idx))
    return float(data[idx])


def _aggregate_category(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    attempts = 0
    errors = 0
    fallbacks = 0
    latencies: list[float] = []
    for row in rows:
        decision = str(row.get("decision") or "")
        predict_error = str(row.get("predict_error") or "")
        predict_ms = row.get("predict_ms")
        has_latency = predict_ms is not None
        has_error = bool(predict_error)
        attempted = (
            decision in ATTEMPTED_DECISIONS or has_error or has_latency
        )
        if attempted:
            attempts += 1
        if has_error:
            errors += 1
        if decision == "fallback":
            fallbacks += 1
        if has_latency:
            try:
                latencies.append(float(predict_ms))
            except (TypeError, ValueError):
                pass
    error_rate = (errors / attempts * 100.0) if attempts else 0.0
    fallback_rate = (fallbacks / attempts * 100.0) if attempts else 0.0
    return {
        "total": total,
        "predict_attempts": attempts,
        "errors": errors,
        "fallbacks": fallbacks,
        "error_rate_pct": round(error_rate, 4),
        "fallback_rate_pct": round(fallback_rate, 4),
        "p95_latency_ms": _percentile(latencies, 95.0),
    }


def _evaluate_thresholds(
    metrics: dict[str, Any], thresholds: GuardThresholds
) -> dict[str, Any]:
    if metrics["predict_attempts"] < thresholds.min_sample_size:
        return {"breaches": [], "insufficient_sample": True}
    breaches: list[dict[str, Any]] = []
    if metrics["error_rate_pct"] > thresholds.max_error_rate_pct:
        breaches.append(
            {
                "metric": "error_rate_pct",
                "value": metrics["error_rate_pct"],
                "threshold": thresholds.max_error_rate_pct,
            }
        )
    if metrics["fallback_rate_pct"] > thresholds.max_fallback_rate_pct:
        breaches.append(
            {
                "metric": "fallback_rate_pct",
                "value": metrics["fallback_rate_pct"],
                "threshold": thresholds.max_fallback_rate_pct,
            }
        )
    p95 = metrics["p95_latency_ms"]
    if p95 is not None and p95 > thresholds.max_p95_latency_ms:
        breaches.append(
            {
                "metric": "p95_latency_ms",
                "value": p95,
                "threshold": thresholds.max_p95_latency_ms,
            }
        )
    return {"breaches": breaches, "insufficient_sample": False}


def evaluate_rollback_guard(
    *,
    window_minutes: int = DEFAULT_WINDOW_MINUTES,
    thresholds: GuardThresholds | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    thresholds = thresholds or GuardThresholds()
    now_dt = now or timezone.now()
    cutoff = now_dt - timedelta(minutes=int(window_minutes))
    rows = list(
        RoadmapMLInvocation.objects.filter(created_at__gte=cutoff).values(
            "category", "decision", "predict_ms", "predict_error"
        )
    )

    by_cat: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        cat = str(row.get("category") or "")
        by_cat.setdefault(cat, []).append(row)

    per_category: dict[str, dict[str, Any]] = {}
    any_breach = False
    for cat, cat_rows in by_cat.items():
        metrics = _aggregate_category(cat_rows)
        evaluation = _evaluate_thresholds(metrics, thresholds)
        per_category[cat] = {**metrics, **evaluation}
        if evaluation["breaches"]:
            any_breach = True

    return {
        "window_minutes": int(window_minutes),
        "cutoff_utc": cutoff.isoformat(),
        "now_utc": now_dt.isoformat(),
        "thresholds": asdict(thresholds),
        "per_category": per_category,
        "any_breach": any_breach,
        "frozen_before": runtime_config.is_runtime_ml_frozen(),
    }


def _breach_summary(per_category: dict[str, dict[str, Any]]) -> str:
    parts: list[str] = []
    for cat in sorted(per_category):
        for breach in per_category[cat].get("breaches", []):
            parts.append(
                f"{cat}:{breach['metric']}={breach['value']:.2f}>{breach['threshold']}"
            )
    return "; ".join(parts)


def enforce_rollback_guard(
    *,
    window_minutes: int = DEFAULT_WINDOW_MINUTES,
    thresholds: GuardThresholds | None = None,
    now: datetime | None = None,
    actor: str = GUARD_ACTOR,
) -> dict[str, Any]:
    report = evaluate_rollback_guard(
        window_minutes=window_minutes, thresholds=thresholds, now=now
    )
    if not report["any_breach"]:
        report["action_taken"] = "none"
        report["frozen_after"] = report["frozen_before"]
        return report
    if report["frozen_before"]:
        report["action_taken"] = "already_frozen"
        report["frozen_after"] = True
        return report
    note = _breach_summary(report["per_category"])[:256]
    runtime_config.set_value(
        runtime_config.FREEZE_KEY,
        "true",
        updated_by=actor,
        note=note,
    )
    report["action_taken"] = "freeze_set"
    report["frozen_after"] = True
    report["freeze_note"] = note
    return report
