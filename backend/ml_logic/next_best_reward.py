from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any


@dataclass
class RFM:
    recency_days: int
    frequency_90d: int
    monetary_90d: float


def compute_rfm(transactions: list[dict[str, Any]], now: datetime) -> RFM:
    # transactions: [{"created_at": datetime, "total_amount": float}, ...]
    if not transactions:
        return RFM(recency_days=9999, frequency_90d=0, monetary_90d=0.0)

    tx_sorted = sorted(transactions, key=lambda t: t["created_at"], reverse=True)
    last = tx_sorted[0]["created_at"]
    recency_days = max(0, (now - last).days)

    window_start = now - timedelta(days=90)
    in_90d = [t for t in transactions if t["created_at"] >= window_start]

    frequency = len(in_90d)
    monetary = float(sum(float(t["total_amount"]) for t in in_90d))

    return RFM(recency_days=recency_days, frequency_90d=frequency, monetary_90d=monetary)


def segment(rfm: RFM) -> str:
    # Простая сегментация MVP
    if rfm.frequency_90d <= 1:
        return "new_or_rare"
    if rfm.recency_days >= 45:
        return "at_risk"
    if rfm.frequency_90d >= 6 or rfm.monetary_90d >= 200:
        return "vip"
    return "active"


def pick_next_offer(
    rfm: RFM,
    segment_name: str,
    offers: list[dict[str, Any]],
    last_assignment_days_ago: int | None,
    budget_left: float,
    context_steps: list[str] | None = None) -> dict[str, Any] | None:
    # Табличные uplift веса по сегментам (MVP)
    uplift = {
        "at_risk": {"points_multiplier": 1.1, "discount": 1.0, "gift": 0.9},
        "vip": {"gift": 1.1, "points_multiplier": 1.0, "discount": 0.7},
        "new_or_rare": {"discount": 1.1, "points_multiplier": 0.9, "gift": 0.6},
        "active": {"points_multiplier": 1.0, "discount": 0.9, "gift": 0.8},
    }

    best = None
    best_score = -10**9

    for o in offers:
        if not o.get("is_active", True):
            continue

        allowed_steps = o.get("allowed_steps") or []
        if allowed_steps:
            if not context_steps:
                continue
            if not set(allowed_steps).intersection(set(context_steps)):
                continue

        # eligibility: min_total_spend_90d
        if float(rfm.monetary_90d) < float(o.get("min_total_spend_90d", 0)):
            continue

        # frequency cap
        cooldown = int(o.get("cooldown_days", 14))
        if last_assignment_days_ago is not None and last_assignment_days_ago < cooldown:
            continue

        cost = float(o.get("estimated_cost", 0))
        if cost > budget_left:
            continue

        otype = o["offer_type"]
        u = uplift.get(segment_name, {}).get(otype, 0.7)

        # simplistic profit proxy:
        # higher monetary/frequency => higher expected value
        base_value = (rfm.frequency_90d * 5.0) + (rfm.monetary_90d * 0.02)
        score = (u * base_value) - cost

        if score > best_score:
            best_score = score
            best = o

    if best is None:
        return None

    return {
        "offer_id": best["id"],
        "score": best_score,
        "reason": {
            "segment": segment_name,
            "context_steps": context_steps,
            "rfm": {
                "recency_days": rfm.recency_days,
                "frequency_90d": rfm.frequency_90d,
                "monetary_90d": rfm.monetary_90d,
            },
            "picked_because": "max(score) under eligibility + cooldown + budget constraints",
        },
    }
