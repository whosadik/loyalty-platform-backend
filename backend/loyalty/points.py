from __future__ import annotations

from decimal import Decimal, InvalidOperation, ROUND_FLOOR, ROUND_HALF_UP


DEFAULT_POINTS_RATE = Decimal("0.01")
MAX_POINTS_RATE = Decimal("0.10")

TIER_POINTS_MULTIPLIERS = {
    "bronze": Decimal("1.00"),
    "silver": Decimal("1.50"),
    "gold": Decimal("2.00"),
}


def get_effective_points_rate(raw_rate) -> Decimal:
    """
    User-facing surfaces across the project advertise roughly 1% back in points.
    Older environments stored rates like 1.00 / 1.50, which produced 100-150% accrual.
    Normalize those legacy values and cap the effective earn rate to 1% until
    differentiated tier economics are explicitly applied by tier multipliers.
    """

    try:
        rate = Decimal(str(raw_rate))
    except (InvalidOperation, TypeError, ValueError):
        return DEFAULT_POINTS_RATE

    if not rate.is_finite() or rate <= 0:
        return DEFAULT_POINTS_RATE

    if rate >= Decimal("1"):
        rate = (rate / Decimal("10")).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

    if rate > DEFAULT_POINTS_RATE:
        return DEFAULT_POINTS_RATE

    return rate.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)


def get_tier_points_multiplier(tier_name) -> Decimal:
    normalized = str(tier_name or "").strip().lower()
    return TIER_POINTS_MULTIPLIERS.get(normalized, Decimal("1.00"))


def cap_earned_points(points: int, paid_amount: Decimal) -> int:
    if paid_amount <= 0:
        return 0
    max_points = int((paid_amount * MAX_POINTS_RATE).quantize(Decimal("1"), rounding=ROUND_HALF_UP))
    return max(0, min(int(points or 0), max_points))


def clamp_redeem_points(requested_points, total_amount: Decimal) -> int:
    if total_amount <= 0:
        return 0

    try:
        requested = int(requested_points or 0)
    except (TypeError, ValueError):
        requested = 0

    if requested <= 0:
        return 0

    max_redeemable = int(total_amount.to_integral_value(rounding=ROUND_FLOOR))
    return max(0, min(requested, max_redeemable))
