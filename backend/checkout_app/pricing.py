from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Iterable

from catalog.models import Product
from loyalty.points import cap_earned_points, clamp_redeem_points, get_effective_points_rate


@dataclass
class Line:
    product: Product
    quantity: int
    unit_price: Decimal


def d2(x: Decimal) -> Decimal:
    return x.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)


def _round_points(value: Decimal) -> int:
    return max(0, int(value.quantize(Decimal("1"), rounding=ROUND_HALF_UP)))


def _normalize_multiplier(value: Decimal) -> Decimal:
    try:
        multiplier = Decimal(str(value))
    except Exception:
        return Decimal("1")
    if not multiplier.is_finite() or multiplier <= 0:
        return Decimal("1")
    return multiplier


def _points_for_amount(amount: Decimal, points_rate: Decimal) -> int:
    return _round_points(max(amount, Decimal("0")) * points_rate)


def calc_gross(lines: Iterable[Line]) -> Decimal:
    total = Decimal("0")
    for ln in lines:
        total += ln.unit_price * ln.quantity
    return d2(total)


def is_eligible(product: Product, target: dict) -> bool:
    target = target or {"scope": "cart"}
    scope = target.get("scope", "cart")
    value = target.get("value")
    cat = target.get("category")

    if scope == "cart":
        return True
    if scope == "product_id":
        return int(product.id) == int(value)
    if scope == "category":
        return product.category == value
    if scope == "product_type":
        if cat and product.category != cat:
            return False
        if value:
            return product.product_type == value
        return True
    return True


def calc_eligible(lines: Iterable[Line], target: dict) -> tuple[Decimal, list[int]]:
    eligible_total = Decimal("0")
    eligible_ids: list[int] = []
    for ln in lines:
        if is_eligible(ln.product, target):
            eligible_total += ln.unit_price * ln.quantity
            eligible_ids.append(ln.product.id)
    return d2(eligible_total), eligible_ids


def _split_applied_amount(
    *,
    payable_total: Decimal,
    eligible_total: Decimal,
    applied_amount: Decimal,
) -> tuple[Decimal, Decimal]:
    normalized_total = d2(max(payable_total, Decimal("0")))
    normalized_eligible = d2(
        min(max(eligible_total, Decimal("0")), normalized_total)
    )

    if normalized_total <= 0 or applied_amount <= 0:
        return normalized_eligible, d2(normalized_total - normalized_eligible)

    normalized_applied = d2(applied_amount)
    rest_total = d2(normalized_total - normalized_eligible)

    if normalized_eligible <= 0:
        eligible_applied = Decimal("0")
    elif normalized_eligible >= normalized_total:
        eligible_applied = normalized_applied
    else:
        eligible_share = normalized_eligible / normalized_total
        eligible_applied = d2(normalized_applied * eligible_share)

    eligible_applied = min(eligible_applied, normalized_eligible, normalized_applied)
    rest_applied = min(normalized_applied - eligible_applied, rest_total)

    eligible_payable = d2(max(normalized_eligible - eligible_applied, Decimal("0")))
    rest_payable = d2(max(rest_total - rest_applied, Decimal("0")))
    return eligible_payable, rest_payable


def apply_offer_to_totals(
    *,
    offer_type: str,
    offer_value: Decimal,
    target: dict,
    lines: list[Line],
    points_rate: Decimal,
    tier_points_multiplier: Decimal = Decimal("1"),
    redeem_points: int = 0,
    gift_card_balance: Decimal = Decimal("0"),
) -> dict[str, Any]:
    normalized_points_rate = get_effective_points_rate(points_rate)
    normalized_tier_multiplier = _normalize_multiplier(tier_points_multiplier)
    gross_total = calc_gross(lines)
    eligible_total, eligible_item_ids = calc_eligible(lines, target)

    scope = (target or {}).get("scope", "cart")
    if scope != "cart" and eligible_total <= 0:
        return {
            "ok": False,
            "message": "No eligible items for this offer in provided items",
            "gross_total": str(gross_total),
            "eligible_total": str(eligible_total),
            "eligible_item_ids": eligible_item_ids,
        }

    discount_amount = Decimal("0")
    payable_total_before_points = gross_total
    multiplier = Decimal("1")

    if offer_type == "discount":
        percent = offer_value
        discount_amount = d2(eligible_total * (percent / Decimal("100")))
        payable_total_before_points = gross_total - discount_amount
        if payable_total_before_points < 0:
            payable_total_before_points = Decimal("0")
        payable_total_before_points = d2(payable_total_before_points)

    elif offer_type == "points_multiplier":
        multiplier = offer_value

    eligible_total_after_offer = eligible_total
    if offer_type == "discount":
        if scope == "cart":
            eligible_total_after_offer = payable_total_before_points
        else:
            eligible_total_after_offer = eligible_total - discount_amount
    eligible_total_after_offer = d2(max(eligible_total_after_offer, Decimal("0")))

    gift_card_applied_amount = d2(min(max(gift_card_balance, Decimal("0")), payable_total_before_points))
    eligible_after_gift_card, rest_after_gift_card = _split_applied_amount(
        payable_total=payable_total_before_points,
        eligible_total=eligible_total_after_offer,
        applied_amount=gift_card_applied_amount,
    )
    payable_total_after_gift_card = d2(eligible_after_gift_card + rest_after_gift_card)

    points_redeemed = clamp_redeem_points(redeem_points, payable_total_after_gift_card)
    eligible_payable_total, rest_payable_total = _split_applied_amount(
        payable_total=payable_total_after_gift_card,
        eligible_total=eligible_after_gift_card,
        applied_amount=Decimal(points_redeemed),
    )
    payable_total = d2(eligible_payable_total + rest_payable_total)

    # Points are earned from the amount the customer actually pays after all deductions.
    base_points = _points_for_amount(payable_total, normalized_points_rate)
    tier_adjusted_points = _round_points(Decimal(base_points) * normalized_tier_multiplier)
    est_points = tier_adjusted_points

    if offer_type == "points_multiplier" and multiplier != Decimal("1"):
        if scope == "cart":
            est_points = _round_points(Decimal(tier_adjusted_points) * multiplier)
        else:
            eligible_base_points = _points_for_amount(eligible_payable_total, normalized_points_rate)
            rest_base_points = _points_for_amount(rest_payable_total, normalized_points_rate)
            eligible_tier_points = _round_points(Decimal(eligible_base_points) * normalized_tier_multiplier)
            rest_tier_points = _round_points(Decimal(rest_base_points) * normalized_tier_multiplier)
            est_points = rest_tier_points + _round_points(Decimal(eligible_tier_points) * multiplier)

    est_points = cap_earned_points(est_points, payable_total)

    return {
        "ok": True,
        "gross_total": str(gross_total),
        "eligible_total": str(eligible_total),
        "eligible_item_ids": eligible_item_ids,
        "discount_amount": str(d2(discount_amount)),
        "net_total_before_points": str(d2(payable_total_after_gift_card)),
        "net_total": str(payable_total),
        "gift_card_applied_amount": str(gift_card_applied_amount),
        "points_redeemed": points_redeemed,
        "points_multiplier": str(multiplier),
        "tier_points_multiplier": str(normalized_tier_multiplier),
        "base_points": base_points,
        "tier_adjusted_points": tier_adjusted_points,
        "estimated_points_earned": est_points,
        "points_rate": str(normalized_points_rate),
    }
