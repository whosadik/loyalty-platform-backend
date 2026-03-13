from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Iterable

from catalog.models import Product
from loyalty.points import clamp_redeem_points, get_effective_points_rate


@dataclass
class Line:
    product: Product
    quantity: int
    unit_price: Decimal


def d2(x: Decimal) -> Decimal:
    return x.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)


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


def _split_redeemed_amount(
    *,
    payable_total_before_points: Decimal,
    eligible_total_before_points: Decimal,
    points_redeemed: int,
) -> tuple[Decimal, Decimal]:
    total_before_points = d2(max(payable_total_before_points, Decimal("0")))
    eligible_before_points = d2(
        min(max(eligible_total_before_points, Decimal("0")), total_before_points)
    )

    if total_before_points <= 0 or points_redeemed <= 0:
        return eligible_before_points, d2(total_before_points - eligible_before_points)

    redeem_amount = Decimal(points_redeemed)
    rest_before_points = d2(total_before_points - eligible_before_points)

    if eligible_before_points <= 0:
        eligible_redeemed = Decimal("0")
    elif eligible_before_points >= total_before_points:
        eligible_redeemed = redeem_amount
    else:
        eligible_share = eligible_before_points / total_before_points
        eligible_redeemed = d2(redeem_amount * eligible_share)

    eligible_redeemed = min(eligible_redeemed, eligible_before_points, redeem_amount)
    rest_redeemed = min(redeem_amount - eligible_redeemed, rest_before_points)

    eligible_payable = d2(max(eligible_before_points - eligible_redeemed, Decimal("0")))
    rest_payable = d2(max(rest_before_points - rest_redeemed, Decimal("0")))
    return eligible_payable, rest_payable


def apply_offer_to_totals(
    *,
    offer_type: str,
    offer_value: Decimal,
    target: dict,
    lines: list[Line],
    points_rate: Decimal,
    redeem_points: int = 0,
) -> dict[str, Any]:
    normalized_points_rate = get_effective_points_rate(points_rate)
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

    points_redeemed = clamp_redeem_points(redeem_points, payable_total_before_points)
    eligible_payable_total, rest_payable_total = _split_redeemed_amount(
        payable_total_before_points=payable_total_before_points,
        eligible_total_before_points=eligible_total_after_offer,
        points_redeemed=points_redeemed,
    )
    payable_total = d2(eligible_payable_total + rest_payable_total)

    # Points are earned from the amount the customer actually pays after all deductions.
    base_points = int(round(float(payable_total * normalized_points_rate)))
    est_points = base_points

    if offer_type == "points_multiplier" and multiplier != Decimal("1"):
        if scope == "cart":
            est_points = int(round(float(Decimal(base_points) * multiplier)))
        else:
            eligible_points = int(round(float(eligible_payable_total * normalized_points_rate)))
            rest_points = int(round(float(rest_payable_total * normalized_points_rate)))
            est_points = rest_points + int(round(float(Decimal(eligible_points) * multiplier)))

    return {
        "ok": True,
        "gross_total": str(gross_total),
        "eligible_total": str(eligible_total),
        "eligible_item_ids": eligible_item_ids,
        "discount_amount": str(d2(discount_amount)),
        "net_total_before_points": str(d2(payable_total_before_points)),
        "net_total": str(payable_total),
        "points_redeemed": points_redeemed,
        "points_multiplier": str(multiplier),
        "base_points": base_points,
        "estimated_points_earned": est_points,
        "points_rate": str(normalized_points_rate),
    }
