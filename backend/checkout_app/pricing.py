from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP
from typing import Iterable, Any

from catalog.models import Product


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


def apply_offer_to_totals(
    *,
    offer_type: str,
    offer_value: Decimal,
    target: dict,
    lines: list[Line],
    points_rate: Decimal,
) -> dict[str, Any]:
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
    net_total = gross_total
    multiplier = Decimal("1")

    if offer_type == "discount":
        percent = offer_value
        discount_amount = d2(eligible_total * (percent / Decimal("100")))
        net_total = gross_total - discount_amount
        if net_total < 0:
            net_total = Decimal("0")
        net_total = d2(net_total)

    elif offer_type == "points_multiplier":
        multiplier = offer_value

    # базовые points от net_total (после скидки)
    base_points = int(round(float(net_total * points_rate)))
    est_points = base_points

    if offer_type == "points_multiplier" and multiplier != Decimal("1"):
        if scope == "cart":
            est_points = int(round(float(Decimal(base_points) * multiplier)))
        else:
            eligible_points = int(round(float(eligible_total * points_rate)))
            rest_total = gross_total - eligible_total
            if rest_total < 0:
                rest_total = Decimal("0")
            rest_points = int(round(float(rest_total * points_rate)))
            est_points = rest_points + int(round(float(Decimal(eligible_points) * multiplier)))

    return {
        "ok": True,
        "gross_total": str(gross_total),
        "eligible_total": str(eligible_total),
        "eligible_item_ids": eligible_item_ids,
        "discount_amount": str(d2(discount_amount)),
        "net_total": str(d2(net_total)),
        "points_multiplier": str(multiplier),
        "base_points": base_points,
        "estimated_points_earned": est_points,
    }
