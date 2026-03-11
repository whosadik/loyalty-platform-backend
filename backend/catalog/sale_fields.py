from __future__ import annotations

from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from typing import Any, Mapping


SALE_PRICE_KEYS = (
    "original_price",
    "old_price",
    "price_old",
    "rrp",
    "compare_at_price",
)
SALE_DISCOUNT_KEYS = (
    "discount",
    "discount_percent",
    "sale_percent",
)

_TWO_PLACES = Decimal("0.01")
_ONE = Decimal("1")
_ONE_HUNDRED = Decimal("100")


def _to_decimal(value: Any) -> Decimal | None:
    if value is None:
        return None
    if isinstance(value, Decimal):
        return value
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return Decimal(str(value))
    if isinstance(value, str):
        cleaned = value.strip().replace(" ", "").replace(",", ".")
        if not cleaned:
            return None
        try:
            return Decimal(cleaned)
        except InvalidOperation:
            return None
    return None


def _as_mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, dict) else {}


def _pick_decimal(mapping: Mapping[str, Any], keys: tuple[str, ...]) -> Decimal | None:
    for key in keys:
        parsed = _to_decimal(mapping.get(key))
        if parsed is not None:
            return parsed
    return None


def _price_components(product: Any) -> tuple[Decimal | None, Decimal | None, Decimal | None]:
    raw_meta = _as_mapping(getattr(product, "raw_meta", {}))
    attrs = _as_mapping(getattr(product, "attrs", {}))
    price = _to_decimal(getattr(product, "price", None))
    original_price = _pick_decimal(raw_meta, SALE_PRICE_KEYS) or _pick_decimal(attrs, SALE_PRICE_KEYS)
    discount = _pick_decimal(raw_meta, SALE_DISCOUNT_KEYS) or _pick_decimal(attrs, SALE_DISCOUNT_KEYS)
    return price, original_price, discount


def get_product_discount_percent(product: Any) -> int | None:
    price, original_price, explicit_discount = _price_components(product)

    if explicit_discount is not None and explicit_discount > 0:
        return min(99, int(explicit_discount.quantize(Decimal("1"), rounding=ROUND_HALF_UP)))

    if price is None or original_price is None or original_price <= 0 or original_price <= price:
        return None

    discount = ((original_price - price) / original_price * _ONE_HUNDRED).quantize(
        Decimal("1"),
        rounding=ROUND_HALF_UP,
    )
    return max(1, int(discount))


def get_product_original_price(product: Any) -> Decimal | None:
    price, original_price, explicit_discount = _price_components(product)

    if original_price is not None:
        original_price = original_price.quantize(_TWO_PLACES, rounding=ROUND_HALF_UP)
        if price is None or original_price > price:
            return original_price

    if price is None or explicit_discount is None or explicit_discount <= 0 or explicit_discount >= _ONE_HUNDRED:
        return None

    multiplier = _ONE - (explicit_discount / _ONE_HUNDRED)
    if multiplier <= 0:
        return None

    derived = (price / multiplier).quantize(_TWO_PLACES, rounding=ROUND_HALF_UP)
    return derived if derived > price else None


def product_has_discount(product: Any) -> bool:
    return (get_product_discount_percent(product) or 0) > 0
