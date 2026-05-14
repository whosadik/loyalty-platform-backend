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


def _catalog_discount_percent(
    price: Decimal | None,
    original_price: Decimal | None,
    explicit_discount: Decimal | None,
) -> int | None:
    if explicit_discount is not None and explicit_discount > 0:
        return min(99, int(explicit_discount.quantize(Decimal("1"), rounding=ROUND_HALF_UP)))

    if price is None or original_price is None or original_price <= 0 or original_price <= price:
        return None

    discount = ((original_price - price) / original_price * _ONE_HUNDRED).quantize(
        Decimal("1"),
        rounding=ROUND_HALF_UP,
    )
    return max(1, int(discount))


def _catalog_original_price(
    price: Decimal | None,
    original_price: Decimal | None,
    explicit_discount: Decimal | None,
) -> Decimal | None:
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


def _public_discount(product: Any) -> dict | None:
    from offers.public_catalog_pricing import get_public_discount_for_product

    return get_public_discount_for_product(product)


def _effective_price_components(product: Any) -> tuple[Decimal | None, Decimal | None, int | None]:
    cache_attr = "_catalog_effective_price_components_cache"
    cached = getattr(product, cache_attr, None)
    if cached is not None:
        return cached

    price, raw_original_price, explicit_discount = _price_components(product)
    if price is None:
        result = (None, None, None)
        setattr(product, cache_attr, result)
        return result

    catalog_original_price = _catalog_original_price(price, raw_original_price, explicit_discount)
    catalog_discount = _catalog_discount_percent(price, raw_original_price, explicit_discount)

    public_discount = _public_discount(product)
    if not public_discount:
        result = (price.quantize(_TWO_PLACES, rounding=ROUND_HALF_UP), catalog_original_price, catalog_discount)
        setattr(product, cache_attr, result)
        return result

    public_percent = _to_decimal(public_discount.get("discount_percent"))
    if public_percent is None or public_percent <= 0:
        result = (price.quantize(_TWO_PLACES, rounding=ROUND_HALF_UP), catalog_original_price, catalog_discount)
        setattr(product, cache_attr, result)
        return result

    public_percent = min(public_percent, Decimal("100"))
    effective_price = (price * (_ONE - (public_percent / _ONE_HUNDRED))).quantize(
        _TWO_PLACES,
        rounding=ROUND_HALF_UP,
    )
    original_price = catalog_original_price if catalog_original_price and catalog_original_price > price else price

    if original_price <= effective_price:
        result = (price.quantize(_TWO_PLACES, rounding=ROUND_HALF_UP), catalog_original_price, catalog_discount)
        setattr(product, cache_attr, result)
        return result

    combined_discount = ((original_price - effective_price) / original_price * _ONE_HUNDRED).quantize(
        Decimal("1"),
        rounding=ROUND_HALF_UP,
    )
    result = (
        effective_price,
        original_price.quantize(_TWO_PLACES, rounding=ROUND_HALF_UP),
        max(1, int(combined_discount)),
    )
    setattr(product, cache_attr, result)
    return result


def get_product_effective_price(product: Any) -> Decimal | None:
    price, _, _ = _effective_price_components(product)
    return price


def get_product_discount_percent(product: Any) -> int | None:
    _, _, discount = _effective_price_components(product)
    return discount


def get_product_original_price(product: Any) -> Decimal | None:
    _, original_price, _ = _effective_price_components(product)
    return original_price


def product_has_discount(product: Any) -> bool:
    return (get_product_discount_percent(product) or 0) > 0
