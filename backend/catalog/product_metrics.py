from __future__ import annotations

from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from typing import Any, Mapping

from django.utils.text import slugify


RATING_KEYS = ("rating", "avg_rating")
REVIEWS_KEYS = ("reviews_count", "reviews", "ratings_count")


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


def _to_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except Exception:
        return None


def _as_mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, dict) else {}


def _pick_decimal(mapping: Mapping[str, Any], keys: tuple[str, ...]) -> Decimal | None:
    for key in keys:
        parsed = _to_decimal(mapping.get(key))
        if parsed is not None:
            return parsed
    return None


def _pick_int(mapping: Mapping[str, Any], keys: tuple[str, ...]) -> int | None:
    for key in keys:
        parsed = _to_int(mapping.get(key))
        if parsed is not None:
            return parsed
    return None


def get_product_brand_slug(product: Any) -> str:
    return slugify(str(getattr(product, "brand", "") or "").strip(), allow_unicode=True)


def get_product_points_earned(product: Any) -> int:
    price = _to_decimal(getattr(product, "price", None)) or Decimal("0")
    points = (price * Decimal("0.10")).quantize(Decimal("1"), rounding=ROUND_HALF_UP)
    return max(0, int(points))


def get_product_rating(product: Any) -> float | None:
    raw_meta = _as_mapping(getattr(product, "raw_meta", {}))
    attrs = _as_mapping(getattr(product, "attrs", {}))

    rating = _pick_decimal(raw_meta, RATING_KEYS) or _pick_decimal(attrs, RATING_KEYS)
    if rating is None:
        return None

    normalized = rating.quantize(Decimal("0.1"), rounding=ROUND_HALF_UP)
    value = float(normalized)
    if value < 0:
        return 0.0
    if value > 5:
        return 5.0
    return value


def get_product_reviews_count(product: Any) -> int:
    raw_meta = _as_mapping(getattr(product, "raw_meta", {}))
    attrs = _as_mapping(getattr(product, "attrs", {}))

    reviews_count = _pick_int(raw_meta, REVIEWS_KEYS)
    if reviews_count is None:
        reviews_count = _pick_int(attrs, REVIEWS_KEYS)
    return max(0, reviews_count or 0)
