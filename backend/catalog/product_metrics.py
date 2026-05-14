from __future__ import annotations

from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from typing import Any, Mapping

from django.db.models import Avg, Count
from django.utils.text import slugify

from catalog.sale_fields import get_product_effective_price
from loyalty.models import LoyaltyAccount
from loyalty.points import (
    DEFAULT_POINTS_RATE,
    cap_earned_points,
    get_effective_points_rate,
    get_tier_points_multiplier,
)


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


def _normalize_rating(rating: Decimal) -> float:
    normalized = rating.quantize(Decimal("0.1"), rounding=ROUND_HALF_UP)
    value = float(normalized)
    if value < 0:
        return 0.0
    if value > 5:
        return 5.0
    return value


def _get_imported_review_stats(product: Any) -> tuple[Decimal | None, int]:
    raw_meta = _as_mapping(getattr(product, "raw_meta", {}))
    attrs = _as_mapping(getattr(product, "attrs", {}))

    rating = _pick_decimal(raw_meta, RATING_KEYS) or _pick_decimal(attrs, RATING_KEYS)

    reviews_count = _pick_int(raw_meta, REVIEWS_KEYS)
    if reviews_count is None:
        reviews_count = _pick_int(attrs, REVIEWS_KEYS)

    return rating, max(0, reviews_count or 0)


def _get_customer_review_stats(product: Any) -> tuple[Decimal | None, int | None]:
    annotated_count = _to_int(getattr(product, "customer_reviews_count", None))
    annotated_rating = _to_decimal(getattr(product, "customer_rating_avg", None))
    if annotated_count is not None:
        return annotated_rating, max(0, annotated_count)

    reviews = getattr(product, "reviews", None)
    if reviews is None or not hasattr(reviews, "aggregate"):
        return None, None

    stats = reviews.aggregate(customer_rating_avg=Avg("rating"), customer_reviews_count=Count("id"))
    return _to_decimal(stats.get("customer_rating_avg")), max(0, _to_int(stats.get("customer_reviews_count")) or 0)


def get_product_brand_slug(product: Any) -> str:
    return slugify(str(getattr(product, "brand", "") or "").strip(), allow_unicode=True)


def _user_loyalty_context(user: Any) -> tuple[Decimal, Decimal]:
    if not user or not getattr(user, "is_authenticated", False):
        return DEFAULT_POINTS_RATE, Decimal("1.00")

    cached = getattr(user, "_catalog_points_context", None)
    if cached is not None:
        return cached

    try:
        account = LoyaltyAccount.objects.select_related("tier").filter(user=user).first()
    except Exception:
        account = None

    tier = getattr(account, "tier", None) if account is not None else None
    if tier is None:
        context = DEFAULT_POINTS_RATE, Decimal("1.00")
        setattr(user, "_catalog_points_context", context)
        return context

    context = (
        get_effective_points_rate(getattr(tier, "points_rate", DEFAULT_POINTS_RATE)),
        get_tier_points_multiplier(getattr(tier, "name", None)),
    )
    setattr(user, "_catalog_points_context", context)
    return context


def get_product_points_earned(product: Any, user: Any = None) -> int:
    price = get_product_effective_price(product) or _to_decimal(getattr(product, "price", None)) or Decimal("0")
    points_rate, tier_multiplier = _user_loyalty_context(user)
    base_points = (price * points_rate).quantize(Decimal("1"), rounding=ROUND_HALF_UP)
    tier_points = (base_points * tier_multiplier).quantize(Decimal("1"), rounding=ROUND_HALF_UP)
    return cap_earned_points(max(0, int(tier_points)), price)


def get_product_rating(product: Any) -> float | None:
    imported_rating, imported_reviews_count = _get_imported_review_stats(product)
    customer_rating, customer_reviews_count = _get_customer_review_stats(product)

    if customer_reviews_count and customer_reviews_count > 0 and customer_rating is not None:
        if imported_rating is not None and imported_reviews_count > 0:
            total_count = imported_reviews_count + customer_reviews_count
            combined_rating = (
                (imported_rating * Decimal(imported_reviews_count))
                + (customer_rating * Decimal(customer_reviews_count))
            ) / Decimal(total_count)
            return _normalize_rating(combined_rating)
        return _normalize_rating(customer_rating)

    if imported_rating is None:
        return None

    return _normalize_rating(imported_rating)


def get_product_reviews_count(product: Any) -> int:
    _, imported_reviews_count = _get_imported_review_stats(product)
    _, customer_reviews_count = _get_customer_review_stats(product)

    if customer_reviews_count and customer_reviews_count > 0:
        return imported_reviews_count + customer_reviews_count

    return imported_reviews_count
