from __future__ import annotations

from datetime import timedelta
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from typing import Any

from django.db.models import Q
from django.utils import timezone

from catalog.models import Product
from offers.models import CampaignBudget, Offer


_TWO_PLACES = Decimal("0.01")
_ONE_HUNDRED = Decimal("100")


def _to_decimal(value: Any) -> Decimal | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, Decimal):
        return value
    try:
        return Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError):
        return None


def _d2(value: Decimal) -> Decimal:
    return value.quantize(_TWO_PLACES, rounding=ROUND_HALF_UP)


def _week_start_date(now):
    return (now - timedelta(days=now.weekday())).date()


def _public_campaigns(now):
    today = now.date()
    return (
        CampaignBudget.objects.filter(is_active=True, campaign_type=CampaignBudget.Type.PUBLIC)
        .filter(Q(start_date__isnull=True) | Q(start_date__lte=today))
        .filter(Q(end_date__isnull=True) | Q(end_date__gte=today))
        .prefetch_related("offers")
        .order_by("priority", "id")
    )


def _campaign_left(campaign: CampaignBudget, now) -> Decimal | None:
    weekly_limit = _to_decimal(campaign.weekly_limit) or Decimal("0")
    if weekly_limit <= 0:
        return None

    spent = _to_decimal(campaign.weekly_spent) or Decimal("0")
    if getattr(campaign, "week_start_date", None) and campaign.week_start_date != _week_start_date(now):
        spent = Decimal("0")
    return weekly_limit - spent


def _clean_strings(values) -> list[str]:
    out: list[str] = []
    for value in values or []:
        normalized = str(value).strip()
        if normalized and normalized not in out:
            out.append(normalized)
    return out


def _clean_ints(values) -> list[int]:
    out: list[int] = []
    for value in values or []:
        try:
            normalized = int(value)
        except (TypeError, ValueError):
            continue
        if normalized > 0 and normalized not in out:
            out.append(normalized)
    return out


def _intersect_or_union(left: list, right: list) -> list:
    if left and right:
        right_set = set(right)
        return [value for value in left if value in right_set]
    return left or right


def _intersect_brands_or_union(left: list[str], right: list[str]) -> list[str]:
    if left and right:
        right_lookup = {value.casefold() for value in right}
        return [value for value in left if value.casefold() in right_lookup]
    return left or right


def _constraints(offer: Offer, campaign: CampaignBudget) -> dict:
    offer_categories = _clean_strings(offer.allowed_categories)
    campaign_categories = _clean_strings(campaign.allowed_categories)
    offer_product_types = _clean_strings(offer.allowed_product_types)
    campaign_product_types = _clean_strings(campaign.allowed_steps)
    offer_brands = _clean_strings(getattr(offer, "allowed_brands", []))
    campaign_brands = _clean_strings(getattr(campaign, "allowed_brands", []))
    offer_product_ids = _clean_ints(getattr(offer, "allowed_product_ids", []))
    campaign_product_ids = _clean_ints(getattr(campaign, "allowed_product_ids", []))

    categories = _intersect_or_union(offer_categories, campaign_categories)
    product_types = _intersect_or_union(offer_product_types, campaign_product_types)
    brands = _intersect_brands_or_union(offer_brands, campaign_brands)
    product_ids = _intersect_or_union(offer_product_ids, campaign_product_ids)
    impossible = (
        (offer_categories and campaign_categories and not categories)
        or (offer_product_types and campaign_product_types and not product_types)
        or (offer_brands and campaign_brands and not brands)
        or (offer_product_ids and campaign_product_ids and not product_ids)
    )
    return {
        "impossible": impossible,
        "categories": categories,
        "product_types": product_types,
        "brands": brands,
        "product_ids": product_ids,
    }


def _product_matches(product, constraints: dict) -> bool:
    if constraints.get("impossible"):
        return False

    categories = constraints.get("categories") or []
    product_types = constraints.get("product_types") or []
    brands = constraints.get("brands") or []
    product_ids = constraints.get("product_ids") or []

    if categories and product.category not in categories:
        return False
    if product_types and product.product_type not in product_types:
        return False
    if product_ids and int(product.id) not in product_ids:
        return False
    if brands:
        brand_lookup = {str(value).strip().casefold() for value in brands}
        if str(product.brand or "").strip().casefold() not in brand_lookup:
            return False
    return True


def product_queryset_for_public_offer(offer: Offer, campaign: CampaignBudget, base_qs=None):
    constraints = _constraints(offer, campaign)
    if constraints.get("impossible"):
        return Product.objects.none()

    qs = base_qs if base_qs is not None else Product.objects.all()
    categories = constraints.get("categories") or []
    product_types = constraints.get("product_types") or []
    brands = constraints.get("brands") or []
    product_ids = constraints.get("product_ids") or []

    if categories:
        qs = qs.filter(category__in=categories)
    if product_types:
        qs = qs.filter(product_type__in=product_types)
    if product_ids:
        qs = qs.filter(id__in=product_ids)
    if brands:
        brand_query = Q()
        for brand in brands:
            brand_query |= Q(brand__iexact=brand)
        qs = qs.filter(brand_query)
    return qs


def get_public_discount_for_product(product, now=None) -> dict | None:
    """Return the best active public discount that can apply to a single product."""

    price = _to_decimal(getattr(product, "price", None))
    if price is None or price <= 0:
        return None

    now = now or timezone.now()
    best: tuple[Decimal, int, int, dict] | None = None

    for campaign in _public_campaigns(now):
        left = _campaign_left(campaign, now)
        if left is not None and left <= 0:
            continue

        for offer in campaign.offers.all():
            if not offer.is_active or offer.offer_type != Offer.Type.DISCOUNT:
                continue
            percent = _to_decimal(offer.value)
            if percent is None or percent <= 0:
                continue
            percent = min(percent, _ONE_HUNDRED)

            constraints = _constraints(offer, campaign)
            if not _product_matches(product, constraints):
                continue

            discount_amount = _d2(price * (percent / _ONE_HUNDRED))
            if discount_amount <= 0:
                continue
            if left is not None and discount_amount > left:
                continue

            payload = {
                "campaign_id": int(campaign.id),
                "campaign_name": campaign.name,
                "offer_id": int(offer.id),
                "discount_percent": percent,
                "discount_amount": discount_amount,
            }
            candidate = (discount_amount, -int(campaign.priority or 0), -int(offer.id), payload)
            if best is None or candidate[:3] > best[:3]:
                best = candidate

    return best[3] if best is not None else None
