from __future__ import annotations

from datetime import timedelta

from django.db.models import Count
from django.utils import timezone
from django.utils.text import slugify

from .models import Product
from .sale_fields import product_has_discount


CATEGORY_LABELS = {
    Product.Category.SKINCARE: "skincare",
    Product.Category.HAIRCARE: "haircare",
    Product.Category.MAKEUP: "makeup",
    Product.Category.FRAGRANCE: "fragrance",
}


def brand_to_slug(value: str) -> str:
    return slugify((value or "").strip(), allow_unicode=True)


def _brand_logo_letter(value: str) -> str:
    normalized = (value or "").strip()
    return normalized[:1].upper() or "B"


def build_brand_summary_payload(brand_name: str, product_count: int) -> dict[str, object]:
    normalized_name = (brand_name or "").strip()
    return {
        "slug": brand_to_slug(normalized_name),
        "name": normalized_name,
        "logo_letter": _brand_logo_letter(normalized_name),
        "product_count": int(product_count),
    }


def list_brand_summary_payloads() -> list[dict[str, object]]:
    rows = (
        Product.objects.exclude(brand="")
        .values("brand")
        .annotate(product_count=Count("id"))
        .order_by("-product_count", "brand")
    )
    return [build_brand_summary_payload(row["brand"], row["product_count"]) for row in rows]


def resolve_brand_name_from_slug(brand_slug: str) -> str | None:
    normalized_slug = brand_to_slug(brand_slug)
    if not normalized_slug:
        return None

    exact_match = (
        Product.objects.exclude(brand="")
        .filter(brand__iexact=brand_slug)
        .values_list("brand", flat=True)
        .first()
    )
    if exact_match:
        return exact_match

    for brand_name in Product.objects.exclude(brand="").order_by().values_list("brand", flat=True).distinct():
        if brand_to_slug(brand_name) == normalized_slug:
            return brand_name

    return None


def _build_brand_description(
    brand_name: str,
    product_count: int,
    categories: list[str],
    top_product_types: list[str],
) -> str:
    category_labels = [CATEGORY_LABELS.get(category, category) for category in categories[:2]]
    top_types = [product_type.replace("_", " ") for product_type in top_product_types[:3]]

    if category_labels and top_types:
        return (
            f"{brand_name} в каталоге: {product_count} товаров в категориях "
            f"{', '.join(category_labels)}. Чаще всего встречаются {', '.join(top_types)}."
        )

    if category_labels:
        return f"{brand_name} в каталоге: {product_count} товаров в категориях {', '.join(category_labels)}."

    return f"{brand_name} в каталоге: {product_count} товаров."


def get_brand_detail_payload(brand_slug: str) -> dict[str, object] | None:
    brand_name = resolve_brand_name_from_slug(brand_slug)
    if not brand_name:
        return None

    products = Product.objects.filter(brand__iexact=brand_name).order_by("-id")
    product_count = products.count()
    if product_count == 0:
        return None

    category_rows = list(
        products.values("category")
        .annotate(product_count=Count("id"))
        .order_by("-product_count", "category")
    )
    categories = [row["category"] for row in category_rows]

    top_type_rows = list(
        products.exclude(product_type="")
        .values("product_type")
        .annotate(product_count=Count("id"))
        .order_by("-product_count", "product_type")
    )
    top_product_types = [row["product_type"] for row in top_type_rows[:3]]

    new_cutoff = timezone.now() - timedelta(days=60)
    new_products_count = products.filter(created_at__gte=new_cutoff).count()
    sale_products_count = sum(
        1
        for product in products.only("id", "price", "raw_meta", "attrs").iterator()
        if product_has_discount(product)
    )

    payload = build_brand_summary_payload(brand_name, product_count)
    payload.update(
        {
            "description": _build_brand_description(brand_name, product_count, categories, top_product_types),
            "categories": categories,
            "top_product_types": top_product_types,
            "new_products_count": new_products_count,
            "sale_products_count": sale_products_count,
        }
    )
    return payload
