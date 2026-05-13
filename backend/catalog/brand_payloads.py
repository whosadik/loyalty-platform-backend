from __future__ import annotations

from datetime import timedelta

from django.db.models import Count
from django.utils import timezone
from django.utils.text import slugify

from backend.request_language import AppLanguage, normalize_language
from roadmap_app.step_presentation import get_roadmap_step_presentation

from .models import Brand, Product
from .sale_fields import product_has_discount


CATEGORY_LABELS: dict[AppLanguage, dict[str, str]] = {
    "ru": {
        Product.Category.SKINCARE: "уход за кожей",
        Product.Category.HAIRCARE: "уход за волосами",
        Product.Category.MAKEUP: "макияж",
        Product.Category.FRAGRANCE: "ароматы",
    },
    "kk": {
        Product.Category.SKINCARE: "тері күтімі",
        Product.Category.HAIRCARE: "шаш күтімі",
        Product.Category.MAKEUP: "макияж",
        Product.Category.FRAGRANCE: "хош иістер",
    },
    "en": {
        Product.Category.SKINCARE: "skincare",
        Product.Category.HAIRCARE: "haircare",
        Product.Category.MAKEUP: "makeup",
        Product.Category.FRAGRANCE: "fragrance",
    },
}


def brand_to_slug(value: str) -> str:
    return slugify((value or "").strip(), allow_unicode=True)


def _brand_logo_letter(value: str) -> str:
    normalized = (value or "").strip()
    return normalized[:1].upper() or "B"


def _brand_logo_url(brand: Brand | None) -> str:
    if brand is None:
        return ""
    if brand.logo_image:
        try:
            return brand.logo_image.url
        except ValueError:
            return ""
    return brand.logo_url or ""


def build_brand_summary_payload(
    brand_name: str,
    product_count: int,
    brand: Brand | None = None,
) -> dict[str, object]:
    normalized_name = (brand_name or "").strip()
    slug = brand.slug if brand is not None else brand_to_slug(normalized_name)
    return {
        "slug": slug,
        "name": normalized_name,
        "logo_letter": _brand_logo_letter(normalized_name),
        "logo_url": _brand_logo_url(brand),
        "product_count": int(product_count),
    }


def list_brand_summary_payloads() -> list[dict[str, object]]:
    rows = (
        Brand.objects.filter(is_active=True)
        .annotate(product_count=Count("products"))
        .filter(product_count__gt=0)
        .order_by("-product_count", "name")
    )
    return [
        build_brand_summary_payload(brand.name, brand.product_count, brand=brand)
        for brand in rows
    ]


def resolve_brand_from_slug(brand_slug: str) -> Brand | None:
    normalized_slug = brand_to_slug(brand_slug)
    if not normalized_slug:
        return None
    return Brand.objects.filter(slug=normalized_slug).first()


def resolve_brand_name_from_slug(brand_slug: str) -> str | None:
    brand = resolve_brand_from_slug(brand_slug)
    return brand.name if brand is not None else None


def _localize_category(category: str, language: AppLanguage) -> str:
    return CATEGORY_LABELS[language].get(category, category)


def _localize_product_type(product_type: str, language: AppLanguage) -> str:
    presentation = get_roadmap_step_presentation(product_type, language)
    title = str(presentation.get("title") or "").strip()
    if title:
        return title

    prepared = product_type.replace("_", " ").strip()
    return prepared.title() if prepared else product_type


def _build_brand_description(
    brand_name: str,
    product_count: int,
    categories: list[str],
    top_product_types: list[str],
    language: AppLanguage,
) -> str:
    localized_categories = [_localize_category(category, language) for category in categories[:2]]
    localized_types = [_localize_product_type(product_type, language) for product_type in top_product_types[:3]]

    if language == "kk":
        if localized_categories and localized_types:
            return (
                f"{brand_name} каталогында {product_count} тауар бар. Негізгі санаттар: "
                f"{', '.join(localized_categories)}. Ең жиі кездесетін түрлер: {', '.join(localized_types)}."
            )
        if localized_categories:
            return f"{brand_name} каталогында {product_count} тауар бар. Негізгі санаттар: {', '.join(localized_categories)}."
        return f"{brand_name} каталогында {product_count} тауар бар."

    if language == "en":
        if localized_categories and localized_types:
            return (
                f"{brand_name} in the catalog: {product_count} items across {', '.join(localized_categories)}. "
                f"Most common product types: {', '.join(localized_types)}."
            )
        if localized_categories:
            return f"{brand_name} in the catalog: {product_count} items across {', '.join(localized_categories)}."
        return f"{brand_name} in the catalog: {product_count} items."

    if localized_categories and localized_types:
        return (
            f"{brand_name} в каталоге: {product_count} товаров в категориях {', '.join(localized_categories)}. "
            f"Чаще всего встречаются {', '.join(localized_types)}."
        )
    if localized_categories:
        return f"{brand_name} в каталоге: {product_count} товаров в категориях {', '.join(localized_categories)}."
    return f"{brand_name} в каталоге: {product_count} товаров."


def get_brand_detail_payload(
    brand_slug: str,
    language: AppLanguage = "ru",
) -> dict[str, object] | None:
    normalized_language = normalize_language(language)
    brand = resolve_brand_from_slug(brand_slug)
    if brand is None:
        return None

    products = Product.objects.filter(brand_ref=brand).order_by("-id")
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

    custom_description = (getattr(brand, f"description_{normalized_language}", "") or "").strip()
    if not custom_description:
        custom_description = (brand.description_ru or "").strip()
    if not custom_description:
        custom_description = _build_brand_description(
            brand.name,
            product_count,
            categories,
            top_product_types,
            normalized_language,
        )

    payload = build_brand_summary_payload(brand.name, product_count, brand=brand)
    payload.update(
        {
            "description": custom_description,
            "categories": categories,
            "top_product_types": top_product_types,
            "new_products_count": new_products_count,
            "sale_products_count": sale_products_count,
        }
    )
    return payload
