from __future__ import annotations

from decimal import Decimal, InvalidOperation
from typing import Iterable

from backend.request_language import AppLanguage
from catalog.models import Product
from roadmap_app.step_presentation import get_roadmap_step_presentation


def _is_record(value) -> bool:
    return isinstance(value, dict)


def _first_string(*values) -> str | None:
    for value in values:
        if isinstance(value, str) and value.strip():
            return value.strip()
        if isinstance(value, (list, tuple)):
            for item in value:
                if isinstance(item, str) and item.strip():
                    return item.strip()
    return None


def _to_number(value) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, str) and value.strip():
        try:
            return float(Decimal(value.strip()))
        except (InvalidOperation, ValueError):
            return None
    return None


def _to_int(value) -> int | None:
    number = _to_number(value)
    if number is None:
        return None
    rounded = int(number)
    return rounded if float(rounded) == float(number) else None


def _format_value(value: float) -> str:
    return str(int(value)) if float(value).is_integer() else f"{value:.1f}"


def _to_promotion_type(offer_type: str | None) -> str:
    if offer_type == "discount":
        return "discount"
    if offer_type == "points_multiplier":
        return "points"
    if offer_type == "gift":
        return "gift"
    return "personal"


OFFER_CATEGORY_LABELS: dict[AppLanguage, dict[str, str]] = {
    "ru": {
        "skincare": "уход за кожей",
        "haircare": "уход за волосами",
        "makeup": "макияж",
        "fragrance": "ароматы",
    },
    "kk": {
        "skincare": "тері күтімі",
        "haircare": "шаш күтімі",
        "makeup": "макияж",
        "fragrance": "хош иістер",
    },
    "en": {
        "skincare": "skincare",
        "haircare": "haircare",
        "makeup": "makeup",
        "fragrance": "fragrance",
    },
}

OFFER_COPY: dict[AppLanguage, dict[str, str]] = {
    "ru": {
        "badge_discount": "Скидка",
        "badge_points": "Баллы",
        "badge_gift": "Подарок",
        "badge_personal": "Для вас",
        "discount_generic": "Скидка {value}% для вас",
        "discount_target": "Скидка {value}% на {target}",
        "points_title": "x{value} баллов на покупку",
        "gift_title": "Подарок к заказу",
        "personal_title": "Персональное предложение",
        "cart_description": "Применяется автоматически к корзине от {amount} ₸.",
        "category_description": "Предложение действует на категорию «{label}».",
        "product_type_description": "Предложение действует на товары типа «{label}».",
        "product_id_description": "Предложение привязано к рекомендованному товару типа «{label}».",
        "roadmap_description": "Оффер связан с вашим текущим шагом roadmap и подобран под следующий шаг покупки.",
        "points_description": "Баллы будут начислены с повышенным коэффициентом x{value}.",
        "gift_description": "Подарок добавится автоматически при выполнении условий предложения.",
        "default_description": "Персональное предложение доступно для вас прямо сейчас.",
        "cta": "Подробнее",
    },
    "kk": {
        "badge_discount": "Жеңілдік",
        "badge_points": "Ұпайлар",
        "badge_gift": "Сыйлық",
        "badge_personal": "Сізге",
        "discount_generic": "Сізге {value}% жеңілдік",
        "discount_target": "{target} үшін {value}% жеңілдік",
        "points_title": "Сатып алуға x{value} ұпай",
        "gift_title": "Тапсырысқа сыйлық",
        "personal_title": "Жеке ұсыныс",
        "cart_description": "{amount} ₸ бастап себетке автоматты түрде қолданылады.",
        "category_description": "Ұсыныс «{label}» санатына қолданылады.",
        "product_type_description": "Ұсыныс «{label}» түріндегі тауарларға қолданылады.",
        "product_id_description": "Ұсыныс «{label}» түріндегі ұсынылған тауарға байланыстырылған.",
        "roadmap_description": "Бұл оффер roadmap-тағы ағымдағы қадамыңызбен байланысты және келесі сатып алу қадамына сай таңдалған.",
        "points_description": "Ұпайлар x{value} жоғары коэффициентпен есептеледі.",
        "gift_description": "Ұсыныс шарттары орындалғанда сыйлық автоматты түрде қосылады.",
        "default_description": "Бұл жеке ұсыныс сізге дәл қазір қолжетімді.",
        "cta": "Толығырақ",
    },
    "en": {
        "badge_discount": "Discount",
        "badge_points": "Points",
        "badge_gift": "Gift",
        "badge_personal": "For you",
        "discount_generic": "{value}% off for you",
        "discount_target": "{value}% off {target}",
        "points_title": "x{value} points on purchase",
        "gift_title": "Gift with order",
        "personal_title": "Personal offer",
        "cart_description": "Applies automatically to carts from {amount} ₸.",
        "category_description": "This offer applies to the “{label}” category.",
        "product_type_description": "This offer applies to products of type “{label}”.",
        "product_id_description": "This offer is linked to a recommended product of type “{label}”.",
        "roadmap_description": "This offer is tied to your current roadmap step and selected for the next purchase step.",
        "points_description": "Points will be credited with an increased x{value} multiplier.",
        "gift_description": "The gift will be added automatically when the offer conditions are met.",
        "default_description": "A personal offer is available for you right now.",
        "cta": "Learn more",
    },
}


def _format_label(value: str | None, language: AppLanguage) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None

    normalized = value.strip()
    if normalized in OFFER_CATEGORY_LABELS[language]:
        return OFFER_CATEGORY_LABELS[language][normalized]

    prepared = normalized.replace("_", " ")
    if normalized in {
        "cleanser",
        "toner",
        "serum",
        "moisturizer",
        "spf",
        "shampoo",
        "conditioner",
        "hair_mask",
        "hair_oil",
        "scalp_serum",
        "foundation",
        "eyeshadow",
        "lipstick",
        "perfume",
    }:
        return get_roadmap_step_presentation(normalized, language)["title"]

    if language == "en":
        return prepared.title()

    return prepared[:1].upper() + prepared[1:]


def _format_target_label(target: dict | None, language: AppLanguage) -> str | None:
    target = target or {}
    product_type_label = _format_label(_first_string(target.get("product_type")), language)
    if product_type_label:
        return product_type_label.lower() if language == "ru" else product_type_label

    category_label = _format_label(_first_string(target.get("category")), language)
    if category_label:
        return category_label.lower() if language == "ru" else category_label

    return None


def _to_badge(promo_type: str, language: AppLanguage) -> str:
    copy = OFFER_COPY[language]
    if promo_type == "discount":
        return copy["badge_discount"]
    if promo_type == "points":
        return copy["badge_points"]
    if promo_type == "gift":
        return copy["badge_gift"]
    return copy["badge_personal"]


def _build_title(
    promo_type: str,
    offer_name: str | None,
    offer_value: float | None,
    target: dict | None,
    language: AppLanguage,
) -> str:
    copy = OFFER_COPY[language]
    target_label = _format_target_label(target, language)

    if promo_type == "discount" and offer_value is not None:
        formatted_value = _format_value(offer_value)
        if target_label:
            return copy["discount_target"].format(value=formatted_value, target=target_label)
        return copy["discount_generic"].format(value=formatted_value)

    if promo_type == "points" and offer_value is not None:
        return copy["points_title"].format(value=_format_value(offer_value))

    if promo_type == "gift":
        return offer_name or copy["gift_title"]

    return offer_name or copy["personal_title"]


def _build_description(
    promo_type: str,
    target: dict | None,
    reason: dict | None,
    offer_value: float | None,
    language: AppLanguage,
) -> str:
    copy = OFFER_COPY[language]
    target = target or {}
    reason = reason or {}
    scope = _first_string(target.get("scope"))
    category_label = _format_label(_first_string(target.get("category")), language)
    product_type_label = _format_label(_first_string(target.get("product_type")), language)
    picked_via = _first_string(target.get("picked_via")) or ""
    min_basket_amount = _to_number(target.get("min_basket_amount"))

    if scope == "cart" and min_basket_amount is not None:
        return copy["cart_description"].format(amount=f"{int(min_basket_amount):,}".replace(",", " "))

    if scope == "category" and category_label:
        return copy["category_description"].format(label=category_label)

    if scope == "product_type" and product_type_label:
        return copy["product_type_description"].format(label=product_type_label)

    if scope == "product_id" and product_type_label:
        return copy["product_id_description"].format(label=product_type_label)

    if picked_via.startswith("roadmap_shortcut"):
        return copy["roadmap_description"]

    if promo_type == "points" and offer_value is not None:
        return copy["points_description"].format(value=_format_value(offer_value))

    if promo_type == "gift":
        return copy["gift_description"]

    return copy["default_description"]


def _pick_product_image(product: Product | None) -> str | None:
    if product is None:
        return None
    if isinstance(product.image_url, str) and product.image_url.strip():
        return product.image_url.strip()
    if isinstance(product.image_urls, list):
        for value in product.image_urls:
            if isinstance(value, str) and value.strip():
                return value.strip()
    return None


def build_offer_product_cache(assignments: Iterable) -> dict[int, Product]:
    product_ids: set[int] = set()
    for assignment in assignments:
        target = assignment.target if isinstance(getattr(assignment, "target", None), dict) else {}
        if _first_string(target.get("scope")) != "product_id":
            continue
        product_id = _to_int(target.get("value"))
        if product_id is not None:
            product_ids.add(product_id)
    return Product.objects.in_bulk(product_ids) if product_ids else {}


def _resolve_target_product(target: dict | None, product_cache: dict[int, Product] | None = None) -> Product | None:
    target = target or {}
    scope = _first_string(target.get("scope"))
    category = _first_string(target.get("category"))
    product_type = _first_string(target.get("product_type"))

    if scope == "product_id":
        product_id = _to_int(target.get("value"))
        if product_id is None:
            return None
        if product_cache and product_id in product_cache:
            return product_cache[product_id]
        return Product.objects.filter(id=product_id).only("id", "image_url", "image_urls").first()

    if not category and not product_type:
        return None

    queryset = Product.objects.filter(in_stock=True)
    if category:
        queryset = queryset.filter(category=category)
    if product_type:
        queryset = queryset.filter(product_type=product_type)

    return queryset.only("id", "image_url", "image_urls").order_by("-id").first()


def build_offer_presentation(
    assignment,
    *,
    language: AppLanguage = "ru",
    product_cache: dict[int, Product] | None = None,
) -> dict:
    target = assignment.target if isinstance(getattr(assignment, "target", None), dict) else {}
    reason = assignment.reason if isinstance(getattr(assignment, "reason", None), dict) else {}
    offer = assignment.offer
    promo_type = _to_promotion_type(getattr(offer, "offer_type", None))
    offer_value = _to_number(getattr(offer, "value", None))
    image_url = _pick_product_image(_resolve_target_product(target, product_cache=product_cache))

    return {
        "title": _build_title(
            promo_type,
            _first_string(getattr(offer, "name", None)),
            offer_value,
            target,
            language,
        ),
        "description": _build_description(promo_type, target, reason, offer_value, language),
        "badge": _to_badge(promo_type, language),
        "cta_label": OFFER_COPY[language]["cta"],
        "image_url": image_url,
    }


def build_offer_assignment_payload(
    assignment,
    *,
    language: AppLanguage = "ru",
    include_assigned_at: bool = False,
    include_estimated_cost: bool = False,
    product_cache: dict[int, Product] | None = None,
) -> dict:
    payload = {
        "assignment_id": assignment.id,
        "expires_at": assignment.expires_at,
        "target": assignment.target,
        "reason": assignment.reason,
        "offer": {
            "id": assignment.offer.id,
            "name": assignment.offer.name,
            "type": assignment.offer.offer_type,
            "value": str(assignment.offer.value),
        },
        "presentation": build_offer_presentation(
            assignment,
            language=language,
            product_cache=product_cache,
        ),
    }

    if include_assigned_at:
        payload["assigned_at"] = assignment.assigned_at

    if include_estimated_cost:
        payload["offer"]["estimated_cost"] = str(assignment.offer.estimated_cost)

    return payload
