from __future__ import annotations

from decimal import Decimal, InvalidOperation
from typing import Iterable

from catalog.models import Product


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


def _format_label(value) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    prepared = value.strip().replace("_", " ")
    return prepared[:1].upper() + prepared[1:]


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


def _to_badge(promo_type: str) -> str:
    if promo_type == "discount":
        return "Скидка"
    if promo_type == "points":
        return "Баллы"
    if promo_type == "gift":
        return "Подарок"
    return "Для вас"


def _build_title(promo_type: str, offer_name: str | None, offer_value: float | None, target: dict | None) -> str:
    category_label = _format_label((target or {}).get("category"))
    product_type_label = _format_label((target or {}).get("product_type"))

    if promo_type == "discount" and offer_value is not None:
        discount = _format_value(offer_value)
        if product_type_label:
            return f"Скидка {discount}% на {product_type_label.lower()}"
        if category_label:
            return f"Скидка {discount}% на {category_label.lower()}"
        return f"Скидка {discount}% для вас"

    if promo_type == "points" and offer_value is not None:
        return f"x{_format_value(offer_value)} баллы на покупку"

    if promo_type == "gift":
        return offer_name or "Подарок к заказу"

    return offer_name or "Персональное предложение"


def _build_description(promo_type: str, target: dict | None, reason: dict | None, offer_value: float | None) -> str:
    target = target or {}
    reason = reason or {}
    scope = _first_string(target.get("scope"))
    category_label = _format_label(target.get("category"))
    product_type_label = _format_label(target.get("product_type"))
    min_basket_amount = _to_number(target.get("min_basket_amount"))
    roadmap_reason = reason.get("roadmap") if _is_record(reason.get("roadmap")) else None

    if scope == "cart" and min_basket_amount is not None:
        return f"Применяется автоматически к корзине от {int(min_basket_amount):,}".replace(",", " ") + " ₸."

    if scope == "category" and category_label:
        return f"Предложение действует на категорию «{category_label}»."

    if scope == "product_type" and product_type_label:
        return f"Предложение действует на товары типа «{product_type_label}»."

    if scope == "product_id" and product_type_label:
        return f"Предложение привязано к рекомендованному товару типа «{product_type_label}»."

    if roadmap_reason:
        return "Оффер связан с вашим текущим шагом roadmap и подобран под следующий шаг покупки."

    if promo_type == "points" and offer_value is not None:
        return f"Баллы будут начислены с повышенным коэффициентом x{_format_value(offer_value)}."

    if promo_type == "gift":
        return "Подарок добавится автоматически при выполнении условий предложения."

    return "Персональное предложение доступно для вас прямо сейчас."


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

    qs = Product.objects.filter(in_stock=True)
    if category:
        qs = qs.filter(category=category)
    if product_type:
        qs = qs.filter(product_type=product_type)

    return qs.only("id", "image_url", "image_urls").order_by("-id").first()


def build_offer_presentation(assignment, *, product_cache: dict[int, Product] | None = None) -> dict:
    target = assignment.target if isinstance(getattr(assignment, "target", None), dict) else {}
    reason = assignment.reason if isinstance(getattr(assignment, "reason", None), dict) else {}
    offer = assignment.offer
    promo_type = _to_promotion_type(getattr(offer, "offer_type", None))
    offer_value = _to_number(getattr(offer, "value", None))
    image_url = _pick_product_image(_resolve_target_product(target, product_cache=product_cache))

    return {
        "title": _build_title(promo_type, _first_string(getattr(offer, "name", None)), offer_value, target),
        "description": _build_description(promo_type, target, reason, offer_value),
        "badge": _to_badge(promo_type),
        "cta_label": "Подробнее",
        "image_url": image_url,
    }


def build_offer_assignment_payload(
    assignment,
    *,
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
        "presentation": build_offer_presentation(assignment, product_cache=product_cache),
    }

    if include_assigned_at:
        payload["assigned_at"] = assignment.assigned_at

    if include_estimated_cost:
        payload["offer"]["estimated_cost"] = str(assignment.offer.estimated_cost)

    return payload
