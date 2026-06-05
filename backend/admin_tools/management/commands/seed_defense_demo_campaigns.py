from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from decimal import Decimal
import json

from django.contrib.auth import get_user_model
from django.core.management.base import BaseCommand, CommandError
from django.db import transaction
from django.db.models import Q
from django.utils import timezone

from catalog.models import Product
from loyalty.models import LoyaltyAccount, Tier
from loyalty.points import DEFAULT_POINTS_RATE
from offers.events import record_offer_event
from offers.models import CampaignBudget, Offer, OfferAssignment, OfferEvent
from roadmap_app.services import (
    get_active_plan,
    get_next_missing_step,
    get_next_visible_missing_step,
    refresh_roadmap,
)
from users_app.models import CustomerProfile


DEFAULT_PASSWORD = "demo12345"
DEFENSE_CLEAN_USERNAME = "defense_clean"
DEFAULT_USERNAMES = (DEFENSE_CLEAN_USERNAME,)
DEMO_POINTS_BALANCE = 1200
OBSOLETE_PUBLIC_CAMPAIGN_NAMES = ("Cleanser Duo Deal -14%",)


@dataclass(frozen=True)
class PublicCampaignSpec:
    name: str
    offer_name: str
    promo_text: str
    category: str
    product_types: tuple[str, ...]
    discount_percent: Decimal
    priority: int
    banner_product_types: tuple[str, ...]
    brands: tuple[str, ...] = ()
    product_ids: tuple[int, ...] = ()
    target_scope: str = "category"


@dataclass(frozen=True)
class PersonalOfferSpec:
    name: str
    offer_type: str
    value: Decimal
    target_scope: str
    allowed_categories: tuple[str, ...] = ()
    allowed_product_types: tuple[str, ...] = ()
    allowed_steps: tuple[str, ...] = ()
    allowed_brands: tuple[str, ...] = ()
    allowed_product_ids: tuple[int, ...] = ()
    estimated_cost: Decimal = Decimal("1000.00")
    cooldown_days: int = 7


@dataclass(frozen=True)
class PersonalCampaignSpec:
    name: str
    priority: int
    weekly_limit: Decimal
    promo_text: str
    allowed_categories: tuple[str, ...] = ()
    allowed_steps: tuple[str, ...] = ()
    recommendation_rules: dict | None = None
    offers: tuple[PersonalOfferSpec, ...] = ()


PUBLIC_CAMPAIGNS: tuple[PublicCampaignSpec, ...] = (
    PublicCampaignSpec(
        name="Beauty Week: уход -15%",
        offer_name="15% на базовый уход",
        promo_text=(
            "Неделя ухода: скидка 15% на очищение, сыворотки, кремы и SPF. "
            "Подходит для демонстрации персональной дорожной карты и каталога."
        ),
        category=Product.Category.SKINCARE,
        product_types=("cleanser", "toner", "serum", "moisturizer", "spf", "mask"),
        discount_percent=Decimal("15.00"),
        priority=10,
        banner_product_types=("serum", "moisturizer", "spf", "cleanser"),
    ),
    PublicCampaignSpec(
        name="Hair Repair Days -12%",
        offer_name="12% на восстановление волос",
        promo_text=(
            "Шампуни, кондиционеры, маски и несмываемый уход со скидкой 12% "
            "для сценария восстановления волос."
        ),
        category=Product.Category.HAIRCARE,
        product_types=("shampoo", "conditioner", "hair_mask", "hair_oil", "leave_in"),
        discount_percent=Decimal("12.00"),
        priority=20,
        banner_product_types=("conditioner", "hair_mask", "shampoo"),
    ),
    PublicCampaignSpec(
        name="Makeup Glow -10%",
        offer_name="10% на макияж для сияния",
        promo_text=(
            "Тон, румяна, тушь и помады со скидкой 10%. Хорошо смотрится "
            "в витрине акций и карточках товаров."
        ),
        category=Product.Category.MAKEUP,
        product_types=("foundation", "blush", "lipstick", "mascara", "primer", "setting_spray"),
        discount_percent=Decimal("10.00"),
        priority=30,
        banner_product_types=("foundation", "blush", "lipstick"),
    ),
    PublicCampaignSpec(
        name="Fragrance Evening -18%",
        offer_name="18% на вечерние ароматы",
        promo_text=(
            "Подборка EDP и EDT со скидкой 18% для дорогого, заметного "
            "оффера в демо корзины."
        ),
        category=Product.Category.FRAGRANCE,
        product_types=("edp", "edt", "body_mist"),
        discount_percent=Decimal("18.00"),
        priority=40,
        banner_product_types=("edp", "edt"),
    ),
    PublicCampaignSpec(
        name="d'Alba Brand Focus -20%",
        offer_name="20% на d'Alba skincare",
        promo_text=(
            "Брендовая акция: d'Alba со скидкой 20%. Подходит для показа "
            "фильтрации offer по бренду и категории."
        ),
        category=Product.Category.SKINCARE,
        product_types=("serum", "moisturizer", "spf", "essence"),
        discount_percent=Decimal("20.00"),
        priority=50,
        banner_product_types=("serum", "moisturizer"),
        brands=("d'Alba",),
        target_scope="brand",
    ),
    PublicCampaignSpec(
        name="3INA Makeup Brand -15%",
        offer_name="15% на 3INA makeup",
        promo_text=(
            "Брендовая акция на 3INA: палетки, румяна, помады и тональные "
            "средства со скидкой 15%."
        ),
        category=Product.Category.MAKEUP,
        product_types=("eyeshadow", "blush", "lipstick", "foundation"),
        discount_percent=Decimal("15.00"),
        priority=60,
        banner_product_types=("eyeshadow", "blush", "lipstick"),
        brands=("3INA",),
        target_scope="brand",
    ),
    PublicCampaignSpec(
        name="Cleanser Duo Deal -22%",
        offer_name="22% на выбранные cleanser",
        promo_text=(
            "Точечная скидка 22% на конкретные товары SUR.MEDIC+ cleanser. "
            "Этот сценарий показывает product_id targeting."
        ),
        category=Product.Category.SKINCARE,
        product_types=("cleanser",),
        discount_percent=Decimal("22.00"),
        priority=70,
        banner_product_types=("cleanser",),
        product_ids=(1400, 1404),
        target_scope="product_id",
    ),
)


PERSONAL_CAMPAIGN_NAME = "Personal Roadmap Step"
PERSONAL_OFFER_NAME = "Персонально: -25% на следующий шаг"
PERSONAL_OFFER_VALUE = Decimal("25.00")

PERSONAL_CAMPAIGNS: tuple[PersonalCampaignSpec, ...] = (
    PersonalCampaignSpec(
        name=PERSONAL_CAMPAIGN_NAME,
        priority=1,
        weekly_limit=Decimal("1500000.00"),
        promo_text=(
            "Roadmap-based персональный offer: повышенная скидка на следующий "
            "товар из индивидуальной дорожной карты."
        ),
        allowed_categories=(
            Product.Category.SKINCARE,
            Product.Category.HAIRCARE,
            Product.Category.MAKEUP,
            Product.Category.FRAGRANCE,
        ),
        recommendation_rules={"demo_seed": "seed_defense_demo_campaigns", "intent": "roadmap_next_step"},
        offers=(
            PersonalOfferSpec(
                name=PERSONAL_OFFER_NAME,
                offer_type=Offer.Type.DISCOUNT,
                value=PERSONAL_OFFER_VALUE,
                target_scope="product_id",
                allowed_categories=(
                    Product.Category.SKINCARE,
                    Product.Category.HAIRCARE,
                    Product.Category.MAKEUP,
                    Product.Category.FRAGRANCE,
                ),
                estimated_cost=Decimal("1200.00"),
                cooldown_days=0,
            ),
        ),
    ),
    PersonalCampaignSpec(
        name="onboarding_first_order",
        priority=5,
        weekly_limit=Decimal("900000.00"),
        promo_text="Welcome-кампания для нового покупателя без завершённых заказов.",
        recommendation_rules={"intent": "first_order_activation", "segment": "new_or_rare"},
        offers=(
            PersonalOfferSpec(
                name="Welcome: -15% на первый заказ",
                offer_type=Offer.Type.DISCOUNT,
                value=Decimal("15.00"),
                target_scope="cart",
                estimated_cost=Decimal("900.00"),
                cooldown_days=0,
            ),
        ),
    ),
    PersonalCampaignSpec(
        name="favorite_category",
        priority=15,
        weekly_limit=Decimal("800000.00"),
        promo_text="Персональная скидка на категорию, в которой пользователь чаще покупает.",
        allowed_categories=(
            Product.Category.SKINCARE,
            Product.Category.HAIRCARE,
            Product.Category.MAKEUP,
            Product.Category.FRAGRANCE,
        ),
        recommendation_rules={"intent": "favorite_category_boost"},
        offers=(
            PersonalOfferSpec(
                name="Любимая категория: -18%",
                offer_type=Offer.Type.DISCOUNT,
                value=Decimal("18.00"),
                target_scope="category",
                allowed_categories=(
                    Product.Category.SKINCARE,
                    Product.Category.HAIRCARE,
                    Product.Category.MAKEUP,
                    Product.Category.FRAGRANCE,
                ),
                estimated_cost=Decimal("1100.00"),
                cooldown_days=10,
            ),
        ),
    ),
    PersonalCampaignSpec(
        name="winback_30d",
        priority=20,
        weekly_limit=Decimal("700000.00"),
        promo_text="Winback-кампания для пользователя, который давно не покупал.",
        recommendation_rules={"intent": "winback", "inactive_days": 30},
        offers=(
            PersonalOfferSpec(
                name="Comeback: -20% на корзину",
                offer_type=Offer.Type.DISCOUNT,
                value=Decimal("20.00"),
                target_scope="cart",
                estimated_cost=Decimal("1400.00"),
                cooldown_days=30,
            ),
        ),
    ),
    PersonalCampaignSpec(
        name="fragrance_crosssell",
        priority=50,
        weekly_limit=Decimal("300000.00"),
        promo_text="Персональный cross-sell в fragrance после покупки или интереса к аромату.",
        allowed_categories=(Product.Category.FRAGRANCE,),
        recommendation_rules={"intent": "fragrance_crosssell"},
        offers=(
            PersonalOfferSpec(
                name="Fragrance discovery: -15%",
                offer_type=Offer.Type.DISCOUNT,
                value=Decimal("15.00"),
                target_scope="product_type",
                allowed_categories=(Product.Category.FRAGRANCE,),
                allowed_product_types=("edp", "edt", "body_mist"),
                estimated_cost=Decimal("1300.00"),
            ),
            PersonalOfferSpec(
                name="x2 баллы на fragrance",
                offer_type=Offer.Type.POINTS_MULTIPLIER,
                value=Decimal("2.00"),
                target_scope="category",
                allowed_categories=(Product.Category.FRAGRANCE,),
                estimated_cost=Decimal("500.00"),
            ),
        ),
    ),
    PersonalCampaignSpec(
        name="skincare_retention",
        priority=60,
        weekly_limit=Decimal("400000.00"),
        promo_text="Retention-кампания для следующего шага skincare routine.",
        allowed_categories=(Product.Category.SKINCARE,),
        allowed_steps=("spf", "serum", "moisturizer"),
        recommendation_rules={"intent": "routine_retention"},
        offers=(
            PersonalOfferSpec(
                name="Skincare next step: -16%",
                offer_type=Offer.Type.DISCOUNT,
                value=Decimal("16.00"),
                target_scope="product_type",
                allowed_categories=(Product.Category.SKINCARE,),
                allowed_product_types=("cleanser", "serum", "moisturizer", "spf", "toner", "mask"),
                allowed_steps=("spf", "serum", "moisturizer"),
                estimated_cost=Decimal("1000.00"),
            ),
            PersonalOfferSpec(
                name="x2 баллы на skincare routine",
                offer_type=Offer.Type.POINTS_MULTIPLIER,
                value=Decimal("2.00"),
                target_scope="category",
                allowed_categories=(Product.Category.SKINCARE,),
                estimated_cost=Decimal("450.00"),
            ),
        ),
    ),
    PersonalCampaignSpec(
        name="makeup_push",
        priority=70,
        weekly_limit=Decimal("350000.00"),
        promo_text="Персональная кампания для makeup next step и brand push.",
        allowed_categories=(Product.Category.MAKEUP,),
        recommendation_rules={"intent": "makeup_next_step"},
        offers=(
            PersonalOfferSpec(
                name="Makeup next step: -12%",
                offer_type=Offer.Type.DISCOUNT,
                value=Decimal("12.00"),
                target_scope="product_type",
                allowed_categories=(Product.Category.MAKEUP,),
                allowed_product_types=("foundation", "mascara", "blush", "lipstick", "eyeshadow"),
                estimated_cost=Decimal("850.00"),
            ),
            PersonalOfferSpec(
                name="3INA personal brand: -15%",
                offer_type=Offer.Type.DISCOUNT,
                value=Decimal("15.00"),
                target_scope="brand",
                allowed_categories=(Product.Category.MAKEUP,),
                allowed_brands=("3INA",),
                estimated_cost=Decimal("900.00"),
            ),
        ),
    ),
    PersonalCampaignSpec(
        name="default",
        priority=100,
        weekly_limit=Decimal("1000000.00"),
        promo_text="Fallback-кампания, если нет более точного персонального сигнала.",
        recommendation_rules={"intent": "fallback"},
        offers=(
            PersonalOfferSpec(
                name="Default cart: -7%",
                offer_type=Offer.Type.DISCOUNT,
                value=Decimal("7.00"),
                target_scope="cart",
                estimated_cost=Decimal("600.00"),
            ),
            PersonalOfferSpec(
                name="Haircare next step: -12%",
                offer_type=Offer.Type.DISCOUNT,
                value=Decimal("12.00"),
                target_scope="product_type",
                allowed_categories=(Product.Category.HAIRCARE,),
                allowed_product_types=("shampoo", "conditioner", "hair_mask", "hair_oil", "leave_in"),
                estimated_cost=Decimal("850.00"),
            ),
            PersonalOfferSpec(
                name="x2 баллы на корзину",
                offer_type=Offer.Type.POINTS_MULTIPLIER,
                value=Decimal("2.00"),
                target_scope="cart",
                estimated_cost=Decimal("350.00"),
            ),
        ),
    ),
)

OBSOLETE_PERSONAL_CAMPAIGN_NAMES = ("8march",)


def _week_start(now):
    return (now - timedelta(days=now.weekday())).date()


def _clean_unique(values):
    out = []
    for value in values or []:
        normalized = str(value).strip()
        if normalized and normalized not in out:
            out.append(normalized)
    return out


def _product_image_url(product: Product | None) -> str:
    if product is None:
        return ""
    if product.image:
        return product.image.url
    if product.image_url:
        return product.image_url
    for url in product.image_urls or []:
        if str(url).strip():
            return str(url).strip()
    return ""


def _available_products(category: str, product_type: str | None = None):
    qs = Product.objects.filter(
        category=category,
        in_stock=True,
        price__isnull=False,
    )
    if product_type:
        qs = qs.filter(product_type=product_type)
    return qs.order_by("-price", "id")


def _pick_product(category: str, preferred_types: tuple[str, ...] = ()) -> Product | None:
    for product_type in preferred_types:
        product = _available_products(category, product_type).exclude(image_url="").first()
        if product is not None:
            return product
    product = _available_products(category).exclude(image_url="").first()
    if product is not None:
        return product
    return _available_products(category).first()


def _campaign_products_queryset(spec: PublicCampaignSpec):
    qs = Product.objects.filter(in_stock=True, price__isnull=False)
    if spec.category:
        qs = qs.filter(category=spec.category)
    if spec.product_types:
        qs = qs.filter(product_type__in=list(spec.product_types))
    if spec.product_ids:
        qs = qs.filter(id__in=list(spec.product_ids))
    if spec.brands:
        brand_query = Q()
        for brand in spec.brands:
            brand_query |= Q(brand__iexact=brand)
        qs = qs.filter(brand_query)
    return qs.order_by("-price", "id")


def _pick_campaign_banner_product(spec: PublicCampaignSpec) -> Product | None:
    qs = _campaign_products_queryset(spec)
    for product_type in spec.banner_product_types:
        product = qs.filter(product_type=product_type).exclude(image_url="").first()
        if product is not None:
            return product
    product = qs.exclude(image_url="").first()
    if product is not None:
        return product
    return qs.first()


def _ensure_bronze_tier() -> Tier:
    tier, _ = Tier.objects.get_or_create(
        name="Bronze",
        defaults={"threshold_spend_90d": 0, "points_rate": DEFAULT_POINTS_RATE},
    )
    return tier


def _ensure_loyalty_account(user) -> None:
    tier = _ensure_bronze_tier()
    account, _ = LoyaltyAccount.objects.get_or_create(
        user=user,
        defaults={"tier": tier, "points_balance": DEMO_POINTS_BALANCE},
    )
    changed_fields = []
    if account.tier_id is None:
        account.tier = tier
        changed_fields.append("tier")
    if int(account.points_balance or 0) < DEMO_POINTS_BALANCE:
        account.points_balance = DEMO_POINTS_BALANCE
        changed_fields.append("points_balance")
    if changed_fields:
        account.save(update_fields=changed_fields)


def _ensure_verified_profile(user, now) -> None:
    profile, _ = CustomerProfile.objects.get_or_create(user=user)
    changed_fields = []
    if profile.email_verified_at is None:
        profile.email_verified_at = now
        changed_fields.append("email_verified_at")
    if profile.profile_completed_at is None:
        profile.profile_completed_at = now
        changed_fields.append("profile_completed_at")
    if changed_fields:
        profile.save(update_fields=changed_fields)


def _verify_email_only(user, now) -> None:
    profile, _ = CustomerProfile.objects.get_or_create(user=user)
    if profile.email_verified_at is None:
        profile.email_verified_at = now
        profile.save(update_fields=["email_verified_at"])


def _configure_clean_defense_profile(user, now) -> None:
    profile, _ = CustomerProfile.objects.get_or_create(user=user)
    profile.first_name = "Amina"
    profile.last_name = "Defense"
    profile.city = "Almaty"
    profile.skin_type = CustomerProfile.SkinType.COMBINATION
    profile.goals = ["hydration", "radiance", "barrier_repair"]
    profile.avoid_flags = ["alcohol"]
    profile.budget = CustomerProfile.Budget.MEDIUM
    profile.hair_profile = {
        "hair_type": "wavy",
        "scalp_type": "normal",
        "hair_thickness": "medium",
        "concerns": ["repair", "shine"],
    }
    profile.makeup_profile = {
        "finish_pref": ["natural", "glow"],
        "coverage_pref": ["medium"],
        "undertone": "neutral",
    }
    profile.fragrance_profile = {
        "liked_families": ["fresh", "floral"],
        "liked_notes": ["bergamot", "jasmine"],
        "intensity_pref": "medium",
    }
    profile.profile_completed_at = now
    profile.email_verified_at = now
    profile.save()


def _refresh_clean_defense_roadmaps(user) -> None:
    for category in (
        Product.Category.SKINCARE,
        Product.Category.HAIRCARE,
        Product.Category.MAKEUP,
        Product.Category.FRAGRANCE,
    ):
        refresh_roadmap(user, category=category, force_new=True)


def _target_from_roadmap(user) -> tuple[dict, dict, Product | None]:
    for category in (
        Product.Category.SKINCARE,
        Product.Category.HAIRCARE,
        Product.Category.MAKEUP,
        Product.Category.FRAGRANCE,
    ):
        plan = get_active_plan(user, category=category)
        step = get_next_visible_missing_step(plan) or get_next_missing_step(plan)
        if step is None:
            continue
        product_type = str(step.product_type or "").strip()
        if not product_type:
            continue

        product = None
        if step.recommended_product_id:
            product = _available_products(category).filter(id=step.recommended_product_id).first()
        if product is None:
            product = _pick_product(category, (product_type,))
        if product is None:
            continue

        target = {
            "scope": "product_id",
            "value": int(product.id),
            "category": category,
            "product_type": product.product_type,
            "picked_via": "defense_demo_roadmap",
            "roadmap_product_type": product_type,
            "demo_label": "Следующий шаг дорожной карты",
        }
        reason = {
            "segment": "defense_demo",
            "picked_because": "Prepared demo offer aligned with the user's next roadmap step",
            "roadmap": {
                "category": category,
                "plan_id": int(plan.id) if plan else None,
                "step_id": int(step.id),
                "step_index": int(step.step_index),
                "next_product_type": product_type,
                "next_product_id": int(product.id),
                "link_type": "direct_target",
            },
            "demo_seed": "seed_defense_demo_campaigns",
        }
        return target, reason, product

    product = _pick_product(Product.Category.SKINCARE, ("serum", "moisturizer", "cleanser"))
    if product is None:
        raise CommandError("Catalog needs at least one available product for personal demo offer")

    target = {
        "scope": "product_id",
        "value": int(product.id),
        "category": product.category,
        "product_type": product.product_type,
        "picked_via": "defense_demo_fallback",
        "demo_label": "Персональная рекомендация",
    }
    reason = {
        "segment": "defense_demo",
        "picked_because": "Prepared fallback demo offer because the user has no active roadmap step",
        "demo_seed": "seed_defense_demo_campaigns",
    }
    return target, reason, product


def _ensure_public_campaigns(now, expires_in_days: int) -> list[dict]:
    today = now.date()
    out = []
    for spec in PUBLIC_CAMPAIGNS:
        banner_product = _pick_campaign_banner_product(spec)
        banner_url = _product_image_url(banner_product)
        if not banner_url:
            raise CommandError(f"No banner image found for campaign {spec.name}")

        allowed_categories = [spec.category] if spec.category else []
        allowed_product_types = list(spec.product_types)
        allowed_brands = list(spec.brands)
        allowed_product_ids = list(spec.product_ids)

        campaign, _ = CampaignBudget.objects.update_or_create(
            name=spec.name,
            defaults={
                "campaign_type": CampaignBudget.Type.PUBLIC,
                "is_active": True,
                "priority": spec.priority,
                "weekly_limit": Decimal("3000000.00"),
                "weekly_spent": Decimal("0.00"),
                "week_start_date": _week_start(now),
                "start_date": today - timedelta(days=1),
                "end_date": today + timedelta(days=expires_in_days),
                "allowed_categories": allowed_categories,
                "allowed_steps": allowed_product_types,
                "allowed_brands": allowed_brands,
                "allowed_product_ids": allowed_product_ids,
                "tiers": [],
                "recommendation_rules": {
                    "demo_seed": "seed_defense_demo_campaigns",
                    "banner_product_id": int(banner_product.id) if banner_product else None,
                    "target_scope": spec.target_scope,
                    "show_in": ["home", "promotions", "catalog", "checkout"],
                },
                "promo_text": spec.promo_text,
                "banner_url": banner_url,
            },
        )
        offer, _ = Offer.objects.update_or_create(
            campaign=campaign,
            name=spec.offer_name,
            defaults={
                "is_active": True,
                "offer_type": Offer.Type.DISCOUNT,
                "value": spec.discount_percent,
                "min_total_spend_90d": Decimal("0.00"),
                "allowed_steps": [],
                "estimated_cost": Decimal("0.00"),
                "cooldown_days": 0,
                "expires_in_days": expires_in_days,
                "allowed_categories": allowed_categories,
                "allowed_product_types": allowed_product_types,
                "target_scope": spec.target_scope,
                "allowed_brands": allowed_brands,
                "allowed_product_ids": allowed_product_ids,
            },
        )
        deleted_old_offers = _delete_offers_with_artifacts(
            Offer.objects.filter(campaign=campaign).exclude(name=spec.offer_name)
        )
        products_count = _campaign_products_queryset(spec).count()
        out.append(
            {
                "campaign_id": campaign.id,
                "campaign": campaign.name,
                "offer_id": offer.id,
                "discount_percent": str(spec.discount_percent),
                "category": spec.category,
                "target_scope": spec.target_scope,
                "brands": allowed_brands,
                "product_ids": allowed_product_ids,
                "products_count": products_count,
                "banner_url": banner_url,
                "deleted_old_offers": deleted_old_offers,
            }
        )
    return out


def _delete_offers_with_artifacts(offers_qs) -> dict[str, int]:
    offer_ids = list(offers_qs.values_list("id", flat=True))
    if not offer_ids:
        return {"offers": 0, "assignments": 0, "events": 0}
    assignment_ids = list(
        OfferAssignment.objects.filter(offer_id__in=offer_ids).values_list("id", flat=True)
    )
    events_by_offer = OfferEvent.objects.filter(offer_id__in=offer_ids).delete()[0]
    events_by_assignment = OfferEvent.objects.filter(assignment_id__in=assignment_ids).delete()[0]
    assignments = OfferAssignment.objects.filter(id__in=assignment_ids).delete()[0]
    offers = Offer.objects.filter(id__in=offer_ids).delete()[0]
    return {
        "offers": int(offers),
        "assignments": int(assignments),
        "events": int(events_by_offer) + int(events_by_assignment),
    }


def _delete_campaigns_with_artifacts(campaigns_qs) -> dict[str, int]:
    campaign_ids = list(campaigns_qs.values_list("id", flat=True))
    if not campaign_ids:
        return {"campaigns": 0, "offers": 0, "assignments": 0, "events": 0}
    offer_stats = _delete_offers_with_artifacts(Offer.objects.filter(campaign_id__in=campaign_ids))
    campaigns = CampaignBudget.objects.filter(id__in=campaign_ids).delete()[0]
    return {
        "campaigns": int(campaigns),
        **offer_stats,
    }


def _delete_obsolete_public_campaigns(now) -> dict[str, int]:
    del now
    demo_names = {spec.name for spec in PUBLIC_CAMPAIGNS}
    placeholder_qs = (
        CampaignBudget.objects.filter(
            campaign_type=CampaignBudget.Type.PUBLIC,
            name__startswith="Demo:",
        )
        .exclude(name__in=demo_names)
    )
    obsolete_qs = CampaignBudget.objects.filter(
        campaign_type=CampaignBudget.Type.PUBLIC,
        name__in=OBSOLETE_PUBLIC_CAMPAIGN_NAMES,
    )
    placeholder_stats = _delete_campaigns_with_artifacts(placeholder_qs)
    obsolete_stats = _delete_campaigns_with_artifacts(obsolete_qs)
    return {
        key: int(placeholder_stats.get(key, 0)) + int(obsolete_stats.get(key, 0))
        for key in {"campaigns", "offers", "assignments", "events"}
    }


def _ensure_personal_offer(
    *,
    campaign: CampaignBudget,
    spec: PersonalOfferSpec,
    expires_in_days: int,
) -> Offer:
    offer, _ = Offer.objects.update_or_create(
        campaign=campaign,
        name=spec.name,
        defaults={
            "is_active": True,
            "offer_type": spec.offer_type,
            "value": spec.value,
            "min_total_spend_90d": Decimal("0.00"),
            "allowed_steps": list(spec.allowed_steps),
            "estimated_cost": spec.estimated_cost,
            "cooldown_days": spec.cooldown_days,
            "expires_in_days": expires_in_days,
            "allowed_categories": list(spec.allowed_categories),
            "allowed_product_types": list(spec.allowed_product_types),
            "target_scope": spec.target_scope,
            "allowed_brands": list(spec.allowed_brands),
            "allowed_product_ids": list(spec.allowed_product_ids),
        },
    )
    return offer


def _ensure_personal_campaign(expires_in_days: int, now=None) -> tuple[CampaignBudget, Offer, list[dict]]:
    now = now or timezone.now()
    today = now.date()
    roadmap_campaign = None
    roadmap_offer = None
    summary = []

    for spec in PERSONAL_CAMPAIGNS:
        rules = dict(spec.recommendation_rules or {})
        rules["demo_seed"] = "seed_defense_demo_campaigns"
        campaign, _ = CampaignBudget.objects.update_or_create(
            name=spec.name,
            defaults={
                "campaign_type": CampaignBudget.Type.PERSONAL,
                "is_active": True,
                "priority": spec.priority,
                "weekly_limit": spec.weekly_limit,
                "weekly_spent": Decimal("0.00"),
                "week_start_date": _week_start(now),
                "start_date": today - timedelta(days=1),
                "end_date": today + timedelta(days=expires_in_days),
                "allowed_categories": list(spec.allowed_categories),
                "allowed_steps": list(spec.allowed_steps),
                "allowed_brands": [],
                "allowed_product_ids": [],
                "tiers": [],
                "recommendation_rules": rules,
                "promo_text": spec.promo_text,
                "banner_url": "",
            },
        )

        active_offer_names = []
        active_offer_ids = []
        for offer_spec in spec.offers:
            offer = _ensure_personal_offer(
                campaign=campaign,
                spec=offer_spec,
                expires_in_days=expires_in_days,
            )
            active_offer_names.append(offer.name)
            active_offer_ids.append(offer.id)
            if spec.name == PERSONAL_CAMPAIGN_NAME and offer.name == PERSONAL_OFFER_NAME:
                roadmap_campaign = campaign
                roadmap_offer = offer

        disabled_offers = (
            Offer.objects.filter(campaign=campaign, is_active=True)
            .exclude(name__in=active_offer_names)
            .update(is_active=False)
        )
        summary.append(
            {
                "campaign_id": campaign.id,
                "campaign": campaign.name,
                "priority": campaign.priority,
                "active_offer_ids": active_offer_ids,
                "active_offers_count": len(active_offer_ids),
                "disabled_old_offers": int(disabled_offers),
            }
        )

    deleted_campaigns = _delete_campaigns_with_artifacts(CampaignBudget.objects.filter(
        campaign_type=CampaignBudget.Type.PERSONAL,
        name__in=OBSOLETE_PERSONAL_CAMPAIGN_NAMES,
    ))
    if deleted_campaigns["campaigns"]:
        summary.append({"deleted_obsolete_personal": deleted_campaigns})

    if roadmap_campaign is None or roadmap_offer is None:
        raise CommandError("Personal roadmap campaign/offer was not created")

    return roadmap_campaign, roadmap_offer, summary


def _ensure_demo_user(username: str, password: str, now):
    User = get_user_model()
    if username == DEFENSE_CLEAN_USERNAME:
        User.objects.filter(username=username, is_staff=False, is_superuser=False).delete()
        user = User.objects.create_user(
            username=username,
            password=password,
            email="defense_clean@uylesim.kz",
            first_name="Amina",
            last_name="Defense",
        )
        _configure_clean_defense_profile(user, now)
        _ensure_loyalty_account(user)
        _refresh_clean_defense_roadmaps(user)
        return user

    user = User.objects.filter(username=username).first()
    if user is None and username == "demo_user":
        user = User.objects.create_user(
            username=username,
            password=password,
            email="demo@uylesim.kz",
            first_name="Amina",
            last_name="Demo",
        )
    if user is None:
        return None
    _ensure_verified_profile(user, now)
    _ensure_loyalty_account(user)
    return user


def _verified_non_staff_usernames() -> list[str]:
    User = get_user_model()
    return list(
        User.objects.filter(
            is_active=True,
            is_staff=False,
            customerprofile__email_verified_at__isnull=False,
        )
        .order_by("username")
        .values_list("username", flat=True)
    )


def _prepare_personal_assignment(
    *,
    user,
    offer: Offer,
    now,
    expires_in_days: int,
    reset_existing: bool,
) -> dict:
    if reset_existing:
        OfferAssignment.objects.filter(
            user=user,
            is_active=True,
            is_redeemed=False,
        ).exclude(offer=offer).update(
            is_active=False,
            superseded_at=now,
        )

    target, reason, product = _target_from_roadmap(user)
    expires_at = now + timedelta(days=expires_in_days)
    assignment = (
        OfferAssignment.objects.filter(user=user, offer=offer, is_redeemed=False)
        .order_by("-id")
        .first()
    )
    if assignment is None:
        assignment = OfferAssignment.objects.create(
            user=user,
            offer=offer,
            reason=reason,
            target=target,
            expires_at=expires_at,
        )
        created = True
    else:
        assignment.reason = reason
        assignment.target = target
        assignment.expires_at = expires_at
        assignment.is_active = True
        assignment.is_redeemed = False
        assignment.redeemed_transaction_id = None
        assignment.superseded_at = None
        assignment.assigned_at = now
        assignment.save(
            update_fields=[
                "reason",
                "target",
                "expires_at",
                "is_active",
                "is_redeemed",
                "redeemed_transaction_id",
                "superseded_at",
                "assigned_at",
            ]
        )
        created = False

    record_offer_event(
        assignment,
        OfferEvent.Type.ASSIGNED,
        context={"source": "seed_defense_demo_campaigns"},
    )
    return {
        "username": user.username,
        "assignment_id": assignment.id,
        "created": created,
        "offer_id": offer.id,
        "offer": offer.name,
        "target": target,
        "eligible_product": {
            "id": int(product.id) if product else None,
            "brand": product.brand if product else "",
            "name": product.name if product else "",
            "price": str(product.price) if product else "",
            "image_url": _product_image_url(product),
        },
    }


class Command(BaseCommand):
    help = "Seed realistic public campaigns and personal offer assignments for diploma defense demo."

    def add_arguments(self, parser):
        parser.add_argument(
            "--username",
            action="append",
            dest="usernames",
            default=[],
            help="Username to prepare with a personal offer. Can be passed multiple times.",
        )
        parser.add_argument(
            "--all-verified-users",
            action="store_true",
            help="Also prepare all verified non-staff users.",
        )
        parser.add_argument(
            "--keep-existing-user-offers",
            action="store_true",
            help="Do not deactivate other active personal offers for target users.",
        )
        parser.add_argument(
            "--verify-only",
            action="store_true",
            help="Only mark existing users as email-verified and ensure loyalty account; do not seed offers.",
        )
        parser.add_argument(
            "--expires-in-days",
            type=int,
            default=21,
            help="Campaign and personal assignment validity window.",
        )
        parser.add_argument(
            "--password",
            type=str,
            default=DEFAULT_PASSWORD,
            help="Password used only if demo_user has to be created.",
        )

    def handle(self, *args, **options):
        expires_in_days = max(1, int(options["expires_in_days"]))
        usernames = _clean_unique(options.get("usernames") or [])
        verify_only = bool(options.get("verify_only"))
        if verify_only and not usernames:
            raise CommandError("--verify-only requires at least one --username")
        if not usernames:
            usernames = list(DEFAULT_USERNAMES)
        if options.get("all_verified_users"):
            usernames = _clean_unique([*usernames, *_verified_non_staff_usernames()])

        now = timezone.now()
        if verify_only:
            User = get_user_model()
            verified = []
            skipped = []
            with transaction.atomic():
                for username in usernames:
                    user = User.objects.filter(username=username).first()
                    if user is None:
                        skipped.append(username)
                        continue
                    _verify_email_only(user, now)
                    _ensure_loyalty_account(user)
                    verified.append(username)
            self.stdout.write(
                json.dumps(
                    {
                        "verify_only": True,
                        "verified_usernames": verified,
                        "skipped_usernames": skipped,
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
            return

        with transaction.atomic():
            public = _ensure_public_campaigns(now, expires_in_days)
            deleted_obsolete_public = _delete_obsolete_public_campaigns(now)
            _, personal_offer, personal_campaigns = _ensure_personal_campaign(
                expires_in_days,
                now=now,
            )

            prepared = []
            skipped = []
            for username in usernames:
                user = _ensure_demo_user(username, str(options["password"]), now)
                if user is None:
                    skipped.append(username)
                    continue
                prepared.append(
                    _prepare_personal_assignment(
                        user=user,
                        offer=personal_offer,
                        now=now,
                        expires_in_days=expires_in_days,
                        reset_existing=not bool(options["keep_existing_user_offers"]),
                    )
                )

        summary = {
            "public_campaigns": public,
            "deleted_obsolete_public": deleted_obsolete_public,
            "personal_campaigns": personal_campaigns,
            "personal_assignments": prepared,
            "skipped_usernames": skipped,
            "expires_in_days": expires_in_days,
            "demo_password_if_created": str(options["password"]),
        }
        self.stdout.write(json.dumps(summary, ensure_ascii=False, indent=2))
