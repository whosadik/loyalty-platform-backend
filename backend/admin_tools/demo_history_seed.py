from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any

from django.contrib.auth import get_user_model
from django.core.management.base import CommandError
from django.db import transaction as db_tx
from django.db.models import Max
from django.utils import timezone
from rest_framework.test import APIClient

from admin_tools.goldapple_catalog import json_default
from admin_tools.goldapple_catalog_curated_v2 import CURATED_V2_CANONICAL_TYPES
from catalog.models import Product
from loyalty.models import LoyaltyAccount, LoyaltyLedgerEntry, Tier
from offers.models import CampaignBudget, Offer, OfferAssignment, OfferEvent
from recs_analytics.models import RecommendationEvent
from roadmap_app.fragrance_slots import SLOTS as FRAGRANCE_SLOTS, slot_of_fragrance
from roadmap_app.models import RoadmapEvent, RoadmapPlan, RoadmapStep
from transactions.models import CartItem, OwnedProduct, Transaction, TransactionItem, WishlistItem
from users_app.models import CustomerProfile


DEMO_USER_PREFIX = "demo_hist"
DEMO_PASSWORD = "demo12345"
REPORTS_DIR = Path("reports")
COHORT_CODES = {
    "haircare_focused": "hc",
    "skincare_focused": "sc",
    "makeup_focused": "mu",
    "fragrance_focused": "fr",
    "mixed_beauty": "mx",
}
COHORT_WEIGHTS = (
    ("haircare_focused", 0.18),
    ("skincare_focused", 0.22),
    ("makeup_focused", 0.18),
    ("fragrance_focused", 0.18),
    ("mixed_beauty", 0.24),
)
PRIMARY_CATEGORY_BY_COHORT = {
    "haircare_focused": "haircare",
    "skincare_focused": "skincare",
    "makeup_focused": "makeup",
    "fragrance_focused": "fragrance",
    "mixed_beauty": "skincare",
}
FIRST_NAMES = [
    "Amina",
    "Dana",
    "Aruzhan",
    "Mira",
    "Aliya",
    "Kamila",
    "Sofia",
    "Diana",
    "Leila",
    "Asem",
    "Madina",
    "Mariam",
]
LAST_NAMES = [
    "Sarsen",
    "Bektas",
    "Nurgali",
    "Karim",
    "Aman",
    "Tuleu",
    "Iskak",
    "Kairat",
    "Suleimen",
]
CITY_POOL = [
    "Almaty",
    "Astana",
    "Shymkent",
    "Atyrau",
    "Aktobe",
    "Karaganda",
    "Kostanay",
    "Taraz",
]
FRAGRANCE_REQUIRED_TRANSITIONS = (
    ("warm_day", "warm_evening"),
    ("warm_day", "cold_day"),
    ("cold_day", "cold_evening"),
    ("warm_evening", "cold_evening"),
)


@dataclass(frozen=True)
class DemoUserSpec:
    username: str
    cohort: str
    cohort_index: int
    first_name: str
    last_name: str
    city: str
    skin_type: str
    goals: list[str]
    avoid_flags: list[str]
    budget: str
    hair_profile: dict[str, Any]
    makeup_profile: dict[str, Any]
    fragrance_profile: dict[str, Any]


def _stable_rng(seed: int, *parts: Any) -> random.Random:
    acc = int(seed)
    for part in parts:
        for char in str(part):
            acc = (acc * 131 + ord(char)) % 2_147_483_647
    return random.Random(acc)


def parse_yes_no(value: str | bool | None, *, default: bool = False) -> bool:
    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def demo_username(*, prefix: str, seed: int, cohort: str, index: int) -> str:
    code = COHORT_CODES[cohort]
    return f"{prefix}_s{seed}_{code}_{index:03d}"


def candidate_summary(product: Product, *, include_slot: bool = False) -> dict[str, Any]:
    item = {
        "id": int(product.id),
        "name": str(product.name or ""),
        "brand": str(product.brand or ""),
        "category": str(product.category or ""),
        "product_type": str(product.product_type or ""),
        "price": str(product.price) if product.price is not None else None,
        "in_stock": bool(product.in_stock),
    }
    if include_slot and product.category == "fragrance":
        item["slot"] = slot_of_fragrance(product.attrs or {}, raw_meta=product.raw_meta or {})
    return item


def load_catalog_seed_pools(*, include_out_of_stock: bool = False) -> dict[str, Any]:
    product_qs = Product.objects.all()
    if not include_out_of_stock:
        product_qs = product_qs.filter(in_stock=True)
    products = list(
        product_qs.order_by("category", "product_type", "brand", "name", "id").only(
            "id",
            "name",
            "brand",
            "price",
            "currency",
            "category",
            "product_type",
            "in_stock",
            "attrs",
            "raw_meta",
            "supported_skin_types",
            "concerns",
            "flags",
        )
    )
    by_category_type: dict[str, dict[str, list[Product]]] = {
        category: {ptype: [] for ptype in product_types}
        for category, product_types in CURATED_V2_CANONICAL_TYPES.items()
    }
    by_slot: dict[str, list[Product]] = {slot: [] for slot in FRAGRANCE_SLOTS}
    by_category = Counter()
    by_product_type = Counter()
    in_stock = Counter()
    duplicates = Counter()

    for product in products:
        by_category[str(product.category)] += 1
        by_product_type[f"{product.category}:{product.product_type}"] += 1
        in_stock["true" if product.in_stock else "false"] += 1
        if product.category in by_category_type and product.product_type in by_category_type[product.category]:
            by_category_type[product.category][product.product_type].append(product)
        if product.category == "fragrance":
            slot = slot_of_fragrance(product.attrs or {}, raw_meta=product.raw_meta or {})
            by_slot[slot].append(product)
        key = (
            product.brand.strip().lower(),
            product.name.strip().lower(),
            product.product_type.strip().lower(),
        )
        duplicates[key] += 1

    canonical_coverage = {
        category: {ptype: len(by_category_type[category][ptype]) for ptype in product_types}
        for category, product_types in CURATED_V2_CANONICAL_TYPES.items()
    }
    duplicate_groups = [
        {"brand": brand, "name": name, "product_type": product_type, "count": int(count)}
        for (brand, name, product_type), count in sorted(duplicates.items())
        if count > 1
    ]
    sample_candidates = {
        category: {
            ptype: [candidate_summary(product) for product in by_category_type[category][ptype][:5]]
            for ptype in product_types
        }
        for category, product_types in CURATED_V2_CANONICAL_TYPES.items()
    }
    fragrance_slot_samples = {
        slot: [candidate_summary(product, include_slot=True) for product in by_slot[slot][:5]]
        for slot in FRAGRANCE_SLOTS
    }
    return {
        "products": products,
        "by_category_type": by_category_type,
        "by_slot": by_slot,
        "counts": {
            "products_total": len(products),
            "by_category": dict(by_category),
            "by_product_type": dict(by_product_type),
            "in_stock": dict(in_stock),
            "canonical_coverage": canonical_coverage,
            "fragrance_slots": {slot: len(by_slot[slot]) for slot in FRAGRANCE_SLOTS},
            "duplicate_groups_count": len(duplicate_groups),
        },
        "sample_candidates": sample_candidates,
        "fragrance_slot_samples": fragrance_slot_samples,
        "duplicate_groups": duplicate_groups,
    }


def build_demo_seed_catalog_coverage_report() -> dict[str, Any]:
    pools = load_catalog_seed_pools(include_out_of_stock=False)
    counts = pools["counts"]
    missing_types = []
    for category, product_types in CURATED_V2_CANONICAL_TYPES.items():
        for product_type in product_types:
            if counts["canonical_coverage"][category][product_type] <= 0:
                missing_types.append({"category": category, "product_type": product_type})
    usable_seed_pools = {
        category: {
            product_type: {
                "count": int(counts["canonical_coverage"][category][product_type]),
                "candidates": pools["sample_candidates"][category][product_type],
            }
            for product_type in product_types
        }
        for category, product_types in CURATED_V2_CANONICAL_TYPES.items()
    }
    fragrance_slots = {
        slot: {
            "count": int(counts["fragrance_slots"][slot]),
            "candidates": pools["fragrance_slot_samples"][slot],
        }
        for slot in FRAGRANCE_SLOTS
    }
    return {
        "generated_at": timezone.now().isoformat(),
        "counts": counts,
        "missing_types": missing_types,
        "usable_seed_pools": usable_seed_pools,
        "fragrance_slot_seed_pools": fragrance_slots,
        "duplicate_groups": pools["duplicate_groups"],
        "ready_for_demo_seeding": not missing_types,
    }


def build_demo_seed_catalog_coverage_md(report: dict[str, Any]) -> str:
    counts = report["counts"]
    lines = [
        "# Demo Seed Catalog Coverage",
        "",
        f"- products_total: **{counts['products_total']}**",
        f"- by_category: `{json.dumps(counts['by_category'], ensure_ascii=False, default=json_default)}`",
        f"- in_stock: `{json.dumps(counts['in_stock'], ensure_ascii=False, default=json_default)}`",
        f"- fragrance_slots: `{json.dumps(counts['fragrance_slots'], ensure_ascii=False, default=json_default)}`",
        f"- duplicate_groups_count: **{counts['duplicate_groups_count']}**",
        "",
        "## Canonical Coverage",
    ]
    for category, mapping in counts["canonical_coverage"].items():
        lines.append(f"- {category}: `{json.dumps(mapping, ensure_ascii=False, default=json_default)}`")
    lines.extend(["", "## Seed Pools"])
    for category, mapping in report["usable_seed_pools"].items():
        lines.append(f"### {category}")
        for product_type, bucket in mapping.items():
            lines.append(f"- {product_type}: count={bucket['count']}")
    lines.extend(["", "## Fragrance Slots"])
    for slot, bucket in report["fragrance_slot_seed_pools"].items():
        lines.append(f"- {slot}: count={bucket['count']}")
    if report["missing_types"]:
        lines.extend(["", "## Missing Types"])
        for item in report["missing_types"]:
            lines.append(f"- {item['category']}:{item['product_type']}")
    return "\n".join(lines) + "\n"


def ensure_demo_offers() -> None:
    default_campaign, _ = CampaignBudget.objects.get_or_create(
        name="default",
        defaults={
            "weekly_limit": Decimal("1000.00"),
            "weekly_spent": Decimal("0.00"),
            "priority": 100,
            "is_active": True,
        },
    )
    skincare_campaign, _ = CampaignBudget.objects.get_or_create(
        name="skincare_retention",
        defaults={
            "weekly_limit": Decimal("1000.00"),
            "weekly_spent": Decimal("0.00"),
            "priority": 60,
            "is_active": True,
            "allowed_categories": ["skincare"],
        },
    )
    makeup_campaign, _ = CampaignBudget.objects.get_or_create(
        name="makeup_push",
        defaults={
            "weekly_limit": Decimal("1000.00"),
            "weekly_spent": Decimal("0.00"),
            "priority": 70,
            "is_active": True,
            "allowed_categories": ["makeup"],
        },
    )
    fragrance_campaign, _ = CampaignBudget.objects.get_or_create(
        name="fragrance_crosssell",
        defaults={
            "weekly_limit": Decimal("1000.00"),
            "weekly_spent": Decimal("0.00"),
            "priority": 50,
            "is_active": True,
            "allowed_categories": ["fragrance"],
        },
    )

    def _ensure_offer(
        *,
        name: str,
        campaign: CampaignBudget,
        allowed_categories: list[str],
        allowed_product_types: list[str],
        target_scope: str,
    ) -> None:
        Offer.objects.get_or_create(
            name=name,
            offer_type=Offer.Type.DISCOUNT,
            defaults={
                "value": Decimal("10.00"),
                "estimated_cost": Decimal("4.00"),
                "is_active": True,
                "target_scope": target_scope,
                "cooldown_days": 0,
                "expires_in_days": 7,
                "allowed_categories": allowed_categories,
                "allowed_product_types": allowed_product_types,
                "campaign": campaign,
            },
        )

    _ensure_offer(
        name="CuratedV2 Haircare Next Step",
        campaign=default_campaign,
        allowed_categories=["haircare"],
        allowed_product_types=list(CURATED_V2_CANONICAL_TYPES["haircare"]),
        target_scope="product_type",
    )
    _ensure_offer(
        name="CuratedV2 Skincare Next Step",
        campaign=skincare_campaign,
        allowed_categories=["skincare"],
        allowed_product_types=list(CURATED_V2_CANONICAL_TYPES["skincare"]),
        target_scope="product_type",
    )
    _ensure_offer(
        name="CuratedV2 Makeup Next Step",
        campaign=makeup_campaign,
        allowed_categories=["makeup"],
        allowed_product_types=list(CURATED_V2_CANONICAL_TYPES["makeup"]),
        target_scope="product_type",
    )
    _ensure_offer(
        name="CuratedV2 Fragrance Slot",
        campaign=fragrance_campaign,
        allowed_categories=["fragrance"],
        allowed_product_types=list(CURATED_V2_CANONICAL_TYPES["fragrance"]),
        target_scope="product_id",
    )


def base_tier() -> Tier:
    tier, _ = Tier.objects.get_or_create(
        name="Bronze",
        defaults={"threshold_spend_90d": Decimal("0.00"), "points_rate": Decimal("0.10")},
    )
    if tier.threshold_spend_90d != Decimal("0.00") or tier.points_rate != Decimal("0.10"):
        tier.threshold_spend_90d = Decimal("0.00")
        tier.points_rate = Decimal("0.10")
        tier.save(update_fields=["threshold_spend_90d", "points_rate"])
    return tier


def ensure_loyalty_account(user) -> LoyaltyAccount:
    bronze = base_tier()
    account, _ = LoyaltyAccount.objects.get_or_create(
        user=user,
        defaults={"tier": bronze, "points_balance": 0},
    )
    if account.tier_id is None:
        account.tier = bronze
        account.save(update_fields=["tier"])
    return account


def _allocate_cohorts(total_users: int) -> dict[str, int]:
    raw = []
    assigned = 0
    for cohort, weight in COHORT_WEIGHTS:
        exact = total_users * weight
        base = int(exact)
        raw.append([cohort, base, exact - base])
        assigned += base
    remaining = total_users - assigned
    raw.sort(key=lambda item: (-item[2], item[0]))
    for idx in range(remaining):
        raw[idx][1] += 1
    return {cohort: count for cohort, count, _ in raw}


def _profile_variant_for_cohort(cohort: str, cohort_index: int, seed: int) -> dict[str, Any]:
    rng = _stable_rng(seed, cohort, cohort_index)
    if cohort == "haircare_focused":
        variants = [
            {
                "skin_type": "normal",
                "goals": ["volume", "scalp_balance"],
                "avoid_flags": ["heavy_oils"],
                "budget": "medium",
                "hair_profile": {
                    "hair_type": "straight",
                    "scalp_type": "oily",
                    "hair_thickness": "fine",
                    "concerns": ["oiliness", "volume"],
                },
                "makeup_profile": {"finish_pref": ["natural"], "coverage_pref": ["light"]},
                "fragrance_profile": {"liked_families": ["citrus"], "liked_notes": ["bergamot"], "intensity_pref": "soft"},
            },
            {
                "skin_type": "dry",
                "goals": ["repair", "frizz_control"],
                "avoid_flags": [],
                "budget": "high",
                "hair_profile": {
                    "hair_type": "wavy",
                    "scalp_type": "dry",
                    "hair_thickness": "medium",
                    "concerns": ["frizz", "repair"],
                },
                "makeup_profile": {"finish_pref": ["dewy"], "coverage_pref": ["medium"]},
                "fragrance_profile": {"liked_families": ["floral"], "liked_notes": ["rose"], "intensity_pref": "medium"},
            },
            {
                "skin_type": "sensitive",
                "goals": ["definition", "repair"],
                "avoid_flags": ["alcohol"],
                "budget": "medium",
                "hair_profile": {
                    "hair_type": "curly",
                    "scalp_type": "sensitive",
                    "hair_thickness": "thick",
                    "concerns": ["definition", "repair"],
                },
                "makeup_profile": {"finish_pref": ["natural"], "coverage_pref": ["medium"]},
                "fragrance_profile": {"liked_families": ["amber"], "liked_notes": ["vanilla"], "intensity_pref": "strong"},
            },
        ]
        return dict(variants[cohort_index % len(variants)])
    if cohort == "skincare_focused":
        variants = [
            {
                "skin_type": "oily",
                "goals": ["acne", "oil_control", "clarity"],
                "avoid_flags": ["heavy_oils"],
                "budget": "medium",
                "hair_profile": {"hair_type": "straight", "scalp_type": "normal", "hair_thickness": "medium", "concerns": []},
                "makeup_profile": {"finish_pref": ["matte"], "coverage_pref": ["medium"]},
                "fragrance_profile": {"liked_families": ["green", "citrus"], "liked_notes": ["lime"], "intensity_pref": "soft"},
            },
            {
                "skin_type": "dry",
                "goals": ["hydration", "repair", "brightening"],
                "avoid_flags": [],
                "budget": "high",
                "hair_profile": {"hair_type": "wavy", "scalp_type": "dry", "hair_thickness": "medium", "concerns": ["dryness"]},
                "makeup_profile": {"finish_pref": ["dewy"], "coverage_pref": ["light"]},
                "fragrance_profile": {"liked_families": ["floral"], "liked_notes": ["peony"], "intensity_pref": "medium"},
            },
            {
                "skin_type": "sensitive",
                "goals": ["calming", "repair", "hydration"],
                "avoid_flags": ["fragrance", "alcohol"],
                "budget": "medium",
                "hair_profile": {"hair_type": "straight", "scalp_type": "sensitive", "hair_thickness": "fine", "concerns": ["sensitivity"]},
                "makeup_profile": {"finish_pref": ["natural"], "coverage_pref": ["light"]},
                "fragrance_profile": {"liked_families": ["musk"], "liked_notes": ["white musk"], "intensity_pref": "soft"},
            },
            {
                "skin_type": "combination",
                "goals": ["brightening", "hydration"],
                "avoid_flags": [],
                "budget": "low",
                "hair_profile": {"hair_type": "wavy", "scalp_type": "normal", "hair_thickness": "medium", "concerns": []},
                "makeup_profile": {"finish_pref": ["natural"], "coverage_pref": ["medium"]},
                "fragrance_profile": {"liked_families": ["woody"], "liked_notes": ["cedar"], "intensity_pref": "medium"},
            },
        ]
        return dict(variants[cohort_index % len(variants)])
    if cohort == "makeup_focused":
        variants = [
            {
                "skin_type": "normal",
                "goals": ["even_tone"],
                "avoid_flags": [],
                "budget": "medium",
                "hair_profile": {"hair_type": "straight", "scalp_type": "normal", "hair_thickness": "medium", "concerns": []},
                "makeup_profile": {
                    "finish_pref": ["natural"],
                    "coverage_pref": ["medium"],
                    "undertone": "warm",
                    "tone_family": "medium",
                    "preferred_actions": ["foundation", "blush", "mascara", "lipstick"],
                },
                "fragrance_profile": {"liked_families": ["citrus"], "liked_notes": ["orange blossom"], "intensity_pref": "soft"},
            },
            {
                "skin_type": "dry",
                "goals": ["radiance"],
                "avoid_flags": [],
                "budget": "high",
                "hair_profile": {"hair_type": "wavy", "scalp_type": "dry", "hair_thickness": "medium", "concerns": []},
                "makeup_profile": {
                    "finish_pref": ["dewy"],
                    "coverage_pref": ["light"],
                    "undertone": "neutral",
                    "tone_family": "light",
                    "preferred_actions": ["foundation", "blush", "lipstick"],
                },
                "fragrance_profile": {"liked_families": ["floral"], "liked_notes": ["jasmine"], "intensity_pref": "medium"},
            },
            {
                "skin_type": "oily",
                "goals": ["longwear"],
                "avoid_flags": [],
                "budget": "medium",
                "hair_profile": {"hair_type": "straight", "scalp_type": "oily", "hair_thickness": "fine", "concerns": []},
                "makeup_profile": {
                    "finish_pref": ["matte"],
                    "coverage_pref": ["full"],
                    "undertone": "cool",
                    "tone_family": "fair",
                    "preferred_actions": ["foundation", "mascara", "eyeshadow", "primer", "setting_spray"],
                },
                "fragrance_profile": {"liked_families": ["amber"], "liked_notes": ["tonka"], "intensity_pref": "strong"},
            },
        ]
        return dict(variants[cohort_index % len(variants)])
    if cohort == "fragrance_focused":
        variants = [
            {
                "skin_type": "normal",
                "goals": ["signature_scent"],
                "avoid_flags": [],
                "budget": "high",
                "hair_profile": {"hair_type": "wavy", "scalp_type": "normal", "hair_thickness": "medium", "concerns": []},
                "makeup_profile": {"finish_pref": ["natural"], "coverage_pref": ["light"]},
                "fragrance_profile": {"liked_families": ["citrus", "green"], "liked_notes": ["bergamot", "neroli"], "intensity_pref": "soft"},
            },
            {
                "skin_type": "normal",
                "goals": ["signature_scent"],
                "avoid_flags": [],
                "budget": "medium",
                "hair_profile": {"hair_type": "straight", "scalp_type": "normal", "hair_thickness": "fine", "concerns": []},
                "makeup_profile": {"finish_pref": ["natural"], "coverage_pref": ["medium"]},
                "fragrance_profile": {"liked_families": ["woody", "aromatic"], "liked_notes": ["cedar", "sage"], "intensity_pref": "medium"},
            },
            {
                "skin_type": "normal",
                "goals": ["signature_scent"],
                "avoid_flags": [],
                "budget": "high",
                "hair_profile": {"hair_type": "curly", "scalp_type": "normal", "hair_thickness": "thick", "concerns": []},
                "makeup_profile": {"finish_pref": ["glow"], "coverage_pref": ["medium"]},
                "fragrance_profile": {"liked_families": ["amber", "gourmand"], "liked_notes": ["vanilla", "amber"], "intensity_pref": "strong"},
            },
            {
                "skin_type": "normal",
                "goals": ["signature_scent"],
                "avoid_flags": [],
                "budget": "medium",
                "hair_profile": {"hair_type": "wavy", "scalp_type": "normal", "hair_thickness": "medium", "concerns": []},
                "makeup_profile": {"finish_pref": ["natural"], "coverage_pref": ["light"]},
                "fragrance_profile": {"liked_families": ["floral", "musk"], "liked_notes": ["rose", "musk"], "intensity_pref": "medium"},
            },
        ]
        return dict(variants[cohort_index % len(variants)])
    variants = [
        {
            "skin_type": "combination",
            "goals": ["hydration", "even_tone", "definition"],
            "avoid_flags": [],
            "budget": rng.choice(["medium", "high"]),
            "hair_profile": {"hair_type": "wavy", "scalp_type": "normal", "hair_thickness": "medium", "concerns": ["definition"]},
            "makeup_profile": {"finish_pref": ["natural"], "coverage_pref": ["medium"], "undertone": "warm", "tone_family": "medium"},
            "fragrance_profile": {"liked_families": ["citrus", "woody"], "liked_notes": ["bergamot", "cedar"], "intensity_pref": "medium"},
        },
        {
            "skin_type": "dry",
            "goals": ["hydration", "repair", "radiance"],
            "avoid_flags": [],
            "budget": rng.choice(["medium", "high"]),
            "hair_profile": {"hair_type": "curly", "scalp_type": "dry", "hair_thickness": "thick", "concerns": ["repair", "definition"]},
            "makeup_profile": {"finish_pref": ["dewy"], "coverage_pref": ["light"], "undertone": "neutral", "tone_family": "light"},
            "fragrance_profile": {"liked_families": ["floral", "amber"], "liked_notes": ["rose", "vanilla"], "intensity_pref": "medium"},
        },
    ]
    return dict(variants[cohort_index % len(variants)])


def generate_demo_user_specs(*, total_users: int, seed: int, prefix: str = DEMO_USER_PREFIX) -> list[DemoUserSpec]:
    if total_users <= 0:
        return []
    allocations = _allocate_cohorts(total_users)
    specs: list[DemoUserSpec] = []
    global_index = 0
    for cohort in [item[0] for item in COHORT_WEIGHTS]:
        for cohort_index in range(allocations.get(cohort, 0)):
            first_name = FIRST_NAMES[(global_index + cohort_index) % len(FIRST_NAMES)]
            last_name = LAST_NAMES[(global_index * 3 + cohort_index) % len(LAST_NAMES)]
            city = CITY_POOL[(global_index * 5 + cohort_index) % len(CITY_POOL)]
            variant = _profile_variant_for_cohort(cohort, cohort_index, seed)
            specs.append(
                DemoUserSpec(
                    username=demo_username(prefix=prefix, seed=seed, cohort=cohort, index=cohort_index + 1),
                    cohort=cohort,
                    cohort_index=cohort_index + 1,
                    first_name=first_name,
                    last_name=last_name,
                    city=city,
                    skin_type=str(variant["skin_type"]),
                    goals=list(variant["goals"]),
                    avoid_flags=list(variant["avoid_flags"]),
                    budget=str(variant["budget"]),
                    hair_profile=dict(variant["hair_profile"]),
                    makeup_profile=dict(variant["makeup_profile"]),
                    fragrance_profile=dict(variant["fragrance_profile"]),
                )
            )
            global_index += 1
    specs.sort(key=lambda item: item.username)
    return specs


def reset_demo_users_and_history(*, prefix: str = DEMO_USER_PREFIX) -> dict[str, int]:
    User = get_user_model()
    demo_users = User.objects.filter(username__startswith=f"{prefix}_", is_staff=False, is_superuser=False)
    counts = {"demo_users": demo_users.count()}
    demo_users.delete()
    return counts


def seed_demo_users(*, seed: int, total_users: int, prefix: str = DEMO_USER_PREFIX) -> dict[str, Any]:
    User = get_user_model()
    deleted = reset_demo_users_and_history(prefix=prefix)
    bronze = base_tier()
    specs = generate_demo_user_specs(total_users=total_users, seed=seed, prefix=prefix)
    created_counts = Counter()
    now = timezone.now()
    for spec in specs:
        user = User.objects.create_user(
            username=spec.username,
            password=DEMO_PASSWORD,
            email=f"{spec.username}@example.test",
        )
        profile, _ = CustomerProfile.objects.get_or_create(user=user)
        profile.first_name = spec.first_name
        profile.last_name = spec.last_name
        profile.city = spec.city
        profile.skin_type = spec.skin_type
        profile.goals = list(spec.goals)
        profile.avoid_flags = list(spec.avoid_flags)
        profile.budget = spec.budget
        profile.hair_profile = dict(spec.hair_profile)
        profile.makeup_profile = dict(spec.makeup_profile)
        profile.fragrance_profile = dict(spec.fragrance_profile)
        profile.profile_completed_at = now
        profile.save()
        LoyaltyAccount.objects.get_or_create(user=user, defaults={"tier": bronze, "points_balance": 0})
        created_counts[spec.cohort] += 1
    return {
        "seed": int(seed),
        "users_requested": int(total_users),
        "users_created": int(len(specs)),
        "deleted_previous_demo_users": int(deleted["demo_users"]),
        "cohorts": {key: int(value) for key, value in created_counts.items()},
        "username_prefix": prefix,
        "password": DEMO_PASSWORD,
    }


def get_demo_users(*, prefix: str = DEMO_USER_PREFIX, seed: int | None = None, limit: int | None = None) -> list[Any]:
    qs = get_user_model().objects.filter(
        username__startswith=f"{prefix}_",
        is_staff=False,
        is_superuser=False,
    ).order_by("username")
    if seed is not None:
        qs = qs.filter(username__contains=f"_s{seed}_")
    if limit is not None:
        return list(qs[:limit])
    return list(qs)


def _profile_for_user(user) -> CustomerProfile:
    return CustomerProfile.objects.get(user=user)


def _product_tokens(product: Product) -> set[str]:
    tokens = set()
    for item in product.concerns or []:
        token = str(item or "").strip().lower()
        if token:
            tokens.add(token)
    for item in product.flags or []:
        token = str(item or "").strip().lower()
        if token:
            tokens.add(token)
    attrs = product.attrs or {}
    for key in (
        "hair_type",
        "scalp_type",
        "hair_thickness",
        "finish",
        "coverage",
        "effect",
        "tone_family",
        "undertone",
        "shade_family",
        "scent_family",
        "intensity",
        "area",
    ):
        value = attrs.get(key)
        if isinstance(value, str) and value.strip():
            tokens.add(value.strip().lower())
    return tokens


def _score_product_for_profile(product: Product, *, category: str, target: str, profile: CustomerProfile) -> int:
    attrs = product.attrs or {}
    score = 0
    if category == "haircare":
        hair = profile.hair_profile or {}
        if str(attrs.get("hair_type") or "").strip().lower() == str(hair.get("hair_type") or "").strip().lower():
            score += 3
        if str(attrs.get("scalp_type") or "").strip().lower() == str(hair.get("scalp_type") or "").strip().lower():
            score += 3
        if str(attrs.get("hair_thickness") or "").strip().lower() == str(hair.get("hair_thickness") or "").strip().lower():
            score += 2
        concerns = {str(x).strip().lower() for x in (hair.get("concerns") or [])}
        if concerns.intersection(_product_tokens(product)):
            score += 2
    elif category == "skincare":
        if profile.skin_type and profile.skin_type in set(product.supported_skin_types or []):
            score += 3
        goals = {str(x).strip().lower() for x in (profile.goals or [])}
        if goals.intersection(_product_tokens(product)):
            score += 2
        avoid = {str(x).strip().lower() for x in (profile.avoid_flags or [])}
        if avoid.intersection(_product_tokens(product)):
            score -= 4
    elif category == "makeup":
        makeup = profile.makeup_profile or {}
        if str(attrs.get("finish") or "").strip().lower() in {
            str(x).strip().lower() for x in (makeup.get("finish_pref") or [])
        }:
            score += 3
        if str(attrs.get("coverage") or "").strip().lower() in {
            str(x).strip().lower() for x in (makeup.get("coverage_pref") or [])
        }:
            score += 2
        if str(attrs.get("undertone") or "").strip().lower() == str(makeup.get("undertone") or "").strip().lower():
            score += 1
        if str(attrs.get("tone_family") or "").strip().lower() == str(makeup.get("tone_family") or "").strip().lower():
            score += 1
    elif category == "fragrance":
        slot = slot_of_fragrance(attrs, raw_meta=product.raw_meta or {})
        if slot == target:
            score += 5
        fragrance = profile.fragrance_profile or {}
        liked_families = {str(x).strip().lower() for x in (fragrance.get("liked_families") or [])}
        liked_notes = {str(x).strip().lower() for x in (fragrance.get("liked_notes") or [])}
        preferred_intensity = str(fragrance.get("intensity_pref") or "").strip().lower()
        family = str(attrs.get("scent_family") or "").strip().lower()
        if family and family in liked_families:
            score += 3
        notes = {str(x).strip().lower() for x in (attrs.get("notes") or [])}
        if notes.intersection(liked_notes):
            score += 2
        if preferred_intensity and str(attrs.get("intensity") or "").strip().lower() == preferred_intensity:
            score += 2
    return score


def pick_product_for_target(
    *,
    pools: dict[str, Any],
    category: str,
    target: str,
    profile: CustomerProfile,
    seed: int,
    user_key: str,
    step_index: int,
    exclude_product_ids: set[int] | None = None,
) -> Product:
    exclude_product_ids = exclude_product_ids or set()
    candidates = list(
        pools["by_slot"].get(target) or []
        if category == "fragrance"
        else (pools["by_category_type"].get(category) or {}).get(target) or []
    )
    if not candidates:
        raise CommandError(f"No in-stock seed candidates for {category}:{target}")

    ranked: list[tuple[int, int, int, Product]] = []
    for product in candidates:
        penalty = 10 if product.id in exclude_product_ids else 0
        score = _score_product_for_profile(product, category=category, target=target, profile=profile)
        tie = _stable_rng(seed, user_key, category, target, step_index, product.id).randint(0, 1_000_000)
        ranked.append((penalty, -score, tie, product))
    ranked.sort(key=lambda item: (item[0], item[1], item[2], item[3].id))
    return ranked[0][3]


def _haircare_chain(profile: CustomerProfile, cohort_index: int) -> list[str]:
    hair = profile.hair_profile or {}
    concerns = {str(x).strip().lower() for x in (hair.get("concerns") or [])}
    hair_type = str(hair.get("hair_type") or "").strip().lower()
    scalp_type = str(hair.get("scalp_type") or "").strip().lower()
    thickness = str(hair.get("hair_thickness") or "").strip().lower()
    chain = ["shampoo", "conditioner"]
    if scalp_type in {"oily", "sensitive"} or "scalp_balance" in set(profile.goals or []):
        chain.append("scalp_serum")
    if (
        hair_type in {"curly", "coily", "wavy"}
        or thickness == "fine"
        or "volume" in set(profile.goals or [])
        or bool(concerns & {"definition", "frizz", "frizz_control"})
    ):
        chain.append("leave_in")
    chain.append("hair_mask")
    if thickness in {"medium", "thick"} or bool(concerns & {"repair", "dryness"}):
        chain.append("hair_oil")
    return _dedupe_preserve(chain)


def _skincare_chain(profile: CustomerProfile, cohort_index: int) -> list[str]:
    goals = {str(x).strip().lower() for x in (profile.goals or [])}
    chain = ["cleanser", "serum", "moisturizer", "spf"]
    if profile.skin_type in {"oily", "combination"} or bool(goals & {"acne", "oil_control", "clarity"}):
        chain.extend(["toner", "mask"])
    if profile.skin_type in {"dry", "sensitive"} or bool(goals & {"hydration", "repair", "calming"}):
        chain.extend(["essence", "eye_cream"])
    if cohort_index % 3 == 0:
        chain.append("mask")
    return _dedupe_preserve(chain)


def _makeup_chain(profile: CustomerProfile, cohort_index: int) -> list[str]:
    makeup = profile.makeup_profile or {}
    finish = {str(x).strip().lower() for x in (makeup.get("finish_pref") or [])}
    coverage = {str(x).strip().lower() for x in (makeup.get("coverage_pref") or [])}
    preferred = [str(x).strip().lower() for x in (makeup.get("preferred_actions") or []) if str(x).strip()]
    chain = ["foundation"]
    if "primer" in preferred or bool(finish & {"matte"}) or bool(coverage & {"full"}):
        chain.append("primer")
    chain.extend(["mascara", "blush"])
    if "eyeshadow" in preferred or cohort_index % 2 == 0:
        chain.append("eyeshadow")
    if "lipstick" in preferred or cohort_index % 3 != 0:
        chain.append("lipstick")
    if "setting_spray" in preferred or bool(coverage & {"full"}):
        chain.append("setting_spray")
    return _dedupe_preserve(chain)


def _fragrance_slot_chain(profile: CustomerProfile, cohort_index: int) -> list[str]:
    fragrance = profile.fragrance_profile or {}
    liked_families = {str(x).strip().lower() for x in (fragrance.get("liked_families") or [])}
    intensity = str(fragrance.get("intensity_pref") or "").strip().lower()
    transition_templates = [
        ["warm_day", "warm_evening", "cold_evening"],
        ["warm_day", "cold_day", "cold_evening"],
        ["cold_day", "cold_evening"],
        ["warm_evening", "cold_evening"],
        ["warm_day", "warm_evening", "cold_day", "cold_evening"],
    ]
    chain = list(transition_templates[cohort_index % len(transition_templates)])
    if intensity == "soft" and chain[0] not in {"warm_day", "cold_day"}:
        chain[0] = "warm_day"
    if bool(liked_families & {"green", "citrus"}) and chain[0] == "warm_evening":
        chain[0] = "warm_day"
    if bool(liked_families & {"amber", "gourmand"}) and chain[-1] != "cold_evening":
        chain.append("cold_evening")
    return _dedupe_preserve(chain)


def _mixed_plan_categories(spec: DemoUserSpec) -> list[str]:
    options = [
        ["skincare", "makeup", "fragrance"],
        ["haircare", "skincare", "fragrance"],
        ["skincare", "haircare", "makeup"],
    ]
    return options[(int(spec.cohort_index) - 1) % len(options)]


def _dedupe_preserve(values: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        token = str(value or "").strip()
        if not token or token in seen:
            continue
        out.append(token)
        seen.add(token)
    return out


def _primary_warmup_count(spec: DemoUserSpec, primary_targets: list[str]) -> int:
    if len(primary_targets) <= 1:
        return 0
    variant = max(0, int(spec.cohort_index) - 1)
    if spec.cohort == "haircare_focused":
        warmup = [0, 1, 2][variant % 3]
    elif spec.cohort == "skincare_focused":
        warmup = [0, 1, 2, 1][variant % 4]
    elif spec.cohort == "makeup_focused":
        warmup = [0, 1, 1][variant % 3]
    elif spec.cohort == "fragrance_focused":
        warmup = [0, 1, 2, 1][variant % 4]
    else:
        warmup = 1 if variant % 2 == 0 else 0
    return max(0, min(int(warmup), len(primary_targets) - 1))


def build_user_purchase_plan(spec: DemoUserSpec, *, max_transactions: int) -> dict[str, Any]:
    profile_stub = type("ProfileStub", (), {})()
    profile_stub.skin_type = spec.skin_type
    profile_stub.goals = spec.goals
    profile_stub.avoid_flags = spec.avoid_flags
    profile_stub.budget = spec.budget
    profile_stub.hair_profile = spec.hair_profile
    profile_stub.makeup_profile = spec.makeup_profile
    profile_stub.fragrance_profile = spec.fragrance_profile

    category_sequences: dict[str, list[str]] = {}
    if spec.cohort == "haircare_focused":
        category_sequences["haircare"] = _haircare_chain(profile_stub, spec.cohort_index)
        category_sequences["skincare"] = ["cleanser", "spf"]
    elif spec.cohort == "skincare_focused":
        category_sequences["skincare"] = _skincare_chain(profile_stub, spec.cohort_index)
        category_sequences["makeup"] = ["foundation"]
    elif spec.cohort == "makeup_focused":
        category_sequences["makeup"] = _makeup_chain(profile_stub, spec.cohort_index)
        category_sequences["skincare"] = ["cleanser", "spf"]
    elif spec.cohort == "fragrance_focused":
        category_sequences["fragrance"] = _fragrance_slot_chain(profile_stub, spec.cohort_index)
        category_sequences["makeup"] = ["lipstick"]
    else:
        for category in _mixed_plan_categories(spec):
            if category == "haircare":
                category_sequences[category] = ["shampoo", "conditioner", "hair_mask"]
            elif category == "skincare":
                category_sequences[category] = ["cleanser", "serum", "moisturizer", "spf"]
            elif category == "makeup":
                category_sequences[category] = ["foundation", "blush", "lipstick"]
            elif category == "fragrance":
                category_sequences[category] = _fragrance_slot_chain(profile_stub, spec.cohort_index)[:2]

    primary_category = PRIMARY_CATEGORY_BY_COHORT[spec.cohort]
    primary_targets = list(category_sequences.get(primary_category) or [])
    warmup_count = _primary_warmup_count(spec, primary_targets)

    background_direct: list[dict[str, Any]] = []
    for category in [cat for cat in category_sequences.keys() if cat != primary_category]:
        for target in category_sequences[category]:
            background_direct.append(
                {
                    "category": category,
                    "target": target,
                    "is_fragrance_slot": category == "fragrance",
                    "mode": "direct",
                    "allow_skip_to_target": False,
                    "scenario_role": "background_direct",
                }
            )

    primary_direct = [
        {
            "category": primary_category,
            "target": target,
            "is_fragrance_slot": primary_category == "fragrance",
            "mode": "direct",
            "allow_skip_to_target": False,
            "scenario_role": "primary_warmup",
        }
        for target in primary_targets[:warmup_count]
    ]
    primary_live = [
        {
            "category": primary_category,
            "target": target,
            "is_fragrance_slot": primary_category == "fragrance",
            "mode": "live",
            "allow_skip_to_target": True,
            "scenario_role": "primary_live",
        }
        for target in primary_targets[warmup_count:]
    ]

    core_items = primary_direct + primary_live
    allowed_background = max(0, int(max_transactions) - len(core_items))
    ordered: list[dict[str, Any]] = background_direct[:allowed_background] + core_items
    if len(ordered) > max_transactions:
        ordered = ordered[:max_transactions]
    return {"primary_category": primary_category, "planned_transactions": ordered}


def _clear_demo_history_for_users(users: list[Any]) -> dict[str, int]:
    user_ids = [int(user.id) for user in users]
    if not user_ids:
        return {}
    account_ids = list(LoyaltyAccount.objects.filter(user_id__in=user_ids).values_list("id", flat=True))
    deleted = {
        "cart_items": CartItem.objects.filter(user_id__in=user_ids).count(),
        "wishlist_items": WishlistItem.objects.filter(user_id__in=user_ids).count(),
        "recommendation_events": RecommendationEvent.objects.filter(user_id__in=user_ids).count(),
        "offer_events": OfferEvent.objects.filter(user_id__in=user_ids).count(),
        "offer_assignments": OfferAssignment.objects.filter(user_id__in=user_ids).count(),
        "roadmap_events": RoadmapEvent.objects.filter(user_id__in=user_ids).count(),
        "roadmap_plans": RoadmapPlan.objects.filter(user_id__in=user_ids).count(),
        "roadmap_steps": RoadmapStep.objects.filter(plan__user_id__in=user_ids).count(),
        "transactions": Transaction.objects.filter(user_id__in=user_ids).count(),
        "transaction_items": TransactionItem.objects.filter(transaction__user_id__in=user_ids).count(),
        "owned_products": OwnedProduct.objects.filter(user_id__in=user_ids).count(),
        "loyalty_ledger": LoyaltyLedgerEntry.objects.filter(account_id__in=account_ids).count(),
    }
    CartItem.objects.filter(user_id__in=user_ids).delete()
    WishlistItem.objects.filter(user_id__in=user_ids).delete()
    RecommendationEvent.objects.filter(user_id__in=user_ids).delete()
    OfferEvent.objects.filter(user_id__in=user_ids).delete()
    OfferAssignment.objects.filter(user_id__in=user_ids).delete()
    RoadmapEvent.objects.filter(user_id__in=user_ids).delete()
    RoadmapPlan.objects.filter(user_id__in=user_ids).delete()
    Transaction.objects.filter(user_id__in=user_ids).delete()
    OwnedProduct.objects.filter(user_id__in=user_ids).delete()
    LoyaltyLedgerEntry.objects.filter(account_id__in=account_ids).delete()
    bronze = base_tier()
    LoyaltyAccount.objects.filter(user_id__in=user_ids).update(points_balance=0, tier=bronze)
    return deleted


def _spaced_datetimes(
    *,
    seed: int,
    label: str,
    count: int,
    start_dt: datetime,
    end_dt: datetime,
) -> list[datetime]:
    if count <= 0:
        return []
    if count == 1 or end_dt <= start_dt:
        return [start_dt + timedelta(minutes=_stable_rng(seed, label, 0).randint(0, 50))]

    total_seconds = max(1, int((end_dt - start_dt).total_seconds()))
    step_seconds = max(1, total_seconds // max(count - 1, 1))
    out: list[datetime] = []
    for idx in range(count):
        jitter = min(1800, max(0, step_seconds // 8))
        offset = _stable_rng(seed, label, idx).randint(0, jitter) if jitter else 0
        candidate = start_dt + timedelta(seconds=step_seconds * idx + offset)
        if out and candidate <= out[-1]:
            candidate = out[-1] + timedelta(minutes=15)
        if candidate > end_dt:
            candidate = end_dt if not out else max(end_dt, out[-1] + timedelta(minutes=1))
        out.append(candidate)
    return out


def _scheduled_transaction_times(*, seed: int, planned: list[dict[str, Any]], days_span: int) -> list[datetime]:
    now = timezone.now().replace(hour=12, minute=0, second=0, microsecond=0)
    direct_count = sum(1 for item in planned if str(item.get("mode") or "direct") == "direct")
    live_count = sum(1 for item in planned if str(item.get("mode") or "") == "live")

    live_end = now - timedelta(days=7)
    live_start = live_end - timedelta(days=max(0, 2 * max(live_count - 1, 0) + 2))
    direct_end = live_start - timedelta(days=10)
    direct_start = now - timedelta(days=max(days_span, 45))

    direct_times = _spaced_datetimes(
        seed=seed,
        label="direct",
        count=direct_count,
        start_dt=direct_start,
        end_dt=direct_end,
    )
    live_times = _spaced_datetimes(
        seed=seed,
        label="live",
        count=live_count,
        start_dt=live_start,
        end_dt=live_end,
    )

    scheduled: list[datetime] = []
    direct_idx = 0
    live_idx = 0
    for item in planned:
        if str(item.get("mode") or "direct") == "live":
            scheduled.append(live_times[live_idx])
            live_idx += 1
        else:
            scheduled.append(direct_times[direct_idx])
            direct_idx += 1
    return scheduled


def _create_direct_transaction(
    *,
    user,
    product: Product,
    when: datetime,
    category: str,
    target: str,
    scenario_meta: dict[str, Any],
) -> Transaction:
    txn = Transaction.objects.create(
        user=user,
        total_amount=product.price or Decimal("0.00"),
        channel="demo_seed",
        pricing_meta={
            "demo_seed": True,
            "mode": "direct_orm",
            "category": category,
            "target": target,
            "scenario": scenario_meta,
        },
    )
    TransactionItem.objects.create(
        transaction=txn,
        product=product,
        quantity=1,
        unit_price=product.price or Decimal("0.00"),
    )
    Transaction.objects.filter(id=txn.id).update(created_at=when)

    owned, created = OwnedProduct.objects.get_or_create(
        user=user,
        product=product,
        defaults={
            "quantity_total": 1,
            "is_active": True,
            "acquired_at": when,
            "last_acquired_at": when,
            "source": "demo_seed",
        },
    )
    if not created:
        owned.quantity_total = int(owned.quantity_total or 0) + 1
        owned.is_active = True
        owned.last_acquired_at = when
        owned.source = "demo_seed"
        owned.save(update_fields=["quantity_total", "is_active", "last_acquired_at", "source"])
    return txn


def _model_watermarks(user) -> dict[str, int]:
    return {
        "transaction": int(Transaction.objects.filter(user=user).aggregate(v=Max("id"))["v"] or 0),
        "offer_assignment": int(OfferAssignment.objects.filter(user=user).aggregate(v=Max("id"))["v"] or 0),
        "offer_event": int(OfferEvent.objects.filter(user=user).aggregate(v=Max("id"))["v"] or 0),
        "roadmap_event": int(RoadmapEvent.objects.filter(user=user).aggregate(v=Max("id"))["v"] or 0),
        "recommendation_event": int(RecommendationEvent.objects.filter(user=user).aggregate(v=Max("id"))["v"] or 0),
        "loyalty_ledger": int(LoyaltyLedgerEntry.objects.filter(account__user=user).aggregate(v=Max("id"))["v"] or 0),
        "roadmap_plan": int(RoadmapPlan.objects.filter(user=user).aggregate(v=Max("id"))["v"] or 0),
        "roadmap_step": int(RoadmapStep.objects.filter(plan__user=user).aggregate(v=Max("id"))["v"] or 0),
    }


def _backdate_queryset_sequential(queryset, *, field_name: str, start_dt: datetime) -> int:
    ids = list(queryset.order_by("id").values_list("id", flat=True))
    for offset, object_id in enumerate(ids):
        queryset.model.objects.filter(id=object_id).update(**{field_name: start_dt + timedelta(seconds=offset)})
    return len(ids)


def _backdate_live_tail_side_effects(*, user, before: dict[str, int], event_dt: datetime) -> dict[str, int]:
    updated = {
        "transactions": _backdate_queryset_sequential(
            Transaction.objects.filter(user=user, id__gt=before["transaction"]),
            field_name="created_at",
            start_dt=event_dt,
        ),
        "offer_assignments": _backdate_queryset_sequential(
            OfferAssignment.objects.filter(user=user, id__gt=before["offer_assignment"]),
            field_name="assigned_at",
            start_dt=event_dt + timedelta(seconds=1),
        ),
        "offer_events": _backdate_queryset_sequential(
            OfferEvent.objects.filter(user=user, id__gt=before["offer_event"]),
            field_name="created_at",
            start_dt=event_dt + timedelta(seconds=2),
        ),
        "roadmap_events": _backdate_queryset_sequential(
            RoadmapEvent.objects.filter(user=user, id__gt=before["roadmap_event"]),
            field_name="created_at",
            start_dt=event_dt + timedelta(seconds=3),
        ),
        "recommendation_events": _backdate_queryset_sequential(
            RecommendationEvent.objects.filter(user=user, id__gt=before["recommendation_event"]),
            field_name="created_at",
            start_dt=event_dt + timedelta(seconds=4),
        ),
        "loyalty_ledger": _backdate_queryset_sequential(
            LoyaltyLedgerEntry.objects.filter(account__user=user, id__gt=before["loyalty_ledger"]),
            field_name="created_at",
            start_dt=event_dt + timedelta(seconds=5),
        ),
        "roadmap_plans": _backdate_queryset_sequential(
            RoadmapPlan.objects.filter(user=user, id__gt=before["roadmap_plan"]),
            field_name="created_at",
            start_dt=event_dt + timedelta(seconds=6),
        ),
        "roadmap_steps": _backdate_queryset_sequential(
            RoadmapStep.objects.filter(plan__user=user, id__gt=before["roadmap_step"]),
            field_name="created_at",
            start_dt=event_dt + timedelta(seconds=7),
        ),
    }
    return updated


def _next_runtime_step_from_roadmap_response(response_data: dict[str, Any]) -> tuple[int | None, str | None, int | None]:
    summary = response_data.get("summary") or {}
    next_step = summary.get("next_step") or {}
    step_id = next_step.get("step_id") or next_step.get("id")
    try:
        step_id_value = int(step_id) if step_id not in (None, "") else None
    except Exception:
        step_id_value = None
    recommended_product_id = next_step.get("recommended_product_id")
    try:
        recommended_value = int(recommended_product_id) if recommended_product_id not in (None, "") else None
    except Exception:
        recommended_value = None
    return step_id_value, str(next_step.get("product_type") or "").strip() or None, recommended_value


def _backdate_roadmap_events_since(*, user, before_event_id: int, start_dt: datetime) -> int:
    return _backdate_queryset_sequential(
        RoadmapEvent.objects.filter(user=user, id__gt=int(before_event_id or 0)),
        field_name="created_at",
        start_dt=start_dt,
    )


def _product_for_runtime_target(
    *,
    category: str,
    runtime_target: str | None,
    recommended_product_id: int | None,
    pools: dict[str, Any],
    profile: CustomerProfile,
    seed: int,
    user_key: str,
    step_index: int,
    exclude_product_ids: set[int],
) -> Product | None:
    if recommended_product_id:
        product = Product.objects.filter(id=int(recommended_product_id), in_stock=True).first()
        if product is not None:
            return product
    if not runtime_target:
        return None
    if category == "fragrance" and runtime_target in FRAGRANCE_SLOTS:
        return pick_product_for_target(
            pools=pools,
            category=category,
            target=runtime_target,
            profile=profile,
            seed=seed,
            user_key=user_key,
            step_index=step_index,
            exclude_product_ids=exclude_product_ids,
        )
    if category != "fragrance" and runtime_target in CURATED_V2_CANONICAL_TYPES.get(category, []):
        return pick_product_for_target(
            pools=pools,
            category=category,
            target=runtime_target,
            profile=profile,
            seed=seed,
            user_key=user_key,
            step_index=step_index,
            exclude_product_ids=exclude_product_ids,
        )
    return None


def _exercise_recs_and_offer(client: APIClient, *, category: str) -> dict[str, Any]:
    recs_resp = client.get("/api/me/recommendations/home", {"category": category, "limit": 5})
    clicked_product_id = None
    if recs_resp.status_code == 200:
        for section in list((recs_resp.data or {}).get("sections") or []):
            for result in list(section.get("results") or []):
                product = result.get("product") or {}
                if product.get("id"):
                    clicked_product_id = int(product["id"])
                    client.post(
                        "/api/me/recommendations/event",
                        {
                            "action": "click",
                            "product_id": clicked_product_id,
                            "page": "home",
                            "section_key": str(section.get("key") or "for_you"),
                        },
                        format="json",
                    )
                    break
            if clicked_product_id is not None:
                break
    offer_resp = client.get("/api/me/next-offer")
    return {
        "recs_status": int(recs_resp.status_code),
        "offer_status": int(offer_resp.status_code),
        "clicked_product_id": clicked_product_id,
    }


def _run_live_tail_checkout(
    *,
    user,
    category: str,
    when: datetime,
    pools: dict[str, Any],
    seed: int,
    step_index: int,
    include_offer_recs_telemetry: bool,
    fallback_target: str | None = None,
    desired_target: str | None = None,
    allow_skip_to_target: bool = False,
    max_skip_steps: int = 3,
) -> dict[str, Any]:
    profile = _profile_for_user(user)
    client = APIClient()
    client.force_authenticate(user)
    before_precheckout = _model_watermarks(user)
    roadmap_resp = client.get("/api/me/roadmap", {"category": category})
    if roadmap_resp.status_code != 200:
        raise CommandError(f"GET /api/me/roadmap failed for {user.username} {category}: {roadmap_resp.status_code}")
    current_step_id, runtime_target, recommended_product_id = _next_runtime_step_from_roadmap_response(roadmap_resp.data)
    if not runtime_target:
        runtime_target = str(desired_target or fallback_target or "").strip() or None

    skipped_targets: list[str] = []
    skip_count = 0
    if allow_skip_to_target and desired_target:
        desired_target = str(desired_target).strip()
        while (
            current_step_id
            and runtime_target
            and runtime_target != desired_target
            and skip_count < max_skip_steps
        ):
            patch_resp = client.patch(
                f"/api/me/roadmap/steps/{int(current_step_id)}",
                {"status": "skipped"},
                format="json",
            )
            if patch_resp.status_code != 200:
                raise CommandError(
                    f"PATCH /api/me/roadmap/steps/{current_step_id} failed for {user.username} {category}: {patch_resp.status_code}"
                )
            skipped_targets.append(str(runtime_target))
            roadmap_resp = client.get("/api/me/roadmap", {"category": category})
            if roadmap_resp.status_code != 200:
                raise CommandError(f"GET /api/me/roadmap failed for {user.username} {category}: {roadmap_resp.status_code}")
            current_step_id, runtime_target, recommended_product_id = _next_runtime_step_from_roadmap_response(
                roadmap_resp.data
            )
            skip_count += 1

    if not runtime_target:
        runtime_target = str(desired_target or fallback_target or "").strip() or None

    _backdate_roadmap_events_since(
        user=user,
        before_event_id=int(before_precheckout["roadmap_event"]),
        start_dt=when,
    )

    before_checkout = _model_watermarks(user)
    exclude_ids = set(OwnedProduct.objects.filter(user=user, is_active=True).values_list("product_id", flat=True))
    product = _product_for_runtime_target(
        category=category,
        runtime_target=runtime_target,
        recommended_product_id=recommended_product_id,
        pools=pools,
        profile=profile,
        seed=seed,
        user_key=user.username,
        step_index=step_index,
        exclude_product_ids=exclude_ids,
    )
    if product is None:
        raise CommandError(f"No runtime checkout product available for {user.username} {category} target={runtime_target}")
    idem = f"demo-tail:{user.username}:{category}:{step_index}:{int(when.timestamp())}"
    checkout_resp = client.post(
        "/api/checkout",
        {
            "channel": "online",
            "idempotency_key": idem,
            "items": [{"product": int(product.id), "quantity": 1}],
        },
        format="json",
    )
    if checkout_resp.status_code != 201:
        raise CommandError(f"POST /api/checkout failed for {user.username} {category}: {checkout_resp.status_code}")
    preflight = {}
    if include_offer_recs_telemetry:
        preflight = _exercise_recs_and_offer(client, category=category)
        client.get("/api/me/roadmap", {"category": category})
    checkout_dt = when + timedelta(minutes=5)
    backdated = _backdate_live_tail_side_effects(user=user, before=before_checkout, event_dt=checkout_dt)
    OwnedProduct.objects.filter(user=user, product=product).update(
        last_acquired_at=checkout_dt,
        acquired_at=checkout_dt,
        source="transaction",
    )
    return {
        "category": category,
        "desired_target": str(desired_target or fallback_target or runtime_target or ""),
        "target": runtime_target,
        "product_id": int(product.id),
        "product_type": str(product.product_type),
        "slot": slot_of_fragrance(product.attrs or {}, raw_meta=product.raw_meta or {}) if product.category == "fragrance" else None,
        "checkout_status_code": int(checkout_resp.status_code),
        "transaction_id": int(checkout_resp.data.get("transaction_id")),
        "next_roadmap_step": (checkout_resp.data.get("next_roadmap_step") or {}).get("product_type"),
        "skipped_targets": list(skipped_targets),
        "side_effects_backdated": backdated,
        "preflight": preflight,
    }


def seed_demo_purchase_history(
    *,
    seed: int,
    users: int,
    max_transactions_per_user: int,
    days_span: int,
    use_checkout_path: bool,
    include_offer_recs_telemetry: bool,
    prefix: str = DEMO_USER_PREFIX,
) -> dict[str, Any]:
    demo_users = get_demo_users(prefix=prefix, seed=seed, limit=users)
    if len(demo_users) < users:
        raise CommandError(
            f"Expected at least {users} demo users for seed={seed}. Run seed_demo_users_and_profiles first."
        )
    pools = load_catalog_seed_pools(include_out_of_stock=False)
    missing = [
        f"{category}:{ptype}"
        for category, product_types in CURATED_V2_CANONICAL_TYPES.items()
        for ptype in product_types
        if len(pools["by_category_type"][category][ptype]) <= 0
    ]
    if missing:
        raise CommandError(f"Catalog is missing in-stock candidates for: {', '.join(missing)}")

    ensure_demo_offers()
    for user in demo_users:
        ensure_loyalty_account(user)

    deleted = _clear_demo_history_for_users(demo_users)
    specs = generate_demo_user_specs(total_users=users, seed=seed, prefix=prefix)
    spec_by_username = {spec.username: spec for spec in specs}

    total_transactions = 0
    total_direct_transactions = 0
    total_live_tail_transactions = 0
    per_category_transactions = Counter()
    live_tail_records: list[dict[str, Any]] = []

    for user_index, user in enumerate(demo_users):
        spec = spec_by_username.get(user.username)
        if spec is None:
            raise CommandError(f"Unexpected demo username encountered: {user.username}")
        profile = _profile_for_user(user)
        plan = build_user_purchase_plan(spec, max_transactions=max_transactions_per_user)
        planned = list(plan["planned_transactions"])
        primary_category = plan["primary_category"]

        planned_times = _scheduled_transaction_times(
            seed=seed + user_index,
            planned=planned,
            days_span=days_span,
        )
        previous_product_id = None
        for ordinal, (item, when) in enumerate(zip(planned, planned_times), start=1):
            if use_checkout_path and str(item.get("mode") or "") == "live":
                result = _run_live_tail_checkout(
                    user=user,
                    category=item["category"],
                    when=when,
                    pools=pools,
                    seed=seed,
                    step_index=ordinal,
                    include_offer_recs_telemetry=include_offer_recs_telemetry,
                    fallback_target=item["target"],
                    desired_target=str(item.get("target") or ""),
                    allow_skip_to_target=bool(item.get("allow_skip_to_target")),
                )
                total_transactions += 1
                total_live_tail_transactions += 1
                per_category_transactions[item["category"]] += 1
                live_tail_records.append(
                    {
                        "user": user.username,
                        "category": item["category"],
                        "planned_target": item["target"],
                        "scenario_role": str(item.get("scenario_role") or ""),
                        **result,
                    }
                )
                previous_product_id = int(result["product_id"])
                continue

            product = pick_product_for_target(
                pools=pools,
                category=item["category"],
                target=item["target"],
                profile=profile,
                seed=seed,
                user_key=user.username,
                step_index=ordinal,
                exclude_product_ids={previous_product_id} if previous_product_id else set(),
            )
            _create_direct_transaction(
                user=user,
                product=product,
                when=when,
                category=item["category"],
                target=item["target"],
                scenario_meta={
                    "cohort": spec.cohort,
                    "cohort_index": spec.cohort_index,
                    "primary_category": primary_category,
                    "scenario_role": str(item.get("scenario_role") or ""),
                },
            )
            total_transactions += 1
            total_direct_transactions += 1
            per_category_transactions[item["category"]] += 1
            previous_product_id = int(product.id)

    fragrance_repairs = _ensure_required_fragrance_transitions(
        demo_users=demo_users,
        pools=pools,
        seed=seed,
        prefix=prefix,
    )
    total_transactions += int(fragrance_repairs["created_transactions"])
    total_direct_transactions += int(fragrance_repairs["created_transactions"])
    per_category_transactions["fragrance"] += int(fragrance_repairs["created_transactions"])

    owned_total = OwnedProduct.objects.filter(user_id__in=[user.id for user in demo_users]).count()
    return {
        "seed": int(seed),
        "users_targeted": int(users),
        "users_seeded": int(len(demo_users)),
        "history_reset": deleted,
        "transactions_created": int(total_transactions),
        "direct_transactions_created": int(total_direct_transactions),
        "live_tail_transactions_created": int(total_live_tail_transactions),
        "owned_products_total": int(owned_total),
        "per_category_transactions": {key: int(value) for key, value in per_category_transactions.items()},
        "live_tail_records_sample": live_tail_records[:25],
        "fragrance_transition_repairs": fragrance_repairs,
        "use_checkout_path": bool(use_checkout_path),
        "include_offer_recs_telemetry": bool(include_offer_recs_telemetry),
    }


def _transaction_rows_for_demo_users(*, prefix: str, seed: int | None = None):
    users = get_demo_users(prefix=prefix, seed=seed)
    user_ids = [int(user.id) for user in users]
    rows = list(
        TransactionItem.objects.filter(transaction__user_id__in=user_ids)
        .order_by("transaction__created_at", "transaction_id", "id")
        .values(
            "transaction__user_id",
            "transaction__created_at",
            "transaction_id",
            "product_id",
            "product__category",
            "product__product_type",
            "product__attrs",
            "product__raw_meta",
        )
    )
    return users, rows


def _fragrance_slot_sequence_for_user(user) -> list[str]:
    rows = list(
        TransactionItem.objects.filter(transaction__user=user, product__category="fragrance")
        .order_by("transaction__created_at", "transaction_id", "id")
        .values("product__attrs", "product__raw_meta")
    )
    return [
        slot_of_fragrance(row.get("product__attrs") or {}, raw_meta=row.get("product__raw_meta") or {})
        for row in rows
    ]


def _ensure_required_fragrance_transitions(
    *,
    demo_users: list[Any],
    pools: dict[str, Any],
    seed: int,
    prefix: str,
) -> dict[str, int]:
    created = 0
    satisfied: Counter = Counter()
    transitions = _category_transition_counts(_transaction_rows_for_demo_users(prefix=prefix, seed=seed)[1])["fragrance_slots"]
    safe_upper_bound = timezone.now().replace(hour=12, minute=0, second=0, microsecond=0) - timedelta(days=6)
    for pair_index, (left, right) in enumerate(FRAGRANCE_REQUIRED_TRANSITIONS, start=1):
        key = f"{left}->{right}"
        if int(transitions.get(key, 0)) > 0:
            satisfied[key] = int(transitions[key])
            continue
        user = demo_users[(pair_index - 1) % len(demo_users)]
        profile = _profile_for_user(user)
        slot_seq = _fragrance_slot_sequence_for_user(user)
        last_slot = slot_seq[-1] if slot_seq else None
        latest_txn_dt = (
            Transaction.objects.filter(user=user)
            .order_by("-created_at", "-id")
            .values_list("created_at", flat=True)
            .first()
        )
        if latest_txn_dt is None:
            base_when = safe_upper_bound - timedelta(days=pair_index)
        else:
            base_when = min(latest_txn_dt + timedelta(hours=6), safe_upper_bound - timedelta(hours=pair_index))
        if last_slot != left:
            left_product = pick_product_for_target(
                pools=pools,
                category="fragrance",
                target=left,
                profile=profile,
                seed=seed,
                user_key=user.username,
                step_index=10_000 + pair_index * 2,
                exclude_product_ids=set(),
            )
            _create_direct_transaction(
                user=user,
                product=left_product,
                when=base_when,
                category="fragrance",
                target=left,
                scenario_meta={"cohort": "transition_guard", "required_pair": key},
            )
            created += 1
        right_product = pick_product_for_target(
            pools=pools,
            category="fragrance",
            target=right,
            profile=profile,
            seed=seed,
            user_key=user.username,
            step_index=10_001 + pair_index * 2,
            exclude_product_ids=set(),
        )
        _create_direct_transaction(
            user=user,
            product=right_product,
            when=base_when + timedelta(hours=6),
            category="fragrance",
            target=right,
            scenario_meta={"cohort": "transition_guard", "required_pair": key},
        )
        created += 1
        transitions[key] += 1
        satisfied[key] = int(transitions[key])
    return {"created_transactions": int(created), "final_transition_counts": dict(satisfied)}


def _category_transition_counts(rows: list[dict[str, Any]]) -> dict[str, Counter]:
    per_user_category: dict[tuple[int, str], list[tuple[datetime, str]]] = defaultdict(list)
    fragrance_per_user: dict[int, list[str]] = defaultdict(list)
    for row in rows:
        user_id = int(row["transaction__user_id"])
        category = str(row["product__category"] or "")
        created_at = row["transaction__created_at"]
        product_type = str(row["product__product_type"] or "")
        per_user_category[(user_id, category)].append((created_at, product_type))
        if category == "fragrance":
            fragrance_per_user[user_id].append(
                slot_of_fragrance(row.get("product__attrs") or {}, raw_meta=row.get("product__raw_meta") or {})
            )
    transitions: dict[str, Counter] = defaultdict(Counter)
    for (_, category), seq in per_user_category.items():
        ordered = [ptype for _, ptype in sorted(seq, key=lambda item: item[0])]
        for left, right in zip(ordered, ordered[1:]):
            transitions[category][f"{left}->{right}"] += 1
    fragrance_slot_transitions = Counter()
    for _, seq in fragrance_per_user.items():
        for left, right in zip(seq, seq[1:]):
            fragrance_slot_transitions[f"{left}->{right}"] += 1
    transitions["fragrance_slots"] = fragrance_slot_transitions
    return transitions


def _users_with_purchases_by_category(rows: list[dict[str, Any]]) -> dict[str, int]:
    buckets: dict[str, set[int]] = defaultdict(set)
    for row in rows:
        buckets[str(row["product__category"] or "")].add(int(row["transaction__user_id"]))
    return {category: len(user_ids) for category, user_ids in buckets.items()}


def _users_with_two_plus_transactions(rows: list[dict[str, Any]]) -> dict[str, int]:
    buckets: dict[tuple[int, str], int] = Counter()
    for row in rows:
        buckets[(int(row["transaction__user_id"]), str(row["product__category"] or ""))] += 1
    out: dict[str, int] = Counter()
    for (_, category), count in buckets.items():
        if count >= 2:
            out[category] += 1
    return {category: int(value) for category, value in out.items()}


def _initial_anchor_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    seen: set[tuple[int, str]] = set()
    counts: Counter = Counter()
    for row in rows:
        key = (int(row["transaction__user_id"]), str(row["product__category"] or ""))
        if key in seen:
            continue
        seen.add(key)
        counts[key[1]] += 1
    return {category: int(value) for category, value in counts.items()}


def _runtime_surface_counts(*, prefix: str, seed: int | None = None) -> dict[str, Any]:
    users = get_demo_users(prefix=prefix, seed=seed)
    user_ids = [int(user.id) for user in users]
    return {
        "roadmap_plan_refreshed": int(
            RoadmapEvent.objects.filter(user_id__in=user_ids, event_type=RoadmapEvent.Type.PLAN_REFRESHED).count()
        ),
        "roadmap_step_completed": int(
            RoadmapEvent.objects.filter(user_id__in=user_ids, event_type=RoadmapEvent.Type.STEP_COMPLETED).count()
        ),
        "roadmap_step_skipped": int(
            RoadmapEvent.objects.filter(user_id__in=user_ids, event_type=RoadmapEvent.Type.STEP_SKIPPED).count()
        ),
        "roadmap_step_exposed": int(
            RoadmapEvent.objects.filter(user_id__in=user_ids, event_type=RoadmapEvent.Type.STEP_EXPOSED).count()
        ),
        "offer_assignments": int(OfferAssignment.objects.filter(user_id__in=user_ids).count()),
        "offer_events": int(OfferEvent.objects.filter(user_id__in=user_ids).count()),
        "recommendation_events": int(RecommendationEvent.objects.filter(user_id__in=user_ids).count()),
    }


def _roadmap_and_offer_smoke(*, prefix: str, seed: int | None = None, sample_per_category: int = 4) -> dict[str, Any]:
    result = {"roadmap": {}, "next_offer": {}, "failures": []}
    users, rows = _transaction_rows_for_demo_users(prefix=prefix, seed=seed)
    category_users: dict[str, list[int]] = defaultdict(list)
    for row in rows:
        category = str(row["product__category"] or "")
        user_id = int(row["transaction__user_id"])
        if user_id not in category_users[category]:
            category_users[category].append(user_id)
    user_map = {int(user.id): user for user in users}

    with db_tx.atomic():
        for category, user_ids in category_users.items():
            ok_count = 0
            next_offer_ok = 0
            sample_ids = user_ids[:sample_per_category]
            for user_id in sample_ids:
                client = APIClient()
                client.force_authenticate(user_map[user_id])
                roadmap_resp = client.get("/api/me/roadmap", {"category": category})
                next_offer_resp = client.get("/api/me/next-offer")
                if roadmap_resp.status_code == 200:
                    ok_count += 1
                else:
                    result["failures"].append(f"roadmap {category} user_id={user_id} status={roadmap_resp.status_code}")
                if next_offer_resp.status_code == 200:
                    next_offer_ok += 1
                else:
                    result["failures"].append(f"next-offer {category} user_id={user_id} status={next_offer_resp.status_code}")
            result["roadmap"][category] = {"sampled_users": len(sample_ids), "ok": ok_count}
            result["next_offer"][category] = {"sampled_users": len(sample_ids), "ok": next_offer_ok}
        db_tx.set_rollback(True)
    return result


def build_demo_history_readiness_report(*, prefix: str = DEMO_USER_PREFIX, seed: int | None = None) -> dict[str, Any]:
    users, rows = _transaction_rows_for_demo_users(prefix=prefix, seed=seed)
    user_ids = [int(user.id) for user in users]
    transitions = _category_transition_counts(rows)
    users_by_category = _users_with_purchases_by_category(rows)
    users_two_plus = _users_with_two_plus_transactions(rows)
    anchors = _initial_anchor_counts(rows)
    runtime_surfaces = _runtime_surface_counts(prefix=prefix, seed=seed)
    smoke = _roadmap_and_offer_smoke(prefix=prefix, seed=seed)
    total_transactions = Transaction.objects.filter(user_id__in=user_ids).count()
    owned_total = OwnedProduct.objects.filter(user_id__in=user_ids).count()
    enough_initial = (
        len(users) >= 120
        and total_transactions >= 800
        and all(int(anchors.get(category, 0)) >= 20 for category in ("haircare", "skincare", "makeup", "fragrance"))
    )
    enough_transition = (
        runtime_surfaces["roadmap_plan_refreshed"] >= 100
        and runtime_surfaces["roadmap_step_completed"] >= 60
        and all(int(transitions["fragrance_slots"].get(f"{left}->{right}", 0)) > 0 for left, right in FRAGRANCE_REQUIRED_TRANSITIONS)
        and all(int(users_two_plus.get(category, 0)) >= 20 for category in ("haircare", "skincare", "makeup", "fragrance"))
    )
    return {
        "generated_at": timezone.now().isoformat(),
        "demo_users_total": len(users),
        "transactions_total": int(total_transactions),
        "owned_products_total": int(owned_total),
        "users_with_purchases_by_category": users_by_category,
        "users_with_two_plus_transactions_by_category": users_two_plus,
        "initial_planner_anchor_counts": anchors,
        "transitions_by_category": {
            category: dict(counter)
            for category, counter in transitions.items()
            if category != "fragrance_slots"
        },
        "fragrance_slot_transitions": dict(transitions["fragrance_slots"]),
        "runtime_surfaces": runtime_surfaces,
        "roadmap_offer_smoke": smoke,
        "verdict": {
            "safe_for_dataset_rebuild": bool(enough_initial and not smoke["failures"]),
            "enough_for_initial_planner_dataset": bool(enough_initial),
            "enough_for_transition_dataset": bool(enough_transition and not smoke["failures"]),
            "blocking_issues": list(smoke["failures"]),
        },
    }


def build_demo_history_readiness_md(report: dict[str, Any]) -> str:
    verdict = report["verdict"]
    lines = [
        "# Demo History Readiness",
        "",
        f"- demo_users_total: **{report['demo_users_total']}**",
        f"- transactions_total: **{report['transactions_total']}**",
        f"- owned_products_total: **{report['owned_products_total']}**",
        f"- users_with_purchases_by_category: `{json.dumps(report['users_with_purchases_by_category'], ensure_ascii=False, default=json_default)}`",
        f"- users_with_two_plus_transactions_by_category: `{json.dumps(report['users_with_two_plus_transactions_by_category'], ensure_ascii=False, default=json_default)}`",
        f"- initial_planner_anchor_counts: `{json.dumps(report['initial_planner_anchor_counts'], ensure_ascii=False, default=json_default)}`",
        f"- fragrance_slot_transitions: `{json.dumps(report['fragrance_slot_transitions'], ensure_ascii=False, default=json_default)}`",
        f"- runtime_surfaces: `{json.dumps(report['runtime_surfaces'], ensure_ascii=False, default=json_default)}`",
        "",
        f"- safe_for_dataset_rebuild = **{'yes' if verdict['safe_for_dataset_rebuild'] else 'no'}**",
        f"- enough_for_initial_planner_dataset = **{'yes' if verdict['enough_for_initial_planner_dataset'] else 'no'}**",
        f"- enough_for_transition_dataset = **{'yes' if verdict['enough_for_transition_dataset'] else 'no'}**",
    ]
    failures = (report.get("roadmap_offer_smoke") or {}).get("failures") or []
    if failures:
        lines.extend(["", "## Failing Checks"])
        lines.extend(f"- {item}" for item in failures)
    return "\n".join(lines) + "\n"


def write_report_files(*, report: dict[str, Any], md_path: str | Path, json_path: str | Path, md_builder) -> None:
    md_file = Path(md_path)
    json_file = Path(json_path)
    md_file.parent.mkdir(parents=True, exist_ok=True)
    json_file.parent.mkdir(parents=True, exist_ok=True)
    md_file.write_text(md_builder(report), encoding="utf-8")
    json_file.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=json_default),
        encoding="utf-8",
    )
