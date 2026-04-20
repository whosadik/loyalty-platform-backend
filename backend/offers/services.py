from __future__ import annotations
from django.conf import settings
from dataclasses import dataclass
from datetime import timedelta, datetime
from decimal import Decimal
from typing import Any

from django.db import transaction as db_tx, connection
from django.utils import timezone
from django.db.models import Sum, Max

from catalog.models import Product
from transactions.models import Transaction, TransactionItem, OwnedProduct
from users_app.models import CustomerProfile
from offers.models import Offer, OfferAssignment, CampaignBudget, OfferEvent
from offers.events import record_offer_event
from offers import ml_scorer as offer_ml_scorer
from ml_logic.recommender import bundle as rec_bundle

from ml_logic.next_best_reward import RFM, segment  # compute_rfm можно не трогать
from ml_logic.recommender import (
    UserProfile as RecUserProfile,
    recommend as rec_recommend,
    build_cooccurrence,
)
from django.db.models import Count
from django.core.cache import cache
from recs_analytics.fatigue import adjust_recs
from recs_analytics.effectiveness import category_uplift
from roadmap_app.fragrance_slots import SLOTS as FRAGRANCE_SLOTS, slot_of_fragrance

ONBOARDING_FIRST_ORDER_CAMPAIGN = "onboarding_first_order"
WINBACK_30D_CAMPAIGN = "winback_30d"
FAVORITE_CATEGORY_CAMPAIGN = "favorite_category"


def _week_start(d: datetime) -> datetime.date:
    # Monday as week start
    return (d - timedelta(days=d.weekday())).date()


def _reset_campaign_week_if_needed(camp: CampaignBudget, now: datetime) -> None:
    if hasattr(camp, "week_start_date"):
        ws = _week_start(now)
        if camp.week_start_date != ws:
            camp.week_start_date = ws
            camp.weekly_spent = Decimal("0.0")
            camp.save(update_fields=["week_start_date", "weekly_spent"])


def _campaign_candidates(context_steps: list[str] | None, post_ctx: dict | None):
    """
    Returns (campaigns_in_order, routing_info)
    Hard preference order:
      fragrance_crosssell (if fragrance in post_ctx.categories)
      skincare_retention (if "spf" in context_steps)
      makeup_push (fallback)
      default (always last)
      then any remaining active campaigns by priority
    """
    cats = set((post_ctx or {}).get("categories") or [])
    steps = set(context_steps or [])

    has_fragrance = "fragrance" in cats
    has_spf = "spf" in steps

    preferred = []
    if has_fragrance:
        preferred.append("fragrance_crosssell")
    if has_spf:
        preferred.append("skincare_retention")
    if not preferred:
        preferred.append("makeup_push")
    preferred.append("default")

    qs = list(CampaignBudget.objects.filter(is_active=True).order_by("priority", "id"))
    by_name = {c.name: c for c in qs}

    def passes_gates(c: CampaignBudget) -> bool:
        # allowed_steps gate
        if getattr(c, "allowed_steps", None):
            if not steps:
                return False
            if not (set(c.allowed_steps or []) & steps):
                return False

        # allowed_categories gate (если есть контекст категорий — требуем пересечение)
        if getattr(c, "allowed_categories", None):
            if cats and not (set(c.allowed_categories or []) & cats):
                return False

        return True

    ordered: list[CampaignBudget] = []
    included_why: list[dict] = []

    def add(name: str, why: str):
        c = by_name.get(name)
        if not c:
            return
        if c in ordered:
            return
        if not passes_gates(c):
            return
        ordered.append(c)
        included_why.append({"campaign": c.name, "why": why, "priority": c.priority})

    # 1) preferred campaigns first
    if has_fragrance:
        add("fragrance_crosssell", "preferred: post_ctx contains fragrance")
    if has_spf:
        add("skincare_retention", "preferred: routine context contains spf")
    if not (has_fragrance or has_spf):
        add("makeup_push", "preferred: default fallback when no fragrance/spf signal")

    # 2) then any other campaigns by priority (excluding ones already added and excluding default for now)
    for c in qs:
        if c in ordered:
            continue
        if c.name in {"default", "fragrance_crosssell", "skincare_retention", "makeup_push"}:
            continue
        if passes_gates(c):
            ordered.append(c)
            included_why.append({"campaign": c.name, "why": "eligible: by priority", "priority": c.priority})

    # 3) default last (if active & passes gates)
    add("default", "always: fallback campaign")

    routing_info = {
        "signals": {
            "categories": sorted(cats),
            "context_steps": sorted(steps),
            "has_fragrance": has_fragrance,
            "has_spf": has_spf,
        },
        "preferred_order": preferred,
        "included": [c.name for c in ordered],
        "included_why": included_why,
    }

    return ordered, routing_info


def _effective_allowed_categories(offer: Offer, camp: CampaignBudget | None) -> list[str]:
    """
    Ограничение по категориям = пересечение offer.allowed_categories и camp.allowed_categories (если оба заданы).
    """
    o = offer.allowed_categories or []
    c = (camp.allowed_categories or []) if camp else []
    if o and c:
        return [x for x in o if x in c]
    return o or c


def _effective_allowed_steps(offer: Offer, camp: CampaignBudget | None) -> list[str]:
    o = offer.allowed_steps or []
    c = (camp.allowed_steps or []) if camp else []
    if o and c:
        return [x for x in o if x in c]
    return o or c



def _rfm(user, now: datetime) -> dict[str, Any]:
    since = now - timedelta(days=90)

    last_txn = Transaction.objects.filter(user=user).order_by("-created_at").first()
    if last_txn:
        recency_days = (now.date() - last_txn.created_at.date()).days
    else:
        recency_days = 9999

    q = Transaction.objects.filter(user=user, created_at__gte=since)
    frequency_90d = q.count()
    monetary_90d = q.aggregate(s=Sum("total_amount"))["s"] or Decimal("0")

    return {
        "recency_days": int(recency_days),
        "frequency_90d": int(frequency_90d),
        "monetary_90d": float(monetary_90d),
    }


def _build_rec_profile(cp: CustomerProfile) -> RecUserProfile:
    return RecUserProfile(
        skin_type=cp.skin_type,
        goals=cp.goals or [],
        avoid_flags=cp.avoid_flags or [],
        budget=cp.budget,
        hair=cp.hair_profile or {},
        makeup=cp.makeup_profile or {},
        fragrance=cp.fragrance_profile or {},
    )


def _load_products_for_recs() -> list[dict[str, Any]]:
    db_name = connection.settings_dict.get("NAME", "default")
    key = f"recs:products:v3:{db_name}"
    cached = cache.get(key)
    if cached is not None:
        sample_ids = [int(p["id"]) for p in cached[:100] if isinstance(p, dict) and p.get("id")]
        if not sample_ids:
            return cached
        existing = set(Product.objects.filter(id__in=sample_ids).values_list("id", flat=True))
        if len(existing) == len(set(sample_ids)):
            return cached

    data = list(
        Product.objects.filter(in_stock=True, price__isnull=False).values(
            "id","name","brand","price","category","product_type",
            "concerns","attrs","raw_meta","actives","flags","supported_skin_types","strength","in_stock",
            "ingredients_inci",
        )
    )
    cache.set(key, data, timeout=600)  # 10 минут
    return data



def _cooccurrence_90d(now: datetime):
    db_name = connection.settings_dict.get("NAME", "default")
    key = f"recs:cooc90d:v2:{db_name}"
    cached = cache.get(key)
    if cached is not None:
        return cached

    since = now - timedelta(days=90)
    rows = (
        TransactionItem.objects.filter(transaction__created_at__gte=since)
        .values("transaction_id", "product_id")
    )
    txn_map: dict[int, list[int]] = {}
    for r in rows:
        txn_map.setdefault(r["transaction_id"], []).append(r["product_id"])

    co = build_cooccurrence(list(txn_map.values()))

    # ✅ ВАЖНО: превратить defaultdict(...) в обычные dict, чтобы pickle/redis не падал
    co_plain = {int(k): {int(kk): int(vv) for kk, vv in dict(v).items()} for k, v in dict(co).items()}

    cache.set(key, co_plain, timeout=600)  # 10 минут
    return co_plain


def _passes_cooldown(user, offer: Offer, now: datetime) -> bool:
    cd = int(getattr(offer, "cooldown_days", 0) or 0)
    if cd <= 0:
        return True
    since = now - timedelta(days=cd)
    return not OfferAssignment.objects.filter(
        user=user, offer=offer, is_redeemed=True, assigned_at__gte=since
    ).exists()

# Порог "уже достаточно" по категориям (MVP)
def _is_onboarding_first_order_campaign(camp: CampaignBudget | None) -> bool:
    return bool(camp and camp.name == ONBOARDING_FIRST_ORDER_CAMPAIGN)


def _user_has_any_transactions(user) -> bool:
    return Transaction.objects.filter(user=user).exists()


def _is_winback_30d_campaign(camp: CampaignBudget | None) -> bool:
    return bool(camp and camp.name == WINBACK_30D_CAMPAIGN)


def _is_favorite_category_campaign(camp: CampaignBudget | None) -> bool:
    return bool(camp and camp.name == FAVORITE_CATEGORY_CAMPAIGN)


def _winback_inactivity_days() -> int:
    return int(getattr(settings, "WINBACK_INACTIVITY_DAYS", 30))


def _winback_reassign_days() -> int:
    return int(getattr(settings, "WINBACK_REASSIGN_DAYS", 30))


def _is_winback_eligible(user, now: datetime) -> bool:
    last_txn = (
        Transaction.objects.filter(user=user)
        .only("created_at")
        .order_by("-created_at")
        .first()
    )
    if not last_txn:
        return False
    inactivity_days = max(0, (now.date() - last_txn.created_at.date()).days)
    return inactivity_days >= _winback_inactivity_days()


def _passes_winback_assignment_cooldown(user, camp: CampaignBudget, now: datetime) -> bool:
    since = now - timedelta(days=_winback_reassign_days())
    return not OfferAssignment.objects.filter(
        user=user,
        offer__campaign=camp,
        assigned_at__gte=since,
    ).exists()


def _favorite_category_window_days() -> int:
    return int(getattr(settings, "FAVORITE_CATEGORY_WINDOW_DAYS", 90))


def _favorite_category_reassign_days() -> int:
    return int(getattr(settings, "FAVORITE_CATEGORY_REASSIGN_DAYS", 14))


def _favorite_category(user, now: datetime) -> str | None:
    since = now - timedelta(days=_favorite_category_window_days())
    rows = (
        TransactionItem.objects.filter(
            transaction__user=user,
            transaction__created_at__gte=since,
        )
        .values("product__category")
        .annotate(
            total_qty=Sum("quantity"),
            line_count=Count("id"),
            last_at=Max("transaction__created_at"),
        )
        .order_by("-total_qty", "-line_count", "-last_at", "product__category")
    )
    best = rows.first()
    return best["product__category"] if best else None


def _passes_favorite_category_assignment_cooldown(user, camp: CampaignBudget, now: datetime) -> bool:
    since = now - timedelta(days=_favorite_category_reassign_days())
    return not OfferAssignment.objects.filter(
        user=user,
        offer__campaign=camp,
        assigned_at__gte=since,
    ).exists()


SATURATION_LIMITS = {
    "makeup": 3,      # 3+ товаров одного product_type (например mascara) = хватит
    "skincare": 2,
    "haircare": 2,
    "fragrance": 2,
}


def _owned_type_count(user, category: str, product_type: str) -> int:
    return (
        OwnedProduct.objects.filter(
            user=user,
            is_active=True,
            product__category=category,
            product__product_type=product_type,
        )
        .count()
    )


def _is_saturated(user, category: str, product_type: str) -> bool:
    limit = SATURATION_LIMITS.get(category, 2)
    return _owned_type_count(user, category, product_type) >= limit

def _pick_target_for_offer(
    user,
    offer: Offer,
    now: datetime,
    context_steps: list[str] | None,
    post_ctx: dict | None,
    campaign: CampaignBudget | None = None,
    roadmap_ctx: dict | None = None,
):
    forced_category = _favorite_category(user, now) if _is_favorite_category_campaign(campaign) else None
    if forced_category and offer.target_scope == "category":
        return {"scope": "category", "value": forced_category, "picked_via": "favorite_category"}

    # 0) roadmap shortcut
    roadmap_category = str((roadmap_ctx or {}).get("category") or "").strip()
    roadmap_next_type = str((roadmap_ctx or {}).get("next_product_type") or "").strip()
    roadmap_next_product_id = (roadmap_ctx or {}).get("next_product_id")
    is_fragrance_slot = roadmap_category == "fragrance" and roadmap_next_type in FRAGRANCE_SLOTS
    if (
        roadmap_category
        and roadmap_next_type
        and offer.target_scope in {"product_type", "product_id"}
    ):
        allowed_pt_values = {
            str(x).strip()
            for x in (offer.allowed_product_types or [])
            if str(x).strip()
        }
        allowed_slot_values = {x for x in allowed_pt_values if x in FRAGRANCE_SLOTS}
        allowed_actual_types = allowed_pt_values - allowed_slot_values

        allowed_cats = _effective_allowed_categories(offer, campaign)
        if forced_category:
            if forced_category != roadmap_category:
                allowed_cats = [forced_category]
            elif not allowed_cats:
                allowed_cats = [forced_category]

        allow_category = (not allowed_cats) or (roadmap_category in allowed_cats)
        if is_fragrance_slot:
            if not allowed_pt_values:
                allow_type = True
            elif allowed_slot_values:
                allow_type = roadmap_next_type in allowed_slot_values
            else:
                # Backward-compatible mode: allowed_product_types may contain actual types (edp/edt)
                allow_type = True
        else:
            allow_type = (not allowed_pt_values) or (roadmap_next_type in allowed_pt_values)
        if allow_category and allow_type:
            if offer.target_scope == "product_type":
                return {
                    "scope": "product_type",
                    "value": roadmap_next_type,
                    "category": roadmap_category,
                    "picked_via": "roadmap_shortcut",
                }

            # product_id target from roadmap preferred recommendation or fresh recs
            owned_ids = set(
                OwnedProduct.objects.filter(user=user, is_active=True).values_list("product_id", flat=True)
            )
            recent_vals = _recent_target_values(user, days=7, now=now)

            pid_from_roadmap = None
            try:
                if roadmap_next_product_id is not None:
                    pid_from_roadmap = int(roadmap_next_product_id)
            except Exception:
                pid_from_roadmap = None

            if pid_from_roadmap:
                prod_qs = Product.objects.filter(
                    id=pid_from_roadmap,
                    category=roadmap_category,
                    in_stock=True,
                )
                if not is_fragrance_slot:
                    prod_qs = prod_qs.filter(product_type=roadmap_next_type)
                prod = prod_qs.first()
                if prod and is_fragrance_slot:
                    slot = slot_of_fragrance(prod.attrs or {}, raw_meta=prod.raw_meta or {})
                    if slot != roadmap_next_type:
                        prod = None
                if prod and is_fragrance_slot and allowed_actual_types:
                    if str(prod.product_type) not in allowed_actual_types:
                        prod = None
                if prod and prod.id not in owned_ids and prod.id not in recent_vals:
                    return {
                        "scope": "product_id",
                        "value": int(prod.id),
                        "category": roadmap_category,
                        "product_type": roadmap_next_type,
                        "actual_product_type": prod.product_type if is_fragrance_slot else None,
                        "picked_via": "roadmap_shortcut",
                    }

            cp, _ = CustomerProfile.objects.get_or_create(user=user)
            prof = _build_rec_profile(cp)
            products = _load_products_for_recs()
            co = _cooccurrence_90d(now)
            recs = rec_recommend(
                prof=prof,
                products=products,
                owned_active_ids=list(owned_ids),
                context_product_ids=((post_ctx or {}).get("product_ids") or list(owned_ids))[:50],
                category=roadmap_category,
                product_type=None if is_fragrance_slot else roadmap_next_type,
                limit=50 if is_fragrance_slot else 10,
                co=co,
            )
            if is_fragrance_slot:
                slot_filtered: list[dict[str, Any]] = []
                for row in recs:
                    product = row.get("product") or {}
                    if allowed_actual_types and str(product.get("product_type") or "") not in allowed_actual_types:
                        continue
                    if slot_of_fragrance(
                        product.get("attrs") or {},
                        raw_meta=product.get("raw_meta") or {},
                    ) == roadmap_next_type:
                        slot_filtered.append(row)
                recs = slot_filtered
            top = _pick_best_rec_target(user, recs, now=now, recent_vals=recent_vals)
            if top:
                p = top["product"]
                return {
                    "scope": "product_id",
                    "value": p["id"],
                    "category": roadmap_category,
                    "product_type": roadmap_next_type if is_fragrance_slot else p["product_type"],
                    "actual_product_type": p.get("product_type") if is_fragrance_slot else None,
                    "rec_score": top.get("score"),
                    "rec_components": top.get("components", {}),
                    "rec_why": (top.get("why") or [])[:6],
                    "adjusted_score": top.get("adjusted_score"),
                    "picked_via": "roadmap_shortcut+recs",
                }

            if is_fragrance_slot:
                blocked = set(owned_ids) | set(recent_vals)
                fallback_rows = list(
                    Product.objects.filter(category="fragrance", in_stock=True)
                    .exclude(id__in=list(blocked))
                    .values("id", "category", "product_type", "attrs", "raw_meta")
                    .order_by("id")[:50]
                )
                for row in fallback_rows:
                    if allowed_actual_types and str(row.get("product_type") or "") not in allowed_actual_types:
                        continue
                    if slot_of_fragrance(
                        row.get("attrs") or {},
                        raw_meta=row.get("raw_meta") or {},
                    ) != roadmap_next_type:
                        continue
                    return {
                        "scope": "product_id",
                        "value": int(row["id"]),
                        "category": "fragrance",
                        "product_type": roadmap_next_type,
                        "actual_product_type": str(row["product_type"]),
                        "picked_via": "roadmap_shortcut_fallback+slot_db",
                    }

            return {
                "scope": "product_type",
                "value": roadmap_next_type,
                "category": roadmap_category,
                "picked_via": "roadmap_shortcut_fallback",
            }

    # 1) routine-context shortcut
    if settings.USE_ROUTINE_SHORTCUT and context_steps and offer.target_scope in {"product_type", "product_id"}:
        if "spf" in context_steps:
            # if offer allows skincare or doesn't restrict
            allowed = _effective_allowed_categories(offer, campaign)
            if not allowed or "skincare" in allowed:
                if offer.target_scope == "product_type":
                    return {"scope": "product_type", "value": "spf", "category": "skincare"}
                # product_id
                # pick recommended SPF
                cp, _ = CustomerProfile.objects.get_or_create(user=user)
                prof = _build_rec_profile(cp)
                products = _load_products_for_recs()
                co = _cooccurrence_90d(now)
                owned_ids = list(
                    OwnedProduct.objects.filter(user=user, is_active=True).values_list("product_id", flat=True)
                )
                recent_vals = _recent_target_values(user, days=7, now=now)

                recs = rec_recommend(
                    prof=prof,
                    products=products,
                    owned_active_ids=owned_ids,
                    context_product_ids=owned_ids[:50],
                    category="skincare",
                    product_type="spf",
                    limit=10,   
                    co=co,
                )

                top = _pick_best_rec_target(user, recs, now=now, recent_vals=recent_vals)
                if top:
                    p = top["product"]
                    return {
                        "scope": "product_id",
                        "value": p["id"],
                        "category": p["category"],
                        "product_type": p["product_type"],
                        "rec_score": top.get("score"),
                        "rec_components": top.get("components", {}),
                        "rec_why": (top.get("why") or [])[:6],
                        "adjusted_score": top.get("adjusted_score"),
                        "picked_via": "routine_shortcut+recs",
                    }

    # 2) if cart
    if offer.target_scope == "cart":
        return {"scope": "cart"}
    
    # ---- Bundle-driven target (post-purchase) ----
    if settings.USE_BUNDLE_TARGETING and post_ctx and offer.target_scope == "product_id":
        allowed_cats = _effective_allowed_categories(offer, campaign)
        if forced_category:
            if allowed_cats and forced_category not in allowed_cats:
                return {"scope": "category", "value": forced_category, "picked_via": "favorite_category_fallback"}
            allowed_cats = [forced_category]
        bundle_cat = allowed_cats[0] if allowed_cats else None

        t = _pick_product_from_bundle(user, now, post_ctx, category=bundle_cat)
        if t:
            return t

    # 3) explicit restrictions
    allowed_cats = _effective_allowed_categories(offer, campaign)
    if forced_category:
        if allowed_cats and forced_category not in allowed_cats:
            return {"scope": "category", "value": forced_category, "picked_via": "favorite_category_fallback"}
        allowed_cats = [forced_category]
    allowed_pts = offer.allowed_product_types or []

    cp, _ = CustomerProfile.objects.get_or_create(user=user)
    prof = _build_rec_profile(cp)

    suggestions = []
    if settings.USE_POST_PURCHASE_RULES:
        suggestions = _derive_cross_sell_suggestions(post_ctx)

    filtered = []
    for s in suggestions:
        if allowed_cats and s["category"] not in allowed_cats:
            continue

        if offer.allowed_product_types and s["product_type"] not in (offer.allowed_product_types or []):
            # если scope=category — тип продукта можно игнорировать
            if offer.target_scope != "category":
                continue

        filtered.append(s)

    # теперь перебираем по приоритету
    for s in filtered:
        cat = s["category"]
        pt = s["product_type"]

        # антиспам по категории
        if not _passes_category_cooldown(user, cat, now, days=3):
            continue

        # saturation: если этого типа уже достаточно — пропускаем
        if _is_saturated(user, cat, pt):
            continue

        # применяем target в зависимости от scope
        if offer.target_scope == "category":
            return {"scope": "category", "value": cat, "picked_via": "post_purchase_rules"}

        if offer.target_scope == "product_type":
            return {"scope": "product_type", "value": pt, "category": cat, "picked_via": "post_purchase_rules"}

        if offer.target_scope == "product_id":
            cp, _ = CustomerProfile.objects.get_or_create(user=user)
            prof = _build_rec_profile(cp)
            products = _load_products_for_recs()
            co = _cooccurrence_90d(now)
            owned_ids = list(
                OwnedProduct.objects.filter(user=user, is_active=True).values_list("product_id", flat=True)
            )

            recent_vals = _recent_target_values(user, days=7, now=now)

            recs = rec_recommend(
                prof=prof,
                products=products,
                owned_active_ids=owned_ids,
                context_product_ids=(post_ctx.get("product_ids") or owned_ids)[:50],
                category=cat,
                product_type=pt,
                limit=10,   
                co=co,
            )

            top = _pick_best_rec_target(user, recs, now=now, recent_vals=recent_vals)
            if top:
                p = top["product"]
                return {
                    "scope": "product_id",
                    "value": p["id"],
                    "category": p["category"],
                    "product_type": p["product_type"],
                    "rec_score": top.get("score"),
                    "rec_components": top.get("components", {}),
                    "rec_why": (top.get("why") or [])[:6],
                    "adjusted_score": top.get("adjusted_score"),
                    "picked_via": "post_purchase_rules+recs",
                }

            # если реков нет — деградируем до product_type
            return {"scope": "product_type", "value": pt, "category": cat, "picked_via": "post_purchase_rules_fallback"}

    # fallback category choice based on filled profiles
    if not allowed_cats:
        if (prof.fragrance or {}):
            allowed_cats = ["fragrance"]
        elif (prof.makeup or {}):
            allowed_cats = ["makeup"]
        elif (prof.hair or {}):
            allowed_cats = ["haircare"]
        else:
            allowed_cats = ["skincare"]

    category = allowed_cats[0]
    product_type = allowed_pts[0] if allowed_pts else None

    if offer.target_scope == "category":
        return {"scope": "category", "value": category}

    if offer.target_scope == "product_type":
        if product_type:
            return {"scope": "product_type", "value": product_type, "category": category}
        return {"scope": "category", "value": category}

    # product_id → choose top recommendation
    products = _load_products_for_recs()
    co = _cooccurrence_90d(now)
    owned_ids = list(
        OwnedProduct.objects.filter(user=user, is_active=True).values_list("product_id", flat=True)
    )
    recent_vals = _recent_target_values(user, days=7, now=now)

    recs = rec_recommend(
        prof=prof,
        products=products,
        owned_active_ids=owned_ids,
        context_product_ids=owned_ids[:50],
        category=category,
        product_type=product_type,
        limit=10,   
        co=co,
    )

    top = _pick_best_rec_target(user, recs, now=now, recent_vals=recent_vals)
    if top:
        p = top["product"]
        return {
            "scope": "product_id",
            "value": p["id"],
            "category": p["category"],
            "product_type": p["product_type"],
            "rec_score": top.get("score"),
            "rec_components": top.get("components", {}),
            "rec_why": (top.get("why") or [])[:6],
            "adjusted_score": top.get("adjusted_score"),
        }

    # degrade
    if product_type:
        return {"scope": "product_type", "value": product_type, "category": category}
    return {"scope": "category", "value": category}

def _derive_cross_sell_suggestions(post_ctx: dict | None) -> list[dict]:
    """
    Возвращает список желаемых целей (по приоритету):
    [{"category": "...", "product_type": "..."}, ...]
    """
    if not post_ctx:
        return []

    cats = set(post_ctx.get("categories") or [])
    pts = set(post_ctx.get("product_types") or [])

    out: list[dict] = []

    # FRAGRANCE
    if "fragrance" in cats:
        if "edp" in pts or "edt" in pts:
            out += [
                {"category": "fragrance", "product_type": "body_mist"},
                {"category": "fragrance", "product_type": "edt"},
            ]

    # HAIRCARE
    if "haircare" in cats:
        if "shampoo" in pts:
            out += [
                {"category": "haircare", "product_type": "conditioner"},
                {"category": "haircare", "product_type": "hair_mask"},
            ]
        if "conditioner" in pts:
            out += [
                {"category": "haircare", "product_type": "hair_mask"},
                {"category": "haircare", "product_type": "hair_oil"},
            ]

    # MAKEUP
    if "makeup" in cats:
        if "foundation" in pts:
            out += [
                {"category": "makeup", "product_type": "mascara"},
                {"category": "makeup", "product_type": "lipstick"},
                {"category": "makeup", "product_type": "blush"},
            ]
        if "lipstick" in pts:
            out += [
                {"category": "makeup", "product_type": "blush"},
                {"category": "makeup", "product_type": "eyeshadow"},
            ]
        if "mascara" in pts:
            out += [
                {"category": "makeup", "product_type": "eyeshadow"},
                {"category": "makeup", "product_type": "blush"},
            ]

    # SKINCARE
    if "skincare" in cats:
        if "serum" in pts or "cleanser" in pts:
            out += [
                {"category": "skincare", "product_type": "moisturizer"},
                {"category": "skincare", "product_type": "spf"},
            ]
        if "moisturizer" in pts:
            out += [{"category": "skincare", "product_type": "spf"}]

    # Удалим дубли, сохраним порядок
    uniq = []
    seen = set()
    for x in out:
        key = (x["category"], x["product_type"])
        if key not in seen:
            seen.add(key)
            uniq.append(x)
    return uniq

def _passes_category_cooldown(user, category: str, now: datetime, days: int = 3) -> bool:
    since = now - timedelta(days=days)
    # target — JSONField, Postgres умеет target__category
    return not OfferAssignment.objects.filter(
        user=user,
        assigned_at__gte=since,
        target__category=category,
    ).exists()

def _select_offer(user, now: datetime, context_steps: list[str] | None, post_ctx: dict | None):
    def _expected_category_for_offer(o: Offer, camp: CampaignBudget | None, context_steps: list[str] | None):
        if context_steps and "spf" in context_steps:
            allowed = _effective_allowed_categories(o, camp)
            if not allowed or "skincare" in allowed:
                return "skincare"

        allowed = _effective_allowed_categories(o, camp)
        if allowed:
            return allowed[0]
        return None

    # сегмент/рфм считаем один раз — и кладём в любые reason
    rfm = _rfm(user, now)
    seg = segment(RFM(**rfm))

    campaigns, routing = _campaign_candidates(context_steps, post_ctx)
    winback_eligible = _is_winback_eligible(user, now)
    favorite_cat = _favorite_category(user, now)
    if not campaigns:
        return None, {
            "segment": seg,
            "rfm": rfm,
            "picked_because": "no active campaigns",
            "context_steps": context_steps or None,
            "campaign_routing": routing,
            "winback_eligible": winback_eligible,
            "favorite_category": favorite_cat,
        }

    # сохранить ПОРЯДОК кампаний из routing, но при этом залочить строки
    ids_in_order = [c.id for c in campaigns]
    locked_map = {
        c.id: c
        for c in CampaignBudget.objects.select_for_update().filter(id__in=ids_in_order)
    }
    locked = [locked_map[i] for i in ids_in_order if i in locked_map]
    has_any_txn = _user_has_any_transactions(user)
    ordered_locked: list[CampaignBudget] = []
    if not has_any_txn:
        ordered_locked.extend([c for c in locked if _is_onboarding_first_order_campaign(c)])
    if winback_eligible:
        ordered_locked.extend([c for c in locked if _is_winback_30d_campaign(c) and c not in ordered_locked])
    if favorite_cat and not post_ctx and not context_steps:
        ordered_locked.extend([c for c in locked if _is_favorite_category_campaign(c) and c not in ordered_locked])
    ordered_locked.extend([c for c in locked if c not in ordered_locked])
    locked = ordered_locked

    # STRICT routing: идём по кампаниям в порядке locked (это routing order)
    for camp in locked:
        if _is_onboarding_first_order_campaign(camp) and has_any_txn:
            continue
        if _is_winback_30d_campaign(camp):
            if not winback_eligible:
                continue
            if not _passes_winback_assignment_cooldown(user, camp, now):
                continue
        if _is_favorite_category_campaign(camp):
            if post_ctx:
                continue
            if not favorite_cat:
                continue
            if not _passes_favorite_category_assignment_cooldown(user, camp, now):
                continue
        _reset_campaign_week_if_needed(camp, now)

        left = Decimal(str(camp.weekly_limit)) - Decimal(str(camp.weekly_spent))
        if left <= 0:
            continue

        offers = list(Offer.objects.filter(is_active=True, campaign=camp))
        if not offers:
            continue

        # ML propensity scores per offer within this campaign. Returns None if
        # the model is unavailable or disabled by flag; we then fall back to
        # pure rule-based scoring for this iteration.
        ml_enabled = bool(getattr(settings, "OFFER_REDEMPTION_ML_ENABLED", False))
        ml_weight = float(getattr(settings, "OFFER_REDEMPTION_ML_WEIGHT", 0.0) or 0.0)
        ml_probs: list[float] | None = None
        if ml_enabled and ml_weight > 0.0:
            ml_probs = offer_ml_scorer.score_offers(
                offers=offers,
                campaign_name=camp.name,
                rfm=rfm,
            )
        algo_used = "ml+rules" if ml_probs is not None else "rules"

        best_offer = None
        best_score = -1e9
        best_reason = None

        for idx, o in enumerate(offers):
            # пересечение ограничений offer/campaign по категориям
            eff_cats = _effective_allowed_categories(o, camp)
            if (o.allowed_categories or []) and (camp.allowed_categories or []) and not eff_cats:
                continue
            if _is_favorite_category_campaign(camp):
                if o.target_scope == "cart":
                    continue
                if eff_cats and favorite_cat not in eff_cats:
                    continue

            cost = Decimal(str(getattr(o, "estimated_cost", 0) or 0))
            if cost > left:
                continue
            if not _passes_cooldown(user, o, now):
                continue

            score = 0.0

            # segment boosts
            if seg == "at_risk" and o.offer_type == "discount":
                score += 2.5
            if seg in {"vip", "active"} and o.offer_type == "points_multiplier":
                score += 2.0
            if seg == "new_or_rare" and o.offer_type == "discount":
                score += 0.5

            # context boosts по steps (offer ∩ campaign)
            eff_steps = _effective_allowed_steps(o, camp)
            if context_steps and eff_steps and (set(eff_steps) & set(context_steps)):
                score += 10.0

            # campaign priority: слабый бонус (routing решает порядок, это просто tie-breaker)
            score += max(0.0, (100 - float(camp.priority))) * 0.01
            if _is_favorite_category_campaign(camp):
                score += 0.8

            # global effectiveness по ожидаемой категории
            exp_cat = _expected_category_for_offer(o, camp, context_steps)
            if _is_favorite_category_campaign(camp) and favorite_cat:
                exp_cat = favorite_cat
            cat_adj = 0.0
            cat_perf = None
            if exp_cat:
                cat_adj, cat_perf = category_uplift(exp_cat, now=now)
                score += cat_adj

                baseline = float(getattr(settings, "RECS_GLOBAL_BASELINE_CR", 0.02))
                if cat_perf and cat_perf.impressions >= int(getattr(settings, "RECS_GLOBAL_MIN_IMP", 20)):
                    if cat_perf.cr >= baseline:
                        if o.offer_type == "points_multiplier":
                            score += 0.6
                        if o.offer_type == "discount":
                            score -= 0.4
                    else:
                        if o.offer_type == "discount":
                            score += 0.6
                        if o.offer_type == "points_multiplier":
                            score -= 0.2

            # предпочтение дешёвых внутри кампании
            score += float(1.0 / (1.0 + float(cost))) * 0.5

            # tie-breaker: таргетные офферы
            if getattr(o, "target_scope", None) and o.target_scope != "cart":
                score += 0.05

            # ML propensity bonus. Model returns P(redeem | exposed, features).
            # We blend it into the rule score so business constraints still
            # dominate hard eligibility, but ML reshapes ties within a campaign.
            rule_score = score
            ml_prob: float | None = None
            if ml_probs is not None and idx < len(ml_probs):
                ml_prob = float(ml_probs[idx])
                score += ml_weight * ml_prob * 10.0

            if score > best_score:
                best_score = score
                best_offer = o
                best_reason = {
                    "segment": seg,
                    "rfm": rfm,
                    "picked_because": "best offer within first eligible campaign (routing order) under cooldown+budget+global_effectiveness",
                    "context_steps": context_steps or None,
                    "campaign": camp.name,
                    "campaign_left": float(left),
                    "campaign_priority": camp.priority,
                    "expected_category": exp_cat,
                    "category_adjust": round(cat_adj, 4),
                    "category_perf": (
                        {
                            "impressions": cat_perf.impressions,
                            "clicks": cat_perf.clicks,
                            "purchases": cat_perf.purchases,
                            "ctr": round(cat_perf.ctr, 4),
                            "cr": round(cat_perf.cr, 4),
                        }
                        if cat_perf else None
                    ),
                    "campaign_routing": routing,
                    "winback_eligible": winback_eligible,
                    "favorite_category": favorite_cat,
                    "algo_used": algo_used,
                    "rule_score": round(float(rule_score), 4),
                    "ml_prob": round(float(ml_prob), 6) if ml_prob is not None else None,
                    "final_score": round(float(score), 4),
                }

        # если в этой кампании нашли оффер — возвращаем СРАЗУ (это и есть “сначала пробуем …”)
        if best_offer:
            return (best_offer, camp), best_reason

    # если прошли все кампании и ничего не нашли
    return None, {
        "segment": seg,
        "rfm": rfm,
        "picked_because": "no eligible offers under constraints across routed campaigns",
        "context_steps": context_steps or None,
        "campaign_routing": routing,
        "winback_eligible": winback_eligible,
        "favorite_category": favorite_cat,
    }


def _to_int(v) -> int | None:
    try:
        if v is None:
            return None
        return int(v)
    except Exception:
        return None


def _roadmap_values(roadmap_ctx: dict | None) -> tuple[str, str, int | None]:
    cat = str((roadmap_ctx or {}).get("category") or "").strip()
    next_pt = str((roadmap_ctx or {}).get("next_product_type") or "").strip()
    next_pid = _to_int((roadmap_ctx or {}).get("next_product_id"))
    return cat, next_pt, next_pid


def _roadmap_reason_reference(
    *,
    roadmap_ctx: dict | None = None,
    reason: dict | None = None,
) -> dict[str, Any]:
    if isinstance(roadmap_ctx, dict) and roadmap_ctx:
        return dict(roadmap_ctx)

    src = reason if isinstance(reason, dict) else {}
    for key in ("roadmap", "roadmap_influence", "roadmap_ctx"):
        value = src.get(key)
        if isinstance(value, dict) and value:
            return dict(value)
    return {}


def _roadmap_reason_payload(roadmap_ctx: dict | None) -> dict[str, Any]:
    ctx = roadmap_ctx if isinstance(roadmap_ctx, dict) else {}
    payload: dict[str, Any] = {}

    category = str(ctx.get("category") or "").strip()
    if category:
        payload["category"] = category

    next_product_type = str(ctx.get("next_product_type") or "").strip()
    if next_product_type:
        payload["next_product_type"] = next_product_type

    plan_id = _to_int(ctx.get("plan_id"))
    if plan_id is not None:
        payload["plan_id"] = plan_id

    step_id = _to_int(ctx.get("step_id"))
    if step_id is not None:
        payload["step_id"] = step_id

    step_index = _to_int(ctx.get("step_index"))
    if step_index is not None:
        payload["step_index"] = step_index

    next_product_id = _to_int(ctx.get("next_product_id"))
    if next_product_id is not None:
        payload["next_product_id"] = next_product_id

    return payload


def _target_has_roadmap_shortcut(target: dict | None) -> bool:
    if not isinstance(target, dict):
        return False
    picked_via = str(target.get("picked_via") or "").strip()
    return picked_via.startswith("roadmap_shortcut")


def _roadmap_influence_payload(
    *,
    roadmap_ctx: dict | None,
    target: dict | None,
    target_matches_step: bool,
) -> dict[str, Any]:
    payload = _roadmap_reason_payload(roadmap_ctx)
    if not payload.get("next_product_type"):
        return {}

    tgt = target if isinstance(target, dict) else {}
    scope = str(tgt.get("scope") or "").strip()
    picked_via = str(tgt.get("picked_via") or "").strip()
    final_category = str(tgt.get("category") or "").strip()
    if not final_category and scope == "category":
        final_category = str(tgt.get("value") or "").strip()
    final_product_type = str(tgt.get("product_type") or "").strip()
    if not final_product_type and scope == "product_type":
        final_product_type = str(tgt.get("value") or "").strip()

    payload.update(
        {
            "link_type": "routing_only",
            "final_target_matches_step": bool(target_matches_step),
        }
    )
    if scope:
        payload["final_scope"] = scope
    if picked_via:
        payload["final_picked_via"] = picked_via
    if final_category:
        payload["final_category"] = final_category
    if final_product_type:
        payload["final_product_type"] = final_product_type
    payload["reason_code"] = (
        "matches_step_without_roadmap_shortcut"
        if target_matches_step
        else "campaign_or_fallback_target"
    )
    return payload


def enforce_assignment_roadmap_reason_contract(
    assignment: OfferAssignment,
    *,
    roadmap_ctx: dict | None = None,
    save: bool = True,
) -> bool:
    if not assignment:
        return False

    reason = assignment.reason if isinstance(getattr(assignment, "reason", None), dict) else {}
    target = assignment.target if isinstance(getattr(assignment, "target", None), dict) else {}
    reference = _roadmap_reason_reference(roadmap_ctx=roadmap_ctx, reason=reason)
    reference_payload = _roadmap_reason_payload(reference)

    new_reason = dict(reason)
    new_reason.pop("roadmap", None)
    new_reason.pop("roadmap_influence", None)

    if reference_payload.get("next_product_type"):
        target_matches_step = _target_matches_roadmap_step(target, reference)
        if _target_has_roadmap_shortcut(target) and target_matches_step:
            new_reason["roadmap"] = {
                **reference_payload,
                "link_type": "direct_target",
            }
        else:
            influence = _roadmap_influence_payload(
                roadmap_ctx=reference,
                target=target,
                target_matches_step=target_matches_step,
            )
            if influence:
                new_reason["roadmap_influence"] = influence

    changed = new_reason != reason
    if changed:
        assignment.reason = new_reason
        if save:
            assignment.save(update_fields=["reason"])
    return changed


def _target_matches_roadmap_step(target: dict | None, roadmap_ctx: dict | None) -> bool:
    if not isinstance(target, dict):
        return False
    roadmap_cat, roadmap_pt, roadmap_pid = _roadmap_values(roadmap_ctx)
    if not roadmap_pt:
        return False
    is_fragrance_slot = roadmap_cat == "fragrance" and roadmap_pt in FRAGRANCE_SLOTS

    scope = str(target.get("scope") or "").strip()
    tgt_cat = str(target.get("category") or "").strip()
    tgt_pt = str(target.get("product_type") or "").strip()
    tgt_val_pt = str(target.get("value") or "").strip() if scope == "product_type" else ""
    tgt_pid = _to_int(target.get("value")) if scope == "product_id" else None

    if roadmap_cat and tgt_cat and roadmap_cat != tgt_cat:
        return False

    if scope == "product_type":
        candidate_pt = tgt_val_pt or tgt_pt
        return candidate_pt == roadmap_pt

    if scope == "product_id":
        if roadmap_pid is not None and tgt_pid is not None and roadmap_pid == tgt_pid:
            if roadmap_cat and tgt_cat:
                return roadmap_cat == tgt_cat
            return True

        if is_fragrance_slot and tgt_pid is not None:
            p = Product.objects.filter(id=tgt_pid).values("category", "attrs", "raw_meta").first()
            if not p:
                return False
            if roadmap_cat and str(p.get("category")) != roadmap_cat:
                return False
            return slot_of_fragrance(
                p.get("attrs") or {},
                raw_meta=p.get("raw_meta") or {},
            ) == roadmap_pt

        if tgt_pt:
            if tgt_pt != roadmap_pt:
                return False
            if not roadmap_cat:
                return True
            if tgt_cat:
                return tgt_cat == roadmap_cat
            if tgt_pid is not None:
                p = Product.objects.filter(id=tgt_pid).values("category").first()
                return bool(p and str(p.get("category")) == roadmap_cat)
            return False

        if tgt_pid is None:
            return False
        p = Product.objects.filter(id=tgt_pid).values("category", "product_type").first()
        if not p:
            return False
        if str(p.get("product_type")) != roadmap_pt:
            return False
        if roadmap_cat and str(p.get("category")) != roadmap_cat:
            return False
        return True

    if scope == "" and tgt_pt:
        if tgt_pt != roadmap_pt:
            return False
        if roadmap_cat and tgt_cat:
            return roadmap_cat == tgt_cat
        return True

    return False


def _assignment_has_roadmap_shortcut_target(assignment: OfferAssignment) -> bool:
    target = assignment.target if isinstance(getattr(assignment, "target", None), dict) else {}
    return _target_has_roadmap_shortcut(target)


def expire_assignment_if_needed(
    assignment: OfferAssignment | None,
    *,
    now: datetime,
    source: str,
    request_id: str | None = None,
    save: bool = True,
) -> bool:
    if not assignment:
        return False

    expires_at = getattr(assignment, "expires_at", None)
    if not expires_at or expires_at > now:
        return False

    was_active = bool(getattr(assignment, "is_active", False))
    if was_active:
        assignment.is_active = False
        if save:
            assignment.save(update_fields=["is_active"])

    if was_active and not getattr(assignment, "is_redeemed", False):
        record_offer_event(
            assignment,
            OfferEvent.Type.EXPIRED,
            request_id=request_id,
            context={"source": source},
        )

    return True


def deactivate_stale_roadmap_assignment(
    assignment: OfferAssignment,
    *,
    now: datetime,
    save: bool = True,
) -> bool:
    del now
    if not assignment or not getattr(assignment, "is_active", False) or getattr(assignment, "is_redeemed", False):
        return False
    if not _assignment_has_roadmap_shortcut_target(assignment):
        return False

    reason = assignment.reason if isinstance(getattr(assignment, "reason", None), dict) else {}
    roadmap_reason = reason.get("roadmap") if isinstance(reason.get("roadmap"), dict) else {}
    target = assignment.target if isinstance(getattr(assignment, "target", None), dict) else {}
    roadmap_category = str(roadmap_reason.get("category") or target.get("category") or "").strip()
    if not roadmap_category:
        assignment.is_active = False
        if save:
            assignment.save(update_fields=["is_active"])
        return True

    from roadmap_app.services import get_active_plan, get_next_missing_step

    plan = get_active_plan(assignment.user, category=roadmap_category)
    step = get_next_missing_step(plan)
    if not plan or not step:
        assignment.is_active = False
        if save:
            assignment.save(update_fields=["is_active"])
        return True

    current_roadmap_ctx = {
        "category": roadmap_category,
        "plan_id": int(plan.id),
        "step_id": int(step.id),
        "step_index": int(step.step_index),
        "next_product_type": str(step.product_type or ""),
    }
    if step.recommended_product_id:
        current_roadmap_ctx["next_product_id"] = int(step.recommended_product_id)

    if _target_matches_roadmap_step(target, current_roadmap_ctx):
        return False

    assignment.is_active = False
    if save:
        assignment.save(update_fields=["is_active"])
    return True


def _can_supersede_existing(existing: OfferAssignment, roadmap_ctx: dict | None, now: datetime) -> bool:
    del now
    _, roadmap_pt, _ = _roadmap_values(roadmap_ctx)
    if not roadmap_pt:
        return False

    if _target_matches_roadmap_step(existing.target, roadmap_ctx):
        return False

    interacted = OfferEvent.objects.filter(
        assignment=existing,
        event_type__in=[OfferEvent.Type.CLICKED, OfferEvent.Type.REDEEMED],
    ).exists()
    if interacted:
        return False

    exposed_count = OfferEvent.objects.filter(
        assignment=existing,
        event_type=OfferEvent.Type.EXPOSED,
    ).count()
    max_exposed = int(getattr(settings, "OFFER_SUPERSEDE_MAX_EXPOSED", 1))
    if exposed_count > max_exposed:
        return False

    return True


def _refund_campaign_budget_for_superseded(existing: OfferAssignment, now: datetime) -> None:
    interacted = OfferEvent.objects.filter(
        assignment=existing,
        event_type__in=[OfferEvent.Type.CLICKED, OfferEvent.Type.REDEEMED],
    ).exists()
    if interacted:
        return

    camp_id = getattr(existing.offer, "campaign_id", None)
    if not camp_id:
        return

    camp = CampaignBudget.objects.select_for_update().filter(id=camp_id).first()
    if not camp:
        return

    _reset_campaign_week_if_needed(camp, now)
    cost = Decimal(str(getattr(existing.offer, "estimated_cost", 0) or 0))
    if cost <= 0:
        return

    current = Decimal(str(camp.weekly_spent or 0))
    camp.weekly_spent = max(Decimal("0"), current - cost)
    camp.save(
        update_fields=["weekly_spent"]
        + (["week_start_date"] if hasattr(camp, "week_start_date") else [])
    )


def get_or_assign_next_offer(
    user,
    now: datetime,
    context_steps: list[str] | None = None,
    post_ctx: dict | None = None,
    roadmap_ctx: dict | None = None,
) -> OfferAssignment | None:
    with db_tx.atomic():
        existing = (
            OfferAssignment.objects.select_for_update()
            .filter(user=user, is_active=True, is_redeemed=False)
            .select_related("offer")
            .order_by("-assigned_at")
            .first()
        )

        supersede_candidate: OfferAssignment | None = None
        if existing:
            existing_campaign = getattr(existing.offer, "campaign", None)
            if expire_assignment_if_needed(
                existing,
                now=now,
                source="get_or_assign_next_offer",
            ):
                existing.refresh_from_db(fields=["is_active"])
            if _is_onboarding_first_order_campaign(existing_campaign) and _user_has_any_transactions(user):
                existing.is_active = False
                existing.save(update_fields=["is_active"])
            if _is_winback_30d_campaign(existing_campaign) and not _is_winback_eligible(user, now):
                existing.is_active = False
                existing.save(update_fields=["is_active"])
            if _is_favorite_category_campaign(existing_campaign) and not _favorite_category(user, now):
                existing.is_active = False
                existing.save(update_fields=["is_active"])

            if existing.is_active and not existing.is_redeemed:
                if deactivate_stale_roadmap_assignment(existing, now=now):
                    existing.refresh_from_db(fields=["is_active"])
                if existing.is_active and not existing.is_redeemed:
                    enforce_assignment_roadmap_reason_contract(existing, save=True)
                    if not roadmap_ctx:
                        return existing
                    if not _can_supersede_existing(existing, roadmap_ctx=roadmap_ctx, now=now):
                        return existing
                    supersede_candidate = existing

        picked, reason = _select_offer(user, now, context_steps, post_ctx)
        if not picked:
            if existing and existing.is_active and not existing.is_redeemed:
                return existing
            return None

        offer, camp = picked

        if post_ctx:
            reason = {
                **(reason or {}),
                "post_purchase": {
                    "categories": post_ctx.get("categories"),
                    "product_types": post_ctx.get("product_types"),
                },
            }

        target = _pick_target_for_offer(
            user,
            offer,
            now,
            context_steps,
            post_ctx,
            campaign=camp,
            roadmap_ctx=roadmap_ctx,
        )

        # replacement requires new target that follows roadmap next-step
        if supersede_candidate and not _target_matches_roadmap_step(target, roadmap_ctx):
            return existing

        if isinstance(target, dict) and str(target.get("picked_via", "")).startswith("bundle"):
            reason = {
                **(reason or {}),
                "bundle": {
                    "based_on_product_id": target.get("based_on_product_id"),
                    "mode": target.get("bundle_mode"),
                    "why": target.get("bundle_why"),
                },
            }

        if supersede_candidate:
            supersede_candidate.is_active = False
            supersede_candidate.superseded_at = now
            supersede_candidate.save(update_fields=["is_active", "superseded_at"])
            record_offer_event(
                supersede_candidate,
                OfferEvent.Type.SUPERSEDED,
                request_id=None,
                context={
                    "source": "roadmap_post_purchase",
                    "roadmap_ctx": roadmap_ctx or {},
                },
            )
            _refund_campaign_budget_for_superseded(supersede_candidate, now=now)

        # spend campaign budget
        cost = Decimal(str(getattr(offer, "estimated_cost", 0) or 0))
        camp.weekly_spent = Decimal(str(camp.weekly_spent)) + cost
        camp.save(
            update_fields=["weekly_spent"]
            + (["week_start_date"] if hasattr(camp, "week_start_date") else [])
        )

        ttl_days = max(1, int(getattr(offer, "expires_in_days", 7) or 7))
        expires_at = now + timedelta(days=ttl_days)

        assignment = OfferAssignment.objects.create(
            user=user,
            offer=offer,
            reason=reason,
            target=target,
            expires_at=expires_at,
        )
        enforce_assignment_roadmap_reason_contract(
            assignment,
            roadmap_ctx=roadmap_ctx,
            save=True,
        )

        if supersede_candidate:
            supersede_candidate.superseded_by = assignment
            supersede_candidate.save(update_fields=["superseded_by"])

        record_offer_event(
            assignment,
            OfferEvent.Type.ASSIGNED,
            request_id=None,
            context={"source": "get_or_assign_next_offer"},
        )
        return assignment

def _pick_product_from_bundle(user, now: datetime, post_ctx: dict, category: str | None = None):
    """
    Возвращает target dict для product_id на основе bundle(cooc+fallback).
    """
    product_ids = post_ctx.get("product_ids") or []
    if not product_ids:
        return None

    cp, _ = CustomerProfile.objects.get_or_create(user=user)
    prof = _build_rec_profile(cp)
    recent_vals = _recent_target_values(user, days=7, now=now)

    products = _load_products_for_recs()
    co = _cooccurrence_90d(now)

    owned_ids = list(
        OwnedProduct.objects.filter(user=user, is_active=True).values_list("product_id", flat=True)
    )
    owned_set = set(owned_ids)

    # попробуем по 3 последним купленным товарам
    for base_id in list(product_ids)[-3:][::-1]:
        res = rec_bundle(
            products=products,
            base_product_id=int(base_id),
            owned_active_ids=owned_ids,
            prof=prof,
            co=co,
            limit=20,
        )
        # фильтры: category (если задана), saturation, in_stock уже учтён
        candidates = []

        for r in res:
            p = r["product"]
            pid = int(p["id"])

            if pid in recent_vals:
                continue
            if pid in owned_set:
                continue
            if category and p.get("category") != category:
                continue
            if _is_saturated(user, p["category"], p["product_type"]):
                continue

            candidates.append(r)
            if len(candidates) >= 20:
                break

        top = _pick_best_rec_target(user, candidates, now=now, recent_vals=recent_vals)
        if top:
            p = top["product"]
            return {
                "scope": "product_id",
                "value": p["id"],
                "category": p["category"],
                "product_type": p["product_type"],
                "bundle_mode": top.get("components", {}).get("mode"),
                "bundle_score": top.get("score"),
                "bundle_components": top.get("components", {}),
                "bundle_why": (top.get("why") or [])[:6],
                "adjusted_score": top.get("adjusted_score"),
                "picked_via": "bundle",
                "based_on_product_id": int(base_id),
            }

    return None

def _recent_target_values(user, days: int, now: datetime) -> set[int]:
    since = now - timedelta(days=days)
    vals = set()
    qs = OfferAssignment.objects.filter(user=user, assigned_at__gte=since).values_list("target", flat=True)
    for t in qs:
        if isinstance(t, dict) and t.get("scope") == "product_id" and t.get("value"):
            try:
                vals.add(int(t["value"]))
            except Exception:
                pass
    return vals

def _pick_best_rec_target(user, recs: list[dict], *, now: datetime, recent_vals: set[int] | None = None):
    recent_vals = recent_vals or set()
    adjusted = adjust_recs(user, recs, now=now)

    for r in adjusted:
        p = r.get("product") or {}
        pid_raw = p.get("id")
        if not pid_raw:
            continue
        pid = int(pid_raw)

        # anti-repeat
        if pid in recent_vals:
            continue

        # fatigue hard-skip
        fat = (r.get("components") or {}).get("fatigue") or {}
        is_fatigued = bool(fat.get("fatigued"))
        if getattr(settings, "RECS_FATIGUE_HARD_SKIP", True) and is_fatigued:
            continue

        # global hard-skip
        glob = (r.get("components") or {}).get("global") or {}
        g_imp = int(glob.get("impressions") or 0)
        g_cr = float(glob.get("cr") or 0.0)
        if getattr(settings, "RECS_GLOBAL_HARD_SKIP", False):
            if g_imp >= int(getattr(settings, "RECS_GLOBAL_MIN_IMP", 20)) and g_cr <= float(
                getattr(settings, "RECS_GLOBAL_HARD_SKIP_CR", 0.002)
            ):
                continue

        return r

    return adjusted[0] if adjusted else None

