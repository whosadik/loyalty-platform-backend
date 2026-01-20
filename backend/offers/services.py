from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta, datetime
from decimal import Decimal
from typing import Any

from django.db import transaction as db_tx
from django.utils import timezone
from django.db.models import Sum

from catalog.models import Product
from transactions.models import Transaction, TransactionItem, OwnedProduct
from users_app.models import CustomerProfile
from offers.models import Offer, OfferAssignment, CampaignBudget

from ml_logic.next_best_reward import RFM, segment  # compute_rfm можно не трогать
from ml_logic.recommender import (
    UserProfile as RecUserProfile,
    recommend as rec_recommend,
    build_cooccurrence,
)
from django.db.models import Count

def _week_start(d: datetime) -> datetime.date:
    # Monday as week start
    return (d - timedelta(days=d.weekday())).date()


def _get_budget_locked(now: datetime) -> CampaignBudget:
    # single row budget (MVP). If you have multiple campaigns, adapt.
    b, _ = CampaignBudget.objects.select_for_update().get_or_create(
        id=1, defaults={"weekly_limit": Decimal("1000.0"), "weekly_spent": Decimal("0.0")}
    )

    # optional weekly reset if you already added week_start_date
    if hasattr(b, "week_start_date"):
        ws = _week_start(now)
        if b.week_start_date != ws:
            b.week_start_date = ws
            b.weekly_spent = Decimal("0.0")
            b.save(update_fields=["week_start_date", "weekly_spent"])

    return b


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
    return list(
        Product.objects.all().values(
            "id",
            "name",
            "brand",
            "price",
            "category",
            "product_type",
            "concerns",
            "attrs",
            "actives",
            "flags",
            "supported_skin_types",
            "strength",
            "in_stock",
        )
    )


def _cooccurrence_90d(now: datetime):
    since = now - timedelta(days=90)
    rows = (
        TransactionItem.objects.filter(transaction__created_at__gte=since)
        .values("transaction_id", "product_id")
    )
    txn_map: dict[int, list[int]] = {}
    for r in rows:
        txn_map.setdefault(r["transaction_id"], []).append(r["product_id"])
    return build_cooccurrence(list(txn_map.values()))


def _passes_cooldown(user, offer: Offer, now: datetime) -> bool:
    cd = int(getattr(offer, "cooldown_days", 0) or 0)
    if cd <= 0:
        return True
    since = now - timedelta(days=cd)
    return not OfferAssignment.objects.filter(
        user=user, offer=offer, is_redeemed=True, assigned_at__gte=since
    ).exists()

# Порог "уже достаточно" по категориям (MVP)
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

def _pick_target_for_offer(user, offer: Offer, now: datetime, context_steps: list[str] | None, post_ctx: dict | None):
    # 1) routine-context shortcut
    if context_steps and offer.target_scope in {"product_type", "product_id"}:
        if "spf" in context_steps:
            # if offer allows skincare or doesn't restrict
            allowed = offer.allowed_categories or []
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
                recs = rec_recommend(
                    prof=prof,
                    products=products,
                    owned_active_ids=owned_ids,
                    context_product_ids=owned_ids[:50],
                    category="skincare",
                    product_type="spf",
                    limit=1,
                    co=co,
                )
                if recs:
                    top = recs[0]
                    p = top["product"]
                    return {
                        "scope": "product_id",
                        "value": p["id"],
                        "category": p["category"],
                        "product_type": p["product_type"],
                        "rec_score": top.get("score"),
                        "rec_components": top.get("components", {}),
                        "rec_why": (top.get("why") or [])[:6],
                    }

    # 2) if cart
    if offer.target_scope == "cart":
        return {"scope": "cart"}

    # 3) explicit restrictions
    allowed_cats = offer.allowed_categories or []
    allowed_pts = offer.allowed_product_types or []

    cp, _ = CustomerProfile.objects.get_or_create(user=user)
    prof = _build_rec_profile(cp)

    suggestions = _derive_cross_sell_suggestions(post_ctx)

    # если offer ограничивает allowed_categories/types — отфильтруем кандидаты
    filtered = []
    for s in suggestions:
        if offer.allowed_categories and s["category"] not in (offer.allowed_categories or []):
            continue
        if offer.allowed_product_types and s["product_type"] not in (offer.allowed_product_types or []):
            # если offer scope=category — можно оставить кандидата по категории
            if offer.target_scope == "category":
                pass
            else:
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

            recs = rec_recommend(
                prof=prof,
                products=products,
                owned_active_ids=owned_ids,
                context_product_ids=(post_ctx.get("product_ids") or owned_ids)[:50],
                category=cat,
                product_type=pt,
                limit=1,
                co=co,
            )
            if recs:
                top = recs[0]
                p = top["product"]
                return {
                    "scope": "product_id",
                    "value": p["id"],
                    "category": p["category"],
                    "product_type": p["product_type"],
                    "rec_score": top.get("score"),
                    "rec_components": top.get("components", {}),
                    "rec_why": (top.get("why") or [])[:6],
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
        return {"scope": "product_type", "value": product_type, "category": category}

    # product_id → choose top recommendation
    products = _load_products_for_recs()
    co = _cooccurrence_90d(now)
    owned_ids = list(
        OwnedProduct.objects.filter(user=user, is_active=True).values_list("product_id", flat=True)
    )
    recs = rec_recommend(
        prof=prof,
        products=products,
        owned_active_ids=owned_ids,
        context_product_ids=owned_ids[:50],
        category=category,
        product_type=product_type,
        limit=1,
        co=co,
    )
    if recs:
        top = recs[0]
        p = top["product"]
        return {
            "scope": "product_id",
            "value": p["id"],
            "category": p["category"],
            "product_type": p["product_type"],
            "rec_score": top.get("score"),
            "rec_components": top.get("components", {}),
            "rec_why": (top.get("why") or [])[:6],
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

def _select_offer(user, now: datetime, context_steps: list[str] | None):
    # Pick best offer under: is_active + cooldown + budget.
    offers = list(Offer.objects.filter(is_active=True))

    budget = _get_budget_locked(now)
    left = Decimal(str(budget.weekly_limit)) - Decimal(str(budget.weekly_spent))

    best = None
    best_score = -1e9

    # segment from RFM
    rfm = _rfm(user, now)
    seg = segment(RFM(**rfm))

    for o in offers:
        cost = Decimal(str(getattr(o, "estimated_cost", 0) or 0))
        if cost > left:
            continue
        if not _passes_cooldown(user, o, now):
            continue

        # simple scoring (MVP). You can refine later.
        score = 0.0

        # context boosts
        if context_steps and (o.allowed_steps or []):
            if set(o.allowed_steps).intersection(set(context_steps)):
                score += 10.0

        # segment boosts
        if seg == "at_risk" and o.offer_type == "discount":
            score += 2.0
        if seg in {"vip", "active"} and o.offer_type == "points_multiplier":
            score += 2.0

        # cheaper offers slightly preferred to stretch budget
        score += float(1.0 / (1.0 + float(cost)))

        if score > best_score:
            best_score = score
            best = o

    if not best:
        return None, {"segment": seg, "rfm": rfm, "picked_because": "no eligible offers under constraints"}

    return best, {
        "segment": seg,
        "rfm": rfm,
        "picked_because": "max(score) under eligibility + cooldown + budget constraints",
        "context_steps": context_steps or None,
    }


def get_or_assign_next_offer(
    user,
    now: datetime,
    context_steps: list[str] | None = None,
    post_ctx: dict | None = None,
) -> OfferAssignment | None:
    """
    Returns existing active assignment if present, else creates a new one.
    Must be called inside an atomic block if you want strict budget consistency.
    """
    # 1) if user already has active unredeemed offer, reuse it
    existing = (
        OfferAssignment.objects.filter(user=user, is_redeemed=False)
        .select_related("offer")
        .order_by("-assigned_at")
        .first()
    )
    if existing and (not existing.expires_at or existing.expires_at > now):
        return existing

    # 2) select offer under constraints (locks budget row)
    offer, reason = _select_offer(user, now, context_steps)
    if not offer:
        return None
    
    if post_ctx:
        reason = {
            **(reason or {}),
            "post_purchase": {
                "categories": post_ctx.get("categories"),
                "product_types": post_ctx.get("product_types"),
            },
        }

    target = _pick_target_for_offer(user, offer, now, context_steps, post_ctx)

    # 3) create assignment and spend budget
    budget = _get_budget_locked(now)
    cost = Decimal(str(getattr(offer, "estimated_cost", 0) or 0))
    budget.weekly_spent = Decimal(str(budget.weekly_spent)) + cost
    budget.save(update_fields=["weekly_spent"] + (["week_start_date"] if hasattr(budget, "week_start_date") else []))

    expires_at = now + timedelta(days=int(getattr(offer, "expires_in_days", 7) or 7)) if hasattr(OfferAssignment, "expires_at") else None

    assignment = OfferAssignment.objects.create(
        user=user,
        offer=offer,
        reason=reason,
        target=target,
        **({"expires_at": expires_at} if hasattr(OfferAssignment, "expires_at") else {}),
    )
    return assignment
