from __future__ import annotations
from django.conf import settings
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
    key = "recs:products:v1"
    cached = cache.get(key)
    if cached is not None:
        return cached

    data = list(
        Product.objects.all().values(
            "id","name","brand","price","category","product_type",
            "concerns","attrs","actives","flags","supported_skin_types","strength","in_stock",
        )
    )
    cache.set(key, data, timeout=600)  # 10 минут
    return data



def _cooccurrence_90d(now: datetime):
    key = "recs:cooc90d:v1"
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
):
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
        bundle_cat = allowed_cats[0] if allowed_cats else None

        t = _pick_product_from_bundle(user, now, post_ctx, category=bundle_cat)
        if t:
            return t

    # 3) explicit restrictions
    allowed_cats = _effective_allowed_categories(offer, campaign)
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
    if not campaigns:
        return None, {
            "segment": seg,
            "rfm": rfm,
            "picked_because": "no active campaigns",
            "context_steps": context_steps or None,
            "campaign_routing": routing,
        }

    # сохранить ПОРЯДОК кампаний из routing, но при этом залочить строки
    ids_in_order = [c.id for c in campaigns]
    locked_map = {
        c.id: c
        for c in CampaignBudget.objects.select_for_update().filter(id__in=ids_in_order)
    }
    locked = [locked_map[i] for i in ids_in_order if i in locked_map]

    # STRICT routing: идём по кампаниям в порядке locked (это routing order)
    for camp in locked:
        _reset_campaign_week_if_needed(camp, now)

        left = Decimal(str(camp.weekly_limit)) - Decimal(str(camp.weekly_spent))
        if left <= 0:
            continue

        offers = list(Offer.objects.filter(is_active=True, campaign=camp))
        if not offers:
            continue

        best_offer = None
        best_score = -1e9
        best_reason = None

        for o in offers:
            # пересечение ограничений offer/campaign по категориям
            eff_cats = _effective_allowed_categories(o, camp)
            if (o.allowed_categories or []) and (camp.allowed_categories or []) and not eff_cats:
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

            # global effectiveness по ожидаемой категории
            exp_cat = _expected_category_for_offer(o, camp, context_steps)
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
    }


def get_or_assign_next_offer(
    user,
    now: datetime,
    context_steps: list[str] | None = None,
    post_ctx: dict | None = None,
) -> OfferAssignment | None:
    existing = (
        OfferAssignment.objects.filter(user=user, is_redeemed=False)
        .select_related("offer")
        .order_by("-assigned_at")
        .first()
    )

    if existing:
        if existing.expires_at and existing.expires_at <= now:
            existing.is_redeemed = True
            existing.save(update_fields=["is_redeemed"])
        else:
            return existing

    # всё, что связано с select_for_update + списанием бюджета — в atomic
    with db_tx.atomic():
        picked, reason = _select_offer(user, now, context_steps, post_ctx)
        if not picked:
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

        target = _pick_target_for_offer(user, offer, now, context_steps, post_ctx, campaign=camp)

        if isinstance(target, dict) and str(target.get("picked_via", "")).startswith("bundle"):
            reason = {
                **(reason or {}),
                "bundle": {
                    "based_on_product_id": target.get("based_on_product_id"),
                    "mode": target.get("bundle_mode"),
                    "why": target.get("bundle_why"),
                },
            }

        # spend campaign budget
        cost = Decimal(str(getattr(offer, "estimated_cost", 0) or 0))
        camp.weekly_spent = Decimal(str(camp.weekly_spent)) + cost
        camp.save(
            update_fields=["weekly_spent"]
            + (["week_start_date"] if hasattr(camp, "week_start_date") else [])
        )

        ttl_days = int(getattr(offer, "expires_in_days", 7) or 7)
        expires_at = now + timedelta(days=ttl_days)

        assignment = OfferAssignment.objects.create(
            user=user,
            offer=offer,
            reason=reason,
            target=target,
            expires_at=expires_at,
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
