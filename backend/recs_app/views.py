from __future__ import annotations

from datetime import timedelta
from decimal import Decimal
from typing import Any

from django.core.cache import cache
from django.db.models import Sum
from django.utils import timezone as dj_timezone
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated
from rest_framework import serializers

from backend.throttles import RecsRateThrottle
from catalog.models import Product
from catalog.serializers import ProductSerializer
from transactions.models import OwnedProduct, TransactionItem, Transaction
from users_app.models import CustomerProfile

from ml_logic.recommender import UserProfile, bundle

from drf_spectacular.utils import extend_schema, OpenApiExample, OpenApiParameter
from drf_spectacular.types import OpenApiTypes

from backend.request_language import get_request_language
from ml_logic.recommender import bundle as rec_bundle
from recs_analytics.experiment import build_event_experiment_context
from recs_analytics.models import RecommendationEvent
from recs_app.reranker import (
    get_reranker_model_version,
    get_runtime_co_map,
    recommend_with_algo,
    rerank_bundle_with_algo,
)

# Use existing offer helpers for profile/products loading.
from offers.services import _build_rec_profile, _load_products_for_recs

HOME_SECTION_TITLES = {
    "ru": {
        "for_you": "Для вас",
        "because_you_bought": "Потому что вы купили",
        "trending": "Популярное",
    },
    "kk": {
        "for_you": "Сізге арналған",
        "because_you_bought": "Сатып алғаныңызға байланысты",
        "trending": "Танымал",
    },
    "en": {
        "for_you": "For you",
        "because_you_bought": "Because you bought",
        "trending": "Trending",
    },
}


class RecommendationsQuerySerializer(serializers.Serializer):
    category = serializers.ChoiceField(
        choices=["skincare", "haircare", "makeup", "fragrance"],
        required=False,
        allow_null=True,
    )
    product_type = serializers.CharField(required=False, allow_blank=True)
    limit = serializers.IntegerField(required=False, min_value=1, max_value=50, default=10)
    algo = serializers.ChoiceField(choices=["cooc", "reranker", "auto"], required=False)


class BundleQuerySerializer(serializers.Serializer):
    product_id = serializers.IntegerField()
    limit = serializers.IntegerField(required=False, min_value=1, max_value=50, default=10)
    algo = serializers.ChoiceField(choices=["cooc", "reranker", "auto"], required=False)


def _dedupe_recent_product_ids(raw_ids: list[int], *, k: int) -> list[int]:
    """
    Input order: newest -> oldest.
    Output order: oldest -> newest (expected by recency-weighted retrieval).
    """
    out: list[int] = []
    seen: set[int] = set()
    lim = max(1, int(k or 1))
    for raw in raw_ids:
        try:
            pid = int(raw)
        except Exception:
            continue
        if pid in seen:
            continue
        seen.add(pid)
        out.append(pid)
        if len(out) >= lim:
            break
    out.reverse()
    return out


def get_user_context_product_ids(user, k: int = 50) -> tuple[list[int], dict[str, Any]]:
    fetch_n = max(int(k) * 20, 200)

    purchase_raw = list(
        TransactionItem.objects.filter(transaction__user=user)
        .order_by("-transaction__created_at", "-id")
        .values_list("product_id", flat=True)[:fetch_n]
    )
    purchase_ctx = _dedupe_recent_product_ids(purchase_raw, k=k)
    if purchase_ctx:
        return purchase_ctx, {
            "context_source": "purchases",
            "context_len": len(purchase_ctx),
            "context_k_used": int(k),
        }

    behavior_raw = list(
        RecommendationEvent.objects.filter(
            user=user,
            action__in=[
                RecommendationEvent.Action.ADD_TO_CART,
                RecommendationEvent.Action.CLICK,
                RecommendationEvent.Action.PURCHASE_ATTRIBUTED,
            ],
        )
        .order_by("-created_at", "-id")
        .values_list("product_id", flat=True)[:fetch_n]
    )
    behavior_ctx = _dedupe_recent_product_ids(behavior_raw, k=k)
    if behavior_ctx:
        return behavior_ctx, {
            "context_source": "behavior",
            "context_len": len(behavior_ctx),
            "context_k_used": int(k),
        }

    return [], {
        "context_source": "none",
        "context_len": 0,
        "context_k_used": int(k),
    }


def _build_profile(cp: CustomerProfile) -> UserProfile:
    return UserProfile(
        skin_type=cp.skin_type,
        goals=cp.goals or [],
        avoid_flags=cp.avoid_flags or [],
        budget=cp.budget,
        hair=cp.hair_profile or {},
        makeup=cp.makeup_profile or {},
        fragrance=cp.fragrance_profile or {},
    )


def _load_products():
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



class MeRecommendationsView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request):
        language = get_request_language(request)
        section_titles = HOME_SECTION_TITLES[language]
        q = RecommendationsQuerySerializer(data=request.query_params)
        q.is_valid(raise_exception=True)

        category = q.validated_data.get("category")
        product_type = (q.validated_data.get("product_type") or "").strip() or None
        limit = q.validated_data["limit"]
        algo_requested = q.validated_data.get("algo")

        cp, _ = CustomerProfile.objects.get_or_create(user=request.user)
        prof = _build_rec_profile(cp)

        owned_active_ids = list(
            OwnedProduct.objects.filter(user=request.user, is_active=True)
            .values_list("product_id", flat=True)
        )

        # Runtime context is built from user purchases/events.
        context_ids, context_meta = get_user_context_product_ids(request.user, k=50)

        products = _load_products()
        co, co_source = get_runtime_co_map()

        results, algo_used, model_version, algo_routing = recommend_with_algo(
            user_id=request.user.id,
            prof=prof,
            products=products,
            owned_active_ids=owned_active_ids,
            context_product_ids=context_ids,
            category=category,
            product_type=product_type,
            limit=limit,
            co=co,
            algo_requested=algo_requested,
            co_source=co_source,
        )
        algo_routing = dict(algo_routing or {})
        algo_routing.update(context_meta)

        return Response(
            {
                "query": {
                    "category": category,
                    "product_type": product_type,
                    "limit": limit,
                    "algo_requested": algo_requested,
                    "algo_used": algo_used,
                    "model_version": model_version,
                    "algo_routing": algo_routing,
                },
                "context": {
                    "owned_active_count": len(owned_active_ids),
                    **context_meta,
                },
                "results": _enrich_results_with_catalog_payload(results),
            }
        )


class MeBundleView(APIView):
    permission_classes = [IsAuthenticated]
    throttle_classes = [RecsRateThrottle]
    @extend_schema(
        tags=["Recommendations"],
        parameters=[
            OpenApiParameter("product_id", OpenApiTypes.INT, required=True, description="Base product id"),
            OpenApiParameter("limit", OpenApiTypes.INT, required=False, description="Max results (default 10)"),
            OpenApiParameter("algo", OpenApiTypes.STR, required=False, description="cooc, reranker or auto"),
        ],
        responses={200: OpenApiTypes.OBJECT},
        examples=[
            OpenApiExample(
                "Bundle response (sample)",
                response_only=True,
                value={
                    "query": {"product_id": 330, "limit": 10},
                    "results": [
                        {
                            "product": {"id": 322, "name": "Eyeshadow 17", "category": "makeup", "product_type": "eyeshadow"},
                            "score": 1.0,
                            "components": {"mode": "cooccurrence", "cooccurrence": 1},
                            "why": ["frequently purchased with product_id=330 (count=1)"],
                        },
                        {
                            "product": {"id": 282, "name": "Blush 2", "category": "makeup", "product_type": "blush"},
                            "score": 1.7,
                            "components": {"mode": "fallback", "similarity": 1.3, "content": 0.8},
                            "why": ["no/weak co-occurrence yet; showing similar items", "same brand", "same finish=matte"],
                        },
                    ],
                },
            ),
        ],
    )
    def get(self, request):
        q = BundleQuerySerializer(data=request.query_params)
        q.is_valid(raise_exception=True)

        base_product_id = q.validated_data["product_id"]
        limit = q.validated_data["limit"]
        algo_requested = q.validated_data.get("algo")

        cp, _ = CustomerProfile.objects.get_or_create(user=request.user)
        prof = _build_profile(cp)

        owned_active_ids = list(
            OwnedProduct.objects.filter(user=request.user, is_active=True)
            .values_list("product_id", flat=True)
        )

        products = _load_products()
        co, co_source = get_runtime_co_map()

        results_raw = bundle(
            products=products,
            base_product_id=base_product_id,
            owned_active_ids=owned_active_ids,
            prof=prof,
            co=co,
            limit=limit * 3,
        )
        results, algo_used, model_version, algo_routing = rerank_bundle_with_algo(
            user_id=request.user.id,
            base_product_id=base_product_id,
            bundle_results=results_raw,
            products=products,
            co=co,
            limit=limit,
            algo_requested=algo_requested,
            owned_active_ids=owned_active_ids,
            co_source=co_source,
        )

        return Response(
            {
                "query": {
                    "product_id": base_product_id,
                    "limit": limit,
                    "algo_requested": algo_requested,
                    "algo_used": algo_used,
                    "model_version": model_version,
                    "algo_routing": algo_routing,
                },
                "results": _enrich_results_with_catalog_payload(results),
            }
        )

def _to_int(v, default=None):
    try:
        return int(v)
    except Exception:
        return default


def _to_decimal(v, default=None):
    try:
        return Decimal(str(v))
    except Exception:
        return default


def _product_obj_to_dict(p: Product) -> dict[str, Any]:
    return {
        "id": p.id,
        "name": p.name,
        "brand": p.brand,
        "price": float(p.price),
        "currency": p.currency,
        "category": p.category,
        "product_type": p.product_type,
        "concerns": p.concerns or [],
        "attrs": p.attrs or {},
        "actives": p.actives or [],
        "flags": p.flags or [],
        "supported_skin_types": p.supported_skin_types or [],
        "strength": p.strength,
        "in_stock": bool(p.in_stock),
        "image_url": p.image_url or None,
        "image_urls": list(p.image_urls or []),
        "raw_meta": p.raw_meta or {},
    }


def _enrich_results_with_catalog_payload(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    product_ids: list[int] = []
    for row in results:
        product = row.get("product") or {}
        product_id = product.get("id")
        if product_id is None:
            continue
        try:
            product_ids.append(int(product_id))
        except Exception:
            continue

    if not product_ids:
        return results

    serialized = ProductSerializer(Product.objects.filter(id__in=product_ids), many=True).data
    by_id = {int(item["id"]): item for item in serialized}

    enriched: list[dict[str, Any]] = []
    for row in results:
        product = row.get("product") or {}
        try:
            product_id = int(product.get("id"))
        except Exception:
            enriched.append(row)
            continue

        payload = by_id.get(product_id)
        if payload is None:
            enriched.append(row)
            continue

        enriched.append({**row, "product": payload})

    return enriched


def _apply_filters_to_product_dict(p: dict[str, Any], *, category=None, product_type=None, price_min=None, price_max=None) -> bool:
    if category and p.get("category") != category:
        return False
    if product_type and p.get("product_type") != product_type:
        return False
    price = _to_decimal(p.get("price"))
    if price_min is not None and price is not None and price < price_min:
        return False
    if price_max is not None and price is not None and price > price_max:
        return False
    if p.get("in_stock") is False:
        return False
    return True


def _dedupe_limit(results: list[dict[str, Any]], seen: set[int], limit: int) -> list[dict[str, Any]]:
    out = []
    for r in results:
        pid = (r.get("product") or {}).get("id")
        if pid is None:
            continue
        pid = int(pid)
        if pid in seen:
            continue
        seen.add(pid)
        out.append(r)
        if len(out) >= limit:
            break
    return out


def _recs_event_context(
    *,
    algo_requested: str | None,
    algo_used: str | None,
    model_version: str | None,
    routing: dict[str, Any] | None,
    section_key: str,
) -> dict[str, Any]:
    out = build_event_experiment_context(
        algo_requested=algo_requested,
        algo_used=algo_used,
        model_version=model_version,
        routing=routing,
    )
    out["section"] = section_key
    return out


def _trending_30d(*, now, category=None, product_type=None, price_min=None, price_max=None, limit=10) -> list[dict[str, Any]]:
    """
    Trending = top products by quantity in last 30 days.
    Cache 10 min.
    """
    key = f"recs:trending30d:v1:{category or 'any'}:{product_type or 'any'}:{price_min or 'none'}:{price_max or 'none'}:{limit}"
    cached = cache.get(key)
    if cached is not None:
        return cached

    since = now - timedelta(days=30)

    items = TransactionItem.objects.filter(transaction__created_at__gte=since, product__in_stock=True)

    if category:
        items = items.filter(product__category=category)
    if product_type:
        items = items.filter(product__product_type=product_type)

    top = (
        items.values("product_id")
        .annotate(qty=Sum("quantity"))
        .order_by("-qty")[: max(limit * 3, 30)]
    )

    ids = [row["product_id"] for row in top]
    qty_map = {row["product_id"]: int(row["qty"] or 0) for row in top}

    products = list(Product.objects.filter(id__in=ids, in_stock=True))
    prod_map = {p.id: p for p in products}

    out: list[dict[str, Any]] = []
    for pid in ids:
        p = prod_map.get(pid)
        if not p:
            continue
        pd = _product_obj_to_dict(p)
        if not _apply_filters_to_product_dict(pd, category=category, product_type=product_type, price_min=price_min, price_max=price_max):
            continue

        out.append({
            "product": pd,
            "score": float(qty_map.get(pid, 0)),
            "components": {"mode": "trending", "qty_30d": qty_map.get(pid, 0)},
            "why": [f"top-selling in last 30d (qty={qty_map.get(pid, 0)})"],
        })
        if len(out) >= limit:
            break

    cache.set(key, out, timeout=600)
    return out


class HomeRecommendationsView(APIView):
    permission_classes = [IsAuthenticated]
    throttle_classes = [RecsRateThrottle]

    def get(self, request):
        language = get_request_language(request)
        section_titles = HOME_SECTION_TITLES[language]
        qp = request.query_params
        limit = _to_int(qp.get("limit"), 10) or 10
        limit = max(1, min(limit, 50))

        category = qp.get("category") or None
        product_type = qp.get("product_type") or None
        price_min = _to_decimal(qp.get("price_min"))
        price_max = _to_decimal(qp.get("price_max"))
        algo_requested = (qp.get("algo") or "").strip().lower() or None

        now = dj_timezone.now()

        cp, _ = CustomerProfile.objects.get_or_create(user=request.user)
        prof = _build_rec_profile(cp)

        products = _load_products_for_recs()
        co, co_source = get_runtime_co_map()

        owned_ids = list(
            OwnedProduct.objects.filter(user=request.user, is_active=True).values_list("product_id", flat=True)
        )
        context_ids, context_meta = get_user_context_product_ids(request.user, k=50)

        # last purchase context
        last_txn = Transaction.objects.filter(user=request.user).order_by("-created_at").first()
        base_product_ids: list[int] = []
        if last_txn:
            base_product_ids = list(
                TransactionItem.objects.filter(transaction=last_txn).values_list("product_id", flat=True)[:10]
            )

        seen: set[int] = set()

        # 1) For you (profile-based)
        for_you_raw, for_you_algo_used, for_you_model_version, for_you_routing = recommend_with_algo(
            user_id=request.user.id,
            prof=prof,
            products=products,
            owned_active_ids=owned_ids,
            context_product_ids=context_ids,
            category=category,
            product_type=product_type,
            limit=limit * 2,
            co=co,
            algo_requested=algo_requested,
            co_source=co_source,
        )
        for_you_routing = dict(for_you_routing or {})
        for_you_routing.update(context_meta)

        for_you = []
        for r in for_you_raw:
            p = r.get("product") or {}
            if not _apply_filters_to_product_dict(p, category=category, product_type=product_type, price_min=price_min, price_max=price_max):
                continue
            for_you.append(r)
        for_you = _dedupe_limit(for_you, seen, limit)

        # 2) Because you bought (bundle)
        because = []
        because_algo_used = "cooc"
        because_model_version = None
        if base_product_ids:
            base_id = int(base_product_ids[0])
            bundle_raw = rec_bundle(
                products=products,
                base_product_id=base_id,
                owned_active_ids=owned_ids,
                prof=prof,
                co=co,
                limit=limit * 2,
            )
            filtered = []
            for r in bundle_raw:
                p = r.get("product") or {}
                if not _apply_filters_to_product_dict(p, category=category, product_type=product_type, price_min=price_min, price_max=price_max):
                    continue
                filtered.append(r)
            reranked, because_algo_used, because_model_version, because_routing = rerank_bundle_with_algo(
                user_id=request.user.id,
                base_product_id=base_id,
                bundle_results=filtered,
                products=products,
                co=co,
                limit=limit * 2,
                algo_requested=algo_requested,
                owned_active_ids=owned_ids,
                co_source=co_source,
            )
            because = _dedupe_limit(reranked, seen, limit)
        else:
            because_routing = {"source": "none"}

        # 3) Trending
        trending_raw = _trending_30d(
            now=now,
            category=category,
            product_type=product_type,
            price_min=price_min,
            price_max=price_max,
            limit=limit,
        )
        # exclude owned + dedupe with global seen
        owned_set = set(map(int, owned_ids))
        trending_filtered = []
        for r in trending_raw:
            pid = int((r.get("product") or {}).get("id"))
            if pid in owned_set:
                continue
            trending_filtered.append(r)
        trending = _dedupe_limit(trending_filtered, seen, limit)

        event_product_ids = set()
        for section in (for_you, because, trending):
            for row in section:
                pid = (row.get("product") or {}).get("id")
                if pid is not None:
                    event_product_ids.add(int(pid))
        existing_event_ids = set(Product.objects.filter(id__in=event_product_ids).values_list("id", flat=True))

        events = []
        rid = getattr(request, "request_id", None)

        def push_impressions(
            section_key: str,
            results: list[dict],
            page: str = "home",
            algo_mode_override: str | None = None,
            static_context: dict[str, Any] | None = None,
        ):
            for rank, r in enumerate(results, start=1):
                p = r.get("product") or {}
                pid = p.get("id")
                if not pid:
                    continue
                if int(pid) not in existing_event_ids:
                    continue
                comps = r.get("components") or {}
                ctx = {"why": (r.get("why") or [])[:6], "rank": rank}
                if static_context:
                    ctx.update(static_context)
                events.append(RecommendationEvent(
                    user=request.user,
                    action=RecommendationEvent.Action.IMPRESSION,
                    page=page,
                    section_key=section_key,
                    request_id=rid,
                    product_id=int(pid),
                    algo_mode=str(algo_mode_override or comps.get("mode") or comps.get("source") or ""),
                    score=float(r.get("score")) if r.get("score") is not None else None,
                    components=comps,
                    context=ctx,
                ))

        for_you_algo_mode = "reranker" if str(for_you_algo_used).startswith("reranker") else "cooc"
        because_algo_mode = "reranker" if str(because_algo_used).startswith("reranker") else "cooc"
        for_you_ctx = _recs_event_context(
            algo_requested=algo_requested,
            algo_used=for_you_algo_used,
            model_version=for_you_model_version,
            routing=for_you_routing,
            section_key="for_you",
        )
        because_ctx = _recs_event_context(
            algo_requested=algo_requested,
            algo_used=because_algo_used,
            model_version=because_model_version,
            routing=because_routing,
            section_key="because_you_bought",
        )
        if base_product_ids:
            because_ctx["base_product_id"] = int(base_product_ids[0])

        trending_ctx = {
            "algo_requested": algo_requested,
            "algo_used": "trending",
            "algo_source": "trending",
            "section": "trending",
        }

        push_impressions("for_you", for_you, algo_mode_override=for_you_algo_mode, static_context=for_you_ctx)
        push_impressions("because_you_bought", because, algo_mode_override=because_algo_mode, static_context=because_ctx)
        push_impressions("trending", trending, static_context=trending_ctx)

        RecommendationEvent.objects.bulk_create(events, batch_size=500)

        for_you_enriched = _enrich_results_with_catalog_payload(for_you)
        because_enriched = _enrich_results_with_catalog_payload(because)
        trending_enriched = _enrich_results_with_catalog_payload(trending)

        return Response({
            "ok": True,
            "query": {
                "limit": limit,
                "category": category,
                "product_type": product_type,
                "price_min": str(price_min) if price_min is not None else None,
                "price_max": str(price_max) if price_max is not None else None,
                "algo_requested": algo_requested,
                "for_you_algo_used": for_you_algo_used,
                "for_you_model_version": for_you_model_version,
                "for_you_routing": for_you_routing,
                "for_you_context": context_meta,
                "because_algo_used": because_algo_used,
                "because_model_version": because_model_version,
                "because_routing": because_routing,
                "reranker_model_available": bool(get_reranker_model_version()),
            },
            "sections": [
                {"key": "for_you", "title": section_titles["for_you"], "results": for_you_enriched},
                {
                    "key": "because_you_bought",
                    "title": section_titles["because_you_bought"],
                    "base_product_id": base_product_ids[0] if base_product_ids else None,
                    "results": because_enriched,
                },
                {"key": "trending", "title": section_titles["trending"], "results": trending_enriched},
            ],
        })

