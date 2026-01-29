from __future__ import annotations

from datetime import datetime, timedelta, timezone
from collections import defaultdict

from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated
from rest_framework import serializers

from backend.throttles import RecsRateThrottle
from catalog.models import Product
from transactions.models import OwnedProduct, TransactionItem, Transaction
from users_app.models import CustomerProfile

from ml_logic.recommender import UserProfile, build_cooccurrence, recommend, bundle

from drf_spectacular.utils import extend_schema, OpenApiExample, OpenApiParameter
from drf_spectacular.types import OpenApiTypes

from decimal import Decimal
from typing import Any

from django.utils import timezone
from django.db.models import Sum
from django.core.cache import cache

from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated

from backend.throttles import RecsRateThrottle

from catalog.models import Product
from transactions.models import Transaction, TransactionItem, OwnedProduct
from users_app.models import CustomerProfile

from ml_logic.recommender import recommend as rec_recommend
from ml_logic.recommender import bundle as rec_bundle
from recs_analytics.models import RecommendationEvent

# ВАЖНО: чтобы не переписывать, используем твои уже готовые хелперы.
# Если эти функции лежат в offers/services.py — импортируй оттуда.
from offers.services import _build_rec_profile, _load_products_for_recs, _cooccurrence_90d
class RecommendationsQuerySerializer(serializers.Serializer):
    category = serializers.ChoiceField(choices=["skincare", "haircare", "makeup", "fragrance"])
    product_type = serializers.CharField(required=False, allow_blank=True)
    limit = serializers.IntegerField(required=False, min_value=1, max_value=50, default=10)


class BundleQuerySerializer(serializers.Serializer):
    product_id = serializers.IntegerField()
    limit = serializers.IntegerField(required=False, min_value=1, max_value=50, default=10)


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


def _cooccurrence_last_90d(now: datetime):
    # строим co-occurrence по всем транзакциям за 90 дней (MVP)
    since = now - timedelta(days=90)

    items = (
        TransactionItem.objects
        .filter(transaction__created_at__gte=since)
        .values("transaction_id", "product_id")
    )

    txn_map = defaultdict(list)
    for row in items:
        txn_map[row["transaction_id"]].append(row["product_id"])

    txn_lists = list(txn_map.values())
    return build_cooccurrence(txn_lists)


class MeRecommendationsView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request):
        q = RecommendationsQuerySerializer(data=request.query_params)
        q.is_valid(raise_exception=True)

        category = q.validated_data["category"]
        product_type = (q.validated_data.get("product_type") or "").strip() or None
        limit = q.validated_data["limit"]

        cp, _ = CustomerProfile.objects.get_or_create(user=request.user)
        prof = _build_profile(cp)

        now = datetime.now(timezone.utc)

        owned_active_ids = list(
            OwnedProduct.objects.filter(user=request.user, is_active=True)
            .values_list("product_id", flat=True)
        )

        # контекст = owned товары (можно потом расширить)
        context_ids = owned_active_ids[:50]

        products = _load_products()
        co = _cooccurrence_last_90d(now)

        results = recommend(
            prof=prof,
            products=products,
            owned_active_ids=owned_active_ids,
            context_product_ids=context_ids,
            category=category,
            product_type=product_type,
            limit=limit,
            co=co,
        )

        return Response(
            {
                "query": {"category": category, "product_type": product_type, "limit": limit},
                "context": {"owned_active_count": len(owned_active_ids)},
                "results": results,
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

        cp, _ = CustomerProfile.objects.get_or_create(user=request.user)
        prof = _build_profile(cp)

        now = datetime.now(timezone.utc)
        owned_active_ids = list(
            OwnedProduct.objects.filter(user=request.user, is_active=True)
            .values_list("product_id", flat=True)
        )

        products = _load_products()
        co = _cooccurrence_last_90d(now)

        results = bundle(
            products=products,
            base_product_id=base_product_id,
            owned_active_ids=owned_active_ids,
            prof=prof,
            co=co,
            limit=limit,
        )

        return Response(
            {
                "query": {"product_id": base_product_id, "limit": limit},
                "results": results,
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
        "category": p.category,
        "product_type": p.product_type,
        "concerns": p.concerns or [],
        "attrs": p.attrs or {},
        "actives": p.actives or [],
        "flags": p.flags or [],
        "supported_skin_types": p.supported_skin_types or [],
        "strength": p.strength,
        "in_stock": bool(p.in_stock),
    }


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


def _trending_30d(*, now, category=None, product_type=None, price_min=None, price_max=None, limit=10) -> list[dict[str, Any]]:
    """
    Trending = top products by quantity in last 30 days.
    Cache 10 min.
    """
    key = f"recs:trending30d:v1:{category or 'any'}:{product_type or 'any'}:{price_min or 'none'}:{price_max or 'none'}:{limit}"
    cached = cache.get(key)
    if cached is not None:
        return cached

    since = now - timezone.timedelta(days=30)

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
        qp = request.query_params
        limit = _to_int(qp.get("limit"), 10) or 10
        limit = max(1, min(limit, 50))

        category = qp.get("category") or None
        product_type = qp.get("product_type") or None
        price_min = _to_decimal(qp.get("price_min"))
        price_max = _to_decimal(qp.get("price_max"))

        now = timezone.now()

        cp, _ = CustomerProfile.objects.get_or_create(user=request.user)
        prof = _build_rec_profile(cp)

        products = _load_products_for_recs()
        co = _cooccurrence_90d(now)

        owned_ids = list(
            OwnedProduct.objects.filter(user=request.user, is_active=True).values_list("product_id", flat=True)
        )

        # last purchase context
        last_txn = Transaction.objects.filter(user=request.user).order_by("-created_at").first()
        base_product_ids: list[int] = []
        if last_txn:
            base_product_ids = list(
                TransactionItem.objects.filter(transaction=last_txn).values_list("product_id", flat=True)[:10]
            )

        seen: set[int] = set()

        # 1) For you (profile-based)
        for_you_raw = rec_recommend(
            prof=prof,
            products=products,
            owned_active_ids=owned_ids,
            context_product_ids=(base_product_ids or owned_ids)[:50],
            category=category,
            product_type=product_type,
            limit=limit * 2,
            co=co,
        )

        for_you = []
        for r in for_you_raw:
            p = r.get("product") or {}
            if not _apply_filters_to_product_dict(p, category=category, product_type=product_type, price_min=price_min, price_max=price_max):
                continue
            for_you.append(r)
        for_you = _dedupe_limit(for_you, seen, limit)

        # 2) Because you bought (bundle)
        because = []
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
            because = _dedupe_limit(filtered, seen, limit)

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
        events = []
        rid = getattr(request, "request_id", None)

        def push_impressions(section_key: str, results: list[dict], page: str = "home"):
            for r in results:
                p = r.get("product") or {}
                pid = p.get("id")
                if not pid:
                    continue
                comps = r.get("components") or {}
                events.append(RecommendationEvent(
                    user=request.user,
                    action=RecommendationEvent.Action.IMPRESSION,
                    page=page,
                    section_key=section_key,
                    request_id=rid,
                    product_id=int(pid),
                    algo_mode=str(comps.get("mode") or comps.get("source") or ""),
                    score=float(r.get("score")) if r.get("score") is not None else None,
                    components=comps,
                    context={"why": (r.get("why") or [])[:6]},
                ))

        push_impressions("for_you", for_you)
        push_impressions("because_you_bought", because)
        push_impressions("trending", trending)

        RecommendationEvent.objects.bulk_create(events, batch_size=500)

        return Response({
            "ok": True,
            "query": {
                "limit": limit,
                "category": category,
                "product_type": product_type,
                "price_min": str(price_min) if price_min is not None else None,
                "price_max": str(price_max) if price_max is not None else None,
            },
            "sections": [
                {"key": "for_you", "title": "For you", "results": for_you},
                {"key": "because_you_bought", "title": "Because you bought", "base_product_id": base_product_ids[0] if base_product_ids else None, "results": because},
                {"key": "trending", "title": "Trending", "results": trending},
            ],
        })
