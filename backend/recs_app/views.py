from datetime import datetime, timedelta, timezone
from collections import defaultdict

from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated
from rest_framework import serializers

from catalog.models import Product
from transactions.models import OwnedProduct, TransactionItem, Transaction
from users_app.models import CustomerProfile

from ml_logic.recommender import UserProfile, build_cooccurrence, recommend, bundle


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
