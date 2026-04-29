from django.db.models import Avg, Count, Q
from rest_framework import status, viewsets
from rest_framework.decorators import action
from rest_framework.exceptions import NotAuthenticated, NotFound
from rest_framework.pagination import PageNumberPagination
from rest_framework.permissions import AllowAny, IsAdminUser, SAFE_METHODS
from rest_framework.response import Response
from rest_framework.views import APIView

from backend.request_language import get_request_language
from .brand_payloads import get_brand_detail_payload, list_brand_summary_payloads
from .home_hero import get_home_hero_payload
from .models import Product, ProductReview
from .new_fields import get_new_products_cutoff
from .product_metrics import get_product_rating, get_product_reviews_count
from .sale_fields import product_has_discount
from .serializers import (
    BrandDetailSerializer,
    BrandSummarySerializer,
    HomeHeroSerializer,
    ProductReviewSerializer,
    ProductSerializer,
)


class ProductPagination(PageNumberPagination):
    page_size = 12
    page_size_query_param = "page_size"
    max_page_size = 48


class ProductViewSet(viewsets.ModelViewSet):
    serializer_class = ProductSerializer
    pagination_class = ProductPagination

    def get_permissions(self):
        if getattr(self, "action", None) in {"reviews", "delete_my_review"}:
            return [AllowAny()]
        if self.request.method in SAFE_METHODS:
            return [AllowAny()]
        return [IsAdminUser()]

    def get_queryset(self):
        qs = Product.objects.annotate(
            customer_rating_avg=Avg("reviews__rating"),
            customer_reviews_count=Count("reviews", distinct=True),
        ).order_by("-id")

        category = self.request.query_params.get("category")
        if category:
            qs = qs.filter(category=category)

        product_type = self.request.query_params.get("product_type")
        if product_type:
            qs = qs.filter(product_type=product_type)

        brand = self.request.query_params.get("brand")
        if brand:
            qs = qs.filter(brand__iexact=brand)

        in_stock = self.request.query_params.get("in_stock")
        if in_stock in {"true", "false"}:
            qs = qs.filter(in_stock=(in_stock == "true"))

        is_new = (self.request.query_params.get("new") or "").strip().lower()
        if is_new in {"1", "true", "yes"}:
            qs = qs.filter(created_at__gte=get_new_products_cutoff())

        search = (self.request.query_params.get("search") or "").strip()
        if search:
            qs = qs.filter(
                Q(name__icontains=search)
                | Q(brand__icontains=search)
                | Q(category__icontains=search)
                | Q(product_type__icontains=search)
                | Q(source_product_id__icontains=search)
            )

        sale = (self.request.query_params.get("sale") or "").strip().lower()
        if sale in {"1", "true", "yes"}:
            sale_ids = [
                product.id
                for product in qs.only("id", "price", "raw_meta", "attrs")
                if product_has_discount(product)
            ]
            qs = qs.filter(id__in=sale_ids)

        return qs

    def _reviews_payload(self, product: Product) -> dict:
        reviews_qs = product.reviews.select_related("user").order_by("-created_at", "-id")
        rating_counts_raw = product.reviews.order_by().values("rating").annotate(count=Count("id"))
        rating_counts = {str(rating): 0 for rating in range(1, 6)}
        for item in rating_counts_raw:
            rating_counts[str(item["rating"])] += item["count"]

        my_review = None
        request_user = getattr(self.request, "user", None)
        if request_user and request_user.is_authenticated:
            my_review = reviews_qs.filter(user=request_user).first()

        return {
            "summary": {
                "product_id": product.id,
                "rating": get_product_rating(product),
                "reviews_count": get_product_reviews_count(product),
                "customer_reviews_count": sum(rating_counts.values()),
                "rating_counts": rating_counts,
            },
            "results": ProductReviewSerializer(
                reviews_qs,
                many=True,
                context={"request": self.request},
            ).data,
            "my_review": ProductReviewSerializer(
                my_review,
                context={"request": self.request},
            ).data
            if my_review is not None
            else None,
        }

    @action(detail=True, methods=["get", "post"], url_path="reviews", permission_classes=[AllowAny])
    def reviews(self, request, pk=None):
        product = self.get_object()

        if request.method == "GET":
            return Response(self._reviews_payload(product))

        if not request.user or not request.user.is_authenticated:
            raise NotAuthenticated("Authentication is required to leave a review.")

        instance = ProductReview.objects.filter(product=product, user=request.user).first()
        serializer = ProductReviewSerializer(
            instance,
            data=request.data,
            partial=instance is not None,
            context={"request": request},
        )
        serializer.is_valid(raise_exception=True)
        serializer.save(product=product, user=request.user)

        product = self.get_queryset().get(pk=product.pk)
        response_status = status.HTTP_200_OK if instance else status.HTTP_201_CREATED
        return Response(self._reviews_payload(product), status=response_status)

    @action(detail=True, methods=["delete"], url_path="reviews/mine", permission_classes=[AllowAny])
    def delete_my_review(self, request, pk=None):
        product = self.get_object()

        if not request.user or not request.user.is_authenticated:
            raise NotAuthenticated("Authentication is required to delete a review.")

        ProductReview.objects.filter(product=product, user=request.user).delete()
        product = self.get_queryset().get(pk=product.pk)
        return Response(self._reviews_payload(product))

    def list(self, request, *args, **kwargs):
        queryset = self.filter_queryset(self.get_queryset())

        if "page" not in request.query_params and "page_size" not in request.query_params:
            serializer = self.get_serializer(queryset, many=True)
            return Response(serializer.data)

        page = self.paginate_queryset(queryset)
        if page is not None:
            serializer = self.get_serializer(page, many=True)
            return self.get_paginated_response(serializer.data)

        serializer = self.get_serializer(queryset, many=True)
        return Response(serializer.data)


class BrandListView(APIView):
    permission_classes = [AllowAny]

    def get(self, request):
        payload = list_brand_summary_payloads()
        return Response(BrandSummarySerializer(payload, many=True).data)


class BrandDetailView(APIView):
    permission_classes = [AllowAny]

    def get(self, request, brand_slug: str):
        payload = get_brand_detail_payload(brand_slug, get_request_language(request))
        if payload is None:
            raise NotFound("Brand not found.")
        return Response(BrandDetailSerializer(payload).data)


class HomeHeroView(APIView):
    permission_classes = [AllowAny]

    def get(self, request):
        payload = get_home_hero_payload(get_request_language(request))
        return Response(HomeHeroSerializer(payload).data)
