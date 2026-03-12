from rest_framework.exceptions import NotFound
from rest_framework.pagination import PageNumberPagination
from rest_framework.response import Response
from rest_framework import viewsets
from django.db.models import Q
from rest_framework.permissions import AllowAny, IsAdminUser, SAFE_METHODS
from rest_framework.views import APIView

from .brand_payloads import get_brand_detail_payload, list_brand_summary_payloads
from .home_hero import get_home_hero_payload
from .models import Product
from .new_fields import get_new_products_cutoff
from .sale_fields import product_has_discount
from .serializers import BrandDetailSerializer, BrandSummarySerializer, HomeHeroSerializer, ProductSerializer


class ProductPagination(PageNumberPagination):
    page_size = 12
    page_size_query_param = "page_size"
    max_page_size = 48


class ProductViewSet(viewsets.ModelViewSet):
    serializer_class = ProductSerializer
    pagination_class = ProductPagination

    def get_permissions(self):
        if self.request.method in SAFE_METHODS:
            return [AllowAny()]
        return [IsAdminUser()]

    def get_queryset(self):
        qs = Product.objects.all().order_by("-id")

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
        payload = get_brand_detail_payload(brand_slug)
        if payload is None:
            raise NotFound("Brand not found.")
        return Response(BrandDetailSerializer(payload).data)


class HomeHeroView(APIView):
    permission_classes = [AllowAny]

    def get(self, request):
        payload = get_home_hero_payload()
        return Response(HomeHeroSerializer(payload).data)
