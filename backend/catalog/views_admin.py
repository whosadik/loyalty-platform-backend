from django.db.models import Count, ProtectedError, Q
from rest_framework import status, viewsets
from rest_framework.parsers import FormParser, JSONParser, MultiPartParser
from rest_framework.permissions import IsAdminUser
from rest_framework.response import Response

from .models import Brand, Product
from .serializers import AdminBrandSerializer, AdminProductSerializer
from .views import ProductPagination


class AdminBrandViewSet(viewsets.ModelViewSet):
    serializer_class = AdminBrandSerializer
    permission_classes = [IsAdminUser]
    parser_classes = [MultiPartParser, FormParser, JSONParser]

    def get_queryset(self):
        qs = Brand.objects.annotate(product_count=Count("products")).order_by("name")
        search = (self.request.query_params.get("search") or "").strip()
        if search:
            qs = qs.filter(Q(name__icontains=search) | Q(slug__icontains=search))
        is_active = self.request.query_params.get("is_active")
        if is_active in {"true", "false"}:
            qs = qs.filter(is_active=(is_active == "true"))
        return qs

    def destroy(self, request, *args, **kwargs):
        instance = self.get_object()
        try:
            instance.delete()
        except ProtectedError:
            return Response(
                {
                    "detail": "Нельзя удалить бренд: к нему привязаны товары. "
                    "Перенесите товары на другой бренд или деактивируйте бренд (is_active=false).",
                    "code": "brand_has_products",
                },
                status=status.HTTP_409_CONFLICT,
            )
        return Response(status=status.HTTP_204_NO_CONTENT)


class AdminProductViewSet(viewsets.ModelViewSet):
    serializer_class = AdminProductSerializer
    permission_classes = [IsAdminUser]
    parser_classes = [MultiPartParser, FormParser, JSONParser]
    pagination_class = ProductPagination

    def get_queryset(self):
        qs = Product.objects.select_related("brand_ref").order_by("-id")

        category = self.request.query_params.get("category")
        if category:
            qs = qs.filter(category=category)

        product_type = self.request.query_params.get("product_type")
        if product_type:
            qs = qs.filter(product_type=product_type)

        brand_ref = self.request.query_params.get("brand_ref")
        if brand_ref:
            qs = qs.filter(brand_ref_id=brand_ref)

        brand = self.request.query_params.get("brand")
        if brand:
            qs = qs.filter(brand__iexact=brand)

        in_stock = self.request.query_params.get("in_stock")
        if in_stock in {"true", "false"}:
            qs = qs.filter(in_stock=(in_stock == "true"))

        search = (self.request.query_params.get("search") or "").strip()
        if search:
            qs = qs.filter(
                Q(name__icontains=search)
                | Q(brand__icontains=search)
                | Q(brand_ref__name__icontains=search)
                | Q(category__icontains=search)
                | Q(product_type__icontains=search)
                | Q(source_product_id__icontains=search)
            )

        return qs
