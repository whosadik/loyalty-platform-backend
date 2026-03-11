from rest_framework import viewsets
from django.db.models import Q
from rest_framework.permissions import IsAdminUser, IsAuthenticated, SAFE_METHODS

from .models import Product
from .sale_fields import product_has_discount
from .serializers import ProductSerializer


class ProductViewSet(viewsets.ModelViewSet):
    serializer_class = ProductSerializer

    def get_permissions(self):
        if self.request.method in SAFE_METHODS:
            return [IsAuthenticated()]
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

        search = (self.request.query_params.get("search") or "").strip()
        if search:
            qs = qs.filter(
                Q(name__icontains=search)
                | Q(brand__icontains=search)
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
