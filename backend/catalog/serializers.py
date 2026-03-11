from rest_framework import serializers

from .models import Product
from .sale_fields import get_product_discount_percent, get_product_original_price, product_has_discount


class ProductSerializer(serializers.ModelSerializer):
    original_price = serializers.SerializerMethodField()
    discount = serializers.SerializerMethodField()
    has_discount = serializers.SerializerMethodField()

    def get_original_price(self, obj: Product) -> str | None:
        original_price = get_product_original_price(obj)
        return str(original_price) if original_price is not None else None

    def get_discount(self, obj: Product) -> int | None:
        return get_product_discount_percent(obj)

    def get_has_discount(self, obj: Product) -> bool:
        return product_has_discount(obj)

    class Meta:
        model = Product
        fields = [
            "id",
            "source_product_id",
            "name",
            "brand",
            "price",
            "currency",
            "category",
            "product_type",
            "concerns",
            "attrs",
            "actives",
            "flags",
            "supported_skin_types",
            "step",
            "strength",
            "in_stock",
            "image_url",
            "image_urls",
            "description",
            "application_text",
            "ingredients_inci",
            "volume_raw",
            "raw_meta",
            "original_price",
            "discount",
            "has_discount",
            "created_at",
            "updated_at",
        ]
        read_only_fields = ["id", "created_at", "updated_at"]
