from rest_framework import serializers

from .new_fields import created_at_is_new
from .models import Product
from .product_metrics import (
    get_product_brand_slug,
    get_product_points_earned,
    get_product_rating,
    get_product_reviews_count,
)
from .sale_fields import get_product_discount_percent, get_product_original_price, product_has_discount


class ProductSerializer(serializers.ModelSerializer):
    brand_slug = serializers.SerializerMethodField()
    original_price = serializers.SerializerMethodField()
    discount = serializers.SerializerMethodField()
    has_discount = serializers.SerializerMethodField()
    points_earned = serializers.SerializerMethodField()
    is_new = serializers.SerializerMethodField()
    rating = serializers.SerializerMethodField()
    reviews_count = serializers.SerializerMethodField()

    def get_brand_slug(self, obj: Product) -> str:
        return get_product_brand_slug(obj)

    def get_original_price(self, obj: Product) -> str | None:
        original_price = get_product_original_price(obj)
        return str(original_price) if original_price is not None else None

    def get_discount(self, obj: Product) -> int | None:
        return get_product_discount_percent(obj)

    def get_has_discount(self, obj: Product) -> bool:
        return product_has_discount(obj)

    def get_points_earned(self, obj: Product) -> int:
        return get_product_points_earned(obj)

    def get_is_new(self, obj: Product) -> bool:
        return created_at_is_new(obj.created_at)

    def get_rating(self, obj: Product) -> float | None:
        return get_product_rating(obj)

    def get_reviews_count(self, obj: Product) -> int:
        return get_product_reviews_count(obj)

    class Meta:
        model = Product
        fields = [
            "id",
            "source_product_id",
            "name",
            "brand",
            "brand_slug",
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
            "points_earned",
            "is_new",
            "rating",
            "reviews_count",
            "created_at",
            "updated_at",
        ]
        read_only_fields = ["id", "created_at", "updated_at"]


class BrandSummarySerializer(serializers.Serializer):
    slug = serializers.CharField()
    name = serializers.CharField()
    logo_letter = serializers.CharField()
    product_count = serializers.IntegerField()


class BrandDetailSerializer(BrandSummarySerializer):
    description = serializers.CharField()
    categories = serializers.ListField(child=serializers.CharField())
    top_product_types = serializers.ListField(child=serializers.CharField())
    new_products_count = serializers.IntegerField()
    sale_products_count = serializers.IntegerField()


class HomeHeroSlideSerializer(serializers.Serializer):
    id = serializers.CharField()
    eyebrow = serializers.CharField(required=False, allow_blank=True, allow_null=True)
    title = serializers.CharField()
    description = serializers.CharField(required=False, allow_blank=True, allow_null=True)
    button_text = serializers.CharField(required=False, allow_blank=True, allow_null=True)
    button_to = serializers.CharField(required=False, allow_blank=True, allow_null=True)


class HomeHeroSerializer(serializers.Serializer):
    ok = serializers.BooleanField()
    slides = HomeHeroSlideSerializer(many=True)
