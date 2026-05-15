from django.utils.text import slugify
from rest_framework import serializers

from .new_fields import created_at_is_new
from .models import Brand, Product, ProductReview
from .product_metrics import (
    get_product_brand_slug,
    get_product_points_earned,
    get_product_rating,
    get_product_reviews_count,
)
from .sale_fields import (
    get_product_discount_percent,
    get_product_effective_price,
    get_product_original_price,
    product_has_discount,
)


class ProductSerializer(serializers.ModelSerializer):
    price = serializers.SerializerMethodField()
    brand_slug = serializers.SerializerMethodField()
    original_price = serializers.SerializerMethodField()
    discount = serializers.SerializerMethodField()
    has_discount = serializers.SerializerMethodField()
    points_earned = serializers.SerializerMethodField()
    is_new = serializers.SerializerMethodField()
    rating = serializers.SerializerMethodField()
    reviews_count = serializers.SerializerMethodField()
    image_url = serializers.SerializerMethodField()

    def get_image_url(self, obj: Product) -> str:
        if obj.image:
            request = self.context.get("request")
            if request:
                return request.build_absolute_uri(obj.image.url)
            return obj.image.url
        return obj.image_url or ""

    def get_brand_slug(self, obj: Product) -> str:
        return get_product_brand_slug(obj)

    def get_price(self, obj: Product) -> str | None:
        price = get_product_effective_price(obj)
        return str(price) if price is not None else None

    def get_original_price(self, obj: Product) -> str | None:
        original_price = get_product_original_price(obj)
        return str(original_price) if original_price is not None else None

    def get_discount(self, obj: Product) -> int | None:
        return get_product_discount_percent(obj)

    def get_has_discount(self, obj: Product) -> bool:
        return product_has_discount(obj)

    def get_points_earned(self, obj: Product) -> int:
        request = self.context.get("request")
        user = getattr(request, "user", None) if request is not None else None
        return get_product_points_earned(obj, user=user)

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


class ProductReviewSerializer(serializers.ModelSerializer):
    author_name = serializers.SerializerMethodField()
    is_mine = serializers.SerializerMethodField()
    rating = serializers.IntegerField(min_value=1, max_value=5)
    title = serializers.CharField(required=False, allow_blank=True, max_length=120)
    body = serializers.CharField(required=False, allow_blank=True, max_length=2000)

    def get_author_name(self, obj: ProductReview) -> str:
        full_name = obj.user.get_full_name().strip()
        if full_name:
            return full_name
        username = (getattr(obj.user, "username", "") or "").strip()
        return username or f"User {obj.user_id}"

    def get_is_mine(self, obj: ProductReview) -> bool:
        request = self.context.get("request")
        user = getattr(request, "user", None)
        return bool(user and user.is_authenticated and obj.user_id == user.id)

    class Meta:
        model = ProductReview
        fields = [
            "id",
            "product",
            "rating",
            "title",
            "body",
            "author_name",
            "is_mine",
            "created_at",
            "updated_at",
        ]
        read_only_fields = ["id", "product", "author_name", "is_mine", "created_at", "updated_at"]


class BrandSummarySerializer(serializers.Serializer):
    slug = serializers.CharField()
    name = serializers.CharField()
    logo_letter = serializers.CharField()
    logo_url = serializers.CharField(required=False, allow_blank=True)
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


def _unique_brand_slug(name: str, exclude_id: int | None = None) -> str:
    base = slugify((name or "").strip(), allow_unicode=True) or "brand"
    candidate = base
    suffix = 2
    qs = Brand.objects.all()
    if exclude_id is not None:
        qs = qs.exclude(pk=exclude_id)
    while qs.filter(slug=candidate).exists():
        candidate = f"{base}-{suffix}"
        suffix += 1
    return candidate


class AdminBrandSerializer(serializers.ModelSerializer):
    product_count = serializers.IntegerField(read_only=True)
    logo_image = serializers.ImageField(required=False, allow_null=True)
    logo_image_url = serializers.SerializerMethodField()

    class Meta:
        model = Brand
        fields = [
            "id",
            "name",
            "slug",
            "description_ru",
            "description_kk",
            "description_en",
            "logo_image",
            "logo_image_url",
            "logo_url",
            "is_active",
            "product_count",
            "created_at",
            "updated_at",
        ]
        read_only_fields = ["id", "slug", "logo_image_url", "product_count", "created_at", "updated_at"]

    def get_logo_image_url(self, obj: Brand) -> str:
        if obj.logo_image:
            request = self.context.get("request")
            if request:
                return request.build_absolute_uri(obj.logo_image.url)
            return obj.logo_image.url
        return obj.logo_url or ""

    def validate_name(self, value: str) -> str:
        normalized = (value or "").strip()
        if not normalized:
            raise serializers.ValidationError("Название бренда не может быть пустым.")
        qs = Brand.objects.filter(name__iexact=normalized)
        if self.instance is not None:
            qs = qs.exclude(pk=self.instance.pk)
        if qs.exists():
            raise serializers.ValidationError("Бренд с таким названием уже существует.")
        return normalized

    def create(self, validated_data):
        validated_data["slug"] = _unique_brand_slug(validated_data["name"])
        return super().create(validated_data)

    def update(self, instance, validated_data):
        new_name = validated_data.get("name")
        if new_name and new_name != instance.name:
            validated_data["slug"] = _unique_brand_slug(new_name, exclude_id=instance.pk)
        return super().update(instance, validated_data)


class AdminProductSerializer(serializers.ModelSerializer):
    brand_ref = serializers.PrimaryKeyRelatedField(
        queryset=Brand.objects.all(),
        required=False,
        allow_null=True,
    )
    brand_name = serializers.CharField(write_only=True, required=False, allow_blank=True)
    image = serializers.ImageField(required=False, allow_null=True)
    image_url = serializers.CharField(
        required=False,
        allow_blank=True,
        max_length=500,
        trim_whitespace=True,
    )
    image_url_display = serializers.SerializerMethodField()
    brand_slug = serializers.SerializerMethodField()

    class Meta:
        model = Product
        fields = [
            "id",
            "source_product_id",
            "name",
            "brand",
            "brand_ref",
            "brand_name",
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
            "stock_quantity",
            "image",
            "image_url",
            "image_url_display",
            "image_urls",
            "description",
            "application_text",
            "ingredients_inci",
            "volume_raw",
            "raw_meta",
            "created_at",
            "updated_at",
        ]
        read_only_fields = ["id", "brand_slug", "image_url_display", "in_stock", "created_at", "updated_at"]

    def get_image_url_display(self, obj: Product) -> str:
        if obj.image:
            request = self.context.get("request")
            if request:
                return request.build_absolute_uri(obj.image.url)
            return obj.image.url
        return obj.image_url or ""

    def get_brand_slug(self, obj: Product) -> str:
        if obj.brand_ref_id:
            return obj.brand_ref.slug
        return get_product_brand_slug(obj)

    def _resolve_brand(self, validated_data: dict) -> Brand | None:
        brand = validated_data.pop("brand_ref", None) if "brand_ref" in validated_data else None
        brand_name = validated_data.pop("brand_name", "") if "brand_name" in validated_data else ""
        brand_name = (brand_name or "").strip()

        if brand is None and brand_name:
            existing = Brand.objects.filter(name__iexact=brand_name).first()
            if existing is not None:
                brand = existing
            else:
                brand = Brand.objects.create(
                    name=brand_name,
                    slug=_unique_brand_slug(brand_name),
                )
        return brand

    @staticmethod
    def _sync_in_stock(validated_data: dict) -> None:
        if "stock_quantity" in validated_data:
            validated_data["in_stock"] = validated_data["stock_quantity"] > 0

    def validate_image_url(self, value):
        return (value or "").strip()

    def create(self, validated_data):
        brand = self._resolve_brand(validated_data)
        if brand is not None:
            validated_data["brand_ref"] = brand
            validated_data["brand"] = brand.name
        self._sync_in_stock(validated_data)
        return super().create(validated_data)

    def update(self, instance, validated_data):
        has_brand_ref_key = "brand_ref" in self.initial_data
        has_brand_name_key = "brand_name" in self.initial_data
        brand = self._resolve_brand(validated_data)

        if brand is not None:
            validated_data["brand_ref"] = brand
            validated_data["brand"] = brand.name
        elif has_brand_ref_key or has_brand_name_key:
            validated_data["brand_ref"] = None
            validated_data["brand"] = ""

        self._sync_in_stock(validated_data)
        return super().update(instance, validated_data)
