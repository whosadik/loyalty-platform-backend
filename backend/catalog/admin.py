from django.contrib import admin
from django.utils.html import format_html
from .models import Product, ProductReview


@admin.register(Product)
class ProductAdmin(admin.ModelAdmin):
    list_display = ("id", "source_product_id", "name", "brand", "category", "product_type", "price", "in_stock", "image_preview")
    list_filter = ("category", "product_type", "strength", "in_stock")
    search_fields = ("name", "brand", "source_product_id")
    readonly_fields = ("image_preview_large",)
    fields = (
        "name", "brand", "price", "currency", "category", "product_type",
        "in_stock", "source_product_id",
        "image", "image_preview_large", "image_url", "image_urls",
        "description", "application_text", "ingredients_inci", "volume_raw",
        "step", "concerns", "actives", "flags", "supported_skin_types", "strength", "attrs", "raw_meta",
    )

    def image_preview(self, obj):
        url = self._get_image_url(obj)
        if url:
            return format_html('<img src="{}" style="height:40px;border-radius:4px;" />', url)
        return "-"
    image_preview.short_description = "Фото"

    def image_preview_large(self, obj):
        url = self._get_image_url(obj)
        if url:
            return format_html('<img src="{}" style="max-height:200px;max-width:300px;border-radius:6px;" />', url)
        return "Нет картинки"
    image_preview_large.short_description = "Превью"

    def _get_image_url(self, obj):
        if obj.image:
            return obj.image.url
        if obj.image_url:
            return obj.image_url
        return None


@admin.register(ProductReview)
class ProductReviewAdmin(admin.ModelAdmin):
    list_display = ("id", "product", "user", "rating", "created_at", "updated_at")
    list_filter = ("rating", "created_at")
    search_fields = ("product__name", "product__brand", "user__username", "title", "body")
    readonly_fields = ("created_at", "updated_at")
