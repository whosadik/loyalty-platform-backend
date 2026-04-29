from django.contrib import admin
from .models import Product, ProductReview


@admin.register(Product)
class ProductAdmin(admin.ModelAdmin):
    list_display = ("id", "source_product_id", "name", "brand", "category", "product_type", "price", "in_stock")
    list_filter = ("category", "product_type", "strength", "in_stock")
    search_fields = ("name", "brand", "source_product_id")


@admin.register(ProductReview)
class ProductReviewAdmin(admin.ModelAdmin):
    list_display = ("id", "product", "user", "rating", "created_at", "updated_at")
    list_filter = ("rating", "created_at")
    search_fields = ("product__name", "product__brand", "user__username", "title", "body")
    readonly_fields = ("created_at", "updated_at")
