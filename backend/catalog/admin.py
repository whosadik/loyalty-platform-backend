from django.contrib import admin
from .models import Product

@admin.register(Product)
class ProductAdmin(admin.ModelAdmin):
    list_display = ("id", "source_product_id", "name", "brand", "category", "product_type", "price", "in_stock")
    list_filter = ("category", "product_type", "strength", "in_stock")
    search_fields = ("name", "brand", "source_product_id")
