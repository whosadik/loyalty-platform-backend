from django.contrib import admin
from .models import Product

@admin.register(Product)
class ProductAdmin(admin.ModelAdmin):
    list_display = ("id", "name", "brand", "step", "strength", "in_stock")
    list_filter = ("step", "strength", "in_stock")
    search_fields = ("name", "brand")
