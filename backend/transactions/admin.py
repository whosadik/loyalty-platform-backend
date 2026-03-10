from django.contrib import admin
from .models import CartItem, Transaction, TransactionItem, WishlistItem


class TransactionItemInline(admin.TabularInline):
    model = TransactionItem
    extra = 0


@admin.register(Transaction)
class TransactionAdmin(admin.ModelAdmin):
    list_display = ("id", "user", "total_amount", "channel", "created_at")
    list_filter = ("channel", "created_at")
    search_fields = ("user__username", "user__email")
    inlines = [TransactionItemInline]


@admin.register(WishlistItem)
class WishlistItemAdmin(admin.ModelAdmin):
    list_display = ("id", "user", "product", "created_at")
    list_filter = ("created_at",)
    search_fields = ("user__username", "product__name", "product__brand")


@admin.register(CartItem)
class CartItemAdmin(admin.ModelAdmin):
    list_display = ("id", "user", "product", "quantity", "updated_at")
    list_filter = ("updated_at",)
    search_fields = ("user__username", "product__name", "product__brand")
