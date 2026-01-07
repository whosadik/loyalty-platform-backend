from django.contrib import admin
from .models import Transaction, TransactionItem


class TransactionItemInline(admin.TabularInline):
    model = TransactionItem
    extra = 0


@admin.register(Transaction)
class TransactionAdmin(admin.ModelAdmin):
    list_display = ("id", "user", "total_amount", "channel", "created_at")
    list_filter = ("channel", "created_at")
    search_fields = ("user__username", "user__email")
    inlines = [TransactionItemInline]
