from django.contrib import admin
from .models import Tier, LoyaltyAccount, LoyaltyLedgerEntry


@admin.register(Tier)
class TierAdmin(admin.ModelAdmin):
    list_display = ("id", "name", "threshold_spend_90d", "points_rate")


@admin.register(LoyaltyAccount)
class LoyaltyAccountAdmin(admin.ModelAdmin):
    list_display = ("id", "user", "tier", "points_balance", "updated_at")
    search_fields = ("user__username", "user__email")


@admin.register(LoyaltyLedgerEntry)
class LoyaltyLedgerEntryAdmin(admin.ModelAdmin):
    list_display = ("id", "account", "entry_type", "points_delta", "reference", "created_at")
    list_filter = ("entry_type", "created_at")
