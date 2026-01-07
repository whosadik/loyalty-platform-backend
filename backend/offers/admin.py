from django.contrib import admin
from .models import Offer, OfferAssignment, CampaignBudget

@admin.register(Offer)
class OfferAdmin(admin.ModelAdmin):
    list_display = ("id", "name", "offer_type", "value", "estimated_cost", "is_active", "cooldown_days")
    list_filter = ("offer_type", "is_active")
    search_fields = ("name",)

@admin.register(OfferAssignment)
class OfferAssignmentAdmin(admin.ModelAdmin):
    list_display = ("id", "user", "offer", "assigned_at", "is_redeemed")
    list_filter = ("is_redeemed", "assigned_at")
    search_fields = ("user__username", "user__email", "offer__name")

@admin.register(CampaignBudget)
class CampaignBudgetAdmin(admin.ModelAdmin):
    list_display = ("id", "name", "weekly_limit", "weekly_spent", "updated_at")
