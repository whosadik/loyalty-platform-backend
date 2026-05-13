from django.contrib import admin
from .models import Offer, OfferAssignment, CampaignBudget, OfferEvent

@admin.register(Offer)
class OfferAdmin(admin.ModelAdmin):
    list_display = ("id", "name", "offer_type", "target_scope", "value", "estimated_cost", "is_active", "cooldown_days")
    list_filter = ("offer_type", "target_scope", "is_active")
    search_fields = ("name",)

@admin.register(OfferAssignment)
class OfferAssignmentAdmin(admin.ModelAdmin):
    list_display = ("id", "user", "offer", "assigned_at", "is_redeemed")
    list_filter = ("is_redeemed", "assigned_at")
    search_fields = ("user__username", "user__email", "offer__name")

@admin.register(CampaignBudget)
class CampaignBudgetAdmin(admin.ModelAdmin):
    list_display = ("id", "name", "campaign_type", "priority", "weekly_limit", "weekly_spent", "updated_at")
    list_filter = ("campaign_type", "is_active")
    search_fields = ("name",)


@admin.register(OfferEvent)
class OfferEventAdmin(admin.ModelAdmin):
    list_display = ("id", "event_type", "assignment", "campaign_name", "created_at")
    list_filter = ("event_type", "campaign_name", "created_at")
    search_fields = ("campaign_name", "assignment__id", "user__username", "offer__name", "request_id")
