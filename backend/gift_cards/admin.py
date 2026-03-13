from django.contrib import admin

from .models import GiftCard, GiftCardLedgerEntry


@admin.register(GiftCard)
class GiftCardAdmin(admin.ModelAdmin):
    list_display = (
        "id",
        "code",
        "purchaser",
        "recipient_email",
        "initial_amount",
        "remaining_amount",
        "status",
        "sent_at",
        "created_at",
    )
    list_filter = ("status", "currency", "created_at", "sent_at")
    search_fields = ("code", "recipient_email", "purchaser__email", "purchaser__username")


@admin.register(GiftCardLedgerEntry)
class GiftCardLedgerEntryAdmin(admin.ModelAdmin):
    list_display = ("id", "gift_card", "entry_type", "amount_delta", "transaction", "created_at")
    list_filter = ("entry_type", "created_at")
    search_fields = ("gift_card__code", "transaction__id")

