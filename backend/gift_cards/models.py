from django.conf import settings
from django.db import models


class GiftCard(models.Model):
    class Status(models.TextChoices):
        ACTIVE = "active", "Active"
        EXHAUSTED = "exhausted", "Exhausted"
        EXPIRED = "expired", "Expired"
        REFUNDED = "refunded", "Refunded"

    code = models.CharField(max_length=32, unique=True, db_index=True)
    purchaser = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="purchased_gift_cards",
    )
    recipient_email = models.EmailField(db_index=True)
    message = models.TextField(blank=True, default="")
    currency = models.CharField(max_length=8, default="KZT")
    initial_amount = models.DecimalField(max_digits=10, decimal_places=2)
    remaining_amount = models.DecimalField(max_digits=10, decimal_places=2)
    status = models.CharField(
        max_length=16,
        choices=Status.choices,
        default=Status.ACTIVE,
        db_index=True,
    )
    expires_at = models.DateTimeField(null=True, blank=True)
    purchase_transaction = models.ForeignKey(
        "transactions.Transaction",
        null=True,
        blank=True,
        on_delete=models.SET_NULL,
        related_name="issued_gift_cards",
    )
    sent_at = models.DateTimeField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["-created_at", "-id"]

    def __str__(self) -> str:
        return f"GiftCard(id={self.id}, recipient={self.recipient_email}, remaining={self.remaining_amount})"


class GiftCardLedgerEntry(models.Model):
    class EntryType(models.TextChoices):
        ISSUE = "issue", "Issue"
        REDEEM = "redeem", "Redeem"
        REFUND = "refund", "Refund"
        EXPIRE = "expire", "Expire"

    gift_card = models.ForeignKey(
        "gift_cards.GiftCard",
        on_delete=models.CASCADE,
        related_name="ledger_entries",
    )
    entry_type = models.CharField(max_length=16, choices=EntryType.choices, db_index=True)
    amount_delta = models.DecimalField(max_digits=10, decimal_places=2)
    transaction = models.ForeignKey(
        "transactions.Transaction",
        null=True,
        blank=True,
        on_delete=models.SET_NULL,
        related_name="gift_card_entries",
    )
    meta = models.JSONField(default=dict, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at", "-id"]

    def __str__(self) -> str:
        return f"GiftCardLedgerEntry(card={self.gift_card_id}, type={self.entry_type}, delta={self.amount_delta})"

