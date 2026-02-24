from django.conf import settings
from django.db import models


class Offer(models.Model):
    class Type(models.TextChoices):
        DISCOUNT = "discount", "Discount"
        POINTS_MULTIPLIER = "points_multiplier", "Points multiplier"
        GIFT = "gift", "Gift"

    is_active = models.BooleanField(default=True)

    campaign = models.ForeignKey(
        "offers.CampaignBudget",
        null=True,
        blank=True,
        on_delete=models.SET_NULL,
        related_name="offers",
    )

    name = models.CharField(max_length=200)
    offer_type = models.CharField(max_length=40, choices=Type.choices)

    # value examples:
    # discount: 10 (percent)
    # points_multiplier: 2 (x2 points)
    value = models.DecimalField(max_digits=10, decimal_places=2, default=0)

    # eligibility constraints (MVP)
    min_total_spend_90d = models.DecimalField(max_digits=10, decimal_places=2, default=0)
    allowed_steps = models.JSONField(default=list, blank=True)  # e.g. ["spf","moisturizer"]; empty = any

    # estimated cost for budget accounting (MVP)
    estimated_cost = models.DecimalField(max_digits=10, decimal_places=2, default=0)

    # frequency cap: do not assign more often than every N days
    cooldown_days = models.PositiveIntegerField(default=14)
    expires_in_days = models.PositiveIntegerField(default=7)

    created_at = models.DateTimeField(auto_now_add=True)
    allowed_categories = models.JSONField(default=list, blank=True)      # ["makeup","fragrance"]
    allowed_product_types = models.JSONField(default=list, blank=True)   # ["lipstick","edp"]

    # как применять оффер (если cart — на весь чек)
    target_scope = models.CharField(
        max_length=20,
        choices=[
            ("cart", "Cart"),
            ("category", "Category"),
            ("product_type", "Product type"),
            ("product_id", "Product"),
        ],
        default="cart",
    )

    def __str__(self) -> str:
        return f"{self.name} ({self.offer_type})"


class CampaignBudget(models.Model):
    name = models.CharField(max_length=64, unique=True)

    is_active = models.BooleanField(default=True)
    priority = models.IntegerField(default=100)  # меньше = выше приоритет

    weekly_limit = models.DecimalField(max_digits=12, decimal_places=2, default=1000)
    weekly_spent = models.DecimalField(max_digits=12, decimal_places=2, default=0)

    week_start_date = models.DateField(null=True, blank=True)
    updated_at = models.DateTimeField(auto_now=True)

    allowed_categories = models.JSONField(default=list, blank=True)  # ["skincare","makeup",...]
    allowed_steps = models.JSONField(default=list, blank=True)       # ["spf","serum",...]

    def __str__(self) -> str:
        return f"{self.name} budget"


class OfferAssignment(models.Model):
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name="offer_assignments")
    offer = models.ForeignKey(Offer, on_delete=models.PROTECT)

    assigned_at = models.DateTimeField(auto_now_add=True)
    expires_at = models.DateTimeField(null=True, blank=True)

    reason = models.JSONField(default=dict, blank=True)  # explainability payload
    is_redeemed = models.BooleanField(default=False)
    redeemed_transaction_id = models.IntegerField(null=True, blank=True)
    target = models.JSONField(default=dict, blank=True)  # например {"scope":"product_type","value":"lipstick","category":"makeup"}
    def __str__(self) -> str:
        return f"Assign(user={self.user_id}, offer={self.offer_id}, redeemed={self.is_redeemed})"
