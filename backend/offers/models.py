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

    # how to apply the offer (cart means whole basket)
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
    priority = models.IntegerField(default=100)  # smaller number = higher priority

    weekly_limit = models.DecimalField(max_digits=12, decimal_places=2, default=1000)
    weekly_spent = models.DecimalField(max_digits=12, decimal_places=2, default=0)

    week_start_date = models.DateField(null=True, blank=True)
    start_date = models.DateField(null=True, blank=True)
    end_date = models.DateField(null=True, blank=True)
    updated_at = models.DateTimeField(auto_now=True)

    allowed_categories = models.JSONField(default=list, blank=True)  # ["skincare","makeup",...]
    allowed_steps = models.JSONField(default=list, blank=True)       # ["spf","serum",...]
    tiers = models.JSONField(default=list, blank=True)
    promo_text = models.TextField(blank=True, default="")
    banner_url = models.URLField(blank=True, default="")

    def __str__(self) -> str:
        return f"{self.name} budget"


class OfferAssignment(models.Model):
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name="offer_assignments")
    offer = models.ForeignKey(Offer, on_delete=models.PROTECT)

    assigned_at = models.DateTimeField(auto_now_add=True)
    expires_at = models.DateTimeField(null=True, blank=True)

    reason = models.JSONField(default=dict, blank=True)  # explainability payload
    is_active = models.BooleanField(default=True, db_index=True)
    is_redeemed = models.BooleanField(default=False)
    redeemed_transaction_id = models.IntegerField(null=True, blank=True)
    superseded_at = models.DateTimeField(null=True, blank=True)
    superseded_by = models.ForeignKey(
        "self",
        null=True,
        blank=True,
        on_delete=models.SET_NULL,
        related_name="superseded_children",
    )
    target = models.JSONField(default=dict, blank=True)  # e.g. {"scope":"product_type","value":"lipstick","category":"makeup"}

    def __str__(self) -> str:
        return f"Assign(user={self.user_id}, offer={self.offer_id}, redeemed={self.is_redeemed})"


class OfferEvent(models.Model):
    class Type(models.TextChoices):
        ASSIGNED = "offer_assigned", "Offer assigned"
        EXPOSED = "offer_exposed", "Offer exposed"
        CLICKED = "offer_clicked", "Offer clicked"
        REDEEMED = "offer_redeemed", "Offer redeemed"
        EXPIRED = "offer_expired", "Offer expired"
        SUPERSEDED = "offer_superseded", "Offer superseded"

    assignment = models.ForeignKey("offers.OfferAssignment", on_delete=models.CASCADE, related_name="events")
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name="offer_events")
    offer = models.ForeignKey("offers.Offer", on_delete=models.CASCADE, related_name="events")

    campaign_name = models.CharField(max_length=128, db_index=True)
    event_type = models.CharField(max_length=32, choices=Type.choices, db_index=True)
    event_key = models.CharField(max_length=128, blank=True, null=True, db_index=True)
    event_version = models.PositiveSmallIntegerField(default=1)

    created_at = models.DateTimeField(auto_now_add=True, db_index=True)
    request_id = models.CharField(max_length=64, blank=True, null=True, db_index=True)
    context = models.JSONField(default=dict, blank=True)

    class Meta:
        constraints = [
            models.UniqueConstraint(
                fields=["event_key"],
                condition=models.Q(event_key__isnull=False),
                name="uq_offer_event_key_not_null",
            ),
        ]
        indexes = [
            models.Index(fields=["campaign_name", "created_at"]),
            models.Index(fields=["event_type", "created_at"]),
        ]

    def __str__(self) -> str:
        return f"OfferEvent(assign={self.assignment_id}, type={self.event_type})"
