from django.conf import settings
from django.db import models


class AuditEvent(models.Model):
    class Action(models.TextChoices):
        CHECKOUT_CREATED = "checkout_created"
        CHECKOUT_REPLAY = "checkout_replay"
        OFFER_PREVIEW = "offer_preview"
        OFFER_REDEEM = "offer_redeem"
        NEXT_OFFER_ASSIGNED = "next_offer_assigned"
        CACHE_INVALIDATE = "cache_invalidate"

    created_at = models.DateTimeField(auto_now_add=True)

    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        null=True,
        blank=True,
        on_delete=models.SET_NULL,
        related_name="audit_events",
    )

    action = models.CharField(max_length=64, choices=Action.choices)

    # “на что” событие было направлено
    entity_type = models.CharField(max_length=64, null=True, blank=True)   # e.g. "Transaction", "OfferAssignment"
    entity_id = models.CharField(max_length=64, null=True, blank=True)     # строкой чтобы не париться с int/uuid

    # request context
    request_id = models.CharField(max_length=64, null=True, blank=True)
    path = models.CharField(max_length=255, null=True, blank=True)
    method = models.CharField(max_length=16, null=True, blank=True)
    status_code = models.IntegerField(null=True, blank=True)
    ip = models.CharField(max_length=64, null=True, blank=True)
    user_agent = models.CharField(max_length=255, null=True, blank=True)

    # любое доп. пояснение (без персональных данных!)
    meta = models.JSONField(default=dict, blank=True)

    class Meta:
        indexes = [
            models.Index(fields=["created_at"]),
            models.Index(fields=["action", "created_at"]),
            models.Index(fields=["user", "created_at"]),
            models.Index(fields=["request_id"]),
        ]

    def __str__(self) -> str:
        return f"{self.created_at} {self.action} user={self.user_id}"
