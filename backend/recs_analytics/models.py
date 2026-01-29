from django.conf import settings
from django.db import models


class RecommendationEvent(models.Model):
    class Action(models.TextChoices):
        IMPRESSION = "impression"
        CLICK = "click"
        ADD_TO_CART = "add_to_cart"
        PURCHASE_ATTRIBUTED = "purchase_attributed"

    created_at = models.DateTimeField(auto_now_add=True)

    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name="rec_events")
    action = models.CharField(max_length=32, choices=Action.choices)

    page = models.CharField(max_length=32, default="home")          # home/bundle/etc
    section_key = models.CharField(max_length=64, null=True, blank=True)  # for_you/because/trending
    request_id = models.CharField(max_length=64, null=True, blank=True)

    product = models.ForeignKey("catalog.Product", on_delete=models.CASCADE)

    # объяснимость/атрибуция
    algo_mode = models.CharField(max_length=32, null=True, blank=True)   # recommend/bundle/trending/fallback
    score = models.FloatField(null=True, blank=True)
    components = models.JSONField(default=dict, blank=True)              # компоненты скора
    context = models.JSONField(default=dict, blank=True)                 # base_product_id, etc.

    class Meta:
        indexes = [
            models.Index(fields=["created_at"]),
            models.Index(fields=["user", "created_at"]),
            models.Index(fields=["action", "created_at"]),
            models.Index(fields=["page", "section_key", "action", "created_at"]),
            models.Index(fields=["product", "action", "created_at"]),
        ]
