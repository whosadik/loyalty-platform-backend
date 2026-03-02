from django.conf import settings
from django.db import models


class RoadmapPlan(models.Model):
    class Category(models.TextChoices):
        SKINCARE = "skincare", "Skincare"
        HAIRCARE = "haircare", "Haircare"
        MAKEUP = "makeup", "Makeup"
        FRAGRANCE = "fragrance", "Fragrance"
        MIXED = "mixed", "Mixed"

    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="roadmap_plans",
    )
    category = models.CharField(max_length=20, choices=Category.choices, db_index=True)
    is_active = models.BooleanField(default=True, db_index=True)
    version = models.PositiveIntegerField(default=1)
    meta = models.JSONField(default=dict, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        indexes = [
            models.Index(fields=["user", "category", "is_active"]),
        ]

    def __str__(self) -> str:
        return f"RoadmapPlan(user={self.user_id}, category={self.category}, active={self.is_active})"


class RoadmapStep(models.Model):
    class Status(models.TextChoices):
        MISSING = "missing", "Missing"
        RECOMMENDED = "recommended", "Recommended"
        OWNED = "owned", "Owned"
        SKIPPED = "skipped", "Skipped"
        COMPLETED = "completed", "Completed"

    class Cadence(models.TextChoices):
        DAILY = "daily", "Daily"
        WEEKLY = "weekly", "Weekly"
        OPTIONAL = "optional", "Optional"

    plan = models.ForeignKey(
        "roadmap_app.RoadmapPlan",
        on_delete=models.CASCADE,
        related_name="steps",
    )
    step_index = models.PositiveIntegerField()
    product_type = models.CharField(max_length=64)
    status = models.CharField(max_length=20, choices=Status.choices, default=Status.MISSING)

    recommended_product = models.ForeignKey(
        "catalog.Product",
        null=True,
        blank=True,
        on_delete=models.SET_NULL,
        related_name="+",
    )
    suggestions = models.JSONField(default=list, blank=True)
    score = models.FloatField(null=True, blank=True)
    confidence = models.FloatField(null=True, blank=True)
    why = models.JSONField(default=list, blank=True)
    cadence = models.CharField(max_length=16, choices=Cadence.choices, blank=True, default="")

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        constraints = [
            models.UniqueConstraint(fields=["plan", "step_index"], name="uq_roadmap_step_plan_index"),
        ]
        indexes = [
            models.Index(fields=["plan", "status"]),
        ]

    def __str__(self) -> str:
        return f"RoadmapStep(plan={self.plan_id}, idx={self.step_index}, type={self.product_type}, status={self.status})"
