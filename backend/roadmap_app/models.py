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


class RoadmapEvent(models.Model):
    class Type(models.TextChoices):
        PLAN_REFRESHED = "roadmap_plan_refreshed", "Roadmap plan refreshed"
        STEP_GENERATED = "roadmap_step_generated", "Roadmap step generated"
        STEP_EXPOSED = "roadmap_step_exposed", "Roadmap step exposed"
        STEP_CLICKED = "roadmap_step_clicked", "Roadmap step clicked"
        STEP_SKIPPED = "roadmap_step_skipped", "Roadmap step skipped"
        STEP_COMPLETED = "roadmap_step_completed", "Roadmap step completed"

    created_at = models.DateTimeField(auto_now_add=True, db_index=True)
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="roadmap_events",
    )
    plan = models.ForeignKey(
        "roadmap_app.RoadmapPlan",
        null=True,
        blank=True,
        on_delete=models.SET_NULL,
        related_name="events",
    )
    step = models.ForeignKey(
        "roadmap_app.RoadmapStep",
        null=True,
        blank=True,
        on_delete=models.SET_NULL,
        related_name="events",
    )
    event_type = models.CharField(max_length=40, choices=Type.choices, db_index=True)
    request_id = models.CharField(max_length=64, null=True, blank=True, db_index=True)
    context = models.JSONField(default=dict, blank=True)

    class Meta:
        indexes = [
            models.Index(fields=["user", "created_at"], name="roadmap_app_user_id_06df4b_idx"),
            models.Index(fields=["event_type", "created_at"], name="roadmap_app_event_t_b2de4d_idx"),
            models.Index(fields=["step", "event_type"], name="roadmap_app_step_id_386a16_idx"),
        ]

    def __str__(self) -> str:
        return f"RoadmapEvent(user={self.user_id}, event={self.event_type}, step={self.step_id})"


class RoadmapMLInvocation(models.Model):
    created_at = models.DateTimeField(auto_now_add=True, db_index=True)
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        null=True,
        blank=True,
        on_delete=models.SET_NULL,
        related_name="roadmap_ml_invocations",
    )
    plan = models.ForeignKey(
        "roadmap_app.RoadmapPlan",
        null=True,
        blank=True,
        on_delete=models.SET_NULL,
        related_name="ml_invocations",
    )
    category = models.CharField(max_length=20, db_index=True)
    refresh_caller = models.CharField(max_length=64, blank=True, default="")
    ml_mode = models.CharField(max_length=32, blank=True, default="")

    decision = models.CharField(max_length=20, blank=True, default="", db_index=True)
    fallback_reason = models.CharField(max_length=64, blank=True, default="")
    disabled_reason = models.CharField(max_length=64, blank=True, default="")
    model_path = models.CharField(max_length=512, blank=True, default="")
    model_version = models.CharField(max_length=128, blank=True, default="")
    model_slot = models.CharField(max_length=32, blank=True, default="")
    predict_ms = models.FloatField(null=True, blank=True)
    predict_error = models.CharField(max_length=512, blank=True, default="")
    active_top_product_type = models.CharField(max_length=64, blank=True, default="")
    active_top_score = models.FloatField(null=True, blank=True)

    shadow_enabled = models.BooleanField(default=False)
    shadow_reason = models.CharField(max_length=64, blank=True, default="")
    shadow_model_path = models.CharField(max_length=512, blank=True, default="")
    shadow_model_version = models.CharField(max_length=128, blank=True, default="")
    shadow_predict_ms = models.FloatField(null=True, blank=True)
    shadow_predict_error = models.CharField(max_length=512, blank=True, default="")
    shadow_top_product_type = models.CharField(max_length=64, blank=True, default="")
    shadow_top_score = models.FloatField(null=True, blank=True)

    planner_mode = models.CharField(max_length=32, blank=True, default="")
    planner_served = models.BooleanField(default=False)
    planner_decision = models.CharField(max_length=20, blank=True, default="")
    planner_model_path = models.CharField(max_length=512, blank=True, default="")
    planner_predict_ms = models.FloatField(null=True, blank=True)
    planner_predict_error = models.CharField(max_length=512, blank=True, default="")

    rollout_mode = models.CharField(max_length=16, blank=True, default="")
    rollout_selected = models.BooleanField(default=False)
    rollout_bucket = models.IntegerField(null=True, blank=True)
    rollout_percent = models.IntegerField(null=True, blank=True)

    planned_target_product_type = models.CharField(max_length=64, blank=True, default="")
    planned_target_step_index = models.PositiveIntegerField(null=True, blank=True)

    class Meta:
        indexes = [
            models.Index(fields=["category", "created_at"]),
            models.Index(fields=["decision", "created_at"]),
        ]

    def __str__(self) -> str:
        return f"RoadmapMLInvocation(category={self.category}, decision={self.decision}, plan={self.plan_id})"


class RoadmapRuntimeConfig(models.Model):
    key = models.CharField(max_length=64, unique=True)
    value = models.TextField(blank=True, default="")
    updated_at = models.DateTimeField(auto_now=True)
    updated_by = models.CharField(max_length=128, blank=True, default="")
    note = models.CharField(max_length=256, blank=True, default="")

    def __str__(self) -> str:
        return f"RoadmapRuntimeConfig(key={self.key})"
