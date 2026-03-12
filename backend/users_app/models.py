from django.conf import settings
from django.db import models


class CustomerProfile(models.Model):
    class SkinType(models.TextChoices):
        DRY = "dry", "Dry"
        OILY = "oily", "Oily"
        COMBINATION = "combination", "Combination"
        SENSITIVE = "sensitive", "Sensitive"
        NORMAL = "normal", "Normal"

    class Budget(models.TextChoices):
        LOW = "low", "Low"
        MEDIUM = "medium", "Medium"
        HIGH = "high", "High"

    user = models.OneToOneField(settings.AUTH_USER_MODEL, on_delete=models.CASCADE)

    first_name = models.CharField(max_length=150, blank=True, default="")
    last_name = models.CharField(max_length=150, blank=True, default="")
    phone = models.CharField(max_length=32, blank=True, default="")
    city = models.CharField(max_length=120, blank=True, default="")
    skin_type = models.CharField(max_length=20, choices=SkinType.choices, default=SkinType.NORMAL)
    goals = models.JSONField(default=list, blank=True)       # пример: ["acne", "hydration"]
    avoid_flags = models.JSONField(default=list, blank=True) # пример: ["fragrance", "alcohol"]
    budget = models.CharField(max_length=20, choices=Budget.choices, default=Budget.MEDIUM)
    hair_profile = models.JSONField(default=dict, blank=True)
    makeup_profile = models.JSONField(default=dict, blank=True)
    fragrance_profile = models.JSONField(default=dict, blank=True)
    profile_completed_at = models.DateTimeField(null=True, blank=True)
    profile_completion_rewarded_at = models.DateTimeField(null=True, blank=True)
    email_verified_at = models.DateTimeField(null=True, blank=True)
    email_verification_sent_at = models.DateTimeField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self) -> str:
        return f"CustomerProfile(user_id={self.user_id}, skin_type={self.skin_type})"
