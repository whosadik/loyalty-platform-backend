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

    skin_type = models.CharField(max_length=20, choices=SkinType.choices, default=SkinType.NORMAL)
    goals = models.JSONField(default=list, blank=True)       # пример: ["acne", "hydration"]
    avoid_flags = models.JSONField(default=list, blank=True) # пример: ["fragrance", "alcohol"]
    budget = models.CharField(max_length=20, choices=Budget.choices, default=Budget.MEDIUM)

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self) -> str:
        return f"CustomerProfile(user_id={self.user_id}, skin_type={self.skin_type})"
