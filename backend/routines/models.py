from django.conf import settings
from django.db import models


class RoutineSnapshot(models.Model):
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name="routine_snapshots")
    created_at = models.DateTimeField(auto_now_add=True)

    # сохраняем только то, что нужно для аналитики
    missing_steps = models.JSONField(default=list, blank=True)   # ["spf", "cleanser"]
    profile_skin_type = models.CharField(max_length=30, blank=True, default="")

    def __str__(self) -> str:
        return f"RoutineSnapshot(user={self.user_id}, missing={self.missing_steps})"
