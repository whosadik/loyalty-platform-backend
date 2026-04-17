from django.conf import settings
from django.db import models


class RoutineSnapshot(models.Model):
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name="routine_snapshots")
    created_at = models.DateTimeField(auto_now_add=True)

    # сохраняем только то, что нужно для аналитики
    missing_steps = models.JSONField(default=list, blank=True)   # ["spf", "cleanser"]
    profile_skin_type = models.CharField(max_length=30, blank=True, default="")

    # full generated routine payload (am/pm/notes) for history view
    payload = models.JSONField(default=dict, blank=True)

    class Meta:
        indexes = [
            models.Index(fields=["user", "-created_at"]),
        ]

    def __str__(self) -> str:
        return f"RoutineSnapshot(user={self.user_id}, missing={self.missing_steps})"


class SavedRoutine(models.Model):
    user = models.OneToOneField(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="saved_routine",
    )
    payload = models.JSONField(default=dict, blank=True)
    updated_at = models.DateTimeField(auto_now=True)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self) -> str:
        return f"SavedRoutine(user={self.user_id}, updated_at={self.updated_at})"
