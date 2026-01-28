from django.conf import settings
from django.db import models


class StaffRole(models.TextChoices):
    ADMIN = "admin"
    MANAGER = "manager"
    ANALYST = "analyst"


ROLE_PERMISSIONS = {
    StaffRole.ADMIN: {"view_metrics", "view_audit", "invalidate_cache"},
    StaffRole.MANAGER: {"view_metrics", "invalidate_cache"},
    StaffRole.ANALYST: {"view_metrics", "view_audit"},
}


class StaffProfile(models.Model):
    user = models.OneToOneField(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name="staff_profile")
    role = models.CharField(max_length=32, choices=StaffRole.choices, default=StaffRole.ANALYST)
    permissions = models.JSONField(default=list, blank=True)  # optional overrides

    def effective_permissions(self) -> set[str]:
        base = set(ROLE_PERMISSIONS.get(self.role, set()))
        extra = set(self.permissions or [])
        return base.union(extra)

    def __str__(self) -> str:
        return f"{self.user_id}:{self.role}"
