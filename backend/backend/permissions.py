from rest_framework.permissions import BasePermission
from admin_tools.models import StaffProfile


class HasStaffPermission(BasePermission):
    """
    Requires user.is_staff and specific staff permission code.
    Usage: permission_classes = [HasStaffPermission.with_perm("view_audit")]
    """

    required_perm: str | None = None

    def has_permission(self, request, view):
        user = request.user
        if not user or not user.is_authenticated:
            return False
        if not user.is_staff:
            return False

        perm = getattr(view, "required_staff_perm", None) or self.required_perm
        if not perm:
            return True  # only staff required

        try:
            sp = StaffProfile.objects.get(user=user)
        except StaffProfile.DoesNotExist:
            return False
        return perm in sp.effective_permissions()

    @classmethod
    def with_perm(cls, perm: str):
        class _P(cls):
            required_perm = perm
        return _P
