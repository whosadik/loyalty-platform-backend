from django.core.cache import cache
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.permissions import IsAdminUser
from audit.logging import log_event
from audit.models import AuditEvent
from backend.permissions import HasStaffPermission


class AdminCacheInvalidateView(APIView):
    permission_classes = [HasStaffPermission.with_perm("invalidate_cache")]

    def post(self, request):
        keys = ["recs:products:v1", "recs:cooc90d:v1"]
        deleted = 0
        for k in keys:
            if cache.delete(k):
                deleted += 1
        log_event(
            request=request,
            action=AuditEvent.Action.CACHE_INVALIDATE,
            status_code=200,
            meta={"keys": keys, "deleted": deleted},
        )

        return Response({"ok": True, "deleted": deleted, "keys": keys})

