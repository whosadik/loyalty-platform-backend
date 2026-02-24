from django.core.cache import cache
from django.db import connection
from rest_framework.views import APIView
from rest_framework.response import Response
from audit.logging import log_event
from audit.models import AuditEvent
from backend.permissions import HasStaffPermission


class AdminCacheInvalidateView(APIView):
    permission_classes = [HasStaffPermission.with_perm("invalidate_cache")]

    def post(self, request):
        db_name = connection.settings_dict.get("NAME", "default")
        keys = [f"recs:products:v1:{db_name}", f"recs:cooc90d:v1:{db_name}"]
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

