from django.core.cache import cache
from django.db import connection
from rest_framework.views import APIView
from rest_framework.response import Response
from audit.logging import log_event
from audit.models import AuditEvent
from backend.permissions import HasStaffPermission


class AdminCacheInvalidateView(APIView):
    permission_classes = [HasStaffPermission.with_perm("invalidate_cache")]

    def _delete_exact(self, key: str) -> int:
        return 1 if cache.delete(key) else 0

    def _delete_pattern(self, pattern: str) -> int:
        delete_pattern = getattr(cache, "delete_pattern", None)
        if not callable(delete_pattern):
            return 0
        try:
            return int(delete_pattern(pattern) or 0)
        except Exception:
            return 0

    def _scope_patterns(self, scope: str) -> list[str]:
        # Keep scope mappings explicit so frontend options work predictably.
        patterns = {
            "product": [
                "recs:products:*",
                "recs:trending30d:*",
            ],
            "user": [
                "admin:overview:*",
                "admin:metrics:*",
                "admin:recs:experiments:*",
            ],
            "recs": [
                "recs:*",
                "admin:recs:experiments:*",
            ],
            "offers": [
                "admin:overview:*",
                "admin:metrics:*",
            ],
            "all": [
                "recs:*",
                "admin:*",
            ],
        }
        return patterns.get(scope, patterns["recs"])

    def post(self, request):
        scope = str(request.data.get("scope") or "recs").strip().lower()
        key = str(request.data.get("key") or "").strip() or None
        deleted = 0
        touched_keys: list[str] = []
        touched_patterns: list[str] = []

        if key is not None:
            deleted += self._delete_exact(key)
            touched_keys.append(key)
        elif scope == "all":
            try:
                cache.clear()
                deleted = 1
                touched_patterns.append("cache.clear()")
            except Exception:
                # Fallback to pattern-based invalidation if clear fails.
                patterns = self._scope_patterns(scope)
                for pattern in patterns:
                    deleted += self._delete_pattern(pattern)
                    touched_patterns.append(pattern)
        else:
            patterns = self._scope_patterns(scope)
            for pattern in patterns:
                deleted += self._delete_pattern(pattern)
                touched_patterns.append(pattern)

        # Backward-compatible fallback for environments without delete_pattern support.
        if deleted == 0 and key is None:
            db_name = connection.settings_dict.get("NAME", "default")
            fallback_keys = [
                f"recs:products:v1:{db_name}",
                f"recs:cooc90d:v1:{db_name}",
                f"recs:products:v2:{db_name}",
                f"recs:cooc90d:v2:{db_name}",
                f"recs:products:v3:{db_name}",
            ]
            for fallback_key in fallback_keys:
                deleted += self._delete_exact(fallback_key)
            touched_keys.extend(fallback_keys)

        log_event(
            request=request,
            action=AuditEvent.Action.CACHE_INVALIDATE,
            status_code=200,
            meta={
                "scope": scope,
                "key": key,
                "deleted": deleted,
                "keys": touched_keys,
                "patterns": touched_patterns,
            },
        )

        return Response(
            {
                "ok": True,
                "scope": scope,
                "key": key,
                "deleted": deleted,
                "keys": touched_keys,
                "patterns": touched_patterns,
            }
        )

