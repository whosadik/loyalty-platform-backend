from typing import Any, Optional
from audit.models import AuditEvent


def _client_ip(request) -> str | None:
    xff = request.headers.get("X-Forwarded-For")
    if xff:
        return xff.split(",")[0].strip()
    return request.META.get("REMOTE_ADDR")


def log_event(
    *,
    request=None,
    user=None,
    action: str,
    entity_type: str | None = None,
    entity_id: str | int | None = None,
    status_code: int | None = None,
    meta: dict[str, Any] | None = None,
    request_id: str | None = None,
    path: str | None = None,
    method: str | None = None,
    ip: str | None = None,
    user_agent: str | None = None,
) -> AuditEvent:
    if request is not None:
        request_id = request_id or getattr(request, "request_id", None)
        path = path or request.path
        method = method or request.method
        ip = ip or _client_ip(request)
        user_agent = user_agent or request.headers.get("User-Agent")
        if user is None and getattr(request, "user", None) and request.user.is_authenticated:
            user = request.user

    ev = AuditEvent.objects.create(
        user=user if (user and getattr(user, "is_authenticated", True)) else None,
        action=action,
        entity_type=entity_type,
        entity_id=str(entity_id) if entity_id is not None else None,
        request_id=request_id,
        path=path,
        method=method,
        status_code=status_code,
        ip=ip,
        user_agent=(user_agent[:255] if user_agent else None),
        meta=meta or {},
    )
    return ev
