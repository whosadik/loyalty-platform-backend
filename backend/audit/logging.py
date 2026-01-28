from typing import Any, Optional
from audit.models import AuditEvent
import re
from copy import deepcopy

PII_KEYS = {
    "email", "phone", "password", "token", "access", "refresh",
    "first_name", "last_name", "full_name", "address",
}

EMAIL_RE = re.compile(r"[^@\s]+@[^@\s]+\.[^@\s]+")
PHONE_RE = re.compile(r"\+?\d[\d\-\s()]{7,}\d")


def scrub_meta(meta: dict) -> dict:
    """
    Best-effort scrub for PII. We redact by key name and by value patterns.
    """
    if not isinstance(meta, dict):
        return {}

    data = deepcopy(meta)

    def scrub_value(v):
        if isinstance(v, str):
            if EMAIL_RE.search(v):
                return "[REDACTED_EMAIL]"
            if PHONE_RE.search(v):
                return "[REDACTED_PHONE]"
        return v

    def walk(obj):
        if isinstance(obj, dict):
            out = {}
            for k, v in obj.items():
                lk = str(k).lower()
                if lk in PII_KEYS or "password" in lk or "token" in lk or "secret" in lk:
                    out[k] = "[REDACTED]"
                    continue
                out[k] = walk(v)
            return out
        if isinstance(obj, list):
            return [walk(x) for x in obj]
        return scrub_value(obj)

    return walk(data)

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
        meta=scrub_meta(meta or {}),
    )
    return ev
