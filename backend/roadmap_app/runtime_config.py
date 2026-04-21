from __future__ import annotations

import threading
import time
from typing import Any

from django.conf import settings

_DEFAULT_TTL_SECONDS = 5.0

_lock = threading.Lock()
_cache: dict[str, str] | None = None
_cache_loaded_at: float = 0.0

FREEZE_KEY = "runtime_freeze_ml"
LOG_ENABLED_KEY = "ml_invocation_log_enabled"

_TRUE_TOKENS = {"1", "true", "yes", "on", "t", "y"}
_FALSE_TOKENS = {"0", "false", "no", "off", "f", "n", ""}


def _ttl_seconds() -> float:
    try:
        return float(getattr(settings, "ROADMAP_RUNTIME_CONFIG_TTL_SECONDS", _DEFAULT_TTL_SECONDS))
    except (TypeError, ValueError):
        return _DEFAULT_TTL_SECONDS


def _fetch_rows() -> dict[str, str]:
    from roadmap_app.models import RoadmapRuntimeConfig

    rows = RoadmapRuntimeConfig.objects.all().values_list("key", "value")
    return {str(key): str(value) for key, value in rows}


def _load_cache() -> dict[str, str]:
    global _cache, _cache_loaded_at
    now = time.monotonic()
    ttl = _ttl_seconds()
    with _lock:
        if _cache is not None and (now - _cache_loaded_at) < ttl:
            return _cache
    try:
        mapping = _fetch_rows()
    except Exception:
        with _lock:
            return _cache if _cache is not None else {}
    with _lock:
        _cache = mapping
        _cache_loaded_at = now
        return _cache


def invalidate_cache() -> None:
    global _cache, _cache_loaded_at
    with _lock:
        _cache = None
        _cache_loaded_at = 0.0


def get_str(key: str, default: str = "") -> str:
    mapping = _load_cache()
    return mapping.get(key, default)


def get_bool(key: str, default: bool = False) -> bool:
    mapping = _load_cache()
    if key not in mapping:
        return default
    token = mapping[key].strip().lower()
    if token in _TRUE_TOKENS:
        return True
    if token in _FALSE_TOKENS:
        return False
    return default


def get_int(key: str, default: int = 0) -> int:
    mapping = _load_cache()
    if key not in mapping:
        return default
    try:
        return int(mapping[key].strip())
    except (TypeError, ValueError, AttributeError):
        return default


def set_value(key: str, value: Any, *, updated_by: str = "", note: str = "") -> None:
    from roadmap_app.models import RoadmapRuntimeConfig

    key = str(key).strip()
    if not key:
        raise ValueError("runtime_config key must be non-empty")
    if len(key) > 64:
        raise ValueError(f"runtime_config key exceeds 64 chars: {key!r}")
    stored_value = "" if value is None else str(value)
    RoadmapRuntimeConfig.objects.update_or_create(
        key=key,
        defaults={
            "value": stored_value,
            "updated_by": (updated_by or "")[:128],
            "note": (note or "")[:256],
        },
    )
    invalidate_cache()


def unset_value(key: str) -> bool:
    from roadmap_app.models import RoadmapRuntimeConfig

    deleted, _ = RoadmapRuntimeConfig.objects.filter(key=key).delete()
    invalidate_cache()
    return bool(deleted)


def list_values() -> dict[str, str]:
    return dict(_load_cache())


def is_runtime_ml_frozen() -> bool:
    settings_default = bool(getattr(settings, "ROADMAP_RUNTIME_FREEZE_ML", True))
    return get_bool(FREEZE_KEY, default=settings_default)


def is_ml_invocation_log_enabled() -> bool:
    settings_default = bool(getattr(settings, "ROADMAP_ML_INVOCATION_LOG_ENABLED", True))
    return get_bool(LOG_ENABLED_KEY, default=settings_default)
