from __future__ import annotations

from collections.abc import Mapping
from typing import Literal

AppLanguage = Literal["ru", "kk", "en"]

SUPPORTED_LANGUAGES: tuple[AppLanguage, ...] = ("ru", "kk", "en")
DEFAULT_LANGUAGE: AppLanguage = "ru"


def normalize_language(value: object) -> AppLanguage:
    if not isinstance(value, str) or not value.strip():
        return DEFAULT_LANGUAGE

    raw = value.strip().lower().replace("_", "-")
    candidates = [part.strip() for part in raw.split(",") if part.strip()]

    for candidate in candidates:
        lang_part = candidate.split(";")[0].strip()
        primary = lang_part.split("-")[0].strip()
        if primary in SUPPORTED_LANGUAGES:
            return primary  # type: ignore[return-value]

    if raw in SUPPORTED_LANGUAGES:
        return raw  # type: ignore[return-value]

    return DEFAULT_LANGUAGE


def get_request_language(request) -> AppLanguage:
    if request is None:
        return DEFAULT_LANGUAGE

    headers = getattr(request, "headers", None) or {}
    query_params = getattr(request, "query_params", None) or {}

    for value in (
        headers.get("X-App-Language"),
        headers.get("X-Language"),
        query_params.get("lang"),
        headers.get("Accept-Language"),
    ):
        language = normalize_language(value)
        if language in SUPPORTED_LANGUAGES:
            return language

    return DEFAULT_LANGUAGE


def get_context_language(context: Mapping[str, object] | None) -> AppLanguage:
    if not isinstance(context, Mapping):
        return DEFAULT_LANGUAGE

    language = normalize_language(context.get("language"))
    if language in SUPPORTED_LANGUAGES:
        return language

    return get_request_language(context.get("request"))
