from math import ceil

from rest_framework.exceptions import APIException, Throttled, ValidationError
from rest_framework.response import Response
from rest_framework.views import exception_handler as drf_exception_handler


def _request_id_from_context(context) -> str | None:
    request = (context or {}).get("request") if context else None
    if request is None:
        return None
    return getattr(request, "request_id", None) or request.headers.get("X-Request-ID")


def exception_handler(exc, context):
    request_id = _request_id_from_context(context)
    resp = drf_exception_handler(exc, context)
    if resp is None:
        return Response(
            {
                "ok": False,
                "code": "server_error",
                "message": "Internal server error",
                "details": None,
                "request_id": request_id,
            },
            status=500,
        )

    data = resp.data
    code = getattr(exc, "default_code", None) or "error"
    message = "Request failed"
    details = data

    if isinstance(exc, ValidationError):
        code = "validation_error"
        message = "Validation error"
    elif isinstance(exc, Throttled):
        code = "rate_limited"
        retry_after_seconds = ceil(exc.wait) if exc.wait is not None else None
        message = (
            f"Too many requests. Try again in {retry_after_seconds} seconds."
            if retry_after_seconds is not None
            else "Too many requests. Try again later."
        )
        details = {"retry_after_seconds": retry_after_seconds}
    elif isinstance(exc, APIException):
        if isinstance(data, dict) and "detail" in data and isinstance(data["detail"], str):
            message = data["detail"]
            details = None
        elif isinstance(data, str):
            message = data
            details = None
        else:
            detail_value = getattr(exc, "detail", None)
            if isinstance(detail_value, str):
                message = detail_value

    resp.data = {
        "ok": False,
        "code": str(code),
        "message": str(message),
        "details": details,
        "request_id": request_id,
    }
    return resp
