from rest_framework.views import exception_handler as drf_exception_handler
from rest_framework.exceptions import APIException, ValidationError
from rest_framework.response import Response


def exception_handler(exc, context):
    resp = drf_exception_handler(exc, context)
    if resp is None:
        # неожиданные ошибки (500)
        return Response(
            {"ok": False, "code": "server_error", "message": "Internal server error", "details": None},
            status=500,
        )

    status = resp.status_code
    data = resp.data

    code = getattr(exc, "default_code", None) or "error"
    message = "Request failed"
    details = data

    if isinstance(exc, ValidationError):
        code = "validation_error"
        message = "Validation error"
    elif isinstance(exc, APIException):
        # DRF часто кладёт строку в detail
        if isinstance(data, dict) and "detail" in data and isinstance(data["detail"], str):
            message = data["detail"]
            details = None
        elif isinstance(data, str):
            message = data
            details = None
        else:
            # оставим как details
            message = getattr(exc, "detail", None) if isinstance(getattr(exc, "detail", None), str) else message

    resp.data = {"ok": False, "code": str(code), "message": str(message), "details": details}
    return resp
