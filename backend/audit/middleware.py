import uuid


class RequestIdMiddleware:
    """
    Adds request.request_id and X-Request-ID header.
    Accepts incoming X-Request-ID (if provided) to support upstream tracing.
    """

    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        rid = request.headers.get("X-Request-ID") or uuid.uuid4().hex
        request.request_id = rid
        response = self.get_response(request)
        response["X-Request-ID"] = rid
        return response
