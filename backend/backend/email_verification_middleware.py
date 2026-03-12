from django.http import JsonResponse

from backend.auth_email import get_email_verification_resend_remaining_seconds
from users_app.models import CustomerProfile


ALLOWED_UNVERIFIED_API_PREFIXES = (
    "/api/auth/",
    "/api/schema/",
    "/api/docs/",
)


class VerifiedEmailRequiredMiddleware:
    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        if self._should_block(request):
            return JsonResponse(
                {
                    "ok": False,
                    "code": "email_not_verified",
                    "message": "Confirm your email before accessing the API.",
                    "details": {
                        "email": request.user.email,
                        "resend_available_in_seconds": get_email_verification_resend_remaining_seconds(
                            request.user
                        ),
                    },
                    "request_id": getattr(request, "request_id", None),
                },
                status=403,
            )

        return self.get_response(request)

    def _should_block(self, request) -> bool:
        if request.method == "OPTIONS":
            return False

        path = request.path
        if not path.startswith("/api/"):
            return False

        if path.startswith(ALLOWED_UNVERIFIED_API_PREFIXES):
            return False

        user = getattr(request, "user", None)
        if not user or not user.is_authenticated:
            return False

        email = (getattr(user, "email", "") or "").strip()
        if not email:
            return False

        profile, _ = CustomerProfile.objects.get_or_create(user=user)
        return profile.email_verified_at is None
