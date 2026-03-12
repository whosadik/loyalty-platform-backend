from urllib.parse import urlencode

from django.conf import settings
from django.contrib.auth import get_user_model
from django.contrib.auth.tokens import default_token_generator
from django.core import signing
from django.core.mail import send_mail
from django.utils import timezone
from django.utils.encoding import force_bytes, force_str
from django.utils.http import urlsafe_base64_decode, urlsafe_base64_encode

from users_app.models import CustomerProfile


User = get_user_model()
EMAIL_VERIFICATION_SALT = "auth.email.verify"


def build_email_verification_token(user) -> str:
    return signing.dumps(
        {"user_id": user.pk, "email": user.email},
        salt=EMAIL_VERIFICATION_SALT,
    )


def resolve_email_verification_user(token: str):
    payload = signing.loads(
        token,
        salt=EMAIL_VERIFICATION_SALT,
        max_age=settings.EMAIL_VERIFICATION_MAX_AGE_SECONDS,
    )
    user = User.objects.get(pk=payload["user_id"])
    if user.email.strip().lower() != str(payload["email"]).strip().lower():
        raise signing.BadSignature("Email does not match token payload.")
    return user


def build_email_verification_url(user) -> str:
    token = build_email_verification_token(user)
    query = urlencode({"token": token})
    return f"{settings.FRONTEND_BASE_URL.rstrip('/')}/verify-email?{query}"


def build_password_reset_uid(user) -> str:
    return urlsafe_base64_encode(force_bytes(user.pk))


def build_password_reset_token(user) -> str:
    return default_token_generator.make_token(user)


def build_password_reset_url(user) -> str:
    query = urlencode(
        {
            "uid": build_password_reset_uid(user),
            "token": build_password_reset_token(user),
        }
    )
    return f"{settings.FRONTEND_BASE_URL.rstrip('/')}/reset-password?{query}"


def resolve_password_reset_user(uid: str, token: str):
    user_id = force_str(urlsafe_base64_decode(uid))
    user = User.objects.get(pk=user_id)
    if not default_token_generator.check_token(user, token):
        raise signing.BadSignature("Password reset link is invalid.")
    return user


def send_email_verification_message(user) -> str:
    verification_url = build_email_verification_url(user)
    send_mail(
        subject="Confirm your Uilesim email",
        message=(
            "Welcome to Uilesim.\n\n"
            "Confirm your email address by opening this link:\n"
            f"{verification_url}\n\n"
            "The link is active for 3 days."
        ),
        from_email=settings.DEFAULT_FROM_EMAIL,
        recipient_list=[user.email],
    )

    profile, _ = CustomerProfile.objects.get_or_create(user=user)
    profile.email_verification_sent_at = timezone.now()
    profile.save(update_fields=["email_verification_sent_at"])
    return verification_url


def send_password_reset_message(user) -> str:
    reset_url = build_password_reset_url(user)
    send_mail(
        subject="Reset your Uilesim password",
        message=(
            "We received a request to reset your Uilesim password.\n\n"
            "Open this link to set a new password:\n"
            f"{reset_url}\n\n"
            "If you did not request it, you can ignore this email."
        ),
        from_email=settings.DEFAULT_FROM_EMAIL,
        recipient_list=[user.email],
    )
    return reset_url


def get_email_verification_resend_remaining_seconds(user) -> int:
    profile, _ = CustomerProfile.objects.get_or_create(user=user)
    if profile.email_verified_at is not None:
        return 0

    if profile.email_verification_sent_at is None:
        return 0

    elapsed = (timezone.now() - profile.email_verification_sent_at).total_seconds()
    remaining = settings.EMAIL_VERIFICATION_RESEND_COOLDOWN_SECONDS - int(elapsed)
    return max(0, remaining)


def mark_email_verified(user):
    profile, _ = CustomerProfile.objects.get_or_create(user=user)
    if profile.email_verified_at is None:
        profile.email_verified_at = timezone.now()
        profile.save(update_fields=["email_verified_at"])
        return False
    return True
