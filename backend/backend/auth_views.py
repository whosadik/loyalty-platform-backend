import logging
import re

from django.conf import settings
from django.contrib.auth import authenticate, get_user_model, login, logout
from django.contrib.auth.password_validation import validate_password
from django.core import signing
from django.core.exceptions import ValidationError as DjangoValidationError
from django.middleware.csrf import get_token, rotate_token
from drf_spectacular.utils import OpenApiResponse, extend_schema
from rest_framework.authentication import CSRFCheck
from rest_framework.exceptions import APIException, PermissionDenied
from rest_framework import serializers
from rest_framework.permissions import AllowAny, IsAuthenticated
from rest_framework.response import Response
from rest_framework.views import APIView

from backend.api_serializers import ApiErrorSerializer
from backend.auth_email import (
    get_email_verification_resend_remaining_seconds,
    mark_email_verified,
    resolve_password_reset_user,
    resolve_email_verification_user,
    send_email_verification_message,
    send_password_reset_message,
)
from backend.auth_serializers import (
    AuthCsrfResponseSerializer,
    AuthLoginRequestSerializer,
    AuthLoginResponseSerializer,
    AuthLogoutResponseSerializer,
    AuthMeResponseSerializer,
    AuthPasswordResetConfirmRequestSerializer,
    AuthPasswordResetConfirmResponseSerializer,
    AuthPasswordResetRequestResponseSerializer,
    AuthPasswordResetRequestSerializer,
    AuthPasswordResetValidateRequestSerializer,
    AuthPasswordResetValidateResponseSerializer,
    AuthRegisterResponseSerializer,
    AuthRegisterRequestSerializer,
    AuthResendVerificationResponseSerializer,
    AuthUserSerializer,
    AuthVerificationStatusResponseSerializer,
    AuthVerifyEmailRequestSerializer,
    AuthVerifyEmailResponseSerializer,
)
from backend.throttles import (
    AuthLoginRateThrottle,
    AuthPasswordResetRequestRateThrottle,
    AuthRegisterRateThrottle,
    AuthVerifyEmailResendRateThrottle,
)
from users_app.models import CustomerProfile

User = get_user_model()
logger = logging.getLogger(__name__)
USERNAME_SANITIZE_RE = re.compile(r"[^A-Za-z0-9.@+-_]+")


class InvalidCredentials(APIException):
    status_code = 400
    default_code = "invalid_credentials"
    default_detail = "Invalid email or password"


def enforce_csrf(request) -> None:
    check = CSRFCheck(lambda req: None)
    check.process_request(request)
    reason = check.process_view(request, None, (), {})
    if reason:
        raise PermissionDenied(f"CSRF Failed: {reason}")


def build_unique_username(email: str, requested_username: str = "") -> str:
    max_length = User._meta.get_field("username").max_length
    base_username = requested_username.strip()

    if not base_username:
        local_part = email.split("@", 1)[0].strip().lower()
        base_username = USERNAME_SANITIZE_RE.sub("-", local_part).strip("._-+@")

    if not base_username:
        base_username = "user"

    candidate = base_username[:max_length]
    if not User.objects.filter(username__iexact=candidate).exists():
        return candidate

    suffix = 2
    while True:
        suffix_text = f"-{suffix}"
        trimmed = base_username[: max_length - len(suffix_text)].rstrip("._-+@")
        if not trimmed:
            trimmed = "user"
        candidate = f"{trimmed}{suffix_text}"
        if not User.objects.filter(username__iexact=candidate).exists():
            return candidate
        suffix += 1


@extend_schema(
    tags=["Auth"],
    responses={
        200: AuthCsrfResponseSerializer,
    },
)
class AuthCsrfView(APIView):
    permission_classes = [AllowAny]

    def get(self, request):
        csrf_token = get_token(request)
        return Response({"ok": True, "csrfToken": csrf_token})


class AuthPasswordResetRequestView(APIView):
    permission_classes = [AllowAny]
    throttle_classes = [AuthPasswordResetRequestRateThrottle]

    @extend_schema(
        tags=["Auth"],
        request=AuthPasswordResetRequestSerializer,
        responses={
            200: AuthPasswordResetRequestResponseSerializer,
            400: OpenApiResponse(response=ApiErrorSerializer, description="Validation error"),
            403: OpenApiResponse(response=ApiErrorSerializer, description="CSRF failed"),
        },
    )
    def post(self, request):
        enforce_csrf(request)
        serializer = AuthPasswordResetRequestSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)

        email = serializer.validated_data["email"].strip().lower()
        user = User.objects.filter(email__iexact=email).first()

        if user and (user.email or "").strip():
            try:
                send_password_reset_message(user)
            except Exception:
                logger.exception("Failed to send password reset email", extra={"user_id": user.pk})

        return Response(
            {
                "ok": True,
                "email": email,
                "sent": True,
                "message": "If an account exists for this email, we sent a password reset link.",
            }
        )


class AuthPasswordResetValidateView(APIView):
    authentication_classes = []
    permission_classes = [AllowAny]

    @extend_schema(
        tags=["Auth"],
        request=AuthPasswordResetValidateRequestSerializer,
        responses={
            200: AuthPasswordResetValidateResponseSerializer,
            400: OpenApiResponse(response=ApiErrorSerializer, description="Validation error"),
        },
    )
    def post(self, request):
        serializer = AuthPasswordResetValidateRequestSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)

        try:
            resolve_password_reset_user(
                serializer.validated_data["uid"],
                serializer.validated_data["token"],
            )
        except (TypeError, ValueError, OverflowError, signing.BadSignature, User.DoesNotExist):
            return Response(
                {
                    "ok": True,
                    "valid": False,
                    "message": "Password reset link is invalid or expired.",
                }
            )

        return Response(
            {
                "ok": True,
                "valid": True,
                "message": "Password reset link is valid.",
            }
        )


class AuthPasswordResetConfirmView(APIView):
    authentication_classes = []
    permission_classes = [AllowAny]

    @extend_schema(
        tags=["Auth"],
        request=AuthPasswordResetConfirmRequestSerializer,
        responses={
            200: AuthPasswordResetConfirmResponseSerializer,
            400: OpenApiResponse(response=ApiErrorSerializer, description="Validation error"),
        },
    )
    def post(self, request):
        serializer = AuthPasswordResetConfirmRequestSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)

        uid = serializer.validated_data["uid"]
        token = serializer.validated_data["token"]
        password = serializer.validated_data["password"]
        password_confirm = serializer.validated_data["password_confirm"]

        details = {}

        if password != password_confirm:
            details["password_confirm"] = ["Passwords do not match."]

        try:
            user = resolve_password_reset_user(uid, token)
        except (TypeError, ValueError, OverflowError, signing.BadSignature, User.DoesNotExist):
            details["token"] = ["Password reset link is invalid or expired."]
            user = None

        if user is not None:
            try:
                validate_password(password, user=user)
            except DjangoValidationError as error:
                details["password"] = list(error.messages)

        if details:
            raise serializers.ValidationError(details)

        user.set_password(password)
        user.save(update_fields=["password"])
        return Response({"ok": True, "message": "Password updated successfully."})


class AuthLoginView(APIView):
    permission_classes = [AllowAny]
    throttle_classes = [AuthLoginRateThrottle]

    @extend_schema(
        tags=["Auth"],
        request=AuthLoginRequestSerializer,
        responses={
            200: AuthLoginResponseSerializer,
            400: OpenApiResponse(response=ApiErrorSerializer, description="Validation or invalid credentials"),
            403: OpenApiResponse(response=ApiErrorSerializer, description="CSRF failed"),
        },
    )
    def post(self, request):
        enforce_csrf(request)
        serializer = AuthLoginRequestSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)

        email = serializer.validated_data["email"].strip().lower()
        password = serializer.validated_data["password"]
        user_record = User.objects.filter(email__iexact=email).first()
        user = None

        if user_record is not None:
            user = authenticate(
                request=request,
                username=user_record.get_username(),
                password=password,
            )
        if user is None:
            raise InvalidCredentials()

        login(request, user)
        rotate_token(request)
        get_token(request)
        return Response({"ok": True, "user": AuthUserSerializer(user).data})


class AuthRegisterView(APIView):
    permission_classes = [AllowAny]
    throttle_classes = [AuthRegisterRateThrottle]

    @extend_schema(
        tags=["Auth"],
        request=AuthRegisterRequestSerializer,
        responses={
            200: AuthRegisterResponseSerializer,
            400: OpenApiResponse(response=ApiErrorSerializer, description="Validation error"),
            403: OpenApiResponse(response=ApiErrorSerializer, description="CSRF failed"),
        },
    )
    def post(self, request):
        enforce_csrf(request)
        serializer = AuthRegisterRequestSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)

        email = serializer.validated_data["email"].strip().lower()
        password = serializer.validated_data["password"]
        password_confirm = serializer.validated_data["password_confirm"]
        requested_username = serializer.validated_data.get("username", "")
        username = build_unique_username(email, requested_username)

        details = {}

        if requested_username.strip() and User.objects.filter(username__iexact=requested_username.strip()).exists():
            details["username"] = ["A user with this username already exists."]

        if User.objects.filter(email__iexact=email).exists():
            details["email"] = ["A user with this email already exists."]

        if password != password_confirm:
            details["password_confirm"] = ["Passwords do not match."]

        try:
            validate_password(password)
        except DjangoValidationError as error:
            details["password"] = list(error.messages)

        if details:
            raise serializers.ValidationError(details)

        user = User.objects.create_user(username=username, email=email, password=password)
        login(request, user)
        rotate_token(request)
        get_token(request)
        verification_email_sent = True

        try:
            send_email_verification_message(user)
        except Exception:
            logger.exception("Failed to send verification email", extra={"user_id": user.pk})
            verification_email_sent = False

        return Response(
            {
                "ok": True,
                "user": AuthUserSerializer(user).data,
                "verification_email": user.email,
                "verification_email_sent": verification_email_sent,
                "resend_available_in_seconds": (
                    settings.EMAIL_VERIFICATION_RESEND_COOLDOWN_SECONDS if verification_email_sent else 0
                ),
            }
        )


class AuthVerifyEmailView(APIView):
    authentication_classes = []
    permission_classes = [AllowAny]

    @extend_schema(
        tags=["Auth"],
        request=AuthVerifyEmailRequestSerializer,
        responses={
            200: AuthVerifyEmailResponseSerializer,
            400: OpenApiResponse(response=ApiErrorSerializer, description="Invalid or expired token"),
        },
    )
    def post(self, request):
        serializer = AuthVerifyEmailRequestSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)

        try:
            user = resolve_email_verification_user(serializer.validated_data["token"])
        except signing.SignatureExpired:
            raise serializers.ValidationError({"token": ["Verification link has expired."]})
        except (signing.BadSignature, User.DoesNotExist):
            raise serializers.ValidationError({"token": ["Verification link is invalid."]})

        already_verified = mark_email_verified(user)
        return Response(
            {
                "ok": True,
                "email": user.email,
                "email_verified": True,
                "already_verified": already_verified,
                "message": "Email confirmed." if not already_verified else "Email was already confirmed.",
            }
        )


class AuthResendVerificationView(APIView):
    permission_classes = [IsAuthenticated]
    throttle_classes = [AuthVerifyEmailResendRateThrottle]

    @extend_schema(
        tags=["Auth"],
        request=None,
        responses={
            200: AuthResendVerificationResponseSerializer,
            400: OpenApiResponse(response=ApiErrorSerializer, description="Validation error"),
            403: OpenApiResponse(response=ApiErrorSerializer, description="CSRF failed"),
        },
    )
    def post(self, request):
        enforce_csrf(request)
        user = request.user

        if not user.email:
            raise serializers.ValidationError({"email": ["Add an email address before requesting verification."]})

        profile, _ = CustomerProfile.objects.get_or_create(user=user)
        if profile.email_verified_at is not None:
            return Response(
                {
                    "ok": True,
                    "email": user.email,
                    "sent": False,
                    "already_verified": True,
                    "message": "Email is already confirmed.",
                    "resend_available_in_seconds": 0,
                }
            )

        resend_available_in_seconds = get_email_verification_resend_remaining_seconds(user)
        if resend_available_in_seconds > 0:
            raise serializers.ValidationError(
                {
                    "resend_available_in_seconds": [
                        f"Wait {resend_available_in_seconds} seconds before requesting another email."
                    ]
                }
            )

        send_email_verification_message(user)
        return Response(
            {
                "ok": True,
                "email": user.email,
                "sent": True,
                "already_verified": False,
                "message": "Verification email sent.",
                "resend_available_in_seconds": settings.EMAIL_VERIFICATION_RESEND_COOLDOWN_SECONDS,
            }
        )


class AuthVerificationStatusView(APIView):
    permission_classes = [IsAuthenticated]

    @extend_schema(
        tags=["Auth"],
        request=None,
        responses={
            200: AuthVerificationStatusResponseSerializer,
            401: OpenApiResponse(response=ApiErrorSerializer),
            403: OpenApiResponse(response=ApiErrorSerializer),
        },
    )
    def get(self, request):
        profile, _ = CustomerProfile.objects.get_or_create(user=request.user)
        if not (request.user.email or "").strip():
            return Response(
                {
                    "ok": True,
                    "email": "",
                    "email_verified": True,
                    "resend_available_in_seconds": 0,
                }
            )
        return Response(
            {
                "ok": True,
                "email": request.user.email or "",
                "email_verified": profile.email_verified_at is not None,
                "resend_available_in_seconds": get_email_verification_resend_remaining_seconds(request.user),
            }
        )


class AuthLogoutView(APIView):
    permission_classes = [IsAuthenticated]

    @extend_schema(
        tags=["Auth"],
        request=None,
        responses={
            200: AuthLogoutResponseSerializer,
            403: OpenApiResponse(response=ApiErrorSerializer, description="CSRF failed"),
        },
    )
    def post(self, request):
        enforce_csrf(request)
        logout(request)
        rotate_token(request)
        return Response({"ok": True})


class AuthMeView(APIView):
    permission_classes = [IsAuthenticated]

    @extend_schema(
        tags=["Auth"],
        responses={
            200: AuthMeResponseSerializer,
            401: OpenApiResponse(response=ApiErrorSerializer),
            403: OpenApiResponse(response=ApiErrorSerializer),
        },
    )
    def get(self, request):
        return Response({"ok": True, "user": AuthUserSerializer(request.user).data})
