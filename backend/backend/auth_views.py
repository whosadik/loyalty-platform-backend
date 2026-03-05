from django.contrib.auth import authenticate, login, logout
from django.middleware.csrf import get_token, rotate_token
from drf_spectacular.utils import OpenApiResponse, extend_schema
from rest_framework.authentication import CSRFCheck
from rest_framework.exceptions import APIException, PermissionDenied
from rest_framework.permissions import AllowAny, IsAuthenticated
from rest_framework.response import Response
from rest_framework.views import APIView

from backend.api_serializers import ApiErrorSerializer
from backend.auth_serializers import (
    AuthCsrfResponseSerializer,
    AuthLoginRequestSerializer,
    AuthLoginResponseSerializer,
    AuthLogoutResponseSerializer,
    AuthMeResponseSerializer,
    AuthUserSerializer,
)


class InvalidCredentials(APIException):
    status_code = 400
    default_code = "invalid_credentials"
    default_detail = "Invalid username or password"


def enforce_csrf(request) -> None:
    check = CSRFCheck(lambda req: None)
    check.process_request(request)
    reason = check.process_view(request, None, (), {})
    if reason:
        raise PermissionDenied(f"CSRF Failed: {reason}")


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


class AuthLoginView(APIView):
    permission_classes = [AllowAny]

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

        user = authenticate(
            request=request,
            username=serializer.validated_data["username"],
            password=serializer.validated_data["password"],
        )
        if user is None:
            raise InvalidCredentials()

        login(request, user)
        rotate_token(request)
        get_token(request)
        return Response({"ok": True, "user": AuthUserSerializer(user).data})


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
