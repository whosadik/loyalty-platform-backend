from django.contrib.auth import get_user_model
from rest_framework import serializers

from users_app.models import CustomerProfile


User = get_user_model()


class AuthUserSerializer(serializers.ModelSerializer):
    email_verified = serializers.SerializerMethodField()

    class Meta:
        model = User
        fields = ["id", "username", "email", "email_verified", "is_staff", "is_superuser"]

    def get_email_verified(self, obj) -> bool:
        if not (obj.email or "").strip():
            return True
        try:
            profile = obj.customerprofile
        except CustomerProfile.DoesNotExist:
            return False
        return profile.email_verified_at is not None


class AuthCsrfResponseSerializer(serializers.Serializer):
    ok = serializers.BooleanField()
    csrfToken = serializers.CharField()


class AuthLoginRequestSerializer(serializers.Serializer):
    email = serializers.EmailField()
    password = serializers.CharField(trim_whitespace=False, write_only=True)


class AuthRegisterRequestSerializer(serializers.Serializer):
    username = serializers.CharField(required=False, allow_blank=True)
    email = serializers.EmailField()
    password = serializers.CharField(trim_whitespace=False, write_only=True)
    password_confirm = serializers.CharField(trim_whitespace=False, write_only=True)


class AuthLoginResponseSerializer(serializers.Serializer):
    ok = serializers.BooleanField()
    user = AuthUserSerializer()


class AuthRegisterResponseSerializer(serializers.Serializer):
    ok = serializers.BooleanField()
    user = AuthUserSerializer()
    verification_email = serializers.EmailField()
    verification_email_sent = serializers.BooleanField()
    resend_available_in_seconds = serializers.IntegerField()


class AuthLogoutResponseSerializer(serializers.Serializer):
    ok = serializers.BooleanField()


class AuthMeResponseSerializer(serializers.Serializer):
    ok = serializers.BooleanField()
    user = AuthUserSerializer()


class AuthVerifyEmailRequestSerializer(serializers.Serializer):
    token = serializers.CharField()


class AuthVerifyEmailResponseSerializer(serializers.Serializer):
    ok = serializers.BooleanField()
    email = serializers.EmailField()
    email_verified = serializers.BooleanField()
    already_verified = serializers.BooleanField()
    message = serializers.CharField()


class AuthResendVerificationResponseSerializer(serializers.Serializer):
    ok = serializers.BooleanField()
    email = serializers.EmailField()
    sent = serializers.BooleanField()
    already_verified = serializers.BooleanField()
    message = serializers.CharField()
    resend_available_in_seconds = serializers.IntegerField()


class AuthVerificationStatusResponseSerializer(serializers.Serializer):
    ok = serializers.BooleanField()
    email = serializers.EmailField(allow_blank=True)
    email_verified = serializers.BooleanField()
    resend_available_in_seconds = serializers.IntegerField()


class AuthPasswordResetRequestSerializer(serializers.Serializer):
    email = serializers.EmailField()


class AuthPasswordResetRequestResponseSerializer(serializers.Serializer):
    ok = serializers.BooleanField()
    email = serializers.EmailField()
    sent = serializers.BooleanField()
    message = serializers.CharField()


class AuthPasswordResetValidateRequestSerializer(serializers.Serializer):
    uid = serializers.CharField()
    token = serializers.CharField()


class AuthPasswordResetValidateResponseSerializer(serializers.Serializer):
    ok = serializers.BooleanField()
    valid = serializers.BooleanField()
    message = serializers.CharField()


class AuthPasswordResetConfirmRequestSerializer(serializers.Serializer):
    uid = serializers.CharField()
    token = serializers.CharField()
    password = serializers.CharField(trim_whitespace=False, write_only=True)
    password_confirm = serializers.CharField(trim_whitespace=False, write_only=True)


class AuthPasswordResetConfirmResponseSerializer(serializers.Serializer):
    ok = serializers.BooleanField()
    message = serializers.CharField()
