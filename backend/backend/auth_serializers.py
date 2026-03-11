from django.contrib.auth import get_user_model
from rest_framework import serializers


User = get_user_model()


class AuthUserSerializer(serializers.ModelSerializer):
    class Meta:
        model = User
        fields = ["id", "username", "is_staff", "is_superuser"]


class AuthCsrfResponseSerializer(serializers.Serializer):
    ok = serializers.BooleanField()
    csrfToken = serializers.CharField()


class AuthLoginRequestSerializer(serializers.Serializer):
    username = serializers.CharField()
    password = serializers.CharField(trim_whitespace=False, write_only=True)


class AuthRegisterRequestSerializer(serializers.Serializer):
    username = serializers.CharField()
    password = serializers.CharField(trim_whitespace=False, write_only=True)
    password_confirm = serializers.CharField(trim_whitespace=False, write_only=True)


class AuthLoginResponseSerializer(serializers.Serializer):
    ok = serializers.BooleanField()
    user = AuthUserSerializer()


class AuthLogoutResponseSerializer(serializers.Serializer):
    ok = serializers.BooleanField()


class AuthMeResponseSerializer(serializers.Serializer):
    ok = serializers.BooleanField()
    user = AuthUserSerializer()
