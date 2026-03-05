from rest_framework import serializers


class ApiErrorSerializer(serializers.Serializer):
    ok = serializers.BooleanField(default=False)
    code = serializers.CharField()
    message = serializers.CharField()
    details = serializers.JSONField(required=False, allow_null=True)
    request_id = serializers.CharField(required=False, allow_null=True)
