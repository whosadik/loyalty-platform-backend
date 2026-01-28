from rest_framework import serializers
from audit.models import AuditEvent


class AuditEventSerializer(serializers.ModelSerializer):
    user_id = serializers.IntegerField(source="user.id", read_only=True)

    class Meta:
        model = AuditEvent
        fields = [
            "id",
            "created_at",
            "action",
            "user_id",
            "entity_type",
            "entity_id",
            "request_id",
            "path",
            "method",
            "status_code",
            "ip",
            "meta",
        ]
