from rest_framework import serializers
from .models import CustomerProfile


class CustomerProfileSerializer(serializers.ModelSerializer):
    class Meta:
        model = CustomerProfile
        fields = [
            "skin_type",
            "goals",
            "avoid_flags",
            "budget",
            "created_at",
            "updated_at",
        ]
        read_only_fields = ["created_at", "updated_at"]
