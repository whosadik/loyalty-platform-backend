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
            "hair_profile",
            "makeup_profile",
            "fragrance_profile",
            "profile_completed_at",
            "profile_completion_rewarded_at",
            "created_at",
            "updated_at",
        ]
        read_only_fields = [
            "profile_completed_at",
            "profile_completion_rewarded_at",
            "created_at",
            "updated_at",
        ]
