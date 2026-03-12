from rest_framework import serializers
from .models import CustomerProfile


class CustomerProfileSerializer(serializers.ModelSerializer):
    class Meta:
        model = CustomerProfile
        fields = [
            "first_name",
            "last_name",
            "phone",
            "city",
            "skin_type",
            "goals",
            "avoid_flags",
            "budget",
            "hair_profile",
            "makeup_profile",
            "fragrance_profile",
            "profile_completed_at",
            "profile_completion_rewarded_at",
            "email_verified_at",
            "email_verification_sent_at",
            "created_at",
            "updated_at",
        ]
        read_only_fields = [
            "profile_completed_at",
            "profile_completion_rewarded_at",
            "email_verified_at",
            "email_verification_sent_at",
            "created_at",
            "updated_at",
        ]


class ProfileCompletionBonusSerializer(serializers.Serializer):
    ok = serializers.BooleanField()
    awarded = serializers.BooleanField()
    points_added = serializers.IntegerField()
    completed = serializers.BooleanField()


class MeProfileUpdateResponseSerializer(serializers.Serializer):
    ok = serializers.BooleanField()
    profile = CustomerProfileSerializer()
    profile_completion_bonus = ProfileCompletionBonusSerializer()
