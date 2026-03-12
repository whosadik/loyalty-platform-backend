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


class ProfileTaxonomyOptionSerializer(serializers.Serializer):
    value = serializers.CharField()
    label = serializers.CharField()
    aliases = serializers.ListField(child=serializers.CharField(), required=False)


class ProfileTaxonomyBudgetOptionSerializer(ProfileTaxonomyOptionSerializer):
    min = serializers.IntegerField(allow_null=True, required=False)
    max = serializers.IntegerField(allow_null=True, required=False)
    currency = serializers.CharField(required=False)


class ProfileTaxonomyStepSerializer(serializers.Serializer):
    id = serializers.IntegerField()
    key = serializers.CharField()
    title = serializers.CharField()
    description = serializers.CharField()
    optional = serializers.BooleanField()


class ProfileTaxonomySerializer(serializers.Serializer):
    steps = ProfileTaxonomyStepSerializer(many=True)
    skin_types = ProfileTaxonomyOptionSerializer(many=True)
    goals = ProfileTaxonomyOptionSerializer(many=True)
    avoid_flags = ProfileTaxonomyOptionSerializer(many=True)
    budget_options = ProfileTaxonomyBudgetOptionSerializer(many=True)
    hair_types = ProfileTaxonomyOptionSerializer(many=True)
    hair_concerns = ProfileTaxonomyOptionSerializer(many=True)
    coverage_options = ProfileTaxonomyOptionSerializer(many=True)
    fragrance_notes = ProfileTaxonomyOptionSerializer(many=True)
    intensity_options = ProfileTaxonomyOptionSerializer(many=True)


class MeProfileTaxonomyResponseSerializer(serializers.Serializer):
    ok = serializers.BooleanField()
    taxonomy = ProfileTaxonomySerializer()
