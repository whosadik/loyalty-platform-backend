from rest_framework import serializers
from offers.models import CampaignBudget


class CampaignSerializer(serializers.ModelSerializer):
    class Meta:
        model = CampaignBudget
        fields = [
            "id",
            "name",
            "is_active",
            "priority",
            "weekly_limit",
            "weekly_spent",
            "week_start_date",
            "allowed_categories",
            "allowed_steps",
        ]
