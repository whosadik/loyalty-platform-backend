from rest_framework import serializers
from offers.models import CampaignBudget


class CampaignSerializer(serializers.ModelSerializer):
    def validate(self, attrs):
        start = attrs.get("week_start_date")
        end = attrs.get("end_date")

        if self.instance is not None:
            if start is None:
                start = self.instance.week_start_date
            if end is None:
                end = self.instance.end_date

        if start and end and end < start:
            raise serializers.ValidationError({"end_date": "End date cannot be earlier than start date."})

        return attrs

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
            "end_date",
            "allowed_categories",
            "allowed_steps",
            "tiers",
            "promo_text",
            "banner_url",
        ]
