from rest_framework import serializers
from offers.models import CampaignBudget, Offer


class CampaignSerializer(serializers.ModelSerializer):
    offers_count = serializers.SerializerMethodField()

    def get_offers_count(self, obj: CampaignBudget) -> int:
        return Offer.objects.filter(campaign=obj, is_active=True).count()

    def validate(self, attrs):
        start = attrs.get("start_date")
        end = attrs.get("end_date")

        if self.instance is not None:
            if start is None:
                start = self.instance.start_date
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
            "start_date",
            "end_date",
            "allowed_categories",
            "allowed_steps",
            "tiers",
            "promo_text",
            "banner_url",
            "offers_count",
        ]
        read_only_fields = ["weekly_spent", "offers_count"]


class OfferAdminSerializer(serializers.ModelSerializer):
    class Meta:
        model = Offer
        fields = [
            "id",
            "campaign",
            "name",
            "offer_type",
            "value",
            "target_scope",
            "estimated_cost",
            "cooldown_days",
            "expires_in_days",
            "allowed_categories",
            "allowed_product_types",
            "allowed_steps",
            "min_total_spend_90d",
            "is_active",
            "created_at",
        ]
        read_only_fields = ["created_at"]

    def validate_value(self, v):
        if v is None or v < 0:
            raise serializers.ValidationError("Value must be >= 0.")
        return v

    def validate(self, attrs):
        offer_type = attrs.get("offer_type") or getattr(self.instance, "offer_type", None)
        value = attrs.get("value") if "value" in attrs else getattr(self.instance, "value", None)
        target_scope = attrs.get("target_scope") or getattr(self.instance, "target_scope", None)

        if offer_type == "discount" and value is not None and value > 100:
            raise serializers.ValidationError({"value": "Discount percent cannot exceed 100."})

        if target_scope in {"category", "product_type", "product_id"}:
            cats = attrs.get("allowed_categories", getattr(self.instance, "allowed_categories", []) or [])
            pts = attrs.get("allowed_product_types", getattr(self.instance, "allowed_product_types", []) or [])
            if target_scope == "category" and not cats:
                raise serializers.ValidationError(
                    {"allowed_categories": "Required when target scope is 'category'."}
                )
            if target_scope == "product_type" and not pts:
                raise serializers.ValidationError(
                    {"allowed_product_types": "Required when target scope is 'product_type'."}
                )

        return attrs
