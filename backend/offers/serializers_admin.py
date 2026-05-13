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
            if start is None:
                start = self.instance.week_start_date
            if end is None:
                end = self.instance.end_date

        if start and end and end < start:
            raise serializers.ValidationError({"end_date": "End date cannot be earlier than start date."})

        for field in ("allowed_categories", "allowed_steps", "allowed_brands"):
            if field in attrs and attrs[field] is None:
                attrs[field] = []
        if "allowed_product_ids" in attrs:
            attrs["allowed_product_ids"] = self._normalize_product_ids(attrs.get("allowed_product_ids"))
        if "recommendation_rules" in attrs and attrs["recommendation_rules"] is None:
            attrs["recommendation_rules"] = {}

        return attrs

    def _normalize_product_ids(self, value) -> list[int]:
        if not value:
            return []
        if not isinstance(value, list):
            raise serializers.ValidationError({"allowed_product_ids": "Expected a list of product ids."})
        out = []
        for item in value:
            try:
                product_id = int(item)
            except (TypeError, ValueError):
                raise serializers.ValidationError({"allowed_product_ids": "Product ids must be integers."})
            if product_id > 0 and product_id not in out:
                out.append(product_id)
        return out

    class Meta:
        model = CampaignBudget
        fields = [
            "id",
            "name",
            "campaign_type",
            "is_active",
            "priority",
            "weekly_limit",
            "weekly_spent",
            "start_date",
            "end_date",
            "allowed_categories",
            "allowed_steps",
            "allowed_brands",
            "allowed_product_ids",
            "tiers",
            "recommendation_rules",
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
            "allowed_brands",
            "allowed_product_ids",
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
        if target_scope == "brand":
            brands = attrs.get("allowed_brands", getattr(self.instance, "allowed_brands", []) or [])
            if not brands:
                raise serializers.ValidationError({"allowed_brands": "Required when target scope is 'brand'."})

        if "allowed_product_ids" in attrs:
            attrs["allowed_product_ids"] = self._normalize_product_ids(attrs.get("allowed_product_ids"))

        return attrs

    def _normalize_product_ids(self, value) -> list[int]:
        if not value:
            return []
        if not isinstance(value, list):
            raise serializers.ValidationError({"allowed_product_ids": "Expected a list of product ids."})
        out = []
        for item in value:
            try:
                product_id = int(item)
            except (TypeError, ValueError):
                raise serializers.ValidationError({"allowed_product_ids": "Product ids must be integers."})
            if product_id > 0 and product_id not in out:
                out.append(product_id)
        return out
