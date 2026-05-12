from rest_framework import serializers

from gift_cards.serializers import GiftCardSnapshotSerializer
from roadmap_app.serializers import RoadmapStepSnapshotSerializer


class CheckoutItemSerializer(serializers.Serializer):
    product = serializers.IntegerField()
    quantity = serializers.IntegerField(min_value=1, default=1)


class CheckoutRequestSerializer(serializers.Serializer):
    channel = serializers.CharField(required=False, default="offline")
    items = CheckoutItemSerializer(many=True)
    idempotency_key = serializers.CharField(required=False, allow_blank=False, max_length=64)

    apply_assignment_id = serializers.IntegerField(required=False)
    redeem_points = serializers.IntegerField(required=False, min_value=1)
    gift_card_code = serializers.CharField(required=False, allow_blank=False, max_length=64)


class CheckoutOfferSerializer(serializers.Serializer):
    id = serializers.IntegerField()
    name = serializers.CharField()
    type = serializers.CharField()
    value = serializers.CharField()
    estimated_cost = serializers.CharField(required=False, allow_blank=True)


class CheckoutNextOfferSerializer(serializers.Serializer):
    assignment_id = serializers.IntegerField()
    offer = CheckoutOfferSerializer()
    target = serializers.JSONField(required=False, allow_null=True)
    reason = serializers.JSONField(required=False, allow_null=True)
    expires_at = serializers.DateTimeField(allow_null=True)


class CheckoutCommitResponseSerializer(serializers.Serializer):
    ok = serializers.BooleanField()
    idempotent_replay = serializers.BooleanField(required=False)
    transaction_id = serializers.IntegerField()
    gross_total = serializers.CharField()
    discount_amount = serializers.CharField()
    net_total = serializers.CharField()
    offer_applied = serializers.BooleanField()
    offer_assignment_id = serializers.IntegerField(required=False, allow_null=True)
    public_campaign_id = serializers.IntegerField(required=False, allow_null=True)
    public_offer_id = serializers.IntegerField(required=False, allow_null=True)
    applied_offer = serializers.JSONField(required=False, allow_null=True)
    target = serializers.JSONField(required=False, allow_null=True)
    eligible_total = serializers.CharField()
    points_redeemed = serializers.IntegerField()
    points_earned = serializers.IntegerField()
    tier_points_multiplier = serializers.CharField(required=False)
    new_balance = serializers.IntegerField()
    gift_card = GiftCardSnapshotSerializer(required=False, allow_null=True)
    tier = serializers.CharField(required=False, allow_null=True)
    new_tier = serializers.CharField(required=False, allow_null=True)
    tier_upgraded = serializers.BooleanField(required=False)
    next_offer = CheckoutNextOfferSerializer(required=False, allow_null=True)
    next_roadmap_step = RoadmapStepSnapshotSerializer(required=False, allow_null=True)


class CheckoutLastResponseSerializer(serializers.Serializer):
    ok = serializers.BooleanField()
    checkout = serializers.JSONField(allow_null=True)
