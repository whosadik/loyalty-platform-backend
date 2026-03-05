from rest_framework import serializers


class CheckoutItemSerializer(serializers.Serializer):
    product = serializers.IntegerField()
    quantity = serializers.IntegerField(min_value=1, default=1)


class CheckoutRequestSerializer(serializers.Serializer):
    channel = serializers.CharField(required=False, default="offline")
    items = CheckoutItemSerializer(many=True)
    idempotency_key = serializers.CharField(required=False, allow_blank=False)

    apply_assignment_id = serializers.IntegerField(required=False)
    redeem_points = serializers.IntegerField(required=False, min_value=1)


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
    target = serializers.JSONField(required=False, allow_null=True)
    eligible_total = serializers.CharField()
    points_redeemed = serializers.IntegerField()
    points_earned = serializers.IntegerField()
    new_balance = serializers.IntegerField()
    tier = serializers.CharField(required=False, allow_null=True)
    next_offer = CheckoutNextOfferSerializer(required=False, allow_null=True)
