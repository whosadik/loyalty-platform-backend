from rest_framework import serializers


class RedeemOfferRequestSerializer(serializers.Serializer):
    assignment_id = serializers.IntegerField()
    transaction_id = serializers.IntegerField()


class RedeemOfferResponseSerializer(serializers.Serializer):
    ok = serializers.BooleanField()
    earned_points = serializers.IntegerField()
    new_balance = serializers.IntegerField()
    tier = serializers.CharField(required=False, allow_null=True)
    discount_amount = serializers.CharField()

class PreviewItemSerializer(serializers.Serializer):
    product = serializers.IntegerField()
    quantity = serializers.IntegerField(min_value=1, default=1)


class OfferPreviewRequestSerializer(serializers.Serializer):
    assignment_id = serializers.IntegerField()
    items = PreviewItemSerializer(many=True)


class OfferClickRequestSerializer(serializers.Serializer):
    assignment_id = serializers.IntegerField()
    context = serializers.JSONField(required=False)
