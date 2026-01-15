from rest_framework import serializers


class RedeemOfferRequestSerializer(serializers.Serializer):
    assignment_id = serializers.IntegerField()
    transaction_id = serializers.IntegerField()

class PreviewItemSerializer(serializers.Serializer):
    product = serializers.IntegerField()
    quantity = serializers.IntegerField(min_value=1, default=1)
    unit_price = serializers.DecimalField(max_digits=10, decimal_places=2)


class OfferPreviewRequestSerializer(serializers.Serializer):
    assignment_id = serializers.IntegerField()
    items = PreviewItemSerializer(many=True)