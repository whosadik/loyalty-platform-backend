from rest_framework import serializers


class CheckoutItemSerializer(serializers.Serializer):
    product = serializers.IntegerField()
    quantity = serializers.IntegerField(min_value=1, default=1)
    unit_price = serializers.DecimalField(max_digits=10, decimal_places=2)


class CheckoutRequestSerializer(serializers.Serializer):
    channel = serializers.CharField(required=False, default="offline")
    items = CheckoutItemSerializer(many=True)

    apply_assignment_id = serializers.IntegerField(required=False)
    redeem_points = serializers.IntegerField(required=False, min_value=1)
