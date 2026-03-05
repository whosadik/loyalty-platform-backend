from rest_framework import serializers


class RedeemPointsRequestSerializer(serializers.Serializer):
    points = serializers.IntegerField(min_value=1)
    reference = serializers.CharField(required=False, allow_blank=True, default="")


class MeLoyaltyResponseSerializer(serializers.Serializer):
    tier = serializers.CharField(allow_null=True)
    points_balance = serializers.IntegerField()


class RedeemPointsResponseSerializer(serializers.Serializer):
    ok = serializers.BooleanField()
    new_balance = serializers.IntegerField()
