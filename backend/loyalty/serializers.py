from rest_framework import serializers


class RedeemPointsRequestSerializer(serializers.Serializer):
    points = serializers.IntegerField(min_value=1)
    reference = serializers.CharField(required=False, allow_blank=True, default="")
