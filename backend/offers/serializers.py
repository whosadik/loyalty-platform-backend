from rest_framework import serializers


class RedeemOfferRequestSerializer(serializers.Serializer):
    assignment_id = serializers.IntegerField()
    transaction_id = serializers.IntegerField()
