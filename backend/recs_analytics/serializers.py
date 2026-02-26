from rest_framework import serializers
from recs_analytics.models import RecommendationEvent


class RecEventCreateSerializer(serializers.Serializer):
    action = serializers.ChoiceField(choices=[
        RecommendationEvent.Action.CLICK,
        RecommendationEvent.Action.ADD_TO_CART,
    ])
    product_id = serializers.IntegerField()
    page = serializers.CharField(required=False, default="home")
    section_key = serializers.CharField(required=False, allow_blank=True, allow_null=True)
    context = serializers.JSONField(required=False)
