from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated

from backend.throttles import RecsRateThrottle
from recs_analytics.models import RecommendationEvent
from recs_analytics.serializers import RecEventCreateSerializer


class RecEventCreateView(APIView):
    permission_classes = [IsAuthenticated]
    throttle_classes = [RecsRateThrottle]

    def post(self, request):
        s = RecEventCreateSerializer(data=request.data)
        s.is_valid(raise_exception=True)
        data = s.validated_data

        rid = getattr(request, "request_id", None)

        RecommendationEvent.objects.create(
            user=request.user,
            action=data["action"],
            product_id=data["product_id"],
            page=data.get("page", "home"),
            section_key=data.get("section_key"),
            request_id=rid,
        )
        return Response({"ok": True})
