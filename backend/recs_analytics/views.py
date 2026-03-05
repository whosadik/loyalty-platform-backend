from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated
from drf_spectacular.utils import OpenApiResponse, extend_schema, inline_serializer
from rest_framework import serializers

from backend.api_serializers import ApiErrorSerializer
from backend.throttles import RecsRateThrottle
from recs_analytics.experiment import extract_experiment_context
from recs_analytics.models import RecommendationEvent
from recs_analytics.serializers import RecEventCreateSerializer


class RecEventCreateView(APIView):
    permission_classes = [IsAuthenticated]
    throttle_classes = [RecsRateThrottle]

    @extend_schema(
        tags=["Recommendations"],
        request=RecEventCreateSerializer,
        responses={
            200: inline_serializer(
                name="RecommendationEventCreateResponse",
                fields={
                    "ok": serializers.BooleanField(),
                },
            ),
            400: OpenApiResponse(response=ApiErrorSerializer),
        },
    )
    def post(self, request):
        s = RecEventCreateSerializer(data=request.data)
        s.is_valid(raise_exception=True)
        data = s.validated_data

        rid = getattr(request, "request_id", None)
        page = data.get("page", "home")
        section_key = data.get("section_key")
        extra_context = data.get("context") if isinstance(data.get("context"), dict) else {}

        # Try to inherit algo metadata from the latest impression for this product.
        imp_qs = RecommendationEvent.objects.filter(
            user=request.user,
            action=RecommendationEvent.Action.IMPRESSION,
            product_id=data["product_id"],
        )
        if rid:
            imp_qs = imp_qs.filter(request_id=rid)
        imp = imp_qs.order_by("-created_at").first()
        if imp and not section_key:
            section_key = imp.section_key
        if imp and not page:
            page = imp.page
        inherited_exp_ctx = extract_experiment_context((imp.context or {}) if imp else {})
        context = {}
        if extra_context:
            context.update(extra_context)
        if imp:
            context["from_impression_id"] = imp.id
            context.update(inherited_exp_ctx)

        RecommendationEvent.objects.create(
            user=request.user,
            action=data["action"],
            product_id=data["product_id"],
            page=page,
            section_key=section_key,
            request_id=rid,
            algo_mode=(imp.algo_mode if imp else None),
            score=(imp.score if imp else None),
            components=((imp.components or {}) if imp else {}),
            context=context,
        )
        return Response({"ok": True})
