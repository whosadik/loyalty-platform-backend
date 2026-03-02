from __future__ import annotations

from django.http import Http404
from rest_framework import status
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework.views import APIView

from drf_spectacular.types import OpenApiTypes
from drf_spectacular.utils import OpenApiExample, OpenApiParameter, extend_schema

from roadmap_app.models import RoadmapStep
from roadmap_app.serializers import (
    RoadmapPlanReadSerializer,
    RoadmapQuerySerializer,
    RoadmapRefreshRequestSerializer,
    RoadmapStepPatchRequestSerializer,
    RoadmapStepReadSerializer,
)
from roadmap_app.services import get_active_plan, patch_step_status, refresh_roadmap


class MeRoadmapView(APIView):
    permission_classes = [IsAuthenticated]

    @extend_schema(
        tags=["Roadmap"],
        parameters=[
            OpenApiParameter(
                name="category",
                required=False,
                type=OpenApiTypes.STR,
                enum=["skincare", "haircare", "makeup", "fragrance"],
                description="Roadmap category. Recommended for deterministic response.",
            )
        ],
        responses={200: OpenApiTypes.OBJECT, 400: OpenApiTypes.OBJECT},
        examples=[
            OpenApiExample(
                "Roadmap response (haircare)",
                response_only=True,
                value={
                    "id": 7,
                    "category": "haircare",
                    "is_active": True,
                    "version": 1,
                    "meta": {"source": "roadmap_v1"},
                    "steps": [
                        {
                            "id": 12,
                            "step_index": 1,
                            "product_type": "shampoo",
                            "status": "completed",
                            "recommended_product": None,
                            "suggestions": [],
                            "why": ["picked via rules", "already owned"],
                        },
                        {
                            "id": 13,
                            "step_index": 2,
                            "product_type": "conditioner",
                            "status": "recommended",
                            "recommended_product": {"id": 18, "name": "Conditioner X"},
                            "suggestions": [18, 19, 20],
                            "why": ["picked via rules", "recommended via reranker/cooc"],
                        },
                    ],
                    "summary": {
                        "next_step": {"id": 13, "step_index": 2, "product_type": "conditioner", "status": "recommended"},
                        "missing_steps_count": 3,
                        "total_steps": 5,
                    },
                },
            ),
        ],
    )
    def get(self, request):
        q = RoadmapQuerySerializer(data=request.query_params)
        q.is_valid(raise_exception=True)
        category = q.validated_data.get("category")

        plan = get_active_plan(request.user, category=category or "")
        if not plan:
            if not category:
                return Response(
                    {"ok": False, "message": "Provide ?category=skincare|haircare|makeup|fragrance"},
                    status=status.HTTP_400_BAD_REQUEST,
                )
            plan = refresh_roadmap(request.user, category=category, post_ctx=None)
        return Response(RoadmapPlanReadSerializer(plan).data)


class MeRoadmapRefreshView(APIView):
    permission_classes = [IsAuthenticated]

    @extend_schema(
        tags=["Roadmap"],
        request=RoadmapRefreshRequestSerializer,
        responses={200: OpenApiTypes.OBJECT, 400: OpenApiTypes.OBJECT},
        examples=[
            OpenApiExample(
                "Refresh request",
                request_only=True,
                value={"category": "haircare"},
            ),
            OpenApiExample(
                "Refresh response (sample)",
                response_only=True,
                value={
                    "id": 7,
                    "category": "haircare",
                    "summary": {
                        "next_step": {"step_index": 2, "product_type": "conditioner"},
                        "missing_steps_count": 3,
                        "total_steps": 5,
                    },
                },
            ),
        ],
    )
    def post(self, request):
        s = RoadmapRefreshRequestSerializer(data=request.data)
        s.is_valid(raise_exception=True)
        category = s.validated_data["category"]
        plan = refresh_roadmap(request.user, category=category, post_ctx=None)
        return Response(RoadmapPlanReadSerializer(plan).data)


class MeRoadmapStepPatchView(APIView):
    permission_classes = [IsAuthenticated]

    @extend_schema(
        tags=["Roadmap"],
        request=RoadmapStepPatchRequestSerializer,
        responses={200: OpenApiTypes.OBJECT, 400: OpenApiTypes.OBJECT, 404: OpenApiTypes.OBJECT},
        examples=[
            OpenApiExample("Patch request", request_only=True, value={"status": "skipped"}),
            OpenApiExample(
                "Patch response",
                response_only=True,
                value={
                    "ok": True,
                    "step": {
                        "id": 13,
                        "step_index": 2,
                        "product_type": "conditioner",
                        "status": "skipped",
                    },
                },
            ),
        ],
    )
    def patch(self, request, step_id: int):
        s = RoadmapStepPatchRequestSerializer(data=request.data)
        s.is_valid(raise_exception=True)
        next_status = s.validated_data["status"]
        try:
            step = patch_step_status(user=request.user, step_id=int(step_id), status=next_status)
        except RoadmapStep.DoesNotExist:
            raise Http404
        except ValueError:
            return Response({"ok": False, "message": "Unsupported status"}, status=status.HTTP_400_BAD_REQUEST)
        return Response({"ok": True, "step": RoadmapStepReadSerializer(step).data})
