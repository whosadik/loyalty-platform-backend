from __future__ import annotations

from django.db import models
from django.http import Http404
from django.utils import timezone
from rest_framework import status
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework.views import APIView

from backend.request_language import get_request_language
from drf_spectacular.types import OpenApiTypes
from drf_spectacular.utils import OpenApiExample, OpenApiParameter, extend_schema

from offers.models import OfferAssignment
from roadmap_app.events import (
    build_step_event_context,
    record_roadmap_event,
    record_step_exposed_dedup,
)
from roadmap_app.models import RoadmapEvent
from roadmap_app.models import RoadmapStep
from roadmap_app.serializers import (
    RoadmapPlanReadSerializer,
    RoadmapQuerySerializer,
    RoadmapRefreshRequestSerializer,
    RoadmapStepPatchRequestSerializer,
    RoadmapStepReadSerializer,
)
from roadmap_app.services import (
    get_active_plan,
    patch_step_status,
    refresh_roadmap,
    resolve_primary_roadmap_category,
)


def _active_offer_assignment_for_step(user, step: RoadmapStep | None) -> OfferAssignment | None:
    if not step:
        return None
    now = timezone.now()
    qs = (
        OfferAssignment.objects.filter(user=user, is_active=True, is_redeemed=False)
        .filter(
            models.Q(expires_at__isnull=True) | models.Q(expires_at__gt=now)
        )
        .order_by("-assigned_at")
    )
    for assignment in qs[:10]:
        target = assignment.target or {}
        scope = str(target.get("scope") or "").strip()
        value = target.get("value")
        category_ok = (not target.get("category")) or (str(target.get("category")) == str(step.plan.category))

        if scope == "product_id" and step.recommended_product_id:
            try:
                if int(value) == int(step.recommended_product_id):
                    return assignment
            except Exception:
                pass
        if scope == "product_type":
            if str(value) == str(step.product_type) and category_ok:
                return assignment
        if scope == "product_id" and target.get("product_type"):
            if str(target.get("product_type")) == str(step.product_type) and category_ok:
                return assignment
    return None


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
                            "step_id": 12,
                            "plan_id": 7,
                            "category": "haircare",
                            "step_index": 1,
                            "product_type": "shampoo",
                            "status": "completed",
                            "title": "Очищение кожи головы",
                            "description": "Выберите шампунь по типу кожи головы и частоте мытья.",
                            "recommended_product_id": None,
                            "recommended_product": None,
                            "suggestions": [],
                            "why": ["picked via rules", "already owned"],
                        },
                        {
                            "id": 13,
                            "step_id": 13,
                            "plan_id": 7,
                            "category": "haircare",
                            "step_index": 2,
                            "product_type": "conditioner",
                            "status": "recommended",
                            "title": "Кондиционирование",
                            "description": "Используйте кондиционер для защиты длины и блеска волос.",
                            "recommended_product_id": 18,
                            "recommended_product": {"id": 18, "name": "Conditioner X", "image_url": "https://example.com/conditioner.jpg"},
                            "suggestions": [18, 19, 20],
                            "why": ["picked via rules", "recommended via reranker/cooc"],
                        },
                    ],
                    "summary": {
                        "next_step": {"id": 13, "step_id": 13, "step_index": 2, "product_type": "conditioner", "status": "recommended", "title": "Кондиционирование", "recommended_product_id": 18},
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
            resolved_category = category or resolve_primary_roadmap_category(request.user)
            plan = refresh_roadmap(request.user, category=resolved_category, post_ctx=None)

        next_step = (
            plan.steps.filter(status__in=[RoadmapStep.Status.MISSING, RoadmapStep.Status.RECOMMENDED])
            .order_by("step_index")
            .first()
        )
        if next_step:
            offer_assignment = _active_offer_assignment_for_step(request.user, next_step)
            request_id = getattr(request, "request_id", None) or request.headers.get("X-Request-ID")
            record_step_exposed_dedup(
                user=request.user,
                plan=plan,
                step=next_step,
                request_id=request_id,
                offer_assignment_id=offer_assignment.id if offer_assignment else None,
            )
        language = get_request_language(request)
        return Response(
            RoadmapPlanReadSerializer(
                plan,
                context={"request": request, "language": language},
            ).data
        )


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
                        "next_step": {"step_id": 13, "step_index": 2, "product_type": "conditioner", "title": "Кондиционирование"},
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
        category = s.validated_data.get("category") or resolve_primary_roadmap_category(request.user)
        plan = refresh_roadmap(request.user, category=category, post_ctx=None)
        language = get_request_language(request)
        return Response(
            RoadmapPlanReadSerializer(
                plan,
                context={"request": request, "language": language},
            ).data
        )


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
        if next_status == RoadmapStep.Status.SKIPPED:
            offer_assignment = _active_offer_assignment_for_step(request.user, step)
            request_id = getattr(request, "request_id", None) or request.headers.get("X-Request-ID")
            record_roadmap_event(
                user=request.user,
                event_type=RoadmapEvent.Type.STEP_SKIPPED,
                plan=step.plan,
                step=step,
                request_id=request_id,
                context=build_step_event_context(
                    category=step.plan.category,
                    step=step,
                    offer_assignment_id=offer_assignment.id if offer_assignment else None,
                ),
            )
        language = get_request_language(request)
        return Response(
            {
                "ok": True,
                "step": RoadmapStepReadSerializer(
                    step,
                    context={
                        "request": request,
                        "language": language,
                        "category": step.plan.category,
                        "plan_id": step.plan_id,
                    },
                ).data,
            }
        )


class MeRoadmapStepClickView(APIView):
    permission_classes = [IsAuthenticated]

    @extend_schema(
        tags=["Roadmap"],
        responses={200: OpenApiTypes.OBJECT, 404: OpenApiTypes.OBJECT},
        examples=[
            OpenApiExample(
                "Click response",
                response_only=True,
                value={"ok": True, "step_id": 42},
            )
        ],
    )
    def post(self, request, step_id: int):
        try:
            step = RoadmapStep.objects.select_related("plan").get(
                id=int(step_id),
                plan__user=request.user,
                plan__is_active=True,
            )
        except RoadmapStep.DoesNotExist:
            raise Http404

        offer_assignment = _active_offer_assignment_for_step(request.user, step)
        request_id = getattr(request, "request_id", None) or request.headers.get("X-Request-ID")
        record_roadmap_event(
            user=request.user,
            event_type=RoadmapEvent.Type.STEP_CLICKED,
            plan=step.plan,
            step=step,
            request_id=request_id,
            context=build_step_event_context(
                category=step.plan.category,
                step=step,
                offer_assignment_id=offer_assignment.id if offer_assignment else None,
            ),
        )
        return Response({"ok": True, "step_id": step.id})
