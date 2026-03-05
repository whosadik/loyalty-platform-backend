from __future__ import annotations

from django.db import transaction as db_tx
from django.utils import timezone

from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import serializers, status

from drf_spectacular.utils import OpenApiParameter, OpenApiResponse, extend_schema, inline_serializer

from backend.api_serializers import ApiErrorSerializer
from backend.permissions import HasStaffPermission
from offers.models import CampaignBudget
from offers.serializers_admin import CampaignSerializer


CampaignDetailResponseSerializer = inline_serializer(
    name="AdminCampaignDetailResponse",
    fields={
        "ok": serializers.BooleanField(),
        "campaign": CampaignSerializer(),
    },
)

CampaignListResponseSerializer = inline_serializer(
    name="AdminCampaignListResponse",
    fields={
        "ok": serializers.BooleanField(),
        "results": CampaignSerializer(many=True),
    },
)

CampaignPatchRequestSerializer = inline_serializer(
    name="AdminCampaignPatchRequest",
    fields={
        "name": serializers.CharField(required=False),
        "is_active": serializers.BooleanField(required=False),
        "priority": serializers.IntegerField(required=False),
        "weekly_limit": serializers.DecimalField(required=False, max_digits=12, decimal_places=2),
        "weekly_spent": serializers.DecimalField(required=False, max_digits=12, decimal_places=2),
        "week_start_date": serializers.DateField(required=False, allow_null=True),
        "allowed_categories": serializers.ListField(child=serializers.CharField(), required=False),
        "allowed_steps": serializers.ListField(child=serializers.CharField(), required=False),
        "reset_weekly_spent": serializers.BooleanField(required=False),
    },
)


class AdminCampaignListCreateView(APIView):
    """
    GET: requires view_metrics
    POST: requires manage_campaigns
    """

    def get_permissions(self):
        if self.request.method == "POST":
            return [HasStaffPermission.with_perm("manage_campaigns")()]
        return [HasStaffPermission.with_perm("view_metrics")()]

    @extend_schema(
        tags=["Admin"],
        description="List campaigns (budgets) with optional filters.",
        parameters=[
            OpenApiParameter(name="is_active", required=False, type=bool),
            OpenApiParameter(name="name", required=False, type=str, description="substring match"),
            OpenApiParameter(name="ordering", required=False, type=str, description="priority|name|-priority|-name"),
        ],
        responses={
            200: CampaignListResponseSerializer,
        },
    )
    def get(self, request):
        qs = CampaignBudget.objects.all()

        is_active = request.query_params.get("is_active")
        if is_active is not None:
            qs = qs.filter(is_active=str(is_active).lower() in {"1", "true", "yes", "on"})

        name = request.query_params.get("name")
        if name:
            qs = qs.filter(name__icontains=name)

        ordering = request.query_params.get("ordering") or "priority"
        if ordering in {"priority", "-priority", "name", "-name", "id", "-id"}:
            qs = qs.order_by(ordering, "id")
        else:
            qs = qs.order_by("priority", "id")

        data = CampaignSerializer(qs, many=True).data
        return Response({"ok": True, "results": data})

    @extend_schema(
        tags=["Admin"],
        description="Create campaign (budget).",
        request=CampaignSerializer,
        responses={
            201: CampaignDetailResponseSerializer,
            400: OpenApiResponse(response=ApiErrorSerializer),
        },
    )
    def post(self, request):
        s = CampaignSerializer(data=request.data)
        s.is_valid(raise_exception=True)
        c = s.save()
        return Response({"ok": True, "campaign": CampaignSerializer(c).data}, status=status.HTTP_201_CREATED)


class AdminCampaignDetailView(APIView):
    """
    GET: requires view_metrics
    PATCH: requires manage_campaigns
    """

    def get_permissions(self):
        if self.request.method == "PATCH":
            return [HasStaffPermission.with_perm("manage_campaigns")()]
        return [HasStaffPermission.with_perm("view_metrics")()]

    def _get(self, pk: int) -> CampaignBudget:
        return CampaignBudget.objects.get(pk=pk)

    @extend_schema(
        tags=["Admin"],
        description="Get campaign details.",
        responses={200: CampaignDetailResponseSerializer},
    )
    def get(self, request, pk: int):
        c = self._get(pk)
        return Response({"ok": True, "campaign": CampaignSerializer(c).data})

    @extend_schema(
        tags=["Admin"],
        description="Patch campaign fields. Optional: reset_weekly_spent=true resets weekly_spent and week_start_date.",
        request=CampaignPatchRequestSerializer,
        responses={
            200: CampaignDetailResponseSerializer,
            400: OpenApiResponse(response=ApiErrorSerializer),
        },
    )
    def patch(self, request, pk: int):
        with db_tx.atomic():
            c = CampaignBudget.objects.select_for_update().get(pk=pk)

            reset = request.data.get("reset_weekly_spent")
            if reset in {True, "true", "1", 1, "yes", "on"}:
                # reset to current week start if field exists
                if hasattr(c, "week_start_date"):
                    now = timezone.now()
                    ws = (now - timezone.timedelta(days=now.weekday())).date()
                    c.week_start_date = ws
                c.weekly_spent = 0
                c.save(update_fields=["weekly_spent"] + (["week_start_date"] if hasattr(c, "week_start_date") else []))

            serializer_data = request.data.copy()
            serializer_data.pop("reset_weekly_spent", None)
            s = CampaignSerializer(instance=c, data=serializer_data, partial=True)
            s.is_valid(raise_exception=True)
            c = s.save()

        return Response({"ok": True, "campaign": CampaignSerializer(c).data})
