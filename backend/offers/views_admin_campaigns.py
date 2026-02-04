from __future__ import annotations

from django.db import transaction as db_tx
from django.utils import timezone

from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status

from drf_spectacular.utils import extend_schema, OpenApiParameter

from backend.permissions import HasStaffPermission
from offers.models import CampaignBudget
from offers.serializers_admin import CampaignSerializer


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

    @extend_schema(tags=["Admin"], description="Get campaign details.")
    def get(self, request, pk: int):
        c = self._get(pk)
        return Response({"ok": True, "campaign": CampaignSerializer(c).data})

    @extend_schema(
        tags=["Admin"],
        description="Patch campaign fields. Optional: reset_weekly_spent=true resets weekly_spent and week_start_date.",
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

            s = CampaignSerializer(instance=c, data=request.data, partial=True)
            s.is_valid(raise_exception=True)
            c = s.save()

        return Response({"ok": True, "campaign": CampaignSerializer(c).data})
