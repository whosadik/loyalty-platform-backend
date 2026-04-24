from __future__ import annotations

from rest_framework import serializers, status
from rest_framework.response import Response
from rest_framework.views import APIView

from drf_spectacular.utils import OpenApiParameter, OpenApiResponse, extend_schema, inline_serializer

from backend.api_serializers import ApiErrorSerializer
from backend.permissions import HasStaffPermission
from offers.models import Offer
from offers.serializers_admin import OfferAdminSerializer


OfferDetailResponseSerializer = inline_serializer(
    name="AdminOfferDetailResponse",
    fields={
        "ok": serializers.BooleanField(),
        "offer": OfferAdminSerializer(),
    },
)

OfferListResponseSerializer = inline_serializer(
    name="AdminOfferListResponse",
    fields={
        "ok": serializers.BooleanField(),
        "results": OfferAdminSerializer(many=True),
    },
)


class AdminOfferListCreateView(APIView):
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
        description="List offers. Filter by campaign_id / is_active / offer_type.",
        parameters=[
            OpenApiParameter(name="campaign_id", required=False, type=int),
            OpenApiParameter(name="is_active", required=False, type=bool),
            OpenApiParameter(name="offer_type", required=False, type=str),
        ],
        responses={200: OfferListResponseSerializer},
    )
    def get(self, request):
        qs = Offer.objects.all().order_by("-id")

        campaign_id = request.query_params.get("campaign_id")
        if campaign_id:
            try:
                qs = qs.filter(campaign_id=int(campaign_id))
            except (TypeError, ValueError):
                qs = qs.none()

        is_active = request.query_params.get("is_active")
        if is_active is not None:
            qs = qs.filter(is_active=str(is_active).lower() in {"1", "true", "yes", "on"})

        offer_type = request.query_params.get("offer_type")
        if offer_type:
            qs = qs.filter(offer_type=offer_type)

        data = OfferAdminSerializer(qs, many=True).data
        return Response({"ok": True, "results": data})

    @extend_schema(
        tags=["Admin"],
        description="Create offer.",
        request=OfferAdminSerializer,
        responses={
            201: OfferDetailResponseSerializer,
            400: OpenApiResponse(response=ApiErrorSerializer),
        },
    )
    def post(self, request):
        s = OfferAdminSerializer(data=request.data)
        s.is_valid(raise_exception=True)
        offer = s.save()
        return Response(
            {"ok": True, "offer": OfferAdminSerializer(offer).data},
            status=status.HTTP_201_CREATED,
        )


class AdminOfferDetailView(APIView):
    """
    GET: requires view_metrics
    PATCH/DELETE: requires manage_campaigns
    """

    def get_permissions(self):
        if self.request.method in {"PATCH", "DELETE"}:
            return [HasStaffPermission.with_perm("manage_campaigns")()]
        return [HasStaffPermission.with_perm("view_metrics")()]

    def _get(self, pk: int) -> Offer:
        return Offer.objects.get(pk=pk)

    @extend_schema(
        tags=["Admin"],
        description="Get offer details.",
        responses={200: OfferDetailResponseSerializer},
    )
    def get(self, request, pk: int):
        offer = self._get(pk)
        return Response({"ok": True, "offer": OfferAdminSerializer(offer).data})

    @extend_schema(
        tags=["Admin"],
        description="Patch offer fields.",
        request=OfferAdminSerializer,
        responses={
            200: OfferDetailResponseSerializer,
            400: OpenApiResponse(response=ApiErrorSerializer),
        },
    )
    def patch(self, request, pk: int):
        offer = self._get(pk)
        s = OfferAdminSerializer(instance=offer, data=request.data, partial=True)
        s.is_valid(raise_exception=True)
        offer = s.save()
        return Response({"ok": True, "offer": OfferAdminSerializer(offer).data})

    @extend_schema(
        tags=["Admin"],
        description="Soft-delete offer: sets is_active=False.",
        responses={200: inline_serializer(name="AdminOfferDeleteResponse", fields={"ok": serializers.BooleanField()})},
    )
    def delete(self, request, pk: int):
        offer = self._get(pk)
        if offer.is_active:
            offer.is_active = False
            offer.save(update_fields=["is_active"])
        return Response({"ok": True})
