from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated
from drf_spectacular.utils import OpenApiResponse, extend_schema

from backend.api_serializers import ApiErrorSerializer
from backend.request_language import get_request_language
from .models import CustomerProfile
from .profile_taxonomy import get_profile_taxonomy_payload
from .serializers import (
    CustomerProfileSerializer,
    MeProfileTaxonomyResponseSerializer,
    MeProfileUpdateResponseSerializer,
)
from .services import (
    favorite_category_snapshot,
    is_profile_complete,
    maybe_award_profile_completion_bonus,
)


class MeProfileView(APIView):
    permission_classes = [IsAuthenticated]

    @extend_schema(
        tags=["Me"],
        responses={200: CustomerProfileSerializer},
    )
    def get(self, request):
        profile, _ = CustomerProfile.objects.get_or_create(user=request.user)
        return Response(CustomerProfileSerializer(profile).data)

    @extend_schema(
        tags=["Me"],
        request=CustomerProfileSerializer,
        responses={
            200: MeProfileUpdateResponseSerializer,
            400: OpenApiResponse(response=ApiErrorSerializer),
        },
    )
    def put(self, request):
        profile, _ = CustomerProfile.objects.get_or_create(user=request.user)
        serializer = CustomerProfileSerializer(profile, data=request.data, partial=True)
        serializer.is_valid(raise_exception=True)
        profile = serializer.save()

        bonus_result = maybe_award_profile_completion_bonus(request.user, profile)

        return Response(
            {
                "ok": True,
                "profile": CustomerProfileSerializer(profile).data,
                "profile_completion_bonus": bonus_result,
            }
        )


class MeFavoriteCategoryView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request):
        profile, _ = CustomerProfile.objects.get_or_create(user=request.user)
        snap = favorite_category_snapshot(request.user)
        return Response(
            {
                "ok": True,
                "favorite_category": snap["favorite_category"],
                "window_days": snap["window_days"],
                "products_bought": snap.get("products_bought", 0),
                "total_spent": snap.get("total_spent", "0"),
                "currency": snap.get("currency"),
                "profile_complete": is_profile_complete(profile),
                "explain": {
                    "window_start": snap["window_start"],
                    "window_end": snap["window_end"],
                    "history_items_considered": snap["history_items_considered"],
                    "picked_by": snap["picked_by"],
                    "signals": snap["signals"],
                },
            }
        )


class MeProfileTaxonomyView(APIView):
    permission_classes = [IsAuthenticated]

    @extend_schema(
        tags=["Me"],
        responses={200: MeProfileTaxonomyResponseSerializer},
    )
    def get(self, request):
        return Response(
            {
                "ok": True,
                "taxonomy": get_profile_taxonomy_payload(get_request_language(request)),
            }
        )
