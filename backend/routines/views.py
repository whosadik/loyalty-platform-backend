from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated

from catalog.models import Product
from users_app.models import CustomerProfile
from .serializers import RoutineGenerateRequestSerializer, RoutineValidateRequestSerializer

from ml_logic.routine_builder import Profile, build_routine
from ml_logic.routine_validator import validate_routine
from .models import RoutineSnapshot

from transactions.models import OwnedProduct

class RoutineGenerateView(APIView):
    permission_classes = [IsAuthenticated]

    def post(self, request):
        req = RoutineGenerateRequestSerializer(data=request.data)
        req.is_valid(raise_exception=True)

        profile_obj, _ = CustomerProfile.objects.get_or_create(user=request.user)

        profile = Profile(
            skin_type=profile_obj.skin_type,
            goals=profile_obj.goals or [],
            avoid_flags=profile_obj.avoid_flags or [],
            budget=profile_obj.budget,
        )

        products = list(
            Product.objects.all().values(
                "id",
                "name",
                "brand",
                "price",
                "step",
                "actives",
                "flags",
                "supported_skin_types",
                "strength",
                "in_stock",
            )
        )
        owned_ids = list(
            OwnedProduct.objects.filter(user=request.user, is_active=True).values_list("product_id", flat=True)
        )

        routine = build_routine(
            profile=profile,
            products=products,
            top_k=3,
            owned_product_ids=owned_ids,
        )

        missing_steps = []
        for item in routine["am"] + routine["pm"]:
            if item["status"] == "missing":
                missing_steps.append(item["step"])

        RoutineSnapshot.objects.create(
            user=request.user,
            missing_steps=missing_steps,
            profile_skin_type=profile.skin_type or "",
        )

        return Response(routine)


class RoutineValidateView(APIView):
    permission_classes = [IsAuthenticated]

    def post(self, request):
        req = RoutineValidateRequestSerializer(data=request.data)
        req.is_valid(raise_exception=True)

        profile_obj, _ = CustomerProfile.objects.get_or_create(user=request.user)
        profile = Profile(
            skin_type=profile_obj.skin_type,
            goals=profile_obj.goals or [],
            avoid_flags=profile_obj.avoid_flags or [],
            budget=profile_obj.budget,
        )

        products = list(
            Product.objects.all().values(
                "id",
                "name",
                "brand",
                "price",
                "step",
                "actives",
                "flags",
                "supported_skin_types",
                "strength",
                "in_stock",
            )
        )

        result = validate_routine(
            profile=profile,
            products=products,
            routine={"am": req.validated_data["am"], "pm": req.validated_data["pm"]},
            top_k=3,
        )
        return Response(result)
