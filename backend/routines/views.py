from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated
from drf_spectacular.types import OpenApiTypes
from drf_spectacular.utils import OpenApiResponse, extend_schema

from backend.api_serializers import ApiErrorSerializer
from catalog.models import Product
from catalog.serializers import ProductSerializer
from users_app.models import CustomerProfile
from .serializers import RoutineGenerateRequestSerializer, RoutineValidateRequestSerializer

from ml_logic.routine_builder import Profile, build_routine
from ml_logic.routine_validator import validate_routine
from .models import RoutineSnapshot

from transactions.models import OwnedProduct


ROUTINE_STEP_META = {
    "cleanser": {
        "display_step": "Очищение",
        "duration_label": "1-2 мин",
    },
    "toner": {
        "display_step": "Тонизирование",
        "duration_label": "30 сек",
    },
    "serum": {
        "display_step": "Сыворотка",
        "duration_label": "1 мин",
    },
    "moisturizer": {
        "display_step": "Увлажнение",
        "duration_label": "1 мин",
    },
    "spf": {
        "display_step": "SPF защита",
        "duration_label": "1 мин",
        "note": "Наносите каждый день как завершающий утренний шаг.",
    },
}


def _format_step_label(step: str) -> str:
    prepared = str(step or "").replace("_", " ").strip()
    if not prepared:
        return "Шаг ухода"

    return prepared[:1].upper() + prepared[1:]


def _normalize_product_id(value):
    if isinstance(value, bool):
        return None

    if isinstance(value, int):
        return value

    if isinstance(value, str) and value.strip():
        try:
            return int(value)
        except ValueError:
            return None

    return None


def _serialize_products_by_id(product_ids: set[int]) -> dict[int, dict]:
    if not product_ids:
        return {}

    products = Product.objects.filter(id__in=product_ids)
    return {
        int(product["id"]): product
        for product in ProductSerializer(products, many=True).data
        if product.get("id") is not None
    }


def _enrich_routine_payload(routine: dict) -> dict:
    items = []
    product_ids = set()

    for bucket in ("am", "pm"):
        bucket_items = routine.get(bucket)
        if not isinstance(bucket_items, list):
            continue

        for item in bucket_items:
            if not isinstance(item, dict):
                continue

            items.append(item)
            product = item.get("product")
            if not isinstance(product, dict):
                continue

            product_id = _normalize_product_id(product.get("id"))
            if product_id is not None:
                product_ids.add(product_id)

    serialized_products = _serialize_products_by_id(product_ids)

    for item in items:
        step_key = str(item.get("step") or "")
        step_meta = ROUTINE_STEP_META.get(step_key, {})
        item["display_step"] = step_meta.get("display_step") or _format_step_label(step_key)

        duration_label = step_meta.get("duration_label")
        if duration_label:
            item["duration_label"] = duration_label

        note = step_meta.get("note")
        if note:
            item["note"] = note

        product = item.get("product")
        if not isinstance(product, dict):
            continue

        product_id = _normalize_product_id(product.get("id"))
        if product_id is None:
            continue

        serialized_product = serialized_products.get(product_id)
        if serialized_product is not None:
            item["product"] = serialized_product

    return routine


def _enrich_routine_validation_payload(result: dict) -> dict:
    suggestions = result.get("suggestions")
    if not isinstance(suggestions, list):
        return result

    product_ids = set()
    suggestion_items = []

    for item in suggestions:
        if not isinstance(item, dict):
            continue

        suggestion_items.append(item)

        current_product_id = _normalize_product_id(item.get("current_product_id"))
        if current_product_id is not None:
            product_ids.add(current_product_id)

        alternatives = item.get("alternatives")
        if not isinstance(alternatives, list):
            continue

        for value in alternatives:
            alternative_id = _normalize_product_id(value)
            if alternative_id is not None:
                product_ids.add(alternative_id)

    serialized_products = _serialize_products_by_id(product_ids)

    for item in suggestion_items:
        step_key = str(item.get("step") or "")
        step_meta = ROUTINE_STEP_META.get(step_key, {})
        item["display_step"] = step_meta.get("display_step") or _format_step_label(step_key)

        current_product_id = _normalize_product_id(item.get("current_product_id"))
        if current_product_id is not None:
            item["current_product"] = serialized_products.get(current_product_id)

        alternative_products = []
        alternatives = item.get("alternatives")
        if isinstance(alternatives, list):
            for value in alternatives:
                alternative_id = _normalize_product_id(value)
                if alternative_id is None:
                    continue

                serialized_product = serialized_products.get(alternative_id)
                if serialized_product is not None:
                    alternative_products.append(serialized_product)

        item["alternative_products"] = alternative_products

    return result

class RoutineGenerateView(APIView):
    permission_classes = [IsAuthenticated]

    @extend_schema(
        tags=["Routine"],
        request=RoutineGenerateRequestSerializer,
        responses={
            200: OpenApiTypes.OBJECT,
            400: OpenApiResponse(response=ApiErrorSerializer),
        },
    )
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
            Product.objects.filter(category="skincare").values(
                "id",
                "name",
                "brand",
                "price",
                "category",
                "product_type",
                "actives",
                "flags",
                "supported_skin_types",
                "strength",
                "in_stock",
                "concerns",
                "attrs",
                "step",
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
        routine = _enrich_routine_payload(routine)

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

    @extend_schema(
        tags=["Routine"],
        request=RoutineValidateRequestSerializer,
        responses={
            200: OpenApiTypes.OBJECT,
            400: OpenApiResponse(response=ApiErrorSerializer),
        },
    )
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
            Product.objects.filter(category="skincare").values(
                "id",
                "name",
                "brand",
                "price",
                "category",
                "product_type",
                "actives",
                "flags",
                "supported_skin_types",
                "strength",
                "in_stock",
                "concerns",
                "attrs",
                "step",
            )
        )

        result = validate_routine(
            profile=profile,
            products=products,
            routine={"am": req.validated_data["am"], "pm": req.validated_data["pm"]},
            top_k=3,
        )
        result = _enrich_routine_validation_payload(result)
        return Response(result)
