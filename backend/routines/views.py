from __future__ import annotations

from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework.views import APIView

from backend.api_serializers import ApiErrorSerializer
from backend.request_language import AppLanguage, get_request_language
from catalog.models import Product
from catalog.serializers import ProductSerializer
from drf_spectacular.types import OpenApiTypes
from drf_spectacular.utils import OpenApiResponse, extend_schema
from ml_logic.routine_builder import Profile, build_routine
from ml_logic.routine_validator import validate_routine
from roadmap_app.step_presentation import get_roadmap_step_presentation
from transactions.models import OwnedProduct
from users_app.models import CustomerProfile

from .models import RoutineSnapshot
from .serializers import RoutineGenerateRequestSerializer, RoutineValidateRequestSerializer

ROUTINE_STEP_META: dict[str, dict[AppLanguage, dict[str, str]]] = {
    "cleanser": {
        "ru": {"duration_label": "1-2 мин"},
        "kk": {"duration_label": "1-2 мин"},
        "en": {"duration_label": "1-2 min"},
    },
    "toner": {
        "ru": {"duration_label": "30 сек"},
        "kk": {"duration_label": "30 сек"},
        "en": {"duration_label": "30 sec"},
    },
    "serum": {
        "ru": {"duration_label": "1 мин"},
        "kk": {"duration_label": "1 мин"},
        "en": {"duration_label": "1 min"},
    },
    "moisturizer": {
        "ru": {"duration_label": "1 мин"},
        "kk": {"duration_label": "1 мин"},
        "en": {"duration_label": "1 min"},
    },
    "spf": {
        "ru": {
            "duration_label": "1 мин",
            "note": "Наносите каждый день как завершающий утренний шаг.",
        },
        "kk": {
            "duration_label": "1 мин",
            "note": "Күн сайын таңертеңгі соңғы қадам ретінде жағыңыз.",
        },
        "en": {
            "duration_label": "1 min",
            "note": "Apply every day as the final morning step.",
        },
    },
}

ROUTINE_FALLBACK_TITLE: dict[AppLanguage, str] = {
    "ru": "Шаг ухода",
    "kk": "Күтім қадамы",
    "en": "Care step",
}

ROUTINE_NOTE_TRANSLATIONS: dict[str, dict[AppLanguage, str]] = {
    "Consider SPF in the morning when using active ingredients.": {
        "ru": "Используйте SPF утром, если в рутине есть активные ингредиенты.",
        "kk": "Егер рутинада белсенді ингредиенттер болса, таңертең SPF қолданыңыз.",
        "en": "Use SPF in the morning when your routine includes active ingredients.",
    }
}


def _format_step_label(step: str, language: AppLanguage) -> str:
    normalized = str(step or "").strip()
    if not normalized:
        return ROUTINE_FALLBACK_TITLE[language]

    presentation = get_roadmap_step_presentation(normalized, language)
    if presentation["title"]:
        return presentation["title"]

    prepared = normalized.replace("_", " ").strip()
    if not prepared:
        return ROUTINE_FALLBACK_TITLE[language]
    if language == "en":
        return prepared.title()
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


def _localize_routine_notes(notes: list, language: AppLanguage) -> list[str]:
    localized_notes: list[str] = []
    for value in notes:
        if not isinstance(value, str) or not value.strip():
            continue
        localized_notes.append(ROUTINE_NOTE_TRANSLATIONS.get(value, {}).get(language, value.strip()))
    return localized_notes


def _enrich_routine_payload(routine: dict, language: AppLanguage) -> dict:
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
        step_meta = ROUTINE_STEP_META.get(step_key, {}).get(language, {})
        item["display_step"] = _format_step_label(step_key, language)

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

    routine["notes"] = _localize_routine_notes(list(routine.get("notes") or []), language)

    return routine


def _enrich_routine_validation_payload(result: dict, language: AppLanguage) -> dict:
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
        item["display_step"] = _format_step_label(step_key, language)

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
        language = get_request_language(request)
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
        routine = _enrich_routine_payload(routine, language)

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
        language = get_request_language(request)
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
        result = _enrich_routine_validation_payload(result, language)
        return Response(result)
