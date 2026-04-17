from __future__ import annotations

import copy as _copy
import json

from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework.views import APIView

from datetime import timedelta

from django.core.serializers.json import DjangoJSONEncoder
from django.db.models import Count
from django.utils import timezone as django_timezone


def _json_safe(value):
    """Round-trip value through DjangoJSONEncoder so that Decimal/datetime etc.
    become primitive types compatible with psycopg3's JSON adapter.
    """
    return json.loads(json.dumps(value, cls=DjangoJSONEncoder))

from backend.api_serializers import ApiErrorSerializer
from backend.request_language import AppLanguage, get_request_language
from catalog.models import Product
from catalog.serializers import ProductSerializer
from drf_spectacular.types import OpenApiTypes
from drf_spectacular.utils import OpenApiResponse, extend_schema
from ml_logic.routine_builder import Profile, build_routine
from ml_logic.routine_validator import validate_routine
from roadmap_app.models import RoadmapEvent, RoadmapStep
from roadmap_app.step_presentation import get_roadmap_step_presentation
from transactions.models import OwnedProduct, Transaction, WishlistItem
from users_app.models import CustomerProfile

from .models import RoutineSnapshot, SavedRoutine
from .serializers import (
    RoutineGenerateRequestSerializer,
    RoutineValidateRequestSerializer,
    SavedRoutineWriteSerializer,
)

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


def _build_ml_context(user, candidate_product_ids: set[int]) -> dict:
    """Gather runtime behavioral signals for the ML ranker.

    Kept lightweight: four cheap aggregate queries scoped to the user and the
    specific candidate set.
    """
    now = django_timezone.now()
    window_90d = now - timedelta(days=90)
    window_30d = now - timedelta(days=30)

    tx_count_90d = Transaction.objects.filter(user=user, created_at__gte=window_90d).count()
    owned_skincare_count = OwnedProduct.objects.filter(
        user=user, is_active=True, product__category="skincare"
    ).count()

    product_signals: dict[int, dict] = {}
    if not candidate_product_ids:
        return {
            "user_tx_count_90d": tx_count_90d,
            "user_owned_skincare_count": owned_skincare_count,
            "product_signals": product_signals,
        }

    for pid in candidate_product_ids:
        product_signals[int(pid)] = {
            "popularity": 0,
            "in_wishlist": False,
            "roadmap_clicks_30d": 0,
            "roadmap_skips_30d": 0,
        }

    popularity_rows = (
        OwnedProduct.objects.filter(is_active=True, product_id__in=candidate_product_ids)
        .values("product_id")
        .annotate(count=Count("id"))
    )
    for row in popularity_rows:
        pid = int(row["product_id"])
        if pid in product_signals:
            product_signals[pid]["popularity"] = int(row["count"])

    wishlist_ids = WishlistItem.objects.filter(
        user=user, product_id__in=candidate_product_ids
    ).values_list("product_id", flat=True)
    for pid in wishlist_ids:
        if int(pid) in product_signals:
            product_signals[int(pid)]["in_wishlist"] = True

    step_to_product = {
        int(row["id"]): int(row["recommended_product_id"])
        for row in RoadmapStep.objects.filter(
            recommended_product_id__in=candidate_product_ids
        ).values("id", "recommended_product_id")
        if row.get("recommended_product_id") is not None
    }
    if step_to_product:
        event_rows = RoadmapEvent.objects.filter(
            created_at__gte=window_30d,
            step_id__in=step_to_product.keys(),
            event_type__in=[
                RoadmapEvent.Type.STEP_CLICKED,
                RoadmapEvent.Type.STEP_SKIPPED,
            ],
        ).values("event_type", "step_id")
        for row in event_rows:
            pid = step_to_product.get(int(row["step_id"]))
            if pid is None or pid not in product_signals:
                continue
            if row["event_type"] == RoadmapEvent.Type.STEP_CLICKED:
                product_signals[pid]["roadmap_clicks_30d"] += 1
            elif row["event_type"] == RoadmapEvent.Type.STEP_SKIPPED:
                product_signals[pid]["roadmap_skips_30d"] += 1

    return {
        "user_tx_count_90d": tx_count_90d,
        "user_owned_skincare_count": owned_skincare_count,
        "product_signals": product_signals,
    }


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

        candidate_product_ids = {int(p["id"]) for p in products if p.get("id") is not None}
        ml_context = _build_ml_context(request.user, candidate_product_ids)

        routine = build_routine(
            profile=profile,
            products=products,
            top_k=3,
            owned_product_ids=owned_ids,
            ml_context=ml_context,
        )
        raw_routine = _json_safe(_copy.deepcopy(routine))
        routine = _enrich_routine_payload(routine, language)

        missing_steps = []
        for item in routine["am"] + routine["pm"]:
            if item["status"] == "missing":
                missing_steps.append(item["step"])

        RoutineSnapshot.objects.create(
            user=request.user,
            missing_steps=missing_steps,
            profile_skin_type=profile.skin_type or "",
            payload=raw_routine,
        )
        SavedRoutine.objects.update_or_create(
            user=request.user,
            defaults={"payload": raw_routine},
        )

        return Response(routine)


def _build_saved_raw_payload(am_items, pm_items, notes) -> dict:
    """Build a raw saved routine payload (unenriched) from {step, product_id} items."""
    product_ids: set[int] = set()
    for group in (am_items, pm_items):
        for entry in group:
            pid = entry.get("product_id")
            if isinstance(pid, int):
                product_ids.add(pid)

    products_by_id: dict[int, dict] = {}
    if product_ids:
        products_by_id = {
            int(p["id"]): p
            for p in Product.objects.filter(id__in=product_ids).values(
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
            if p.get("id") is not None
        }

    def _pack(entries):
        packed = []
        for entry in entries:
            step = entry.get("step") or "routine_step"
            pid = entry.get("product_id")
            product = products_by_id.get(int(pid)) if isinstance(pid, int) else None
            packed.append({
                "step": step,
                "status": "filled" if product else "missing",
                "source": "saved",
                "scorer": "saved",
                "product": product,
                "why": [],
                "suggestions": [],
            })
        return packed

    clean_notes = [n for n in (notes or []) if isinstance(n, str) and n.strip()]

    return {
        "am": _pack(am_items),
        "pm": _pack(pm_items),
        "notes": clean_notes,
    }


class SavedRoutineView(APIView):
    permission_classes = [IsAuthenticated]

    @extend_schema(
        tags=["Routine"],
        responses={200: OpenApiTypes.OBJECT},
    )
    def get(self, request):
        language = get_request_language(request)
        try:
            saved = SavedRoutine.objects.get(user=request.user)
        except SavedRoutine.DoesNotExist:
            return Response({"routine": None, "updated_at": None})

        payload = saved.payload or {}
        if payload:
            payload = _enrich_routine_payload(_copy.deepcopy(payload), language)

        return Response(
            {
                "routine": payload if payload else None,
                "updated_at": saved.updated_at.isoformat() if saved.updated_at else None,
            }
        )

    @extend_schema(
        tags=["Routine"],
        request=SavedRoutineWriteSerializer,
        responses={200: OpenApiTypes.OBJECT},
    )
    def post(self, request):
        language = get_request_language(request)
        req = SavedRoutineWriteSerializer(data=request.data)
        req.is_valid(raise_exception=True)

        raw_payload = _json_safe(
            _build_saved_raw_payload(
                req.validated_data["am"],
                req.validated_data["pm"],
                req.validated_data.get("notes") or [],
            )
        )
        saved, _ = SavedRoutine.objects.update_or_create(
            user=request.user,
            defaults={"payload": raw_payload},
        )
        enriched = _enrich_routine_payload(_copy.deepcopy(raw_payload), language)
        return Response(
            {
                "routine": enriched,
                "updated_at": saved.updated_at.isoformat() if saved.updated_at else None,
            }
        )

    @extend_schema(
        tags=["Routine"],
        responses={204: None},
    )
    def delete(self, request):
        SavedRoutine.objects.filter(user=request.user).delete()
        return Response(status=204)


class RoutineHistoryView(APIView):
    permission_classes = [IsAuthenticated]

    @extend_schema(
        tags=["Routine"],
        responses={200: OpenApiTypes.OBJECT},
    )
    def get(self, request):
        language = get_request_language(request)
        snapshots = list(
            RoutineSnapshot.objects.filter(user=request.user)
            .order_by("-created_at")[:20]
        )

        items = []
        for snap in snapshots:
            payload = snap.payload or {}
            enriched = (
                _enrich_routine_payload(_copy.deepcopy(payload), language)
                if payload
                else None
            )
            items.append(
                {
                    "id": snap.id,
                    "created_at": snap.created_at.isoformat(),
                    "missing_steps": snap.missing_steps,
                    "profile_skin_type": snap.profile_skin_type,
                    "routine": enriched,
                }
            )

        return Response({"items": items})


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
