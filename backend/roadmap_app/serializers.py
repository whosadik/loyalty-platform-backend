from __future__ import annotations

from decimal import Decimal

from rest_framework import serializers

from backend.request_language import AppLanguage, get_context_language
from roadmap_app.models import RoadmapPlan, RoadmapStep
from roadmap_app.runtime_status import roadmap_step_explainability
from roadmap_app.services import build_plan_summary, get_next_missing_step
from roadmap_app.step_presentation import (
    build_roadmap_step_presentation,
    get_roadmap_step_presentation,
)

ROADMAP_CATEGORY_CHOICES = [
    RoadmapPlan.Category.SKINCARE,
    RoadmapPlan.Category.HAIRCARE,
    RoadmapPlan.Category.MAKEUP,
    RoadmapPlan.Category.FRAGRANCE,
]


def _coerce_points_earned(price) -> int:
    try:
        normalized = Decimal(str(price or "0"))
    except Exception:
        normalized = Decimal("0")
    return int(max(0, round(float(normalized) * 0.1)))


def _coerce_int(value) -> int | None:
    try:
        if value in (None, ""):
            return None
        return int(value)
    except Exception:
        return None


def _coerce_string(value) -> str:
    if value is None:
        return ""
    return str(value)


def _serialize_snapshot_product_dict(product: dict | None) -> dict | None:
    if not isinstance(product, dict):
        return None

    return {
        "id": _coerce_int(product.get("id")),
        "name": _coerce_string(product.get("name")),
        "brand": _coerce_string(product.get("brand")),
        "price": None if product.get("price") in (None, "") else str(product.get("price")),
        "currency": product.get("currency"),
        "category": _coerce_string(product.get("category")),
        "product_type": _coerce_string(product.get("product_type")),
        "in_stock": bool(product.get("in_stock")),
        "image_url": product.get("image_url") or None,
        "image_urls": list(product.get("image_urls") or []),
        "points_earned": _coerce_int(product.get("points_earned"))
        if _coerce_int(product.get("points_earned")) is not None
        else _coerce_points_earned(product.get("price")),
    }


def serialize_roadmap_recommended_product(product) -> dict | None:
    if not product:
        return None

    if isinstance(product, dict):
        return _serialize_snapshot_product_dict(product)

    return {
        "id": int(product.id),
        "name": product.name,
        "brand": product.brand,
        "price": str(product.price) if product.price is not None else None,
        "currency": product.currency,
        "category": product.category,
        "product_type": product.product_type,
        "in_stock": bool(product.in_stock),
        "image_url": product.image_url or None,
        "image_urls": list(product.image_urls or []),
        "points_earned": _coerce_points_earned(product.price),
    }


def _serialize_snapshot_from_dict(
    payload: dict,
    *,
    category: str | None = None,
    plan_id: int | None = None,
    plan_meta: dict | None = None,
    language: AppLanguage = "ru",
) -> dict:
    product_type = _coerce_string(payload.get("product_type"))
    step_texts = get_roadmap_step_presentation(product_type, language)
    step_presentation = build_roadmap_step_presentation(product_type, language=language)
    resolved_plan_id = plan_id if plan_id is not None else _coerce_int(payload.get("plan_id"))
    resolved_category = category if category is not None else payload.get("category")
    explainability = roadmap_step_explainability(
        why=list(payload.get("why") or []),
        plan_meta=plan_meta,
    )
    recommended_product = serialize_roadmap_recommended_product(payload.get("recommended_product"))
    is_fragrance = str(resolved_category or "").strip().lower() == "fragrance"
    recommended_actual_product_type = (
        _coerce_string((recommended_product or {}).get("product_type")) or None
    )

    return {
        "id": _coerce_int(payload.get("id") or payload.get("step_id")),
        "step_id": _coerce_int(payload.get("step_id") or payload.get("id")),
        "plan_id": resolved_plan_id,
        "category": str(resolved_category) if resolved_category else None,
        "step_index": int(_coerce_int(payload.get("step_index")) or 0),
        "product_type": product_type,
        "status": _coerce_string(payload.get("status")),
        "title": step_texts["title"],
        "description": step_texts["description"],
        "presentation": step_presentation,
        "why": list(explainability["why"] or []),
        "picked_via": _coerce_string(explainability.get("picked_via")),
        "decision_source": _coerce_string(explainability.get("decision_source")),
        "continuation_reason": explainability.get("continuation_reason"),
        "continuation_markers": list(explainability.get("continuation_markers") or []),
        "fragrance_slot": product_type if is_fragrance and product_type else None,
        "recommended_actual_product_type": recommended_actual_product_type if is_fragrance else None,
        "cadence": _coerce_string(payload.get("cadence")),
        "recommended_product_id": _coerce_int(payload.get("recommended_product_id")),
        "recommended_product": recommended_product,
    }


def serialize_roadmap_step_snapshot(
    step: RoadmapStep | dict | None,
    *,
    category: str | None = None,
    plan_id: int | None = None,
    plan_meta: dict | None = None,
    language: AppLanguage = "ru",
) -> dict | None:
    if not step:
        return None

    if isinstance(step, dict):
        return _serialize_snapshot_from_dict(
            step,
            category=category,
            plan_id=plan_id,
            plan_meta=plan_meta,
            language=language,
        )

    step_texts = get_roadmap_step_presentation(step.product_type, language)
    step_presentation = build_roadmap_step_presentation(step.product_type, language=language)
    resolved_plan_id = plan_id if plan_id is not None else getattr(step, "plan_id", None)
    resolved_category = category
    if resolved_category is None:
        plan = getattr(step, "plan", None)
        resolved_category = getattr(plan, "category", None)
    resolved_plan_meta = plan_meta
    if resolved_plan_meta is None:
        plan = getattr(step, "plan", None)
        resolved_plan_meta = getattr(plan, "meta", None) if plan is not None else None
    explainability = roadmap_step_explainability(
        why=list(step.why or []),
        plan_meta=resolved_plan_meta,
    )
    recommended_product = serialize_roadmap_recommended_product(step.recommended_product)
    is_fragrance = str(resolved_category or "").strip().lower() == "fragrance"
    recommended_actual_product_type = (
        _coerce_string((recommended_product or {}).get("product_type")) or None
    )

    return {
        "id": int(step.id),
        "step_id": int(step.id),
        "plan_id": int(resolved_plan_id) if resolved_plan_id is not None else None,
        "category": str(resolved_category) if resolved_category else None,
        "step_index": int(step.step_index),
        "product_type": _coerce_string(step.product_type),
        "status": _coerce_string(step.status),
        "title": step_texts["title"],
        "description": step_texts["description"],
        "presentation": step_presentation,
        "why": list(explainability["why"] or []),
        "picked_via": _coerce_string(explainability.get("picked_via")),
        "decision_source": _coerce_string(explainability.get("decision_source")),
        "continuation_reason": explainability.get("continuation_reason"),
        "continuation_markers": list(explainability.get("continuation_markers") or []),
        "fragrance_slot": _coerce_string(step.product_type) if is_fragrance else None,
        "recommended_actual_product_type": recommended_actual_product_type if is_fragrance else None,
        "cadence": _coerce_string(step.cadence),
        "recommended_product_id": int(step.recommended_product_id) if step.recommended_product_id else None,
        "recommended_product": recommended_product,
    }


class RoadmapQuerySerializer(serializers.Serializer):
    category = serializers.ChoiceField(choices=ROADMAP_CATEGORY_CHOICES, required=False)


class RoadmapRefreshRequestSerializer(serializers.Serializer):
    category = serializers.ChoiceField(choices=ROADMAP_CATEGORY_CHOICES, required=False)


class RoadmapStepPatchRequestSerializer(serializers.Serializer):
    status = serializers.ChoiceField(choices=RoadmapStep.Status.choices)


class RoadmapRecommendedProductSerializer(serializers.Serializer):
    id = serializers.IntegerField()
    name = serializers.CharField()
    brand = serializers.CharField(allow_blank=True, required=False)
    price = serializers.CharField(allow_null=True, required=False)
    currency = serializers.CharField(allow_null=True, required=False)
    category = serializers.CharField(allow_blank=True, required=False)
    product_type = serializers.CharField(allow_blank=True, required=False)
    in_stock = serializers.BooleanField()
    image_url = serializers.CharField(allow_null=True, required=False)
    image_urls = serializers.ListField(child=serializers.CharField(), required=False)
    points_earned = serializers.IntegerField(required=False)


class RoadmapStepPresentationSerializer(serializers.Serializer):
    title = serializers.CharField()
    description = serializers.CharField()
    points = serializers.IntegerField()
    why = serializers.CharField()
    improves = serializers.CharField()
    benefit = serializers.CharField()


class RoadmapStepSnapshotSerializer(serializers.Serializer):
    id = serializers.IntegerField(required=False)
    step_id = serializers.IntegerField(required=False)
    plan_id = serializers.IntegerField(required=False, allow_null=True)
    category = serializers.CharField(required=False, allow_null=True)
    step_index = serializers.IntegerField()
    product_type = serializers.CharField()
    status = serializers.CharField()
    title = serializers.CharField()
    description = serializers.CharField()
    presentation = RoadmapStepPresentationSerializer(required=False)
    why = serializers.ListField(child=serializers.CharField(), required=False)
    picked_via = serializers.CharField(required=False, allow_blank=True)
    decision_source = serializers.CharField(required=False, allow_blank=True)
    continuation_reason = serializers.CharField(required=False, allow_null=True, allow_blank=True)
    continuation_markers = serializers.ListField(child=serializers.CharField(), required=False)
    fragrance_slot = serializers.CharField(required=False, allow_null=True, allow_blank=True)
    recommended_actual_product_type = serializers.CharField(required=False, allow_null=True, allow_blank=True)
    cadence = serializers.CharField(required=False, allow_blank=True)
    recommended_product_id = serializers.IntegerField(required=False, allow_null=True)
    recommended_product = RoadmapRecommendedProductSerializer(required=False, allow_null=True)

    def to_representation(self, obj):
        if isinstance(obj, dict):
            return serialize_roadmap_step_snapshot(
                obj,
                category=self.context.get("category"),
                plan_id=self.context.get("plan_id"),
                plan_meta=self.context.get("plan_meta"),
                language=get_context_language(self.context),
            )

        return serialize_roadmap_step_snapshot(
            obj,
            category=self.context.get("category"),
            plan_id=self.context.get("plan_id"),
            plan_meta=self.context.get("plan_meta"),
            language=get_context_language(self.context),
        )


class RoadmapStepReadSerializer(RoadmapStepSnapshotSerializer):
    suggestions = serializers.ListField(child=serializers.IntegerField(), required=False)
    score = serializers.FloatField(required=False, allow_null=True)
    confidence = serializers.FloatField(required=False, allow_null=True)

    def to_representation(self, obj: RoadmapStep):
        data = serialize_roadmap_step_snapshot(
            obj,
            category=self.context.get("category"),
            plan_id=self.context.get("plan_id"),
            plan_meta=self.context.get("plan_meta"),
            language=get_context_language(self.context),
        )
        data["suggestions"] = list(obj.suggestions or [])
        data["score"] = obj.score
        data["confidence"] = obj.confidence
        return data


class RoadmapPlanReadSerializer(serializers.ModelSerializer):
    steps = serializers.SerializerMethodField()
    summary = serializers.SerializerMethodField()

    class Meta:
        model = RoadmapPlan
        fields = [
            "id",
            "category",
            "is_active",
            "version",
            "meta",
            "created_at",
            "updated_at",
            "steps",
            "summary",
        ]

    def get_steps(self, obj: RoadmapPlan):
        rows = obj.steps.select_related("recommended_product").order_by("step_index")
        language = get_context_language(self.context)
        return RoadmapStepReadSerializer(
            rows,
            many=True,
            context={
                "request": self.context.get("request"),
                "language": language,
                "category": obj.category,
                "plan_id": obj.id,
                "plan_meta": obj.meta,
            },
        ).data

    def get_summary(self, obj: RoadmapPlan):
        summary = build_plan_summary(obj)
        next_step = get_next_missing_step(obj)
        summary["next_step"] = serialize_roadmap_step_snapshot(
            next_step,
            category=obj.category,
            plan_id=obj.id,
            plan_meta=obj.meta,
            language=get_context_language(self.context),
        )
        return summary
