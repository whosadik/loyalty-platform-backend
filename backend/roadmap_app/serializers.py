from decimal import Decimal

from rest_framework import serializers

from roadmap_app.models import RoadmapPlan, RoadmapStep
from roadmap_app.services import build_plan_summary, get_next_missing_step


ROADMAP_CATEGORY_CHOICES = [
    RoadmapPlan.Category.SKINCARE,
    RoadmapPlan.Category.HAIRCARE,
    RoadmapPlan.Category.MAKEUP,
    RoadmapPlan.Category.FRAGRANCE,
]


ROADMAP_STEP_PRESENTATION_BY_TYPE = {
    "cleanser": {
        "title": "Очищение",
        "description": "Начните с мягкого очищающего средства для вашего типа кожи.",
    },
    "toner": {
        "title": "Тонизирование",
        "description": "Восстановите баланс кожи с помощью подходящего тоника.",
    },
    "serum": {
        "title": "Сыворотка",
        "description": "Добавьте активный этап для решения конкретной задачи кожи.",
    },
    "moisturizer": {
        "title": "Увлажнение",
        "description": "Закрепите уход увлажняющим средством для поддержания барьера кожи.",
    },
    "spf": {
        "title": "Защита SPF",
        "description": "Завершите дневной уход средством с солнцезащитой.",
    },
    "shampoo": {
        "title": "Очищение кожи головы",
        "description": "Выберите шампунь по типу кожи головы и частоте мытья.",
    },
    "conditioner": {
        "title": "Кондиционирование",
        "description": "Используйте кондиционер для защиты длины и блеска волос.",
    },
    "hair_mask": {
        "title": "Маска для волос",
        "description": "Добавьте еженедельный восстановительный этап ухода.",
    },
    "hair_oil": {
        "title": "Масло для волос",
        "description": "Используйте масло для защиты и гладкости длины.",
    },
    "scalp_serum": {
        "title": "Сыворотка для кожи головы",
        "description": "Добавьте целевой уход для кожи головы и корней.",
    },
    "foundation": {
        "title": "Тон",
        "description": "Подберите основу, подходящую по тону и типу кожи.",
    },
    "eyeshadow": {
        "title": "Акцент для глаз",
        "description": "Добавьте продукт для акцента и завершения макияжа.",
    },
    "lipstick": {
        "title": "Акцент для губ",
        "description": "Завершите образ подходящим оттенком для губ.",
    },
    "perfume": {
        "title": "Парфюмерная база",
        "description": "Подберите аромат, который соответствует вашим предпочтениям.",
    },
}


def _coerce_points_earned(price) -> int:
    try:
        normalized = Decimal(str(price or "0"))
    except Exception:
        normalized = Decimal("0")
    return int(max(0, round(float(normalized) * 0.1)))


def get_roadmap_step_presentation(product_type: str | None) -> dict[str, str]:
    normalized = str(product_type or "").strip()
    if normalized in ROADMAP_STEP_PRESENTATION_BY_TYPE:
        return ROADMAP_STEP_PRESENTATION_BY_TYPE[normalized]

    label = normalized.replace("_", " ").strip() or "Шаг ухода"
    title = label[:1].upper() + label[1:] if label else "Шаг ухода"
    return {
        "title": title,
        "description": "Персональный шаг, добавленный в ваш roadmap.",
    }


def serialize_roadmap_recommended_product(product) -> dict | None:
    if not product:
        return None
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


def serialize_roadmap_step_snapshot(
    step: RoadmapStep | None,
    *,
    category: str | None = None,
    plan_id: int | None = None,
) -> dict | None:
    if not step:
        return None

    presentation = get_roadmap_step_presentation(step.product_type)
    resolved_plan_id = plan_id if plan_id is not None else getattr(step, "plan_id", None)
    resolved_category = category
    if resolved_category is None:
        plan = getattr(step, "plan", None)
        resolved_category = getattr(plan, "category", None)

    return {
        "id": int(step.id),
        "step_id": int(step.id),
        "plan_id": int(resolved_plan_id) if resolved_plan_id is not None else None,
        "category": str(resolved_category) if resolved_category else None,
        "step_index": int(step.step_index),
        "product_type": str(step.product_type or ""),
        "status": str(step.status or ""),
        "title": presentation["title"],
        "description": presentation["description"],
        "why": list(step.why or []),
        "cadence": str(step.cadence or ""),
        "recommended_product_id": int(step.recommended_product_id) if step.recommended_product_id else None,
        "recommended_product": serialize_roadmap_recommended_product(step.recommended_product),
    }


class RoadmapQuerySerializer(serializers.Serializer):
    category = serializers.ChoiceField(choices=ROADMAP_CATEGORY_CHOICES, required=False)


class RoadmapRefreshRequestSerializer(serializers.Serializer):
    category = serializers.ChoiceField(choices=ROADMAP_CATEGORY_CHOICES)


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
    why = serializers.ListField(child=serializers.CharField(), required=False)
    cadence = serializers.CharField(required=False, allow_blank=True)
    recommended_product_id = serializers.IntegerField(required=False, allow_null=True)
    recommended_product = RoadmapRecommendedProductSerializer(required=False, allow_null=True)

    def to_representation(self, obj):
        if isinstance(obj, dict):
            return obj
        category = self.context.get("category")
        plan_id = self.context.get("plan_id")
        return serialize_roadmap_step_snapshot(obj, category=category, plan_id=plan_id)


class RoadmapStepReadSerializer(RoadmapStepSnapshotSerializer):
    suggestions = serializers.ListField(child=serializers.IntegerField(), required=False)
    score = serializers.FloatField(required=False, allow_null=True)
    confidence = serializers.FloatField(required=False, allow_null=True)

    def to_representation(self, obj: RoadmapStep):
        data = serialize_roadmap_step_snapshot(
            obj,
            category=self.context.get("category"),
            plan_id=self.context.get("plan_id"),
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
        return RoadmapStepReadSerializer(
            rows,
            many=True,
            context={"category": obj.category, "plan_id": obj.id},
        ).data

    def get_summary(self, obj: RoadmapPlan):
        summary = build_plan_summary(obj)
        next_step = get_next_missing_step(obj)
        summary["next_step"] = serialize_roadmap_step_snapshot(
            next_step,
            category=obj.category,
            plan_id=obj.id,
        )
        return summary
