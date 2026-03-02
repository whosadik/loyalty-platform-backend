from rest_framework import serializers

from roadmap_app.models import RoadmapPlan, RoadmapStep
from roadmap_app.services import build_plan_summary


ROADMAP_CATEGORY_CHOICES = [
    RoadmapPlan.Category.SKINCARE,
    RoadmapPlan.Category.HAIRCARE,
    RoadmapPlan.Category.MAKEUP,
    RoadmapPlan.Category.FRAGRANCE,
]


class RoadmapQuerySerializer(serializers.Serializer):
    category = serializers.ChoiceField(choices=ROADMAP_CATEGORY_CHOICES, required=False)


class RoadmapRefreshRequestSerializer(serializers.Serializer):
    category = serializers.ChoiceField(choices=ROADMAP_CATEGORY_CHOICES)


class RoadmapStepPatchRequestSerializer(serializers.Serializer):
    status = serializers.ChoiceField(choices=RoadmapStep.Status.choices)


class RoadmapStepReadSerializer(serializers.ModelSerializer):
    recommended_product = serializers.SerializerMethodField()

    class Meta:
        model = RoadmapStep
        fields = [
            "id",
            "step_index",
            "product_type",
            "status",
            "recommended_product",
            "suggestions",
            "score",
            "confidence",
            "why",
            "cadence",
        ]

    def get_recommended_product(self, obj: RoadmapStep):
        p = obj.recommended_product
        if not p:
            return None
        return {
            "id": p.id,
            "name": p.name,
            "brand": p.brand,
            "price": str(p.price) if p.price is not None else None,
            "category": p.category,
            "product_type": p.product_type,
            "in_stock": bool(p.in_stock),
        }


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
        return RoadmapStepReadSerializer(rows, many=True).data

    def get_summary(self, obj: RoadmapPlan):
        return build_plan_summary(obj)
