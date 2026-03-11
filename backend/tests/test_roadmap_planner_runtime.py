from __future__ import annotations

from decimal import Decimal
from unittest.mock import patch

from django.contrib.auth import get_user_model
from django.test import TestCase, override_settings

from catalog.models import Product
from roadmap_app.models import RoadmapEvent
from roadmap_app.services import refresh_roadmap


def _planner_result(chain: list[str]) -> dict[str, object]:
    return {
        "category": "makeup",
        "decision": "model_used",
        "fallback_reason": None,
        "disabled_reason": None,
        "chain": list(chain),
        "source_by_type": {
            item: {"source": "ml_planner", "score": round(1.0 - idx * 0.1, 4)}
            for idx, item in enumerate(chain)
        },
        "trace": [
            {"selected": item, "score": round(1.0 - idx * 0.1, 4)}
            for idx, item in enumerate(chain)
        ],
        "model_path": "C:/tmp/planner.pkl",
        "model_version": "planner_test_v1",
        "selected_feature_set": "baseline_only",
    }


class RoadmapPlannerRuntimeTests(TestCase):
    def setUp(self):
        User = get_user_model()
        self.user = User.objects.create_user(username="planner_runtime_u1", password="pass12345")

        for product_type in ["foundation", "mascara", "blush", "lipstick", "eyeshadow", "primer", "setting_spray"]:
            Product.objects.create(
                name=f"Planner Runtime {product_type}",
                brand="B",
                price=Decimal("10.00"),
                category="makeup",
                product_type=product_type,
                in_stock=True,
            )

    @override_settings(
        ROADMAP_NEXTSTEP_V4_ENABLED=False,
        ROADMAP_NEXTSTEP_V3_ENABLED=False,
        ROADMAP_PLANNER_V1_MODE="shadow",
    )
    def test_shadow_mode_keeps_rules_plan_and_stores_planner_shadow_meta(self):
        with patch(
            "roadmap_app.services.generate_planner_chain",
            return_value=_planner_result(["mascara", "foundation", "blush"]),
        ):
            plan = refresh_roadmap(self.user, category="makeup", post_ctx=None)

        steps = list(plan.steps.order_by("step_index"))
        self.assertGreaterEqual(len(steps), 3)
        self.assertEqual(steps[0].product_type, "foundation")
        self.assertEqual(str(plan.meta.get("source")), "roadmap_v1")
        self.assertEqual(str((plan.meta.get("planner") or {}).get("mode")), "shadow")
        self.assertFalse(bool((plan.meta.get("planner") or {}).get("served")))
        self.assertEqual(str((plan.meta.get("planner") or {}).get("decision")), "model_used")
        self.assertEqual((plan.meta.get("planner") or {}).get("chain")[:3], ["mascara", "foundation", "blush"])

    @override_settings(
        ROADMAP_NEXTSTEP_V4_ENABLED=False,
        ROADMAP_NEXTSTEP_V3_ENABLED=False,
        ROADMAP_PLANNER_V1_MODE="serve",
    )
    def test_serve_mode_uses_planner_generated_chain(self):
        with patch(
            "roadmap_app.services.generate_planner_chain",
            return_value=_planner_result(["mascara", "foundation", "blush"]),
        ):
            plan = refresh_roadmap(self.user, category="makeup", post_ctx=None)

        steps = list(plan.steps.order_by("step_index"))
        self.assertGreaterEqual(len(steps), 3)
        self.assertEqual(steps[0].product_type, "mascara")
        self.assertEqual(steps[1].product_type, "foundation")
        self.assertEqual(str(plan.meta.get("source")), "roadmap_planner_v1")
        self.assertTrue(bool((plan.meta.get("planner") or {}).get("served")))
        self.assertEqual(str((plan.meta.get("planner") or {}).get("model_version")), "planner_test_v1")
        self.assertEqual(str(((steps[0].why or [None])[0] or "")), "picked via ML planner")

        plan_event = RoadmapEvent.objects.filter(
            user=self.user,
            plan=plan,
            event_type=RoadmapEvent.Type.PLAN_REFRESHED,
        ).order_by("-id").first()
        self.assertIsNotNone(plan_event)
        self.assertEqual(str((plan_event.context or {}).get("source")), "roadmap_planner_v1")
        self.assertTrue(bool(((plan_event.context or {}).get("planner") or {}).get("served")))

        generated = RoadmapEvent.objects.filter(
            user=self.user,
            plan=plan,
            event_type=RoadmapEvent.Type.STEP_GENERATED,
            step__step_index=1,
        ).order_by("-id").first()
        self.assertIsNotNone(generated)
        self.assertEqual(str((generated.context or {}).get("plan_source")), "roadmap_planner_v1")
        self.assertEqual(str((generated.context or {}).get("source")), "planner")
