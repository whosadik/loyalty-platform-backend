from __future__ import annotations

from decimal import Decimal
from unittest.mock import patch

from django.contrib.auth import get_user_model
from django.test import TestCase, override_settings

from catalog.models import Product
from roadmap_app.ml_planner import generate_planner_chain
from roadmap_app.models import RoadmapEvent
from roadmap_app.services import refresh_roadmap
from users_app.models import CustomerProfile


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

    @override_settings(
        ROADMAP_PLANNER_V1_MODE="serve",
        ROADMAP_PLANNER_V1_ENABLED_CATEGORIES=["makeup"],
        ROADMAP_PLANNER_V1_MODEL_PATH="C:/tmp/planner.pkl",
    )
    def test_generate_planner_chain_uses_content_features(self):
        profile = CustomerProfile.objects.get(user=self.user)
        profile.makeup_profile = {
            "finish_pref": ["dewy"],
            "coverage_pref": ["medium"],
            "undertone": "neutral",
            "tone_family": "light",
            "concerns": ["long_wear"],
        }
        profile.save(update_fields=["makeup_profile"])

        primer = Product.objects.create(
            name="Planner Runtime Primer",
            brand="B",
            price=Decimal("9.00"),
            category="makeup",
            product_type="primer",
            attrs={"finish": "dewy"},
            ingredients_inci="dimethicone, silica",
            in_stock=True,
        )
        foundation = Product.objects.get(category="makeup", product_type="foundation")
        foundation.attrs = {
            "finish": "dewy",
            "coverage": "medium",
            "undertone": "neutral",
            "tone_family": "light",
        }
        foundation.concerns = ["long_wear"]
        foundation.ingredients_inci = "water, dimethicone, iron_oxides"
        foundation.save(update_fields=["attrs", "concerns", "ingredients_inci"])

        mascara = Product.objects.get(category="makeup", product_type="mascara")
        mascara.attrs = {"finish": "matte"}
        mascara.ingredients_inci = "water, beeswax"
        mascara.save(update_fields=["attrs", "ingredients_inci"])

        User = get_user_model()
        # Reuse checkout-style history via transactions.
        from transactions.models import Transaction, TransactionItem

        tx = Transaction.objects.create(
            user=self.user,
            total_amount=Decimal("9.00"),
            channel="web",
            idempotency_key="planner-runtime-content-1",
        )
        TransactionItem.objects.create(
            transaction=tx,
            product=primer,
            quantity=1,
            unit_price=Decimal("9.00"),
        )

        class DummyPlannerModel:
            def __init__(self):
                self.columns: list[str] = []
                self.seen_frames: list[object] = []

            def predict(self, X):
                self.columns = list(X.columns)
                self.seen_frames.append(X.copy())
                score = (
                    X["candidate_profile_makeup_finish_match_rate"].astype(float) * 10.0
                    + X["candidate_anchor_shared_inci_rate"].astype(float) * 3.0
                    - X["candidate_is_stop"].astype(float) * 5.0
                )
                return score.to_numpy()

        model = DummyPlannerModel()
        artifact = {
            "task": "roadmap_planner_v1_ranking",
            "model": model,
            "preprocessor": None,
            "model_type": "lightgbm_ranker",
            "model_version": "planner_content_test_v1",
            "selected_feature_set": "full",
            "feature_columns": [
                "category",
                "candidate_type",
                "profile_makeup_finish_pref_primary",
                "anchor_product_type",
                "candidate_profile_makeup_finish_match_rate",
                "candidate_anchor_shared_inci_rate",
                "candidate_is_stop",
            ],
            "categorical_features": [
                "category",
                "candidate_type",
                "profile_makeup_finish_pref_primary",
                "anchor_product_type",
            ],
            "numeric_features": [
                "candidate_profile_makeup_finish_match_rate",
                "candidate_anchor_shared_inci_rate",
                "candidate_is_stop",
            ],
            "candidate_types_by_category": {"makeup": ["foundation", "mascara", "__stop__"]},
            "candidate_popularity_priors": {"makeup": {"foundation": 0.5, "mascara": 0.4, "__stop__": 0.1}},
        }

        with patch("roadmap_app.ml_planner._load_planner_artifact", return_value=artifact):
            result = generate_planner_chain(
                user=self.user,
                category="makeup",
                candidate_types=["foundation", "mascara"],
                purchased_types=[],
                owned_types_ordered=[],
                min_steps=1,
                max_steps=3,
                refresh_caller="refresh_roadmap",
            )

        self.assertEqual(str(result["decision"]), "model_used")
        self.assertGreaterEqual(len(result["chain"]), 1)
        self.assertEqual(str(result["chain"][0]), "foundation")
        self.assertIn("candidate_profile_makeup_finish_match_rate", model.columns)
        self.assertIn("candidate_anchor_shared_inci_rate", model.columns)
        self.assertTrue(bool(model.seen_frames))
        first_seen = model.seen_frames[0]
        foundation_row = first_seen[first_seen["candidate_type"].astype(str) == "foundation"].iloc[0]
        self.assertEqual(str(foundation_row["profile_makeup_finish_pref_primary"]), "dewy")
        self.assertEqual(str(foundation_row["anchor_product_type"]), "primer")
        self.assertGreater(float(foundation_row["candidate_profile_makeup_finish_match_rate"]), 0.0)
