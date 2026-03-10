from __future__ import annotations

from io import StringIO

from django.contrib.auth import get_user_model
from django.core.management import call_command
from django.test import TestCase
from django.test.utils import override_settings

from catalog.models import Product
from roadmap_app.models import RoadmapPlan
from roadmap_app.services import refresh_roadmap, update_roadmap_from_purchase


class ReportRoadmapQualityMlRuntimeTests(TestCase):
    def _create_skincare_products(self, suffix: str = "") -> None:
        Product.objects.create(
            name=f"RQ Cleanser {suffix}",
            brand="B",
            category="skincare",
            product_type="cleanser",
            in_stock=True,
        )
        Product.objects.create(
            name=f"RQ Serum {suffix}",
            brand="B",
            category="skincare",
            product_type="serum",
            in_stock=True,
        )
        Product.objects.create(
            name=f"RQ Moisturizer {suffix}",
            brand="B",
            category="skincare",
            product_type="moisturizer",
            in_stock=True,
        )
        Product.objects.create(
            name=f"RQ SPF {suffix}",
            brand="B",
            category="skincare",
            product_type="spf",
            in_stock=True,
        )

    def _create_plan(self, *, username: str, meta: dict, category: str = "skincare") -> RoadmapPlan:
        User = get_user_model()
        user = User.objects.create_user(username=username, password="pass12345")
        return RoadmapPlan.objects.create(
            user=user,
            category=category,
            is_active=True,
            meta=meta,
        )

    def test_report_splits_ml_decisions_and_keeps_missing_out_of_fallback(self):
        self._create_plan(
            username="rq_model_used",
            meta={
                "source": "roadmap_v1",
                "ml": {
                    "decision": "model_used",
                    "mode": "v4_ranking",
                    "model_path": "/tmp/model.pkl",
                    "fallback_reason": "low_confidence",
                },
            },
        )
        self._create_plan(
            username="rq_fallback",
            meta={
                "source": "roadmap_v1",
                "ml": {
                    "decision": "fallback",
                    "mode": "v4_ranking",
                    "model_path": "/tmp/model.pkl",
                    "fallback_reason": "low_confidence",
                },
            },
        )
        self._create_plan(
            username="rq_disabled",
            meta={
                "source": "roadmap_v1",
                "ml": {
                    "decision": "disabled",
                    "mode": "legacy",
                    "model_path": "/tmp/model.pkl",
                    "disabled_reason": "ml_disabled",
                },
            },
        )
        self._create_plan(
            username="rq_missing",
            meta={
                "source": "roadmap_v1",
                "ml": {
                    "mode": "legacy",
                    "model_path": "/tmp/model.pkl",
                },
            },
        )

        out = StringIO()
        call_command("report_roadmap_quality", days=7, include_ga=True, stdout=out)
        text = out.getvalue()

        self.assertIn("model_used=1/4", text)
        self.assertIn("fallback=1/4", text)
        self.assertIn("disabled=1/4", text)
        self.assertIn("missing_ml_meta=1/4", text)

        self.assertIn("| model_used | 1 |", text)
        self.assertIn("| fallback | 1 |", text)
        self.assertIn("| disabled | 1 |", text)
        self.assertIn("| missing_ml_meta | 1 |", text)

        start = text.find("### ML fallback reasons")
        end = text.find("### ML fallback reasons (raw)", start)
        self.assertGreaterEqual(start, 0)
        self.assertGreater(end, start)
        fallback_block = text[start:end]
        self.assertIn("low_confidence", fallback_block)
        self.assertNotIn("category_disabled", fallback_block)

        start_disabled = text.find("### ML disabled reasons")
        end_disabled = text.find("### ML disabled reasons (raw)", start_disabled)
        self.assertGreaterEqual(start_disabled, 0)
        self.assertGreater(end_disabled, start_disabled)
        disabled_block = text[start_disabled:end_disabled]
        self.assertIn("ml_disabled", disabled_block)
        self.assertNotIn("low_confidence", disabled_block)

    def test_report_groups_category_guard_and_category_disabled_reasons(self):
        self._create_plan(
            username="rq_guard_failed",
            meta={
                "source": "roadmap_v1",
                "ml": {
                    "decision": "fallback",
                    "mode": "v4_ranking",
                    "model_path": "/tmp/model.pkl",
                    "fallback_reason": "category_guard_failed",
                    "category_guard": {
                        "final_status": "HOLD",
                        "reason": "7d_unstable",
                        "hold_reason": "severe_negative_offer_ctr_lift",
                        "stability_gate_failures": ["7d:severe_negative_offer_ctr_lift"],
                    },
                },
            },
        )
        self._create_plan(
            username="rq_category_disabled",
            category="fragrance",
            meta={
                "source": "roadmap_v1",
                "category": "fragrance",
                "ml": {
                    "decision": "disabled",
                    "mode": "v4_ranking",
                    "model_path": "/tmp/model.pkl",
                    "disabled_reason": "category_disabled",
                    "category_guard": {
                        "final_status": "DISABLE",
                        "reason": "category_disabled",
                        "stability_gate_failures": [],
                    },
                },
            },
        )

        out = StringIO()
        call_command("report_roadmap_quality", days=7, include_ga=True, stdout=out)
        text = out.getvalue()

        self.assertIn("| category_guard_failed | 1 |", text)
        self.assertIn("| category_disabled | 1 |", text)
        self.assertIn("### ML runtime decisions by category", text)
        self.assertIn("| fragrance | 0 | 0 | 1 | 0 |", text)
        self.assertIn("### Category guard reasons", text)
        self.assertIn("### ML per-category detail", text)
        self.assertIn("### Rollout status by category", text)
        self.assertIn("### HOLD reasons by category", text)
        self.assertIn("### Stability gate failures by category", text)

    def test_report_counts_partial_rollout_modes_and_reasons(self):
        self._create_plan(
            username="rq_partial_selected",
            category="makeup",
            meta={
                "source": "roadmap_v1",
                "category": "makeup",
                "ml": {
                    "decision": "model_used",
                    "mode": "v4_ranking",
                    "model_path": "/tmp/model.pkl",
                    "rollout_mode": "partial",
                    "rollout_selected": True,
                    "rollout_reason": "selected",
                    "rollout_bucket": 17,
                    "rollout_percent": 30,
                    "partial_match_product_type": "foundation",
                    "partial_match_step_index": 1,
                },
            },
        )
        self._create_plan(
            username="rq_partial_not_selected",
            category="makeup",
            meta={
                "source": "roadmap_v1",
                "category": "makeup",
                "ml": {
                    "decision": "fallback",
                    "mode": "v4_ranking",
                    "model_path": "/tmp/model.pkl",
                    "fallback_reason": "partial_rollout_not_selected",
                    "rollout_mode": "partial",
                    "rollout_selected": False,
                    "rollout_reason": "bucket_out_of_range",
                    "rollout_bucket": 83,
                    "rollout_percent": 30,
                    "partial_match_product_type": "foundation",
                    "partial_match_step_index": 1,
                },
            },
        )
        self._create_plan(
            username="rq_full_model",
            category="haircare",
            meta={
                "source": "roadmap_v1",
                "category": "haircare",
                "ml": {
                    "decision": "model_used",
                    "mode": "v4_ranking",
                    "model_path": "/tmp/model.pkl",
                    "rollout_mode": "full",
                    "rollout_selected": True,
                    "rollout_reason": "category_enabled",
                },
            },
        )

        out = StringIO()
        call_command("report_roadmap_quality", days=7, include_ga=True, stdout=out)
        text = out.getvalue()

        self.assertIn("### Rollout mode distribution", text)
        self.assertIn("| partial | 2 |", text)
        self.assertIn("| full | 1 |", text)
        self.assertIn("### Model-used split (full vs partial)", text)
        self.assertIn("| full | 1 |", text)
        self.assertIn("| partial | 1 |", text)
        self.assertIn("| partial_rollout_not_selected | 1 |", text)
        self.assertIn("### Partial rollout by category", text)
        self.assertIn("| makeup | 2 | 1 |", text)

    def test_runtime_paths_always_write_meta_ml_decision(self):
        User = get_user_model()
        user = User.objects.create_user(username="rq_runtime_u1", password="pass12345")

        serum = Product.objects.create(
            name="R Serum",
            brand="B",
            category="skincare",
            product_type="serum",
            in_stock=True,
        )
        Product.objects.create(
            name="R Cleanser",
            brand="B",
            category="skincare",
            product_type="cleanser",
            in_stock=True,
        )
        Product.objects.create(
            name="R Moisturizer",
            brand="B",
            category="skincare",
            product_type="moisturizer",
            in_stock=True,
        )
        Product.objects.create(
            name="R SPF",
            brand="B",
            category="skincare",
            product_type="spf",
            in_stock=True,
        )

        plan = refresh_roadmap(user, category="skincare", post_ctx=None)
        ml_meta = (plan.meta or {}).get("ml") if isinstance(plan.meta, dict) else None
        self.assertIsInstance(ml_meta, dict)
        self.assertTrue(str(ml_meta.get("decision") or "").strip())
        self.assertTrue(str(ml_meta.get("mode") or "").strip())
        self.assertTrue(str(ml_meta.get("model_path") or "").strip())
        self.assertIn(str(ml_meta.get("decision")), {"model_used", "fallback", "disabled"})
        self.assertIn(str(ml_meta.get("rollout_mode")), {"full", "partial", "none"})
        self.assertIn(bool(ml_meta.get("rollout_selected")), {True, False})
        if str(ml_meta.get("decision")) == "disabled":
            self.assertTrue(str(ml_meta.get("disabled_reason") or "").strip())

        updated = update_roadmap_from_purchase(
            user,
            {
                "categories": ["skincare"],
                "product_ids": [int(serum.id)],
            },
        )
        self.assertIsNotNone(updated)
        updated_plan = updated["plan"]
        ml_meta_updated = (updated_plan.meta or {}).get("ml") if isinstance(updated_plan.meta, dict) else None
        self.assertIsInstance(ml_meta_updated, dict)
        self.assertTrue(str(ml_meta_updated.get("decision") or "").strip())
        self.assertTrue(str(ml_meta_updated.get("mode") or "").strip())
        self.assertTrue(str(ml_meta_updated.get("model_path") or "").strip())
        self.assertIn(str(ml_meta_updated.get("decision")), {"model_used", "fallback", "disabled"})
        self.assertIn(str(ml_meta_updated.get("rollout_mode")), {"full", "partial", "none"})
        self.assertIn(bool(ml_meta_updated.get("rollout_selected")), {True, False})
        if str(ml_meta_updated.get("decision")) == "disabled":
            self.assertTrue(str(ml_meta_updated.get("disabled_reason") or "").strip())

    def test_disabled_decision_always_has_explicit_reason(self):
        User = get_user_model()
        user = User.objects.create_user(username="rq_disabled_reason_u1", password="pass12345")

        Product.objects.create(
            name="D Serum",
            brand="B",
            category="skincare",
            product_type="serum",
            in_stock=True,
        )
        Product.objects.create(
            name="D Cleanser",
            brand="B",
            category="skincare",
            product_type="cleanser",
            in_stock=True,
        )
        Product.objects.create(
            name="D Moisturizer",
            brand="B",
            category="skincare",
            product_type="moisturizer",
            in_stock=True,
        )
        Product.objects.create(
            name="D SPF",
            brand="B",
            category="skincare",
            product_type="spf",
            in_stock=True,
        )

        with override_settings(ROADMAP_NEXTSTEP_V4_ENABLED=False, ROADMAP_NEXTSTEP_V3_ENABLED=False):
            plan = refresh_roadmap(user, category="skincare", post_ctx=None)

        ml_meta = (plan.meta or {}).get("ml") if isinstance(plan.meta, dict) else None
        self.assertIsInstance(ml_meta, dict)
        self.assertEqual(str(ml_meta.get("decision")), "disabled")
        self.assertTrue(str(ml_meta.get("disabled_reason") or "").strip())

    def test_category_disabled_sets_disabled_reason(self):
        User = get_user_model()
        user = User.objects.create_user(username="rq_cat_disabled_u1", password="pass12345")

        with override_settings(
            ROADMAP_NEXTSTEP_V4_ENABLED=True,
            ROADMAP_NEXTSTEP_V3_ENABLED=False,
            ROADMAP_NEXTSTEP_V4_ENABLED_CATEGORIES=["skincare", "haircare", "makeup"],
            ROADMAP_NEXTSTEP_V4_DISABLED_CATEGORIES=["fragrance"],
        ):
            plan = refresh_roadmap(user, category="fragrance", post_ctx=None)

        ml_meta = (plan.meta or {}).get("ml") if isinstance(plan.meta, dict) else None
        self.assertIsInstance(ml_meta, dict)
        self.assertEqual(str(ml_meta.get("decision")), "disabled")
        self.assertEqual(str(ml_meta.get("disabled_reason")), "category_disabled")
        guard = ml_meta.get("category_guard")
        self.assertIsInstance(guard, dict)
        self.assertFalse(bool(guard.get("passed")))

    def test_category_allow_deny_precedence_global_off_wins(self):
        User = get_user_model()
        user = User.objects.create_user(username="rq_precedence_u1", password="pass12345")

        with override_settings(
            ROADMAP_NEXTSTEP_V4_ENABLED=False,
            ROADMAP_NEXTSTEP_V3_ENABLED=False,
            ROADMAP_NEXTSTEP_V4_ENABLED_CATEGORIES=["fragrance"],
            ROADMAP_NEXTSTEP_V4_DISABLED_CATEGORIES=["fragrance"],
        ):
            plan = refresh_roadmap(user, category="fragrance", post_ctx=None)

        ml_meta = (plan.meta or {}).get("ml") if isinstance(plan.meta, dict) else None
        self.assertIsInstance(ml_meta, dict)
        self.assertEqual(str(ml_meta.get("decision")), "disabled")
        self.assertEqual(str(ml_meta.get("disabled_reason")), "ml_disabled")

    def test_allowed_category_with_passing_guards_can_use_model(self):
        from unittest.mock import patch

        User = get_user_model()
        user = User.objects.create_user(username="rq_guard_pass_u1", password="pass12345")
        self._create_skincare_products("guard-pass")

        with override_settings(
            ROADMAP_NEXTSTEP_V4_ENABLED=True,
            ROADMAP_NEXTSTEP_V3_ENABLED=False,
            ROADMAP_NEXTSTEP_V4_ENABLED_CATEGORIES=["skincare", "haircare", "makeup"],
            ROADMAP_NEXTSTEP_V4_DISABLED_CATEGORIES=["fragrance"],
            ROADMAP_NEXTSTEP_V4_CONFIDENCE_THRESHOLD=0.1,
        ), patch(
            "roadmap_app.services.v4_min_lift_guard_status",
            return_value={"passed": True, "reason": "ok"},
        ), patch(
            "roadmap_app.services.v4_category_staged_rollout_status",
            return_value={
                "passed": True,
                "final_status": "ENABLE",
                "current_decision": "ENABLE",
                "reason": "passed",
                "hold_reason": None,
                "category": "skincare",
                "recommendation_7d": "ENABLE",
                "recommendation_30d": "ENABLE",
                "stability_gate_failures": [],
                "guard_7d": {"passed": True, "reason": "passed"},
                "guard_30d": {"passed": True, "reason": "passed"},
            },
        ), patch(
            "roadmap_app.services.predict_next_product_types",
            return_value=[
                {"candidate_type": "serum", "score": 0.91},
                {"candidate_type": "cleanser", "score": 0.12},
            ],
        ):
            plan = refresh_roadmap(user, category="skincare", post_ctx=None)

        ml_meta = (plan.meta or {}).get("ml") if isinstance(plan.meta, dict) else None
        self.assertIsInstance(ml_meta, dict)
        self.assertEqual(str(ml_meta.get("decision")), "model_used")
        self.assertIsNone(ml_meta.get("fallback_reason"))
        self.assertIsNone(ml_meta.get("disabled_reason"))
        guard = ml_meta.get("category_guard")
        self.assertIsInstance(guard, dict)
        self.assertTrue(bool(guard.get("passed")))

    def test_failing_category_uplift_guard_falls_back(self):
        from unittest.mock import patch

        User = get_user_model()
        user = User.objects.create_user(username="rq_guard_fail_u1", password="pass12345")
        self._create_skincare_products("guard-fail")

        with override_settings(
            ROADMAP_NEXTSTEP_V4_ENABLED=True,
            ROADMAP_NEXTSTEP_V3_ENABLED=False,
            ROADMAP_NEXTSTEP_V4_ENABLED_CATEGORIES=["skincare", "haircare", "makeup"],
            ROADMAP_NEXTSTEP_V4_DISABLED_CATEGORIES=["fragrance"],
        ), patch(
            "roadmap_app.services.v4_min_lift_guard_status",
            return_value={"passed": True, "reason": "ok"},
        ), patch(
            "roadmap_app.services.v4_category_staged_rollout_status",
            return_value={
                "passed": False,
                "final_status": "HOLD",
                "current_decision": "HOLD",
                "reason": "7d_unstable",
                "hold_reason": "severe_negative_offer_ctr_lift",
                "category": "skincare",
                "recommendation_7d": "HOLD",
                "recommendation_30d": "ENABLE",
                "stability_gate_failures": ["7d:severe_negative_offer_ctr_lift"],
                "guard_7d": {"passed": False, "reason": "severe_negative_offer_ctr_lift"},
                "guard_30d": {"passed": True, "reason": "passed"},
            },
        ), patch(
            "roadmap_app.services.predict_next_product_types",
        ) as predict_mock:
            plan = refresh_roadmap(user, category="skincare", post_ctx=None)

        ml_meta = (plan.meta or {}).get("ml") if isinstance(plan.meta, dict) else None
        self.assertIsInstance(ml_meta, dict)
        self.assertEqual(str(ml_meta.get("decision")), "fallback")
        self.assertEqual(str(ml_meta.get("fallback_reason")), "category_guard_failed")
        guard = ml_meta.get("category_guard")
        self.assertIsInstance(guard, dict)
        self.assertFalse(bool(guard.get("passed")))
        self.assertEqual(str(guard.get("final_status")), "HOLD")
        predict_mock.assert_not_called()
