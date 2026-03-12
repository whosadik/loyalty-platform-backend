from __future__ import annotations

from io import StringIO
from unittest.mock import patch

from django.contrib.auth import get_user_model
from django.core.management import call_command
from django.test import TestCase, override_settings

from roadmap_app.models import RoadmapPlan, RoadmapStep


class BackfillRoadmapRuntimePolicyMetaTests(TestCase):
    def _plan(self, username: str, *, category: str = "haircare", meta: dict) -> RoadmapPlan:
        User = get_user_model()
        user = User.objects.create_user(username=username, password="pass12345")
        plan = RoadmapPlan.objects.create(
            user=user,
            category=category,
            is_active=True,
            meta=meta,
        )
        RoadmapStep.objects.create(
            plan=plan,
            step_index=1,
            product_type="shampoo",
            status=RoadmapStep.Status.COMPLETED,
        )
        RoadmapStep.objects.create(
            plan=plan,
            step_index=2,
            product_type="leave_in",
            status=RoadmapStep.Status.RECOMMENDED,
        )
        return plan

    def test_dry_run_reports_pending_runtime_policy_backfill_without_writing(self):
        plan = self._plan(
            "runtime_policy_backfill_dry",
            meta={
                "source": "roadmap_v1",
                "ml": {
                    "decision": "model_used",
                    "mode": "v4_ranking",
                    "model_path": "/tmp/active-model.pkl",
                    "predictions": [{"candidate_type": "leave_in", "score": 0.9}],
                },
                "context": {"post_ctx_product_ids": [101, 202]},
            },
        )

        out = StringIO()
        with patch(
            "roadmap_app.management.commands.backfill_roadmap_runtime_policy_meta.nextstep_model_artifact_summary",
            return_value={
                "exists": True,
                "model_version": "active_v4",
                "selected_feature_set": "full",
            },
        ), patch(
            "roadmap_app.management.commands.backfill_roadmap_runtime_policy_meta.predict_next_product_types_for_model_path",
            return_value=[
                {
                    "candidate_type": "leave_in",
                    "score": 0.81,
                    "runtime_policies": ["haircare_leavein_rerank"],
                    "runtime_policy_biases": {"haircare_leavein_rerank": 1.31},
                }
            ],
        ):
            call_command(
                "backfill_roadmap_runtime_policy_meta",
                days=30,
                include_ga=True,
                model_path="/tmp/active-model.pkl",
                stdout=out,
            )

        text = out.getvalue()
        plan.refresh_from_db()
        self.assertIn("mode: `dry-run`", text)
        self.assertIn("plans needing runtime policy backfill: `1`", text)
        self.assertIn("haircare_leavein_rerank: `1`", text)
        self.assertEqual(((plan.meta or {}).get("ml") or {}).get("runtime_policies"), None)

    def test_write_backfills_runtime_policy_payload(self):
        plan = self._plan(
            "runtime_policy_backfill_write",
            meta={
                "source": "roadmap_v1",
                "ml": {
                    "decision": "model_used",
                    "mode": "v4_ranking",
                    "model_path": "/tmp/active-model.pkl",
                    "predictions": [{"candidate_type": "leave_in", "score": 0.88}],
                },
            },
        )

        out = StringIO()
        with patch(
            "roadmap_app.management.commands.backfill_roadmap_runtime_policy_meta.nextstep_model_artifact_summary",
            return_value={
                "exists": True,
                "model_version": "active_v4",
                "selected_feature_set": "full",
            },
        ), patch(
            "roadmap_app.management.commands.backfill_roadmap_runtime_policy_meta.predict_next_product_types_for_model_path",
            return_value=[
                {
                    "candidate_type": "leave_in",
                    "score": 0.93,
                    "runtime_policies": ["haircare_leavein_rerank", "haircare_progression_bias"],
                    "runtime_policy_biases": {
                        "haircare_leavein_rerank": 1.4,
                        "haircare_progression_bias": 0.12,
                    },
                }
            ],
        ) as predict_mock:
            call_command(
                "backfill_roadmap_runtime_policy_meta",
                days=30,
                include_ga=True,
                model_path="/tmp/active-model.pkl",
                write=True,
                stdout=out,
            )

        text = out.getvalue()
        plan.refresh_from_db()
        ml = ((plan.meta or {}).get("ml") or {})

        self.assertIn("mode: `write`", text)
        self.assertIn("plans updated: `1`", text)
        self.assertEqual(
            ml.get("runtime_policies"),
            ["haircare_leavein_rerank", "haircare_progression_bias"],
        )
        self.assertEqual(
            ml.get("runtime_policy_max_abs_bias"),
            {
                "haircare_leavein_rerank": 1.4,
                "haircare_progression_bias": 0.12,
            },
        )
        self.assertEqual(str(ml.get("runtime_policy_meta_source")), "backfill_projection")
        self.assertTrue(str(ml.get("runtime_policy_meta_updated_at") or ""))
        predict_mock.assert_called_once()

    @override_settings(ROADMAP_NEXTSTEP_V4_MODEL_PATH="/tmp/current-active-model.pkl")
    def test_falls_back_to_current_default_model_when_stored_model_path_is_missing(self):
        plan = self._plan(
            "runtime_policy_backfill_default_fallback",
            meta={
                "source": "roadmap_v1",
                "ml": {
                    "decision": "model_used",
                    "mode": "v4_ranking",
                    "model_path": "/tmp/missing-stored-model.pkl",
                    "predictions": [{"candidate_type": "leave_in", "score": 0.77}],
                },
            },
        )

        def _artifact_summary(path: str):
            raw = str(path)
            if raw.endswith("missing-stored-model.pkl"):
                return {"exists": False}
            if raw.endswith("current-active-model.pkl"):
                return {
                    "exists": True,
                    "model_version": "active_v4_current",
                    "selected_feature_set": "full",
                }
            return {"exists": False}

        out = StringIO()
        with patch(
            "roadmap_app.management.commands.backfill_roadmap_runtime_policy_meta.nextstep_model_artifact_summary",
            side_effect=_artifact_summary,
        ), patch(
            "roadmap_app.management.commands.backfill_roadmap_runtime_policy_meta.predict_next_product_types_for_model_path",
            return_value=[
                {
                    "candidate_type": "leave_in",
                    "score": 0.9,
                    "runtime_policies": ["haircare_leavein_rerank"],
                    "runtime_policy_biases": {"haircare_leavein_rerank": 1.22},
                }
            ],
        ) as predict_mock:
            call_command(
                "backfill_roadmap_runtime_policy_meta",
                days=30,
                include_ga=True,
                write=True,
                stdout=out,
            )

        plan.refresh_from_db()
        ml = ((plan.meta or {}).get("ml") or {})
        self.assertEqual(ml.get("runtime_policies"), ["haircare_leavein_rerank"])
        self.assertEqual(str(ml.get("runtime_policy_meta_source")), "backfill_projection")
        predict_mock.assert_called_once()
        args, _ = predict_mock.call_args
        self.assertTrue(str(args[0]).endswith("current-active-model.pkl"))
