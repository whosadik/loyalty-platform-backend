from __future__ import annotations

from io import StringIO
from unittest.mock import patch

from django.contrib.auth import get_user_model
from django.core.management import call_command
from django.test import TestCase

from roadmap_app.models import RoadmapPlan, RoadmapStep


class BackfillRoadmapShadowMetaTests(TestCase):
    def _plan(self, username: str, *, meta: dict) -> RoadmapPlan:
        User = get_user_model()
        user = User.objects.create_user(username=username, password="pass12345")
        plan = RoadmapPlan.objects.create(
            user=user,
            category="skincare",
            is_active=True,
            meta=meta,
        )
        RoadmapStep.objects.create(
            plan=plan,
            step_index=1,
            product_type="serum",
            status=RoadmapStep.Status.RECOMMENDED,
        )
        RoadmapStep.objects.create(
            plan=plan,
            step_index=2,
            product_type="moisturizer",
            status=RoadmapStep.Status.RECOMMENDED,
        )
        return plan

    def test_dry_run_reports_pending_shadow_without_writing(self):
        plan = self._plan(
            "shadow_backfill_dry",
            meta={
                "source": "roadmap_v1",
                "ml": {
                    "decision": "model_used",
                    "mode": "v4_ranking",
                    "model_path": "/tmp/active-model.pkl",
                    "predictions": [{"candidate_type": "serum", "score": 0.9}],
                },
                "context": {"post_ctx_product_ids": [101, 202]},
            },
        )

        out = StringIO()
        with patch(
            "roadmap_app.management.commands.backfill_roadmap_shadow_meta.nextstep_model_artifact_summary",
            return_value={
                "exists": True,
                "model_version": "shadow_semantic_v1",
                "selected_feature_set": "full",
            },
        ), patch(
            "roadmap_app.management.commands.backfill_roadmap_shadow_meta.predict_next_product_types_for_model_path",
            return_value=[{"candidate_type": "moisturizer", "score": 0.81}],
        ):
            call_command(
                "backfill_roadmap_shadow_meta",
                days=30,
                include_ga=True,
                model_path="/tmp/shadow-model.pkl",
                stdout=out,
            )

        text = out.getvalue()
        plan.refresh_from_db()
        self.assertIn("mode: `dry-run`", text)
        self.assertIn("plans needing shadow backfill: `1`", text)
        self.assertEqual(((plan.meta or {}).get("ml") or {}).get("shadow"), None)

    def test_write_backfills_shadow_payload(self):
        plan = self._plan(
            "shadow_backfill_write",
            meta={
                "source": "roadmap_v1",
                "ml": {
                    "decision": "fallback",
                    "mode": "v4_ranking",
                    "model_path": "/tmp/active-model.pkl",
                    "predictions": [{"candidate_type": "serum", "score": 0.64}],
                },
            },
        )

        out = StringIO()
        with patch(
            "roadmap_app.management.commands.backfill_roadmap_shadow_meta.nextstep_model_artifact_summary",
            return_value={
                "exists": True,
                "model_version": "shadow_semantic_v2",
                "selected_feature_set": "full",
            },
        ), patch(
            "roadmap_app.management.commands.backfill_roadmap_shadow_meta.predict_next_product_types_for_model_path",
            return_value=[{"candidate_type": "moisturizer", "score": 0.88}],
        ) as predict_mock:
            call_command(
                "backfill_roadmap_shadow_meta",
                days=30,
                include_ga=True,
                model_path="/tmp/shadow-model.pkl",
                write=True,
                stdout=out,
            )

        text = out.getvalue()
        plan.refresh_from_db()
        shadow = (((plan.meta or {}).get("ml") or {}).get("shadow") or {})

        self.assertIn("mode: `write`", text)
        self.assertIn("plans updated: `1`", text)
        self.assertEqual(str(shadow.get("reason")), "ok")
        self.assertTrue(bool(shadow.get("enabled")))
        self.assertEqual(str(shadow.get("model_version")), "shadow_semantic_v2")
        self.assertEqual(str(shadow.get("selected_feature_set")), "full")
        self.assertEqual(str((shadow.get("predictions") or [])[0].get("candidate_type")), "moisturizer")
        predict_mock.assert_called_once()
