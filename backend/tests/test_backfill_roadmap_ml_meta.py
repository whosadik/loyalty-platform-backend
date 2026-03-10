from __future__ import annotations

from io import StringIO

from django.contrib.auth import get_user_model
from django.core.management import call_command
from django.test import TestCase

from roadmap_app.models import RoadmapPlan


class BackfillRoadmapMlMetaTests(TestCase):
    def _plan(self, username: str, meta: dict) -> RoadmapPlan:
        User = get_user_model()
        user = User.objects.create_user(username=username, password="pass12345")
        return RoadmapPlan.objects.create(
            user=user,
            category="skincare",
            is_active=True,
            meta=meta,
        )

    def test_dry_run_reports_changes_without_writing(self):
        plan = self._plan(
            "backfill_dry_u1",
            {
                "source": "roadmap_v1",
                "ml": {
                    "used": True,
                },
            },
        )

        out = StringIO()
        call_command("backfill_roadmap_ml_meta", days=30, include_ga=True, stdout=out)
        text = out.getvalue()
        plan.refresh_from_db()

        self.assertIn("mode: `dry-run`", text)
        self.assertIn("plans needing normalization: `1`", text)
        self.assertEqual((plan.meta or {}).get("ml", {}).get("decision"), None)

    def test_write_normalizes_legacy_ml_payload(self):
        plan_model = self._plan(
            "backfill_write_model",
            {
                "source": "roadmap_v1",
                "ml": {
                    "used": True,
                },
            },
        )
        plan_fallback = self._plan(
            "backfill_write_fb",
            {
                "source": "roadmap_v1",
                "ml": {
                    "fallback_reason": "low_confidence",
                },
            },
        )

        out = StringIO()
        call_command("backfill_roadmap_ml_meta", days=30, include_ga=True, write=True, stdout=out)
        text = out.getvalue()
        plan_model.refresh_from_db()
        plan_fallback.refresh_from_db()

        model_ml = (plan_model.meta or {}).get("ml") or {}
        fallback_ml = (plan_fallback.meta or {}).get("ml") or {}

        self.assertIn("mode: `write`", text)
        self.assertIn("plans updated: `2`", text)
        self.assertEqual(str(model_ml.get("decision")), "model_used")
        self.assertEqual(str(fallback_ml.get("decision")), "fallback")
        self.assertTrue(str(model_ml.get("mode") or "").strip())
        self.assertTrue(str(fallback_ml.get("model_path") or "").strip())
