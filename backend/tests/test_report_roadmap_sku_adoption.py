from __future__ import annotations

import json
from datetime import timedelta
from decimal import Decimal
from io import StringIO

from django.contrib.auth import get_user_model
from django.core.management import call_command
from django.test import TestCase
from django.utils import timezone

from catalog.models import Product
from roadmap_app.models import RoadmapEvent, RoadmapPlan, RoadmapStep


class ReportRoadmapSkuAdoptionTests(TestCase):
    def _run_report_json(self, **kwargs) -> dict:
        out = StringIO()
        params = {
            "days": 30,
            "format": "json",
            "cohort_mode": "all",
            "include_ga": True,
            "category": "all",
        }
        params.update(kwargs)
        call_command("report_roadmap_sku_adoption", stdout=out, **params)
        return json.loads(out.getvalue())

    def _user(self, username: str):
        User = get_user_model()
        return User.objects.create_user(username=username, password="pass12345")

    def _event(
        self,
        *,
        user,
        plan: RoadmapPlan | None,
        step: RoadmapStep | None,
        event_type: str,
        created_at,
        context: dict,
    ) -> RoadmapEvent:
        event = RoadmapEvent.objects.create(
            user=user,
            plan=plan,
            step=step,
            event_type=event_type,
            context=context,
        )
        RoadmapEvent.objects.filter(id=event.id).update(created_at=created_at)
        event.refresh_from_db()
        return event

    def test_report_counts_exact_and_semantic_adoption_by_recommended_product(self):
        base = timezone.now() - timedelta(days=1)
        rec = Product.objects.create(
            name="Foundation Hero",
            brand="Glow",
            price=Decimal("100.00"),
            category="makeup",
            product_type="foundation",
            in_stock=True,
        )
        alt = Product.objects.create(
            name="Foundation Alt",
            brand="Glow",
            price=Decimal("95.00"),
            category="makeup",
            product_type="foundation",
            in_stock=True,
        )

        user_exact = self._user("sku_adoption_exact")
        plan_exact = RoadmapPlan.objects.create(
            user=user_exact,
            category="makeup",
            is_active=True,
            meta={"source": "roadmap_v1", "ml": {"decision": "model_used"}},
        )
        step_exact = RoadmapStep.objects.create(
            plan=plan_exact,
            step_index=1,
            product_type="foundation",
            status=RoadmapStep.Status.RECOMMENDED,
            recommended_product=rec,
        )
        self._event(
            user=user_exact,
            plan=plan_exact,
            step=None,
            event_type=RoadmapEvent.Type.PLAN_REFRESHED,
            created_at=base + timedelta(minutes=1),
            context={
                "plan_id": plan_exact.id,
                "category": "makeup",
                "source": "roadmap_v1",
                "next_step_id": step_exact.id,
                "ml": {"decision": "model_used"},
            },
        )
        self._event(
            user=user_exact,
            plan=plan_exact,
            step=step_exact,
            event_type=RoadmapEvent.Type.STEP_GENERATED,
            created_at=base + timedelta(minutes=2),
            context={
                "plan_id": plan_exact.id,
                "step_id": step_exact.id,
                "category": "makeup",
                "product_type": "foundation",
                "recommended_product_id": rec.id,
                "has_recommendation": True,
                "ml": {"decision": "model_used"},
            },
        )
        self._event(
            user=user_exact,
            plan=plan_exact,
            step=step_exact,
            event_type=RoadmapEvent.Type.STEP_CLICKED,
            created_at=base + timedelta(minutes=3),
            context={"category": "makeup"},
        )
        self._event(
            user=user_exact,
            plan=plan_exact,
            step=step_exact,
            event_type=RoadmapEvent.Type.STEP_COMPLETED,
            created_at=base + timedelta(minutes=4),
            context={
                "category": "makeup",
                "matched_by": "recommended_product_id",
                "match_meta": {
                    "recommended_product_id": rec.id,
                    "purchased_product_id": rec.id,
                    "purchased_product_type": "foundation",
                },
            },
        )

        user_semantic = self._user("sku_adoption_semantic")
        plan_semantic = RoadmapPlan.objects.create(
            user=user_semantic,
            category="makeup",
            is_active=True,
            meta={"source": "roadmap_v1", "ml": {"decision": "model_used"}},
        )
        step_semantic = RoadmapStep.objects.create(
            plan=plan_semantic,
            step_index=1,
            product_type="foundation",
            status=RoadmapStep.Status.RECOMMENDED,
            recommended_product=rec,
        )
        self._event(
            user=user_semantic,
            plan=plan_semantic,
            step=None,
            event_type=RoadmapEvent.Type.PLAN_REFRESHED,
            created_at=base + timedelta(minutes=5),
            context={
                "plan_id": plan_semantic.id,
                "category": "makeup",
                "source": "roadmap_v1",
                "next_step_id": step_semantic.id,
                "ml": {"decision": "model_used"},
            },
        )
        self._event(
            user=user_semantic,
            plan=plan_semantic,
            step=step_semantic,
            event_type=RoadmapEvent.Type.STEP_GENERATED,
            created_at=base + timedelta(minutes=6),
            context={
                "plan_id": plan_semantic.id,
                "step_id": step_semantic.id,
                "category": "makeup",
                "product_type": "foundation",
                "recommended_product_id": rec.id,
                "has_recommendation": True,
                "ml": {"decision": "model_used"},
            },
        )
        self._event(
            user=user_semantic,
            plan=plan_semantic,
            step=step_semantic,
            event_type=RoadmapEvent.Type.STEP_COMPLETED,
            created_at=base + timedelta(minutes=7),
            context={
                "category": "makeup",
                "matched_by": "semantic_content_match",
                "match_meta": {
                    "recommended_product_id": rec.id,
                    "purchased_product_id": alt.id,
                    "purchased_product_type": "foundation",
                },
            },
        )

        payload = self._run_report_json(category="makeup")

        overall = payload["overall"]
        self.assertEqual(int(overall["recommended_steps"]), 2)
        self.assertEqual(int(overall["exact_recommended_product_checkout"]), 1)
        self.assertEqual(int(overall["semantic_alternative_checkout"]), 1)
        self.assertEqual(int(overall["clicked_after_generated"]), 1)
        self.assertAlmostEqual(float(overall["exact_adoption_rate"]), 0.5, places=6)
        self.assertAlmostEqual(float(overall["semantic_alternative_rate"]), 0.5, places=6)

        rows = payload["breakdowns"]["by_recommended_product_rows"]
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(int(row["recommended_product_id"]), int(rec.id))
        self.assertEqual(int(row["recommended_steps"]), 2)
        self.assertEqual(int(row["exact_recommended_product_checkout"]), 1)
        self.assertEqual(int(row["semantic_alternative_checkout"]), 1)
        self.assertEqual(int(row["top_semantic_alternatives"][0]["purchased_product_id"]), int(alt.id))
        self.assertEqual(int(row["top_semantic_alternatives"][0]["count"]), 1)

    def test_report_isolates_next_step_generated_instances(self):
        base = timezone.now() - timedelta(days=1)
        rec_primary = Product.objects.create(
            name="Primary Serum",
            brand="Derma",
            price=Decimal("50.00"),
            category="skincare",
            product_type="serum",
            in_stock=True,
        )
        rec_non_next = Product.objects.create(
            name="Non Next Cream",
            brand="Derma",
            price=Decimal("60.00"),
            category="skincare",
            product_type="moisturizer",
            in_stock=True,
        )
        user = self._user("sku_adoption_next_step_only")
        plan = RoadmapPlan.objects.create(
            user=user,
            category="skincare",
            is_active=True,
            meta={"source": "roadmap_v1", "ml": {"decision": "model_used"}},
        )
        step_next = RoadmapStep.objects.create(
            plan=plan,
            step_index=1,
            product_type="serum",
            status=RoadmapStep.Status.RECOMMENDED,
            recommended_product=rec_primary,
        )
        step_other = RoadmapStep.objects.create(
            plan=plan,
            step_index=2,
            product_type="moisturizer",
            status=RoadmapStep.Status.RECOMMENDED,
            recommended_product=rec_non_next,
        )
        self._event(
            user=user,
            plan=plan,
            step=None,
            event_type=RoadmapEvent.Type.PLAN_REFRESHED,
            created_at=base + timedelta(minutes=1),
            context={
                "plan_id": plan.id,
                "category": "skincare",
                "source": "roadmap_v1",
                "next_step_id": step_next.id,
                "ml": {"decision": "model_used"},
            },
        )
        self._event(
            user=user,
            plan=plan,
            step=step_other,
            event_type=RoadmapEvent.Type.STEP_GENERATED,
            created_at=base + timedelta(minutes=2),
            context={
                "plan_id": plan.id,
                "step_id": step_other.id,
                "category": "skincare",
                "product_type": "moisturizer",
                "recommended_product_id": rec_non_next.id,
                "has_recommendation": True,
                "ml": {"decision": "model_used"},
            },
        )
        self._event(
            user=user,
            plan=plan,
            step=step_next,
            event_type=RoadmapEvent.Type.STEP_GENERATED,
            created_at=base + timedelta(minutes=3),
            context={
                "plan_id": plan.id,
                "step_id": step_next.id,
                "category": "skincare",
                "product_type": "serum",
                "recommended_product_id": rec_primary.id,
                "has_recommendation": True,
                "ml": {"decision": "model_used"},
            },
        )

        payload = self._run_report_json(category="skincare")

        overall = payload["overall"]
        self.assertEqual(int(overall["recommended_steps"]), 1)
        rows = payload["breakdowns"]["by_recommended_product_rows"]
        self.assertEqual(len(rows), 1)
        self.assertEqual(int(rows[0]["recommended_product_id"]), int(rec_primary.id))
