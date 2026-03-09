from __future__ import annotations

import json
from io import StringIO

from django.contrib.auth import get_user_model
from django.core.management import call_command
from django.test import TestCase

from offers.models import CampaignBudget, Offer, OfferAssignment, OfferEvent
from roadmap_app.models import RoadmapEvent, RoadmapPlan, RoadmapStep


class ReportRoadmapMlUpliftTests(TestCase):
    def _run_report_json(self, **kwargs) -> dict:
        out = StringIO()
        params = {
            "days": 30,
            "format": "json",
            "cohort_mode": "fresh",
            "control": "non_model",
            "include_ga": True,
        }
        params.update(kwargs)
        call_command("report_roadmap_ml_uplift", stdout=out, **params)
        return json.loads(out.getvalue())

    def _user(self, username: str):
        User = get_user_model()
        return User.objects.create_user(username=username, password="pass12345")

    def _plan(self, *, user, decision: str | None, category: str = "skincare", product_type: str = "serum"):
        ml_meta = {
            "mode": "v4_ranking",
            "model_path": "/tmp/model.pkl",
        }
        if decision is not None:
            ml_meta["decision"] = decision
        plan = RoadmapPlan.objects.create(
            user=user,
            category=category,
            is_active=True,
            meta={"source": "roadmap_v1", "ml": ml_meta},
        )
        step = RoadmapStep.objects.create(
            plan=plan,
            step_index=1,
            product_type=product_type,
            status=RoadmapStep.Status.RECOMMENDED,
        )
        return plan, step

    def _offer(self, suffix: str = "1") -> Offer:
        camp = CampaignBudget.objects.create(name=f"camp_{suffix}", weekly_limit=1000)
        return Offer.objects.create(
            name=f"offer_{suffix}",
            offer_type=Offer.Type.DISCOUNT,
            campaign=camp,
        )

    def test_fresh_mode_excludes_missing_ml_meta(self):
        u1 = self._user("uplift_fresh_u1")
        u2 = self._user("uplift_fresh_u2")
        self._plan(user=u1, decision="model_used")
        self._plan(user=u2, decision=None)

        payload = self._run_report_json(cohort_mode="fresh", control="non_model")

        self.assertEqual(payload["runtime_observability"]["decision_counts"]["missing_ml_meta"], 1)
        self.assertEqual(payload["unattributed_excluded"]["fresh_mode_excluded_missing_ml_meta_plans"], 1)
        self.assertEqual(payload["overall"]["plans_total_in_scope"], 2)
        self.assertEqual(payload["overall"]["plans_total_after_cohort_mode"], 1)

    def test_control_non_model_combines_fallback_and_disabled(self):
        u1 = self._user("uplift_ctrl_u1")
        u2 = self._user("uplift_ctrl_u2")
        u3 = self._user("uplift_ctrl_u3")
        self._plan(user=u1, decision="model_used")
        self._plan(user=u2, decision="fallback")
        self._plan(user=u3, decision="disabled")

        payload = self._run_report_json(control="non_model")

        self.assertEqual(payload["cohorts"]["model_used"]["plans_total"], 1)
        self.assertEqual(payload["cohorts"]["control"]["plans_total"], 2)
        self.assertEqual(payload["overall"]["control_plans_total"], 2)

    def test_effective_plan_id_attribution_works_for_plan_and_step_path(self):
        u_model = self._user("uplift_attr_u1")
        u_ctrl = self._user("uplift_attr_u2")
        model_plan, model_step = self._plan(user=u_model, decision="model_used")
        self._plan(user=u_ctrl, decision="disabled")

        RoadmapEvent.objects.create(
            user=u_model,
            plan=model_plan,
            step=model_step,
            event_type=RoadmapEvent.Type.STEP_EXPOSED,
            context={"sources": ["roadmap_api"]},
        )
        # plan=None, only step -> must still map through step__plan_id
        RoadmapEvent.objects.create(
            user=u_model,
            plan=None,
            step=model_step,
            event_type=RoadmapEvent.Type.STEP_CLICKED,
            context={},
        )

        payload = self._run_report_json(control="disabled")
        model = payload["cohorts"]["model_used"]

        self.assertEqual(model["roadmap_step_exposed"], 1)
        self.assertEqual(model["roadmap_step_clicked"], 1)

    def test_unattributed_offers_are_not_forced_into_cohorts(self):
        u_model = self._user("uplift_off_u1")
        u_ctrl = self._user("uplift_off_u2")
        self._plan(user=u_model, decision="model_used")
        self._plan(user=u_ctrl, decision="disabled")

        offer = self._offer("unattr")
        assignment = OfferAssignment.objects.create(
            user=u_model,
            offer=offer,
            reason={"source": "roadmap_post_purchase"},
            target={"picked_via": "roadmap_shortcut"},
        )
        OfferEvent.objects.create(
            assignment=assignment,
            user=u_model,
            offer=offer,
            campaign_name="camp_unattr",
            event_type=OfferEvent.Type.EXPOSED,
            context={},
        )

        payload = self._run_report_json(control="disabled")
        self.assertEqual(payload["unattributed_excluded"]["roadmap_assignments_unattributed"], 1)
        self.assertEqual(payload["cohorts"]["model_used"]["offers_assigned_total"], 0)
        self.assertEqual(payload["cohorts"]["control"]["offers_assigned_total"], 0)

    def test_uplift_abs_and_rel_are_computed_correctly(self):
        u_model = self._user("uplift_math_u1")
        u_ctrl = self._user("uplift_math_u2")
        model_plan, model_step = self._plan(user=u_model, decision="model_used")
        ctrl_plan, ctrl_step = self._plan(user=u_ctrl, decision="disabled")

        for _ in range(10):
            RoadmapEvent.objects.create(
                user=u_model,
                plan=model_plan,
                step=model_step,
                event_type=RoadmapEvent.Type.STEP_EXPOSED,
                context={"sources": ["roadmap_api"]},
            )
            RoadmapEvent.objects.create(
                user=u_ctrl,
                plan=ctrl_plan,
                step=ctrl_step,
                event_type=RoadmapEvent.Type.STEP_EXPOSED,
                context={"sources": ["roadmap_api"]},
            )
        for _ in range(6):
            RoadmapEvent.objects.create(
                user=u_model,
                plan=None,
                step=model_step,
                event_type=RoadmapEvent.Type.STEP_CLICKED,
                context={},
            )
        for _ in range(4):
            RoadmapEvent.objects.create(
                user=u_ctrl,
                plan=None,
                step=ctrl_step,
                event_type=RoadmapEvent.Type.STEP_CLICKED,
                context={},
            )

        payload = self._run_report_json(control="disabled")
        metric = payload["uplift"]["overall"]["step_funnel"]["step_ctr"]
        self.assertAlmostEqual(metric["abs_lift"], 0.2, places=6)
        self.assertAlmostEqual(metric["rel_lift"], 0.5, places=6)

    def test_low_sample_path_does_not_crash(self):
        u_model = self._user("uplift_low_u1")
        u_ctrl = self._user("uplift_low_u2")
        model_plan, model_step = self._plan(user=u_model, decision="model_used")
        ctrl_plan, ctrl_step = self._plan(user=u_ctrl, decision="disabled")

        RoadmapEvent.objects.create(
            user=u_model,
            plan=model_plan,
            step=model_step,
            event_type=RoadmapEvent.Type.STEP_EXPOSED,
            context={"sources": ["roadmap_api"]},
        )
        RoadmapEvent.objects.create(
            user=u_ctrl,
            plan=ctrl_plan,
            step=ctrl_step,
            event_type=RoadmapEvent.Type.STEP_EXPOSED,
            context={"sources": ["roadmap_api"]},
        )

        payload = self._run_report_json(control="disabled")
        metric = payload["uplift"]["overall"]["step_funnel"]["step_ctr"]
        self.assertTrue(metric["low_sample"])

    def test_include_ga_respected(self):
        u_non_ga = self._user("uplift_ga_non")
        u_ga = self._user("ga_uplift_ga_yes")
        self._plan(user=u_non_ga, decision="model_used")
        self._plan(user=u_ga, decision="model_used")

        without_ga = self._run_report_json(include_ga=False, control="disabled")
        with_ga = self._run_report_json(include_ga=True, control="disabled")

        self.assertEqual(without_ga["overall"]["plans_total_in_scope"], 1)
        self.assertEqual(with_ga["overall"]["plans_total_in_scope"], 2)
