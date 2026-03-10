from __future__ import annotations

import json
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory

from django.contrib.auth import get_user_model
from django.core.management import call_command
from django.test import TestCase
from django.test.utils import override_settings

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

    def _plan(
        self,
        *,
        user,
        decision: str | None,
        category: str = "skincare",
        product_type: str = "serum",
        step_index: int = 1,
        ml_overrides: dict | None = None,
    ):
        ml_meta = {
            "mode": "v4_ranking",
            "model_path": "/tmp/model.pkl",
        }
        if decision is not None:
            ml_meta["decision"] = decision
        if isinstance(ml_overrides, dict):
            ml_meta.update(ml_overrides)
        plan = RoadmapPlan.objects.create(
            user=user,
            category=category,
            is_active=True,
            meta={"source": "roadmap_v1", "ml": ml_meta},
        )
        step = RoadmapStep.objects.create(
            plan=plan,
            step_index=step_index,
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

    def test_partial_makeup_uplift_block_has_overall_and_slice_breakdowns(self):
        u_canary = self._user("uplift_partial_canary")
        u_ctrl = self._user("uplift_partial_ctrl")
        u_ctrl_2 = self._user("uplift_partial_ctrl_2")

        canary_plan, canary_step = self._plan(
            user=u_canary,
            decision="model_used",
            category="makeup",
            product_type="foundation",
            step_index=1,
            ml_overrides={
                "rollout_mode": "partial",
                "rollout_selected": True,
                "rollout_reason": "selected",
                "rollout_bucket": 12,
                "rollout_percent": 30,
                "partial_match_product_type": "foundation",
                "partial_match_step_index": 1,
            },
        )
        ctrl_plan, ctrl_step = self._plan(
            user=u_ctrl,
            decision="fallback",
            category="makeup",
            product_type="foundation",
            step_index=1,
            ml_overrides={
                "rollout_mode": "partial",
                "rollout_selected": False,
                "rollout_reason": "bucket_out_of_range",
                "fallback_reason": "partial_rollout_not_selected",
            },
        )
        ctrl_plan_2, ctrl_step_2 = self._plan(
            user=u_ctrl_2,
            decision="disabled",
            category="makeup",
            product_type="mascara",
            step_index=2,
            ml_overrides={"rollout_mode": "none", "rollout_selected": False},
        )

        for _ in range(10):
            RoadmapEvent.objects.create(
                user=u_canary,
                plan=canary_plan,
                step=canary_step,
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
        for _ in range(8):
            RoadmapEvent.objects.create(
                user=u_canary,
                plan=None,
                step=canary_step,
                event_type=RoadmapEvent.Type.STEP_COMPLETED,
                context={},
            )
        for _ in range(4):
            RoadmapEvent.objects.create(
                user=u_ctrl,
                plan=None,
                step=ctrl_step,
                event_type=RoadmapEvent.Type.STEP_COMPLETED,
                context={},
            )

        for _ in range(5):
            RoadmapEvent.objects.create(
                user=u_ctrl_2,
                plan=ctrl_plan_2,
                step=ctrl_step_2,
                event_type=RoadmapEvent.Type.STEP_EXPOSED,
                context={"sources": ["roadmap_api"]},
            )

        payload = self._run_report_json(control="non_model", category="makeup", min_plans=1)

        partial = payload["partial_makeup_uplift"]
        self.assertEqual(partial["overall"]["canary"]["plans_total"], 1)
        self.assertEqual(partial["overall"]["control"]["plans_total"], 2)
        metric = partial["overall"]["uplift"]["step_funnel"]["step_completion_rate"]
        self.assertGreater(float(metric["abs_lift"] or 0.0), 0.0)
        self.assertIn("foundation", partial["by_product_type"])
        self.assertIn("step_1", partial["by_step_index"])

    def test_sync_runtime_artifact_writes_model_owned_json(self):
        with TemporaryDirectory() as model_tmp:
            model_dir = Path(model_tmp)
            out = StringIO()
            err = StringIO()
            with override_settings(
                ROADMAP_NEXTSTEP_V4_ENABLED=True,
                ROADMAP_NEXTSTEP_V4_MODEL_PATH=str(model_dir / "model.pkl"),
            ):
                call_command(
                    "report_roadmap_ml_uplift",
                    days=7,
                    category="all",
                    format="json",
                    cohort_mode="fresh",
                    control="non_model",
                    include_ga=True,
                    sync_runtime_artifact=True,
                    stdout=out,
                    stderr=err,
                )

            payload = json.loads(out.getvalue())
            runtime_path = model_dir / "uplift_report_7d.json"
            self.assertTrue(runtime_path.exists())
            self.assertTrue("synced runtime artifact" in err.getvalue())
            runtime_payload = json.loads(runtime_path.read_text(encoding="utf-8"))
            self.assertTrue(bool(runtime_payload["params"]["sync_runtime_artifact"]))
            self.assertEqual(runtime_payload["params"]["days"], 7)
            self.assertEqual(payload["params"]["days"], 7)
