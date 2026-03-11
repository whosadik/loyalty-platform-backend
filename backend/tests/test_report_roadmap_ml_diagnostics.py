from __future__ import annotations

import json
from io import StringIO
from unittest.mock import patch

from django.contrib.auth import get_user_model
from django.core.management import call_command
from django.test import TestCase

from offers.models import CampaignBudget, Offer, OfferAssignment, OfferEvent
from roadmap_app.models import RoadmapEvent, RoadmapPlan, RoadmapStep


class ReportRoadmapMlDiagnosticsTests(TestCase):
    def _run_report_json(self, **kwargs) -> dict:
        out = StringIO()
        params = {
            "days": 30,
            "format": "json",
            "cohort_mode": "fresh",
            "control": "non_model",
            "include_ga": True,
            "categories": "skincare,makeup",
            "min_sample": 30,
        }
        params.update(kwargs)
        call_command("report_roadmap_ml_diagnostics", stdout=out, **params)
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
        steps: list[tuple[int, str]] | None = None,
    ) -> tuple[RoadmapPlan, list[RoadmapStep]]:
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
        out_steps: list[RoadmapStep] = []
        for idx, product_type in (steps or [(1, "serum")]):
            out_steps.append(
                RoadmapStep.objects.create(
                    plan=plan,
                    step_index=idx,
                    product_type=product_type,
                    status=RoadmapStep.Status.RECOMMENDED,
                )
            )
        return plan, out_steps

    def _emit_step_events(
        self,
        *,
        user,
        plan: RoadmapPlan,
        step: RoadmapStep,
        exposed: int = 0,
        clicked: int = 0,
        completed: int = 0,
        skipped: int = 0,
        source: str = "roadmap_api",
    ) -> None:
        for _ in range(exposed):
            RoadmapEvent.objects.create(
                user=user,
                plan=plan,
                step=step,
                event_type=RoadmapEvent.Type.STEP_EXPOSED,
                context={"sources": [source]},
            )
        for _ in range(clicked):
            RoadmapEvent.objects.create(
                user=user,
                plan=None,
                step=step,
                event_type=RoadmapEvent.Type.STEP_CLICKED,
                context={},
            )
        for _ in range(completed):
            RoadmapEvent.objects.create(
                user=user,
                plan=None,
                step=step,
                event_type=RoadmapEvent.Type.STEP_COMPLETED,
                context={},
            )
        for _ in range(skipped):
            RoadmapEvent.objects.create(
                user=user,
                plan=None,
                step=step,
                event_type=RoadmapEvent.Type.STEP_SKIPPED,
                context={},
            )

    def _offer(self, suffix: str) -> Offer:
        camp = CampaignBudget.objects.create(name=f"diag_camp_{suffix}", weekly_limit=1000)
        return Offer.objects.create(
            name=f"diag_offer_{suffix}",
            offer_type=Offer.Type.DISCOUNT,
            campaign=camp,
        )

    def _slice_row(self, payload: dict, *, category: str, slice_type: str, slice_value: str) -> dict:
        for row in payload.get("slice_rows", []):
            if (
                row.get("category") == category
                and row.get("slice_type") == slice_type
                and row.get("slice_value") == slice_value
            ):
                return row
        self.fail(f"slice row not found: {category}/{slice_type}/{slice_value}")

    def _policy_row(self, payload: dict, policy: str) -> dict:
        for row in payload.get("policy_simulation", []):
            if row.get("policy") == policy:
                return row
        self.fail(f"policy row not found: {policy}")

    def test_cohort_split_model_vs_non_model_is_correct(self):
        u_model = self._user("diag_cohort_model")
        u_fallback = self._user("diag_cohort_fallback")
        u_disabled = self._user("diag_cohort_disabled")
        u_missing = self._user("diag_cohort_missing")

        self._plan(user=u_model, decision="model_used", category="skincare")
        self._plan(user=u_fallback, decision="fallback", category="skincare")
        self._plan(user=u_disabled, decision="disabled", category="skincare")
        self._plan(user=u_missing, decision=None, category="skincare")

        payload = self._run_report_json(control="non_model", categories="skincare")

        self.assertEqual(payload["overall"]["plans_total_in_scope"], 4)
        self.assertEqual(payload["overall"]["plans_total_after_cohort_mode"], 3)
        self.assertEqual(payload["overall"]["model_used_plans_total"], 1)
        self.assertEqual(payload["overall"]["control_plans_total"], 2)
        self.assertEqual(payload["runtime_observability"]["decision_counts"]["missing_ml_meta"], 1)

    def test_slice_aggregation_by_step_product_type(self):
        u_model = self._user("diag_pt_model")
        u_ctrl = self._user("diag_pt_ctrl")
        model_plan, model_steps = self._plan(
            user=u_model,
            decision="model_used",
            category="skincare",
            steps=[(1, "cleanser"), (2, "serum")],
        )
        ctrl_plan, ctrl_steps = self._plan(
            user=u_ctrl,
            decision="disabled",
            category="skincare",
            steps=[(1, "cleanser")],
        )
        model_cleanser = model_steps[0]
        ctrl_cleanser = ctrl_steps[0]

        self._emit_step_events(user=u_model, plan=model_plan, step=model_cleanser, exposed=5, clicked=3, completed=2)
        self._emit_step_events(user=u_ctrl, plan=ctrl_plan, step=ctrl_cleanser, exposed=4, clicked=1, completed=1)

        payload = self._run_report_json(control="disabled", categories="skincare", min_sample=1)
        row = self._slice_row(
            payload,
            category="skincare",
            slice_type="step_product_type",
            slice_value="cleanser",
        )

        self.assertEqual(row["model_exposed"], 5)
        self.assertEqual(row["control_exposed"], 4)
        self.assertAlmostEqual(row["step_ctr_lift"], 0.35, places=6)
        self.assertAlmostEqual(row["step_completion_lift"], 0.15, places=6)

    def test_slice_aggregation_by_step_index(self):
        u_model = self._user("diag_idx_model")
        u_ctrl = self._user("diag_idx_ctrl")
        model_plan, model_steps = self._plan(
            user=u_model,
            decision="model_used",
            category="makeup",
            steps=[(2, "foundation")],
        )
        ctrl_plan, ctrl_steps = self._plan(
            user=u_ctrl,
            decision="disabled",
            category="makeup",
            steps=[(2, "foundation")],
        )

        self._emit_step_events(user=u_model, plan=model_plan, step=model_steps[0], exposed=3, completed=2)
        self._emit_step_events(user=u_ctrl, plan=ctrl_plan, step=ctrl_steps[0], exposed=2, completed=1)

        payload = self._run_report_json(control="disabled", categories="makeup", min_sample=1)
        row = self._slice_row(
            payload,
            category="makeup",
            slice_type="step_index",
            slice_value="step_2",
        )
        self.assertEqual(row["model_exposed"], 3)
        self.assertEqual(row["control_exposed"], 2)
        self.assertAlmostEqual(row["step_completion_lift"], (2 / 3) - (1 / 2), places=6)

    def test_unattributed_offers_are_not_force_assigned(self):
        u_model = self._user("diag_offer_model")
        u_ctrl = self._user("diag_offer_ctrl")
        self._plan(user=u_model, decision="model_used", category="skincare")
        self._plan(user=u_ctrl, decision="disabled", category="skincare")

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
            campaign_name="diag_camp_unattr",
            event_type=OfferEvent.Type.EXPOSED,
            context={},
        )

        payload = self._run_report_json(control="disabled", categories="skincare")

        self.assertEqual(payload["unattributed"]["roadmap_assignments_unattributed"], 1)
        skincare_summary = next(
            row for row in payload["category_summary"] if row["category"] == "skincare"
        )
        self.assertEqual(skincare_summary["model"]["offer_assigned"], 0)
        self.assertEqual(skincare_summary["control"]["offer_assigned"], 0)

    def test_low_sample_path_does_not_crash(self):
        u_model = self._user("diag_low_model")
        u_ctrl = self._user("diag_low_ctrl")
        model_plan, model_steps = self._plan(user=u_model, decision="model_used", category="skincare")
        ctrl_plan, ctrl_steps = self._plan(user=u_ctrl, decision="disabled", category="skincare")

        self._emit_step_events(user=u_model, plan=model_plan, step=model_steps[0], exposed=1)
        self._emit_step_events(user=u_ctrl, plan=ctrl_plan, step=ctrl_steps[0], exposed=1)

        payload = self._run_report_json(control="disabled", categories="skincare", min_sample=10)
        row = self._slice_row(
            payload,
            category="skincare",
            slice_type="step_product_type",
            slice_value="serum",
        )
        self.assertEqual(row["verdict"], "LOW_SAMPLE")

    def test_partial_rollout_simulation_is_deterministic(self):
        u_mf = self._user("diag_pol_m_found")
        u_mm = self._user("diag_pol_m_masc")
        u_cf = self._user("diag_pol_c_found")
        u_cm = self._user("diag_pol_c_masc")

        p_mf, s_mf = self._plan(
            user=u_mf,
            decision="model_used",
            category="makeup",
            steps=[(1, "foundation")],
        )
        p_mm, s_mm = self._plan(
            user=u_mm,
            decision="model_used",
            category="makeup",
            steps=[(1, "mascara")],
        )
        p_cf, s_cf = self._plan(
            user=u_cf,
            decision="disabled",
            category="makeup",
            steps=[(1, "foundation")],
        )
        p_cm, s_cm = self._plan(
            user=u_cm,
            decision="disabled",
            category="makeup",
            steps=[(1, "mascara")],
        )

        self._emit_step_events(user=u_mf, plan=p_mf, step=s_mf[0], exposed=10, clicked=6, completed=8)
        self._emit_step_events(user=u_mm, plan=p_mm, step=s_mm[0], exposed=10, clicked=4, completed=1)
        self._emit_step_events(user=u_cf, plan=p_cf, step=s_cf[0], exposed=10, clicked=5, completed=4)
        self._emit_step_events(user=u_cm, plan=p_cm, step=s_cm[0], exposed=10, clicked=4, completed=5)

        payload_1 = self._run_report_json(control="disabled", categories="makeup", min_sample=1)
        payload_2 = self._run_report_json(control="disabled", categories="makeup", min_sample=1)

        self.assertEqual(payload_1["policy_simulation"], payload_2["policy_simulation"])

        policy_a = self._policy_row(payload_1, "Policy A - current")
        policy_b = self._policy_row(payload_1, "Policy B - makeup partial")
        self.assertGreater(float(policy_b["step_completion_lift"] or 0.0), float(policy_a["step_completion_lift"] or 0.0))

    def test_report_includes_active_and_candidate_artifact_inventory(self):
        nextstep_active = {
            "model_path": "/models/nextstep-active/model.pkl",
            "model_version": "nextstep_active_v1",
            "selected_feature_set": "full",
            "exists": True,
            "metrics_test": {"ndcg_at_5": 0.62, "recall_at_1": 0.34},
            "runtime_guard": {"passed": True},
        }
        nextstep_candidate = {
            "model_path": "/models/nextstep-candidate/model.pkl",
            "model_version": "nextstep_semantic_v2",
            "selected_feature_set": "full",
            "exists": True,
            "metrics_test": {"ndcg_at_5": 0.68, "recall_at_1": 0.39},
            "runtime_guard": {"passed": True},
        }
        planner_active = {
            "model_path": "/models/planner-active/model.pkl",
            "model_version": "planner_active_v1",
            "selected_feature_set": "baseline_only",
            "exists": True,
            "metrics_test": {"ndcg_at_5": 0.88, "recall_at_1": 0.74},
            "planner_guard": {"passed": True},
        }
        planner_candidate = {
            "model_path": "/models/planner-candidate/model.pkl",
            "model_version": "planner_semantic_v2",
            "selected_feature_set": "full",
            "exists": True,
            "metrics_test": {"ndcg_at_5": 0.90, "recall_at_1": 0.71},
            "planner_guard": {"passed": False},
        }

        with patch(
            "roadmap_app.management.commands.report_roadmap_ml_diagnostics.nextstep_model_artifact_summary",
            side_effect=[nextstep_active, nextstep_candidate],
        ), patch(
            "roadmap_app.management.commands.report_roadmap_ml_diagnostics.planner_model_artifact_summary",
            side_effect=[planner_active, planner_candidate],
        ):
            payload = self._run_report_json(
                categories="skincare",
                nextstep_candidate_model_path="/models/nextstep-candidate/model.pkl",
                planner_candidate_model_path="/models/planner-candidate/model.pkl",
            )

        self.assertEqual(
            payload["artifacts"]["nextstep"]["active"]["model_version"],
            "nextstep_active_v1",
        )
        self.assertEqual(
            payload["artifacts"]["nextstep"]["candidate"]["model_version"],
            "nextstep_semantic_v2",
        )
        self.assertEqual(
            payload["artifacts"]["planner"]["active"]["selected_feature_set"],
            "baseline_only",
        )
        self.assertFalse(bool(payload["artifacts"]["planner"]["candidate"]["planner_guard"]["passed"]))
