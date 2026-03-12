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

    def _run_report_md(self, **kwargs) -> str:
        out = StringIO()
        params = {
            "days": 30,
            "format": "md",
            "cohort_mode": "fresh",
            "control": "non_model",
            "include_ga": True,
            "categories": "skincare,makeup",
            "min_sample": 30,
        }
        params.update(kwargs)
        call_command("report_roadmap_ml_diagnostics", stdout=out, **params)
        return out.getvalue()

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
        ml_extra: dict | None = None,
    ) -> tuple[RoadmapPlan, list[RoadmapStep]]:
        ml_meta = {
            "mode": "v4_ranking",
            "model_path": "/tmp/model.pkl",
        }
        if decision is not None:
            ml_meta["decision"] = decision
        if ml_extra:
            ml_meta.update(ml_extra)
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

    def _served_slot_row(self, payload: dict, *, category: str, model_slot: str) -> dict:
        rows = payload["runtime_observability"].get("served_model_slots", {}).get("model_used_outcomes_by_slot", [])
        for row in rows:
            if row.get("category") == category and row.get("model_slot") == model_slot:
                return row
        self.fail(f"served slot row not found: {category}/{model_slot}")

    def _served_slot_target_row(
        self,
        payload: dict,
        *,
        category: str,
        model_slot: str,
        planned_target_product_type: str,
    ) -> dict:
        rows = (
            payload["runtime_observability"]
            .get("served_model_slots", {})
            .get("model_used_outcomes_by_slot_and_planned_target", [])
        )
        for row in rows:
            if (
                row.get("category") == category
                and row.get("model_slot") == model_slot
                and row.get("planned_target_product_type") == planned_target_product_type
            ):
                return row
        self.fail(
            f"served slot target row not found: {category}/{model_slot}/{planned_target_product_type}"
        )

    def _shadow_outcome_by_top1_row(self, payload: dict, *, category: str, shadow_top1: str) -> dict:
        rows = payload["runtime_observability"]["shadow"].get("outcome_by_shadow_top1", [])
        for row in rows:
            if row.get("category") == category and row.get("shadow_top1") == shadow_top1:
                return row
        self.fail(f"shadow outcome by top1 row not found: {category}/{shadow_top1}")

    def _shadow_outcome_by_pair_row(
        self,
        payload: dict,
        *,
        category: str,
        active_top1: str,
        shadow_top1: str,
    ) -> dict:
        rows = payload["runtime_observability"]["shadow"].get("outcome_by_swap_pair", [])
        for row in rows:
            if (
                row.get("category") == category
                and row.get("active_top1") == active_top1
                and row.get("shadow_top1") == shadow_top1
            ):
                return row
        self.fail(f"shadow outcome by pair row not found: {category}/{active_top1}->{shadow_top1}")

    def _candidate_outcome_by_top1_row(self, payload: dict, *, category: str, candidate_top1: str) -> dict:
        rows = payload["runtime_observability"].get("candidate_path_compare", {}).get("outcome_by_candidate_top1", [])
        for row in rows:
            if row.get("category") == category and row.get("candidate_top1") == candidate_top1:
                return row
        self.fail(f"candidate outcome by top1 row not found: {category}/{candidate_top1}")

    def _candidate_outcome_by_pair_row(
        self,
        payload: dict,
        *,
        category: str,
        active_top1: str,
        candidate_top1: str,
    ) -> dict:
        rows = payload["runtime_observability"].get("candidate_path_compare", {}).get("outcome_by_swap_pair", [])
        for row in rows:
            if (
                row.get("category") == category
                and row.get("active_top1") == active_top1
                and row.get("candidate_top1") == candidate_top1
            ):
                return row
        self.fail(f"candidate outcome by pair row not found: {category}/{active_top1}->{candidate_top1}")

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

    def test_report_includes_shadow_top1_agreement_and_swaps(self):
        u_same = self._user("diag_shadow_same")
        u_diff = self._user("diag_shadow_diff")

        self._plan(
            user=u_same,
            decision="model_used",
            category="skincare",
            ml_extra={
                "predictions": [{"candidate_type": "serum", "score": 0.91}],
                "shadow": {
                    "enabled": True,
                    "reason": "ok",
                    "model_version": "shadow_semantic_v1",
                    "predictions": [{"candidate_type": "serum", "score": 0.87}],
                },
            },
        )
        self._plan(
            user=u_diff,
            decision="disabled",
            category="skincare",
            ml_extra={
                "predictions": [{"candidate_type": "cleanser", "score": 0.66}],
                "shadow": {
                    "enabled": True,
                    "reason": "ok",
                    "model_version": "shadow_semantic_v1",
                    "predictions": [{"candidate_type": "moisturizer", "score": 0.71}],
                },
            },
        )

        payload = self._run_report_json(control="disabled", categories="skincare", min_sample=1)
        shadow = payload["runtime_observability"]["shadow"]
        top1 = shadow["top1_comparison"]

        self.assertEqual(shadow["plans_with_shadow_meta"], 2)
        self.assertEqual(shadow["shadow_enabled_plans"], 2)
        self.assertEqual(shadow["reason_counts"]["ok"], 2)
        self.assertEqual(shadow["model_version_counts"]["shadow_semantic_v1"], 2)
        self.assertEqual(top1["eligible_plans"], 2)
        self.assertEqual(top1["same_top1_plans"], 1)
        self.assertEqual(top1["different_top1_plans"], 1)
        self.assertAlmostEqual(float(top1["agreement_rate"]), 0.5, places=6)
        skincare = shadow["by_category"]["skincare"]
        self.assertEqual(skincare["different_top1_plans"], 1)
        self.assertAlmostEqual(float(skincare["agreement_rate"]), 0.5, places=6)
        self.assertEqual(shadow["top_swaps"][0]["category"], "skincare")
        self.assertEqual(shadow["top_swaps"][0]["active_top1"], "cleanser")
        self.assertEqual(shadow["top_swaps"][0]["shadow_top1"], "moisturizer")
        self.assertEqual(int(shadow["top_swaps"][0]["plans"]), 1)

    def test_report_includes_shadow_vs_real_completion_outcome(self):
        u_active = self._user("diag_shadow_outcome_active")
        u_shadow = self._user("diag_shadow_outcome_shadow")

        plan_active, steps_active = self._plan(
            user=u_active,
            decision="model_used",
            category="haircare",
            steps=[(1, "serum")],
            ml_extra={
                "predictions": [{"candidate_type": "serum", "score": 0.91}],
                "shadow": {
                    "enabled": True,
                    "reason": "ok",
                    "model_version": "shadow_semantic_v1",
                    "predictions": [{"candidate_type": "conditioner", "score": 0.83}],
                },
            },
        )
        plan_shadow, steps_shadow = self._plan(
            user=u_shadow,
            decision="disabled",
            category="haircare",
            steps=[(1, "conditioner")],
            ml_extra={
                "predictions": [{"candidate_type": "shampoo", "score": 0.65}],
                "shadow": {
                    "enabled": True,
                    "reason": "ok",
                    "model_version": "shadow_semantic_v1",
                    "predictions": [{"candidate_type": "conditioner", "score": 0.82}],
                },
            },
        )

        self._emit_step_events(user=u_active, plan=plan_active, step=steps_active[0], completed=1)
        self._emit_step_events(user=u_shadow, plan=plan_shadow, step=steps_shadow[0], completed=1)

        payload = self._run_report_json(control="disabled", categories="haircare", min_sample=1)
        outcome = payload["runtime_observability"]["shadow"]["outcome_comparison"]

        self.assertEqual(outcome["eligible_plans"], 2)
        self.assertEqual(outcome["active_hits"], 1)
        self.assertEqual(outcome["shadow_hits"], 1)
        self.assertEqual(outcome["active_only_hits"], 1)
        self.assertEqual(outcome["shadow_only_hits"], 1)
        self.assertEqual(outcome["both_hits"], 0)
        self.assertEqual(outcome["neither_hits"], 0)
        self.assertAlmostEqual(float(outcome["active_hit_rate"]), 0.5, places=6)
        self.assertAlmostEqual(float(outcome["shadow_hit_rate"]), 0.5, places=6)
        self.assertAlmostEqual(float(outcome["shadow_delta_vs_active"]), 0.0, places=6)
        haircare = outcome["by_category"]["haircare"]
        self.assertEqual(haircare["eligible_plans"], 2)
        self.assertEqual(haircare["active_only_hits"], 1)
        self.assertEqual(haircare["shadow_only_hits"], 1)
        by_top1 = self._shadow_outcome_by_top1_row(payload, category="haircare", shadow_top1="conditioner")
        self.assertEqual(by_top1["eligible_plans"], 2)
        self.assertEqual(by_top1["active_hits"], 1)
        self.assertEqual(by_top1["shadow_hits"], 1)
        self.assertEqual(by_top1["top_actual_outcomes"][0]["product_type"], "conditioner")
        self.assertEqual(by_top1["top_actual_outcomes"][1]["product_type"], "serum")
        self.assertEqual(by_top1["top_actual_outcomes_text"], "conditioner:1, serum:1")
        pair_active = self._shadow_outcome_by_pair_row(
            payload,
            category="haircare",
            active_top1="serum",
            shadow_top1="conditioner",
        )
        self.assertEqual(pair_active["eligible_plans"], 1)
        self.assertEqual(pair_active["active_only_hits"], 1)
        self.assertEqual(pair_active["shadow_only_hits"], 0)
        self.assertAlmostEqual(float(pair_active["shadow_delta_vs_active"]), -1.0, places=6)
        pair_shadow = self._shadow_outcome_by_pair_row(
            payload,
            category="haircare",
            active_top1="shampoo",
            shadow_top1="conditioner",
        )
        self.assertEqual(pair_shadow["eligible_plans"], 1)
        self.assertEqual(pair_shadow["active_only_hits"], 0)
        self.assertEqual(pair_shadow["shadow_only_hits"], 1)
        self.assertAlmostEqual(float(pair_shadow["shadow_delta_vs_active"]), 1.0, places=6)

    def test_report_markdown_includes_shadow_pair_outcome_sections(self):
        user = self._user("diag_shadow_md")
        plan, steps = self._plan(
            user=user,
            decision="model_used",
            category="haircare",
            steps=[(1, "conditioner")],
            ml_extra={
                "predictions": [{"candidate_type": "shampoo", "score": 0.61}],
                "shadow": {
                    "enabled": True,
                    "reason": "ok",
                    "model_version": "shadow_semantic_v1",
                    "predictions": [{"candidate_type": "conditioner", "score": 0.84}],
                },
            },
        )
        self._emit_step_events(user=user, plan=plan, step=steps[0], completed=1)

        markdown = self._run_report_md(control="disabled", categories="haircare", min_sample=1)

        self.assertIn("### shadow outcome by predicted top1", markdown)
        self.assertIn("### shadow outcome by swap pair", markdown)
        self.assertIn("haircare | shampoo | conditioner", markdown)

    def test_report_includes_read_only_candidate_path_compare(self):
        u_active = self._user("diag_candidate_active")
        u_candidate = self._user("diag_candidate_only")

        plan_active, steps_active = self._plan(
            user=u_active,
            decision="model_used",
            category="haircare",
            steps=[(1, "shampoo"), (2, "conditioner")],
            ml_extra={
                "predictions": [{"candidate_type": "shampoo", "score": 0.91}],
                "planned_target_product_type": "conditioner",
                "planned_target_step_index": 2,
            },
        )
        plan_candidate, steps_candidate = self._plan(
            user=u_candidate,
            decision="model_used",
            category="haircare",
            steps=[(1, "shampoo"), (2, "conditioner")],
            ml_extra={
                "predictions": [{"candidate_type": "shampoo", "score": 0.88}],
                "planned_target_product_type": "conditioner",
                "planned_target_step_index": 2,
            },
        )

        self._emit_step_events(user=u_active, plan=plan_active, step=steps_active[0], completed=1)
        self._emit_step_events(user=u_candidate, plan=plan_candidate, step=steps_candidate[1], completed=1)

        nextstep_active = {
            "model_path": "/models/nextstep-active/model.pkl",
            "model_version": "nextstep_active_v1",
            "selected_feature_set": "full",
            "exists": True,
        }
        nextstep_candidate = {
            "model_path": "/models/nextstep-candidate/model.pkl",
            "model_version": "nextstep_primary14_v1",
            "selected_feature_set": "full",
            "exists": True,
        }

        def _predict_candidate(model_path, *, user, **kwargs):
            self.assertEqual(model_path, "/models/nextstep-candidate/model.pkl")
            if int(user) == int(u_active.id):
                return [{"candidate_type": "conditioner", "score": 0.83}]
            if int(user) == int(u_candidate.id):
                return [{"candidate_type": "conditioner", "score": 0.81}]
            return []

        with patch(
            "roadmap_app.management.commands.report_roadmap_ml_diagnostics.nextstep_model_artifact_summary",
            side_effect=[nextstep_active, nextstep_candidate],
        ), patch(
            "roadmap_app.management.commands.report_roadmap_ml_diagnostics.predict_next_product_types_for_model_path",
            side_effect=_predict_candidate,
        ):
            payload = self._run_report_json(
                control="disabled",
                categories="haircare",
                min_sample=1,
                nextstep_candidate_model_path="/models/nextstep-candidate/model.pkl",
            )

        compare = payload["runtime_observability"]["candidate_path_compare"]
        self.assertEqual(compare["model_version"], "nextstep_primary14_v1")
        self.assertEqual(compare["plans_scanned"], 2)
        self.assertEqual(compare["predicted_plans"], 2)
        self.assertEqual(compare["top1_comparison"]["eligible_plans"], 2)
        self.assertEqual(compare["top1_comparison"]["same_top1_plans"], 0)
        self.assertEqual(compare["top1_comparison"]["different_top1_plans"], 2)
        self.assertAlmostEqual(float(compare["top1_comparison"]["agreement_rate"]), 0.0, places=6)
        self.assertEqual(compare["by_category"]["haircare"]["different_top1_plans"], 2)
        self.assertEqual(compare["top_swaps"][0]["active_top1"], "shampoo")
        self.assertEqual(compare["top_swaps"][0]["candidate_top1"], "conditioner")

        outcome = compare["outcome_comparison"]
        self.assertEqual(outcome["eligible_plans"], 2)
        self.assertEqual(outcome["active_hits"], 1)
        self.assertEqual(outcome["candidate_hits"], 1)
        self.assertEqual(outcome["active_only_hits"], 1)
        self.assertEqual(outcome["candidate_only_hits"], 1)
        self.assertEqual(outcome["both_hits"], 0)
        self.assertEqual(outcome["neither_hits"], 0)
        self.assertAlmostEqual(float(outcome["active_hit_rate"]), 0.5, places=6)
        self.assertAlmostEqual(float(outcome["candidate_hit_rate"]), 0.5, places=6)
        self.assertAlmostEqual(float(outcome["candidate_delta_vs_active"]), 0.0, places=6)

        by_top1 = self._candidate_outcome_by_top1_row(payload, category="haircare", candidate_top1="conditioner")
        self.assertEqual(by_top1["eligible_plans"], 2)
        self.assertEqual(by_top1["candidate_hits"], 1)
        self.assertEqual(by_top1["top_actual_outcomes"][0]["product_type"], "conditioner")
        self.assertEqual(by_top1["top_actual_outcomes"][1]["product_type"], "shampoo")
        pair = self._candidate_outcome_by_pair_row(
            payload,
            category="haircare",
            active_top1="shampoo",
            candidate_top1="conditioner",
        )
        self.assertEqual(pair["eligible_plans"], 2)
        self.assertEqual(pair["active_only_hits"], 1)
        self.assertEqual(pair["candidate_only_hits"], 1)

    def test_report_markdown_includes_candidate_path_compare_sections(self):
        user = self._user("diag_candidate_md")
        plan, steps = self._plan(
            user=user,
            decision="model_used",
            category="haircare",
            steps=[(1, "shampoo"), (2, "conditioner")],
            ml_extra={
                "predictions": [{"candidate_type": "shampoo", "score": 0.74}],
                "planned_target_product_type": "conditioner",
                "planned_target_step_index": 2,
            },
        )
        self._emit_step_events(user=user, plan=plan, step=steps[1], completed=1)

        with patch(
            "roadmap_app.management.commands.report_roadmap_ml_diagnostics.nextstep_model_artifact_summary",
            side_effect=[
                {"model_path": "/models/nextstep-active/model.pkl", "model_version": "active_v1", "exists": True},
                {"model_path": "/models/nextstep-candidate/model.pkl", "model_version": "primary14_v1", "exists": True},
            ],
        ), patch(
            "roadmap_app.management.commands.report_roadmap_ml_diagnostics.predict_next_product_types_for_model_path",
            return_value=[{"candidate_type": "conditioner", "score": 0.9}],
        ):
            markdown = self._run_report_md(
                control="disabled",
                categories="haircare",
                min_sample=1,
                nextstep_candidate_model_path="/models/nextstep-candidate/model.pkl",
            )

        self.assertIn("### candidate path compare", markdown)
        self.assertIn("### candidate path vs outcome", markdown)
        self.assertIn("### candidate path outcome by swap pair", markdown)
        self.assertIn("haircare | shampoo | conditioner", markdown)

    def test_report_includes_served_model_slot_and_planned_target_outcomes(self):
        u_partial = self._user("diag_slot_partial")
        u_active = self._user("diag_slot_active")

        plan_partial, steps_partial = self._plan(
            user=u_partial,
            decision="model_used",
            category="haircare",
            steps=[(1, "conditioner")],
            ml_extra={
                "model_slot": "partial_candidate",
                "model_version": "semantic_v4_canary",
                "planned_target_product_type": "conditioner",
            },
        )
        plan_active, steps_active = self._plan(
            user=u_active,
            decision="model_used",
            category="haircare",
            steps=[(1, "hair_mask")],
            ml_extra={
                "model_slot": "active",
                "model_version": "active_v4",
                "planned_target_product_type": "hair_mask",
            },
        )

        self._emit_step_events(user=u_partial, plan=plan_partial, step=steps_partial[0], exposed=4, completed=3)
        self._emit_step_events(user=u_active, plan=plan_active, step=steps_active[0], exposed=5, completed=1)

        payload = self._run_report_json(control="disabled", categories="haircare", min_sample=1)
        served = payload["runtime_observability"]["served_model_slots"]

        self.assertEqual(served["slot_counts"]["active"], 1)
        self.assertEqual(served["slot_counts"]["partial_candidate"], 1)
        self.assertEqual(served["model_version_counts"]["active_v4"], 1)
        self.assertEqual(served["model_version_counts"]["semantic_v4_canary"], 1)
        self.assertEqual(served["by_category"]["haircare"]["active"], 1)
        self.assertEqual(served["by_category"]["haircare"]["partial_candidate"], 1)

        partial_row = self._served_slot_row(payload, category="haircare", model_slot="partial_candidate")
        self.assertEqual(partial_row["plans"], 1)
        self.assertEqual(partial_row["step_exposed"], 4)
        self.assertEqual(partial_row["step_completed"], 3)
        self.assertAlmostEqual(float(partial_row["step_completion_rate"]), 0.75, places=6)

        active_row = self._served_slot_row(payload, category="haircare", model_slot="active")
        self.assertEqual(active_row["plans"], 1)
        self.assertEqual(active_row["step_exposed"], 5)
        self.assertEqual(active_row["step_completed"], 1)
        self.assertAlmostEqual(float(active_row["step_completion_rate"]), 0.2, places=6)

        partial_target = self._served_slot_target_row(
            payload,
            category="haircare",
            model_slot="partial_candidate",
            planned_target_product_type="conditioner",
        )
        self.assertEqual(partial_target["plans"], 1)
        self.assertAlmostEqual(float(partial_target["step_completion_rate"]), 0.75, places=6)

    def test_report_markdown_includes_served_model_slot_sections(self):
        user = self._user("diag_slot_md")
        plan, steps = self._plan(
            user=user,
            decision="model_used",
            category="haircare",
            steps=[(1, "conditioner")],
            ml_extra={
                "model_slot": "partial_candidate",
                "model_version": "semantic_v4_canary",
                "planned_target_product_type": "conditioner",
            },
        )
        self._emit_step_events(user=user, plan=plan, step=steps[0], exposed=2, completed=1)

        markdown = self._run_report_md(control="disabled", categories="haircare", min_sample=1)

        self.assertIn("### served model slots", markdown)
        self.assertIn("### model-used outcome by served slot", markdown)
        self.assertIn("### model-used outcome by served slot and planned target", markdown)
        self.assertIn("haircare | partial_candidate | conditioner", markdown)
