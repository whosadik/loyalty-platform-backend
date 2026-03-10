from __future__ import annotations

import json
from datetime import timedelta
from io import StringIO

from django.contrib.auth import get_user_model
from django.core.management import call_command
from django.test import TestCase
from django.utils import timezone

from roadmap_app.models import RoadmapEvent, RoadmapPlan, RoadmapStep


class ReportRoadmapGenerationGapTests(TestCase):
    def _run_report_json(self, **kwargs) -> dict:
        out = StringIO()
        params = {
            "days": 30,
            "format": "json",
            "cohort_mode": "all",
            "include_ga": True,
        }
        params.update(kwargs)
        call_command("report_roadmap_generation_gap", stdout=out, **params)
        return json.loads(out.getvalue())

    def _user(self, username: str):
        User = get_user_model()
        return User.objects.create_user(username=username, password="pass12345")

    def _plan(
        self,
        *,
        user,
        category: str,
        decision: str | None,
        step_index: int,
        product_type: str,
    ) -> tuple[RoadmapPlan, RoadmapStep]:
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
            step_index=step_index,
            product_type=product_type,
            status=RoadmapStep.Status.RECOMMENDED,
        )
        return plan, step

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

    def test_report_builds_overall_and_breakdowns(self):
        base = timezone.now() - timedelta(days=1)

        user_model = self._user("gap_model_u1")
        user_disabled = self._user("gap_disabled_u1")
        user_missing = self._user("gap_missing_u1")

        model_plan, model_step = self._plan(
            user=user_model,
            category="makeup",
            decision="model_used",
            step_index=1,
            product_type="foundation",
        )
        disabled_plan, disabled_step = self._plan(
            user=user_disabled,
            category="makeup",
            decision="disabled",
            step_index=2,
            product_type="concealer",
        )
        missing_plan, missing_step = self._plan(
            user=user_missing,
            category="skincare",
            decision=None,
            step_index=1,
            product_type="serum",
        )

        self._event(
            user=user_model,
            plan=model_plan,
            step=None,
            event_type=RoadmapEvent.Type.PLAN_REFRESHED,
            created_at=base + timedelta(minutes=1),
            context={
                "plan_id": model_plan.id,
                "category": "makeup",
                "next_step_id": model_step.id,
                "next_step_index": 1,
                "next_product_type": "foundation",
                "ml": {"decision": "model_used", "mode": "v4_ranking", "rollout_mode": "full"},
            },
        )
        self._event(
            user=user_model,
            plan=model_plan,
            step=model_step,
            event_type=RoadmapEvent.Type.STEP_GENERATED,
            created_at=base + timedelta(minutes=2),
            context={
                "plan_id": model_plan.id,
                "step_id": model_step.id,
                "step_index": 1,
                "category": "makeup",
                "product_type": "foundation",
                "status": "recommended",
                "recommended_product_id": 501,
                "has_recommendation": True,
                "source": "ml_next_step",
                "ml": {"decision": "model_used", "rollout_mode": "full"},
            },
        )
        self._event(
            user=user_model,
            plan=model_plan,
            step=model_step,
            event_type=RoadmapEvent.Type.STEP_EXPOSED,
            created_at=base + timedelta(minutes=3),
            context={"sources": ["roadmap_api"], "category": "makeup"},
        )
        self._event(
            user=user_model,
            plan=model_plan,
            step=model_step,
            event_type=RoadmapEvent.Type.STEP_CLICKED,
            created_at=base + timedelta(minutes=4),
            context={"category": "makeup"},
        )
        self._event(
            user=user_model,
            plan=model_plan,
            step=model_step,
            event_type=RoadmapEvent.Type.STEP_COMPLETED,
            created_at=base + timedelta(minutes=5),
            context={"category": "makeup", "matched_by": "recommended_product_id"},
        )

        self._event(
            user=user_disabled,
            plan=disabled_plan,
            step=None,
            event_type=RoadmapEvent.Type.PLAN_REFRESHED,
            created_at=base + timedelta(minutes=6),
            context={
                "plan_id": disabled_plan.id,
                "category": "makeup",
                "next_step_id": disabled_step.id,
                "next_step_index": 2,
                "next_product_type": "concealer",
                "ml": {"decision": "disabled", "mode": "v4_ranking", "rollout_mode": "disable"},
            },
        )
        self._event(
            user=user_disabled,
            plan=disabled_plan,
            step=disabled_step,
            event_type=RoadmapEvent.Type.STEP_GENERATED,
            created_at=base + timedelta(minutes=7),
            context={
                "plan_id": disabled_plan.id,
                "step_id": disabled_step.id,
                "step_index": 2,
                "category": "makeup",
                "product_type": "concealer",
                "status": "recommended",
                "recommended_product_id": 601,
                "has_recommendation": True,
                "source": "rules",
                "ml": {"decision": "disabled", "rollout_mode": "disable"},
            },
        )
        self._event(
            user=user_disabled,
            plan=disabled_plan,
            step=disabled_step,
            event_type=RoadmapEvent.Type.STEP_EXPOSED,
            created_at=base + timedelta(minutes=8),
            context={"sources": ["offers"], "category": "makeup", "offer_assignment_id": 123},
        )

        self._event(
            user=user_missing,
            plan=missing_plan,
            step=None,
            event_type=RoadmapEvent.Type.PLAN_REFRESHED,
            created_at=base + timedelta(minutes=9),
            context={
                "plan_id": missing_plan.id,
                "category": "skincare",
                "ml": {"mode": "v4_ranking", "rollout_mode": "none"},
            },
        )
        self._event(
            user=user_missing,
            plan=missing_plan,
            step=missing_step,
            event_type=RoadmapEvent.Type.STEP_GENERATED,
            created_at=base + timedelta(minutes=10),
            context={
                "plan_id": missing_plan.id,
                "step_id": missing_step.id,
                "step_index": 1,
                "category": "skincare",
                "product_type": "serum",
                "status": "recommended",
                "source": "rules",
                "ml": {"rollout_mode": "none"},
            },
        )

        payload = self._run_report_json(cohort_mode="fresh")

        overall_raw = payload["overall"]["raw"]
        overall_analysis = payload["overall"]["analysis"]
        next_step_only = payload["next_step_only"]["raw"]
        adherence = payload["recommended_product_adherence"]["raw"]
        self.assertEqual(overall_raw["plans_refreshed"], 3)
        self.assertEqual(overall_raw["steps_generated"], 3)
        self.assertEqual(overall_raw["steps_exposed"], 2)
        self.assertEqual(overall_raw["steps_clicked"], 1)
        self.assertEqual(overall_raw["steps_completed_after_generated"], 1)

        self.assertEqual(overall_analysis["plans_refreshed"], 2)
        self.assertEqual(overall_analysis["steps_generated"], 2)
        self.assertEqual(overall_analysis["excluded_missing_ml_meta_plans"], 1)
        self.assertEqual(overall_analysis["excluded_missing_ml_meta_steps"], 1)
        self.assertEqual(next_step_only["plans_refreshed"], 2)
        self.assertEqual(next_step_only["steps_generated"], 2)
        self.assertEqual(next_step_only["steps_exposed"], 2)
        self.assertEqual(next_step_only["steps_clicked"], 1)
        self.assertEqual(next_step_only["steps_completed_after_generated"], 1)
        self.assertEqual(adherence["next_step_with_recommendation"], 2)
        self.assertEqual(adherence["checkout_targeted_next_step"], 1)
        self.assertEqual(adherence["checkout_targeted_recommended_product"], 1)

        by_category = payload["breakdowns"]["by_category"]
        self.assertEqual(by_category["makeup"]["plans_refreshed"], 2)
        self.assertEqual(by_category["makeup"]["steps_generated"], 2)
        self.assertEqual(by_category["makeup"]["steps_exposed"], 2)
        self.assertEqual(by_category["makeup"]["steps_clicked"], 1)
        self.assertEqual(by_category["makeup"]["steps_completed_after_generated"], 1)
        next_step_by_category = payload["next_step_only"]["by_category"]
        self.assertEqual(next_step_by_category["makeup"]["plans_refreshed"], 2)
        self.assertEqual(next_step_by_category["makeup"]["steps_generated"], 2)
        self.assertEqual(next_step_by_category["makeup"]["steps_exposed"], 2)
        adherence_by_category = payload["recommended_product_adherence"]["by_category"]
        self.assertEqual(adherence_by_category["makeup"]["next_step_with_recommendation"], 2)
        self.assertEqual(adherence_by_category["makeup"]["checkout_targeted_next_step"], 1)
        self.assertEqual(adherence_by_category["makeup"]["checkout_targeted_recommended_product"], 1)

        by_step_index = payload["breakdowns"]["by_step_index"]
        self.assertEqual(by_step_index["step_1"]["steps_generated"], 1)
        self.assertEqual(by_step_index["step_2"]["steps_generated"], 1)

        by_product_type = payload["breakdowns"]["by_product_type"]
        self.assertEqual(by_product_type["foundation"]["steps_clicked"], 1)
        self.assertEqual(by_product_type["concealer"]["steps_exposed"], 1)

        by_ml_decision = payload["breakdowns"]["by_ml_decision"]
        self.assertEqual(by_ml_decision["model_used"]["plans_refreshed"], 1)
        self.assertEqual(by_ml_decision["disabled"]["steps_generated"], 1)
        self.assertEqual(by_ml_decision["missing_ml_meta"]["steps_generated"], 1)
        adherence_by_ml = payload["recommended_product_adherence"]["by_ml_decision"]
        self.assertEqual(adherence_by_ml["model_used"]["checkout_targeted_recommended_product"], 1)
        self.assertEqual(adherence_by_ml["disabled"]["next_step_with_recommendation"], 1)

        by_expose_source = payload["breakdowns"]["by_expose_source"]
        self.assertEqual(by_expose_source["roadmap_api"]["steps_generated"], 1)
        self.assertEqual(by_expose_source["offers"]["steps_generated"], 1)

    def test_category_filter_uses_plan_category_when_event_context_is_missing(self):
        base = timezone.now() - timedelta(days=1)
        user = self._user("gap_category_u1")
        plan, step = self._plan(
            user=user,
            category="makeup",
            decision="model_used",
            step_index=1,
            product_type="foundation",
        )

        self._event(
            user=user,
            plan=plan,
            step=None,
            event_type=RoadmapEvent.Type.PLAN_REFRESHED,
            created_at=base + timedelta(minutes=1),
            context={"plan_id": plan.id, "category": "makeup", "ml": {"decision": "model_used"}},
        )
        self._event(
            user=user,
            plan=plan,
            step=step,
            event_type=RoadmapEvent.Type.STEP_GENERATED,
            created_at=base + timedelta(minutes=2),
            context={
                "plan_id": plan.id,
                "step_id": step.id,
                "step_index": 1,
                "category": "makeup",
                "product_type": "foundation",
                "ml": {"decision": "model_used"},
            },
        )
        self._event(
            user=user,
            plan=plan,
            step=step,
            event_type=RoadmapEvent.Type.STEP_EXPOSED,
            created_at=base + timedelta(minutes=3),
            context={},
        )
        self._event(
            user=user,
            plan=plan,
            step=step,
            event_type=RoadmapEvent.Type.STEP_CLICKED,
            created_at=base + timedelta(minutes=4),
            context={},
        )
        self._event(
            user=user,
            plan=plan,
            step=step,
            event_type=RoadmapEvent.Type.STEP_COMPLETED,
            created_at=base + timedelta(minutes=5),
            context={},
        )

        payload = self._run_report_json(category="makeup")

        overall = payload["overall"]["raw"]
        self.assertEqual(overall["steps_generated"], 1)
        self.assertEqual(overall["steps_exposed"], 1)
        self.assertEqual(overall["steps_clicked"], 1)
        self.assertEqual(overall["steps_completed_after_generated"], 1)

    def test_repeated_generation_windows_are_ordered_by_event_time(self):
        base = timezone.now() - timedelta(days=1)
        user = self._user("gap_window_u1")
        plan, step = self._plan(
            user=user,
            category="haircare",
            decision="model_used",
            step_index=1,
            product_type="conditioner",
        )

        self._event(
            user=user,
            plan=plan,
            step=None,
            event_type=RoadmapEvent.Type.PLAN_REFRESHED,
            created_at=base + timedelta(minutes=1),
            context={"plan_id": plan.id, "category": "haircare", "ml": {"decision": "model_used"}},
        )
        self._event(
            user=user,
            plan=plan,
            step=step,
            event_type=RoadmapEvent.Type.STEP_GENERATED,
            created_at=base + timedelta(minutes=2),
            context={
                "plan_id": plan.id,
                "step_id": step.id,
                "step_index": 1,
                "category": "haircare",
                "product_type": "conditioner",
                "ml": {"decision": "model_used"},
            },
        )
        self._event(
            user=user,
            plan=plan,
            step=step,
            event_type=RoadmapEvent.Type.STEP_EXPOSED,
            created_at=base + timedelta(minutes=11),
            context={"sources": ["roadmap_api"]},
        )
        self._event(
            user=user,
            plan=plan,
            step=step,
            event_type=RoadmapEvent.Type.STEP_GENERATED,
            created_at=base + timedelta(minutes=10),
            context={
                "plan_id": plan.id,
                "step_id": step.id,
                "step_index": 1,
                "category": "haircare",
                "product_type": "conditioner",
                "ml": {"decision": "model_used"},
            },
        )
        self._event(
            user=user,
            plan=plan,
            step=step,
            event_type=RoadmapEvent.Type.STEP_EXPOSED,
            created_at=base + timedelta(minutes=3),
            context={"sources": ["roadmap_api"]},
        )
        self._event(
            user=user,
            plan=plan,
            step=step,
            event_type=RoadmapEvent.Type.STEP_COMPLETED,
            created_at=base + timedelta(minutes=12),
            context={},
        )

        payload = self._run_report_json(category="haircare")

        overall = payload["overall"]["raw"]
        self.assertEqual(overall["steps_generated"], 2)
        self.assertEqual(overall["steps_exposed"], 2)
        self.assertEqual(overall["steps_completed_after_generated"], 1)
        self.assertAlmostEqual(overall["generated_to_completed_rate"], 0.5, places=6)

    def test_next_step_only_tracks_plan_refresh_next_step_not_first_generated_step(self):
        base = timezone.now() - timedelta(days=1)
        user = self._user("gap_next_step_u1")
        plan = RoadmapPlan.objects.create(
            user=user,
            category="makeup",
            is_active=True,
            meta={
                "source": "roadmap_v1",
                "ml": {"decision": "fallback", "mode": "v4_ranking", "model_path": "/tmp/model.pkl"},
            },
        )
        step_1 = RoadmapStep.objects.create(
            plan=plan,
            step_index=1,
            product_type="primer",
            status=RoadmapStep.Status.RECOMMENDED,
        )
        step_2 = RoadmapStep.objects.create(
            plan=plan,
            step_index=2,
            product_type="foundation",
            status=RoadmapStep.Status.RECOMMENDED,
        )

        self._event(
            user=user,
            plan=plan,
            step=None,
            event_type=RoadmapEvent.Type.PLAN_REFRESHED,
            created_at=base + timedelta(minutes=1),
            context={
                "plan_id": plan.id,
                "category": "makeup",
                "next_step_id": step_2.id,
                "next_step_index": 2,
                "next_product_type": "foundation",
                "ml": {"decision": "fallback"},
            },
        )
        self._event(
            user=user,
            plan=plan,
            step=step_1,
            event_type=RoadmapEvent.Type.STEP_GENERATED,
            created_at=base + timedelta(minutes=2),
            context={
                "plan_id": plan.id,
                "step_id": step_1.id,
                "step_index": 1,
                "category": "makeup",
                "product_type": "primer",
                "recommended_product_id": 701,
                "has_recommendation": True,
                "ml": {"decision": "fallback"},
            },
        )
        self._event(
            user=user,
            plan=plan,
            step=step_2,
            event_type=RoadmapEvent.Type.STEP_GENERATED,
            created_at=base + timedelta(minutes=3),
            context={
                "plan_id": plan.id,
                "step_id": step_2.id,
                "step_index": 2,
                "category": "makeup",
                "product_type": "foundation",
                "recommended_product_id": 702,
                "has_recommendation": True,
                "ml": {"decision": "fallback"},
            },
        )
        self._event(
            user=user,
            plan=plan,
            step=step_2,
            event_type=RoadmapEvent.Type.STEP_EXPOSED,
            created_at=base + timedelta(minutes=4),
            context={"sources": ["roadmap_api"], "category": "makeup"},
        )
        self._event(
            user=user,
            plan=plan,
            step=step_2,
            event_type=RoadmapEvent.Type.STEP_CLICKED,
            created_at=base + timedelta(minutes=5),
            context={"category": "makeup"},
        )
        self._event(
            user=user,
            plan=plan,
            step=step_2,
            event_type=RoadmapEvent.Type.STEP_COMPLETED,
            created_at=base + timedelta(minutes=6),
            context={"category": "makeup", "matched_by": "product_type"},
        )

        payload = self._run_report_json(category="makeup")

        overall = payload["overall"]["raw"]
        self.assertEqual(overall["steps_generated"], 2)
        self.assertEqual(overall["steps_exposed"], 1)

        next_step_only = payload["next_step_only"]["raw"]
        self.assertEqual(next_step_only["plans_refreshed"], 1)
        self.assertEqual(next_step_only["steps_generated"], 1)
        self.assertEqual(next_step_only["steps_exposed"], 1)
        self.assertEqual(next_step_only["steps_clicked"], 1)
        self.assertEqual(next_step_only["steps_completed_after_exposure"], 1)
        self.assertEqual(next_step_only["steps_completed_after_generated"], 1)

        next_step_by_category = payload["next_step_only"]["by_category"]
        self.assertEqual(next_step_by_category["makeup"]["plans_refreshed"], 1)
        self.assertEqual(next_step_by_category["makeup"]["steps_generated"], 1)
        adherence = payload["recommended_product_adherence"]["raw"]
        self.assertEqual(adherence["next_step_with_recommendation"], 1)
        self.assertEqual(adherence["checkout_targeted_next_step"], 1)
        self.assertEqual(adherence["checkout_targeted_recommended_product"], 0)
        adherence_by_step = payload["recommended_product_adherence"]["by_step_bucket"]
        self.assertEqual(adherence_by_step["step_2_plus"]["next_step_with_recommendation"], 1)
        self.assertEqual(adherence_by_step["step_2_plus"]["checkout_targeted_next_step"], 1)
        self.assertEqual(adherence_by_step["step_2_plus"]["checkout_targeted_recommended_product"], 0)

    def test_same_timestamp_events_use_event_id_tie_break_for_completion_window(self):
        base = timezone.now() - timedelta(days=1)
        user = self._user("gap_same_ts_u1")
        plan, step = self._plan(
            user=user,
            category="haircare",
            decision="fallback",
            step_index=1,
            product_type="shampoo",
        )

        self._event(
            user=user,
            plan=plan,
            step=None,
            event_type=RoadmapEvent.Type.PLAN_REFRESHED,
            created_at=base,
            context={"plan_id": plan.id, "category": "haircare", "ml": {"decision": "fallback"}},
        )
        generated_first = self._event(
            user=user,
            plan=plan,
            step=step,
            event_type=RoadmapEvent.Type.STEP_GENERATED,
            created_at=base,
            context={
                "plan_id": plan.id,
                "step_id": step.id,
                "step_index": 1,
                "category": "haircare",
                "product_type": "shampoo",
                "ml": {"decision": "fallback"},
            },
        )
        exposed = self._event(
            user=user,
            plan=plan,
            step=step,
            event_type=RoadmapEvent.Type.STEP_EXPOSED,
            created_at=base,
            context={"sources": ["roadmap_api"]},
        )
        completed = self._event(
            user=user,
            plan=plan,
            step=step,
            event_type=RoadmapEvent.Type.STEP_COMPLETED,
            created_at=base,
            context={},
        )
        generated_second = self._event(
            user=user,
            plan=plan,
            step=step,
            event_type=RoadmapEvent.Type.STEP_GENERATED,
            created_at=base,
            context={
                "plan_id": plan.id,
                "step_id": step.id,
                "step_index": 1,
                "category": "haircare",
                "product_type": "shampoo",
                "ml": {"decision": "fallback"},
            },
        )

        self.assertLess(int(generated_first.id), int(exposed.id))
        self.assertLess(int(exposed.id), int(completed.id))
        self.assertLess(int(completed.id), int(generated_second.id))

        payload = self._run_report_json(category="haircare")

        overall = payload["overall"]["raw"]
        self.assertEqual(overall["steps_generated"], 2)
        self.assertEqual(overall["steps_exposed"], 1)
        self.assertEqual(overall["steps_completed_after_exposure"], 1)
        self.assertEqual(overall["steps_completed_after_generated"], 1)
