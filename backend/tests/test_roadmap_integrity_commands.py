from decimal import Decimal
from io import StringIO

from django.contrib.auth import get_user_model
from django.core.management import call_command
from django.test import TestCase

from catalog.models import Product
from roadmap_app.models import RoadmapEvent, RoadmapPlan, RoadmapStep


class RoadmapIntegrityCommandTests(TestCase):
    def setUp(self):
        User = get_user_model()
        self.user = User.objects.create_user(username="roadmap_integrity_u1", password="pass12345")

        self.warm_day = Product.objects.create(
            name="Integrity Warm Day",
            brand="F",
            price=Decimal("40.00"),
            category="fragrance",
            product_type="edp",
            concerns=[],
            attrs={"scent_family": "citrus", "notes": ["bergamot"], "intensity": "soft"},
            actives=[],
            flags=[],
            supported_skin_types=["normal"],
            strength="medium",
            in_stock=True,
        )
        self.warm_evening = Product.objects.create(
            name="Integrity Warm Evening",
            brand="F",
            price=Decimal("45.00"),
            category="fragrance",
            product_type="edp",
            concerns=[],
            attrs={"scent_family": "citrus", "notes": ["neroli"], "intensity": "strong"},
            actives=[],
            flags=[],
            supported_skin_types=["normal"],
            strength="high",
            in_stock=True,
        )

    def _create_mismatched_plan(self) -> tuple[RoadmapPlan, RoadmapStep]:
        plan = RoadmapPlan.objects.create(user=self.user, category="fragrance", is_active=True, meta={})
        step = RoadmapStep.objects.create(
            plan=plan,
            step_index=1,
            product_type="warm_evening",
            status=RoadmapStep.Status.RECOMMENDED,
            recommended_product=self.warm_day,
            suggestions=[self.warm_day.id],
            score=0.7,
            why=[],
            cadence=RoadmapStep.Cadence.OPTIONAL,
        )
        return plan, step

    @staticmethod
    def _parse_output(buffer: StringIO) -> dict[str, str]:
        parsed: dict[str, str] = {}
        for raw_line in buffer.getvalue().splitlines():
            line = str(raw_line or "").strip()
            if not line or "=" not in line:
                continue
            key, value = line.split("=", 1)
            parsed[key.strip()] = value.strip()
        return parsed

    def test_fix_command_dry_run_does_not_change_data(self):
        plan, step = self._create_mismatched_plan()
        out = StringIO()

        call_command(
            "fix_fragrance_roadmap_slot_integrity",
            dry_run=True,
            only_mismatched=True,
            batch_size=1,
            plan_ids=[plan.id],
            stdout=out,
        )

        step.refresh_from_db()
        self.assertEqual(step.recommended_product_id, self.warm_day.id)
        self.assertEqual(step.status, RoadmapStep.Status.RECOMMENDED)
        parsed = self._parse_output(out)
        self.assertEqual(parsed["scanned_plans"], "1")
        self.assertEqual(parsed["touched_plans"], "1")
        self.assertEqual(parsed["fixed_mismatches"], "1")
        self.assertEqual(parsed["remaining_mismatches"], "1")

    def test_fix_command_repairs_wrong_slot_recommendation(self):
        plan, step = self._create_mismatched_plan()

        call_command(
            "fix_fragrance_roadmap_slot_integrity",
            only_mismatched=True,
            batch_size=1,
            plan_ids=[plan.id],
            stdout=StringIO(),
        )

        step.refresh_from_db()
        self.assertIsNone(step.recommended_product_id)
        self.assertEqual(step.status, RoadmapStep.Status.MISSING)
        self.assertEqual(list(step.suggestions or []), [])
        self.assertIsNone(step.score)

    def test_runtime_integrity_report_is_clean_after_repair(self):
        plan, _step = self._create_mismatched_plan()
        call_command(
            "fix_fragrance_roadmap_slot_integrity",
            only_mismatched=True,
            batch_size=1,
            plan_ids=[plan.id],
            stdout=StringIO(),
        )

        out = StringIO()
        call_command("report_roadmap_runtime_integrity", stdout=out)

        parsed = self._parse_output(out)
        self.assertEqual(parsed["active_fragrance_next_steps_total"], "1")
        self.assertEqual(parsed["active_fragrance_slot_mismatch_count"], "0")

    def test_legacy_bad_completion_report_counts_corrupted_exact_match(self):
        plan, step = self._create_mismatched_plan()
        RoadmapEvent.objects.create(
            user=self.user,
            plan=plan,
            step=step,
            event_type=RoadmapEvent.Type.STEP_COMPLETED,
            context={
                "matched_by": "recommended_product_id",
                "recommended_product_id": self.warm_day.id,
                "purchased_product_id": self.warm_day.id,
            },
        )

        out = StringIO()
        call_command("report_roadmap_legacy_label_noise", stdout=out)

        parsed = self._parse_output(out)
        self.assertEqual(parsed["bad_fragrance_completed_exact_match_count"], "1")
        self.assertEqual(parsed["bad_fragrance_completed_exact_match_recent_30d"], "1")
        self.assertEqual(parsed["affected_users_count"], "1")
        self.assertEqual(parsed["affected_plans_count"], "1")
