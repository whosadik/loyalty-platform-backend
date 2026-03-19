from decimal import Decimal

from django.contrib.auth import get_user_model
from django.core.cache import cache
from django.test import TestCase, override_settings

from catalog.models import Product
from roadmap_app.fragrance_slots import slot_of_fragrance
from roadmap_app.models import RoadmapPlan, RoadmapStep
from roadmap_app.services import match_completed_steps_for_purchase, refresh_roadmap


@override_settings(
    ROADMAP_NEXTSTEP_V4_ENABLED=False,
    ROADMAP_NEXTSTEP_V3_ENABLED=False,
)
class RoadmapFragranceSlotGuardTests(TestCase):
    def setUp(self):
        User = get_user_model()
        self.user = User.objects.create_user(username="roadmap_frag_guard_u1", password="pass12345")

        self.warm_day = Product.objects.create(
            name="Guard Warm Day",
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
            name="Guard Warm Evening",
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

    def _refresh_fragrance(self) -> RoadmapPlan:
        cache.clear()
        return refresh_roadmap(self.user, category="fragrance", post_ctx=None)

    def _warm_evening_step(self) -> RoadmapStep:
        plan = RoadmapPlan.objects.get(user=self.user, category="fragrance", is_active=True)
        return plan.steps.get(product_type="warm_evening")

    def test_fragrance_recommended_product_matches_slot(self):
        self._refresh_fragrance()

        step = self._warm_evening_step()
        self.assertEqual(step.status, RoadmapStep.Status.RECOMMENDED)
        self.assertIsNotNone(step.recommended_product_id)

        recommended = Product.objects.get(id=step.recommended_product_id)
        self.assertEqual(slot_of_fragrance(recommended.attrs or {}, raw_meta=recommended.raw_meta or {}), "warm_evening")
        self.assertNotIn(self.warm_day.id, list(step.suggestions or []))

    def test_fragrance_slot_step_stays_missing_when_no_slot_candidates(self):
        self.warm_evening.in_stock = False
        self.warm_evening.save(update_fields=["in_stock"])

        self._refresh_fragrance()

        step = self._warm_evening_step()
        self.assertEqual(step.status, RoadmapStep.Status.MISSING)
        self.assertIsNone(step.recommended_product_id)
        self.assertEqual(list(step.suggestions or []), [])

    def test_fragrance_wrong_slot_exact_sku_does_not_complete_step(self):
        plan = RoadmapPlan.objects.create(user=self.user, category="fragrance", is_active=True, meta={})
        step = RoadmapStep.objects.create(
            plan=plan,
            step_index=1,
            product_type="warm_evening",
            status=RoadmapStep.Status.RECOMMENDED,
            recommended_product=self.warm_day,
            suggestions=[self.warm_day.id],
            why=[],
            cadence=RoadmapStep.Cadence.OPTIONAL,
        )

        matches = match_completed_steps_for_purchase(
            self.user,
            {
                "categories": ["fragrance"],
                "product_types": [self.warm_day.product_type],
                "product_ids": [self.warm_day.id],
            },
        )

        self.assertEqual(matches, [])
        self.assertFalse(
            any(
                int(match["step"].id) == int(step.id) and match["matched_by"] == "recommended_product_id"
                for match in matches
            )
        )

    def test_fragrance_slot_purchase_completes_next_step(self):
        warm_evening_alt = Product.objects.create(
            name="Guard Warm Evening Alt",
            brand="F",
            price=Decimal("49.00"),
            category="fragrance",
            product_type="edp",
            concerns=[],
            attrs={"scent_family": "citrus", "notes": ["orange_blossom"], "intensity": "strong"},
            actives=[],
            flags=[],
            supported_skin_types=["normal"],
            strength="high",
            in_stock=True,
        )
        plan = RoadmapPlan.objects.create(user=self.user, category="fragrance", is_active=True, meta={})
        step = RoadmapStep.objects.create(
            plan=plan,
            step_index=1,
            product_type="warm_evening",
            status=RoadmapStep.Status.RECOMMENDED,
            recommended_product=self.warm_day,
            suggestions=[self.warm_day.id],
            why=[],
            cadence=RoadmapStep.Cadence.OPTIONAL,
        )

        matches = match_completed_steps_for_purchase(
            self.user,
            {
                "categories": ["fragrance"],
                "product_types": [warm_evening_alt.product_type],
                "product_ids": [warm_evening_alt.id],
            },
        )

        self.assertEqual(len(matches), 1)
        self.assertEqual(int(matches[0]["step"].id), int(step.id))
        self.assertEqual(matches[0]["matched_by"], "fragrance_slot")
        self.assertEqual((matches[0]["match_meta"] or {}).get("purchased_slot"), "warm_evening")
