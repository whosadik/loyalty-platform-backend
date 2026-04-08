from __future__ import annotations

from datetime import timedelta
from decimal import Decimal
from unittest.mock import patch

from django.contrib.auth import get_user_model
from django.core.management import call_command
from django.test import TestCase
from django.test.utils import override_settings
from django.utils import timezone

from catalog.models import Product
from offers.models import CampaignBudget, Offer, OfferAssignment
from offers.services import get_or_assign_next_offer
from roadmap_app.events import build_step_event_context, record_exposed_from_offer_assignment, record_roadmap_event
from roadmap_app.integrity import legacy_bad_fragrance_completion_details
from roadmap_app.models import RoadmapEvent, RoadmapPlan, RoadmapStep
from roadmap_app.services import (
    get_active_plan,
    match_completed_steps_for_purchase,
    patch_step_status,
    refresh_roadmap,
    update_roadmap_from_purchase,
)
from transactions.models import OwnedProduct
from users_app.models import CustomerProfile


class RoadmapRuntimeIntegrityTests(TestCase):
    def setUp(self):
        User = get_user_model()
        self.user = User.objects.create_user(username="roadmap_integrity_u1", password="pass12345")

    def _create_skincare_products(self) -> Product:
        serum = Product.objects.create(
            name="Integrity Serum",
            brand="B",
            price=Decimal("12.00"),
            category="skincare",
            product_type="serum",
            in_stock=True,
        )
        Product.objects.create(
            name="Integrity Cleanser",
            brand="B",
            price=Decimal("11.00"),
            category="skincare",
            product_type="cleanser",
            in_stock=True,
        )
        Product.objects.create(
            name="Integrity Moisturizer",
            brand="B",
            price=Decimal("13.00"),
            category="skincare",
            product_type="moisturizer",
            in_stock=True,
        )
        Product.objects.create(
            name="Integrity SPF",
            brand="B",
            price=Decimal("10.00"),
            category="skincare",
            product_type="spf",
            in_stock=True,
        )
        return serum

    def _campaign(self, name: str) -> CampaignBudget:
        return CampaignBudget.objects.create(
            name=name,
            weekly_limit=Decimal("1000.00"),
            weekly_spent=Decimal("0.00"),
            priority=100,
            is_active=True,
        )

    def _offer(self, campaign: CampaignBudget, name: str) -> Offer:
        return Offer.objects.create(
            name=name,
            offer_type=Offer.Type.DISCOUNT,
            value=Decimal("10.00"),
            estimated_cost=Decimal("2.00"),
            is_active=True,
            target_scope="product_type",
            cooldown_days=0,
            expires_in_days=7,
            allowed_categories=["skincare"],
            allowed_product_types=["serum", "cleanser", "moisturizer", "spf"],
            campaign=campaign,
        )

    def _set_hair_profile(self, *, hair_type: str, scalp_type: str, hair_thickness: str, concerns: list[str] | None = None) -> None:
        profile, _ = CustomerProfile.objects.get_or_create(user=self.user)
        profile.hair_profile = {
            "hair_type": hair_type,
            "scalp_type": scalp_type,
            "hair_thickness": hair_thickness,
            "concerns": list(concerns or []),
        }
        profile.save(update_fields=["hair_profile", "updated_at"])

    def _set_skin_profile(self, *, skin_type: str, goals: list[str] | None = None, avoid_flags: list[str] | None = None) -> None:
        profile, _ = CustomerProfile.objects.get_or_create(user=self.user)
        profile.skin_type = skin_type
        profile.goals = list(goals or [])
        profile.avoid_flags = list(avoid_flags or [])
        profile.save(update_fields=["skin_type", "goals", "avoid_flags", "updated_at"])

    def _create_haircare_products(self) -> dict[str, Product]:
        products: dict[str, Product] = {}
        for product_type in ["shampoo", "conditioner", "hair_mask", "hair_oil", "scalp_serum", "leave_in"]:
            products[product_type] = Product.objects.create(
                name=f"Integrity {product_type}",
                brand="B",
                price=Decimal("12.00"),
                category="haircare",
                product_type=product_type,
                in_stock=True,
            )
        return products

    def _create_skincare_tail_products(self) -> dict[str, Product]:
        products: dict[str, Product] = {}
        for product_type in ["cleanser", "serum", "moisturizer", "spf", "toner", "mask", "eye_cream", "essence"]:
            products[product_type] = Product.objects.create(
                name=f"Integrity {product_type}",
                brand="B",
                price=Decimal("13.00"),
                category="skincare",
                product_type=product_type,
                in_stock=True,
            )
        return products

    def test_update_roadmap_from_purchase_includes_next_step_identity(self):
        serum = self._create_skincare_products()

        updated = update_roadmap_from_purchase(
            self.user,
            {
                "categories": ["skincare"],
                "product_ids": [int(serum.id)],
            },
        )

        self.assertIsNotNone(updated)
        next_step = updated["next_missing_step"]
        roadmap_ctx = updated["roadmap_ctx"]
        self.assertIsNotNone(next_step)
        self.assertEqual(int(roadmap_ctx["step_id"]), int(next_step.id))
        self.assertEqual(int(roadmap_ctx["step_index"]), int(next_step.step_index))
        self.assertEqual(str(roadmap_ctx["next_product_type"]), str(next_step.product_type))

    def test_update_roadmap_from_purchase_advances_haircare_to_next_rule_step(self):
        shampoo = Product.objects.create(
            name="Integrity Hair Shampoo",
            brand="B",
            price=Decimal("11.00"),
            category="haircare",
            product_type="shampoo",
            in_stock=True,
        )
        conditioner = Product.objects.create(
            name="Integrity Hair Conditioner",
            brand="B",
            price=Decimal("12.00"),
            category="haircare",
            product_type="conditioner",
            in_stock=True,
        )
        Product.objects.create(
            name="Integrity Hair Mask",
            brand="B",
            price=Decimal("14.00"),
            category="haircare",
            product_type="hair_mask",
            in_stock=True,
        )

        now = timezone.now()
        OwnedProduct.objects.create(
            user=self.user,
            product=shampoo,
            quantity_total=1,
            is_active=True,
            last_acquired_at=now - timedelta(days=5),
        )
        OwnedProduct.objects.create(
            user=self.user,
            product=conditioner,
            quantity_total=1,
            is_active=True,
            last_acquired_at=now,
        )

        updated = update_roadmap_from_purchase(
            self.user,
            {
                "categories": ["haircare"],
                "product_ids": [int(conditioner.id)],
            },
        )

        self.assertIsNotNone(updated)
        next_step = updated["next_missing_step"]
        roadmap_ctx = updated["roadmap_ctx"]
        self.assertIsNotNone(next_step)
        self.assertEqual(str(next_step.product_type), "hair_mask")
        self.assertEqual(str(roadmap_ctx["next_product_type"]), "hair_mask")

    def test_refresh_roadmap_ignores_stale_haircare_owned_products(self):
        shampoo = Product.objects.create(
            name="Stale Hair Shampoo",
            brand="B",
            price=Decimal("11.00"),
            category="haircare",
            product_type="shampoo",
            in_stock=True,
        )
        Product.objects.create(
            name="Stale Hair Conditioner",
            brand="B",
            price=Decimal("12.00"),
            category="haircare",
            product_type="conditioner",
            in_stock=True,
        )
        Product.objects.create(
            name="Stale Hair Mask",
            brand="B",
            price=Decimal("14.00"),
            category="haircare",
            product_type="hair_mask",
            in_stock=True,
        )

        now = timezone.now()
        OwnedProduct.objects.create(
            user=self.user,
            product=shampoo,
            quantity_total=1,
            is_active=True,
            last_acquired_at=now - timedelta(days=90),
        )

        plan = refresh_roadmap(self.user, category="haircare", post_ctx=None)
        next_step = next((step for step in plan.steps.all() if step.status in {RoadmapStep.Status.MISSING, RoadmapStep.Status.RECOMMENDED}), None)
        ml_meta = (plan.meta or {}).get("ml") or {}

        self.assertIsNotNone(next_step)
        self.assertEqual(str(next_step.product_type), "shampoo")
        self.assertEqual(str(ml_meta.get("planned_target_product_type")), "shampoo")
        self.assertEqual(int(ml_meta.get("planned_target_step_index") or 0), 1)

    def test_refresh_roadmap_respects_future_finish_date_even_if_last_acquired_is_old(self):
        shampoo = Product.objects.create(
            name="Fresh Hair Shampoo",
            brand="B",
            price=Decimal("11.00"),
            category="haircare",
            product_type="shampoo",
            in_stock=True,
        )
        Product.objects.create(
            name="Fresh Hair Conditioner",
            brand="B",
            price=Decimal("12.00"),
            category="haircare",
            product_type="conditioner",
            in_stock=True,
        )

        now = timezone.now()
        OwnedProduct.objects.create(
            user=self.user,
            product=shampoo,
            quantity_total=1,
            is_active=True,
            last_acquired_at=now - timedelta(days=120),
            finish_date=timezone.localdate(now) + timedelta(days=5),
        )

        plan = refresh_roadmap(self.user, category="haircare", post_ctx=None)
        next_step = next((step for step in plan.steps.all() if step.status in {RoadmapStep.Status.MISSING, RoadmapStep.Status.RECOMMENDED}), None)
        ml_meta = (plan.meta or {}).get("ml") or {}

        self.assertIsNotNone(next_step)
        self.assertEqual(str(next_step.product_type), "conditioner")
        self.assertEqual(str(ml_meta.get("planned_target_product_type")), "conditioner")
        self.assertEqual(int(ml_meta.get("planned_target_step_index") or 0), 2)

    def test_get_or_assign_next_offer_copies_step_identity_into_reason(self):
        campaign = self._campaign("integrity_default")
        offer = self._offer(campaign, "Integrity Roadmap Offer")
        roadmap_ctx = {
            "category": "skincare",
            "plan_id": 42,
            "step_id": 77,
            "step_index": 2,
            "next_product_type": "serum",
        }

        with patch(
            "offers.services._select_offer",
            return_value=((offer, campaign), {"source": "patched"}),
        ), patch(
            "offers.services._pick_target_for_offer",
            return_value={
                "scope": "product_type",
                "value": "serum",
                "category": "skincare",
                "picked_via": "roadmap_shortcut",
            },
        ):
            assignment = get_or_assign_next_offer(
                user=self.user,
                now=timezone.now(),
                context_steps=None,
                post_ctx=None,
                roadmap_ctx=roadmap_ctx,
            )

        self.assertIsNotNone(assignment)
        roadmap_reason = (assignment.reason or {}).get("roadmap") or {}
        self.assertEqual(int(roadmap_reason["plan_id"]), 42)
        self.assertEqual(int(roadmap_reason["step_id"]), 77)
        self.assertEqual(int(roadmap_reason["step_index"]), 2)
        self.assertEqual(str(roadmap_reason["next_product_type"]), "serum")
        self.assertEqual(str(roadmap_reason.get("link_type") or ""), "direct_target")
        self.assertFalse(bool((assignment.reason or {}).get("roadmap_influence")))

    def test_get_or_assign_next_offer_separates_roadmap_routing_only_reason(self):
        campaign = self._campaign("integrity_default_routing_only")
        offer = self._offer(campaign, "Integrity Routing Only Offer")
        roadmap_ctx = {
            "category": "skincare",
            "plan_id": 52,
            "step_id": 88,
            "step_index": 3,
            "next_product_type": "moisturizer",
        }

        with patch(
            "offers.services._select_offer",
            return_value=((offer, campaign), {"source": "patched"}),
        ), patch(
            "offers.services._pick_target_for_offer",
            return_value={
                "scope": "category",
                "value": "skincare",
                "picked_via": "campaign_restriction_fallback",
            },
        ):
            assignment = get_or_assign_next_offer(
                user=self.user,
                now=timezone.now(),
                context_steps=None,
                post_ctx=None,
                roadmap_ctx=roadmap_ctx,
            )

        self.assertIsNotNone(assignment)
        self.assertFalse(bool((assignment.reason or {}).get("roadmap")))
        influence = (assignment.reason or {}).get("roadmap_influence") or {}
        self.assertEqual(str(influence.get("link_type") or ""), "routing_only")
        self.assertEqual(str(influence.get("category") or ""), "skincare")
        self.assertEqual(str(influence.get("next_product_type") or ""), "moisturizer")
        self.assertEqual(str(influence.get("final_scope") or ""), "category")
        self.assertEqual(str(influence.get("final_picked_via") or ""), "campaign_restriction_fallback")
        self.assertFalse(bool(influence.get("final_target_matches_step")))

    def test_get_or_assign_next_offer_without_next_step_does_not_persist_roadmap_reason(self):
        campaign = self._campaign("integrity_default_no_next_step")
        offer = self._offer(campaign, "Integrity No Next Step Offer")
        roadmap_ctx = {
            "category": "skincare",
            "plan_id": 61,
        }

        with patch(
            "offers.services._select_offer",
            return_value=((offer, campaign), {"source": "patched"}),
        ), patch(
            "offers.services._pick_target_for_offer",
            return_value={
                "scope": "category",
                "value": "skincare",
                "picked_via": "campaign_restriction_fallback",
            },
        ):
            assignment = get_or_assign_next_offer(
                user=self.user,
                now=timezone.now(),
                context_steps=None,
                post_ctx=None,
                roadmap_ctx=roadmap_ctx,
            )

        self.assertIsNotNone(assignment)
        self.assertFalse(bool((assignment.reason or {}).get("roadmap")))
        self.assertFalse(bool((assignment.reason or {}).get("roadmap_influence")))

    def test_cleanup_offers_sanitize_roadmap_normalizes_reason_and_deactivates_stale_shortcuts(self):
        campaign = self._campaign("integrity_cleanup")
        offer = self._offer(campaign, "Integrity Cleanup Offer")

        stale_plan = RoadmapPlan.objects.create(
            user=self.user,
            category="skincare",
            is_active=True,
            meta={},
        )
        stale_step = RoadmapStep.objects.create(
            plan=stale_plan,
            step_index=1,
            product_type="serum",
            status=RoadmapStep.Status.COMPLETED,
            suggestions=[],
            why=["picked via rules"],
        )
        stale_assignment = OfferAssignment.objects.create(
            user=self.user,
            offer=offer,
            reason={
                "roadmap": {
                    "plan_id": stale_plan.id,
                    "step_id": stale_step.id,
                    "step_index": stale_step.step_index,
                    "category": "skincare",
                    "next_product_type": "serum",
                }
            },
            target={
                "scope": "product_type",
                "value": "serum",
                "category": "skincare",
                "picked_via": "roadmap_shortcut",
            },
        )
        misleading_assignment = OfferAssignment.objects.create(
            user=self.user,
            offer=offer,
            reason={
                "roadmap": {
                    "plan_id": 75,
                    "step_id": 91,
                    "step_index": 2,
                    "category": "skincare",
                    "next_product_type": "moisturizer",
                }
            },
            target={
                "scope": "category",
                "value": "skincare",
                "picked_via": "campaign_restriction_fallback",
            },
        )

        call_command("cleanup_offers", sanitize_roadmap=True)

        stale_assignment.refresh_from_db()
        misleading_assignment.refresh_from_db()

        self.assertFalse(stale_assignment.is_active)
        self.assertFalse(bool((misleading_assignment.reason or {}).get("roadmap")))
        influence = (misleading_assignment.reason or {}).get("roadmap_influence") or {}
        self.assertEqual(str(influence.get("link_type") or ""), "routing_only")
        self.assertEqual(str(influence.get("reason_code") or ""), "campaign_or_fallback_target")

    def test_record_exposed_from_offer_assignment_prefers_explicit_step_index(self):
        plan = RoadmapPlan.objects.create(
            user=self.user,
            category="skincare",
            is_active=True,
            meta={},
        )
        step_1 = RoadmapStep.objects.create(
            plan=plan,
            step_index=1,
            product_type="cleanser",
            status=RoadmapStep.Status.RECOMMENDED,
        )
        step_2 = RoadmapStep.objects.create(
            plan=plan,
            step_index=2,
            product_type="serum",
            status=RoadmapStep.Status.RECOMMENDED,
        )
        campaign = self._campaign("integrity_exposed")
        offer = self._offer(campaign, "Integrity Exposed Offer")
        assignment = OfferAssignment.objects.create(
            user=self.user,
            offer=offer,
            reason={
                "roadmap": {
                    "plan_id": plan.id,
                    "category": "skincare",
                    "next_product_type": "serum",
                    "step_id": step_2.id,
                    "step_index": 2,
                }
            },
            target={
                "picked_via": "roadmap_shortcut",
                "scope": "product_type",
                "value": "serum",
                "category": "skincare",
            },
        )

        event, created = record_exposed_from_offer_assignment(assignment=assignment, request_id="integrity-1")

        self.assertTrue(created)
        self.assertIsNotNone(event)
        self.assertEqual(int(event.step_id), int(step_2.id))
        self.assertEqual(int(event.plan_id), int(plan.id))
        self.assertEqual(int((event.context or {}).get("step_index")), 2)
        self.assertNotEqual(int(event.step_id), int(step_1.id))

    @override_settings(ROADMAP_NEXTSTEP_V4_ENABLED=False, ROADMAP_NEXTSTEP_V3_ENABLED=False)
    def test_haircare_optional_tail_can_stop_after_skip(self):
        products = self._create_haircare_products()
        self._set_hair_profile(
            hair_type="straight",
            scalp_type="normal",
            hair_thickness="medium",
            concerns=[],
        )

        now = timezone.now()
        for product_type in ["shampoo", "conditioner", "hair_mask", "hair_oil"]:
            OwnedProduct.objects.create(
                user=self.user,
                product=products[product_type],
                quantity_total=1,
                is_active=True,
                last_acquired_at=now,
            )

        plan = refresh_roadmap(self.user, category="haircare", post_ctx=None)
        next_step = next(
            step for step in plan.steps.all()
            if step.status in {RoadmapStep.Status.MISSING, RoadmapStep.Status.RECOMMENDED}
        )
        self.assertEqual(str(next_step.product_type), "scalp_serum")

        patch_step_status(user=self.user, step_id=int(next_step.id), status=RoadmapStep.Status.SKIPPED)

        updated_plan = get_active_plan(self.user, category="haircare")
        self.assertIsNotNone(updated_plan)
        self.assertIsNone(
            next(
                (
                    step
                    for step in updated_plan.steps.all()
                    if step.status in {RoadmapStep.Status.MISSING, RoadmapStep.Status.RECOMMENDED}
                ),
                None,
            )
        )
        continuation = (updated_plan.meta or {}).get("continuation") or {}
        self.assertEqual(str(continuation.get("action") or ""), "__stop__")
        self.assertIn(
            str(continuation.get("reason") or ""),
            {"stopped_due_to_weak_tail_signal", "stopped_after_optional_tail"},
        )

    @override_settings(ROADMAP_NEXTSTEP_V4_ENABLED=False, ROADMAP_NEXTSTEP_V3_ENABLED=False)
    def test_skincare_optional_tail_can_stop_after_skip(self):
        products = self._create_skincare_tail_products()
        self._set_skin_profile(skin_type="normal", goals=["hydration"], avoid_flags=[])

        now = timezone.now()
        for product_type in ["cleanser", "serum", "moisturizer", "spf"]:
            OwnedProduct.objects.create(
                user=self.user,
                product=products[product_type],
                quantity_total=1,
                is_active=True,
                last_acquired_at=now,
            )

        plan = refresh_roadmap(self.user, category="skincare", post_ctx=None)
        next_step = next(
            step for step in plan.steps.all()
            if step.status in {RoadmapStep.Status.MISSING, RoadmapStep.Status.RECOMMENDED}
        )
        self.assertEqual(str(next_step.product_type), "toner")

        patch_step_status(user=self.user, step_id=int(next_step.id), status=RoadmapStep.Status.SKIPPED)

        updated_plan = get_active_plan(self.user, category="skincare")
        self.assertIsNotNone(updated_plan)
        self.assertIsNone(
            next(
                (
                    step
                    for step in updated_plan.steps.all()
                    if step.status in {RoadmapStep.Status.MISSING, RoadmapStep.Status.RECOMMENDED}
                ),
                None,
            )
        )
        continuation = (updated_plan.meta or {}).get("continuation") or {}
        self.assertEqual(str(continuation.get("action") or ""), "__stop__")
        self.assertIn(
            str(continuation.get("reason") or ""),
            {"stopped_due_to_weak_tail_signal", "stopped_after_optional_tail"},
        )

    @override_settings(ROADMAP_NEXTSTEP_V4_ENABLED=False, ROADMAP_NEXTSTEP_V3_ENABLED=False)
    def test_haircare_core_gap_still_continues_after_completed_purchase(self):
        products = self._create_haircare_products()
        self._set_hair_profile(
            hair_type="straight",
            scalp_type="normal",
            hair_thickness="medium",
            concerns=["repair"],
        )

        now = timezone.now()
        OwnedProduct.objects.create(
            user=self.user,
            product=products["shampoo"],
            quantity_total=1,
            is_active=True,
            last_acquired_at=now - timedelta(days=5),
        )
        initial_plan = refresh_roadmap(self.user, category="haircare", post_ctx=None)
        initial_next = next(
            step for step in initial_plan.steps.all()
            if step.status in {RoadmapStep.Status.MISSING, RoadmapStep.Status.RECOMMENDED}
        )
        self.assertEqual(str(initial_next.product_type), "conditioner")

        updated = update_roadmap_from_purchase(
            self.user,
            {"categories": ["haircare"], "product_ids": [int(products["conditioner"].id)]},
        )

        self.assertIsNotNone(updated)
        next_step = updated["next_missing_step"]
        self.assertIsNotNone(next_step)
        self.assertEqual(str(next_step.product_type), "hair_mask")
        self.assertIn("continued_due_to_core_gap", list(next_step.why or []))

    @override_settings(ROADMAP_NEXTSTEP_V4_ENABLED=False, ROADMAP_NEXTSTEP_V3_ENABLED=False)
    def test_fragrance_runtime_patch_scope_is_unchanged(self):
        warm_day = Product.objects.create(
            name="Runtime Warm Day",
            brand="B",
            price=Decimal("20.00"),
            category="fragrance",
            product_type="edp",
            in_stock=True,
            attrs={"scent_family": "citrus", "notes": ["bergamot"], "intensity": "soft"},
            raw_meta={"notes": ["bergamot"], "scent_family": "citrus", "intensity": "soft"},
        )
        Product.objects.create(
            name="Runtime Warm Evening",
            brand="B",
            price=Decimal("22.00"),
            category="fragrance",
            product_type="edp",
            in_stock=True,
            attrs={"scent_family": "amber", "notes": ["amber", "vanilla"], "intensity": "strong"},
            raw_meta={"notes": ["amber", "vanilla"], "scent_family": "amber", "intensity": "strong"},
        )

        updated = update_roadmap_from_purchase(
            self.user,
            {"categories": ["fragrance"], "product_ids": [int(warm_day.id)]},
        )

        self.assertIsNotNone(updated)
        next_step = updated["next_missing_step"]
        self.assertIsNotNone(next_step)
        self.assertEqual(str(next_step.product_type), "warm_evening")
        continuation = ((updated["plan"].meta or {}).get("continuation") or {})
        self.assertEqual(continuation, {})

    def test_current_fragrance_runtime_completion_does_not_create_bad_legacy_exact_artifact(self):
        warm_day = Product.objects.create(
            name="Integrity Fresh Warm Day",
            brand="B",
            price=Decimal("24.00"),
            category="fragrance",
            product_type="edp",
            in_stock=True,
            attrs={"scent_family": "citrus", "notes": ["bergamot"], "intensity": "soft"},
            raw_meta={"notes": ["bergamot"], "scent_family": "citrus", "intensity": "soft"},
        )
        plan = RoadmapPlan.objects.create(
            user=self.user,
            category="fragrance",
            is_active=True,
            meta={},
        )
        step = RoadmapStep.objects.create(
            plan=plan,
            step_index=1,
            product_type="warm_day",
            status=RoadmapStep.Status.RECOMMENDED,
            recommended_product=warm_day,
            suggestions=[warm_day.id],
            why=["picked via rules"],
        )

        matches = match_completed_steps_for_purchase(
            self.user,
            {"categories": ["fragrance"], "product_ids": [int(warm_day.id)]},
        )

        self.assertEqual(len(matches), 1)
        self.assertEqual(str(matches[0]["matched_by"]), "recommended_product_id")

        record_roadmap_event(
            user=self.user,
            event_type=RoadmapEvent.Type.STEP_COMPLETED,
            plan=plan,
            step=step,
            context=build_step_event_context(
                category="fragrance",
                step=step,
                extra={
                    "matched_by": matches[0]["matched_by"],
                    "match_meta": matches[0]["match_meta"],
                },
            ),
        )

        counts = legacy_bad_fragrance_completion_details(recent_days=30)
        self.assertEqual(int(counts["bad_fragrance_completed_exact_match_count"]), 0)
        self.assertEqual(int(counts["step_state_drift_count"]), 0)
