from __future__ import annotations

from datetime import timedelta
from decimal import Decimal
from unittest.mock import patch

from django.contrib.auth import get_user_model
from django.test import TestCase
from django.utils import timezone

from catalog.models import Product
from offers.models import CampaignBudget, Offer, OfferAssignment
from offers.services import get_or_assign_next_offer
from roadmap_app.events import record_exposed_from_offer_assignment
from roadmap_app.models import RoadmapPlan, RoadmapStep
from roadmap_app.services import update_roadmap_from_purchase
from transactions.models import OwnedProduct


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
