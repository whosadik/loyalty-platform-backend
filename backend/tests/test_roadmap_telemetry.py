from decimal import Decimal

from django.contrib.auth import get_user_model
from rest_framework.test import APITestCase

from catalog.models import Product
from loyalty.models import LoyaltyAccount, Tier
from offers.models import CampaignBudget, Offer, OfferEvent
from roadmap_app.models import RoadmapEvent, RoadmapPlan, RoadmapStep
from users_app.models import CustomerProfile


class RoadmapTelemetryTests(APITestCase):
    def setUp(self):
        User = get_user_model()
        self.user = User.objects.create_user(username="roadmap_tele_u1", password="pass12345")
        self.client.force_authenticate(self.user)

        CustomerProfile.objects.get_or_create(user=self.user)
        bronze, _ = Tier.objects.get_or_create(
            name="Bronze",
            defaults={"threshold_spend_90d": 0, "points_rate": 1.0},
        )
        LoyaltyAccount.objects.get_or_create(
            user=self.user,
            defaults={"tier": bronze, "points_balance": 0},
        )

        self.p_shampoo = Product.objects.create(
            name="Telemetry Shampoo",
            brand="B",
            price=Decimal("8.00"),
            category="haircare",
            product_type="shampoo",
            concerns=[],
            attrs={},
            actives=[],
            flags=[],
            supported_skin_types=["normal"],
            strength="low",
            in_stock=True,
        )
        self.p_conditioner = Product.objects.create(
            name="Telemetry Conditioner",
            brand="B",
            price=Decimal("9.00"),
            category="haircare",
            product_type="conditioner",
            concerns=[],
            attrs={},
            actives=[],
            flags=[],
            supported_skin_types=["normal"],
            strength="low",
            in_stock=True,
        )
        Product.objects.create(
            name="Telemetry Hair Mask",
            brand="B",
            price=Decimal("11.00"),
            category="haircare",
            product_type="hair_mask",
            concerns=[],
            attrs={},
            actives=[],
            flags=[],
            supported_skin_types=["normal"],
            strength="low",
            in_stock=True,
        )
        Product.objects.create(
            name="Telemetry Hair Oil",
            brand="B",
            price=Decimal("10.00"),
            category="haircare",
            product_type="hair_oil",
            concerns=[],
            attrs={},
            actives=[],
            flags=[],
            supported_skin_types=["normal"],
            strength="low",
            in_stock=True,
        )

    def _checkout(self, product_id: int):
        return self.client.post(
            "/api/checkout",
            {"channel": "offline", "items": [{"product": product_id, "quantity": 1}]},
            format="json",
        )

    def test_roadmap_exposed_skipped_and_clicked_events(self):
        c = self._checkout(self.p_shampoo.id)
        self.assertEqual(c.status_code, 201)

        roadmap = self.client.get("/api/me/roadmap?category=haircare")
        self.assertEqual(roadmap.status_code, 200)
        next_step = (roadmap.data.get("summary") or {}).get("next_step") or {}
        step_id = int(next_step["id"])

        exposed = RoadmapEvent.objects.filter(
            user=self.user,
            step_id=step_id,
            event_type=RoadmapEvent.Type.STEP_EXPOSED,
        ).first()
        self.assertIsNotNone(exposed)
        self.assertIn("category", exposed.context)
        self.assertIn("step_index", exposed.context)
        self.assertIn("product_type", exposed.context)

        patch = self.client.patch(
            f"/api/me/roadmap/steps/{step_id}",
            {"status": "skipped"},
            format="json",
        )
        self.assertEqual(patch.status_code, 200)
        self.assertTrue(
            RoadmapEvent.objects.filter(
                user=self.user,
                step_id=step_id,
                event_type=RoadmapEvent.Type.STEP_SKIPPED,
            ).exists()
        )

        click = self.client.post(f"/api/me/roadmap/steps/{step_id}/click", format="json")
        self.assertEqual(click.status_code, 200)
        self.assertTrue(
            RoadmapEvent.objects.filter(
                user=self.user,
                step_id=step_id,
                event_type=RoadmapEvent.Type.STEP_CLICKED,
            ).exists()
        )

    def test_checkout_logs_roadmap_step_completed(self):
        first = self._checkout(self.p_shampoo.id)
        self.assertEqual(first.status_code, 201)

        roadmap = self.client.get("/api/me/roadmap?category=haircare")
        self.assertEqual(roadmap.status_code, 200)
        next_step = (roadmap.data.get("summary") or {}).get("next_step") or {}
        step_id = int(next_step["id"])

        second = self._checkout(self.p_conditioner.id)
        self.assertEqual(second.status_code, 201)
        txn_id = int(second.data["transaction_id"])

        event = RoadmapEvent.objects.filter(
            user=self.user,
            step_id=step_id,
            event_type=RoadmapEvent.Type.STEP_COMPLETED,
        ).order_by("-id").first()
        self.assertIsNotNone(event)
        self.assertEqual(int(event.context.get("transaction_id")), txn_id)
        self.assertIn(event.context.get("matched_by"), {"product_type", "recommended_product_id"})

    def test_roadmap_exposed_daily_dedup_on_get(self):
        first = self._checkout(self.p_shampoo.id)
        self.assertEqual(first.status_code, 201)

        r1 = self.client.get("/api/me/roadmap?category=haircare")
        r2 = self.client.get("/api/me/roadmap?category=haircare")
        self.assertEqual(r1.status_code, 200)
        self.assertEqual(r2.status_code, 200)

        next_step = (r1.data.get("summary") or {}).get("next_step") or {}
        step_id = int(next_step["id"])
        self.assertEqual(
            RoadmapEvent.objects.filter(
                user=self.user,
                step_id=step_id,
                event_type=RoadmapEvent.Type.STEP_EXPOSED,
            ).count(),
            1,
        )

    def test_offer_exposed_writes_roadmap_exposed_when_roadmap_shortcut(self):
        default_campaign, _ = CampaignBudget.objects.get_or_create(
            name="default",
            defaults={
                "weekly_limit": Decimal("1000.00"),
                "weekly_spent": Decimal("0.00"),
                "priority": 100,
                "is_active": True,
            },
        )
        Offer.objects.create(
            name="Telemetry Haircare Roadmap Offer",
            offer_type=Offer.Type.DISCOUNT,
            value=Decimal("10.00"),
            estimated_cost=Decimal("2.00"),
            is_active=True,
            target_scope="product_type",
            cooldown_days=0,
            expires_in_days=7,
            allowed_categories=["haircare"],
            allowed_product_types=["conditioner", "hair_mask", "hair_oil"],
            campaign=default_campaign,
        )

        checkout = self._checkout(self.p_shampoo.id)
        self.assertEqual(checkout.status_code, 201)
        checkout_next_offer = checkout.data.get("next_offer") or {}
        assignment_id = int(checkout_next_offer.get("assignment_id"))

        plan = RoadmapPlan.objects.get(user=self.user, category="haircare", is_active=True)
        next_step = (
            RoadmapStep.objects.filter(
                plan=plan,
                status__in=[RoadmapStep.Status.MISSING, RoadmapStep.Status.RECOMMENDED],
            )
            .order_by("step_index")
            .first()
        )
        self.assertIsNotNone(next_step)
        self.assertEqual(
            RoadmapEvent.objects.filter(
                user=self.user,
                step=next_step,
                event_type=RoadmapEvent.Type.STEP_EXPOSED,
            ).count(),
            0,
        )

        r1 = self.client.get("/api/me/next-offer")
        self.assertEqual(r1.status_code, 200)
        target = r1.data.get("target") or {}
        self.assertTrue(str(target.get("picked_via") or "").startswith("roadmap_shortcut"))

        self.assertEqual(
            OfferEvent.objects.filter(
                assignment_id=assignment_id,
                event_type=OfferEvent.Type.EXPOSED,
            ).count(),
            1,
        )
        self.assertEqual(
            RoadmapEvent.objects.filter(
                user=self.user,
                step=next_step,
                event_type=RoadmapEvent.Type.STEP_EXPOSED,
            ).count(),
            1,
        )

        r2 = self.client.get("/api/me/next-offer")
        self.assertEqual(r2.status_code, 200)
        self.assertEqual(
            OfferEvent.objects.filter(
                assignment_id=assignment_id,
                event_type=OfferEvent.Type.EXPOSED,
            ).count(),
            1,
        )
        self.assertEqual(
            RoadmapEvent.objects.filter(
                user=self.user,
                step=next_step,
                event_type=RoadmapEvent.Type.STEP_EXPOSED,
            ).count(),
            1,
        )
