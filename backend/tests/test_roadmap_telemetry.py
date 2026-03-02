from decimal import Decimal

from django.contrib.auth import get_user_model
from rest_framework.test import APITestCase

from catalog.models import Product
from loyalty.models import LoyaltyAccount, Tier
from roadmap_app.models import RoadmapEvent
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
