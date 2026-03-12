from decimal import Decimal

from django.contrib.auth import get_user_model
from rest_framework.test import APITestCase

from catalog.models import Product
from loyalty.models import LoyaltyAccount, Tier
from users_app.models import CustomerProfile


class RoadmapStepPresentationApiTests(APITestCase):
    def setUp(self):
        user_model = get_user_model()
        self.user = user_model.objects.create_user(
            username="roadmap_presentation_u1",
            password="pass12345",
        )
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

        self.shampoo = Product.objects.create(
            name="Presentation Shampoo",
            brand="Glow Lab",
            price=Decimal("8.00"),
            category="haircare",
            product_type="shampoo",
            image_url="https://example.com/shampoo.jpg",
            image_urls=["https://example.com/shampoo.jpg"],
            concerns=[],
            attrs={},
            actives=[],
            flags=[],
            supported_skin_types=["normal"],
            strength="low",
            in_stock=True,
        )
        Product.objects.create(
            name="Presentation Conditioner",
            brand="Glow Lab",
            price=Decimal("9.00"),
            category="haircare",
            product_type="conditioner",
            image_url="https://example.com/conditioner.jpg",
            image_urls=["https://example.com/conditioner.jpg"],
            concerns=[],
            attrs={},
            actives=[],
            flags=[],
            supported_skin_types=["normal"],
            strength="low",
            in_stock=True,
        )
        Product.objects.create(
            name="Presentation Hair Mask",
            brand="Glow Lab",
            price=Decimal("11.00"),
            category="haircare",
            product_type="hair_mask",
            image_url="https://example.com/hair-mask.jpg",
            image_urls=["https://example.com/hair-mask.jpg"],
            concerns=[],
            attrs={},
            actives=[],
            flags=[],
            supported_skin_types=["normal"],
            strength="low",
            in_stock=True,
        )
        Product.objects.create(
            name="Presentation Hair Oil",
            brand="Glow Lab",
            price=Decimal("10.00"),
            category="haircare",
            product_type="hair_oil",
            image_url="https://example.com/hair-oil.jpg",
            image_urls=["https://example.com/hair-oil.jpg"],
            concerns=[],
            attrs={},
            actives=[],
            flags=[],
            supported_skin_types=["normal"],
            strength="low",
            in_stock=True,
        )

    def test_roadmap_steps_include_ui_ready_presentation(self):
        checkout = self.client.post(
            "/api/checkout",
            {"channel": "offline", "items": [{"product": self.shampoo.id, "quantity": 1}]},
            format="json",
        )
        self.assertEqual(checkout.status_code, 201)

        roadmap = self.client.get("/api/me/roadmap?category=haircare")
        self.assertEqual(roadmap.status_code, 200)

        steps = roadmap.data.get("steps") or []
        self.assertTrue(steps)

        conditioner_step = next((step for step in steps if step.get("product_type") == "conditioner"), None)
        self.assertIsNotNone(conditioner_step)
        presentation = conditioner_step.get("presentation") or {}
        self.assertEqual(presentation.get("title"), conditioner_step.get("title"))
        self.assertEqual(presentation.get("description"), conditioner_step.get("description"))
        self.assertEqual(presentation.get("points"), 110)
        self.assertTrue(str(presentation.get("why") or "").strip())
        self.assertTrue(str(presentation.get("improves") or "").strip())
        self.assertTrue(str(presentation.get("benefit") or "").strip())

        summary_next_step = (roadmap.data.get("summary") or {}).get("next_step") or {}
        self.assertIn("presentation", summary_next_step)
        self.assertEqual(summary_next_step["presentation"]["title"], summary_next_step.get("title"))

    def test_checkout_response_next_step_includes_ui_ready_presentation(self):
        checkout = self.client.post(
            "/api/checkout",
            {"channel": "offline", "items": [{"product": self.shampoo.id, "quantity": 1}]},
            format="json",
        )
        self.assertEqual(checkout.status_code, 201)

        next_step = checkout.data.get("next_roadmap_step") or {}
        presentation = next_step.get("presentation") or {}
        self.assertEqual(presentation.get("title"), next_step.get("title"))
        self.assertEqual(presentation.get("description"), next_step.get("description"))
        self.assertTrue(isinstance(presentation.get("points"), int))
