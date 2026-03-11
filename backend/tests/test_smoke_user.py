from decimal import Decimal
from django.contrib.auth import get_user_model
from rest_framework.test import APITestCase

from catalog.models import Product
from users_app.models import CustomerProfile
from loyalty.models import Tier, LoyaltyAccount


class UserSmokeTests(APITestCase):
    def setUp(self):
        User = get_user_model()
        self.user = User.objects.create_user(username="u1", password="pass12345")
        self.client.force_authenticate(self.user)

        bronze, _ = Tier.objects.get_or_create(name="Bronze", defaults={"threshold_spend_90d": 0, "points_rate": 1.0})
        LoyaltyAccount.objects.get_or_create(user=self.user, defaults={"tier": bronze, "points_balance": 0})
        profile, _ = CustomerProfile.objects.get_or_create(user=self.user)
        profile.skin_type = "normal"
        profile.save(update_fields=["skin_type"])

        # products for checkout/routine/recs
        Product.objects.create(
            name="Cleanser 1", brand="B", price=Decimal("0.01"),
            category="skincare", product_type="cleanser",
            concerns=[], attrs={}, actives=[], flags=[],
            supported_skin_types=["normal"], strength="low", in_stock=True,
            image_url="https://example.com/cleanser-1.jpg",
            image_urls=["https://example.com/cleanser-1.jpg"],
            application_text="Наносите на влажную кожу мягкими движениями.",
        )
        Product.objects.create(
            name="SPF 1", brand="B", price=Decimal("0.02"),
            category="skincare", product_type="spf",
            concerns=[], attrs={}, actives=[], flags=[],
            supported_skin_types=["normal"], strength="low", in_stock=True,
        )
        self.serum_current = Product.objects.create(
            name="Serum Conflict", brand="B", price=Decimal("0.03"),
            category="skincare", product_type="serum",
            concerns=[], attrs={}, actives=["vitamin_c"], flags=[],
            supported_skin_types=["normal"], strength="medium", in_stock=True,
            image_url="https://example.com/serum-conflict.jpg",
        )
        self.serum_alternative = Product.objects.create(
            name="Serum Calm", brand="B", price=Decimal("0.04"),
            category="skincare", product_type="serum",
            concerns=[], attrs={}, actives=["niacinamide"], flags=[],
            supported_skin_types=["normal"], strength="low", in_stock=True,
            image_url="https://example.com/serum-calm.jpg",
        )
        self.moisturizer_conflict = Product.objects.create(
            name="Moisturizer AHA", brand="B", price=Decimal("0.05"),
            category="skincare", product_type="moisturizer",
            concerns=[], attrs={}, actives=["aha"], flags=[],
            supported_skin_types=["normal"], strength="medium", in_stock=True,
        )

    def test_me_loyalty(self):
        r = self.client.get("/api/me/loyalty")
        self.assertEqual(r.status_code, 200)
        self.assertIn("tier", r.data)
        self.assertIn("points_balance", r.data)

    def test_routine_generate(self):
        r = self.client.post("/api/routine/generate", {"skin_type": "sensitive"}, format="json")
        self.assertEqual(r.status_code, 200)
        self.assertIn("am", r.data)
        self.assertIn("pm", r.data)
        self.assertIn("notes", r.data)
        cleanser_step = r.data["am"][0]
        self.assertEqual(cleanser_step["display_step"], "Очищение")
        self.assertEqual(cleanser_step["duration_label"], "1-2 мин")
        self.assertEqual(cleanser_step["product"]["image_url"], "https://example.com/cleanser-1.jpg")
        self.assertEqual(
            cleanser_step["product"]["application_text"],
            "Наносите на влажную кожу мягкими движениями.",
        )

    def test_routine_validate_returns_enriched_alternatives(self):
        r = self.client.post(
            "/api/routine/validate",
            {
                "am": [],
                "pm": [
                    {"step": "serum", "product_id": self.serum_current.id},
                    {"step": "moisturizer", "product_id": self.moisturizer_conflict.id},
                ],
            },
            format="json",
        )
        self.assertEqual(r.status_code, 200)
        self.assertFalse(r.data["is_valid"])
        self.assertTrue(r.data["conflicts"])

        serum_suggestion = next(
            item for item in r.data["suggestions"]
            if item.get("step") == "serum"
        )
        self.assertEqual(serum_suggestion["display_step"], "Сыворотка")
        self.assertEqual(serum_suggestion["current_product"]["id"], self.serum_current.id)
        self.assertTrue(
            any(
                product["id"] == self.serum_alternative.id
                for product in serum_suggestion["alternative_products"]
            )
        )
