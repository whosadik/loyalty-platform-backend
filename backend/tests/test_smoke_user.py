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
        CustomerProfile.objects.get_or_create(user=self.user)

        # products for checkout/routine/recs
        Product.objects.create(
            name="Cleanser 1", brand="B", price=Decimal("9.99"),
            category="skincare", product_type="cleanser",
            concerns=[], attrs={}, actives=[], flags=[],
            supported_skin_types=["sensitive"], strength="low", in_stock=True,
        )
        Product.objects.create(
            name="SPF 1", brand="B", price=Decimal("9.99"),
            category="skincare", product_type="spf",
            concerns=[], attrs={}, actives=[], flags=[],
            supported_skin_types=["sensitive"], strength="low", in_stock=True,
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
