from decimal import Decimal

from django.contrib.auth import get_user_model
from rest_framework.test import APITestCase

from catalog.models import Product
from users_app.models import CustomerProfile


class RoadmapSinglePlanApiTests(APITestCase):
    def setUp(self):
        User = get_user_model()
        self.user = User.objects.create_user(username="roadmap_single_user", password="pass12345")
        self.client.force_authenticate(self.user)
        CustomerProfile.objects.get_or_create(user=self.user)

        self._create_product("Cleanser Seed", "skincare", "cleanser")
        self._create_product("Serum Seed", "skincare", "serum")
        self._create_product("Moisturizer Seed", "skincare", "moisturizer")
        self._create_product("SPF Seed", "skincare", "spf")

        self._create_product("Shampoo Seed", "haircare", "shampoo")
        self._create_product("Conditioner Seed", "haircare", "conditioner")
        self._create_product("Hair Mask Seed", "haircare", "hair_mask")
        self._create_product("Hair Oil Seed", "haircare", "hair_oil")

    def _create_product(self, name: str, category: str, product_type: str):
        return Product.objects.create(
            name=name,
            brand="Test Brand",
            price=Decimal("10.00"),
            category=category,
            product_type=product_type,
            concerns=[],
            attrs={},
            actives=[],
            flags=[],
            supported_skin_types=["normal"],
            strength="low",
            in_stock=True,
            image_url=f"https://example.com/{product_type}.jpg",
            image_urls=[f"https://example.com/{product_type}.jpg"],
        )

    def test_get_without_category_creates_single_default_roadmap(self):
        response = self.client.get("/api/me/roadmap")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.data["category"], "skincare")
        self.assertGreaterEqual(len(response.data.get("steps") or []), 4)
        self.assertIsNotNone((response.data.get("summary") or {}).get("next_step"))

    def test_refresh_without_category_reuses_current_active_plan(self):
        first = self.client.post("/api/me/roadmap/refresh", {"category": "haircare"}, format="json")
        self.assertEqual(first.status_code, 200)

        second = self.client.post("/api/me/roadmap/refresh", {}, format="json")

        self.assertEqual(second.status_code, 200)
        self.assertEqual(second.data["category"], "haircare")
        self.assertEqual(second.data["id"], first.data["id"])
