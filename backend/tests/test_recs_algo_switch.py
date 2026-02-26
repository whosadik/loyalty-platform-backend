from decimal import Decimal

from django.contrib.auth import get_user_model
from django.test import override_settings
from rest_framework.test import APITestCase

from catalog.models import Product
from transactions.models import OwnedProduct
from users_app.models import CustomerProfile


class RecsAlgoSwitchTests(APITestCase):
    def setUp(self):
        User = get_user_model()
        self.user = User.objects.create_user(username="algo_user", password="pass12345")
        self.client.force_authenticate(self.user)
        CustomerProfile.objects.get_or_create(user=self.user)

        self.owned = Product.objects.create(
            name="Owned Lipstick",
            brand="B1",
            price=Decimal("9.99"),
            category="makeup",
            product_type="lipstick",
            concerns=[],
            attrs={"finish": "matte"},
            actives=[],
            flags=[],
            supported_skin_types=[],
            strength="low",
            in_stock=True,
        )
        self.candidate = Product.objects.create(
            name="Candidate Mascara",
            brand="B2",
            price=Decimal("10.99"),
            category="makeup",
            product_type="mascara",
            concerns=[],
            attrs={"waterproof": True},
            actives=[],
            flags=[],
            supported_skin_types=[],
            strength="low",
            in_stock=True,
        )
        OwnedProduct.objects.create(
            user=self.user,
            product=self.owned,
            quantity_total=1,
            is_active=True,
        )

    @override_settings(RECS_RERANKER_MODEL_PATH="Z:/not_existing/model.pkl")
    def test_me_recommendations_reranker_fallback(self):
        r = self.client.get("/api/me/recommendations?category=makeup&limit=5&algo=reranker")
        self.assertEqual(r.status_code, 200)
        self.assertIn("query", r.data)
        self.assertIn("results", r.data)
        self.assertEqual(r.data["query"]["algo_requested"], "reranker")
        self.assertTrue(str(r.data["query"]["algo_used"]).startswith("cooc_fallback"))

    @override_settings(RECS_RERANKER_MODEL_PATH="Z:/not_existing/model.pkl")
    def test_home_recommendations_reranker_fallback(self):
        r = self.client.get("/api/me/recommendations/home?limit=5&algo=reranker")
        self.assertEqual(r.status_code, 200)
        self.assertIn("query", r.data)
        self.assertEqual(r.data["query"]["algo_requested"], "reranker")
        self.assertTrue(str(r.data["query"]["for_you_algo_used"]).startswith("cooc_fallback"))
