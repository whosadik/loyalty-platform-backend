from decimal import Decimal

from django.contrib.auth import get_user_model
from django.test import override_settings
from rest_framework.test import APITestCase

from catalog.models import Product
from recs_analytics.models import RecommendationEvent
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
            image_url="https://example.com/owned.jpg",
            raw_meta={"rating": "4.4", "reviews_count": 15},
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
            image_url="https://example.com/candidate.jpg",
            raw_meta={"discount": 20, "original_price": "13.74"},
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

    @override_settings(RECS_RERANKER_MODEL_PATH="Z:/not_existing/model.pkl")
    def test_bundle_recommendations_reranker_fallback(self):
        r = self.client.get(f"/api/me/recommendations/bundle?product_id={self.owned.id}&limit=5&algo=reranker")
        self.assertEqual(r.status_code, 200)
        self.assertIn("query", r.data)
        self.assertEqual(r.data["query"]["algo_requested"], "reranker")
        self.assertTrue(str(r.data["query"]["algo_used"]).startswith("cooc_fallback"))
        if r.data["results"]:
            product = r.data["results"][0]["product"]
            self.assertIn("image_url", product)
            self.assertIn("points_earned", product)
            self.assertIn("brand_slug", product)

    @override_settings(
        RECS_RERANKER_MODEL_PATH="Z:/not_existing/model.pkl",
        RECS_AB_ENABLED=True,
        RECS_AB_RERANKER_PERCENT=100,
        RECS_AB_SALT="test_salt",
    )
    def test_auto_algo_uses_stable_ab_routing(self):
        r1 = self.client.get("/api/me/recommendations?category=makeup&limit=5")
        r2 = self.client.get("/api/me/recommendations?category=makeup&limit=5")
        self.assertEqual(r1.status_code, 200)
        self.assertEqual(r2.status_code, 200)
        self.assertEqual(r1.data["query"]["algo_routing"]["source"], "ab")
        self.assertEqual(r2.data["query"]["algo_routing"]["source"], "ab")
        self.assertEqual(r1.data["query"]["algo_routing"]["ab_bucket"], r2.data["query"]["algo_routing"]["ab_bucket"])
        self.assertEqual(r1.data["query"]["algo_routing"]["ab_variant"], r2.data["query"]["algo_routing"]["ab_variant"])

    @override_settings(
        RECS_RERANKER_MODEL_PATH="Z:/not_existing/model.pkl",
        RECS_AB_ENABLED=True,
        RECS_AB_RERANKER_PERCENT=100,
        RECS_GUARDRAIL_ENABLED=True,
        RECS_GUARDRAIL_MIN_IMPRESSIONS=20,
        RECS_GUARDRAIL_MIN_DELTA_CR=-0.001,
        RECS_GUARDRAIL_WINDOW_DAYS=30,
    )
    def test_guardrail_forces_cooc(self):
        bulk = []
        for _ in range(30):
            bulk.append(
                RecommendationEvent(
                    user=self.user,
                    action=RecommendationEvent.Action.IMPRESSION,
                    product=self.candidate,
                    page="home",
                    section_key="for_you",
                    algo_mode="cooc",
                )
            )
        for _ in range(6):
            bulk.append(
                RecommendationEvent(
                    user=self.user,
                    action=RecommendationEvent.Action.PURCHASE_ATTRIBUTED,
                    product=self.candidate,
                    page="home",
                    section_key="for_you",
                    algo_mode="cooc",
                )
            )
        for _ in range(30):
            bulk.append(
                RecommendationEvent(
                    user=self.user,
                    action=RecommendationEvent.Action.IMPRESSION,
                    product=self.candidate,
                    page="home",
                    section_key="for_you",
                    algo_mode="reranker",
                )
            )
        RecommendationEvent.objects.bulk_create(bulk, batch_size=200)

        r = self.client.get("/api/me/recommendations?category=makeup&limit=5")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.data["query"]["algo_used"], "cooc")
        self.assertTrue(r.data["query"]["algo_routing"]["guardrail_forced"])
