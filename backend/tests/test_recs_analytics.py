from decimal import Decimal

from django.contrib.auth import get_user_model
from rest_framework.test import APITestCase

from catalog.models import Product
from recs_analytics.models import RecommendationEvent
from recs_analytics.services import attribute_purchase
from users_app.models import CustomerProfile


class RecsAnalyticsTests(APITestCase):
    def setUp(self):
        User = get_user_model()
        self.user = User.objects.create_user(username="u3", password="pass12345")
        self.client.force_authenticate(self.user)
        CustomerProfile.objects.get_or_create(user=self.user)

        self.p = Product.objects.create(
            name="Lipstick",
            brand="B",
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

    def test_impressions_written_on_home(self):
        r = self.client.get("/api/me/recommendations/home?limit=5")
        self.assertEqual(r.status_code, 200)
        self.assertTrue(RecommendationEvent.objects.filter(user=self.user, action="impression").exists())

    def test_home_titles_follow_language_header(self):
        r = self.client.get(
            "/api/me/recommendations/home?limit=5",
            HTTP_X_APP_LANGUAGE="en",
            HTTP_ACCEPT_LANGUAGE="en",
        )
        self.assertEqual(r.status_code, 200)
        self.assertEqual(
            [section["title"] for section in r.data["sections"]],
            ["For you", "Because you bought", "Trending"],
        )

    def test_click_inherits_experiment_context_from_impression(self):
        imp = RecommendationEvent.objects.create(
            user=self.user,
            action=RecommendationEvent.Action.IMPRESSION,
            product=self.p,
            page="home",
            section_key="for_you",
            request_id="rid-ctx-1",
            algo_mode="reranker",
            context={
                "experiment_id": "recs_algo_ab_v1",
                "experiment_variant": "test",
                "algo_used": "reranker",
                "algo_source": "ab",
                "model_version": "recs_reranker_lr_v2",
            },
        )

        r = self.client.post(
            "/api/me/recommendations/event",
            {"action": "click", "product_id": self.p.id},
            format="json",
            HTTP_X_REQUEST_ID="rid-ctx-1",
        )
        self.assertEqual(r.status_code, 200)

        click = RecommendationEvent.objects.filter(
            user=self.user,
            product=self.p,
            action=RecommendationEvent.Action.CLICK,
        ).order_by("-id").first()
        self.assertIsNotNone(click)
        self.assertEqual(click.context.get("from_impression_id"), imp.id)
        self.assertEqual(click.context.get("experiment_id"), "recs_algo_ab_v1")
        self.assertEqual(click.context.get("experiment_variant"), "test")
        self.assertEqual(click.context.get("algo_used"), "reranker")

    def test_purchase_attributed_inherits_experiment_context(self):
        imp = RecommendationEvent.objects.create(
            user=self.user,
            action=RecommendationEvent.Action.IMPRESSION,
            product=self.p,
            page="home",
            section_key="for_you",
            algo_mode="cooc",
            context={
                "experiment_id": "recs_algo_ab_v1",
                "experiment_variant": "control",
                "algo_used": "cooc",
                "algo_source": "ab",
            },
        )
        attribute_purchase(
            user=self.user,
            purchased_product_ids=[self.p.id],
            window_days=7,
            request_id="rid-purchase-1",
        )

        ev = RecommendationEvent.objects.filter(
            user=self.user,
            product=self.p,
            action=RecommendationEvent.Action.PURCHASE_ATTRIBUTED,
        ).order_by("-id").first()
        self.assertIsNotNone(ev)
        self.assertEqual(ev.context.get("attributed_from_event_id"), imp.id)
        self.assertEqual(ev.context.get("experiment_id"), "recs_algo_ab_v1")
        self.assertEqual(ev.context.get("experiment_variant"), "control")
        self.assertEqual(ev.request_id, "rid-purchase-1")
