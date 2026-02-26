from django.contrib.auth import get_user_model
from django.test import TestCase

from catalog.models import Product
from recs_analytics.admin_metrics import recs_metrics_30d
from recs_analytics.models import RecommendationEvent


class RecsMetricsByAlgoTests(TestCase):
    def test_recs_metrics_contains_by_algo(self):
        User = get_user_model()
        u = User.objects.create_user(username="recs_metric_user", password="pass12345")
        p = Product.objects.create(
            name="P",
            brand="B",
            price="10.00",
            category="makeup",
            product_type="lipstick",
            concerns=[],
            attrs={},
            actives=[],
            flags=[],
            supported_skin_types=[],
            strength="low",
            in_stock=True,
        )

        RecommendationEvent.objects.create(
            user=u,
            action=RecommendationEvent.Action.IMPRESSION,
            product=p,
            page="home",
            section_key="for_you",
            algo_mode="cooc",
        )
        RecommendationEvent.objects.create(
            user=u,
            action=RecommendationEvent.Action.CLICK,
            product=p,
            page="home",
            section_key="for_you",
            algo_mode="cooc",
        )
        RecommendationEvent.objects.create(
            user=u,
            action=RecommendationEvent.Action.IMPRESSION,
            product=p,
            page="home",
            section_key="for_you",
            algo_mode="reranker",
        )

        metrics = recs_metrics_30d()
        self.assertIn("by_algo", metrics)
        self.assertIn("cooc", metrics["by_algo"])
        self.assertIn("reranker", metrics["by_algo"])
        self.assertEqual(metrics["by_algo"]["cooc"]["impression"], 1)
        self.assertEqual(metrics["by_algo"]["cooc"]["click"], 1)
