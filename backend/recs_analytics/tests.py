from django.contrib.auth import get_user_model
from django.test import TestCase

from catalog.models import Product
from recs_analytics.admin_metrics import recs_experiments_metrics, recs_metrics_30d
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
            context={"experiment_id": "recs_algo_ab_v1", "experiment_variant": "control"},
        )
        RecommendationEvent.objects.create(
            user=u,
            action=RecommendationEvent.Action.CLICK,
            product=p,
            page="home",
            section_key="for_you",
            algo_mode="cooc",
            context={"experiment_id": "recs_algo_ab_v1", "experiment_variant": "control"},
        )
        RecommendationEvent.objects.create(
            user=u,
            action=RecommendationEvent.Action.IMPRESSION,
            product=p,
            page="home",
            section_key="for_you",
            algo_mode="reranker",
            context={"experiment_id": "recs_algo_ab_v1", "experiment_variant": "test"},
        )

        metrics = recs_metrics_30d()
        self.assertIn("by_algo", metrics)
        self.assertIn("cooc", metrics["by_algo"])
        self.assertIn("reranker", metrics["by_algo"])
        self.assertEqual(metrics["by_algo"]["cooc"]["impression"], 1)
        self.assertEqual(metrics["by_algo"]["cooc"]["click"], 1)
        self.assertIn("by_experiment", metrics)
        self.assertIn("recs_algo_ab_v1", metrics["by_experiment"])
        exp = metrics["by_experiment"]["recs_algo_ab_v1"]
        self.assertEqual(exp["totals"]["impression"], 2)
        self.assertEqual(exp["variants"]["control"]["click"], 1)
        self.assertEqual(exp["variants"]["test"]["impression"], 1)


class RecsExperimentsMetricsFiltersTests(TestCase):
    def test_filters_by_experiment_and_variant(self):
        User = get_user_model()
        u = User.objects.create_user(username="recs_exp_user", password="pass12345")
        p = Product.objects.create(
            name="P2",
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
            context={"experiment_id": "exp_a", "experiment_variant": "control"},
        )
        RecommendationEvent.objects.create(
            user=u,
            action=RecommendationEvent.Action.CLICK,
            product=p,
            page="home",
            section_key="for_you",
            algo_mode="cooc",
            context={"experiment_id": "exp_a", "experiment_variant": "control"},
        )
        RecommendationEvent.objects.create(
            user=u,
            action=RecommendationEvent.Action.IMPRESSION,
            product=p,
            page="home",
            section_key="for_you",
            algo_mode="reranker",
            context={"experiment_id": "exp_b", "experiment_variant": "test"},
        )

        payload = recs_experiments_metrics(days=30, experiment_id="exp_a", variant="control")
        self.assertEqual(payload["summary"]["experiments_count"], 1)
        self.assertIn("exp_a", payload["experiments"])
        self.assertNotIn("exp_b", payload["experiments"])
        self.assertEqual(payload["experiments"]["exp_a"]["totals"]["impression"], 1)
        self.assertEqual(payload["experiments"]["exp_a"]["variants"]["control"]["click"], 1)
