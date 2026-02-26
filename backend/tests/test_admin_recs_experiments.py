from decimal import Decimal

from django.contrib.auth import get_user_model
from rest_framework.test import APITestCase

from admin_tools.models import StaffProfile, StaffRole
from catalog.models import Product
from recs_analytics.models import RecommendationEvent


class AdminRecsExperimentsTests(APITestCase):
    def setUp(self):
        User = get_user_model()
        self.admin = User.objects.create_user(username="recs_exp_admin", password="pass12345")
        self.admin.is_staff = True
        self.admin.save(update_fields=["is_staff"])
        StaffProfile.objects.update_or_create(
            user=self.admin,
            defaults={"role": StaffRole.ANALYST, "permissions": ["view_metrics"]},
        )

        self.user = User.objects.create_user(username="recs_exp_user1", password="pass12345")
        self.product = Product.objects.create(
            name="Recs Exp Product",
            brand="B",
            price=Decimal("12.00"),
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
            user=self.user,
            action=RecommendationEvent.Action.IMPRESSION,
            product=self.product,
            page="home",
            section_key="for_you",
            algo_mode="cooc",
            context={"experiment_id": "recs_algo_ab_v1", "experiment_variant": "control"},
        )
        RecommendationEvent.objects.create(
            user=self.user,
            action=RecommendationEvent.Action.CLICK,
            product=self.product,
            page="home",
            section_key="for_you",
            algo_mode="cooc",
            context={"experiment_id": "recs_algo_ab_v1", "experiment_variant": "control"},
        )
        RecommendationEvent.objects.create(
            user=self.user,
            action=RecommendationEvent.Action.IMPRESSION,
            product=self.product,
            page="home",
            section_key="for_you",
            algo_mode="reranker",
            context={"experiment_id": "recs_algo_ab_v1", "experiment_variant": "test"},
        )
        RecommendationEvent.objects.create(
            user=self.user,
            action=RecommendationEvent.Action.IMPRESSION,
            product=self.product,
            page="home",
            section_key="for_you",
            algo_mode="cooc",
            context={"experiment_id": "other_exp", "experiment_variant": "control"},
        )

    def test_admin_recs_experiments_returns_metrics(self):
        self.client.force_authenticate(self.admin)
        r = self.client.get("/api/admin/recs/experiments?days=30")
        self.assertEqual(r.status_code, 200)
        self.assertTrue(r.data["ok"])
        self.assertIn("summary", r.data)
        self.assertIn("experiments", r.data)
        self.assertIn("recs_algo_ab_v1", r.data["experiments"])
        self.assertGreaterEqual(r.data["summary"]["experiments_count"], 2)
        exp = r.data["experiments"]["recs_algo_ab_v1"]
        self.assertEqual(exp["variants"]["control"]["click"], 1)
        self.assertEqual(exp["variants"]["test"]["impression"], 1)

    def test_admin_recs_experiments_filters(self):
        self.client.force_authenticate(self.admin)
        r = self.client.get("/api/admin/recs/experiments?days=30&experiment_id=recs_algo_ab_v1&variant=control")
        self.assertEqual(r.status_code, 200)
        self.assertTrue(r.data["ok"])
        self.assertEqual(r.data["summary"]["experiments_count"], 1)
        self.assertIn("recs_algo_ab_v1", r.data["experiments"])
        exp = r.data["experiments"]["recs_algo_ab_v1"]
        self.assertIn("control", exp["variants"])
        self.assertNotIn("test", exp["variants"])
