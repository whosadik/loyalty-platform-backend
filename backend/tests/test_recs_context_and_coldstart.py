from decimal import Decimal
from unittest.mock import patch

import numpy as np
from django.contrib.auth import get_user_model
from rest_framework.test import APITestCase

from catalog.models import Product
from transactions.models import Transaction, TransactionItem
from users_app.models import CustomerProfile


class _DummyModel:
    def predict_proba(self, x):
        n = int(getattr(x, "shape", [len(x)])[0])
        probs = np.linspace(0.9, 0.6, num=max(n, 1), dtype=float)[:n]
        return np.column_stack([1.0 - probs, probs])


class RecsContextAndColdStartTests(APITestCase):
    def setUp(self):
        user_model = get_user_model()
        self.user = user_model.objects.create_user(username="ctx_user", password="pass12345")
        self.client.force_authenticate(self.user)
        CustomerProfile.objects.get_or_create(user=self.user)

        self.p_ctx = Product.objects.create(
            name="Ctx Product",
            brand="B1",
            price=Decimal("11.00"),
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
        self.p_cand_1 = Product.objects.create(
            name="Cand 1",
            brand="B2",
            price=Decimal("12.00"),
            category="makeup",
            product_type="mascara",
            concerns=[],
            attrs={},
            actives=[],
            flags=[],
            supported_skin_types=[],
            strength="low",
            in_stock=True,
        )
        self.p_cand_2 = Product.objects.create(
            name="Cand 2",
            brand="B3",
            price=Decimal("13.00"),
            category="makeup",
            product_type="blush",
            concerns=[],
            attrs={},
            actives=[],
            flags=[],
            supported_skin_types=[],
            strength="low",
            in_stock=True,
        )

    @patch("recs_app.reranker._load_reranker_model")
    def test_new_user_gets_cold_start_instead_of_no_context_fallback(self, mock_load):
        mock_load.return_value = (_DummyModel(), "dummy_v1", None)

        r = self.client.get("/api/me/recommendations?category=makeup&limit=5&algo=reranker")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.data["query"]["algo_used"], "cold_start:trending")
        self.assertGreater(len(r.data["results"]), 0)
        self.assertEqual(r.data["query"]["algo_routing"]["context_source"], "none")
        self.assertEqual(r.data["query"]["algo_routing"]["context_len"], 0)
        self.assertEqual((r.data["results"][0].get("components") or {}).get("mode"), "cold_start")

    @patch("recs_app.reranker._load_reranker_model")
    def test_purchase_context_is_used_for_recommendations_and_home(self, mock_load):
        mock_load.return_value = (_DummyModel(), "dummy_v1", None)

        tx = Transaction.objects.create(user=self.user, total_amount=Decimal("11.00"), channel="online")
        TransactionItem.objects.create(transaction=tx, product=self.p_ctx, quantity=1, unit_price=Decimal("11.00"))

        r1 = self.client.get("/api/me/recommendations?category=makeup&limit=5&algo=reranker")
        self.assertEqual(r1.status_code, 200)
        self.assertEqual(r1.data["query"]["algo_used"], "reranker")
        self.assertEqual(r1.data["query"]["algo_routing"]["context_source"], "purchases")
        self.assertGreater(r1.data["query"]["algo_routing"]["context_len"], 0)

        r2 = self.client.get("/api/me/recommendations/home?limit=5&algo=reranker")
        self.assertEqual(r2.status_code, 200)
        self.assertNotEqual(r2.data["query"]["for_you_algo_used"], "cooc_fallback:no_context")
        self.assertEqual(
            (r2.data["query"]["for_you_routing"] or {}).get("context_source"),
            "purchases",
        )
