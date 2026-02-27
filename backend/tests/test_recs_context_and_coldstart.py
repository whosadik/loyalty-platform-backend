from decimal import Decimal
from unittest.mock import patch

import numpy as np
from django.contrib.auth import get_user_model
from django.core.cache import cache
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
        cache.clear()
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
    def test_recommendations_without_category_returns_200(self, mock_load):
        mock_load.return_value = (_DummyModel(), "dummy_v1", None)

        r = self.client.get("/api/me/recommendations?limit=5&algo=reranker")
        self.assertEqual(r.status_code, 200)
        self.assertIsNone(r.data["query"]["category"])
        self.assertIn("algo_used", r.data["query"])
        self.assertIn("algo_routing", r.data["query"])
        self.assertGreater(len(r.data["results"]), 0)

    @patch("recs_app.reranker._load_reranker_model")
    @patch("recs_app.views.get_runtime_co_map")
    def test_purchase_context_is_used_for_recommendations_and_home(self, mock_get_co, mock_load):
        mock_load.return_value = (_DummyModel(), "dummy_v1", None)
        mock_get_co.return_value = (
            {
                int(self.p_ctx.id): {
                    int(self.p_cand_1.id): 7,
                    int(self.p_cand_2.id): 4,
                }
            },
            "artifact:test",
        )

        tx = Transaction.objects.create(user=self.user, total_amount=Decimal("11.00"), channel="online")
        TransactionItem.objects.create(transaction=tx, product=self.p_ctx, quantity=1, unit_price=Decimal("11.00"))

        r1 = self.client.get("/api/me/recommendations?category=makeup&limit=5&algo=reranker")
        self.assertEqual(r1.status_code, 200)
        self.assertEqual(r1.data["query"]["algo_used"], "reranker")
        self.assertEqual(r1.data["query"]["algo_routing"]["context_source"], "purchases")
        self.assertGreater(r1.data["query"]["algo_routing"]["context_len"], 0)
        retrieval = (r1.data["query"]["algo_routing"] or {}).get("retrieval") or {}
        self.assertTrue(retrieval.get("co_present"))
        self.assertGreater(int(retrieval.get("co_keys") or 0), 0)

        r2 = self.client.get("/api/me/recommendations/home?limit=5&algo=reranker")
        self.assertEqual(r2.status_code, 200)
        self.assertNotEqual(r2.data["query"]["for_you_algo_used"], "cooc_fallback:no_context")
        self.assertEqual(
            (r2.data["query"]["for_you_routing"] or {}).get("context_source"),
            "purchases",
        )

    @patch("recs_app.reranker.heuristic_recommend", return_value=[])
    @patch("recs_app.reranker._load_reranker_model")
    @patch("recs_app.views.get_runtime_co_map")
    def test_no_candidates_falls_back_to_cold_start(
        self,
        mock_get_co,
        mock_load,
        _mock_heur,
    ):
        mock_load.return_value = (_DummyModel(), "dummy_v1", None)
        mock_get_co.return_value = ({}, "artifact:none")

        tx = Transaction.objects.create(user=self.user, total_amount=Decimal("11.00"), channel="online")
        TransactionItem.objects.create(transaction=tx, product=self.p_ctx, quantity=1, unit_price=Decimal("11.00"))

        r = self.client.get("/api/me/recommendations?category=makeup&limit=5&algo=reranker")
        self.assertEqual(r.status_code, 200)
        self.assertNotEqual(r.data["query"]["algo_used"], "cooc_fallback:no_candidates")
        self.assertGreater(len(r.data["results"]), 0)

    @patch("recs_app.reranker._load_reranker_model")
    def test_event_based_co_map_makes_cooc_candidates_nonzero(self, mock_load):
        mock_load.return_value = (_DummyModel(), "dummy_v1", None)

        r_evt_1 = self.client.post(
            "/api/me/recommendations/event",
            {"action": "click", "product_id": int(self.p_ctx.id)},
            format="json",
        )
        self.assertEqual(r_evt_1.status_code, 200)
        r_evt_2 = self.client.post(
            "/api/me/recommendations/event",
            {"action": "click", "product_id": int(self.p_cand_1.id)},
            format="json",
        )
        self.assertEqual(r_evt_2.status_code, 200)

        r = self.client.get("/api/me/recommendations?category=makeup&limit=10&algo=reranker")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.data["query"]["algo_routing"]["context_source"], "behavior")

        retrieval = (r.data["query"]["algo_routing"] or {}).get("retrieval") or {}
        self.assertTrue(bool(retrieval.get("co_present")))
        self.assertTrue(
            int(retrieval.get("cooc_candidates") or 0) > 0
            or int(retrieval.get("transitions_len") or 0) > 0
        )

    @patch("recs_app.reranker._load_reranker_model")
    @patch("recs_app.views.get_runtime_co_map")
    def test_sources_are_honest(self, mock_get_co, mock_load):
        mock_load.return_value = (_DummyModel(), "dummy_v1", None)
        mock_get_co.return_value = ({}, "db_all_time_empty")

        tx = Transaction.objects.create(user=self.user, total_amount=Decimal("11.00"), channel="online")
        TransactionItem.objects.create(transaction=tx, product=self.p_ctx, quantity=1, unit_price=Decimal("11.00"))

        r = self.client.get("/api/me/recommendations?category=makeup&limit=5&algo=reranker")
        self.assertEqual(r.status_code, 200)

        retrieval = (r.data["query"]["algo_routing"] or {}).get("retrieval") or {}
        sources = list(retrieval.get("sources") or [])
        self.assertTrue(all(not str(src).startswith("cooc_") for src in sources))
