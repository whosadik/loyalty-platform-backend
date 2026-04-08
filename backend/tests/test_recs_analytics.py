from decimal import Decimal
from unittest.mock import patch

from django.contrib.auth import get_user_model
from django.core.cache import cache
from django.test import override_settings
from rest_framework.test import APITestCase

from catalog.models import Product
from recs_analytics.models import RecommendationEvent
from recs_analytics.services import attribute_purchase
from transactions.models import OwnedProduct, Transaction, TransactionItem
from users_app.models import CustomerProfile


class RecsAnalyticsTests(APITestCase):
    def setUp(self):
        cache.clear()
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
        self.base_product = Product.objects.create(
            name="Base Serum",
            brand="B",
            price=Decimal("12.99"),
            category="makeup",
            product_type="foundation",
            concerns=[],
            attrs={"finish": "natural"},
            actives=[],
            flags=[],
            supported_skin_types=[],
            strength="low",
            in_stock=True,
        )
        self.rec_candidate_1 = Product.objects.create(
            name="Candidate Mascara",
            brand="B2",
            price=Decimal("13.99"),
            category="makeup",
            product_type="mascara",
            concerns=[],
            attrs={"finish": "matte"},
            actives=[],
            flags=[],
            supported_skin_types=[],
            strength="low",
            in_stock=True,
        )
        self.rec_candidate_2 = Product.objects.create(
            name="Candidate Blush",
            brand="B3",
            price=Decimal("14.99"),
            category="makeup",
            product_type="blush",
            concerns=[],
            attrs={"finish": "satin"},
            actives=[],
            flags=[],
            supported_skin_types=[],
            strength="low",
            in_stock=True,
        )
        OwnedProduct.objects.create(
            user=self.user,
            product=self.base_product,
            quantity_total=1,
            is_active=True,
        )
        txn = Transaction.objects.create(
            user=self.user,
            total_amount=Decimal("12.99"),
            channel="online",
        )
        TransactionItem.objects.create(
            transaction=txn,
            product=self.base_product,
            quantity=1,
            unit_price=Decimal("12.99"),
        )

    def _runtime_co_map(self):
        return (
            {
                int(self.base_product.id): {
                    int(self.rec_candidate_1.id): 9,
                    int(self.rec_candidate_2.id): 7,
                }
            },
            "artifact:test",
        )

    def _assert_impression_context(
        self,
        event: RecommendationEvent,
        *,
        page: str,
        section_key: str,
        request_id: str,
        expect_context_source: str | None = None,
        expect_base_product_id: int | None = None,
    ) -> None:
        self.assertEqual(event.page, page)
        self.assertEqual(event.section_key, section_key)
        self.assertEqual(event.request_id, request_id)
        self.assertEqual(event.context.get("page"), page)
        self.assertEqual(event.context.get("section"), section_key)
        self.assertEqual(event.context.get("section_key"), section_key)
        self.assertEqual(event.context.get("product_id"), event.product_id)
        self.assertGreaterEqual(int(event.context.get("rank") or 0), 1)
        self.assertTrue(str(event.context.get("algo_used") or "").strip())
        if expect_context_source is not None:
            self.assertEqual(event.context.get("context_source"), expect_context_source)
        if expect_base_product_id is not None:
            self.assertEqual(event.context.get("base_product_id"), expect_base_product_id)

    @patch("recs_app.views.get_runtime_co_map")
    def test_impressions_written_on_home(self, mock_get_co):
        mock_get_co.return_value = self._runtime_co_map()
        r = self.client.get(
            "/api/me/recommendations/home?limit=5&category=makeup",
            HTTP_X_REQUEST_ID="rid-home-1",
        )
        self.assertEqual(r.status_code, 200)
        expected_ids = {
            int(result["product"]["id"])
            for section in list(r.data["sections"] or [])
            for result in list(section.get("results") or [])
        }
        self.assertTrue(expected_ids)

        events = RecommendationEvent.objects.filter(
            user=self.user,
            action=RecommendationEvent.Action.IMPRESSION,
            page="home",
            request_id="rid-home-1",
        ).order_by("id")
        self.assertEqual(events.count(), len(expected_ids))

        first = events.first()
        self.assertIsNotNone(first)
        self._assert_impression_context(
            first,
            page="home",
            section_key=str(first.section_key),
            request_id="rid-home-1",
        )

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

    @override_settings(
        RECS_RERANKER_MODEL_PATH="Z:/not_existing/model.pkl",
        RECS_AB_ENABLED=True,
        RECS_AB_RERANKER_PERCENT=100,
        RECS_AB_SALT="telemetry-test-salt",
    )
    @patch("recs_app.views.get_runtime_co_map")
    def test_recommendations_endpoint_writes_impressions(self, mock_get_co):
        mock_get_co.return_value = self._runtime_co_map()

        r = self.client.get(
            "/api/me/recommendations?category=makeup&limit=5",
            HTTP_X_REQUEST_ID="rid-recs-1",
        )
        self.assertEqual(r.status_code, 200)

        expected_ids = {int(row["product"]["id"]) for row in list(r.data["results"] or [])}
        self.assertTrue(expected_ids)

        events = RecommendationEvent.objects.filter(
            user=self.user,
            action=RecommendationEvent.Action.IMPRESSION,
            page="recommendations",
            section_key="results",
            request_id="rid-recs-1",
        ).order_by("id")
        self.assertEqual(events.count(), len(expected_ids))

        first = events.first()
        self.assertIsNotNone(first)
        self._assert_impression_context(
            first,
            page="recommendations",
            section_key="results",
            request_id="rid-recs-1",
            expect_context_source="purchases",
        )
        self.assertEqual(first.context.get("experiment_id"), "recs_algo_ab_v1")
        self.assertIn(first.context.get("experiment_variant"), {"test", "control"})
        self.assertEqual(first.context.get("variant"), first.context.get("experiment_variant"))
        self.assertEqual(first.context.get("algo_used"), r.data["query"]["algo_used"])

    @patch("recs_app.views.get_runtime_co_map")
    def test_bundle_endpoint_writes_impressions(self, mock_get_co):
        mock_get_co.return_value = self._runtime_co_map()

        r = self.client.get(
            f"/api/me/recommendations/bundle?product_id={self.base_product.id}&limit=5&algo=cooc",
            HTTP_X_REQUEST_ID="rid-bundle-1",
        )
        self.assertEqual(r.status_code, 200)

        expected_ids = {int(row["product"]["id"]) for row in list(r.data["results"] or [])}
        self.assertTrue(expected_ids)

        events = RecommendationEvent.objects.filter(
            user=self.user,
            action=RecommendationEvent.Action.IMPRESSION,
            page="bundle",
            section_key="bundle",
            request_id="rid-bundle-1",
        ).order_by("id")
        self.assertEqual(events.count(), len(expected_ids))

        first = events.first()
        self.assertIsNotNone(first)
        self._assert_impression_context(
            first,
            page="bundle",
            section_key="bundle",
            request_id="rid-bundle-1",
            expect_base_product_id=self.base_product.id,
        )
        self.assertEqual(first.context.get("algo_used"), r.data["query"]["algo_used"])

    @patch("recs_app.views.get_runtime_co_map")
    def test_repeated_recommendation_reads_append_one_impression_batch_per_response(self, mock_get_co):
        mock_get_co.return_value = self._runtime_co_map()

        first = self.client.get(
            "/api/me/recommendations?category=makeup&limit=5&algo=cooc",
            HTTP_X_REQUEST_ID="rid-recs-repeat-1",
        )
        second = self.client.get(
            "/api/me/recommendations?category=makeup&limit=5&algo=cooc",
            HTTP_X_REQUEST_ID="rid-recs-repeat-2",
        )
        self.assertEqual(first.status_code, 200)
        self.assertEqual(second.status_code, 200)

        first_ids = {int(row["product"]["id"]) for row in list(first.data["results"] or [])}
        second_ids = {int(row["product"]["id"]) for row in list(second.data["results"] or [])}

        first_events = RecommendationEvent.objects.filter(
            user=self.user,
            action=RecommendationEvent.Action.IMPRESSION,
            page="recommendations",
            section_key="results",
            request_id="rid-recs-repeat-1",
        )
        second_events = RecommendationEvent.objects.filter(
            user=self.user,
            action=RecommendationEvent.Action.IMPRESSION,
            page="recommendations",
            section_key="results",
            request_id="rid-recs-repeat-2",
        )

        self.assertEqual(first_events.count(), len(first_ids))
        self.assertEqual(second_events.count(), len(second_ids))
        self.assertEqual(
            RecommendationEvent.objects.filter(
                user=self.user,
                action=RecommendationEvent.Action.IMPRESSION,
                page="recommendations",
                section_key="results",
            ).count(),
            len(first_ids) + len(second_ids),
        )

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
