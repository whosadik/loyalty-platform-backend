from datetime import timedelta
from decimal import Decimal

from django.contrib.auth import get_user_model
from django.test import override_settings
from django.utils import timezone
from rest_framework.test import APITestCase

from catalog.models import Product
from transactions.models import Transaction, TransactionItem


@override_settings(FAVORITE_CATEGORY_WINDOW_DAYS=90)
class MeFavoriteCategoryEndpointTests(APITestCase):
    def setUp(self):
        User = get_user_model()
        self.user = User.objects.create_user(username="favcat_api_u1", password="pass12345")
        self.client.force_authenticate(self.user)

        self.p_makeup = Product.objects.create(
            name="Favorite Lipstick",
            brand="B",
            price=Decimal("10.00"),
            category="makeup",
            product_type="lipstick",
            concerns=[],
            attrs={},
            actives=[],
            flags=[],
            supported_skin_types=["normal"],
            strength="low",
            in_stock=True,
        )
        self.p_skincare = Product.objects.create(
            name="Favorite Serum",
            brand="B",
            price=Decimal("15.00"),
            category="skincare",
            product_type="serum",
            concerns=[],
            attrs={},
            actives=[],
            flags=[],
            supported_skin_types=["normal"],
            strength="low",
            in_stock=True,
        )

    def _create_txn_days_ago(self, days: int, items: list[tuple[Product, int]]) -> None:
        txn = Transaction.objects.create(
            user=self.user,
            total_amount=Decimal("0.00"),
            channel="offline",
        )
        total = Decimal("0.00")
        for p, qty in items:
            TransactionItem.objects.create(
                transaction=txn,
                product=p,
                quantity=qty,
                unit_price=Decimal(str(p.price)),
            )
            total += Decimal(str(p.price)) * qty
        Transaction.objects.filter(id=txn.id).update(
            total_amount=total,
            created_at=timezone.now() - timedelta(days=days),
        )

    def test_returns_none_for_empty_history_and_incomplete_profile(self):
        r = self.client.get("/api/me/favorite-category")
        self.assertEqual(r.status_code, 200)
        self.assertTrue(r.data["ok"])
        self.assertIsNone(r.data["favorite_category"])
        self.assertFalse(r.data["profile_complete"])
        self.assertEqual(r.data["explain"]["history_items_considered"], 0)
        self.assertEqual(r.data["explain"]["signals"], [])

    def test_returns_favorite_category_with_explainability(self):
        self._create_txn_days_ago(12, [(self.p_makeup, 3), (self.p_skincare, 1)])
        self._create_txn_days_ago(5, [(self.p_makeup, 2)])

        r = self.client.get("/api/me/favorite-category")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.data["favorite_category"], "makeup")
        self.assertGreaterEqual(r.data["explain"]["history_items_considered"], 2)
        self.assertEqual(r.data["explain"]["signals"][0]["category"], "makeup")
        self.assertEqual(r.data["explain"]["picked_by"], ["total_qty", "line_count", "last_at", "category"])

    def test_ignores_history_outside_window(self):
        self._create_txn_days_ago(120, [(self.p_makeup, 4)])

        r = self.client.get("/api/me/favorite-category")
        self.assertEqual(r.status_code, 200)
        self.assertIsNone(r.data["favorite_category"])
        self.assertEqual(r.data["explain"]["history_items_considered"], 0)
