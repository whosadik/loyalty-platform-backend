from decimal import Decimal
from django.urls import reverse
from django.contrib.auth import get_user_model
from rest_framework.test import APITestCase

from catalog.models import Product
from transactions.models import Transaction
from loyalty.models import LoyaltyLedgerEntry


class CheckoutIdempotencyTests(APITestCase):
    def setUp(self):
        User = get_user_model()
        self.user = User.objects.create_user(username="t1", password="t1pass")
        self.client.login(username="t1", password="t1pass")

        self.product = Product.objects.create(
            name="P1",
            brand="B",
            price=Decimal("9.99"),
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

    def test_checkout_idempotency_replay(self):
        url = "/api/checkout"  
        body = {
            "idempotency_key": "idem-test-1",
            "channel": "offline",
            "items": [{"product": self.product.id, "quantity": 1}],
        }

        r1 = self.client.post(url, body, format="json")
        self.assertEqual(r1.status_code, 201)
        txn_id_1 = r1.data["transaction_id"]

        r2 = self.client.post(url, body, format="json")
        self.assertEqual(r2.status_code, 200)
        self.assertTrue(r2.data.get("idempotent_replay", False))
        self.assertEqual(r2.data["transaction_id"], txn_id_1)

        self.assertEqual(Transaction.objects.filter(user=self.user, idempotency_key="idem-test-1").count(), 1)
        self.assertEqual(LoyaltyLedgerEntry.objects.filter(reference=f"checkout:txn:{txn_id_1}").count(), 1)
