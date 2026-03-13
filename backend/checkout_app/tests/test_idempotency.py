from decimal import Decimal
from django.urls import reverse
from django.contrib.auth import get_user_model
from rest_framework.test import APITestCase

from catalog.models import Product
from loyalty.models import LoyaltyAccount, Tier
from transactions.models import Transaction
from loyalty.models import LoyaltyLedgerEntry
from users_app.models import CustomerProfile


class CheckoutIdempotencyTests(APITestCase):
    def setUp(self):
        User = get_user_model()
        self.user = User.objects.create_user(username="t1", password="t1pass")
        self.client.login(username="t1", password="t1pass")
        CustomerProfile.objects.get_or_create(user=self.user)
        bronze, _ = Tier.objects.get_or_create(
            name="Bronze",
            defaults={"threshold_spend_90d": Decimal("0.00"), "points_rate": Decimal("0.10")},
        )
        LoyaltyAccount.objects.get_or_create(
            user=self.user,
            defaults={"tier": bronze, "points_balance": 0},
        )

        self.product = Product.objects.create(
            name="P1",
            brand="B",
            price=Decimal("9.99"),
            category="haircare",
            product_type="shampoo",
            image_url="https://example.com/shampoo.jpg",
            image_urls=["https://example.com/shampoo.jpg"],
            currency="KZT",
            concerns=[],
            attrs={},
            actives=[],
            flags=[],
            supported_skin_types=[],
            strength="low",
            in_stock=True,
        )
        Product.objects.create(
            name="P2",
            brand="B",
            price=Decimal("12.99"),
            category="haircare",
            product_type="conditioner",
            image_url="https://example.com/conditioner.jpg",
            image_urls=["https://example.com/conditioner.jpg"],
            currency="KZT",
            concerns=[],
            attrs={},
            actives=[],
            flags=[],
            supported_skin_types=[],
            strength="low",
            in_stock=True,
        )
        Product.objects.create(
            name="P3",
            brand="B",
            price=Decimal("13.99"),
            category="haircare",
            product_type="hair_mask",
            image_url="https://example.com/mask.jpg",
            image_urls=["https://example.com/mask.jpg"],
            currency="KZT",
            concerns=[],
            attrs={},
            actives=[],
            flags=[],
            supported_skin_types=[],
            strength="low",
            in_stock=True,
        )
        Product.objects.create(
            name="P4",
            brand="B",
            price=Decimal("11.99"),
            category="haircare",
            product_type="hair_oil",
            image_url="https://example.com/oil.jpg",
            image_urls=["https://example.com/oil.jpg"],
            currency="KZT",
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
        self.assertEqual(r2.data.get("next_roadmap_step"), r1.data.get("next_roadmap_step"))

        self.assertEqual(Transaction.objects.filter(user=self.user, idempotency_key="idem-test-1").count(), 1)
        self.assertEqual(LoyaltyLedgerEntry.objects.filter(reference=f"checkout:txn:{txn_id_1}").count(), 1)
