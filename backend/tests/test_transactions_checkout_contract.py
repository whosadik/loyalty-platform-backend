from decimal import Decimal

from django.contrib.auth import get_user_model
from rest_framework.test import APITestCase

from catalog.models import Product
from loyalty.models import LoyaltyAccount, Tier
from transactions.models import CartItem


def _first_result(data):
    if isinstance(data, list):
        return data[0]
    if isinstance(data, dict) and isinstance(data.get("results"), list):
        return data["results"][0]
    if isinstance(data, dict) and isinstance(data.get("transactions"), list):
        return data["transactions"][0]
    raise AssertionError("Transactions response does not contain a list payload")


class TransactionsCheckoutContractTests(APITestCase):
    def setUp(self):
        User = get_user_model()
        self.user = User.objects.create_user(username="tx_contract_u1", password="pass12345")
        self.client.force_authenticate(self.user)

        bronze, _ = Tier.objects.get_or_create(
            name="Bronze",
            defaults={"threshold_spend_90d": Decimal("0.00"), "points_rate": Decimal("1.00")},
        )
        bronze.threshold_spend_90d = Decimal("0.00")
        bronze.points_rate = Decimal("1.00")
        bronze.save(update_fields=["threshold_spend_90d", "points_rate"])

        silver, _ = Tier.objects.get_or_create(
            name="Silver",
            defaults={"threshold_spend_90d": Decimal("100.00"), "points_rate": Decimal("1.50")},
        )
        silver.threshold_spend_90d = Decimal("100.00")
        silver.points_rate = Decimal("1.50")
        silver.save(update_fields=["threshold_spend_90d", "points_rate"])
        account, _ = LoyaltyAccount.objects.get_or_create(
            user=self.user,
            defaults={"tier": bronze, "points_balance": 0},
        )
        account.tier = bronze
        account.points_balance = 0
        account.save(update_fields=["tier", "points_balance"])

        self.product = Product.objects.create(
            name="Barrier Serum",
            brand="Uilesim",
            price=Decimal("120.00"),
            currency="KZT",
            category="skincare",
            product_type="serum",
            image_url="https://example.com/serum.jpg",
            image_urls=["https://example.com/serum.jpg"],
            in_stock=True,
            concerns=[],
            attrs={},
            actives=[],
            flags=[],
            supported_skin_types=["normal"],
            strength="medium",
        )

    def test_transactions_and_last_checkout_expose_rich_snapshot(self):
        CartItem.objects.create(user=self.user, product=self.product, quantity=2)

        checkout = self.client.post(
            "/api/checkout",
            {
                "channel": "online",
                "items": [{"product": self.product.id, "quantity": 1}],
            },
            format="json",
        )
        self.assertEqual(checkout.status_code, 201)
        self.assertTrue(checkout.data["ok"])
        txn_id = int(checkout.data["transaction_id"])
        expected_ref = f"TXN-{txn_id:08d}"
        self.assertEqual(checkout.data["gross_total"], "120.00")
        self.assertEqual(checkout.data["discount_amount"], "0")
        self.assertEqual(checkout.data["net_total"], "120.00")
        self.assertEqual(checkout.data["points_earned"], 180)
        self.assertEqual(checkout.data["points_redeemed"], 0)
        self.assertEqual(checkout.data["new_balance"], 180)
        self.assertEqual(checkout.data["new_tier"], "Silver")
        self.assertTrue(checkout.data["tier_upgraded"])
        self.assertEqual(CartItem.objects.get(user=self.user, product=self.product).quantity, 1)

        transactions = self.client.get("/api/transactions/")
        self.assertEqual(transactions.status_code, 200)
        txn = _first_result(transactions.data)
        self.assertEqual(txn["id"], txn_id)
        self.assertEqual(txn["transaction_id"], expected_ref)
        self.assertEqual(txn["type"], "purchase")
        self.assertEqual(txn["status"], "completed")
        self.assertEqual(txn["gross_total"], "120.00")
        self.assertEqual(txn["discount_amount"], "0")
        self.assertEqual(txn["net_total"], "120.00")
        self.assertEqual(txn["points_earned"], 180)
        self.assertEqual(txn["points_redeemed"], 0)
        self.assertEqual(txn["points_change"], 180)
        self.assertEqual(txn["new_balance"], 180)
        self.assertEqual(txn["tier_after"], "Silver")
        self.assertEqual(txn["new_tier"], "Silver")
        self.assertTrue(txn["tier_upgraded"])
        self.assertIn("Покупка", txn["description"])
        self.assertEqual(len(txn["items"]), 1)
        self.assertEqual(txn["items"][0]["product"], self.product.id)
        self.assertEqual(txn["items"][0]["product_summary"]["id"], self.product.id)
        self.assertEqual(txn["items"][0]["product_summary"]["name"], "Barrier Serum")
        self.assertEqual(txn["items"][0]["product_summary"]["image_url"], "https://example.com/serum.jpg")

        detail = self.client.get(f"/api/transactions/{txn['id']}/")
        self.assertEqual(detail.status_code, 200)
        self.assertEqual(detail.data["transaction_id"], expected_ref)
        self.assertEqual(detail.data["points_change"], 180)
        self.assertEqual(detail.data["items"][0]["product_summary"]["name"], "Barrier Serum")

        last_checkout = self.client.get("/api/checkout/last")
        self.assertEqual(last_checkout.status_code, 200)
        self.assertTrue(last_checkout.data["ok"])
        self.assertIsNotNone(last_checkout.data["checkout"])
        self.assertEqual(last_checkout.data["checkout"]["id"], txn["id"])
        self.assertEqual(last_checkout.data["checkout"]["transaction_id"], expected_ref)
        self.assertEqual(last_checkout.data["checkout"]["new_tier"], "Silver")
        self.assertTrue(last_checkout.data["checkout"]["tier_upgraded"])

    def test_checkout_deletes_cart_line_when_all_quantity_is_committed(self):
        CartItem.objects.create(user=self.user, product=self.product, quantity=1)

        checkout = self.client.post(
            "/api/checkout",
            {
                "channel": "online",
                "items": [{"product": self.product.id, "quantity": 1}],
            },
            format="json",
        )
        self.assertEqual(checkout.status_code, 201)
        self.assertFalse(CartItem.objects.filter(user=self.user, product=self.product).exists())

    def test_last_checkout_returns_null_when_user_has_no_transactions(self):
        response = self.client.get("/api/checkout/last")
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.data["ok"])
        self.assertIsNone(response.data["checkout"])
