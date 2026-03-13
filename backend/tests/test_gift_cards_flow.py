from decimal import Decimal

from django.contrib.auth import get_user_model
from django.core import mail
from django.test import override_settings
from rest_framework.test import APITestCase

from catalog.models import Product
from gift_cards.models import GiftCard, GiftCardLedgerEntry
from loyalty.models import LoyaltyAccount, LoyaltyLedgerEntry, Tier
from transactions.models import OwnedProduct, Transaction


@override_settings(
    EMAIL_BACKEND="django.core.mail.backends.locmem.EmailBackend",
    FRONTEND_BASE_URL="http://localhost:5173",
)
class GiftCardsFlowTests(APITestCase):
    def setUp(self):
        User = get_user_model()
        self.user = User.objects.create_user(username="gift_card_u1", password="pass12345")
        self.user.email = "sender@example.com"
        self.user.save(update_fields=["email"])
        self.client.force_authenticate(self.user)

        bronze, _ = Tier.objects.get_or_create(
            name="Bronze",
            defaults={"threshold_spend_90d": Decimal("0.00"), "points_rate": Decimal("0.10")},
        )
        bronze.threshold_spend_90d = Decimal("0.00")
        bronze.points_rate = Decimal("0.10")
        bronze.save(update_fields=["threshold_spend_90d", "points_rate"])

        account, _ = LoyaltyAccount.objects.get_or_create(user=self.user, defaults={"tier": bronze, "points_balance": 0})
        account.tier = bronze
        account.points_balance = 100
        account.save(update_fields=["tier", "points_balance"])
        self.account = account

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

    def test_purchase_gift_card_creates_dedicated_transaction_without_loyalty_or_owned_side_effects(self):
        response = self.client.post(
            "/api/gift-cards/purchase",
            {
                "amount": 3000,
                "recipient_email": "friend@example.com",
                "message": "Happy birthday",
                "idempotency_key": "gift-purchase-1",
            },
            format="json",
        )

        self.assertEqual(response.status_code, 201)
        self.assertTrue(response.data["ok"])
        self.assertEqual(response.data["gross_total"], "3000.00")
        self.assertEqual(response.data["net_total"], "3000.00")
        self.assertEqual(response.data["points_earned"], 0)
        self.assertEqual(response.data["points_redeemed"], 0)
        self.assertTrue(response.data["email_sent"])
        self.assertEqual(len(mail.outbox), 1)
        self.assertEqual(mail.outbox[0].to, ["friend@example.com"])

        txn = Transaction.objects.get(id=response.data["transaction_id"])
        self.assertEqual(txn.total_amount, Decimal("3000.00"))
        self.assertEqual((txn.pricing_meta or {}).get("type"), "gift_card_purchase")
        self.assertFalse(txn.items.exists())

        card = GiftCard.objects.get(purchase_transaction=txn)
        self.assertEqual(card.initial_amount, Decimal("3000.00"))
        self.assertEqual(card.remaining_amount, Decimal("3000.00"))
        self.assertEqual(card.recipient_email, "friend@example.com")
        self.assertIsNotNone(card.sent_at)

        self.assertEqual(GiftCardLedgerEntry.objects.filter(gift_card=card, entry_type="issue").count(), 1)
        self.assertEqual(OwnedProduct.objects.filter(user=self.user).count(), 0)
        self.assertEqual(LoyaltyLedgerEntry.objects.filter(account=self.account).count(), 0)

        sent = self.client.get("/api/me/gift-cards/sent")
        self.assertEqual(sent.status_code, 200)
        self.assertEqual(sent.data["count"], 1)
        self.assertEqual(sent.data["items"][0]["recipient_email"], "friend@example.com")

    def test_checkout_preview_and_commit_apply_gift_card_before_points(self):
        card = GiftCard.objects.create(
            code="ABCDWXYZEFGHJKLM",
            purchaser=self.user,
            recipient_email="friend@example.com",
            currency="KZT",
            initial_amount=Decimal("50.00"),
            remaining_amount=Decimal("50.00"),
            status=GiftCard.Status.ACTIVE,
        )

        payload = {
            "channel": "online",
            "items": [{"product": self.product.id, "quantity": 1}],
            "gift_card_code": "ABCD-WXYZ-EFGH-JKLM",
            "redeem_points": 20,
        }

        preview = self.client.post("/api/checkout/preview", payload, format="json")
        self.assertEqual(preview.status_code, 200)
        self.assertTrue(preview.data["ok"])
        self.assertEqual(preview.data["gross_total"], "120.00")
        self.assertEqual(preview.data["net_total"], "50.00")
        self.assertEqual(preview.data["points_redeemed"], 20)
        self.assertEqual(preview.data["estimated_points_earned"], 5)
        self.assertEqual(preview.data["gift_card"]["applied_amount"], "50.00")
        self.assertEqual(preview.data["gift_card"]["balance_before"], "50.00")
        self.assertEqual(preview.data["gift_card"]["balance_after"], "0.00")

        checkout = self.client.post(
            "/api/checkout",
            {**payload, "idempotency_key": "checkout-with-gift-card-1"},
            format="json",
        )
        self.assertEqual(checkout.status_code, 201)
        self.assertTrue(checkout.data["ok"])
        self.assertEqual(checkout.data["net_total"], "50.00")
        self.assertEqual(checkout.data["points_redeemed"], 20)
        self.assertEqual(checkout.data["points_earned"], 5)
        self.assertEqual(checkout.data["gift_card"]["applied_amount"], "50.00")
        self.assertEqual(checkout.data["gift_card"]["balance_after"], "0.00")

        card.refresh_from_db()
        self.assertEqual(card.remaining_amount, Decimal("0.00"))
        self.assertEqual(card.status, GiftCard.Status.EXHAUSTED)
        self.assertEqual(GiftCardLedgerEntry.objects.filter(gift_card=card, entry_type="redeem").count(), 1)

        txn = Transaction.objects.get(id=checkout.data["transaction_id"])
        self.assertEqual(txn.total_amount, Decimal("50.00"))
        self.assertEqual((txn.pricing_meta or {}).get("gift_card", {}).get("applied_amount"), "50.00")

        self.account.refresh_from_db()
        self.assertEqual(self.account.points_balance, 85)

    def test_checkout_last_excludes_gift_card_purchase_transactions(self):
        purchase = self.client.post(
            "/api/gift-cards/purchase",
            {
                "amount": 1000,
                "recipient_email": "friend@example.com",
                "idempotency_key": "gift-purchase-last-1",
            },
            format="json",
        )
        self.assertEqual(purchase.status_code, 201)

        checkout = self.client.post(
            "/api/checkout",
            {
                "channel": "online",
                "items": [{"product": self.product.id, "quantity": 1}],
                "idempotency_key": "plain-checkout-last-1",
            },
            format="json",
        )
        self.assertEqual(checkout.status_code, 201)

        last_checkout = self.client.get("/api/checkout/last")
        self.assertEqual(last_checkout.status_code, 200)
        self.assertTrue(last_checkout.data["ok"])
        self.assertEqual(last_checkout.data["checkout"]["id"], checkout.data["transaction_id"])
        self.assertNotEqual(last_checkout.data["checkout"]["id"], purchase.data["transaction_id"])

    def test_received_gift_cards_are_visible_to_matching_recipient_account_only(self):
        User = get_user_model()
        recipient = User.objects.create_user(
            username="gift_card_recipient",
            password="pass12345",
            email="recipient@example.com",
            first_name="Aida",
        )
        stranger = User.objects.create_user(
            username="gift_card_stranger",
            password="pass12345",
            email="stranger@example.com",
        )

        received = GiftCard.objects.create(
            code="ABCDWXYZEFGHJKLM",
            purchaser=self.user,
            recipient_email="recipient@example.com",
            currency="KZT",
            initial_amount=Decimal("3000.00"),
            remaining_amount=Decimal("1800.00"),
            status=GiftCard.Status.ACTIVE,
        )
        GiftCard.objects.create(
            code="MNPRSTUVWXZYQWER",
            purchaser=self.user,
            recipient_email="nobody@example.com",
            currency="KZT",
            initial_amount=Decimal("1000.00"),
            remaining_amount=Decimal("1000.00"),
            status=GiftCard.Status.ACTIVE,
        )
        GiftCard.objects.create(
            code="ZXCVBNMASDFGHJKL",
            purchaser=recipient,
            recipient_email="recipient@example.com",
            currency="KZT",
            initial_amount=Decimal("5000.00"),
            remaining_amount=Decimal("5000.00"),
            status=GiftCard.Status.ACTIVE,
        )

        self.client.force_authenticate(recipient)
        recipient_response = self.client.get("/api/me/gift-cards/received")
        self.assertEqual(recipient_response.status_code, 200)
        self.assertTrue(recipient_response.data["ok"])
        self.assertEqual(recipient_response.data["count"], 1)
        item = recipient_response.data["items"][0]
        self.assertEqual(item["id"], received.id)
        self.assertEqual(item["sender_email"], "sender@example.com")
        self.assertEqual(item["sender_name"], "gift_card_u1")
        self.assertEqual(item["code"], "ABCD-WXYZ-EFGH-JKLM")
        self.assertEqual(item["snapshot"]["remaining_amount"], "1800.00")

        self.client.force_authenticate(stranger)
        stranger_response = self.client.get("/api/me/gift-cards/received")
        self.assertEqual(stranger_response.status_code, 200)
        self.assertEqual(stranger_response.data["count"], 0)
