from decimal import Decimal
from django.contrib.auth import get_user_model
from rest_framework.test import APITestCase

from catalog.models import Product
from offers.models import Offer, OfferAssignment, CampaignBudget
from users_app.models import CustomerProfile
from loyalty.models import Tier, LoyaltyAccount


class CheckoutOfferFlowTests(APITestCase):
    def setUp(self):
        User = get_user_model()
        self.user = User.objects.create_user(username="u2", password="pass12345")
        self.client.force_authenticate(self.user)

        CustomerProfile.objects.get_or_create(user=self.user)

        bronze, _ = Tier.objects.get_or_create(name="Bronze", defaults={"threshold_spend_90d": 0, "points_rate": 1.0})
        LoyaltyAccount.objects.get_or_create(user=self.user, defaults={"tier": bronze, "points_balance": 0})

        self.p1 = Product.objects.create(
            name="Moisturizer", brand="B", price=Decimal("9.99"),
            category="skincare", product_type="moisturizer",
            concerns=[], attrs={}, actives=[], flags=[],
            supported_skin_types=["normal"], strength="low", in_stock=True,
        )

        self.camp, _ = CampaignBudget.objects.get_or_create(
            name="default",
            defaults={"weekly_limit": Decimal("1000.00"), "weekly_spent": Decimal("0.00"), "priority": 100, "is_active": True},
        )

        self.offer = Offer.objects.create(
            name="Test Discount",
            offer_type="discount",
            value=Decimal("2.00"),
            estimated_cost=Decimal("5.00"),
            is_active=True,
            target_scope="cart",
            cooldown_days=0,
            expires_in_days=7,
            allowed_categories=["skincare"],
            allowed_product_types=[],
            campaign=self.camp,
        )

    def test_next_offer_then_checkout_redeems(self):
        # get offer
        r = self.client.get("/api/me/next-offer")
        self.assertEqual(r.status_code, 200)
        self.assertIn("assignment_id", r.data)
        aid = r.data["assignment_id"]

        # apply offer in checkout
        payload = {
            "channel": "offline",
            "apply_assignment_id": aid,
            "items": [{"product": self.p1.id, "quantity": 1}],
        }
        r2 = self.client.post("/api/checkout", payload, format="json")
        self.assertEqual(r2.status_code, 201)

        # assignment redeemed
        a = OfferAssignment.objects.get(id=aid)
        self.assertTrue(a.is_redeemed)

    def test_next_offer_reuse_active(self):
        r1 = self.client.get("/api/me/next-offer")
        r2 = self.client.get("/api/me/next-offer")
        self.assertEqual(r1.status_code, 200)
        self.assertEqual(r2.status_code, 200)
        self.assertEqual(r1.data.get("assignment_id"), r2.data.get("assignment_id"))
