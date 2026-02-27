from decimal import Decimal

from django.contrib.auth import get_user_model
from rest_framework.test import APITestCase

from admin_tools.models import StaffProfile, StaffRole
from catalog.models import Product
from loyalty.models import LoyaltyAccount, Tier
from offers.models import CampaignBudget, Offer
from users_app.models import CustomerProfile


class AdminMetricsOffersTests(APITestCase):
    def setUp(self):
        User = get_user_model()

        self.admin = User.objects.create_user(username="metrics_admin", password="pass12345")
        self.admin.is_staff = True
        self.admin.save(update_fields=["is_staff"])
        StaffProfile.objects.update_or_create(
            user=self.admin,
            defaults={"role": StaffRole.ANALYST, "permissions": ["view_metrics"]},
        )

        self.user = User.objects.create_user(username="metrics_user", password="pass12345")
        CustomerProfile.objects.get_or_create(user=self.user)

        bronze, _ = Tier.objects.get_or_create(name="Bronze", defaults={"threshold_spend_90d": 0, "points_rate": 1.0})
        LoyaltyAccount.objects.get_or_create(user=self.user, defaults={"tier": bronze, "points_balance": 0})

        self.product = Product.objects.create(
            name="Metrics Product",
            brand="B",
            price=Decimal("19.99"),
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

        campaign, _ = CampaignBudget.objects.get_or_create(
            name="default",
            defaults={"weekly_limit": Decimal("1000.00"), "weekly_spent": Decimal("0.00"), "priority": 100, "is_active": True},
        )
        Offer.objects.create(
            name="Metrics Offer",
            offer_type=Offer.Type.DISCOUNT,
            value=Decimal("10.00"),
            estimated_cost=Decimal("5.00"),
            is_active=True,
            target_scope="cart",
            cooldown_days=0,
            expires_in_days=7,
            campaign=campaign,
        )

    def test_admin_metrics_contains_event_and_efficiency_blocks(self):
        self.client.force_authenticate(self.user)
        r = self.client.get("/api/me/next-offer")
        self.assertEqual(r.status_code, 200)
        aid = r.data["assignment_id"]

        c = self.client.post("/api/offers/click", {"assignment_id": aid}, format="json")
        self.assertEqual(c.status_code, 200)

        checkout = self.client.post(
            "/api/checkout",
            {
                "channel": "offline",
                "apply_assignment_id": aid,
                "items": [{"product": self.product.id, "quantity": 1}],
            },
            format="json",
        )
        self.assertEqual(checkout.status_code, 201)

        self.client.force_authenticate(self.admin)
        metrics = self.client.get("/api/admin/metrics")
        self.assertEqual(metrics.status_code, 200)

        offers = metrics.data["offers"]
        self.assertIn("events_kpis", offers)
        self.assertIn("promo_efficiency_30d", offers)
        self.assertGreaterEqual(offers["events_kpis"]["clicked_7d"], 1)
        self.assertGreaterEqual(offers["events_kpis"]["ctr_clicks_exposed_7d"], 0.0)
        promo = offers["promo_efficiency_30d"]
        self.assertGreaterEqual(promo["redeemed_count"], 1)
        self.assertIn("redeemed_with_transaction_count", promo)
        self.assertIn("redeemed_without_transaction_count", promo)
        self.assertIn("redeemed_with_real_transaction_count", promo)
        self.assertIn("redeemed_with_synthetic_transaction_count", promo)
        self.assertEqual(
            promo["redeemed_with_transaction_count"] + promo["redeemed_without_transaction_count"],
            promo["redeemed_count"],
        )
        self.assertGreaterEqual(promo["redeemed_with_real_transaction_count"], 1)
