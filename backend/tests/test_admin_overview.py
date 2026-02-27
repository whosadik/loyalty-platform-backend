from decimal import Decimal

from django.contrib.auth import get_user_model
from rest_framework.test import APITestCase

from admin_tools.models import StaffProfile, StaffRole
from catalog.models import Product
from loyalty.models import LoyaltyAccount, Tier
from offers.models import CampaignBudget, Offer
from users_app.models import CustomerProfile


class AdminOverviewTests(APITestCase):
    def setUp(self):
        User = get_user_model()

        self.admin = User.objects.create_user(username="overview_admin", password="pass12345")
        self.admin.is_staff = True
        self.admin.save(update_fields=["is_staff"])
        StaffProfile.objects.update_or_create(
            user=self.admin,
            defaults={"role": StaffRole.ANALYST, "permissions": ["view_metrics"]},
        )

        self.user = User.objects.create_user(username="overview_user", password="pass12345")
        CustomerProfile.objects.get_or_create(user=self.user)

        bronze, _ = Tier.objects.get_or_create(name="Bronze", defaults={"threshold_spend_90d": 0, "points_rate": 1.0})
        LoyaltyAccount.objects.get_or_create(user=self.user, defaults={"tier": bronze, "points_balance": 0})

        self.product = Product.objects.create(
            name="Overview Product",
            brand="B",
            price=Decimal("14.99"),
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
            name="Overview Offer",
            offer_type=Offer.Type.DISCOUNT,
            value=Decimal("5.00"),
            estimated_cost=Decimal("3.00"),
            is_active=True,
            target_scope="cart",
            cooldown_days=0,
            expires_in_days=7,
            campaign=campaign,
        )

    def test_admin_overview_has_dashboard_sections(self):
        self.client.force_authenticate(self.user)
        nxt = self.client.get("/api/me/next-offer")
        self.assertEqual(nxt.status_code, 200)
        aid = nxt.data["assignment_id"]
        self.client.post("/api/offers/click", {"assignment_id": aid}, format="json")
        self.client.post(
            "/api/checkout",
            {
                "channel": "offline",
                "apply_assignment_id": aid,
                "items": [{"product": self.product.id, "quantity": 1}],
            },
            format="json",
        )

        self.client.force_authenticate(self.admin)
        r = self.client.get("/api/admin/overview")
        self.assertEqual(r.status_code, 200)

        self.assertTrue(r.data["ok"])
        self.assertIn("transactions", r.data)
        self.assertIn("offers", r.data)
        self.assertIn("retention", r.data)
        self.assertIn("recs", r.data)

        self.assertIn("7d", r.data["transactions"])
        self.assertIn("30d", r.data["transactions"])
        self.assertIn("7d", r.data["offers"])
        self.assertIn("30d", r.data["offers"])
        self.assertIn("promo_efficiency_30d", r.data["offers"])
        self.assertIn("redeemed_with_transaction_count", r.data["offers"]["promo_efficiency_30d"])
        self.assertIn("redeemed_with_real_transaction_count", r.data["offers"]["promo_efficiency_30d"])
        self.assertIn("redeemed_with_synthetic_transaction_count", r.data["offers"]["promo_efficiency_30d"])
        self.assertIn("experiments_30d", r.data["recs"])
