from datetime import timedelta
from decimal import Decimal

from django.core.management import call_command
from django.utils import timezone
from django.contrib.auth import get_user_model
from rest_framework.test import APITestCase

from catalog.models import Product
from loyalty.models import LoyaltyAccount, Tier
from offers.admin_metrics import offers_events_kpis
from offers.models import CampaignBudget, Offer, OfferAssignment, OfferEvent
from users_app.models import CustomerProfile


class OfferEventsFlowTests(APITestCase):
    def setUp(self):
        User = get_user_model()
        self.user = User.objects.create_user(username="events_u1", password="pass12345")
        self.client.force_authenticate(self.user)

        CustomerProfile.objects.get_or_create(user=self.user)

        bronze, _ = Tier.objects.get_or_create(name="Bronze", defaults={"threshold_spend_90d": 0, "points_rate": 1.0})
        LoyaltyAccount.objects.get_or_create(user=self.user, defaults={"tier": bronze, "points_balance": 0})

        self.product = Product.objects.create(
            name="Hydrating Serum",
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

        self.default_campaign, _ = CampaignBudget.objects.get_or_create(
            name="default",
            defaults={"weekly_limit": Decimal("1000.00"), "weekly_spent": Decimal("0.00"), "priority": 100, "is_active": True},
        )
        Offer.objects.create(
            name="Events Test Discount",
            offer_type=Offer.Type.DISCOUNT,
            value=Decimal("5.00"),
            estimated_cost=Decimal("3.00"),
            is_active=True,
            target_scope="cart",
            cooldown_days=0,
            expires_in_days=7,
            campaign=self.default_campaign,
        )

    def test_next_offer_writes_exposed_per_request(self):
        r1 = self.client.get("/api/me/next-offer")
        r2 = self.client.get("/api/me/next-offer")

        self.assertEqual(r1.status_code, 200)
        self.assertEqual(r2.status_code, 200)
        aid = r1.data["assignment_id"]

        self.assertEqual(
            OfferEvent.objects.filter(assignment_id=aid, event_type=OfferEvent.Type.ASSIGNED).count(),
            1,
        )
        self.assertEqual(
            OfferEvent.objects.filter(assignment_id=aid, event_type=OfferEvent.Type.EXPOSED).count(),
            2,
        )

    def test_next_offer_exposed_idempotent_with_same_request_id(self):
        rid = "same-request-id-1"
        r1 = self.client.get("/api/me/next-offer", HTTP_X_REQUEST_ID=rid)
        r2 = self.client.get("/api/me/next-offer", HTTP_X_REQUEST_ID=rid)

        self.assertEqual(r1.status_code, 200)
        self.assertEqual(r2.status_code, 200)
        aid = r1.data["assignment_id"]
        self.assertEqual(
            OfferEvent.objects.filter(assignment_id=aid, event_type=OfferEvent.Type.EXPOSED).count(),
            1,
        )

    def test_checkout_writes_redeemed_event(self):
        r = self.client.get("/api/me/next-offer")
        self.assertEqual(r.status_code, 200)
        aid = r.data["assignment_id"]

        payload = {
            "channel": "offline",
            "apply_assignment_id": aid,
            "items": [{"product": self.product.id, "quantity": 1}],
        }
        checkout = self.client.post("/api/checkout", payload, format="json")
        self.assertEqual(checkout.status_code, 201)

        self.assertEqual(
            OfferEvent.objects.filter(assignment_id=aid, event_type=OfferEvent.Type.REDEEMED).count(),
            1,
        )

    def test_offer_click_writes_clicked_per_request(self):
        r = self.client.get("/api/me/next-offer")
        self.assertEqual(r.status_code, 200)
        aid = r.data["assignment_id"]

        c1 = self.client.post("/api/offers/click", {"assignment_id": aid}, format="json")
        c2 = self.client.post("/api/offers/click", {"assignment_id": aid}, format="json")
        self.assertEqual(c1.status_code, 200)
        self.assertEqual(c2.status_code, 200)
        self.assertTrue(c1.data["clicked_recorded"])
        self.assertTrue(c2.data["clicked_recorded"])

        self.assertEqual(
            OfferEvent.objects.filter(assignment_id=aid, event_type=OfferEvent.Type.CLICKED).count(),
            2,
        )

        kpis = offers_events_kpis()
        self.assertGreaterEqual(kpis["clicked_7d"], 1)
        self.assertGreaterEqual(kpis["ctr_clicks_exposed_7d"], 0.0)

    def test_offer_click_idempotent_with_same_request_id(self):
        r = self.client.get("/api/me/next-offer")
        self.assertEqual(r.status_code, 200)
        aid = r.data["assignment_id"]

        rid = "same-click-request-id"
        c1 = self.client.post("/api/offers/click", {"assignment_id": aid}, format="json", HTTP_X_REQUEST_ID=rid)
        c2 = self.client.post("/api/offers/click", {"assignment_id": aid}, format="json", HTTP_X_REQUEST_ID=rid)
        self.assertEqual(c1.status_code, 200)
        self.assertEqual(c2.status_code, 200)
        self.assertTrue(c1.data["clicked_recorded"])
        self.assertFalse(c2.data["clicked_recorded"])
        self.assertEqual(
            OfferEvent.objects.filter(assignment_id=aid, event_type=OfferEvent.Type.CLICKED).count(),
            1,
        )

    def test_cleanup_offers_writes_expired_event(self):
        r = self.client.get("/api/me/next-offer")
        self.assertEqual(r.status_code, 200)
        aid = r.data["assignment_id"]

        OfferAssignment.objects.filter(id=aid).update(expires_at=timezone.now() - timedelta(minutes=1))

        call_command("cleanup_offers")

        a = OfferAssignment.objects.get(id=aid)
        self.assertTrue(a.is_redeemed)
        self.assertEqual(
            OfferEvent.objects.filter(assignment_id=aid, event_type=OfferEvent.Type.EXPIRED).count(),
            1,
        )
