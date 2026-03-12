from decimal import Decimal

from django.contrib.auth import get_user_model
from rest_framework.test import APITestCase

from audit.models import AuditEvent
from catalog.models import Product
from offers.models import CampaignBudget, Offer, OfferAssignment, OfferEvent
from recs_analytics.models import RecommendationEvent
from transactions.models import Transaction


class AdminHealthViewTests(APITestCase):
    def setUp(self):
        User = get_user_model()

        self.admin = User.objects.create_user(username="health_admin", password="pass12345")
        self.admin.is_staff = True
        self.admin.save(update_fields=["is_staff"])

        self.user = User.objects.create_user(username="health_user", password="pass12345")

        self.product = Product.objects.create(
            name="Health Product",
            brand="Brand",
            price=Decimal("99.00"),
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

        self.transaction = Transaction.objects.create(
            user=self.user,
            total_amount=Decimal("99.00"),
            channel="online",
        )

        self.campaign = CampaignBudget.objects.create(
            name="health-campaign",
            weekly_limit=Decimal("1000.00"),
            weekly_spent=Decimal("0.00"),
            priority=10,
            is_active=True,
        )
        self.offer = Offer.objects.create(
            name="Health Offer",
            offer_type=Offer.Type.DISCOUNT,
            value=Decimal("10.00"),
            estimated_cost=Decimal("5.00"),
            is_active=True,
            target_scope="category",
            cooldown_days=0,
            expires_in_days=7,
            campaign=self.campaign,
        )
        self.assignment = OfferAssignment.objects.create(
            user=self.user,
            offer=self.offer,
            target={"category": "makeup"},
            is_active=True,
        )
        OfferEvent.objects.create(
            assignment=self.assignment,
            user=self.user,
            offer=self.offer,
            campaign_name=self.campaign.name,
            event_type=OfferEvent.Type.EXPOSED,
            context={},
        )

        AuditEvent.objects.create(
            user=self.admin,
            action=AuditEvent.Action.CHECKOUT_CREATED,
            entity_type="Transaction",
            entity_id=str(self.transaction.id),
            request_id="health-request",
            path="/api/checkout",
            method="POST",
            status_code=200,
            ip="127.0.0.1",
            meta={},
        )

        RecommendationEvent.objects.create(
            user=self.user,
            product=self.product,
            action=RecommendationEvent.Action.IMPRESSION,
            page="home",
            section_key="for_you",
            request_id="health-rec-request",
            algo_mode="recommend",
            context={},
        )

    def test_admin_health_returns_live_services_snapshot(self):
        self.client.force_authenticate(self.admin)

        response = self.client.get("/api/admin/health")
        self.assertEqual(response.status_code, 200)

        self.assertEqual(response.data["overall_status"], "ok")
        self.assertEqual(response.data["summary"]["total_services"], 6)
        self.assertEqual(response.data["counts"]["transactions"], 1)
        self.assertEqual(response.data["counts"]["offer_assignments"], 1)
        self.assertEqual(response.data["counts"]["offer_events"], 1)
        self.assertEqual(response.data["counts"]["audit_events"], 1)
        self.assertEqual(response.data["counts"]["recommendation_events"], 1)
        self.assertIn("uptime_seconds", response.data["app"])

        services = {service["name"]: service for service in response.data["services"]}
        self.assertSetEqual(
            set(services.keys()),
            {"db", "cache", "transactions", "offers", "audit", "recommendations"},
        )

        self.assertEqual(services["db"]["status"], "ok")
        self.assertEqual(services["cache"]["status"], "ok")
        self.assertEqual(services["transactions"]["status"], "ok")
        self.assertEqual(services["offers"]["status"], "ok")
        self.assertEqual(services["audit"]["status"], "ok")
        self.assertEqual(services["recommendations"]["status"], "ok")
        self.assertGreaterEqual(float(services["db"]["latency_ms"]), 0.0)
        self.assertGreaterEqual(float(services["transactions"]["latency_ms"]), 0.0)
