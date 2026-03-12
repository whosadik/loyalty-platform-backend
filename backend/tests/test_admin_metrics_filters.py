from datetime import timedelta
from decimal import Decimal

from django.contrib.auth import get_user_model
from django.utils import timezone
from rest_framework.test import APITestCase

from admin_tools.models import StaffProfile, StaffRole
from catalog.models import Product
from offers.models import CampaignBudget, Offer, OfferAssignment, OfferEvent
from transactions.models import Transaction, TransactionItem
from users_app.models import CustomerProfile


class AdminMetricsFiltersTests(APITestCase):
    def setUp(self):
        User = get_user_model()

        self.admin = User.objects.create_user(username="metrics_filters_admin", password="pass12345")
        self.admin.is_staff = True
        self.admin.save(update_fields=["is_staff"])
        StaffProfile.objects.update_or_create(
            user=self.admin,
            defaults={"role": StaffRole.ANALYST, "permissions": ["view_metrics"]},
        )

        self.user = User.objects.create_user(username="metrics_filters_user", password="pass12345")
        CustomerProfile.objects.get_or_create(user=self.user)

        self.product_makeup = Product.objects.create(
            name="Filter Makeup",
            brand="B",
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
        self.product_skincare = Product.objects.create(
            name="Filter Skincare",
            brand="B",
            price=Decimal("89.00"),
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

        campaign, _ = CampaignBudget.objects.get_or_create(
            name="default",
            defaults={"weekly_limit": Decimal("1000.00"), "weekly_spent": Decimal("0.00"), "priority": 100, "is_active": True},
        )
        self.offer_discount = Offer.objects.create(
            name="Filter Discount",
            offer_type=Offer.Type.DISCOUNT,
            value=Decimal("10.00"),
            estimated_cost=Decimal("5.00"),
            is_active=True,
            target_scope="category",
            cooldown_days=0,
            expires_in_days=7,
            campaign=campaign,
        )
        self.offer_points = Offer.objects.create(
            name="Filter Points",
            offer_type=Offer.Type.POINTS_MULTIPLIER,
            value=Decimal("2.00"),
            estimated_cost=Decimal("4.00"),
            is_active=True,
            target_scope="category",
            cooldown_days=0,
            expires_in_days=7,
            campaign=campaign,
        )

        now = timezone.now()
        self.recent_dt = now - timedelta(days=2)
        self.old_dt = now - timedelta(days=40)

        self.tx_offline = Transaction.objects.create(
            user=self.user,
            total_amount=Decimal("99.00"),
            channel="offline",
        )
        self.tx_online = Transaction.objects.create(
            user=self.user,
            total_amount=Decimal("89.00"),
            channel="online",
        )
        TransactionItem.objects.create(
            transaction=self.tx_offline,
            product=self.product_makeup,
            quantity=1,
            unit_price=Decimal("99.00"),
        )
        TransactionItem.objects.create(
            transaction=self.tx_online,
            product=self.product_skincare,
            quantity=1,
            unit_price=Decimal("89.00"),
        )
        Transaction.objects.filter(id=self.tx_offline.id).update(created_at=self.recent_dt)
        Transaction.objects.filter(id=self.tx_online.id).update(created_at=self.old_dt)

        self.assignment_makeup = OfferAssignment.objects.create(
            user=self.user,
            offer=self.offer_discount,
            is_redeemed=True,
            redeemed_transaction_id=self.tx_offline.id,
            target={"category": "makeup"},
        )
        self.assignment_skincare = OfferAssignment.objects.create(
            user=self.user,
            offer=self.offer_points,
            is_redeemed=True,
            redeemed_transaction_id=self.tx_online.id,
            target={"category": "skincare"},
        )
        OfferAssignment.objects.filter(id=self.assignment_makeup.id).update(assigned_at=self.recent_dt)
        OfferAssignment.objects.filter(id=self.assignment_skincare.id).update(assigned_at=self.old_dt)

        self._seed_offer_events(self.assignment_makeup, self.recent_dt)
        self._seed_offer_events(self.assignment_skincare, self.old_dt)

    def _seed_offer_events(self, assignment: OfferAssignment, created_at):
        for event_type in (
            OfferEvent.Type.EXPOSED,
            OfferEvent.Type.CLICKED,
            OfferEvent.Type.REDEEMED,
        ):
            event = OfferEvent.objects.create(
                assignment=assignment,
                user=assignment.user,
                offer=assignment.offer,
                campaign_name="default",
                event_type=event_type,
                context={},
            )
            OfferEvent.objects.filter(id=event.id).update(created_at=created_at)

    def test_admin_metrics_applies_filters(self):
        self.client.force_authenticate(self.admin)

        all_metrics = self.client.get("/api/admin/metrics")
        self.assertEqual(all_metrics.status_code, 200)
        self.assertEqual(all_metrics.data["offers"]["assignments_total"], 2)
        self.assertGreaterEqual(len(all_metrics.data["series"]), 1)
        self.assertEqual(
            {row["channel"] for row in all_metrics.data["channels"]},
            {"offline"},
        )

        discount_metrics = self.client.get("/api/admin/metrics", {"offer_type": "discount"})
        self.assertEqual(discount_metrics.status_code, 200)
        self.assertEqual(discount_metrics.data["offers"]["assignments_total"], 1)

        category_metrics = self.client.get("/api/admin/metrics", {"category": "makeup"})
        self.assertEqual(category_metrics.status_code, 200)
        self.assertEqual(category_metrics.data["offers"]["assignments_total"], 1)

        channel_metrics = self.client.get("/api/admin/metrics", {"channel": "offline"})
        self.assertEqual(channel_metrics.status_code, 200)
        self.assertEqual(channel_metrics.data["offers"]["assignments_total"], 1)
        self.assertEqual(len(channel_metrics.data["channels"]), 1)
        self.assertEqual(channel_metrics.data["channels"][0]["channel"], "offline")
        self.assertEqual(channel_metrics.data["channels"][0]["transactions"], 1)
        self.assertEqual(channel_metrics.data["channels"][0]["offer_redemptions"], 1)
        self.assertEqual(len(channel_metrics.data["series"]), 1)
        self.assertEqual(channel_metrics.data["series"][0]["transactions"], 1)

        date_from = (self.recent_dt - timedelta(days=1)).date().isoformat()
        date_to = timezone.now().date().isoformat()
        date_metrics = self.client.get(
            "/api/admin/metrics",
            {"date_from": date_from, "date_to": date_to},
        )
        self.assertEqual(date_metrics.status_code, 200)
        self.assertEqual(date_metrics.data["offers"]["assignments_total"], 1)

    def test_admin_metrics_export_csv_applies_filters(self):
        self.client.force_authenticate(self.admin)

        response = self.client.get("/api/admin/metrics/export", {"offer_type": "discount"})
        self.assertEqual(response.status_code, 200)
        self.assertIn("text/csv", response["Content-Type"])
        self.assertIn("attachment; filename=", response["Content-Disposition"])

        content = response.content.decode("utf-8-sig")
        self.assertIn("section;metric;value", content)
        self.assertIn("summary;assignments_total;1", content)
        self.assertIn("summary;redemptions_total;1", content)
        self.assertIn("channels;offline_transactions;1", content)
        self.assertIn("series;", content)
