from datetime import timedelta
from decimal import Decimal

from django.contrib.auth import get_user_model
from django.utils import timezone
from rest_framework.test import APITestCase

from catalog.models import Product
from loyalty.models import LoyaltyAccount, Tier
from offers.models import CampaignBudget, Offer, OfferAssignment
from users_app.models import CustomerProfile


class OfferPresentationApiTests(APITestCase):
    def setUp(self):
        user_model = get_user_model()
        self.user = user_model.objects.create_user(
            username="offer_presentation_u1",
            password="pass12345",
        )
        self.client.force_authenticate(self.user)

        CustomerProfile.objects.get_or_create(user=self.user)
        bronze, _ = Tier.objects.get_or_create(
            name="Bronze",
            defaults={"threshold_spend_90d": 0, "points_rate": 1.0},
        )
        LoyaltyAccount.objects.get_or_create(user=self.user, defaults={"tier": bronze, "points_balance": 0})

        self.product = Product.objects.create(
            name="Target Serum",
            brand="Glow Lab",
            price=Decimal("100.00"),
            category="skincare",
            product_type="serum",
            concerns=[],
            attrs={},
            actives=[],
            flags=[],
            supported_skin_types=["normal"],
            strength="low",
            in_stock=True,
            image_url="https://example.com/serum.jpg",
        )
        self.campaign, _ = CampaignBudget.objects.get_or_create(
            name="offer-presentation-default",
            defaults={
                "weekly_limit": Decimal("1000.00"),
                "weekly_spent": Decimal("0.00"),
                "priority": 100,
                "is_active": True,
            },
        )
        self.offer = Offer.objects.create(
            name="Target Serum Discount",
            offer_type=Offer.Type.DISCOUNT,
            value=Decimal("10.00"),
            estimated_cost=Decimal("5.00"),
            is_active=True,
            target_scope="product_id",
            cooldown_days=0,
            expires_in_days=7,
            campaign=self.campaign,
        )
        self.assignment = OfferAssignment.objects.create(
            user=self.user,
            offer=self.offer,
            expires_at=timezone.now() + timedelta(days=3),
            target={
                "scope": "product_id",
                "value": self.product.id,
                "category": self.product.category,
                "product_type": self.product.product_type,
            },
            reason={"segment": "active"},
            is_active=True,
            is_redeemed=False,
        )

    def test_me_offers_includes_ui_ready_presentation_fields(self):
        resp = self.client.get("/api/me/offers")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(len(resp.data), 1)

        item = resp.data[0]
        self.assertIn("presentation", item)
        self.assertTrue(item["presentation"]["badge"])
        self.assertTrue(item["presentation"]["cta_label"])
        self.assertEqual(item["presentation"]["image_url"], "https://example.com/serum.jpg")
        self.assertTrue(item["presentation"]["title"])
        self.assertTrue(item["presentation"]["description"])

    def test_next_offer_reuses_active_assignment_with_ui_ready_presentation_fields(self):
        resp = self.client.get("/api/me/next-offer")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.data["assignment_id"], self.assignment.id)
        self.assertIn("presentation", resp.data)
        self.assertEqual(resp.data["presentation"]["image_url"], "https://example.com/serum.jpg")
        self.assertTrue(resp.data["presentation"]["cta_label"])

    def test_home_promotions_returns_banner_collection_with_presentation_fields(self):
        second_product = Product.objects.create(
            name="Target Toner",
            brand="Glow Lab",
            price=Decimal("80.00"),
            category="skincare",
            product_type="toner",
            concerns=[],
            attrs={},
            actives=[],
            flags=[],
            supported_skin_types=["normal"],
            strength="low",
            in_stock=True,
            image_url="https://example.com/toner.jpg",
        )
        second_offer = Offer.objects.create(
            name="Target Toner Discount",
            offer_type=Offer.Type.DISCOUNT,
            value=Decimal("15.00"),
            estimated_cost=Decimal("4.00"),
            is_active=True,
            target_scope="product_id",
            cooldown_days=0,
            expires_in_days=7,
            campaign=self.campaign,
        )
        OfferAssignment.objects.create(
            user=self.user,
            offer=second_offer,
            expires_at=timezone.now() + timedelta(days=2),
            target={
                "scope": "product_id",
                "value": second_product.id,
                "category": second_product.category,
                "product_type": second_product.product_type,
            },
            reason={"segment": "active"},
            is_active=True,
            is_redeemed=False,
        )

        resp = self.client.get("/api/me/home-promotions?limit=2")
        self.assertEqual(resp.status_code, 200)
        self.assertTrue(resp.data["ok"])
        self.assertEqual(resp.data["count"], 2)
        self.assertEqual(resp.data["limit"], 2)
        self.assertEqual(len(resp.data["banners"]), 2)
        self.assertIn("presentation", resp.data["banners"][0])
        self.assertTrue(resp.data["banners"][0]["presentation"]["cta_label"])

    def test_home_promotions_auto_assigns_first_offer_when_no_active_assignment_exists(self):
        user_model = get_user_model()
        pending_user = user_model.objects.create_user(
            username="offer_presentation_u2",
            password="pass12345",
        )
        CustomerProfile.objects.get_or_create(user=pending_user)
        bronze = Tier.objects.get(name="Bronze")
        LoyaltyAccount.objects.get_or_create(user=pending_user, defaults={"tier": bronze, "points_balance": 0})

        onboarding_campaign = CampaignBudget.objects.create(
            name="onboarding_first_order",
            weekly_limit=Decimal("1000.00"),
            weekly_spent=Decimal("0.00"),
            priority=1,
            is_active=True,
        )
        Offer.objects.create(
            name="Welcome Discount",
            offer_type=Offer.Type.DISCOUNT,
            value=Decimal("10.00"),
            estimated_cost=Decimal("3.00"),
            is_active=True,
            target_scope="cart",
            cooldown_days=0,
            expires_in_days=7,
            campaign=onboarding_campaign,
        )

        self.client.force_authenticate(pending_user)
        resp = self.client.get("/api/me/home-promotions?limit=1")
        self.assertEqual(resp.status_code, 200)
        self.assertTrue(resp.data["ok"])
        self.assertEqual(resp.data["count"], 1)
        self.assertEqual(resp.data["limit"], 1)
        self.assertEqual(len(resp.data["banners"]), 1)
        self.assertIn("presentation", resp.data["banners"][0])
