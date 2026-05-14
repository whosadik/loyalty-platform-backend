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

        bronze, _ = Tier.objects.get_or_create(
            name="Bronze",
            defaults={"threshold_spend_90d": 0, "points_rate": Decimal("0.10")},
        )
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

    def test_public_campaign_discount_preview_and_checkout_without_assignment(self):
        fragrance = Product.objects.create(
            name="Perfume",
            brand="B",
            price=Decimal("100.00"),
            category="fragrance",
            product_type="edp",
            concerns=[],
            attrs={},
            actives=[],
            flags=[],
            supported_skin_types=["normal"],
            strength="low",
            in_stock=True,
        )
        public_campaign = CampaignBudget.objects.create(
            name="summer_public",
            campaign_type=CampaignBudget.Type.PUBLIC,
            weekly_limit=Decimal("100.00"),
            weekly_spent=Decimal("0.00"),
            priority=1,
            is_active=True,
            allowed_categories=["fragrance"],
            banner_url="https://cdn.example.com/campaigns/summer.jpg",
        )
        Offer.objects.create(
            name="11 percent fragrance",
            offer_type="discount",
            value=Decimal("11.00"),
            estimated_cost=Decimal("0.00"),
            is_active=True,
            target_scope="category",
            cooldown_days=0,
            expires_in_days=7,
            allowed_categories=["fragrance"],
            allowed_product_types=[],
            campaign=public_campaign,
        )
        payload = {
            "channel": "online",
            "items": [{"product": fragrance.id, "quantity": 1}],
        }

        preview = self.client.post("/api/checkout/preview", payload, format="json")
        self.assertEqual(preview.status_code, 200)
        self.assertTrue(preview.data["offer_applied"])
        self.assertEqual(preview.data["gross_total"], "100.00")
        self.assertEqual(preview.data["discount_amount"], "11.00")
        self.assertEqual(preview.data["net_total"], "89.00")
        self.assertEqual(preview.data["target"], {"scope": "category", "value": "fragrance"})
        self.assertIsNone(preview.data["applied_offer"]["assignment_id"])
        self.assertTrue(preview.data["applied_offer"]["public_campaign"])

        public_campaign.refresh_from_db()
        self.assertEqual(public_campaign.weekly_spent, Decimal("0.00"))

        checkout = self.client.post("/api/checkout", payload, format="json")
        self.assertEqual(checkout.status_code, 201)
        self.assertTrue(checkout.data["offer_applied"])
        self.assertIsNone(checkout.data["offer_assignment_id"])
        self.assertEqual(checkout.data["public_campaign_id"], public_campaign.id)
        self.assertEqual(checkout.data["discount_amount"], "11.00")
        self.assertEqual(checkout.data["net_total"], "89.00")
        self.assertEqual(checkout.data["target"], {"scope": "category", "value": "fragrance"})
        self.assertTrue(checkout.data["applied_offer"]["public_campaign"])

        public_campaign.refresh_from_db()
        self.assertEqual(public_campaign.weekly_spent, Decimal("11.00"))

    def test_public_campaign_zero_budget_auto_applies_as_unlimited(self):
        fragrance = Product.objects.create(
            name="Unlimited Budget Perfume",
            brand="B",
            price=Decimal("100.00"),
            category="fragrance",
            product_type="edp",
            concerns=[],
            attrs={},
            actives=[],
            flags=[],
            supported_skin_types=["normal"],
            strength="low",
            in_stock=True,
        )
        public_campaign = CampaignBudget.objects.create(
            name="zero_budget_public",
            campaign_type=CampaignBudget.Type.PUBLIC,
            weekly_limit=Decimal("0.00"),
            weekly_spent=Decimal("0.00"),
            priority=1,
            is_active=True,
            allowed_categories=["fragrance"],
        )
        Offer.objects.create(
            name="Zero budget fragrance",
            offer_type=Offer.Type.DISCOUNT,
            value=Decimal("15.00"),
            estimated_cost=Decimal("0.00"),
            is_active=True,
            target_scope="category",
            cooldown_days=0,
            expires_in_days=7,
            allowed_categories=["fragrance"],
            campaign=public_campaign,
        )
        payload = {
            "channel": "online",
            "items": [{"product": fragrance.id, "quantity": 1}],
        }

        preview = self.client.post("/api/checkout/preview", payload, format="json")
        self.assertEqual(preview.status_code, 200)
        self.assertTrue(preview.data["offer_applied"])
        self.assertEqual(preview.data["discount_amount"], "15.00")
        self.assertEqual(preview.data["net_total"], "85.00")

    def test_campaign_without_banner_is_hidden_from_public_promotions(self):
        listing = self.client.get("/api/promotions/banners")
        self.assertEqual(listing.status_code, 200)
        self.assertNotIn(self.camp.id, [item["id"] for item in listing.data["banners"]])

        detail = self.client.get(f"/api/promotions/banners/{self.camp.id}")
        self.assertEqual(detail.status_code, 404)

    def test_public_campaign_detail_includes_matching_product_cards(self):
        matched = Product.objects.create(
            name="Brand Fragrance",
            brand="Boucheron",
            price=Decimal("100.00"),
            category="fragrance",
            product_type="edp",
            concerns=[],
            attrs={},
            actives=[],
            flags=[],
            supported_skin_types=["normal"],
            strength="low",
            in_stock=True,
        )
        Product.objects.create(
            name="Other Brand Fragrance",
            brand="Other",
            price=Decimal("100.00"),
            category="fragrance",
            product_type="edp",
            concerns=[],
            attrs={},
            actives=[],
            flags=[],
            supported_skin_types=["normal"],
            strength="low",
            in_stock=True,
        )
        Product.objects.create(
            name="Brand Skincare",
            brand="Boucheron",
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
        )
        public_campaign = CampaignBudget.objects.create(
            name="brand_fragrance_public",
            campaign_type=CampaignBudget.Type.PUBLIC,
            weekly_limit=Decimal("0.00"),
            weekly_spent=Decimal("0.00"),
            priority=1,
            is_active=True,
            allowed_brands=["Boucheron"],
            banner_url="https://cdn.example.com/campaigns/brand-fragrance.jpg",
        )
        offer = Offer.objects.create(
            name="Boucheron fragrance 20",
            offer_type=Offer.Type.DISCOUNT,
            value=Decimal("20.00"),
            estimated_cost=Decimal("0.00"),
            is_active=True,
            target_scope="category",
            cooldown_days=0,
            expires_in_days=7,
            allowed_categories=["fragrance"],
            campaign=public_campaign,
        )

        detail = self.client.get(f"/api/promotions/banners/{public_campaign.id}")
        self.assertEqual(detail.status_code, 200)
        self.assertEqual(detail.data["products_count"], 1)
        self.assertEqual(detail.data["offers"][0]["id"], offer.id)
        self.assertEqual(detail.data["offers"][0]["products_count"], 1)
        self.assertEqual([item["id"] for item in detail.data["products"]], [matched.id])
        self.assertEqual(detail.data["products"][0]["price"], "80.00")
        self.assertEqual(detail.data["products"][0]["original_price"], "100.00")
        self.assertEqual(detail.data["products"][0]["discount"], 20)
