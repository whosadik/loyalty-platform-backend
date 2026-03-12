from decimal import Decimal
from datetime import date
from django.contrib.auth import get_user_model
from rest_framework.test import APITestCase

from admin_tools.models import StaffProfile, StaffRole
from offers.models import CampaignBudget


class AdminCampaignsTests(APITestCase):
    def setUp(self):
        User = get_user_model()
        self.admin = User.objects.create_user(username="admin1", password="pass12345")
        self.admin.is_staff = True
        self.admin.is_superuser = True
        self.admin.save(update_fields=["is_staff", "is_superuser"])

        # staff profile with manage_campaigns
        StaffProfile.objects.update_or_create(
            user=self.admin,
            defaults={
                "role": StaffRole.ADMIN,
                "permissions": ["manage_campaigns", "manage_offers", "view_metrics", "invalidate_cache"],
            },
        )
        self.client.force_authenticate(self.admin)

    def test_list_campaigns(self):
        CampaignBudget.objects.get_or_create(name="default", defaults={"weekly_limit": Decimal("1000.00"), "weekly_spent": Decimal("0.00")})
        r = self.client.get("/api/admin/campaigns")
        self.assertEqual(r.status_code, 200)
        self.assertTrue(r.data["ok"])
        self.assertIn("results", r.data)

    def test_create_campaign(self):
        payload = {
            "name": "haircare_growth",
            "is_active": True,
            "priority": 25,
            "weekly_limit": "250.00",
            "week_start_date": "2026-03-01",
            "end_date": "2026-03-31",
            "allowed_categories": ["haircare"],
            "allowed_steps": [],
            "tiers": ["Gold", "Platinum"],
            "promo_text": "Double points on haircare essentials.",
            "banner_url": "https://cdn.example.com/campaigns/haircare-growth.jpg",
        }
        r = self.client.post("/api/admin/campaigns", payload, format="json")
        self.assertEqual(r.status_code, 201)
        self.assertTrue(r.data["ok"])
        self.assertEqual(r.data["campaign"]["name"], "haircare_growth")
        self.assertEqual(r.data["campaign"]["end_date"], "2026-03-31")
        self.assertEqual(r.data["campaign"]["tiers"], ["Gold", "Platinum"])
        self.assertEqual(r.data["campaign"]["promo_text"], "Double points on haircare essentials.")
        self.assertEqual(r.data["campaign"]["banner_url"], "https://cdn.example.com/campaigns/haircare-growth.jpg")

    def test_patch_campaign(self):
        c = CampaignBudget.objects.create(name="c1", weekly_limit=Decimal("10.00"), weekly_spent=Decimal("0.00"), priority=50, is_active=True)
        r = self.client.patch(
            f"/api/admin/campaigns/{c.id}",
            {
                "is_active": False,
                "priority": 5,
                "end_date": "2026-04-05",
                "tiers": ["Silver"],
                "promo_text": "Updated promo text",
                "banner_url": "https://cdn.example.com/campaigns/c1.jpg",
            },
            format="json",
        )
        self.assertEqual(r.status_code, 200)
        c.refresh_from_db()
        self.assertFalse(c.is_active)
        self.assertEqual(c.priority, 5)
        self.assertEqual(str(c.end_date), "2026-04-05")
        self.assertEqual(c.tiers, ["Silver"])
        self.assertEqual(c.promo_text, "Updated promo text")
        self.assertEqual(c.banner_url, "https://cdn.example.com/campaigns/c1.jpg")

    def test_patch_campaign_rejects_end_date_before_start(self):
        c = CampaignBudget.objects.create(
            name="c2",
            weekly_limit=Decimal("20.00"),
            weekly_spent=Decimal("0.00"),
            priority=50,
            is_active=True,
            week_start_date=date(2026, 3, 10),
        )
        r = self.client.patch(
            f"/api/admin/campaigns/{c.id}",
            {"end_date": "2026-03-01"},
            format="json",
        )
        self.assertEqual(r.status_code, 400)
        self.assertIn("end_date", r.data["details"])
