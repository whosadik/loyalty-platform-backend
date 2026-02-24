from decimal import Decimal
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
            "allowed_categories": ["haircare"],
            "allowed_steps": [],
        }
        r = self.client.post("/api/admin/campaigns", payload, format="json")
        self.assertEqual(r.status_code, 201)
        self.assertTrue(r.data["ok"])
        self.assertEqual(r.data["campaign"]["name"], "haircare_growth")

    def test_patch_campaign(self):
        c = CampaignBudget.objects.create(name="c1", weekly_limit=Decimal("10.00"), weekly_spent=Decimal("0.00"), priority=50, is_active=True)
        r = self.client.patch(f"/api/admin/campaigns/{c.id}", {"is_active": False, "priority": 5}, format="json")
        self.assertEqual(r.status_code, 200)
        c.refresh_from_db()
        self.assertFalse(c.is_active)
        self.assertEqual(c.priority, 5)
