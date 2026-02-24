from django.contrib.auth import get_user_model
from rest_framework.test import APITestCase


class ProfileCompletionBonusTests(APITestCase):
    def setUp(self):
        User = get_user_model()
        self.user = User.objects.create_user(username="bonus_u1", password="pass12345")
        self.client.force_authenticate(self.user)

    def test_profile_completion_bonus_awarded_once(self):
        r1 = self.client.put(
            "/api/me/profile",
            {
                "skin_type": "sensitive",
                "goals": [],
            },
            format="json",
        )
        self.assertEqual(r1.status_code, 200)
        self.assertTrue(r1.data["ok"])
        self.assertFalse(r1.data["profile_completion_bonus"]["completed"])
        self.assertFalse(r1.data["profile_completion_bonus"]["awarded"])
        self.assertEqual(r1.data["profile_completion_bonus"]["points_added"], 0)

        r2 = self.client.put(
            "/api/me/profile",
            {
                "skin_type": "sensitive",
                "goals": ["hydration"],
                "budget": "medium",
            },
            format="json",
        )
        self.assertEqual(r2.status_code, 200)
        self.assertTrue(r2.data["ok"])
        self.assertTrue(r2.data["profile_completion_bonus"]["completed"])
        self.assertTrue(r2.data["profile_completion_bonus"]["awarded"])
        self.assertEqual(r2.data["profile_completion_bonus"]["points_added"], 50)

        r3 = self.client.put(
            "/api/me/profile",
            {
                "skin_type": "sensitive",
                "goals": ["hydration"],
                "budget": "medium",
            },
            format="json",
        )
        self.assertEqual(r3.status_code, 200)
        self.assertTrue(r3.data["ok"])
        self.assertTrue(r3.data["profile_completion_bonus"]["completed"])
        self.assertFalse(r3.data["profile_completion_bonus"]["awarded"])
        self.assertEqual(r3.data["profile_completion_bonus"]["points_added"], 0)

        loyalty = self.client.get("/api/me/loyalty")
        self.assertEqual(loyalty.status_code, 200)
        self.assertEqual(loyalty.data["points_balance"], 50)
