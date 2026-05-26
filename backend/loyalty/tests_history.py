from django.contrib.auth import get_user_model
from rest_framework.test import APITestCase

from loyalty.models import LoyaltyAccount, LoyaltyLedgerEntry


class LoyaltyHistoryTests(APITestCase):
    def setUp(self):
        User = get_user_model()
        self.user = User.objects.create_user(username="ledger_u1", password="pass12345")
        self.client.force_authenticate(self.user)

    def test_profile_completion_bonus_appears_in_history(self):
        profile_payload = {
            "skin_type": "sensitive",
            "goals": ["hydration"],
            "budget": "medium",
        }
        update = self.client.put("/api/me/profile", profile_payload, format="json")
        self.assertEqual(update.status_code, 200)
        self.assertTrue(update.data["profile_completion_bonus"]["awarded"])

        resp = self.client.get("/api/me/loyalty/history")
        self.assertEqual(resp.status_code, 200)
        self.assertTrue(resp.data["ok"])
        self.assertEqual(resp.data["points_balance"], 50)

        results = resp.data["results"]
        self.assertEqual(len(results), 1)
        entry = results[0]
        self.assertEqual(entry["entry_type"], "earn")
        self.assertEqual(entry["points_delta"], 50)
        self.assertEqual(entry["kind"], "profile_completion")
        self.assertEqual(entry["description"], "Бонус за заполнение профиля")
        self.assertIn("profile_completion", entry["reference"])

    def test_history_filters_by_entry_type(self):
        account, _ = LoyaltyAccount.objects.get_or_create(user=self.user)
        LoyaltyLedgerEntry.objects.create(
            account=account,
            entry_type=LoyaltyLedgerEntry.Type.EARN,
            points_delta=30,
            reference="roadmap_step:7",
        )
        LoyaltyLedgerEntry.objects.create(
            account=account,
            entry_type=LoyaltyLedgerEntry.Type.REDEEM,
            points_delta=-10,
            reference="manual_redeem",
        )

        resp_earn = self.client.get("/api/me/loyalty/history?entry_type=earn")
        self.assertEqual(resp_earn.status_code, 200)
        self.assertEqual(len(resp_earn.data["results"]), 1)
        self.assertEqual(resp_earn.data["results"][0]["kind"], "roadmap_step")

        resp_redeem = self.client.get("/api/me/loyalty/history?entry_type=redeem")
        self.assertEqual(resp_redeem.status_code, 200)
        self.assertEqual(len(resp_redeem.data["results"]), 1)
        self.assertEqual(resp_redeem.data["results"][0]["kind"], "manual_redeem")
        self.assertEqual(resp_redeem.data["results"][0]["points_delta"], -10)

    def test_history_requires_auth(self):
        self.client.force_authenticate(None)
        resp = self.client.get("/api/me/loyalty/history")
        self.assertIn(resp.status_code, (401, 403))

    def test_transaction_earn_kind_classified(self):
        account, _ = LoyaltyAccount.objects.get_or_create(user=self.user)
        LoyaltyLedgerEntry.objects.create(
            account=account,
            entry_type=LoyaltyLedgerEntry.Type.EARN,
            points_delta=15,
            reference="txn:42",
        )
        resp = self.client.get("/api/me/loyalty/history")
        self.assertEqual(resp.status_code, 200)
        results = resp.data["results"]
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["kind"], "txn_earn")
        self.assertEqual(results[0]["transaction_id"], 42)
