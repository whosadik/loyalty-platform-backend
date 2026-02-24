from datetime import timedelta
from decimal import Decimal

from django.contrib.auth import get_user_model
from django.test import TestCase, override_settings
from django.utils import timezone

from offers.models import CampaignBudget, Offer, OfferAssignment
from offers.services import get_or_assign_next_offer
from transactions.models import Transaction


@override_settings(WINBACK_INACTIVITY_DAYS=30, WINBACK_REASSIGN_DAYS=30)
class Winback30DaysTests(TestCase):
    def setUp(self):
        User = get_user_model()
        self.user = User.objects.create_user(username="winback_u1", password="pass12345")

        self.default_campaign = CampaignBudget.objects.create(
            name="default",
            weekly_limit=Decimal("1000.00"),
            weekly_spent=Decimal("0.00"),
            priority=100,
            is_active=True,
        )
        self.winback_campaign = CampaignBudget.objects.create(
            name="winback_30d",
            weekly_limit=Decimal("1000.00"),
            weekly_spent=Decimal("0.00"),
            priority=10,
            is_active=True,
        )

        self.default_offer = Offer.objects.create(
            name="Default Discount",
            offer_type=Offer.Type.DISCOUNT,
            value=Decimal("3.00"),
            estimated_cost=Decimal("2.00"),
            is_active=True,
            target_scope="cart",
            cooldown_days=0,
            expires_in_days=7,
            campaign=self.default_campaign,
        )
        self.winback_offer = Offer.objects.create(
            name="Winback Discount",
            offer_type=Offer.Type.DISCOUNT,
            value=Decimal("15.00"),
            estimated_cost=Decimal("8.00"),
            is_active=True,
            target_scope="cart",
            cooldown_days=0,
            expires_in_days=7,
            campaign=self.winback_campaign,
        )

    def _create_txn_days_ago(self, days: int, total: str = "20.00") -> Transaction:
        txn = Transaction.objects.create(
            user=self.user,
            total_amount=Decimal(total),
            channel="offline",
        )
        when = timezone.now() - timedelta(days=days)
        Transaction.objects.filter(id=txn.id).update(created_at=when)
        txn.refresh_from_db()
        return txn

    def test_winback_assigned_when_inactive_30_days(self):
        self._create_txn_days_ago(40)

        assignment = get_or_assign_next_offer(self.user, now=timezone.now(), context_steps=None, post_ctx=None)

        self.assertIsNotNone(assignment)
        self.assertEqual(assignment.offer_id, self.winback_offer.id)

    def test_winback_not_assigned_for_recently_active_user(self):
        self._create_txn_days_ago(5)

        assignment = get_or_assign_next_offer(self.user, now=timezone.now(), context_steps=None, post_ctx=None)

        self.assertIsNotNone(assignment)
        self.assertEqual(assignment.offer_id, self.default_offer.id)

    def test_winback_respects_reassign_cooldown(self):
        now = timezone.now()
        self._create_txn_days_ago(45)

        first = get_or_assign_next_offer(self.user, now=now, context_steps=None, post_ctx=None)
        self.assertIsNotNone(first)
        self.assertEqual(first.offer_id, self.winback_offer.id)

        OfferAssignment.objects.filter(id=first.id).update(
            is_redeemed=True,
            assigned_at=now - timedelta(days=10),
        )

        second = get_or_assign_next_offer(self.user, now=timezone.now(), context_steps=None, post_ctx=None)

        self.assertIsNotNone(second)
        self.assertEqual(second.offer_id, self.default_offer.id)

    def test_existing_winback_assignment_invalidated_after_new_purchase(self):
        self._create_txn_days_ago(40)
        winback_assignment = get_or_assign_next_offer(self.user, now=timezone.now(), context_steps=None, post_ctx=None)
        self.assertIsNotNone(winback_assignment)
        self.assertEqual(winback_assignment.offer_id, self.winback_offer.id)

        Transaction.objects.create(user=self.user, total_amount=Decimal("30.00"), channel="offline")

        next_assignment = get_or_assign_next_offer(self.user, now=timezone.now(), context_steps=None, post_ctx=None)
        winback_assignment.refresh_from_db()

        self.assertTrue(winback_assignment.is_redeemed)
        self.assertIsNotNone(next_assignment)
        self.assertEqual(next_assignment.offer_id, self.default_offer.id)
