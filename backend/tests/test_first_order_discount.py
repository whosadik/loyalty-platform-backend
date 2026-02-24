from decimal import Decimal

from django.contrib.auth import get_user_model
from django.test import TestCase
from django.utils import timezone

from offers.models import CampaignBudget, Offer
from offers.services import get_or_assign_next_offer
from transactions.models import Transaction


class FirstOrderDiscountRulesTests(TestCase):
    def setUp(self):
        User = get_user_model()
        self.user = User.objects.create_user(username="first_order_u1", password="pass12345")

        self.onboarding_campaign = CampaignBudget.objects.create(
            name="onboarding_first_order",
            weekly_limit=Decimal("1000.00"),
            weekly_spent=Decimal("0.00"),
            priority=1,
            is_active=True,
        )

        self.onboarding_offer = Offer.objects.create(
            name="First Order Discount",
            offer_type="discount",
            value=Decimal("10.00"),
            estimated_cost=Decimal("5.00"),
            is_active=True,
            target_scope="cart",
            cooldown_days=0,
            expires_in_days=7,
            campaign=self.onboarding_campaign,
        )

    def test_assigns_onboarding_offer_for_user_without_transactions(self):
        assignment = get_or_assign_next_offer(user=self.user, now=timezone.now(), context_steps=None, post_ctx=None)

        self.assertIsNotNone(assignment)
        self.assertEqual(assignment.offer_id, self.onboarding_offer.id)
        self.assertFalse(assignment.is_redeemed)

    def test_does_not_assign_onboarding_offer_when_user_has_transaction(self):
        Transaction.objects.create(user=self.user, total_amount=Decimal("20.00"), channel="offline")

        assignment = get_or_assign_next_offer(user=self.user, now=timezone.now(), context_steps=None, post_ctx=None)

        self.assertIsNone(assignment)

    def test_invalidates_existing_onboarding_assignment_after_first_purchase(self):
        assignment = get_or_assign_next_offer(user=self.user, now=timezone.now(), context_steps=None, post_ctx=None)
        self.assertIsNotNone(assignment)

        Transaction.objects.create(user=self.user, total_amount=Decimal("35.00"), channel="offline")

        next_assignment = get_or_assign_next_offer(user=self.user, now=timezone.now(), context_steps=None, post_ctx=None)

        assignment.refresh_from_db()
        self.assertTrue(assignment.is_redeemed)
        self.assertIsNone(next_assignment)
