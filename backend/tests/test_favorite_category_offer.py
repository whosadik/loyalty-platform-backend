from datetime import timedelta
from decimal import Decimal

from django.contrib.auth import get_user_model
from django.test import TestCase, override_settings
from django.utils import timezone

from catalog.models import Product
from offers.models import CampaignBudget, Offer, OfferAssignment
from offers.services import get_or_assign_next_offer
from transactions.models import Transaction, TransactionItem


@override_settings(FAVORITE_CATEGORY_WINDOW_DAYS=90, FAVORITE_CATEGORY_REASSIGN_DAYS=14)
class FavoriteCategoryOfferTests(TestCase):
    def setUp(self):
        User = get_user_model()
        self.user = User.objects.create_user(username="favorite_u1", password="pass12345")

        self.default_campaign = CampaignBudget.objects.create(
            name="default",
            weekly_limit=Decimal("1000.00"),
            weekly_spent=Decimal("0.00"),
            priority=100,
            is_active=True,
        )
        self.favorite_campaign = CampaignBudget.objects.create(
            name="favorite_category",
            weekly_limit=Decimal("1000.00"),
            weekly_spent=Decimal("0.00"),
            priority=20,
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
        self.favorite_offer = Offer.objects.create(
            name="Favorite Category Discount",
            offer_type=Offer.Type.DISCOUNT,
            value=Decimal("12.00"),
            estimated_cost=Decimal("7.00"),
            is_active=True,
            target_scope="category",
            cooldown_days=0,
            expires_in_days=10,
            campaign=self.favorite_campaign,
        )

        self.p_makeup = Product.objects.create(
            name="Lipstick",
            brand="B",
            price=Decimal("10.00"),
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
        self.p_skincare = Product.objects.create(
            name="Serum",
            brand="B",
            price=Decimal("20.00"),
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

    def _create_txn_days_ago(self, days: int, items: list[tuple[Product, int]]) -> None:
        txn = Transaction.objects.create(
            user=self.user,
            total_amount=Decimal("0.00"),
            channel="offline",
        )
        total = Decimal("0.00")
        for p, qty in items:
            TransactionItem.objects.create(
                transaction=txn,
                product=p,
                quantity=qty,
                unit_price=Decimal(str(p.price)),
            )
            total += Decimal(str(p.price)) * qty
        Transaction.objects.filter(id=txn.id).update(
            total_amount=total,
            created_at=timezone.now() - timedelta(days=days),
        )

    def test_assigns_favorite_category_offer(self):
        self._create_txn_days_ago(7, [(self.p_makeup, 3), (self.p_skincare, 1)])

        assignment = get_or_assign_next_offer(self.user, now=timezone.now(), context_steps=None, post_ctx=None)

        self.assertIsNotNone(assignment)
        self.assertEqual(assignment.offer_id, self.favorite_offer.id)
        self.assertEqual((assignment.target or {}).get("scope"), "category")
        self.assertEqual((assignment.target or {}).get("value"), "makeup")
        self.assertEqual((assignment.reason or {}).get("favorite_category"), "makeup")

    def test_falls_back_to_default_when_no_transactions(self):
        assignment = get_or_assign_next_offer(self.user, now=timezone.now(), context_steps=None, post_ctx=None)

        self.assertIsNotNone(assignment)
        self.assertEqual(assignment.offer_id, self.default_offer.id)

    def test_respects_favorite_category_reassign_cooldown(self):
        now = timezone.now()
        self._create_txn_days_ago(10, [(self.p_makeup, 2), (self.p_skincare, 1)])

        first = get_or_assign_next_offer(self.user, now=now, context_steps=None, post_ctx=None)
        self.assertIsNotNone(first)
        self.assertEqual(first.offer_id, self.favorite_offer.id)

        OfferAssignment.objects.filter(id=first.id).update(
            is_redeemed=True,
            assigned_at=now - timedelta(days=5),
        )

        second = get_or_assign_next_offer(self.user, now=timezone.now(), context_steps=None, post_ctx=None)
        self.assertIsNotNone(second)
        self.assertEqual(second.offer_id, self.default_offer.id)

    def test_skips_favorite_campaign_if_offer_category_conflicts(self):
        self.favorite_offer.allowed_categories = ["skincare"]
        self.favorite_offer.save(update_fields=["allowed_categories"])
        self._create_txn_days_ago(6, [(self.p_makeup, 4), (self.p_skincare, 1)])

        assignment = get_or_assign_next_offer(self.user, now=timezone.now(), context_steps=None, post_ctx=None)

        self.assertIsNotNone(assignment)
        self.assertEqual(assignment.offer_id, self.default_offer.id)
