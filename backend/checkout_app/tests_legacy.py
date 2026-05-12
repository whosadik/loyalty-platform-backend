from decimal import Decimal

from django.test import SimpleTestCase

from catalog.models import Product
from checkout_app.pricing import Line, apply_offer_to_totals


def _line(price: str = "10000.00") -> Line:
    product = Product(
        id=1,
        name="Test product",
        brand="Test",
        category="skincare",
        product_type="serum",
    )
    return Line(product=product, quantity=1, unit_price=Decimal(price))


class PricingTierMultiplierTests(SimpleTestCase):
    def test_tier_multiplier_increases_base_points(self):
        bronze = apply_offer_to_totals(
            offer_type="discount",
            offer_value=Decimal("0"),
            target={"scope": "cart"},
            lines=[_line()],
            points_rate=Decimal("0.01"),
            tier_points_multiplier=Decimal("1"),
        )
        silver = apply_offer_to_totals(
            offer_type="discount",
            offer_value=Decimal("0"),
            target={"scope": "cart"},
            lines=[_line()],
            points_rate=Decimal("0.01"),
            tier_points_multiplier=Decimal("1.5"),
        )
        gold = apply_offer_to_totals(
            offer_type="discount",
            offer_value=Decimal("0"),
            target={"scope": "cart"},
            lines=[_line()],
            points_rate=Decimal("0.01"),
            tier_points_multiplier=Decimal("2"),
        )

        self.assertEqual(bronze["estimated_points_earned"], 100)
        self.assertEqual(silver["estimated_points_earned"], 150)
        self.assertEqual(gold["estimated_points_earned"], 200)

    def test_offer_multiplier_stacks_with_tier_but_is_capped(self):
        boosted = apply_offer_to_totals(
            offer_type="points_multiplier",
            offer_value=Decimal("2"),
            target={"scope": "cart"},
            lines=[_line()],
            points_rate=Decimal("0.01"),
            tier_points_multiplier=Decimal("2"),
        )
        capped = apply_offer_to_totals(
            offer_type="points_multiplier",
            offer_value=Decimal("20"),
            target={"scope": "cart"},
            lines=[_line()],
            points_rate=Decimal("0.01"),
            tier_points_multiplier=Decimal("2"),
        )

        self.assertEqual(boosted["estimated_points_earned"], 400)
        self.assertEqual(capped["estimated_points_earned"], 1000)
