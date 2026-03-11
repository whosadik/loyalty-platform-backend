from __future__ import annotations

from decimal import Decimal, ROUND_HALF_UP

from django.core.management.base import BaseCommand

from catalog.models import Product
from catalog.sale_fields import product_has_discount


DISCOUNT_SEQUENCE = (10, 15, 20, 25, 30)
_ONE_HUNDRED = Decimal("100")


class Command(BaseCommand):
    help = "Backfill demo sale metadata into products that do not have discount fields yet."

    def add_arguments(self, parser):
        parser.add_argument(
            "--limit",
            type=int,
            default=60,
            help="Maximum number of products to enrich with sale metadata.",
        )

    def handle(self, *args, **options):
        limit = max(0, int(options["limit"]))
        updated = 0

        queryset = Product.objects.exclude(price__isnull=True).order_by("id")
        for index, product in enumerate(queryset.iterator()):
            if updated >= limit:
                break
            if product_has_discount(product):
                continue

            price = Decimal(str(product.price))
            if price <= 0:
                continue

            discount = DISCOUNT_SEQUENCE[index % len(DISCOUNT_SEQUENCE)]
            original_price = (price / (Decimal("1") - (Decimal(discount) / _ONE_HUNDRED))).quantize(
                Decimal("0.01"),
                rounding=ROUND_HALF_UP,
            )
            if original_price <= price:
                continue

            raw_meta = product.raw_meta if isinstance(product.raw_meta, dict) else {}
            raw_meta = {
                **raw_meta,
                "original_price": str(original_price),
                "discount": discount,
            }
            product.raw_meta = raw_meta
            product.save(update_fields=["raw_meta", "updated_at"])
            updated += 1

        self.stdout.write(self.style.SUCCESS(f"Backfilled sale metadata for {updated} products."))
