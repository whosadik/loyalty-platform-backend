from __future__ import annotations

from decimal import Decimal, ROUND_HALF_UP

from django.core.management.base import BaseCommand

from catalog.models import Product
from catalog.product_metrics import get_product_rating, get_product_reviews_count


class Command(BaseCommand):
    help = "Backfill demo rating and reviews_count into products missing social-proof metadata."

    def add_arguments(self, parser):
        parser.add_argument(
            "--limit",
            type=int,
            default=120,
            help="Maximum number of products to enrich with rating/reviews_count metadata.",
        )

    def handle(self, *args, **options):
        limit = max(0, int(options["limit"]))
        updated = 0

        queryset = Product.objects.order_by("id")
        for product in queryset.iterator():
            if updated >= limit:
                break
            if get_product_rating(product) is not None and get_product_reviews_count(product) > 0:
                continue

            base_rating = Decimal("4.1") + (Decimal(product.id % 9) * Decimal("0.1"))
            rating = min(Decimal("4.9"), base_rating).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP)
            reviews_count = 12 + (product.id % 89)

            raw_meta = product.raw_meta if isinstance(product.raw_meta, dict) else {}
            raw_meta = {
                **raw_meta,
                "rating": str(rating),
                "reviews_count": reviews_count,
            }
            product.raw_meta = raw_meta
            product.save(update_fields=["raw_meta", "updated_at"])
            updated += 1

        self.stdout.write(self.style.SUCCESS(f"Backfilled product social proof for {updated} products."))
