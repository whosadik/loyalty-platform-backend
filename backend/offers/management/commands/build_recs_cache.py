from datetime import datetime
from django.core.management.base import BaseCommand
from django.utils import timezone
from django.core.cache import cache

from offers.services import _load_products_for_recs, _cooccurrence_90d


class Command(BaseCommand):
    help = "Precompute recommendation caches (products + co-occurrence)."

    def add_arguments(self, parser):
        parser.add_argument("--clear", action="store_true", help="Clear recs caches before building")

    def handle(self, *args, **opt):
        if opt["clear"]:
            cache.delete("recs:products:v1")
            cache.delete("recs:cooc90d:v1")
            self.stdout.write(self.style.WARNING("Cleared recs cache keys."))

        now = timezone.now()

        products = _load_products_for_recs()
        cooc = _cooccurrence_90d(now)

        self.stdout.write(self.style.SUCCESS(f"Built products cache: {len(products)} items"))
        self.stdout.write(self.style.SUCCESS(f"Built co-occurrence cache: {len(cooc)} base nodes"))
