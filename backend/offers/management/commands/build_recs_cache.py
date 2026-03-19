from django.core.management.base import BaseCommand
from django.utils import timezone
from django.core.cache import cache
from django.db import connection

from offers.services import _load_products_for_recs, _cooccurrence_90d


class Command(BaseCommand):
    help = "Precompute recommendation caches (products + co-occurrence)."

    def add_arguments(self, parser):
        parser.add_argument("--clear", action="store_true", help="Clear recs caches before building")

    def handle(self, *args, **opt):
        if opt["clear"]:
            db_name = connection.settings_dict.get("NAME", "default")
            cache.delete(f"recs:products:v1:{db_name}")
            cache.delete(f"recs:cooc90d:v1:{db_name}")
            cache.delete(f"recs:products:v2:{db_name}")
            cache.delete(f"recs:cooc90d:v2:{db_name}")
            cache.delete(f"recs:products:v3:{db_name}")
            self.stdout.write(self.style.WARNING("Cleared recs cache keys."))

        now = timezone.now()

        products = _load_products_for_recs()
        cooc = _cooccurrence_90d(now)

        self.stdout.write(self.style.SUCCESS(f"Built products cache: {len(products)} items"))
        self.stdout.write(self.style.SUCCESS(f"Built co-occurrence cache: {len(cooc)} base nodes"))
