from datetime import timedelta
from django.core.management.base import BaseCommand
from django.conf import settings
from django.utils import timezone

from offers.models import OfferAssignment


class Command(BaseCommand):
    help = "Expire stale offers and delete very old assignments (optional)."

    def add_arguments(self, parser):
        parser.add_argument("--delete-old", action="store_true", help="Delete old redeemed/expired assignments")
        parser.add_argument("--days", type=int, default=None, help="Override retention days for delete-old")
        parser.add_argument("--dry-run", action="store_true")

    def handle(self, *args, **opt):
        now = timezone.now()

        # 1) mark expired as redeemed (so they aren't active)
        expired_qs = OfferAssignment.objects.filter(is_redeemed=False, expires_at__isnull=False, expires_at__lte=now)
        expired_cnt = expired_qs.count()

        if opt["dry_run"]:
            self.stdout.write(self.style.WARNING(f"DRY RUN: would mark {expired_cnt} expired offers as redeemed"))
        else:
            updated = expired_qs.update(is_redeemed=True)
            self.stdout.write(self.style.SUCCESS(f"Marked {updated} expired offers as redeemed"))

        # 2) delete very old (optional)
        if opt["delete_old"]:
            days = opt["days"] if opt["days"] is not None else getattr(settings, "OFFERS_RETENTION_DAYS", 180)
            cutoff = now - timedelta(days=int(days))

            old_qs = OfferAssignment.objects.filter(assigned_at__lt=cutoff, is_redeemed=True)
            old_cnt = old_qs.count()

            if opt["dry_run"]:
                self.stdout.write(self.style.WARNING(f"DRY RUN: would delete {old_cnt} old redeemed offers older than {days} days"))
            else:
                deleted, _ = old_qs.delete()
                self.stdout.write(self.style.SUCCESS(f"Deleted {deleted} old redeemed offers older than {days} days"))
