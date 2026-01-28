from datetime import timedelta
from django.core.management.base import BaseCommand
from django.conf import settings
from django.utils import timezone

from audit.models import AuditEvent


class Command(BaseCommand):
    help = "Delete audit events older than retention window."

    def add_arguments(self, parser):
        parser.add_argument("--days", type=int, default=None, help="Override retention days")
        parser.add_argument("--dry-run", action="store_true", help="Only show how many rows would be deleted")

    def handle(self, *args, **opt):
        days = opt["days"] if opt["days"] is not None else getattr(settings, "AUDIT_RETENTION_DAYS", 90)
        cutoff = timezone.now() - timedelta(days=int(days))

        qs = AuditEvent.objects.filter(created_at__lt=cutoff)
        cnt = qs.count()

        if opt["dry_run"]:
            self.stdout.write(self.style.WARNING(f"DRY RUN: would delete {cnt} audit rows older than {days} days"))
            return

        deleted, _ = qs.delete()
        self.stdout.write(self.style.SUCCESS(f"Deleted {deleted} audit rows older than {days} days"))
