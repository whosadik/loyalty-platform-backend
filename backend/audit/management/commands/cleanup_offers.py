from datetime import timedelta
from django.core.management.base import BaseCommand
from django.conf import settings
from django.db.models import Q
from django.utils import timezone

from offers.models import OfferAssignment
from offers.services import (
    deactivate_stale_roadmap_assignment,
    enforce_assignment_roadmap_reason_contract,
    expire_assignment_if_needed,
)


class Command(BaseCommand):
    help = "Deactivate expired offers, sanitize stale roadmap assignments, and optionally delete old inactive history."

    def add_arguments(self, parser):
        parser.add_argument("--delete-old", action="store_true", help="Delete old redeemed/expired assignments")
        parser.add_argument("--days", type=int, default=None, help="Override retention days for delete-old")
        parser.add_argument(
            "--sanitize-roadmap",
            action="store_true",
            help="Deactivate stale roadmap_shortcut assignments and normalize misleading roadmap reason payloads",
        )
        parser.add_argument(
            "--skip-expired",
            action="store_true",
            help="Skip the expired-offer cleanup portion and run only the explicitly requested cleanup flags",
        )
        parser.add_argument("--dry-run", action="store_true")

    def handle(self, *args, **opt):
        now = timezone.now()

        # 1) deactivate expired offers so persisted state matches runtime semantics
        if not opt["skip_expired"]:
            expired_qs = (
                OfferAssignment.objects
                .select_related("offer", "offer__campaign", "user")
                .filter(is_active=True, expires_at__isnull=False, expires_at__lte=now)
                .order_by("id")
            )
            expired_cnt = expired_qs.count()

            if opt["dry_run"]:
                self.stdout.write(self.style.WARNING(f"DRY RUN: would deactivate {expired_cnt} expired active offers"))
            else:
                updated = 0
                for a in expired_qs.iterator():
                    if expire_assignment_if_needed(
                        a,
                        now=now,
                        source="cleanup_offers",
                    ):
                        updated += 1
                self.stdout.write(self.style.SUCCESS(f"Deactivated {updated} expired active offers"))

        if opt["sanitize_roadmap"]:
            active_qs = (
                OfferAssignment.objects
                .select_related("offer", "offer__campaign", "user")
                .filter(is_active=True, is_redeemed=False)
                .order_by("id")
            )
            deactivated = 0
            sanitized = 0
            for assignment in active_qs.iterator():
                if deactivate_stale_roadmap_assignment(assignment, now=now, save=not opt["dry_run"]):
                    deactivated += 1
                    continue

                if opt["dry_run"]:
                    clone = OfferAssignment(
                        id=assignment.id,
                        user=assignment.user,
                        offer=assignment.offer,
                        reason=assignment.reason,
                        target=assignment.target,
                    )
                    if enforce_assignment_roadmap_reason_contract(clone, save=False):
                        sanitized += 1
                else:
                    if enforce_assignment_roadmap_reason_contract(assignment, save=True):
                        sanitized += 1

            prefix = "DRY RUN: would" if opt["dry_run"] else "Applied"
            self.stdout.write(
                self.style.SUCCESS(
                    f"{prefix} roadmap cleanup: deactivated={deactivated}, sanitized_reason={sanitized}"
                )
            )

        # 2) delete very old inactive/redeemed history (optional)
        if opt["delete_old"]:
            days = opt["days"] if opt["days"] is not None else getattr(settings, "OFFERS_RETENTION_DAYS", 180)
            cutoff = now - timedelta(days=int(days))

            old_qs = OfferAssignment.objects.filter(assigned_at__lt=cutoff).filter(
                Q(is_redeemed=True)
                | Q(is_active=False, expires_at__isnull=False, expires_at__lt=cutoff)
            )
            old_cnt = old_qs.count()

            if opt["dry_run"]:
                self.stdout.write(
                    self.style.WARNING(f"DRY RUN: would delete {old_cnt} old inactive/redeemed offers older than {days} days")
                )
            else:
                deleted, _ = old_qs.delete()
                self.stdout.write(self.style.SUCCESS(f"Deleted {deleted} old inactive/redeemed offers older than {days} days"))
