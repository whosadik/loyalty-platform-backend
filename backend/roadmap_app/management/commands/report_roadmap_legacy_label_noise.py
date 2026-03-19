from __future__ import annotations

from django.core.management.base import BaseCommand

from roadmap_app.integrity import legacy_bad_fragrance_completion_counts


class Command(BaseCommand):
    help = "Report legacy roadmap completion label noise for fragrance exact-match events."

    def handle(self, *args, **options):
        counts = legacy_bad_fragrance_completion_counts(recent_days=30)
        self.stdout.write(
            f"bad_fragrance_completed_exact_match_count={counts['bad_fragrance_completed_exact_match_count']}"
        )
        self.stdout.write(
            "bad_fragrance_completed_exact_match_recent_30d="
            f"{counts['bad_fragrance_completed_exact_match_recent_30d']}"
        )
        self.stdout.write(f"affected_users_count={counts['affected_users_count']}")
        self.stdout.write(f"affected_plans_count={counts['affected_plans_count']}")
