from __future__ import annotations

from django.core.management.base import BaseCommand

from roadmap_app.integrity import legacy_bad_fragrance_completion_details


class Command(BaseCommand):
    help = "Report historical fragrance completion analytics noise without mixing it with current runtime truth."

    def handle(self, *args, **options):
        counts = legacy_bad_fragrance_completion_details(recent_days=30)
        self.stdout.write(
            f"bad_fragrance_completed_exact_match_count={counts['bad_fragrance_completed_exact_match_count']}"
        )
        self.stdout.write(
            "bad_fragrance_completed_exact_match_recent_30d="
            f"{counts['bad_fragrance_completed_exact_match_recent_30d']}"
        )
        self.stdout.write(f"affected_users_count={counts['affected_users_count']}")
        self.stdout.write(f"affected_plans_count={counts['affected_plans_count']}")
        self.stdout.write(f"step_state_drift_count={counts['step_state_drift_count']}")
        self.stdout.write(f"step_state_drift_recent_30d={counts['step_state_drift_recent_30d']}")
        self.stdout.write(
            "unresolved_missing_event_product_type_count="
            f"{counts['unresolved_missing_event_product_type_count']}"
        )
        self.stdout.write(f"legacy_bucket={counts['legacy_bucket']}")
