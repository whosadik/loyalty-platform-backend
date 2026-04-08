from __future__ import annotations

from django.core.management.base import BaseCommand

from roadmap_app.integrity import (
    active_fragrance_runtime_integrity_counts,
    legacy_bad_fragrance_completion_details,
)


class Command(BaseCommand):
    help = "Report fragrance runtime integrity and separate historical completion-step drift from true bad exact matches."

    def handle(self, *args, **options):
        runtime_counts = active_fragrance_runtime_integrity_counts()
        legacy_counts = legacy_bad_fragrance_completion_details(recent_days=30)

        self.stdout.write(
            f"active_fragrance_next_steps_total={runtime_counts['active_fragrance_next_steps_total']}"
        )
        self.stdout.write(
            "active_fragrance_next_steps_with_recommended_product="
            f"{runtime_counts['active_fragrance_next_steps_with_recommended_product']}"
        )
        self.stdout.write(
            f"active_fragrance_slot_mismatch_count={runtime_counts['active_fragrance_slot_mismatch_count']}"
        )
        self.stdout.write(
            "fragrance_runtime_status="
            f"{'pass' if runtime_counts['active_fragrance_slot_mismatch_count'] == 0 else 'fail'}"
        )
        self.stdout.write(
            f"bad_fragrance_completed_exact_match_count={legacy_counts['bad_fragrance_completed_exact_match_count']}"
        )
        self.stdout.write(
            "bad_fragrance_completed_exact_match_recent_30d="
            f"{legacy_counts['bad_fragrance_completed_exact_match_recent_30d']}"
        )
        self.stdout.write(f"fragrance_completed_step_state_drift_count={legacy_counts['step_state_drift_count']}")
        self.stdout.write(
            "fragrance_completed_step_state_drift_recent_30d="
            f"{legacy_counts['step_state_drift_recent_30d']}"
        )
        self.stdout.write(f"fragrance_legacy_bucket={legacy_counts['legacy_bucket']}")
