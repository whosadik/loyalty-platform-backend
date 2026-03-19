from __future__ import annotations

from django.core.management.base import BaseCommand

from roadmap_app.integrity import active_fragrance_runtime_integrity_counts, legacy_bad_fragrance_completion_counts


class Command(BaseCommand):
    help = "Report runtime integrity counters for fragrance roadmap next steps and completions."

    def handle(self, *args, **options):
        runtime_counts = active_fragrance_runtime_integrity_counts()
        legacy_counts = legacy_bad_fragrance_completion_counts(recent_days=30)

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
            f"bad_fragrance_completed_exact_match_count={legacy_counts['bad_fragrance_completed_exact_match_count']}"
        )
