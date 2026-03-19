from __future__ import annotations

from django.core.management.base import BaseCommand

from roadmap_app.models import RoadmapPlan, RoadmapStep
from roadmap_app.services import (
    _is_slot_consistent_fragrance_product_id,
    get_next_missing_step,
    refresh_roadmap,
)


class Command(BaseCommand):
    help = "Refresh fragrance roadmaps and report next-step slot mismatches before and after."

    def add_arguments(self, parser):
        parser.add_argument(
            "--active-only",
            action="store_true",
            default=False,
            help="Refresh only active fragrance plans.",
        )

    def handle(self, *args, **options):
        plans_qs = RoadmapPlan.objects.filter(category=RoadmapPlan.Category.FRAGRANCE).select_related("user")
        if options["active_only"]:
            plans_qs = plans_qs.filter(is_active=True)

        plans = list(plans_qs.order_by("user_id", "-is_active", "-updated_at", "-id"))
        refreshed_users: set[int] = set()
        mismatch_before = 0
        mismatch_after = 0
        refreshed_count = 0

        for plan in plans:
            if int(plan.user_id) in refreshed_users:
                continue
            refreshed_users.add(int(plan.user_id))

            step_before = get_next_missing_step(plan)
            if self._step_has_slot_mismatch(step_before):
                mismatch_before += 1

            refreshed_plan = refresh_roadmap(plan.user, category="fragrance", post_ctx=None)
            refreshed_count += 1

            step_after = get_next_missing_step(refreshed_plan)
            if self._step_has_slot_mismatch(step_after):
                mismatch_after += 1

        self.stdout.write(
            self.style.SUCCESS(
                "\n".join(
                    [
                        f"plans_refreshed={refreshed_count}",
                        f"next_step_mismatch_before={mismatch_before}",
                        f"next_step_mismatch_after={mismatch_after}",
                    ]
                )
            )
        )

    @staticmethod
    def _step_has_slot_mismatch(step: RoadmapStep | None) -> bool:
        if step is None or not step.recommended_product_id:
            return False
        return not _is_slot_consistent_fragrance_product_id(
            product_id=step.recommended_product_id,
            expected_slot=step.product_type,
        )
