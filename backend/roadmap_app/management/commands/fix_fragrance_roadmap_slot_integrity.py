from __future__ import annotations

from django.core.management.base import BaseCommand
from django.db import transaction
from django.db.models import Prefetch
from django.utils import timezone

from roadmap_app.integrity import collect_mismatched_fragrance_step_ids, is_fragrance_slot_mismatch_step
from roadmap_app.models import RoadmapPlan, RoadmapStep


class Command(BaseCommand):
    help = "Repair legacy wrong-slot fragrance roadmap recommendations in safe batches."

    def add_arguments(self, parser):
        parser.add_argument("--batch-size", type=int, default=200)
        parser.add_argument("--limit", type=int, default=None)
        parser.add_argument("--offset", type=int, default=0)
        parser.add_argument("--plan-ids", nargs="*", type=int, default=None)
        parser.add_argument("--dry-run", action="store_true", default=False)
        parser.add_argument("--only-mismatched", action="store_true", default=False)
        parser.add_argument("--progress-every", type=int, default=100)

    def handle(self, *args, **options):
        batch_size = max(1, int(options["batch_size"]))
        progress_every = max(1, int(options["progress_every"]))
        requested_plan_ids = list(dict.fromkeys(int(pid) for pid in (options.get("plan_ids") or []) if int(pid) > 0))

        selected_plan_ids = self._selected_plan_ids(
            requested_plan_ids=requested_plan_ids,
            only_mismatched=bool(options["only_mismatched"]),
            offset=max(0, int(options["offset"] or 0)),
            limit=options["limit"],
            batch_size=batch_size,
        )

        precomputed_mismatches = collect_mismatched_fragrance_step_ids(
            plan_ids=selected_plan_ids,
            active_only=True,
            chunk_size=batch_size,
        )

        scanned_plans = 0
        touched_plans = 0
        fixed_mismatches = 0
        dry_run = bool(options["dry_run"])

        for batch_index, batch_plan_ids in enumerate(self._chunks(selected_plan_ids, batch_size), start=1):
            plans = list(
                RoadmapPlan.objects.filter(
                    category=RoadmapPlan.Category.FRAGRANCE,
                    is_active=True,
                    id__in=batch_plan_ids,
                )
                .prefetch_related(
                    Prefetch(
                        "steps",
                        queryset=RoadmapStep.objects.select_related("recommended_product").order_by("step_index", "id"),
                        to_attr="prefetched_steps",
                    )
                )
                .order_by("id")
            )

            batch_touched_plan_ids: list[int] = []
            batch_updates: list[RoadmapStep] = []
            batch_now = timezone.now()

            for plan in plans:
                scanned_plans += 1
                mismatch_step_ids = set(precomputed_mismatches.get(int(plan.id), []))
                mismatch_steps = [
                    step
                    for step in (getattr(plan, "prefetched_steps", []) or [])
                    if int(step.id) in mismatch_step_ids or (not mismatch_step_ids and is_fragrance_slot_mismatch_step(step))
                ]
                mismatch_steps = [step for step in mismatch_steps if is_fragrance_slot_mismatch_step(step)]
                if not mismatch_steps:
                    if scanned_plans % progress_every == 0:
                        self.stdout.write(f"progress scanned_plans={scanned_plans} touched_plans={touched_plans}")
                    continue

                touched_plans += 1
                fixed_mismatches += len(mismatch_steps)
                batch_touched_plan_ids.append(int(plan.id))

                if not dry_run:
                    for step in mismatch_steps:
                        step.recommended_product_id = None
                        step.suggestions = []
                        step.score = None
                        if step.status == RoadmapStep.Status.RECOMMENDED:
                            step.status = RoadmapStep.Status.MISSING
                        step.updated_at = batch_now
                        batch_updates.append(step)

                if scanned_plans % progress_every == 0:
                    self.stdout.write(f"progress scanned_plans={scanned_plans} touched_plans={touched_plans}")

            if not dry_run and batch_updates:
                with transaction.atomic():
                    RoadmapStep.objects.bulk_update(
                        batch_updates,
                        ["recommended_product", "suggestions", "score", "status", "updated_at"],
                    )
                    RoadmapPlan.objects.filter(id__in=batch_touched_plan_ids).update(updated_at=batch_now)

            if plans and batch_index % 1 == 0 and scanned_plans % progress_every != 0:
                self.stdout.write(f"progress scanned_plans={scanned_plans} touched_plans={touched_plans}")

        remaining_mismatches = sum(
            len(step_ids)
            for step_ids in collect_mismatched_fragrance_step_ids(
                plan_ids=selected_plan_ids,
                active_only=True,
                chunk_size=batch_size,
            ).values()
        )

        self.stdout.write(f"scanned_plans={scanned_plans}")
        self.stdout.write(f"touched_plans={touched_plans}")
        self.stdout.write(f"fixed_mismatches={fixed_mismatches}")
        self.stdout.write(f"remaining_mismatches={remaining_mismatches}")

    @staticmethod
    def _chunks(values: list[int], size: int):
        for start in range(0, len(values), size):
            yield values[start:start + size]

    def _selected_plan_ids(
        self,
        *,
        requested_plan_ids: list[int],
        only_mismatched: bool,
        offset: int,
        limit: int | None,
        batch_size: int,
    ) -> list[int]:
        if requested_plan_ids:
            base_plan_ids = list(
                RoadmapPlan.objects.filter(
                    category=RoadmapPlan.Category.FRAGRANCE,
                    is_active=True,
                    id__in=requested_plan_ids,
                )
                .order_by("id")
                .values_list("id", flat=True)
            )
            if only_mismatched:
                mismatched = collect_mismatched_fragrance_step_ids(
                    plan_ids=base_plan_ids,
                    active_only=True,
                    chunk_size=batch_size,
                )
                base_plan_ids = sorted(int(plan_id) for plan_id in mismatched.keys())
        elif only_mismatched:
            mismatched = collect_mismatched_fragrance_step_ids(active_only=True, chunk_size=batch_size)
            base_plan_ids = sorted(int(plan_id) for plan_id in mismatched.keys())
        else:
            base_plan_ids = list(
                RoadmapPlan.objects.filter(
                    category=RoadmapPlan.Category.FRAGRANCE,
                    is_active=True,
                )
                .order_by("id")
                .values_list("id", flat=True)
            )

        if offset:
            base_plan_ids = base_plan_ids[offset:]
        if limit is not None:
            base_plan_ids = base_plan_ids[: max(0, int(limit))]
        return list(base_plan_ids)
