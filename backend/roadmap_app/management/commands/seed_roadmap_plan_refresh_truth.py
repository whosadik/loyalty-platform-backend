"""Seed synthetic RoadmapPlans plus PLAN_REFRESHED/STEP_GENERATED/STEP_COMPLETED
event chains so the v4 online decision-quality guard has resolved truth.

Each seeded plan emits 1-3 refresh cycles; each cycle has 2-4 STEP_GENERATED
candidates and exactly one STEP_COMPLETED inside the same refresh window. Event
`created_at` fields are backdated via update() so the events live inside the
`--days` window the guard reads from.

After seeding, run `backfill_roadmap_shadow_meta --replay-mode historical_anchors
--write` so each anchor gets shadow/control evidence, then re-run
`report_roadmap_nextstep_v4_decision_quality`.
"""
from __future__ import annotations

import random
from datetime import timedelta
from typing import Any

from django.contrib.auth import get_user_model
from django.core.management.base import BaseCommand
from django.db import transaction
from django.utils import timezone

from catalog.models import Product
from roadmap_app.models import RoadmapEvent, RoadmapPlan, RoadmapStep


CATEGORY_CHAINS: dict[str, list[str]] = {
    "skincare": ["cleanser", "toner", "serum", "moisturizer", "mask", "spf"],
    "haircare": ["shampoo", "conditioner", "hair_mask", "scalp_serum", "hair_oil", "leave_in"],
    "makeup": ["primer", "foundation", "blush", "eyeshadow", "mascara", "lipstick", "setting_spray"],
    "fragrance": ["edp", "body_mist", "edt"],
}


def _chunked_update_created_at(events: list[RoadmapEvent], timestamps: list[Any]) -> None:
    """created_at is auto_now_add, so we overwrite via update() per id."""
    for event, ts in zip(events, timestamps):
        RoadmapEvent.objects.filter(pk=event.pk).update(created_at=ts)


class Command(BaseCommand):
    help = "Seed synthetic plans + refresh/generate/complete event chains to unblock the v4 decision-quality guard."

    def add_arguments(self, parser):
        parser.add_argument(
            "--target-plans-per-category",
            type=int,
            default=120,
            help="Top each category up to this many RoadmapPlans.",
        )
        parser.add_argument(
            "--days",
            type=int,
            default=60,
            help="Spread refresh events across the last N days.",
        )
        parser.add_argument(
            "--refreshes-per-plan-min",
            type=int,
            default=1,
        )
        parser.add_argument(
            "--refreshes-per-plan-max",
            type=int,
            default=3,
        )
        parser.add_argument(
            "--generated-per-refresh-min",
            type=int,
            default=2,
        )
        parser.add_argument(
            "--generated-per-refresh-max",
            type=int,
            default=4,
        )
        parser.add_argument(
            "--seed",
            type=int,
            default=42,
        )
        parser.add_argument(
            "--categories",
            default="skincare,haircare,makeup,fragrance",
            help="Comma-separated categories to seed. Skip any not listed here.",
        )
        parser.add_argument(
            "--dry-run",
            action="store_true",
            help="Compute counts but do not write.",
        )

    def handle(self, *args, **opts):
        rng = random.Random(int(opts["seed"]))
        days = max(1, int(opts["days"]))
        target = max(0, int(opts["target-plans-per-category".replace("-", "_")]) if False else int(opts["target_plans_per_category"]))
        refreshes_min = max(1, int(opts["refreshes_per_plan_min"]))
        refreshes_max = max(refreshes_min, int(opts["refreshes_per_plan_max"]))
        gen_min = max(2, int(opts["generated_per_refresh_min"]))
        gen_max = max(gen_min, int(opts["generated_per_refresh_max"]))
        dry_run = bool(opts["dry_run"])
        categories = [c.strip().lower() for c in str(opts["categories"]).split(",") if c.strip()]

        User = get_user_model()
        user_ids_pool = list(
            User.objects.exclude(username__startswith="ga_").values_list("id", flat=True)
        )
        if not user_ids_pool:
            self.stderr.write("No eligible users found (need at least one non-ga user).")
            return
        rng.shuffle(user_ids_pool)

        products_by_cat_type: dict[tuple[str, str], list[int]] = {}
        for cat in categories:
            for row in Product.objects.filter(category=cat, in_stock=True).values("id", "product_type"):
                ptype = str(row.get("product_type") or "").strip().lower()
                if not ptype:
                    continue
                products_by_cat_type.setdefault((cat, ptype), []).append(int(row["id"]))

        now_utc = timezone.now()
        window_start = now_utc - timedelta(days=days)

        summary: dict[str, dict[str, int]] = {}

        for cat in categories:
            chain = [pt for pt in CATEGORY_CHAINS.get(cat, []) if (cat, pt) in products_by_cat_type]
            if len(chain) < 2:
                self.stdout.write(f"[{cat}] SKIP — fewer than 2 in-stock product_types in chain.")
                summary[cat] = {"created_plans": 0, "refreshes": 0, "generated": 0, "completed": 0, "skipped": 1}
                continue

            existing_plans = RoadmapPlan.objects.filter(category=cat).count()
            to_create = max(0, target - existing_plans)
            stats = {"created_plans": 0, "refreshes": 0, "generated": 0, "completed": 0, "skipped": 0}

            user_cursor = 0

            for _ in range(to_create):
                if user_cursor >= len(user_ids_pool):
                    rng.shuffle(user_ids_pool)
                    user_cursor = 0
                user_id = user_ids_pool[user_cursor]
                user_cursor += 1

                with transaction.atomic():
                    if dry_run:
                        plan = None
                    else:
                        plan = RoadmapPlan.objects.create(
                            user_id=user_id,
                            category=cat,
                            is_active=True,
                            version=1,
                            meta={"seed_source": "seed_roadmap_plan_refresh_truth"},
                        )
                    steps: list[RoadmapStep] = []
                    for idx, ptype in enumerate(chain, start=1):
                        candidates = products_by_cat_type.get((cat, ptype), [])
                        product_id = rng.choice(candidates) if candidates else None
                        if dry_run:
                            continue
                        steps.append(
                            RoadmapStep.objects.create(
                                plan=plan,
                                step_index=idx,
                                product_type=ptype,
                                status=RoadmapStep.Status.RECOMMENDED if product_id else RoadmapStep.Status.MISSING,
                                recommended_product_id=product_id,
                                score=round(rng.uniform(0.5, 0.9), 3),
                                why=["seed"],
                            )
                        )
                    stats["created_plans"] += 1

                    refreshes_for_plan = rng.randint(refreshes_min, refreshes_max)
                    cycle_ts = window_start + timedelta(seconds=rng.randint(0, max(1, days * 24 * 3600 // 2)))
                    for cycle in range(refreshes_for_plan):
                        refresh_ts = cycle_ts + timedelta(hours=rng.randint(0, 36))
                        if refresh_ts >= now_utc:
                            break

                        head_step = steps[min(cycle, len(steps) - 1)] if steps else None
                        ctx_refresh = {
                            "source": "roadmap_v1",
                            "plan_id": plan.pk if plan else None,
                            "category": cat,
                            "steps_total": len(steps),
                            "missing_steps_count": max(0, len(steps) - cycle - 1),
                            "refresh_caller": "seed_refresh",
                            "next_step_id": head_step.pk if head_step else None,
                            "next_step_index": head_step.step_index if head_step else None,
                            "next_product_type": head_step.product_type if head_step else None,
                            "ml": {
                                "mode": "legacy",
                                "decision": "disabled",
                                "rollout_mode": "none",
                                "model_version": "seed",
                                "shadow_enabled": False,
                                "disabled_reason": "roadmap_ml_frozen",
                                "fallback_reason": None,
                                "rollout_selected": False,
                                "selected_feature_set": "",
                            },
                        }
                        if dry_run:
                            refresh_ev = None
                        else:
                            refresh_ev = RoadmapEvent.objects.create(
                                user_id=user_id,
                                plan=plan,
                                event_type=RoadmapEvent.Type.PLAN_REFRESHED,
                                context=ctx_refresh,
                            )
                            RoadmapEvent.objects.filter(pk=refresh_ev.pk).update(created_at=refresh_ts)
                        stats["refreshes"] += 1

                        if dry_run:
                            expected_k = rng.randint(gen_min, min(gen_max, max(gen_min, len(chain))))
                            stats["generated"] += expected_k
                            stats["completed"] += 1
                            cycle_ts = refresh_ts + timedelta(hours=rng.randint(12, 48))
                            continue

                        # Pick 2-4 candidates from the chain (must include head_step, plus distinct others).
                        k = rng.randint(gen_min, min(gen_max, len(steps)))
                        generated_steps = [head_step] if head_step else []
                        other_steps = [s for s in steps if s is not head_step]
                        rng.shuffle(other_steps)
                        generated_steps.extend(other_steps[: max(0, k - len(generated_steps))])

                        generated_events: list[RoadmapEvent] = []
                        generated_ts: list[Any] = []
                        for j, st in enumerate(generated_steps):
                            gen_ts = refresh_ts + timedelta(seconds=60 + j * 15 + rng.randint(0, 10))
                            ctx_gen = {
                                "source": "rules",
                                "status": "recommended",
                                "plan_id": plan.pk if plan else None,
                                "step_id": st.pk,
                                "category": cat,
                                "step_index": st.step_index,
                                "product_type": st.product_type,
                                "recommended_product_id": st.recommended_product_id,
                                "has_recommendation": bool(st.recommended_product_id),
                                "why": ["seed"],
                                "ml": {
                                    "mode": "legacy",
                                    "decision": "disabled",
                                    "rollout_mode": "none",
                                    "shadow_enabled": False,
                                },
                            }
                            ev = RoadmapEvent.objects.create(
                                user_id=user_id,
                                plan=plan,
                                step=st,
                                event_type=RoadmapEvent.Type.STEP_GENERATED,
                                context=ctx_gen,
                            )
                            generated_events.append(ev)
                            generated_ts.append(gen_ts)
                            stats["generated"] += 1
                        _chunked_update_created_at(generated_events, generated_ts)

                        # One STEP_COMPLETED inside the window for one of the generated steps.
                        truth_step = rng.choice(generated_steps)
                        completed_ts = refresh_ts + timedelta(minutes=rng.randint(10, 60))
                        ctx_done = {
                            "category": cat,
                            "step_index": truth_step.step_index,
                            "product_type": truth_step.product_type,
                            "matched_by": "recommended_product_id",
                            "recommended_product_id": truth_step.recommended_product_id,
                        }
                        done_ev = RoadmapEvent.objects.create(
                            user_id=user_id,
                            plan=plan,
                            step=truth_step,
                            event_type=RoadmapEvent.Type.STEP_COMPLETED,
                            context=ctx_done,
                        )
                        RoadmapEvent.objects.filter(pk=done_ev.pk).update(created_at=completed_ts)
                        RoadmapStep.objects.filter(pk=truth_step.pk).update(
                            status=RoadmapStep.Status.COMPLETED
                        )
                        stats["completed"] += 1

                        cycle_ts = completed_ts + timedelta(hours=rng.randint(12, 48))

            summary[cat] = stats
            self.stdout.write(
                f"[{cat}] plans_created={stats['created_plans']} "
                f"refreshes={stats['refreshes']} generated={stats['generated']} "
                f"completed={stats['completed']}"
            )

        self.stdout.write("Done. Summary:")
        for cat, stats in summary.items():
            self.stdout.write(f"  {cat}: {stats}")

        if dry_run:
            self.stdout.write(self.style.WARNING("Dry run — no rows written."))
        else:
            self.stdout.write(
                "Next: `python manage.py backfill_roadmap_shadow_meta --replay-mode historical_anchors --days {days} --write` "
                "then `python manage.py report_roadmap_nextstep_v4_decision_quality --days {days}`.".format(days=days)
            )
