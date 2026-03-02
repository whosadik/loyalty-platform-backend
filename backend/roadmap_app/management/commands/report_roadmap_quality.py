from __future__ import annotations

from collections import Counter, defaultdict
from datetime import timedelta
from pathlib import Path
from typing import Any

from django.core.management.base import BaseCommand, CommandError
from django.db import connection
from django.db.migrations.recorder import MigrationRecorder
from django.db.models import Count, Q
from django.utils import timezone

from catalog.models import Product
from offers.models import OfferAssignment
from roadmap_app.fragrance_slots import (
    SLOTS,
    normalize_intensity,
    normalize_notes,
    normalize_scent_family,
    slot_of_fragrance,
)
from roadmap_app.models import RoadmapEvent, RoadmapPlan, RoadmapStep


CATEGORY_CHOICES = ["skincare", "haircare", "makeup", "fragrance", "mixed", "all"]


def _pct(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return round((float(numerator) / float(denominator)) * 100.0, 2)


def _to_int(value: Any) -> int | None:
    try:
        if value is None:
            return None
        return int(value)
    except Exception:
        return None


def _safe_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _safe_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _truncate(value: Any, max_len: int = 140) -> str:
    text = str(value or "").replace("\n", " ").strip()
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


def _why_to_text(why: Any) -> str:
    if isinstance(why, list):
        return " | ".join(str(x) for x in why).lower()
    if isinstance(why, dict):
        return " | ".join(f"{k}:{v}" for k, v in sorted(why.items())).lower()
    return str(why or "").lower()


def _md_table(headers: list[str], rows: list[list[Any]]) -> str:
    if not rows:
        rows = [["-"] * len(headers)]
    out = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        out.append("| " + " | ".join(str(x) for x in row) + " |")
    return "\n".join(out)


class Command(BaseCommand):
    help = "Read-only quality report for Roadmap v1 and Offers<->Roadmap linkage."

    def add_arguments(self, parser):
        parser.add_argument("--days", type=int, default=30)
        parser.add_argument("--category", type=str, default="all", choices=CATEGORY_CHOICES)
        parser.add_argument(
            "--include-ga",
            action="store_true",
            default=False,
            help='Include users with username starting with "ga_".',
        )
        parser.add_argument("--output", type=str, default=None)

    def handle(self, *args, **options):
        days = int(options["days"] or 30)
        if days <= 0:
            raise CommandError("--days must be > 0")
        category = str(options["category"] or "all").strip().lower()
        include_ga = bool(options["include_ga"])
        output_path = options.get("output")

        now_utc = timezone.now()
        since = now_utc - timedelta(days=days)

        plan_qs = RoadmapPlan.objects.filter(updated_at__gte=since)
        step_qs = RoadmapStep.objects.filter(updated_at__gte=since)
        offer_qs = OfferAssignment.objects.filter(assigned_at__gte=since)
        event_qs = RoadmapEvent.objects.filter(created_at__gte=since)

        if category != "all":
            plan_qs = plan_qs.filter(category=category)
            step_qs = step_qs.filter(plan__category=category)
            offer_qs = offer_qs.filter(
                Q(target__category=category) | Q(reason__roadmap__category=category)
            )
            event_qs = event_qs.filter(Q(plan__category=category) | Q(context__category=category))

        if not include_ga:
            plan_qs = plan_qs.exclude(user__username__startswith="ga_")
            step_qs = step_qs.exclude(plan__user__username__startswith="ga_")
            offer_qs = offer_qs.exclude(user__username__startswith="ga_")
            event_qs = event_qs.exclude(user__username__startswith="ga_")

        plans_updated_total = plan_qs.count()
        steps_updated_total = step_qs.count()
        offers_assigned_total = offer_qs.count()

        shortcut_qs = offer_qs.filter(target__picked_via__startswith="roadmap_shortcut")
        offers_with_roadmap_shortcut_total = shortcut_qs.count()

        slot_offer_qs = offer_qs.filter(target__category="fragrance", target__product_type__in=SLOTS)
        offers_with_roadmap_slot_total = slot_offer_qs.count()
        offers_with_roadmap_slot_product_id_total = slot_offer_qs.filter(target__scope="product_id").count()
        offer_slot_product_id_rate = _pct(
            offers_with_roadmap_slot_product_id_total,
            offers_with_roadmap_slot_total,
        )

        exposed_events_total = event_qs.filter(event_type=RoadmapEvent.Type.STEP_EXPOSED).count()
        telemetry_coverage_rate = _pct(exposed_events_total, plans_updated_total)

        # Fragrance catalog health (all-time, all fragrance products)
        fragrance_rows = list(
            Product.objects.filter(category="fragrance").values("id", "product_type", "attrs", "raw_meta")
        )
        fragrance_total = len(fragrance_rows)
        intensity_present_cnt = 0
        scent_family_present_cnt = 0
        notes_present_cnt = 0
        raw_meta_present_cnt = 0
        notes_len_sum = 0
        intensity_dist: Counter[str] = Counter()
        family_dist: Counter[str] = Counter()
        examples_missing: list[dict[str, Any]] = []

        for row in fragrance_rows:
            attrs = _safe_dict(row.get("attrs"))
            raw_meta = _safe_dict(row.get("raw_meta"))

            attrs_intensity = attrs.get("intensity")
            attrs_family = attrs.get("scent_family")
            attrs_notes = attrs.get("notes")

            if str(attrs_intensity or "").strip():
                intensity_present_cnt += 1
            if str(attrs_family or "").strip():
                scent_family_present_cnt += 1
            if normalize_notes(attrs_notes):
                notes_present_cnt += 1
            if raw_meta:
                raw_meta_present_cnt += 1

            norm_intensity = normalize_intensity(attrs.get("intensity") or raw_meta.get("intensity"))
            intensity_dist[norm_intensity] += 1

            norm_family = normalize_scent_family(
                attrs.get("scent_family") or raw_meta.get("scent_family")
            ) or "unknown"
            family_dist[norm_family] += 1

            notes_norm = normalize_notes(
                attrs.get("notes") or raw_meta.get("notes") or raw_meta.get("note_list")
            )
            notes_len_sum += len(notes_norm)

            missing_flags: list[str] = []
            if not attrs:
                missing_flags.append("attrs_empty")
            if not str(attrs_intensity or "").strip():
                missing_flags.append("attrs.intensity_missing")
            if not str(attrs_family or "").strip():
                missing_flags.append("attrs.scent_family_missing")
            if not normalize_notes(attrs_notes):
                missing_flags.append("attrs.notes_missing")
            if missing_flags and len(examples_missing) < 10:
                examples_missing.append(
                    {
                        "product_id": int(row["id"]),
                        "missing": ",".join(missing_flags),
                    }
                )

        intensity_present_pct = _pct(intensity_present_cnt, fragrance_total)
        scent_family_present_pct = _pct(scent_family_present_cnt, fragrance_total)
        notes_present_pct = _pct(notes_present_cnt, fragrance_total)
        raw_meta_present_pct = _pct(raw_meta_present_cnt, fragrance_total)
        notes_avg = round((float(notes_len_sum) / float(fragrance_total)), 2) if fragrance_total else 0.0

        # Roadmap generation quality
        plans_by_category_rows = (
            plan_qs.values("category").annotate(c=Count("id")).order_by("-c", "category")
        )
        plans_by_category = {str(x["category"]): int(x["c"]) for x in plans_by_category_rows}

        status_counts = defaultdict(int)
        for row in step_qs.values("status").annotate(c=Count("id")):
            status_counts[str(row["status"])] = int(row["c"])

        plan_ids = list(plan_qs.values_list("id", flat=True))
        rec_rows = RoadmapStep.objects.filter(plan_id__in=plan_ids).exclude(
            recommended_product_id__isnull=True
        ).values_list("plan_id", "recommended_product_id")
        rec_by_plan: dict[int, list[int]] = defaultdict(list)
        for plan_id, rec_pid in rec_rows:
            rec_by_plan[int(plan_id)].append(int(rec_pid))

        plans_no_repeat = 0
        for plan_id in plan_ids:
            ids = rec_by_plan.get(int(plan_id), [])
            if len(ids) == len(set(ids)):
                plans_no_repeat += 1
        unique_recommended_plan_ratio = _pct(plans_no_repeat, len(plan_ids))

        step_with_recommended_qs = step_qs.exclude(recommended_product_id__isnull=True)
        step_with_recommended_total = step_with_recommended_qs.count()
        step_with_recommended_in_stock = step_with_recommended_qs.filter(
            recommended_product__in_stock=True
        ).count()
        in_stock_sanity_rate = _pct(step_with_recommended_in_stock, step_with_recommended_total)

        fr_slot_steps_qs = step_qs.filter(plan__category="fragrance", product_type__in=SLOTS)
        steps_slots_total = fr_slot_steps_qs.count()
        slot_distribution_rows = (
            fr_slot_steps_qs.values("product_type").annotate(c=Count("id")).order_by("-c", "product_type")
        )
        slot_distribution = {str(x["product_type"]): int(x["c"]) for x in slot_distribution_rows}

        slot_relaxed_count = 0
        no_suitable_count = 0
        examples_fallback_steps: list[dict[str, Any]] = []
        fr_slot_rows = fr_slot_steps_qs.values(
            "id", "plan_id", "product_type", "status", "suggestions", "why"
        )
        for row in fr_slot_rows:
            why_text = _why_to_text(row.get("why"))
            suggestions = _safe_list(row.get("suggestions"))

            slot_relaxed = ("slot relaxed" in why_text) or ("fallback_db_no_slot" in why_text)
            no_suitable = ("no suitable candidates" in why_text) or (
                str(row.get("status")) == RoadmapStep.Status.MISSING and len(suggestions) == 0
            )
            if slot_relaxed:
                slot_relaxed_count += 1
            if no_suitable:
                no_suitable_count += 1

            if (slot_relaxed or no_suitable) and len(examples_fallback_steps) < 10:
                examples_fallback_steps.append(
                    {
                        "step_id": int(row["id"]),
                        "plan_id": int(row["plan_id"]),
                        "slot": str(row["product_type"]),
                        "why": _truncate(row.get("why"), max_len=180),
                        "suggested_count": len(suggestions),
                    }
                )

        slot_relaxed_rate = _pct(slot_relaxed_count, steps_slots_total)
        no_suitable_rate = _pct(no_suitable_count, steps_slots_total)
        fragrance_slot_fallback_rate = slot_relaxed_rate

        # Offers <-> Roadmap quality
        offers_with_target_scope_product_id_total = offer_qs.filter(target__scope="product_id").count()

        spot_qs = slot_offer_qs.filter(target__scope="product_id").order_by("-assigned_at").values(
            "id", "target"
        )[:50]
        spot_rows = list(spot_qs)
        pid_list: list[int] = []
        for row in spot_rows:
            target = _safe_dict(row.get("target"))
            pid = _to_int(target.get("value"))
            if pid is not None:
                pid_list.append(pid)
        product_map = {
            int(p["id"]): p
            for p in Product.objects.filter(id__in=pid_list).values("id", "attrs", "raw_meta")
        }

        mismatch_count = 0
        examples_mismatch: list[dict[str, Any]] = []
        for row in spot_rows:
            assignment_id = int(row["id"])
            target = _safe_dict(row.get("target"))
            target_slot = str(target.get("product_type") or "")
            pid = _to_int(target.get("value"))
            computed_slot = ""
            if pid is not None and pid in product_map:
                p = product_map[pid]
                computed_slot = slot_of_fragrance(
                    _safe_dict(p.get("attrs")),
                    raw_meta=_safe_dict(p.get("raw_meta")),
                )
            if not computed_slot or computed_slot != target_slot:
                mismatch_count += 1
                if len(examples_mismatch) < 10:
                    examples_mismatch.append(
                        {
                            "assignment_id": assignment_id,
                            "product_id": pid,
                            "target_slot": target_slot or "-",
                            "computed_slot": computed_slot or "missing_product",
                        }
                    )

        mismatch_rate = _pct(mismatch_count, len(spot_rows))

        fallback_target_qs = shortcut_qs.filter(target__picked_via__icontains="fallback")
        fallback_target_total = fallback_target_qs.count()
        fallback_target_rate = _pct(fallback_target_total, offers_with_roadmap_shortcut_total)
        fallback_examples: list[dict[str, Any]] = []
        for row in fallback_target_qs.values("id", "target", "reason")[:10]:
            target = _safe_dict(row.get("target"))
            reason = _safe_dict(row.get("reason"))
            fallback_examples.append(
                {
                    "assignment_id": int(row["id"]),
                    "picked_via": str(target.get("picked_via") or ""),
                    "target": _truncate(target, max_len=180),
                    "reason": _truncate(
                        reason.get("picked_because") or reason.get("roadmap") or reason,
                        max_len=180,
                    ),
                }
            )

        # Telemetry coverage
        event_counts = {
            RoadmapEvent.Type.STEP_EXPOSED: event_qs.filter(
                event_type=RoadmapEvent.Type.STEP_EXPOSED
            ).count(),
            RoadmapEvent.Type.STEP_CLICKED: event_qs.filter(
                event_type=RoadmapEvent.Type.STEP_CLICKED
            ).count(),
            RoadmapEvent.Type.STEP_SKIPPED: event_qs.filter(
                event_type=RoadmapEvent.Type.STEP_SKIPPED
            ).count(),
            RoadmapEvent.Type.STEP_COMPLETED: event_qs.filter(
                event_type=RoadmapEvent.Type.STEP_COMPLETED
            ).count(),
        }
        clicked_exposed_ctr_proxy = _pct(
            event_counts[RoadmapEvent.Type.STEP_CLICKED],
            event_counts[RoadmapEvent.Type.STEP_EXPOSED],
        )
        completed_exposed_proxy = _pct(
            event_counts[RoadmapEvent.Type.STEP_COMPLETED],
            event_counts[RoadmapEvent.Type.STEP_EXPOSED],
        )
        top_exposed_steps_rows = (
            event_qs.filter(event_type=RoadmapEvent.Type.STEP_EXPOSED, step_id__isnull=False)
            .values("step_id")
            .annotate(c=Count("id"))
            .order_by("-c", "step_id")[:10]
        )
        top_exposed_steps = [[int(x["step_id"]), int(x["c"])] for x in top_exposed_steps_rows]

        # Sanity checks
        warnings: list[str] = []

        fragrance_products_with_slot_type = Product.objects.filter(
            category="fragrance",
            product_type__in=SLOTS,
        ).count()
        if fragrance_products_with_slot_type > 0:
            warnings.append(
                f"Found {fragrance_products_with_slot_type} fragrance products with slot-like product_type."
            )

        slot_pid_targets_global_qs = OfferAssignment.objects.filter(
            target__scope="product_id",
            target__category="fragrance",
            target__product_type__in=SLOTS,
        )
        if not include_ga:
            slot_pid_targets_global_qs = slot_pid_targets_global_qs.exclude(user__username__startswith="ga_")
        missing_actual_product_type_cnt = slot_pid_targets_global_qs.filter(
            Q(target__actual_product_type__isnull=True) | Q(target__actual_product_type="")
        ).count()
        if missing_actual_product_type_cnt > 0:
            warnings.append(
                "Found fragrance slot product_id targets with empty target.actual_product_type: "
                f"{missing_actual_product_type_cnt}"
            )

        migration_applied = False
        migration_table_exists = False
        migration_check_error = ""
        try:
            applied = set(MigrationRecorder(connection).applied_migrations())
            migration_applied = ("roadmap_app", "0003_roadmapevent") in applied
            migration_table_exists = "roadmap_app_roadmapevent" in connection.introspection.table_names()
        except Exception as exc:
            migration_check_error = str(exc)
            migration_table_exists = "roadmap_app_roadmapevent" in connection.introspection.table_names()
        if not migration_applied and not migration_table_exists:
            warnings.append("Migration roadmap_app.0003_roadmapevent not detected and table missing.")

        # users_touched
        users_touched_ids = set(int(x) for x in plan_qs.values_list("user_id", flat=True))
        users_touched_ids.update(int(x) for x in step_qs.values_list("plan__user_id", flat=True))
        users_touched_ids.update(int(x) for x in offer_qs.values_list("user_id", flat=True))
        users_touched_ids.update(int(x) for x in event_qs.values_list("user_id", flat=True))
        users_touched = len(users_touched_ids)

        # Markdown output
        lines: list[str] = []
        lines.append("# Roadmap Quality Report")
        lines.append("")
        lines.append(f"- Generated at (UTC): `{now_utc.isoformat()}`")
        lines.append(
            f"- Analysis window: `{since.isoformat()}` .. `{now_utc.isoformat()}` "
            f"(last **{days}** days)"
        )
        lines.append(f"- Category filter: `{category}`")
        lines.append(f"- Include ga_* users: `{include_ga}`")
        lines.append("")

        lines.append("## 1) Summary")
        lines.append(f"- users_touched: **{users_touched}**")
        lines.append(f"- plans_updated: **{plans_updated_total}**")
        lines.append(f"- steps_updated: **{steps_updated_total}**")
        lines.append(f"- offers_assigned: **{offers_assigned_total}**")
        lines.append(f"- offers_with_roadmap_shortcut: **{offers_with_roadmap_shortcut_total}**")
        lines.append(f"- fragrance_slot_fallback_rate: **{fragrance_slot_fallback_rate}%**")
        lines.append(f"- offer_roadmap_slot_product_id_rate: **{offer_slot_product_id_rate}%**")
        lines.append(f"- telemetry_coverage_rate (exposed/plans_updated): **{telemetry_coverage_rate}%**")
        lines.append("")

        lines.append("## 2) Catalog health (fragrance signals, all-time)")
        lines.append(f"- fragrance_total: **{fragrance_total}**")
        lines.append(f"- attrs.intensity present: **{intensity_present_cnt} ({intensity_present_pct}%)**")
        lines.append(
            f"- attrs.scent_family present: **{scent_family_present_cnt} ({scent_family_present_pct}%)**"
        )
        lines.append(f"- attrs.notes present: **{notes_present_cnt} ({notes_present_pct}%)**")
        lines.append(f"- raw_meta present: **{raw_meta_present_cnt} ({raw_meta_present_pct}%)**")
        lines.append(f"- avg normalized notes per product: **{notes_avg}**")
        lines.append("")
        lines.append("### Intensity distribution (normalized, top-5)")
        lines.append(
            _md_table(
                ["intensity", "count"],
                [[k, v] for k, v in intensity_dist.most_common(5)],
            )
        )
        lines.append("")
        lines.append("### Scent family distribution (normalized, top-10)")
        lines.append(
            _md_table(
                ["scent_family", "count"],
                [[k, v] for k, v in family_dist.most_common(10)],
            )
        )
        lines.append("")
        lines.append("### Examples missing signals (up to 10)")
        lines.append(
            _md_table(
                ["product_id", "missing"],
                [[x["product_id"], x["missing"]] for x in examples_missing],
            )
        )
        lines.append("")

        lines.append("## 3) Roadmap generation quality (last N days)")
        lines.append(f"- plans_updated_total: **{plans_updated_total}**")
        lines.append(
            _md_table(
                ["category", "plans_updated"],
                [[k, v] for k, v in sorted(plans_by_category.items(), key=lambda kv: (-kv[1], kv[0]))],
            )
        )
        lines.append("")
        lines.append(f"- steps_total: **{steps_updated_total}**")
        lines.append(f"- steps_recommended: **{status_counts.get(RoadmapStep.Status.RECOMMENDED, 0)}**")
        lines.append(f"- steps_missing: **{status_counts.get(RoadmapStep.Status.MISSING, 0)}**")
        lines.append(f"- steps_owned: **{status_counts.get(RoadmapStep.Status.OWNED, 0)}**")
        lines.append(f"- steps_completed: **{status_counts.get(RoadmapStep.Status.COMPLETED, 0)}**")
        lines.append(f"- steps_skipped: **{status_counts.get(RoadmapStep.Status.SKIPPED, 0)}**")
        lines.append(
            f"- unique recommended_product_id ratio (plans without duplicates): "
            f"**{unique_recommended_plan_ratio}%**"
        )
        lines.append(
            f"- in_stock sanity (recommended_product only): "
            f"**{step_with_recommended_in_stock}/{step_with_recommended_total} ({in_stock_sanity_rate}%)**"
        )
        lines.append("")
        lines.append("### Fragrance slots specific")
        lines.append(f"- steps_slots_total: **{steps_slots_total}**")
        lines.append(
            _md_table(
                ["slot", "count"],
                [[k, v] for k, v in sorted(slot_distribution.items(), key=lambda kv: (-kv[1], kv[0]))],
            )
        )
        lines.append("")
        lines.append(f"- slot_relaxed_rate: **{slot_relaxed_rate}%**")
        lines.append(f"- no_suitable_candidates_rate: **{no_suitable_rate}%**")
        lines.append("### Examples fallback steps (up to 10)")
        lines.append(
            _md_table(
                ["step_id", "plan_id", "slot", "suggested_count", "why"],
                [
                    [x["step_id"], x["plan_id"], x["slot"], x["suggested_count"], x["why"]]
                    for x in examples_fallback_steps
                ],
            )
        )
        lines.append("")

        lines.append("## 4) Offers <-> Roadmap quality (last N days)")
        lines.append(f"- offers_assigned_total: **{offers_assigned_total}**")
        lines.append(
            f"- offers_with_target_scope_product_id_total: **{offers_with_target_scope_product_id_total}**"
        )
        lines.append(f"- offers_with_roadmap_shortcut_total: **{offers_with_roadmap_shortcut_total}**")
        lines.append(f"- offers_with_roadmap_slot_total: **{offers_with_roadmap_slot_total}**")
        lines.append(
            f"- offers_with_roadmap_slot_product_id_total: **{offers_with_roadmap_slot_product_id_total}**"
        )
        lines.append(f"- offer_slot_product_id_rate: **{offer_slot_product_id_rate}%**")
        lines.append(f"- spot-check mismatch_rate (top-50): **{mismatch_rate}%**")
        lines.append("### Spot-check mismatches (up to 10)")
        lines.append(
            _md_table(
                ["assignment_id", "product_id", "target_slot", "computed_slot"],
                [
                    [x["assignment_id"], x["product_id"], x["target_slot"], x["computed_slot"]]
                    for x in examples_mismatch
                ],
            )
        )
        lines.append("")
        lines.append(
            f"- fallback_target_rate among roadmap_shortcut targets: "
            f"**{fallback_target_total}/{offers_with_roadmap_shortcut_total} ({fallback_target_rate}%)**"
        )
        lines.append("### Fallback target examples (up to 10)")
        lines.append(
            _md_table(
                ["assignment_id", "picked_via", "target", "reason"],
                [
                    [x["assignment_id"], x["picked_via"], x["target"], x["reason"]]
                    for x in fallback_examples
                ],
            )
        )
        lines.append("")

        lines.append("## 5) Telemetry coverage (last N days)")
        lines.append(
            f"- roadmap_step_exposed: **{event_counts[RoadmapEvent.Type.STEP_EXPOSED]}**"
        )
        lines.append(
            f"- roadmap_step_clicked: **{event_counts[RoadmapEvent.Type.STEP_CLICKED]}**"
        )
        lines.append(
            f"- roadmap_step_skipped: **{event_counts[RoadmapEvent.Type.STEP_SKIPPED]}**"
        )
        lines.append(
            f"- roadmap_step_completed: **{event_counts[RoadmapEvent.Type.STEP_COMPLETED]}**"
        )
        lines.append(
            f"- plans_updated_total vs exposed_events_total: **{plans_updated_total} vs "
            f"{event_counts[RoadmapEvent.Type.STEP_EXPOSED]}**"
        )
        lines.append(f"- clicked/exposed CTR proxy: **{clicked_exposed_ctr_proxy}%**")
        lines.append(f"- completed/exposed completion proxy: **{completed_exposed_proxy}%**")
        lines.append("### Top exposed steps (top-10)")
        lines.append(_md_table(["step_id", "exposed_count"], top_exposed_steps))
        lines.append("")

        lines.append("## 6) Sanity checks")
        lines.append(
            f"- fragrance products with slot-like product_type: **{fragrance_products_with_slot_type}** "
            "(expected: 0)"
        )
        lines.append(
            "- fragrance slot product_id targets with missing actual_product_type: "
            f"**{missing_actual_product_type_cnt}** (expected: 0)"
        )
        lines.append(f"- migration roadmap_app.0003_roadmapevent applied: **{migration_applied}**")
        lines.append(
            f"- table roadmap_app_roadmapevent exists: **{migration_table_exists}**"
        )
        if migration_check_error:
            lines.append(f"- migration check error: `{_truncate(migration_check_error, 220)}`")
        if warnings:
            lines.append("")
            lines.append("### WARNING")
            for item in warnings:
                lines.append(f"- {item}")
        else:
            lines.append("")
            lines.append("- No WARNING checks triggered.")
        lines.append("")

        lines.append("## Run commands")
        lines.append(r"- `.\.venv\Scripts\python.exe backend/manage.py report_roadmap_quality`")
        lines.append(
            r"- `.\.venv\Scripts\python.exe backend/manage.py report_roadmap_quality --days 14 --category fragrance`"
        )
        lines.append(
            r"- `.\.venv\Scripts\python.exe backend/manage.py report_roadmap_quality --days 30 --output reports/roadmap_quality.md`"
        )
        lines.append("")

        report = "\n".join(lines)
        self.stdout.write(report)

        if output_path:
            out_path = Path(output_path)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(report, encoding="utf-8")
            self.stdout.write(f"\n[report_roadmap_quality] wrote markdown to: {out_path}")
