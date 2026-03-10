from __future__ import annotations

import json
from collections import defaultdict
from datetime import timedelta
from pathlib import Path
from typing import Any

from django.core.management.base import BaseCommand, CommandError
from django.utils import timezone

from roadmap_app.models import RoadmapEvent


CATEGORY_CHOICES = ["all", "skincare", "haircare", "makeup", "fragrance", "mixed"]
FORMAT_CHOICES = ["md", "json", "both"]
COHORT_MODE_CHOICES = ["fresh", "all"]
VALID_DECISIONS = {"model_used", "fallback", "disabled"}


def _safe_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _safe_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _to_int(value: Any) -> int | None:
    try:
        if value is None:
            return None
        return int(value)
    except Exception:
        return None


def _pct(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value * 100.0:.2f}%"


def _rate(numerator: int, denominator: int) -> float | None:
    if denominator <= 0:
        return None
    return float(numerator) / float(denominator)


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


def _resolve_out_stem(*, out: str | None, days: int, category: str) -> Path:
    if out:
        p = Path(out)
        if p.suffix.lower() in {".md", ".json"}:
            return p.with_suffix("")
        return p
    return Path("reports") / f"roadmap_generation_gap_{days}d_{category}"


def _decision_from_ml(ml: dict[str, Any]) -> str:
    decision = str(ml.get("decision") or "").strip().lower()
    if decision in VALID_DECISIONS:
        return decision
    return "missing_ml_meta"


def _step_index_bucket(step_index: int | None) -> str:
    if step_index is None:
        return "__unknown__"
    if step_index <= 1:
        return "step_1"
    if step_index == 2:
        return "step_2"
    if step_index == 3:
        return "step_3"
    return "step_4_plus"


def _source_from_expose_context(ctx: dict[str, Any]) -> str:
    sources = _safe_list(ctx.get("sources"))
    normalized = {str(x).strip().lower() for x in sources if str(x).strip()}
    if "offers" in normalized:
        return "offers"
    if "roadmap_api" in normalized:
        return "roadmap_api"
    if ctx.get("offer_assignment_id") not in (None, ""):
        return "offers"
    return "roadmap_api"


def _event_category(row: dict[str, Any], ctx: dict[str, Any]) -> str:
    category = (
        row.get("plan__category")
        or row.get("step__plan__category")
        or ctx.get("category")
        or ""
    )
    return str(category).strip().lower()


def _new_bucket() -> dict[str, int]:
    return {
        "plans_refreshed": 0,
        "steps_generated": 0,
        "steps_exposed": 0,
        "steps_clicked": 0,
        "steps_completed_after_exposure": 0,
        "steps_completed_after_generated": 0,
    }


def _new_adherence_bucket() -> dict[str, int]:
    return {
        "next_step_with_recommendation": 0,
        "checkout_targeted_next_step": 0,
        "checkout_targeted_recommended_product": 0,
    }


def _finalize_bucket(bucket: dict[str, int]) -> dict[str, Any]:
    generated = int(bucket.get("steps_generated", 0))
    exposed = int(bucket.get("steps_exposed", 0))
    return {
        **bucket,
        "generated_to_exposed_rate": _rate(exposed, generated),
        "exposed_to_clicked_rate": _rate(int(bucket.get("steps_clicked", 0)), exposed),
        "exposed_to_completed_rate": _rate(int(bucket.get("steps_completed_after_exposure", 0)), exposed),
        "generated_to_completed_rate": _rate(int(bucket.get("steps_completed_after_generated", 0)), generated),
    }


def _finalize_adherence_bucket(bucket: dict[str, int]) -> dict[str, Any]:
    with_recommendation = int(bucket.get("next_step_with_recommendation", 0))
    targeted_next_step = int(bucket.get("checkout_targeted_next_step", 0))
    targeted_recommended = int(bucket.get("checkout_targeted_recommended_product", 0))
    return {
        **bucket,
        "targeted_next_step_rate": _rate(targeted_next_step, with_recommendation),
        "recommended_product_adoption_rate": _rate(targeted_recommended, with_recommendation),
        "recommended_share_within_targeted_next_step": _rate(targeted_recommended, targeted_next_step),
    }


def _adherence_step_bucket(step_index: int | None) -> str:
    if step_index is None:
        return "__unknown__"
    if step_index <= 1:
        return "step_1"
    return "step_2_plus"


def _event_key(created_at, event_id: int | None) -> tuple[Any, int]:
    return created_at, int(event_id or 0)


def _first_exposure_in_range(
    events: list[tuple[Any, int, str]],
    *,
    start_key: tuple[Any, int],
    end_key: tuple[Any, int] | None,
) -> tuple[Any, int, str] | None:
    for created_at, event_id, source in events:
        key = _event_key(created_at, event_id)
        if key < start_key:
            continue
        if end_key is not None and key >= end_key:
            break
        return created_at, int(event_id), source
    return None


def _any_event_in_range(events: list[tuple[Any, int]], *, start_key: tuple[Any, int], end_key: tuple[Any, int] | None) -> bool:
    for created_at, event_id in events:
        key = _event_key(created_at, event_id)
        if key < start_key:
            continue
        if end_key is not None and key >= end_key:
            break
        return True
    return False


def _any_completion_match_in_range(
    events: list[tuple[Any, int, str]],
    *,
    start_key: tuple[Any, int],
    end_key: tuple[Any, int] | None,
    matched_by_values: set[str] | None = None,
) -> bool:
    for created_at, event_id, matched_by in events:
        key = _event_key(created_at, event_id)
        if key < start_key:
            continue
        if end_key is not None and key >= end_key:
            break
        if matched_by_values is None:
            return True
        if str(matched_by or "").strip().lower() in matched_by_values:
            return True
    return False


def _bucket_rows(buckets: dict[str, dict[str, Any]], *, include_plan_refresh: bool = True) -> list[list[Any]]:
    rows: list[list[Any]] = []
    for key, bucket in sorted(
        buckets.items(),
        key=lambda kv: (-int(kv[1].get("steps_generated", 0)), str(kv[0])),
    ):
        row = [
            key,
            int(bucket.get("steps_generated", 0)),
            int(bucket.get("steps_exposed", 0)),
            int(bucket.get("steps_clicked", 0)),
            int(bucket.get("steps_completed_after_generated", 0)),
            _pct(bucket.get("generated_to_exposed_rate")),
            _pct(bucket.get("exposed_to_clicked_rate")),
            _pct(bucket.get("exposed_to_completed_rate")),
            _pct(bucket.get("generated_to_completed_rate")),
        ]
        if include_plan_refresh:
            row.insert(1, int(bucket.get("plans_refreshed", 0)))
        rows.append(row)
    return rows


def _adherence_rows(buckets: dict[str, dict[str, Any]]) -> list[list[Any]]:
    rows: list[list[Any]] = []
    for key, bucket in sorted(
        buckets.items(),
        key=lambda kv: (-int(kv[1].get("next_step_with_recommendation", 0)), str(kv[0])),
    ):
        rows.append(
            [
                key,
                int(bucket.get("next_step_with_recommendation", 0)),
                int(bucket.get("checkout_targeted_next_step", 0)),
                int(bucket.get("checkout_targeted_recommended_product", 0)),
                _pct(bucket.get("targeted_next_step_rate")),
                _pct(bucket.get("recommended_product_adoption_rate")),
                _pct(bucket.get("recommended_share_within_targeted_next_step")),
            ]
        )
    return rows


def _build_markdown(payload: dict[str, Any]) -> str:
    params = _safe_dict(payload.get("params"))
    overall_raw = _safe_dict(_safe_dict(payload.get("overall")).get("raw"))
    overall_analysis = _safe_dict(_safe_dict(payload.get("overall")).get("analysis"))
    next_step_only = _safe_dict(payload.get("next_step_only"))
    next_step_raw = _safe_dict(next_step_only.get("raw"))
    adherence = _safe_dict(payload.get("recommended_product_adherence"))
    adherence_raw = _safe_dict(adherence.get("raw"))
    breakdowns = _safe_dict(payload.get("breakdowns"))
    notes = _safe_list(payload.get("notes"))

    lines: list[str] = []
    lines.append("# Roadmap Generation Gap Report")
    lines.append("")
    lines.append(f"- Generated at (UTC): `{payload.get('generated_at_utc')}`")
    lines.append(
        f"- Analysis window: `{payload.get('window_start_utc')}` .. `{payload.get('window_end_utc')}` "
        f"(last **{params.get('days')}** days)"
    )
    lines.append(f"- Category filter: `{params.get('category')}`")
    lines.append(f"- Include ga_* users: `{params.get('include_ga')}`")
    lines.append(f"- Cohort mode: `{params.get('cohort_mode')}`")
    lines.append("")
    lines.append("## 1) Overall funnel")
    lines.append(
        f"- raw plans_refreshed: **{int(overall_raw.get('plans_refreshed', 0))}**"
    )
    lines.append(
        f"- raw steps_generated: **{int(overall_raw.get('steps_generated', 0))}**, "
        f"steps_exposed: **{int(overall_raw.get('steps_exposed', 0))}**, "
        f"steps_clicked: **{int(overall_raw.get('steps_clicked', 0))}**, "
        f"steps_completed_after_generated: **{int(overall_raw.get('steps_completed_after_generated', 0))}**"
    )
    lines.append(
        f"- generated -> exposed: **{_pct(overall_raw.get('generated_to_exposed_rate'))}**, "
        f"exposed -> clicked: **{_pct(overall_raw.get('exposed_to_clicked_rate'))}**, "
        f"exposed -> completed: **{_pct(overall_raw.get('exposed_to_completed_rate'))}**, "
        f"generated -> completed: **{_pct(overall_raw.get('generated_to_completed_rate'))}**"
    )
    lines.append(
        f"- analysis steps_generated: **{int(overall_analysis.get('steps_generated', 0))}**, "
        f"excluded_missing_ml_meta_steps: **{int(overall_analysis.get('excluded_missing_ml_meta_steps', 0))}**"
    )
    lines.append("")
    lines.append("## 1b) Next Step Only")
    lines.append(
        f"- next_step_refreshed: **{int(next_step_raw.get('plans_refreshed', 0))}**, "
        f"next_step_generated: **{int(next_step_raw.get('steps_generated', 0))}**, "
        f"next_step_exposed: **{int(next_step_raw.get('steps_exposed', 0))}**, "
        f"next_step_clicked: **{int(next_step_raw.get('steps_clicked', 0))}**, "
        f"next_step_completed: **{int(next_step_raw.get('steps_completed_after_generated', 0))}**"
    )
    lines.append(
        f"- generated -> exposed: **{_pct(next_step_raw.get('generated_to_exposed_rate'))}**, "
        f"exposed -> clicked: **{_pct(next_step_raw.get('exposed_to_clicked_rate'))}**, "
        f"exposed -> completed: **{_pct(next_step_raw.get('exposed_to_completed_rate'))}**, "
        f"generated -> completed: **{_pct(next_step_raw.get('generated_to_completed_rate'))}**"
    )
    lines.append("")
    lines.append("## 1c) Next Step Only By Category")
    lines.append(
        _md_table(
            [
                "category",
                "plans_refreshed",
                "steps_generated",
                "steps_exposed",
                "steps_clicked",
                "steps_completed",
                "gen_to_exp",
                "exp_to_click",
                "exp_to_complete",
                "gen_to_complete",
            ],
            _bucket_rows(_safe_dict(next_step_only.get("by_category")), include_plan_refresh=True),
        )
    )
    lines.append("")
    lines.append("## 1d) Recommended Product Adherence")
    lines.append(
        f"- next_step_with_recommendation: **{int(adherence_raw.get('next_step_with_recommendation', 0))}**, "
        f"checkout_targeted_next_step: **{int(adherence_raw.get('checkout_targeted_next_step', 0))}**, "
        f"checkout_targeted_recommended_product: **{int(adherence_raw.get('checkout_targeted_recommended_product', 0))}**"
    )
    lines.append(
        f"- targeted next-step rate: **{_pct(adherence_raw.get('targeted_next_step_rate'))}**, "
        f"recommended product adoption rate: **{_pct(adherence_raw.get('recommended_product_adoption_rate'))}**, "
        f"recommended share within targeted next-step: **{_pct(adherence_raw.get('recommended_share_within_targeted_next_step'))}**"
    )
    lines.append("")
    lines.append("## 1e) Recommended Product Adherence By Category")
    lines.append(
        _md_table(
            [
                "category",
                "next_step_with_recommendation",
                "checkout_targeted_next_step",
                "checkout_targeted_recommended_product",
                "targeted_next_step_rate",
                "recommended_adoption_rate",
                "recommended_share_within_targeted",
            ],
            _adherence_rows(_safe_dict(adherence.get("by_category"))),
        )
    )
    lines.append("")
    lines.append("## 1f) Recommended Product Adherence By Step Bucket")
    lines.append(
        _md_table(
            [
                "step_bucket",
                "next_step_with_recommendation",
                "checkout_targeted_next_step",
                "checkout_targeted_recommended_product",
                "targeted_next_step_rate",
                "recommended_adoption_rate",
                "recommended_share_within_targeted",
            ],
            _adherence_rows(_safe_dict(adherence.get("by_step_bucket"))),
        )
    )
    lines.append("")
    lines.append("## 1g) Recommended Product Adherence By ML Decision")
    lines.append(
        _md_table(
            [
                "ml_decision",
                "next_step_with_recommendation",
                "checkout_targeted_next_step",
                "checkout_targeted_recommended_product",
                "targeted_next_step_rate",
                "recommended_adoption_rate",
                "recommended_share_within_targeted",
            ],
            _adherence_rows(_safe_dict(adherence.get("by_ml_decision"))),
        )
    )
    lines.append("")
    lines.append("## 2) By Category")
    lines.append(
        _md_table(
            [
                "category",
                "plans_refreshed",
                "steps_generated",
                "steps_exposed",
                "steps_clicked",
                "steps_completed",
                "gen_to_exp",
                "exp_to_click",
                "exp_to_complete",
                "gen_to_complete",
            ],
            _bucket_rows(_safe_dict(breakdowns.get("by_category")), include_plan_refresh=True),
        )
    )
    lines.append("")
    lines.append("## 3) By Step Index")
    lines.append(
        _md_table(
            [
                "step_index",
                "steps_generated",
                "steps_exposed",
                "steps_clicked",
                "steps_completed",
                "gen_to_exp",
                "exp_to_click",
                "exp_to_complete",
                "gen_to_complete",
            ],
            _bucket_rows(_safe_dict(breakdowns.get("by_step_index")), include_plan_refresh=False),
        )
    )
    lines.append("")
    lines.append("## 4) By Product Type")
    lines.append(
        _md_table(
            [
                "product_type",
                "steps_generated",
                "steps_exposed",
                "steps_clicked",
                "steps_completed",
                "gen_to_exp",
                "exp_to_click",
                "exp_to_complete",
                "gen_to_complete",
            ],
            _bucket_rows(_safe_dict(breakdowns.get("by_product_type")), include_plan_refresh=False),
        )
    )
    lines.append("")
    lines.append("## 5) By ML Decision")
    lines.append(
        _md_table(
            [
                "ml_decision",
                "plans_refreshed",
                "steps_generated",
                "steps_exposed",
                "steps_clicked",
                "steps_completed",
                "gen_to_exp",
                "exp_to_click",
                "exp_to_complete",
                "gen_to_complete",
            ],
            _bucket_rows(_safe_dict(breakdowns.get("by_ml_decision")), include_plan_refresh=True),
        )
    )
    lines.append("")
    lines.append("## 6) By Expose Source")
    lines.append(
        _md_table(
            [
                "expose_source",
                "steps_generated",
                "steps_exposed",
                "steps_clicked",
                "steps_completed",
                "gen_to_exp",
                "exp_to_click",
                "exp_to_complete",
                "gen_to_complete",
            ],
            _bucket_rows(_safe_dict(breakdowns.get("by_expose_source")), include_plan_refresh=False),
        )
    )
    lines.append("")
    lines.append("## 7) Notes")
    for note in notes:
        lines.append(f"- {note}")
    lines.append("")
    return "\n".join(lines)


class Command(BaseCommand):
    help = "Read-only report for Roadmap generation -> exposure -> interaction funnel."

    def add_arguments(self, parser):
        parser.add_argument("--days", type=int, default=7)
        parser.add_argument("--category", type=str, default="all", choices=CATEGORY_CHOICES)
        parser.add_argument(
            "--include-ga",
            action="store_true",
            default=False,
            help='Include users with username starting with "ga_".',
        )
        parser.add_argument("--out", type=str, default=None)
        parser.add_argument("--format", type=str, default="both", choices=FORMAT_CHOICES)
        parser.add_argument("--cohort-mode", type=str, default="all", choices=COHORT_MODE_CHOICES)

    def handle(self, *args, **options):
        days = int(options["days"] or 7)
        if days <= 0:
            raise CommandError("--days must be > 0")

        category = str(options["category"] or "all").strip().lower()
        include_ga = bool(options["include_ga"])
        out_raw = options.get("out")
        out_format = str(options["format"] or "both").strip().lower()
        cohort_mode = str(options["cohort_mode"] or "all").strip().lower()

        now_utc = timezone.now()
        since = now_utc - timedelta(days=days)

        event_qs = RoadmapEvent.objects.filter(
            created_at__gte=since,
            created_at__lte=now_utc,
            event_type__in=[
                RoadmapEvent.Type.PLAN_REFRESHED,
                RoadmapEvent.Type.STEP_GENERATED,
                RoadmapEvent.Type.STEP_EXPOSED,
                RoadmapEvent.Type.STEP_CLICKED,
                RoadmapEvent.Type.STEP_COMPLETED,
            ],
        )
        if not include_ga:
            event_qs = event_qs.exclude(user__username__startswith="ga_")

        rows = list(
            event_qs.order_by("user_id", "step_id", "created_at", "id").values(
                "id",
                "user_id",
                "plan_id",
                "plan__category",
                "step_id",
                "step__recommended_product_id",
                "step__plan__category",
                "event_type",
                "created_at",
                "context",
            )
        )

        plan_instances: list[dict[str, Any]] = []
        generated_by_key: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
        exposures_by_key: dict[tuple[int, int], list[tuple[Any, int, str]]] = defaultdict(list)
        clicks_by_key: dict[tuple[int, int], list[tuple[Any, int]]] = defaultdict(list)
        completions_by_key: dict[tuple[int, int], list[tuple[Any, int]]] = defaultdict(list)
        completion_matches_by_key: dict[tuple[int, int], list[tuple[Any, int, str]]] = defaultdict(list)

        for row in rows:
            ctx = _safe_dict(row.get("context"))
            category_key = _event_category(row, ctx)
            if category != "all" and category_key != category:
                continue

            user_id = _to_int(row.get("user_id"))
            event_type = str(row.get("event_type") or "")
            created_at = row.get("created_at")
            if user_id is None or created_at is None:
                continue

            if event_type == RoadmapEvent.Type.PLAN_REFRESHED:
                ml = _safe_dict(ctx.get("ml"))
                plan_instances.append(
                    {
                        "event_id": _to_int(row.get("id")),
                        "user_id": int(user_id),
                        "plan_id": _to_int(row.get("plan_id")) or _to_int(ctx.get("plan_id")),
                        "category": category_key or "__unknown__",
                        "generated_at": created_at,
                        "next_step_id": _to_int(ctx.get("next_step_id")),
                        "next_step_index": _to_int(ctx.get("next_step_index")),
                        "next_product_type": str(ctx.get("next_product_type") or "").strip().lower() or "__unknown__",
                        "ml_decision": _decision_from_ml(ml),
                    }
                )
                continue

            step_id = _to_int(row.get("step_id"))
            if step_id is None:
                step_id = _to_int(ctx.get("step_id"))
            if step_id is None:
                continue
            key = (int(user_id), int(step_id))

            if event_type == RoadmapEvent.Type.STEP_GENERATED:
                ml = _safe_dict(ctx.get("ml"))
                event_id = _to_int(row.get("id"))
                recommended_product_id = _to_int(ctx.get("recommended_product_id")) or _to_int(
                    row.get("step__recommended_product_id")
                )
                has_recommendation = bool(ctx.get("has_recommendation")) or recommended_product_id is not None
                generated_by_key[key].append(
                    {
                        "generated_event_id": event_id,
                        "user_id": int(user_id),
                        "plan_id": _to_int(row.get("plan_id")) or _to_int(ctx.get("plan_id")),
                        "step_id": int(step_id),
                        "generated_at": created_at,
                        "category": category_key or "__unknown__",
                        "step_index": _to_int(ctx.get("step_index")),
                        "step_index_bucket": _step_index_bucket(_to_int(ctx.get("step_index"))),
                        "product_type": str(ctx.get("product_type") or "").strip().lower() or "__unknown__",
                        "status": str(ctx.get("status") or "").strip().lower() or "__unknown__",
                        "generated_source": str(ctx.get("source") or "").strip().lower() or "__unknown__",
                        "recommended_product_id": recommended_product_id,
                        "has_recommendation": bool(has_recommendation),
                        "ml_decision": _decision_from_ml(ml),
                        "ml_rollout_mode": str(ml.get("rollout_mode") or "none").strip().lower() or "none",
                        "ml_rollout_selected": bool(ml.get("rollout_selected")),
                    }
                )
            elif event_type == RoadmapEvent.Type.STEP_EXPOSED:
                exposures_by_key[key].append(
                    (created_at, _to_int(row.get("id")) or 0, _source_from_expose_context(ctx))
                )
            elif event_type == RoadmapEvent.Type.STEP_CLICKED:
                clicks_by_key[key].append((created_at, _to_int(row.get("id")) or 0))
            elif event_type == RoadmapEvent.Type.STEP_COMPLETED:
                event_id = _to_int(row.get("id")) or 0
                matched_by = str(ctx.get("matched_by") or "").strip().lower()
                completions_by_key[key].append((created_at, event_id))
                completion_matches_by_key[key].append((created_at, event_id, matched_by))

        step_instances: list[dict[str, Any]] = []
        for key, items in generated_by_key.items():
            items.sort(key=lambda item: (item["generated_at"], int(item.get("generated_event_id") or 0)))
            for idx, item in enumerate(items):
                generated_key = _event_key(item["generated_at"], int(item.get("generated_event_id") or 0))
                next_generated_key = (
                    _event_key(
                        items[idx + 1]["generated_at"],
                        int(items[idx + 1].get("generated_event_id") or 0),
                    )
                    if idx + 1 < len(items)
                    else None
                )
                exposure = _first_exposure_in_range(
                    exposures_by_key.get(key, []),
                    start_key=generated_key,
                    end_key=next_generated_key,
                )
                first_exposed_at = exposure[0] if exposure else None
                first_exposed_event_id = int(exposure[1]) if exposure else None
                first_expose_source = exposure[2] if exposure else "__not_exposed__"
                item["next_generated_at"] = next_generated_key[0] if next_generated_key else None
                item["has_exposed"] = exposure is not None
                item["first_exposed_at"] = first_exposed_at
                item["first_exposed_event_id"] = first_exposed_event_id
                item["first_expose_source"] = first_expose_source
                item["has_clicked_after_exposure"] = bool(
                    first_exposed_at
                    and _any_event_in_range(
                        clicks_by_key.get(key, []),
                        start_key=_event_key(first_exposed_at, first_exposed_event_id or 0),
                        end_key=next_generated_key,
                    )
                )
                item["has_completed_after_exposure"] = bool(
                    first_exposed_at
                    and _any_event_in_range(
                        completions_by_key.get(key, []),
                        start_key=_event_key(first_exposed_at, first_exposed_event_id or 0),
                        end_key=next_generated_key,
                    )
                )
                item["has_completed_after_generated"] = _any_event_in_range(
                    completions_by_key.get(key, []),
                    start_key=generated_key,
                    end_key=next_generated_key,
                )
                item["has_targeted_next_step_checkout"] = _any_completion_match_in_range(
                    completion_matches_by_key.get(key, []),
                    start_key=generated_key,
                    end_key=next_generated_key,
                )
                item["has_recommended_product_checkout"] = _any_completion_match_in_range(
                    completion_matches_by_key.get(key, []),
                    start_key=generated_key,
                    end_key=next_generated_key,
                    matched_by_values={"recommended_product_id"},
                )
                step_instances.append(item)

        analysis_plan_instances = list(plan_instances)
        analysis_step_instances = list(step_instances)
        next_step_generated_event_ids: set[int] = set()
        generated_by_plan_step: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
        plan_refreshes_by_plan: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for item in step_instances:
            plan_id = _to_int(item.get("plan_id"))
            step_id = _to_int(item.get("step_id"))
            if plan_id is None or step_id is None:
                continue
            generated_by_plan_step[(int(plan_id), int(step_id))].append(item)
        for items in generated_by_plan_step.values():
            items.sort(key=lambda item: (item["generated_at"], int(item.get("generated_event_id") or 0)))
        for plan_item in plan_instances:
            plan_id = _to_int(plan_item.get("plan_id"))
            if plan_id is None:
                continue
            plan_refreshes_by_plan[int(plan_id)].append(plan_item)
        for items in plan_refreshes_by_plan.values():
            items.sort(key=lambda item: (item["generated_at"], int(item.get("event_id") or 0)))
        for plan_id, refreshes in plan_refreshes_by_plan.items():
            for idx, refresh in enumerate(refreshes):
                next_step_id = _to_int(refresh.get("next_step_id"))
                if next_step_id is None:
                    continue
                refresh_key = _event_key(refresh.get("generated_at"), _to_int(refresh.get("event_id")) or 0)
                next_refresh_key = (
                    _event_key(refreshes[idx + 1].get("generated_at"), _to_int(refreshes[idx + 1].get("event_id")) or 0)
                    if idx + 1 < len(refreshes)
                    else None
                )
                for generated_item in generated_by_plan_step.get((int(plan_id), int(next_step_id)), []):
                    generated_key = _event_key(
                        generated_item.get("generated_at"),
                        _to_int(generated_item.get("generated_event_id")) or 0,
                    )
                    if generated_key < refresh_key:
                        continue
                    if next_refresh_key is not None and generated_key >= next_refresh_key:
                        break
                    next_step_generated_event_ids.add(int(generated_item.get("generated_event_id") or 0))
                    break
        excluded_missing_plan_count = 0
        excluded_missing_step_count = 0
        if cohort_mode == "fresh":
            excluded_missing_plan_count = sum(
                1 for item in plan_instances if str(item.get("ml_decision") or "") == "missing_ml_meta"
            )
            excluded_missing_step_count = sum(
                1 for item in step_instances if str(item.get("ml_decision") or "") == "missing_ml_meta"
            )
            analysis_plan_instances = [
                item for item in plan_instances if str(item.get("ml_decision") or "") != "missing_ml_meta"
            ]
            analysis_step_instances = [
                item for item in step_instances if str(item.get("ml_decision") or "") != "missing_ml_meta"
            ]

        overall_raw = _new_bucket()
        overall_analysis = _new_bucket()
        next_step_only_raw = _new_bucket()
        next_step_only_by_category: dict[str, dict[str, int]] = defaultdict(_new_bucket)
        adherence_raw = _new_adherence_bucket()
        adherence_by_category: dict[str, dict[str, int]] = defaultdict(_new_adherence_bucket)
        adherence_by_step_bucket: dict[str, dict[str, int]] = defaultdict(_new_adherence_bucket)
        adherence_by_ml_decision: dict[str, dict[str, int]] = defaultdict(_new_adherence_bucket)
        by_category: dict[str, dict[str, int]] = defaultdict(_new_bucket)
        by_step_index: dict[str, dict[str, int]] = defaultdict(_new_bucket)
        by_product_type: dict[str, dict[str, int]] = defaultdict(_new_bucket)
        by_ml_decision: dict[str, dict[str, int]] = defaultdict(_new_bucket)
        by_expose_source: dict[str, dict[str, int]] = defaultdict(_new_bucket)

        for item in plan_instances:
            overall_raw["plans_refreshed"] += 1
            by_ml_decision[str(item.get("ml_decision") or "missing_ml_meta")]["plans_refreshed"] += 1
        for item in analysis_plan_instances:
            by_category[str(item.get("category") or "__unknown__")]["plans_refreshed"] += 1
            overall_analysis["plans_refreshed"] += 1
            if _to_int(item.get("next_step_id")) is not None:
                next_step_only_raw["plans_refreshed"] += 1
                next_step_only_by_category[str(item.get("category") or "__unknown__")]["plans_refreshed"] += 1

        for item in step_instances:
            overall_raw["steps_generated"] += 1
            overall_raw["steps_exposed"] += int(bool(item.get("has_exposed")))
            overall_raw["steps_clicked"] += int(bool(item.get("has_clicked_after_exposure")))
            overall_raw["steps_completed_after_exposure"] += int(bool(item.get("has_completed_after_exposure")))
            overall_raw["steps_completed_after_generated"] += int(bool(item.get("has_completed_after_generated")))

            ml_bucket = by_ml_decision[str(item.get("ml_decision") or "missing_ml_meta")]
            ml_bucket["steps_generated"] += 1
            ml_bucket["steps_exposed"] += int(bool(item.get("has_exposed")))
            ml_bucket["steps_clicked"] += int(bool(item.get("has_clicked_after_exposure")))
            ml_bucket["steps_completed_after_exposure"] += int(bool(item.get("has_completed_after_exposure")))
            ml_bucket["steps_completed_after_generated"] += int(bool(item.get("has_completed_after_generated")))

        for item in analysis_step_instances:
            overall_analysis["steps_generated"] += 1
            overall_analysis["steps_exposed"] += int(bool(item.get("has_exposed")))
            overall_analysis["steps_clicked"] += int(bool(item.get("has_clicked_after_exposure")))
            overall_analysis["steps_completed_after_exposure"] += int(bool(item.get("has_completed_after_exposure")))
            overall_analysis["steps_completed_after_generated"] += int(bool(item.get("has_completed_after_generated")))

            category_bucket = by_category[str(item.get("category") or "__unknown__")]
            category_bucket["steps_generated"] += 1
            category_bucket["steps_exposed"] += int(bool(item.get("has_exposed")))
            category_bucket["steps_clicked"] += int(bool(item.get("has_clicked_after_exposure")))
            category_bucket["steps_completed_after_exposure"] += int(bool(item.get("has_completed_after_exposure")))
            category_bucket["steps_completed_after_generated"] += int(bool(item.get("has_completed_after_generated")))

            step_bucket = by_step_index[str(item.get("step_index_bucket") or "__unknown__")]
            step_bucket["steps_generated"] += 1
            step_bucket["steps_exposed"] += int(bool(item.get("has_exposed")))
            step_bucket["steps_clicked"] += int(bool(item.get("has_clicked_after_exposure")))
            step_bucket["steps_completed_after_exposure"] += int(bool(item.get("has_completed_after_exposure")))
            step_bucket["steps_completed_after_generated"] += int(bool(item.get("has_completed_after_generated")))

            pt_bucket = by_product_type[str(item.get("product_type") or "__unknown__")]
            pt_bucket["steps_generated"] += 1
            pt_bucket["steps_exposed"] += int(bool(item.get("has_exposed")))
            pt_bucket["steps_clicked"] += int(bool(item.get("has_clicked_after_exposure")))
            pt_bucket["steps_completed_after_exposure"] += int(bool(item.get("has_completed_after_exposure")))
            pt_bucket["steps_completed_after_generated"] += int(bool(item.get("has_completed_after_generated")))

            source_bucket = by_expose_source[str(item.get("first_expose_source") or "__not_exposed__")]
            source_bucket["steps_generated"] += 1
            source_bucket["steps_exposed"] += int(bool(item.get("has_exposed")))
            source_bucket["steps_clicked"] += int(bool(item.get("has_clicked_after_exposure")))
            source_bucket["steps_completed_after_exposure"] += int(bool(item.get("has_completed_after_exposure")))
            source_bucket["steps_completed_after_generated"] += int(bool(item.get("has_completed_after_generated")))

            if int(item.get("generated_event_id") or 0) in next_step_generated_event_ids:
                next_step_only_raw["steps_generated"] += 1
                next_step_only_raw["steps_exposed"] += int(bool(item.get("has_exposed")))
                next_step_only_raw["steps_clicked"] += int(bool(item.get("has_clicked_after_exposure")))
                next_step_only_raw["steps_completed_after_exposure"] += int(bool(item.get("has_completed_after_exposure")))
                next_step_only_raw["steps_completed_after_generated"] += int(bool(item.get("has_completed_after_generated")))

                next_cat_bucket = next_step_only_by_category[str(item.get("category") or "__unknown__")]
                next_cat_bucket["steps_generated"] += 1
                next_cat_bucket["steps_exposed"] += int(bool(item.get("has_exposed")))
                next_cat_bucket["steps_clicked"] += int(bool(item.get("has_clicked_after_exposure")))
                next_cat_bucket["steps_completed_after_exposure"] += int(bool(item.get("has_completed_after_exposure")))
                next_cat_bucket["steps_completed_after_generated"] += int(bool(item.get("has_completed_after_generated")))

                if bool(item.get("has_recommendation")):
                    adherence_raw["next_step_with_recommendation"] += 1
                    adherence_by_category[str(item.get("category") or "__unknown__")]["next_step_with_recommendation"] += 1
                    adherence_by_step_bucket[_adherence_step_bucket(_to_int(item.get("step_index")))]["next_step_with_recommendation"] += 1
                    adherence_by_ml_decision[str(item.get("ml_decision") or "missing_ml_meta")]["next_step_with_recommendation"] += 1

                    if bool(item.get("has_targeted_next_step_checkout")):
                        adherence_raw["checkout_targeted_next_step"] += 1
                        adherence_by_category[str(item.get("category") or "__unknown__")]["checkout_targeted_next_step"] += 1
                        adherence_by_step_bucket[_adherence_step_bucket(_to_int(item.get("step_index")))]["checkout_targeted_next_step"] += 1
                        adherence_by_ml_decision[str(item.get("ml_decision") or "missing_ml_meta")]["checkout_targeted_next_step"] += 1

                    if bool(item.get("has_recommended_product_checkout")):
                        adherence_raw["checkout_targeted_recommended_product"] += 1
                        adherence_by_category[str(item.get("category") or "__unknown__")]["checkout_targeted_recommended_product"] += 1
                        adherence_by_step_bucket[_adherence_step_bucket(_to_int(item.get("step_index")))]["checkout_targeted_recommended_product"] += 1
                        adherence_by_ml_decision[str(item.get("ml_decision") or "missing_ml_meta")]["checkout_targeted_recommended_product"] += 1

        overall_raw_final = _finalize_bucket(overall_raw)
        overall_analysis_final = _finalize_bucket(overall_analysis)
        overall_analysis_final["excluded_missing_ml_meta_plans"] = int(excluded_missing_plan_count)
        overall_analysis_final["excluded_missing_ml_meta_steps"] = int(excluded_missing_step_count)

        payload: dict[str, Any] = {
            "generated_at_utc": now_utc.isoformat(),
            "window_start_utc": since.isoformat(),
            "window_end_utc": now_utc.isoformat(),
            "params": {
                "days": days,
                "category": category,
                "include_ga": include_ga,
                "format": out_format,
                "cohort_mode": cohort_mode,
            },
            "overall": {
                "raw": overall_raw_final,
                "analysis": overall_analysis_final,
            },
            "next_step_only": {
                "raw": _finalize_bucket(next_step_only_raw),
                "by_category": {k: _finalize_bucket(v) for k, v in sorted(next_step_only_by_category.items())},
            },
            "recommended_product_adherence": {
                "raw": _finalize_adherence_bucket(adherence_raw),
                "by_category": {k: _finalize_adherence_bucket(v) for k, v in sorted(adherence_by_category.items())},
                "by_step_bucket": {k: _finalize_adherence_bucket(v) for k, v in sorted(adherence_by_step_bucket.items())},
                "by_ml_decision": {k: _finalize_adherence_bucket(v) for k, v in sorted(adherence_by_ml_decision.items())},
            },
            "breakdowns": {
                "by_category": {k: _finalize_bucket(v) for k, v in sorted(by_category.items())},
                "by_step_index": {k: _finalize_bucket(v) for k, v in sorted(by_step_index.items())},
                "by_product_type": {k: _finalize_bucket(v) for k, v in sorted(by_product_type.items())},
                "by_ml_decision": {k: _finalize_bucket(v) for k, v in sorted(by_ml_decision.items())},
                "by_expose_source": {k: _finalize_bucket(v) for k, v in sorted(by_expose_source.items())},
            },
            "notes": [
                "Read-only report: no DB writes, no runtime logic changes.",
                "Step funnel is built from STEP_GENERATED instances and attributes exposed/clicked/completed only until the next STEP_GENERATED for the same step.",
                "By category/step/product_type/expose_source breakdowns use analysis cohort; by_ml_decision always keeps missing_ml_meta separate.",
                "next_step_only uses PLAN_REFRESHED.next_step_id to isolate the active next-step funnel per refresh.",
                "recommended_product_adherence uses next_step_only generated instances with has_recommendation=true and STEP_COMPLETED.matched_by to separate product-type completion from exact recommended_product_id adoption.",
                "steps_completed_after_generated can be higher than steps_completed_after_exposure when purchases happen without a tracked exposure in the same generated-instance window.",
            ],
        }

        markdown = _build_markdown(payload)
        json_text = json.dumps(payload, ensure_ascii=False, indent=2)
        out_stem = _resolve_out_stem(out=out_raw, days=days, category=category)
        wrote_paths: list[Path] = []

        if out_format in {"md", "both"}:
            if out_raw:
                md_path = out_stem.with_suffix(".md")
                md_path.parent.mkdir(parents=True, exist_ok=True)
                md_path.write_text(markdown, encoding="utf-8")
                wrote_paths.append(md_path)
            else:
                self.stdout.write(markdown)

        if out_format in {"json", "both"}:
            if out_raw:
                json_path = out_stem.with_suffix(".json")
                json_path.parent.mkdir(parents=True, exist_ok=True)
                json_path.write_text(json_text, encoding="utf-8")
                wrote_paths.append(json_path)
            else:
                if out_format == "json":
                    self.stdout.write(json_text)
                else:
                    self.stdout.write("\n---\n")
                    self.stdout.write(json_text)

        for path in wrote_paths:
            self.stdout.write(f"[report_roadmap_generation_gap] wrote: {path}")
