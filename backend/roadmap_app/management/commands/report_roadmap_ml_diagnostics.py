from __future__ import annotations

import json
from collections import Counter, defaultdict
from datetime import timedelta
from pathlib import Path
from typing import Any

from django.conf import settings
from django.core.management.base import BaseCommand, CommandError
from django.db.models import Count
from django.db.models.functions import Coalesce
from django.utils import timezone

from offers.models import OfferAssignment, OfferEvent
from roadmap_app.ml_next_step import v4_category_staged_rollout_status
from roadmap_app.models import RoadmapEvent, RoadmapPlan, RoadmapStep
from transactions.models import Transaction


FORMAT_CHOICES = ["md", "json", "both"]
COHORT_MODE_CHOICES = ["fresh", "all"]
CONTROL_CHOICES = ["non_model", "fallback", "disabled"]
VALID_CATEGORIES = {"skincare", "haircare", "makeup", "fragrance"}
VALID_DECISIONS = {"model_used", "fallback", "disabled"}
DEFAULT_CATEGORIES = ["skincare", "makeup"]


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


def _rate(n: float, d: float) -> float | None:
    if d <= 0:
        return None
    return float(n) / float(d)


def _round_or_none(value: float | None, ndigits: int = 6) -> float | None:
    if value is None:
        return None
    return round(float(value), ndigits)


def _pct(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{float(value) * 100.0:.2f}%"


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


def _new_bucket() -> dict[str, Any]:
    return {
        "plans": set(),
        "users": set(),
        "step_exposed": 0,
        "step_clicked": 0,
        "step_completed": 0,
        "step_skipped": 0,
        "offer_assigned": 0,
        "offer_exposed": 0,
        "offer_clicked": 0,
        "offer_redeemed": 0,
    }


def _decision_from_meta(meta: dict[str, Any]) -> str:
    ml = _safe_dict(meta.get("ml"))
    decision = str(ml.get("decision") or "").strip().lower()
    if decision in VALID_DECISIONS:
        return decision
    return "missing_ml_meta"


def _decision_to_cohort(decision: str, *, cohort_mode: str, control_decisions: set[str]) -> str | None:
    if cohort_mode == "fresh" and decision == "missing_ml_meta":
        return None
    if decision == "model_used":
        return "model_used"
    if decision in control_decisions:
        return "control"
    return None


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


def _is_roadmap_related_assignment(*, reason: dict[str, Any], target: dict[str, Any]) -> bool:
    picked_via = str(target.get("picked_via") or "").strip().lower()
    if picked_via.startswith("roadmap_shortcut"):
        return True
    roadmap_reason = reason.get("roadmap")
    if isinstance(roadmap_reason, dict) and roadmap_reason:
        return True
    roadmap_ctx = reason.get("roadmap_ctx")
    if isinstance(roadmap_ctx, dict) and roadmap_ctx:
        return True
    source = str(reason.get("source") or "").strip().lower()
    if source.startswith("roadmap"):
        return True
    return False


def _parse_categories(raw: str | None) -> list[str]:
    text = str(raw or "").strip()
    if not text:
        return list(DEFAULT_CATEGORIES)
    if text.lower() == "all":
        return sorted(VALID_CATEGORIES)
    out: list[str] = []
    for token in text.split(","):
        cat = str(token or "").strip().lower()
        if not cat:
            continue
        if cat not in VALID_CATEGORIES:
            raise CommandError(f"Unknown category in --categories: {cat}")
        out.append(cat)
    if not out:
        raise CommandError("--categories resolved to empty set")
    return sorted(set(out))


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


def _resolve_out_stem(*, out: str | None, days: int) -> Path:
    if out:
        p = Path(out)
        if p.suffix.lower() in {".md", ".json"}:
            return p.with_suffix("")
        return p
    return Path("reports") / f"roadmap_ml_diagnostics_{days}d"


def _bucket_user_activity(tx_count_90d: int) -> str:
    if tx_count_90d <= 1:
        return "new_or_rare"
    if tx_count_90d <= 5:
        return "mid"
    return "active_frequent"


def _lift(model_rate: float | None, control_rate: float | None) -> float | None:
    if model_rate is None or control_rate is None:
        return None
    return float(model_rate) - float(control_rate)


def _serialize_bucket(bucket: dict[str, Any]) -> dict[str, Any]:
    return {
        "plans": int(len(bucket["plans"])),
        "users": int(len(bucket["users"])),
        "step_exposed": int(bucket["step_exposed"]),
        "step_clicked": int(bucket["step_clicked"]),
        "step_completed": int(bucket["step_completed"]),
        "step_skipped": int(bucket["step_skipped"]),
        "offer_assigned": int(bucket["offer_assigned"]),
        "offer_exposed": int(bucket["offer_exposed"]),
        "offer_clicked": int(bucket["offer_clicked"]),
        "offer_redeemed": int(bucket["offer_redeemed"]),
        "step_ctr": _round_or_none(_rate(bucket["step_clicked"], bucket["step_exposed"])),
        "step_completion_rate": _round_or_none(_rate(bucket["step_completed"], bucket["step_exposed"])),
        "skip_rate": _round_or_none(_rate(bucket["step_skipped"], bucket["step_exposed"])),
        "offer_ctr": _round_or_none(_rate(bucket["offer_clicked"], bucket["offer_exposed"])),
        "offer_redeem_rate": _round_or_none(_rate(bucket["offer_redeemed"], bucket["offer_exposed"])),
    }


def _slice_verdict(
    *,
    model_plans: int,
    control_plans: int,
    step_completion_lift: float | None,
    offer_redeem_lift: float | None,
    step_ctr_lift: float | None,
    offer_ctr_lift: float | None,
    min_sample: int,
    min_step_completion_lift: float,
    min_offer_redeem_lift: float,
    max_negative_step_ctr_lift_soft: float,
    max_negative_offer_ctr_lift_soft: float,
) -> str:
    if model_plans < min_sample or control_plans < min_sample:
        return "LOW_SAMPLE"
    primary_passed = bool(
        (step_completion_lift is not None and step_completion_lift >= min_step_completion_lift)
        or (offer_redeem_lift is not None and offer_redeem_lift >= min_offer_redeem_lift)
    )
    if not primary_passed:
        return "HOLD"
    severe_ctr = bool(
        (step_ctr_lift is not None and step_ctr_lift < max_negative_step_ctr_lift_soft)
        or (offer_ctr_lift is not None and offer_ctr_lift < max_negative_offer_ctr_lift_soft)
    )
    if severe_ctr:
        return "HOLD"
    return "ENABLE_CANDIDATE"


def _safe_number(value: float | int | None, default: float = 0.0) -> float:
    if value is None:
        return float(default)
    return float(value)


def _build_markdown(payload: dict[str, Any]) -> str:
    params = _safe_dict(payload.get("params"))
    executive = _safe_dict(payload.get("executive_summary"))
    offenders = _safe_list(payload.get("worst_offenders"))
    candidates = _safe_list(payload.get("best_enable_candidates"))
    slice_breakdowns = _safe_dict(payload.get("slice_breakdowns"))
    recommendations = _safe_dict(payload.get("recommendations"))
    simulations = _safe_list(payload.get("policy_simulation"))
    runtime = _safe_dict(payload.get("runtime_observability"))
    unattributed = _safe_dict(payload.get("unattributed"))

    lines: list[str] = []
    lines.append("# Roadmap ML Diagnostics Report")
    lines.append("")
    lines.append(f"- Generated at (UTC): `{payload.get('generated_at_utc')}`")
    lines.append(
        f"- Window: `{payload.get('window_start_utc')}` .. `{payload.get('window_end_utc')}` "
        f"(days={params.get('days')})"
    )
    lines.append(
        f"- Categories: `{', '.join(_safe_list(params.get('categories')) or [])}` | "
        f"Cohort mode: `{params.get('cohort_mode')}` | Control: `{params.get('control')}` | "
        f"include_ga: `{params.get('include_ga')}`"
    )
    lines.append("")

    lines.append("## 1) Executive summary")
    for cat in _safe_list(params.get("categories")):
        cat_summary = _safe_dict(executive.get(str(cat)))
        lines.append(
            f"- {cat}: hold_driver=`{cat_summary.get('hold_driver')}`, "
            f"partial_enable=`{cat_summary.get('partial_enable')}`, "
            f"recommendation=`{cat_summary.get('recommendation')}`"
        )
    lines.append("")

    lines.append("## 2) Worst offenders")
    lines.append(
        _md_table(
            [
                "category",
                "slice_type",
                "slice_value",
                "model_plans",
                "control_plans",
                "step_ctr_lift_pp",
                "step_completion_lift_pp",
                "offer_ctr_lift_pp",
                "offer_redeem_lift_pp",
                "verdict",
            ],
            [
                [
                    row.get("category"),
                    row.get("slice_type"),
                    row.get("slice_value"),
                    row.get("model_plans"),
                    row.get("control_plans"),
                    f"{_safe_number(row.get('step_ctr_lift')) * 100.0:.2f}",
                    f"{_safe_number(row.get('step_completion_lift')) * 100.0:.2f}",
                    f"{_safe_number(row.get('offer_ctr_lift')) * 100.0:.2f}",
                    f"{_safe_number(row.get('offer_redeem_lift')) * 100.0:.2f}",
                    row.get("verdict"),
                ]
                for row in offenders
            ],
        )
    )
    lines.append("")

    lines.append("## 3) Best candidates for partial enable")
    lines.append(
        _md_table(
            [
                "category",
                "slice_type",
                "slice_value",
                "model_plans",
                "control_plans",
                "step_completion_lift_pp",
                "offer_redeem_lift_pp",
                "recommendation",
            ],
            [
                [
                    row.get("category"),
                    row.get("slice_type"),
                    row.get("slice_value"),
                    row.get("model_plans"),
                    row.get("control_plans"),
                    f"{_safe_number(row.get('step_completion_lift')) * 100.0:.2f}",
                    f"{_safe_number(row.get('offer_redeem_lift')) * 100.0:.2f}",
                    row.get("verdict"),
                ]
                for row in candidates
            ],
        )
    )
    lines.append("")

    lines.append("## 4) Slice diagnostics")
    for cat in _safe_list(params.get("categories")):
        cat_block = _safe_dict(slice_breakdowns.get(str(cat)))
        lines.append(f"### {cat}")
        for slice_type in [
            "step_product_type",
            "step_index",
            "offer_presence",
            "expose_source",
            "user_activity",
        ]:
            rows_payload = _safe_list(cat_block.get(slice_type))
            lines.append(f"#### {slice_type}")
            lines.append(
                _md_table(
                    [
                        "slice_value",
                        "model_plans",
                        "control_plans",
                        "model_exposed",
                        "control_exposed",
                        "step_ctr_lift_pp",
                        "step_completion_lift_pp",
                        "offer_ctr_lift_pp",
                        "offer_redeem_lift_pp",
                        "verdict",
                    ],
                    [
                        [
                            row.get("slice_value"),
                            row.get("model_plans"),
                            row.get("control_plans"),
                            row.get("model_exposed"),
                            row.get("control_exposed"),
                            f"{_safe_number(row.get('step_ctr_lift')) * 100.0:.2f}",
                            f"{_safe_number(row.get('step_completion_lift')) * 100.0:.2f}",
                            f"{_safe_number(row.get('offer_ctr_lift')) * 100.0:.2f}",
                            f"{_safe_number(row.get('offer_redeem_lift')) * 100.0:.2f}",
                            row.get("verdict"),
                        ]
                        for row in rows_payload[:20]
                    ],
                )
            )
            lines.append("")

    lines.append("## 5) Recommendation")
    lines.append(
        _md_table(
            ["category", "current_rollout_status", "decision", "why", "partial_candidate_count", "partial_plan_coverage_pct"],
            [
                [
                    cat,
                    _safe_dict(rec).get("current_rollout_status"),
                    _safe_dict(rec).get("decision"),
                    _safe_dict(rec).get("why"),
                    _safe_dict(rec).get("partial_candidate_count"),
                    f"{_safe_number(_safe_dict(rec).get('partial_plan_coverage')) * 100.0:.2f}",
                ]
                for cat, rec in sorted(recommendations.items())
            ],
        )
    )
    lines.append("")

    lines.append("## 6) Policy simulation (offline what-if)")
    lines.append(
        _md_table(
            [
                "policy",
                "plans_covered",
                "model_used_share_pct",
                "expected_step_completion_lift_pp",
                "expected_offer_redeem_lift_pp",
                "expected_step_ctr_lift_pp",
                "expected_offer_ctr_lift_pp",
            ],
            [
                [
                    row.get("policy"),
                    row.get("plans_covered"),
                    f"{_safe_number(row.get('model_used_share')) * 100.0:.2f}",
                    f"{_safe_number(row.get('step_completion_lift')) * 100.0:.2f}",
                    f"{_safe_number(row.get('offer_redeem_lift')) * 100.0:.2f}",
                    f"{_safe_number(row.get('step_ctr_lift')) * 100.0:.2f}",
                    f"{_safe_number(row.get('offer_ctr_lift')) * 100.0:.2f}",
                ]
                for row in simulations
            ],
        )
    )
    lines.append("")

    lines.append("## 7) Runtime observability")
    decision_counts = _safe_dict(runtime.get("decision_counts"))
    lines.append(
        _md_table(
            ["decision", "count"],
            [[k, v] for k, v in sorted(decision_counts.items(), key=lambda kv: kv[0])],
        )
    )
    lines.append("### fallback reasons")
    lines.append(
        _md_table(
            ["reason", "count"],
            [[k, v] for k, v in sorted(_safe_dict(runtime.get("fallback_reasons")).items(), key=lambda kv: (-kv[1], kv[0]))],
        )
    )
    lines.append("### disabled reasons")
    lines.append(
        _md_table(
            ["reason", "count"],
            [[k, v] for k, v in sorted(_safe_dict(runtime.get("disabled_reasons")).items(), key=lambda kv: (-kv[1], kv[0]))],
        )
    )
    lines.append("")

    lines.append("## 8) Unattributed / excluded")
    lines.append(
        _md_table(
            ["bucket", "count"],
            [[k, v] for k, v in sorted(unattributed.items(), key=lambda kv: kv[0])],
        )
    )
    lines.append("")

    lines.append("## 9) Notes")
    for note in _safe_list(payload.get("notes")):
        lines.append(f"- {note}")
    lines.append("")

    return "\n".join(lines)


class Command(BaseCommand):
    help = "Read-only diagnostics report for Roadmap ML runtime (category + sub-slice breakdowns)."

    def add_arguments(self, parser):
        parser.add_argument("--days", type=int, default=7)
        parser.add_argument("--include-ga", action="store_true", default=False)
        parser.add_argument("--categories", type=str, default="skincare,makeup")
        parser.add_argument("--format", type=str, default="both", choices=FORMAT_CHOICES)
        parser.add_argument("--out", type=str, default=None)
        parser.add_argument("--min-sample", type=int, default=30)
        parser.add_argument("--cohort-mode", type=str, default="fresh", choices=COHORT_MODE_CHOICES)
        parser.add_argument("--control", type=str, default="non_model", choices=CONTROL_CHOICES)

    def handle(self, *args, **options):
        days = int(options["days"] or 7)
        if days <= 0:
            raise CommandError("--days must be > 0")

        include_ga = bool(options["include_ga"])
        categories = _parse_categories(options.get("categories"))
        out_format = str(options["format"] or "both").strip().lower()
        out_raw = options.get("out")
        min_sample = int(options["min_sample"] or 30)
        if min_sample <= 0:
            raise CommandError("--min-sample must be > 0")
        cohort_mode = str(options["cohort_mode"] or "fresh").strip().lower()
        control = str(options["control"] or "non_model").strip().lower()

        if control == "fallback":
            control_decisions = {"fallback"}
        elif control == "disabled":
            control_decisions = {"disabled"}
        else:
            control_decisions = {"fallback", "disabled"}

        min_step_completion_lift = float(getattr(settings, "ROADMAP_NEXTSTEP_V4_MIN_STEP_COMPLETION_LIFT", 0.01))
        min_offer_redeem_lift = float(getattr(settings, "ROADMAP_NEXTSTEP_V4_MIN_OFFER_REDEEM_LIFT", 0.005))
        max_negative_step_ctr_lift_soft = float(
            getattr(settings, "ROADMAP_NEXTSTEP_V4_MAX_NEGATIVE_STEP_CTR_LIFT_SOFT", -0.02)
        )
        max_negative_offer_ctr_lift_soft = float(
            getattr(settings, "ROADMAP_NEXTSTEP_V4_MAX_NEGATIVE_OFFER_CTR_LIFT_SOFT", -0.03)
        )

        now_utc = timezone.now()
        since = now_utc - timedelta(days=days)

        plan_qs = RoadmapPlan.objects.filter(updated_at__gte=since, updated_at__lte=now_utc, category__in=categories)
        if not include_ga:
            plan_qs = plan_qs.exclude(user__username__startswith="ga_")

        plan_rows = list(plan_qs.values("id", "user_id", "category", "updated_at", "meta"))

        all_scope_plan_ids: set[int] = set()
        plan_category: dict[int, str] = {}
        plan_user: dict[int, int] = {}
        plan_updated: dict[int, Any] = {}
        plan_decision: dict[int, str] = {}
        cohort_by_plan: dict[int, str] = {}
        decision_counts: Counter[str] = Counter()
        fallback_reason_counts: Counter[str] = Counter()
        disabled_reason_counts: Counter[str] = Counter()
        mode_counts: Counter[str] = Counter()

        for row in plan_rows:
            pid = int(row["id"])
            cat = str(row["category"] or "")
            user_id = int(row["user_id"])
            meta = _safe_dict(row.get("meta"))
            ml = _safe_dict(meta.get("ml"))
            decision = _decision_from_meta(meta)

            all_scope_plan_ids.add(pid)
            plan_category[pid] = cat
            plan_user[pid] = user_id
            plan_updated[pid] = row["updated_at"]
            plan_decision[pid] = decision

            decision_counts[decision] += 1
            if decision == "fallback":
                fallback_reason_counts[str(ml.get("fallback_reason") or "__missing_reason__")] += 1
            elif decision == "disabled":
                disabled_reason_counts[str(ml.get("disabled_reason") or "__missing_reason__")] += 1
            mode_counts[str(ml.get("mode") or "none")] += 1

        if cohort_mode == "fresh":
            cohort_scope_plan_ids = {pid for pid in all_scope_plan_ids if plan_decision.get(pid) != "missing_ml_meta"}
        else:
            cohort_scope_plan_ids = set(all_scope_plan_ids)

        for pid in cohort_scope_plan_ids:
            cohort = _decision_to_cohort(
                plan_decision.get(pid, "missing_ml_meta"),
                cohort_mode=cohort_mode,
                control_decisions=control_decisions,
            )
            if cohort:
                cohort_by_plan[pid] = cohort

        analysis_plan_ids = set(cohort_by_plan.keys())
        model_plan_ids = {pid for pid in analysis_plan_ids if cohort_by_plan.get(pid) == "model_used"}
        control_plan_ids = {pid for pid in analysis_plan_ids if cohort_by_plan.get(pid) == "control"}

        tx_since = now_utc - timedelta(days=90)
        tx_qs = Transaction.objects.filter(created_at__gte=tx_since, created_at__lte=now_utc)
        if not include_ga:
            tx_qs = tx_qs.exclude(user__username__startswith="ga_")
        tx_counts = {
            int(row["user_id"]): int(row["c"] or 0)
            for row in tx_qs.values("user_id").annotate(c=Count("id"))
        }
        plan_activity_bucket: dict[int, str] = {
            pid: _bucket_user_activity(int(tx_counts.get(plan_user[pid], 0)))
            for pid in analysis_plan_ids
        }

        plan_step_product_types: dict[int, set[str]] = defaultdict(set)
        plan_step_index_buckets: dict[int, set[str]] = defaultdict(set)
        step_meta: dict[int, dict[str, Any]] = {}
        if analysis_plan_ids:
            step_rows = RoadmapStep.objects.filter(plan_id__in=analysis_plan_ids).values(
                "id",
                "plan_id",
                "step_index",
                "product_type",
            )
            for row in step_rows:
                sid = int(row["id"])
                pid = int(row["plan_id"])
                product_type = str(row["product_type"] or "").strip() or "__unknown__"
                idx_bucket = _step_index_bucket(_to_int(row.get("step_index")))
                plan_step_product_types[pid].add(product_type)
                plan_step_index_buckets[pid].add(idx_bucket)
                step_meta[sid] = {
                    "plan_id": pid,
                    "category": plan_category.get(pid, "__unknown__"),
                    "product_type": product_type,
                    "step_index_bucket": idx_bucket,
                }

        slice_buckets: dict[str, dict[str, dict[str, dict[str, Any]]]] = defaultdict(
            lambda: defaultdict(lambda: defaultdict(lambda: {"model_used": _new_bucket(), "control": _new_bucket()}))
        )
        slice_plan_sets: dict[str, dict[str, dict[str, set[int]]]] = defaultdict(
            lambda: defaultdict(lambda: defaultdict(set))
        )
        slice_active_plan_sets: dict[str, dict[str, dict[str, set[int]]]] = defaultdict(
            lambda: defaultdict(lambda: defaultdict(set))
        )
        category_overall: dict[str, dict[str, Any]] = defaultdict(
            lambda: {"model_used": _new_bucket(), "control": _new_bucket()}
        )
        plan_metrics: dict[int, dict[str, float]] = defaultdict(
            lambda: {
                "step_exposed": 0.0,
                "step_clicked": 0.0,
                "step_completed": 0.0,
                "step_skipped": 0.0,
                "offer_assigned": 0.0,
                "offer_exposed": 0.0,
                "offer_clicked": 0.0,
                "offer_redeemed": 0.0,
            }
        )

        def _touch_plan_slice(cat: str, slice_type: str, slice_value: str, cohort: str, *, pid: int, uid: int) -> None:
            bucket = slice_buckets[cat][slice_type][slice_value][cohort]
            bucket["plans"].add(pid)
            bucket["users"].add(uid)
            slice_plan_sets[cat][slice_type][slice_value].add(pid)

        def _touch_active_slice(cat: str, slice_type: str, slice_value: str, *, pid: int) -> None:
            slice_active_plan_sets[cat][slice_type][slice_value].add(pid)

        for pid in analysis_plan_ids:
            cohort = str(cohort_by_plan.get(pid))
            cat = str(plan_category.get(pid) or "__unknown__")
            uid = int(plan_user.get(pid))
            category_overall[cat][cohort]["plans"].add(pid)
            category_overall[cat][cohort]["users"].add(uid)
            _touch_plan_slice(cat, "user_activity", plan_activity_bucket.get(pid, "__unknown__"), cohort, pid=pid, uid=uid)
            for pt in sorted(plan_step_product_types.get(pid, set())):
                _touch_plan_slice(cat, "step_product_type", pt, cohort, pid=pid, uid=uid)
            for idx_bucket in sorted(plan_step_index_buckets.get(pid, set())):
                _touch_plan_slice(cat, "step_index", idx_bucket, cohort, pid=pid, uid=uid)

        step_sources_by_key: dict[tuple[str, int], set[str]] = defaultdict(set)
        category_plan_ids: dict[str, set[int]] = defaultdict(set)
        for pid in analysis_plan_ids:
            category_plan_ids[str(plan_category.get(pid) or "__unknown__")].add(pid)

        unattributed: Counter[str] = Counter()

        def _increment_step_metric(
            *,
            pid: int,
            sid: int | None,
            metric_key: str,
            source: str | None = None,
            update_primary: bool = True,
        ) -> None:
            cohort = str(cohort_by_plan.get(pid) or "")
            if cohort not in {"model_used", "control"}:
                return
            cat = str(plan_category.get(pid) or "__unknown__")
            uid = int(plan_user.get(pid))

            if update_primary:
                category_bucket = category_overall[cat][cohort]
                category_bucket[metric_key] += 1
                category_bucket["plans"].add(pid)
                category_bucket["users"].add(uid)
                plan_metrics[pid][metric_key] += 1.0

                step_info = _safe_dict(step_meta.get(int(sid or 0)))
                product_type = str(step_info.get("product_type") or "__unknown__")
                step_index_bucket = str(step_info.get("step_index_bucket") or "__unknown__")

                for slice_type, slice_value in (
                    ("step_product_type", product_type),
                    ("step_index", step_index_bucket),
                ):
                    _touch_plan_slice(cat, slice_type, slice_value, cohort, pid=pid, uid=uid)
                    slice_buckets[cat][slice_type][slice_value][cohort][metric_key] += 1
                    _touch_active_slice(cat, slice_type, slice_value, pid=pid)

            if source:
                _touch_plan_slice(cat, "expose_source", source, cohort, pid=pid, uid=uid)
                slice_buckets[cat]["expose_source"][source][cohort][metric_key] += 1
                _touch_active_slice(cat, "expose_source", source, pid=pid)

        if analysis_plan_ids:
            exposed_qs = (
                RoadmapEvent.objects.filter(
                    created_at__gte=since,
                    created_at__lte=now_utc,
                    event_type=RoadmapEvent.Type.STEP_EXPOSED,
                )
                .annotate(_effective_plan_id=Coalesce("plan_id", "step__plan_id"))
                .filter(_effective_plan_id__in=analysis_plan_ids)
            )
            if not include_ga:
                exposed_qs = exposed_qs.exclude(user__username__startswith="ga_")

            for row in exposed_qs.values("step_id", "context", "_effective_plan_id"):
                pid = int(row["_effective_plan_id"])
                sid = _to_int(row.get("step_id"))
                source = _source_from_expose_context(_safe_dict(row.get("context")))
                _increment_step_metric(
                    pid=pid,
                    sid=sid,
                    metric_key="step_exposed",
                    source=source,
                )
                if sid is not None:
                    step_sources_by_key[(str(cohort_by_plan.get(pid)), sid)].add(source)

            interaction_qs = (
                RoadmapEvent.objects.filter(
                    created_at__gte=since,
                    created_at__lte=now_utc,
                    event_type__in=[
                        RoadmapEvent.Type.STEP_CLICKED,
                        RoadmapEvent.Type.STEP_COMPLETED,
                        RoadmapEvent.Type.STEP_SKIPPED,
                    ],
                )
                .annotate(_effective_plan_id=Coalesce("plan_id", "step__plan_id"))
                .filter(_effective_plan_id__in=analysis_plan_ids)
            )
            if not include_ga:
                interaction_qs = interaction_qs.exclude(user__username__startswith="ga_")

            event_map = {
                RoadmapEvent.Type.STEP_CLICKED: "step_clicked",
                RoadmapEvent.Type.STEP_COMPLETED: "step_completed",
                RoadmapEvent.Type.STEP_SKIPPED: "step_skipped",
            }
            for row in interaction_qs.values("step_id", "event_type", "_effective_plan_id"):
                pid = int(row["_effective_plan_id"])
                sid = _to_int(row.get("step_id"))
                metric = event_map.get(str(row.get("event_type") or ""))
                if not metric:
                    continue
                _increment_step_metric(pid=pid, sid=sid, metric_key=metric, source=None)
                if sid is None:
                    unattributed["step_interaction_missing_step_id"] += 1
                    continue
                sources = step_sources_by_key.get((str(cohort_by_plan.get(pid)), sid), set())
                if not sources:
                    unattributed["step_interaction_without_exposed_source"] += 1
                    continue
                for source in sources:
                    _increment_step_metric(
                        pid=pid,
                        sid=sid,
                        metric_key=metric,
                        source=source,
                        update_primary=False,
                    )

        plans_by_user_scope: dict[int, list[dict[str, Any]]] = defaultdict(list)
        plan_step_types: dict[int, set[str]] = defaultdict(set)
        plan_steps_for_product_type: dict[tuple[int, str], list[int]] = defaultdict(list)
        for pid in analysis_plan_ids:
            plans_by_user_scope[int(plan_user[pid])].append(
                {
                    "id": int(pid),
                    "updated_at": plan_updated[pid],
                    "category": str(plan_category[pid]),
                }
            )
        for sid, info in step_meta.items():
            pid = int(info["plan_id"])
            pt = str(info["product_type"])
            step_idx_bucket = str(info["step_index_bucket"])
            if pid not in analysis_plan_ids:
                continue
            plan_step_types[pid].add(pt)
            if step_idx_bucket == "step_1":
                step_index_val = 1
            elif step_idx_bucket == "step_2":
                step_index_val = 2
            elif step_idx_bucket == "step_3":
                step_index_val = 3
            else:
                step_index_val = 4
            plan_steps_for_product_type[(pid, pt)].append(step_index_val)

        roadmap_assignment_total = 0
        roadmap_assignment_attributed = 0
        roadmap_assignment_unattributed = 0
        roadmap_assignment_out_of_scope = 0
        roadmap_assignment_excluded_non_cohort = 0

        assignment_state: dict[int, dict[str, Any]] = {}
        roadmap_assignment_ids: set[int] = set()
        plan_offer_presence: dict[int, dict[str, int]] = defaultdict(
            lambda: {"assigned": 0, "exposed": 0, "clicked": 0, "redeemed": 0}
        )

        assignment_qs = OfferAssignment.objects.filter(assigned_at__gte=since, assigned_at__lte=now_utc)
        if not include_ga:
            assignment_qs = assignment_qs.exclude(user__username__startswith="ga_")

        for row in assignment_qs.values("id", "user_id", "assigned_at", "reason", "target"):
            reason = _safe_dict(row.get("reason"))
            target = _safe_dict(row.get("target"))
            if not _is_roadmap_related_assignment(reason=reason, target=target):
                continue

            roadmap_assignment_total += 1
            assignment_id = int(row["id"])
            roadmap_assignment_ids.add(assignment_id)
            user_id = int(row["user_id"])
            assigned_at = row["assigned_at"]

            roadmap_reason = _safe_dict(reason.get("roadmap"))
            roadmap_ctx = _safe_dict(reason.get("roadmap_ctx"))
            attributed_plan_id = _to_int(roadmap_reason.get("plan_id"))
            attribution_kind = "explicit_plan_id"
            if attributed_plan_id is None:
                attributed_plan_id = _to_int(roadmap_ctx.get("plan_id"))
                if attributed_plan_id is not None:
                    attribution_kind = "explicit_ctx_plan_id"

            category_hint = str(
                roadmap_reason.get("category")
                or roadmap_ctx.get("category")
                or target.get("category")
                or ""
            ).strip().lower()
            product_type_hint = str(
                roadmap_reason.get("next_product_type")
                or roadmap_ctx.get("next_product_type")
                or target.get("product_type")
                or ""
            ).strip().lower()

            step_index_hint = _to_int(
                roadmap_reason.get("step_index")
                or roadmap_ctx.get("step_index")
                or target.get("step_index")
            )

            if attributed_plan_id is None:
                if not category_hint and not product_type_hint:
                    roadmap_assignment_unattributed += 1
                    assignment_state[assignment_id] = {"state": "unattributed", "why": "insufficient_context"}
                    continue

                candidates: list[dict[str, Any]] = []
                for plan_ref in plans_by_user_scope.get(user_id, []):
                    pid = int(plan_ref["id"])
                    if category_hint and str(plan_ref["category"]) != category_hint:
                        continue
                    if product_type_hint and product_type_hint not in plan_step_types.get(pid, set()):
                        continue
                    candidates.append(plan_ref)

                if len(candidates) == 1:
                    attributed_plan_id = int(candidates[0]["id"])
                    attribution_kind = "fallback_unique"
                elif len(candidates) > 1:
                    ranked = sorted(
                        [
                            (
                                abs((assigned_at - candidate["updated_at"]).total_seconds()),
                                int(candidate["id"]),
                            )
                            for candidate in candidates
                        ],
                        key=lambda x: x[0],
                    )
                    if ranked and ranked[0][0] <= 6 * 3600:
                        if len(ranked) == 1 or (ranked[0][0] + 60.0) < ranked[1][0]:
                            attributed_plan_id = int(ranked[0][1])
                            attribution_kind = "fallback_nearest"

            if attributed_plan_id is None:
                roadmap_assignment_unattributed += 1
                assignment_state[assignment_id] = {"state": "unattributed", "why": "ambiguous_or_no_match"}
                continue

            if attributed_plan_id not in analysis_plan_ids:
                roadmap_assignment_out_of_scope += 1
                assignment_state[assignment_id] = {
                    "state": "out_of_scope",
                    "plan_id": int(attributed_plan_id),
                    "attribution_kind": attribution_kind,
                }
                continue

            cohort = str(cohort_by_plan.get(int(attributed_plan_id)) or "")
            if cohort not in {"model_used", "control"}:
                roadmap_assignment_excluded_non_cohort += 1
                assignment_state[assignment_id] = {
                    "state": "non_cohort",
                    "plan_id": int(attributed_plan_id),
                    "attribution_kind": attribution_kind,
                }
                continue

            roadmap_assignment_attributed += 1
            assignment_state[assignment_id] = {
                "state": "cohort",
                "cohort": cohort,
                "plan_id": int(attributed_plan_id),
                "attribution_kind": attribution_kind,
                "category_hint": category_hint,
                "product_type_hint": product_type_hint,
                "step_index_hint": step_index_hint,
            }

            pid = int(attributed_plan_id)
            cat = str(plan_category.get(pid) or "__unknown__")
            uid = int(plan_user.get(pid))
            category_overall[cat][cohort]["offer_assigned"] += 1
            plan_metrics[pid]["offer_assigned"] += 1.0
            plan_offer_presence[pid]["assigned"] += 1

            pt_offer = product_type_hint
            if not pt_offer:
                plan_pts = sorted(plan_step_types.get(pid, set()))
                if len(plan_pts) == 1:
                    pt_offer = str(plan_pts[0])
            if pt_offer:
                _touch_plan_slice(cat, "step_product_type", pt_offer, cohort, pid=pid, uid=uid)
                slice_buckets[cat]["step_product_type"][pt_offer][cohort]["offer_assigned"] += 1
                _touch_active_slice(cat, "step_product_type", pt_offer, pid=pid)
            else:
                unattributed["offer_assignment_missing_product_type_hint"] += 1

            idx_offer_bucket: str | None = None
            if step_index_hint is not None:
                idx_offer_bucket = _step_index_bucket(step_index_hint)
            elif pt_offer:
                candidate_steps = plan_steps_for_product_type.get((pid, pt_offer), [])
                if len(candidate_steps) == 1:
                    idx_offer_bucket = _step_index_bucket(int(candidate_steps[0]))
            if idx_offer_bucket:
                _touch_plan_slice(cat, "step_index", idx_offer_bucket, cohort, pid=pid, uid=uid)
                slice_buckets[cat]["step_index"][idx_offer_bucket][cohort]["offer_assigned"] += 1
                _touch_active_slice(cat, "step_index", idx_offer_bucket, pid=pid)
            else:
                unattributed["offer_assignment_missing_step_index_hint"] += 1

        offer_unattributed_event_counts: Counter[str] = Counter()
        offer_qs = OfferEvent.objects.filter(
            created_at__gte=since,
            created_at__lte=now_utc,
            event_type__in=[
                OfferEvent.Type.EXPOSED,
                OfferEvent.Type.CLICKED,
                OfferEvent.Type.REDEEMED,
            ],
        )
        if not include_ga:
            offer_qs = offer_qs.exclude(user__username__startswith="ga_")

        offer_event_map = {
            OfferEvent.Type.EXPOSED: "offer_exposed",
            OfferEvent.Type.CLICKED: "offer_clicked",
            OfferEvent.Type.REDEEMED: "offer_redeemed",
        }
        for row in offer_qs.values("assignment_id", "event_type"):
            assignment_id = _to_int(row.get("assignment_id"))
            if assignment_id is None or assignment_id not in roadmap_assignment_ids:
                continue
            metric = offer_event_map.get(str(row.get("event_type") or ""))
            if not metric:
                continue
            state = assignment_state.get(assignment_id) or {}
            if state.get("state") != "cohort":
                offer_unattributed_event_counts[metric] += 1
                continue

            cohort = str(state.get("cohort"))
            pid = int(state.get("plan_id"))
            cat = str(plan_category.get(pid) or "__unknown__")
            uid = int(plan_user.get(pid))

            category_overall[cat][cohort][metric] += 1
            plan_metrics[pid][metric] += 1.0
            metric_short = metric.split("_", 1)[-1]
            if metric_short in {"exposed", "clicked", "redeemed"}:
                plan_offer_presence[pid][metric_short] += 1

            pt_offer = str(state.get("product_type_hint") or "").strip().lower()
            if not pt_offer:
                plan_pts = sorted(plan_step_types.get(pid, set()))
                if len(plan_pts) == 1:
                    pt_offer = str(plan_pts[0])
            if pt_offer:
                _touch_plan_slice(cat, "step_product_type", pt_offer, cohort, pid=pid, uid=uid)
                slice_buckets[cat]["step_product_type"][pt_offer][cohort][metric] += 1
                _touch_active_slice(cat, "step_product_type", pt_offer, pid=pid)
            else:
                unattributed["offer_event_missing_product_type_hint"] += 1

            idx_offer_bucket: str | None = None
            step_index_hint = _to_int(state.get("step_index_hint"))
            if step_index_hint is not None:
                idx_offer_bucket = _step_index_bucket(step_index_hint)
            elif pt_offer:
                candidate_steps = plan_steps_for_product_type.get((pid, pt_offer), [])
                if len(candidate_steps) == 1:
                    idx_offer_bucket = _step_index_bucket(int(candidate_steps[0]))
            if idx_offer_bucket:
                _touch_plan_slice(cat, "step_index", idx_offer_bucket, cohort, pid=pid, uid=uid)
                slice_buckets[cat]["step_index"][idx_offer_bucket][cohort][metric] += 1
                _touch_active_slice(cat, "step_index", idx_offer_bucket, pid=pid)
            else:
                unattributed["offer_event_missing_step_index_hint"] += 1

        for pid in analysis_plan_ids:
            cohort = str(cohort_by_plan.get(pid))
            cat = str(plan_category.get(pid) or "__unknown__")
            uid = int(plan_user.get(pid))
            p_metrics = plan_metrics.get(pid) or {}

            _touch_plan_slice(cat, "user_activity", plan_activity_bucket.get(pid, "__unknown__"), cohort, pid=pid, uid=uid)
            user_bucket = slice_buckets[cat]["user_activity"][plan_activity_bucket.get(pid, "__unknown__")][cohort]
            for metric_key in [
                "step_exposed",
                "step_clicked",
                "step_completed",
                "step_skipped",
                "offer_assigned",
                "offer_exposed",
                "offer_clicked",
                "offer_redeemed",
            ]:
                user_bucket[metric_key] += _safe_number(p_metrics.get(metric_key))

            offer_presence_key = (
                "with_offer_followup"
                if int(plan_offer_presence.get(pid, {}).get("assigned", 0)) > 0
                else "without_offer_followup"
            )
            _touch_plan_slice(cat, "offer_presence", offer_presence_key, cohort, pid=pid, uid=uid)
            offer_bucket = slice_buckets[cat]["offer_presence"][offer_presence_key][cohort]
            for metric_key in [
                "step_exposed",
                "step_clicked",
                "step_completed",
                "step_skipped",
                "offer_assigned",
                "offer_exposed",
                "offer_clicked",
                "offer_redeemed",
            ]:
                offer_bucket[metric_key] += _safe_number(p_metrics.get(metric_key))

        slice_rows: list[dict[str, Any]] = []
        for cat in sorted(slice_buckets.keys()):
            for slice_type in sorted(slice_buckets[cat].keys()):
                for slice_value in sorted(slice_buckets[cat][slice_type].keys()):
                    model_serialized = _serialize_bucket(slice_buckets[cat][slice_type][slice_value]["model_used"])
                    control_serialized = _serialize_bucket(slice_buckets[cat][slice_type][slice_value]["control"])
                    step_ctr_lift = _lift(model_serialized.get("step_ctr"), control_serialized.get("step_ctr"))
                    step_completion_lift = _lift(
                        model_serialized.get("step_completion_rate"),
                        control_serialized.get("step_completion_rate"),
                    )
                    offer_ctr_lift = _lift(model_serialized.get("offer_ctr"), control_serialized.get("offer_ctr"))
                    offer_redeem_lift = _lift(
                        model_serialized.get("offer_redeem_rate"),
                        control_serialized.get("offer_redeem_rate"),
                    )

                    verdict = _slice_verdict(
                        model_plans=int(model_serialized.get("plans") or 0),
                        control_plans=int(control_serialized.get("plans") or 0),
                        step_completion_lift=step_completion_lift,
                        offer_redeem_lift=offer_redeem_lift,
                        step_ctr_lift=step_ctr_lift,
                        offer_ctr_lift=offer_ctr_lift,
                        min_sample=min_sample,
                        min_step_completion_lift=min_step_completion_lift,
                        min_offer_redeem_lift=min_offer_redeem_lift,
                        max_negative_step_ctr_lift_soft=max_negative_step_ctr_lift_soft,
                        max_negative_offer_ctr_lift_soft=max_negative_offer_ctr_lift_soft,
                    )

                    slice_rows.append(
                        {
                            "category": cat,
                            "slice_type": slice_type,
                            "slice_value": slice_value,
                            "model": model_serialized,
                            "control": control_serialized,
                            "model_plans": int(model_serialized.get("plans") or 0),
                            "control_plans": int(control_serialized.get("plans") or 0),
                            "model_exposed": int(model_serialized.get("step_exposed") or 0),
                            "control_exposed": int(control_serialized.get("step_exposed") or 0),
                            "step_ctr_lift": _round_or_none(step_ctr_lift),
                            "step_completion_lift": _round_or_none(step_completion_lift),
                            "offer_ctr_lift": _round_or_none(offer_ctr_lift),
                            "offer_redeem_lift": _round_or_none(offer_redeem_lift),
                            "verdict": verdict,
                            "low_sample": bool(
                                int(model_serialized.get("plans") or 0) < min_sample
                                or int(control_serialized.get("plans") or 0) < min_sample
                            ),
                        }
                    )

        def _offender_rank(row: dict[str, Any]) -> tuple[float, float, float]:
            comp = float(row.get("step_completion_lift") or 0.0)
            redeem = float(row.get("offer_redeem_lift") or 0.0)
            ctr = float(row.get("step_ctr_lift") or 0.0)
            return (min(comp, redeem), comp + redeem, ctr)

        worst_offenders: list[dict[str, Any]] = []
        for cat in categories:
            cat_rows = [
                row
                for row in slice_rows
                if row["category"] == cat
                and row["slice_type"] in {"step_product_type", "step_index", "offer_presence", "expose_source", "user_activity"}
                and row["verdict"] == "HOLD"
                and not bool(row.get("low_sample"))
            ]
            cat_rows.sort(key=_offender_rank)
            worst_offenders.extend(cat_rows[:8])

        best_enable_candidates: list[dict[str, Any]] = []
        for cat in categories:
            cat_rows = [
                row
                for row in slice_rows
                if row["category"] == cat
                and row["slice_type"] in {"step_product_type", "step_index"}
                and row["verdict"] == "ENABLE_CANDIDATE"
            ]
            cat_rows.sort(
                key=lambda row: (
                    float(row.get("step_completion_lift") or 0.0) + float(row.get("offer_redeem_lift") or 0.0),
                    float(row.get("step_completion_lift") or 0.0),
                    float(row.get("offer_redeem_lift") or 0.0),
                ),
                reverse=True,
            )
            best_enable_candidates.extend(cat_rows[:12])

        partial_candidate_plan_ids: dict[str, set[int]] = defaultdict(set)
        for row in best_enable_candidates:
            cat = str(row["category"])
            stype = str(row["slice_type"])
            sval = str(row["slice_value"])
            active_ids = set(slice_active_plan_sets.get(cat, {}).get(stype, {}).get(sval, set()))
            if active_ids:
                partial_candidate_plan_ids[cat] |= active_ids
            else:
                partial_candidate_plan_ids[cat] |= set(slice_plan_sets.get(cat, {}).get(stype, {}).get(sval, set()))

        recommendations: dict[str, dict[str, Any]] = {}
        executive_summary: dict[str, dict[str, Any]] = {}

        for cat in categories:
            staged = v4_category_staged_rollout_status(cat)
            current_rollout_status = str(staged.get("final_status") or staged.get("current_decision") or "HOLD")

            cat_offenders = [row for row in worst_offenders if row["category"] == cat]
            if cat_offenders:
                top = cat_offenders[0]
                hold_driver = (
                    f"{top['slice_type']}={top['slice_value']}; "
                    f"step_completion_lift_pp={_safe_number(top.get('step_completion_lift')) * 100.0:.2f}; "
                    f"offer_redeem_lift_pp={_safe_number(top.get('offer_redeem_lift')) * 100.0:.2f}"
                )
            else:
                hold_driver = "insufficient_or_no_negative_signal"

            candidate_count = int(len([x for x in best_enable_candidates if x["category"] == cat]))
            cat_total_plans = int(len(category_plan_ids.get(cat, set())))
            cat_partial_plans = int(len(partial_candidate_plan_ids.get(cat, set())))
            partial_coverage = _rate(cat_partial_plans, cat_total_plans) or 0.0

            if current_rollout_status == "DISABLE":
                decision = "KEEP HOLD"
                why = "category is explicitly disabled by rollout policy"
            elif candidate_count > 0 and partial_coverage >= 0.1:
                decision = "ENABLE PARTIAL"
                why = "actionable positive slices exist with meaningful plan coverage"
            elif current_rollout_status == "ENABLE":
                decision = "ENABLE FULL"
                why = "category already in ENABLE and no blocker in diagnostics window"
            else:
                decision = "KEEP HOLD"
                why = "no robust positive partial slices above guard thresholds"

            recommendations[cat] = {
                "current_rollout_status": current_rollout_status,
                "decision": decision,
                "why": why,
                "partial_candidate_count": candidate_count,
                "partial_plan_coverage": _round_or_none(partial_coverage),
                "hold_driver": hold_driver,
            }
            executive_summary[cat] = {
                "hold_driver": hold_driver,
                "partial_enable": "yes" if decision == "ENABLE PARTIAL" else "no",
                "recommendation": decision,
            }

        def _simulate_policy(
            *,
            policy: str,
            category_partial_allow: dict[str, set[int]] | None = None,
        ) -> dict[str, Any]:
            category_partial_allow = category_partial_allow or {}
            model_counts: dict[str, float] = defaultdict(float)
            control_counts: dict[str, float] = defaultdict(float)
            model_plans = 0
            control_plans = 0

            for pid in sorted(analysis_plan_ids):
                cat = str(plan_category.get(pid) or "__unknown__")
                actual = str(cohort_by_plan.get(pid) or "")
                use_model = actual == "model_used"
                allowed_subset = category_partial_allow.get(cat)
                if allowed_subset is not None and actual == "model_used":
                    use_model = pid in allowed_subset
                target = model_counts if use_model else control_counts
                if use_model:
                    model_plans += 1
                else:
                    control_plans += 1
                for key, value in (plan_metrics.get(pid) or {}).items():
                    target[str(key)] += float(value or 0.0)

            total_plans = model_plans + control_plans
            model_completion = _rate(
                model_counts.get("step_completed", 0.0),
                model_counts.get("step_exposed", 0.0),
            )
            control_completion = _rate(
                control_counts.get("step_completed", 0.0),
                control_counts.get("step_exposed", 0.0),
            )
            model_offer_redeem = _rate(
                model_counts.get("offer_redeemed", 0.0),
                model_counts.get("offer_exposed", 0.0),
            )
            control_offer_redeem = _rate(
                control_counts.get("offer_redeemed", 0.0),
                control_counts.get("offer_exposed", 0.0),
            )
            model_step_ctr = _rate(
                model_counts.get("step_clicked", 0.0),
                model_counts.get("step_exposed", 0.0),
            )
            control_step_ctr = _rate(
                control_counts.get("step_clicked", 0.0),
                control_counts.get("step_exposed", 0.0),
            )
            model_offer_ctr = _rate(
                model_counts.get("offer_clicked", 0.0),
                model_counts.get("offer_exposed", 0.0),
            )
            control_offer_ctr = _rate(
                control_counts.get("offer_clicked", 0.0),
                control_counts.get("offer_exposed", 0.0),
            )

            return {
                "policy": policy,
                "plans_covered": int(total_plans),
                "model_used_share": _round_or_none(_rate(model_plans, total_plans)),
                "step_completion_lift": _round_or_none(_lift(model_completion, control_completion)),
                "offer_redeem_lift": _round_or_none(_lift(model_offer_redeem, control_offer_redeem)),
                "step_ctr_lift": _round_or_none(_lift(model_step_ctr, control_step_ctr)),
                "offer_ctr_lift": _round_or_none(_lift(model_offer_ctr, control_offer_ctr)),
                "model_plans": int(model_plans),
                "control_plans": int(control_plans),
            }

        policy_simulation = [
            _simulate_policy(policy="Policy A - current"),
            _simulate_policy(
                policy="Policy B - makeup partial",
                category_partial_allow={"makeup": set(partial_candidate_plan_ids.get("makeup", set()))},
            ),
            _simulate_policy(
                policy="Policy C - skincare partial",
                category_partial_allow={"skincare": set(partial_candidate_plan_ids.get("skincare", set()))},
            ),
        ]

        category_summary_rows: list[dict[str, Any]] = []
        for cat in categories:
            model_serialized = _serialize_bucket(category_overall[cat]["model_used"])
            control_serialized = _serialize_bucket(category_overall[cat]["control"])
            category_summary_rows.append(
                {
                    "category": cat,
                    "model": model_serialized,
                    "control": control_serialized,
                    "step_ctr_lift": _round_or_none(
                        _lift(model_serialized.get("step_ctr"), control_serialized.get("step_ctr"))
                    ),
                    "step_completion_lift": _round_or_none(
                        _lift(model_serialized.get("step_completion_rate"), control_serialized.get("step_completion_rate"))
                    ),
                    "offer_ctr_lift": _round_or_none(
                        _lift(model_serialized.get("offer_ctr"), control_serialized.get("offer_ctr"))
                    ),
                    "offer_redeem_lift": _round_or_none(
                        _lift(model_serialized.get("offer_redeem_rate"), control_serialized.get("offer_redeem_rate"))
                    ),
                }
            )

        by_category_and_type: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
        for row in slice_rows:
            by_category_and_type[str(row["category"])][str(row["slice_type"])].append(row)
        for cat in by_category_and_type:
            for stype in by_category_and_type[cat]:
                by_category_and_type[cat][stype].sort(
                    key=lambda row: (
                        -int(row.get("model_plans") or 0) - int(row.get("control_plans") or 0),
                        str(row.get("slice_value")),
                    )
                )

        fresh_excluded_missing = len(all_scope_plan_ids - cohort_scope_plan_ids) if cohort_mode == "fresh" else 0
        excluded_non_selected = len(cohort_scope_plan_ids - analysis_plan_ids)

        payload: dict[str, Any] = {
            "generated_at_utc": now_utc.isoformat(),
            "window_start_utc": since.isoformat(),
            "window_end_utc": now_utc.isoformat(),
            "params": {
                "days": days,
                "include_ga": include_ga,
                "categories": categories,
                "format": out_format,
                "cohort_mode": cohort_mode,
                "control": control,
                "min_sample": min_sample,
            },
            "overall": {
                "plans_total_in_scope": len(all_scope_plan_ids),
                "plans_total_after_cohort_mode": len(cohort_scope_plan_ids),
                "analysis_plans_total": len(analysis_plan_ids),
                "model_used_plans_total": len(model_plan_ids),
                "control_plans_total": len(control_plan_ids),
            },
            "runtime_observability": {
                "decision_counts": {
                    "model_used": int(decision_counts.get("model_used", 0)),
                    "fallback": int(decision_counts.get("fallback", 0)),
                    "disabled": int(decision_counts.get("disabled", 0)),
                    "missing_ml_meta": int(decision_counts.get("missing_ml_meta", 0)),
                },
                "fallback_reasons": {
                    str(k): int(v)
                    for k, v in sorted(fallback_reason_counts.items(), key=lambda kv: (-kv[1], kv[0]))
                },
                "disabled_reasons": {
                    str(k): int(v)
                    for k, v in sorted(disabled_reason_counts.items(), key=lambda kv: (-kv[1], kv[0]))
                },
                "mode_distribution": {
                    str(k): int(v)
                    for k, v in sorted(mode_counts.items(), key=lambda kv: (-kv[1], kv[0]))
                },
            },
            "category_summary": category_summary_rows,
            "slice_rows": slice_rows,
            "slice_breakdowns": by_category_and_type,
            "worst_offenders": worst_offenders,
            "best_enable_candidates": best_enable_candidates,
            "recommendations": recommendations,
            "executive_summary": executive_summary,
            "policy_simulation": policy_simulation,
            "partial_candidate_plan_counts": {
                str(cat): int(len(ids))
                for cat, ids in sorted(partial_candidate_plan_ids.items(), key=lambda kv: kv[0])
            },
            "unattributed": {
                "fresh_mode_excluded_missing_ml_meta_plans": int(fresh_excluded_missing),
                "cohort_scope_excluded_non_selected_plans": int(excluded_non_selected),
                "roadmap_assignments_total": int(roadmap_assignment_total),
                "roadmap_assignments_attributed_to_cohorts": int(roadmap_assignment_attributed),
                "roadmap_assignments_unattributed": int(roadmap_assignment_unattributed),
                "roadmap_assignments_out_of_scope_plan": int(roadmap_assignment_out_of_scope),
                "roadmap_assignments_excluded_non_cohort": int(roadmap_assignment_excluded_non_cohort),
                "roadmap_offer_events_unattributed_exposed": int(offer_unattributed_event_counts.get("offer_exposed", 0)),
                "roadmap_offer_events_unattributed_clicked": int(offer_unattributed_event_counts.get("offer_clicked", 0)),
                "roadmap_offer_events_unattributed_redeemed": int(offer_unattributed_event_counts.get("offer_redeemed", 0)),
                **{str(k): int(v) for k, v in sorted(unattributed.items(), key=lambda kv: kv[0])},
            },
            "notes": [
                "Read-only diagnostics: no DB writes, no runtime logic changes.",
                "Conservative attribution only: explicit roadmap plan links first; fallback attribution requires reliable uniqueness.",
                "Ambiguous assignments/events are reported under unattributed buckets and never forced into cohorts.",
                "Default cohort-mode=fresh excludes missing_ml_meta from active model vs control comparison.",
                "Policy simulation is offline what-if analysis over observed plans; runtime behavior is unchanged.",
            ],
        }

        markdown = _build_markdown(payload)
        json_text = json.dumps(payload, ensure_ascii=False, indent=2)

        out_stem = _resolve_out_stem(out=out_raw, days=days)
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

        for p in wrote_paths:
            self.stdout.write(f"[report_roadmap_ml_diagnostics] wrote: {p}")
