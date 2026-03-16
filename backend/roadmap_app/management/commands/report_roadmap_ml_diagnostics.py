from __future__ import annotations

import json
import hashlib
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
from roadmap_app.ml_next_step import (
    blend_prediction_rows,
    nextstep_model_artifact_summary,
    predict_next_product_types_for_model_path,
    v4_category_staged_rollout_status,
)
from roadmap_app.ml_planner import planner_model_artifact_summary
from roadmap_app.models import RoadmapEvent, RoadmapPlan, RoadmapStep
from transactions.models import Transaction


FORMAT_CHOICES = ["md", "json", "both"]
COHORT_MODE_CHOICES = ["fresh", "all"]
CONTROL_CHOICES = ["non_model", "fallback", "disabled"]
NEXTSTEP_CANDIDATE_COMPARE_COHORT_CHOICES = ["model_used", "analysis"]
VALID_CATEGORIES = {"skincare", "haircare", "makeup", "fragrance"}
VALID_DECISIONS = {"model_used", "fallback", "disabled"}
DEFAULT_CATEGORIES = ["skincare", "makeup"]


def _safe_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _safe_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _normalize_product_ids(value: Any) -> list[int]:
    out: list[int] = []
    for item in _safe_list(value):
        try:
            out.append(int(item))
        except Exception:
            continue
    return out


def _normalize_runtime_policies(value: Any) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for item in _safe_list(value):
        policy = str(item or "").strip()
        if not policy or policy in seen:
            continue
        seen.add(policy)
        out.append(policy)
    out.sort()
    return out


def _normalized_token_set(value: Any) -> set[str]:
    out: set[str] = set()
    if isinstance(value, str):
        values = value.split(",")
    elif isinstance(value, (list, tuple, set)):
        values = list(value)
    else:
        values = []
    for raw in values:
        token = str(raw or "").strip().lower()
        if token:
            out.add(token)
    return out


def _normalized_int_set(value: Any) -> set[int]:
    out: set[int] = set()
    if isinstance(value, str):
        values = value.split(",")
    elif isinstance(value, (list, tuple, set)):
        values = list(value)
    else:
        values = []
    for raw in values:
        parsed = _to_int(raw)
        if parsed is None or int(parsed) <= 0:
            continue
        out.add(int(parsed))
    return out


def _current_partial_candidate_model_path(category: str) -> str:
    category_norm = str(category or "").strip().lower()
    if category_norm:
        category_setting = f"ROADMAP_NEXTSTEP_V4_PARTIAL_{category_norm.upper()}_MODEL_PATH"
        category_value = str(getattr(settings, category_setting, "") or "").strip()
        if category_value:
            return category_value
    return str(getattr(settings, "ROADMAP_NEXTSTEP_V4_PARTIAL_MODEL_PATH", "") or "").strip()


def _current_partial_rollout_bucket(*, user_id: int, category: str, salt: str) -> int:
    raw = f"{str(salt)}:{str(category)}:{int(user_id)}".encode("utf-8")
    digest = hashlib.sha1(raw).hexdigest()[:8]
    return int(digest, 16) % 100


def _current_partial_rollout_percent(category: str) -> int:
    category_norm = str(category or "").strip().lower()
    category_percent = None
    if category_norm:
        setting_name = f"ROADMAP_NEXTSTEP_V4_PARTIAL_{category_norm.upper()}_PERCENT"
        category_percent = getattr(settings, setting_name, None)
    raw_percent = (
        category_percent
        if category_percent not in (None, "")
        else getattr(settings, "ROADMAP_NEXTSTEP_V4_PARTIAL_PERCENT", 0)
    )
    try:
        percent = int(raw_percent or 0)
    except Exception:
        percent = 0
    return max(0, min(100, percent))


def _projected_partial_slot_for_plan(
    *,
    user_id: int,
    category: str,
    planned_target_product_type: str,
    planned_target_step_index: int,
    refresh_caller: str = "",
) -> str | None:
    category_norm = str(category or "").strip().lower()
    if not category_norm:
        return None
    enabled_categories = _normalized_token_set(
        getattr(settings, "ROADMAP_NEXTSTEP_V4_PARTIAL_ENABLED_CATEGORIES", [])
    )
    if category_norm not in enabled_categories:
        return None

    percent = _current_partial_rollout_percent(category_norm)
    if percent <= 0:
        return None

    product_setting = f"ROADMAP_NEXTSTEP_V4_PARTIAL_{category_norm.upper()}_PRODUCT_TYPES"
    step_setting = f"ROADMAP_NEXTSTEP_V4_PARTIAL_{category_norm.upper()}_STEP_INDEXES"
    override_setting = (
        f"ROADMAP_NEXTSTEP_V4_PARTIAL_{category_norm.upper()}_ACTIVE_MODEL_PRODUCT_TYPES"
    )
    allow_product_types = _normalized_token_set(getattr(settings, product_setting, []))
    allow_step_indexes = _normalized_int_set(getattr(settings, step_setting, []))
    target_product_type = str(planned_target_product_type or "").strip().lower()
    target_step_index = int(planned_target_step_index or 0)

    product_match = bool(target_product_type and target_product_type in allow_product_types)
    step_match = bool(target_step_index > 0 and target_step_index in allow_step_indexes)

    if allow_product_types and allow_step_indexes:
        allowlist_passed = product_match or step_match
    elif allow_product_types:
        allowlist_passed = product_match
    elif allow_step_indexes:
        allowlist_passed = step_match
    else:
        allowlist_passed = False
    if not allowlist_passed:
        return None

    purchase_context_only_categories = _normalized_token_set(
        getattr(settings, "ROADMAP_NEXTSTEP_V4_PARTIAL_PURCHASE_CONTEXT_ONLY_CATEGORIES", [])
    )
    if (
        category_norm in purchase_context_only_categories
        and str(refresh_caller or "").strip() != "update_roadmap_from_purchase"
    ):
        return None

    salt = str(
        getattr(settings, "ROADMAP_NEXTSTEP_V4_PARTIAL_SALT", "roadmap_nextstep_v4_partial_v1")
        or "roadmap_nextstep_v4_partial_v1"
    ).strip()
    if _current_partial_rollout_bucket(user_id=user_id, category=category_norm, salt=salt) >= percent:
        return None

    active_override_types = _normalized_token_set(getattr(settings, override_setting, []))
    if target_product_type and target_product_type in active_override_types:
        return "partial_active_override"

    if _current_partial_candidate_model_path(category_norm):
        return "partial_candidate"

    return "active"


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


def _fmt_metric(value: Any, ndigits: int = 4) -> str:
    try:
        return f"{float(value):.{ndigits}f}"
    except Exception:
        return "n/a"


def _event_key(created_at: Any, event_id: int | None) -> tuple[Any, int]:
    return created_at, int(event_id or 0)


def _artifact_rows(payload: dict[str, Any]) -> list[list[Any]]:
    artifacts = _safe_dict(payload.get("artifacts"))
    rows: list[list[Any]] = []
    for family in ["nextstep", "planner"]:
        family_block = _safe_dict(artifacts.get(family))
        for slot in ["active", "candidate"]:
            artifact = _safe_dict(family_block.get(slot))
            if not artifact:
                continue
            metrics_test = _safe_dict(artifact.get("metrics_test"))
            guard = _safe_dict(
                artifact.get("runtime_guard") if family == "nextstep" else artifact.get("planner_guard")
            )
            rows.append(
                [
                    family,
                    slot,
                    artifact.get("model_version") or "n/a",
                    artifact.get("selected_feature_set") or "n/a",
                    "yes" if bool(artifact.get("exists")) else "no",
                    _fmt_metric(metrics_test.get("ndcg_at_5")),
                    _fmt_metric(metrics_test.get("recall_at_1")),
                    "yes" if bool(guard.get("passed")) else "no",
                    artifact.get("model_path") or "",
                ]
            )
    return rows


def _top_prediction_token(rows: Any) -> str | None:
    for row in _safe_list(rows):
        item = _safe_dict(row)
        token = str(item.get("product_type") or item.get("candidate_type") or "").strip().lower()
        if token:
            return token
    return None


def _challenger_outcome_summary(counter: Counter[str], *, challenger_label: str) -> dict[str, Any]:
    eligible = int(counter.get("eligible_plans", 0))
    active_hits = int(counter.get("active_hits", 0))
    challenger_hits = int(counter.get(f"{challenger_label}_hits", 0))
    both_hits = int(counter.get("both_hits", 0))
    active_only_hits = int(counter.get("active_only_hits", 0))
    challenger_only_hits = int(counter.get(f"{challenger_label}_only_hits", 0))
    neither_hits = int(counter.get("neither_hits", 0))
    active_rate = _rate(active_hits, eligible)
    challenger_rate = _rate(challenger_hits, eligible)
    delta = None
    if active_rate is not None and challenger_rate is not None:
        delta = challenger_rate - active_rate
    return {
        "eligible_plans": eligible,
        "active_hits": active_hits,
        f"{challenger_label}_hits": challenger_hits,
        "both_hits": both_hits,
        "active_only_hits": active_only_hits,
        f"{challenger_label}_only_hits": challenger_only_hits,
        "neither_hits": neither_hits,
        "active_hit_rate": _round_or_none(active_rate),
        f"{challenger_label}_hit_rate": _round_or_none(challenger_rate),
        f"{challenger_label}_delta_vs_active": _round_or_none(delta),
    }


def _top_counter_rows(counter: Counter[str], *, limit: int = 3) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for token, count in sorted(counter.items(), key=lambda kv: (-kv[1], kv[0])):
        label = str(token or "").strip().lower()
        if not label:
            continue
        rows.append(
            {
                "product_type": label,
                "plans": int(count),
            }
        )
        if len(rows) >= limit:
            break
    return rows


def _challenger_outcome_detail(
    counter: Counter[str],
    *,
    challenger_label: str,
    actual_outcomes: Counter[str] | None = None,
) -> dict[str, Any]:
    summary = _challenger_outcome_summary(counter, challenger_label=challenger_label)
    top_actual_outcomes = _top_counter_rows(actual_outcomes or Counter())
    summary["top_actual_outcomes"] = top_actual_outcomes
    summary["top_actual_outcomes_text"] = ", ".join(
        f"{row['product_type']}:{row['plans']}" for row in top_actual_outcomes
    )
    if top_actual_outcomes:
        summary["dominant_actual_outcome"] = str(top_actual_outcomes[0]["product_type"])
        summary["dominant_actual_outcome_plans"] = int(top_actual_outcomes[0]["plans"])
    else:
        summary["dominant_actual_outcome"] = ""
        summary["dominant_actual_outcome_plans"] = 0
    return summary


def _shadow_outcome_summary(counter: Counter[str]) -> dict[str, Any]:
    return _challenger_outcome_summary(counter, challenger_label="shadow")


def _shadow_outcome_detail(
    counter: Counter[str],
    *,
    actual_outcomes: Counter[str] | None = None,
) -> dict[str, Any]:
    return _challenger_outcome_detail(
        counter,
        challenger_label="shadow",
        actual_outcomes=actual_outcomes,
    )


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
    served_slots = _safe_dict(runtime.get("served_model_slots"))
    lines.append("### served model slots")
    lines.append(
        _md_table(
            ["model_slot", "model_used_plans"],
            [[k, v] for k, v in sorted(_safe_dict(served_slots.get("slot_counts")).items(), key=lambda kv: (-kv[1], kv[0]))],
        )
    )
    lines.append("### served model versions")
    lines.append(
        _md_table(
            ["model_version", "model_used_plans"],
            [
                [k, v]
                for k, v in sorted(_safe_dict(served_slots.get("model_version_counts")).items(), key=lambda kv: (-kv[1], kv[0]))
            ],
        )
    )
    lines.append("### served model slots by category")
    lines.append(
        _md_table(
            ["category", "slot_counts"],
            [
                [
                    cat,
                    ", ".join(f"{slot}:{count}" for slot, count in sorted(_safe_dict(row).items(), key=lambda kv: (-kv[1], kv[0])))
                    or "-",
                ]
                for cat, row in sorted(_safe_dict(served_slots.get("by_category")).items())
            ],
        )
    )
    lines.append("### model-used outcome by served slot")
    lines.append(
        _md_table(
            [
                "category",
                "model_slot",
                "plans",
                "step_exposed",
                "step_completed",
                "step_completion_rate_pct",
                "offer_redeem_rate_pct",
            ],
            [
                [
                    row.get("category"),
                    row.get("model_slot"),
                    row.get("plans", 0),
                    row.get("step_exposed", 0),
                    row.get("step_completed", 0),
                    f"{_safe_number(row.get('step_completion_rate')) * 100.0:.2f}",
                    f"{_safe_number(row.get('offer_redeem_rate')) * 100.0:.2f}",
                ]
                for row in _safe_list(served_slots.get("model_used_outcomes_by_slot"))[:15]
            ],
        )
    )
    lines.append("### model-used outcome by served slot and planned target")
    lines.append(
        _md_table(
            [
                "category",
                "model_slot",
                "planned_target_product_type",
                "plans",
                "step_exposed",
                "step_completed",
                "step_completion_rate_pct",
            ],
            [
                [
                    row.get("category"),
                    row.get("model_slot"),
                    row.get("planned_target_product_type"),
                    row.get("plans", 0),
                    row.get("step_exposed", 0),
                    row.get("step_completed", 0),
                    f"{_safe_number(row.get('step_completion_rate')) * 100.0:.2f}",
                ]
                for row in _safe_list(served_slots.get("model_used_outcomes_by_slot_and_planned_target"))[:15]
            ],
        )
    )
    projected_slots = _safe_dict(runtime.get("projected_partial_slots"))
    lines.append("### projected partial slots")
    lines.append(
        _md_table(
            ["model_slot", "projected_plans"],
            [[k, v] for k, v in sorted(_safe_dict(projected_slots.get("slot_counts")).items(), key=lambda kv: (-kv[1], kv[0]))],
        )
    )
    lines.append("### projected partial slots by category")
    lines.append(
        _md_table(
            ["category", "slot_counts"],
            [
                [
                    cat,
                    ", ".join(
                        f"{slot}:{count}" for slot, count in sorted(_safe_dict(row).items(), key=lambda kv: (-kv[1], kv[0]))
                    )
                    or "-",
                ]
                for cat, row in sorted(_safe_dict(projected_slots.get("by_category")).items())
            ],
        )
    )
    lines.append("### projected partial slots by planned target")
    lines.append(
        _md_table(
            ["category", "model_slot", "planned_target_product_type", "plans"],
            [
                [
                    row.get("category"),
                    row.get("model_slot"),
                    row.get("planned_target_product_type"),
                    row.get("plans", 0),
                ]
                for row in _safe_list(projected_slots.get("by_slot_and_planned_target"))[:20]
            ],
        )
    )
    lines.append("### projected outcome by partial slot")
    lines.append(
        _md_table(
            [
                "category",
                "model_slot",
                "plans",
                "step_exposed",
                "step_completed",
                "step_completion_rate_pct",
                "offer_redeem_rate_pct",
            ],
            [
                [
                    row.get("category"),
                    row.get("model_slot"),
                    row.get("plans", 0),
                    row.get("step_exposed", 0),
                    row.get("step_completed", 0),
                    f"{_safe_number(row.get('step_completion_rate')) * 100.0:.2f}",
                    f"{_safe_number(row.get('offer_redeem_rate')) * 100.0:.2f}",
                ]
                for row in _safe_list(projected_slots.get("projected_outcomes_by_slot"))[:15]
            ],
        )
    )
    lines.append("### projected outcome by partial slot and planned target")
    lines.append(
        _md_table(
            [
                "category",
                "model_slot",
                "planned_target_product_type",
                "plans",
                "step_exposed",
                "step_completed",
                "step_completion_rate_pct",
            ],
            [
                [
                    row.get("category"),
                    row.get("model_slot"),
                    row.get("planned_target_product_type"),
                    row.get("plans", 0),
                    row.get("step_exposed", 0),
                    row.get("step_completed", 0),
                    f"{_safe_number(row.get('step_completion_rate')) * 100.0:.2f}",
                ]
                for row in _safe_list(projected_slots.get("projected_outcomes_by_slot_and_planned_target"))[:20]
            ],
        )
    )
    runtime_policies = _safe_dict(runtime.get("runtime_policies"))
    lines.append("### runtime policies")
    lines.append(
        _md_table(
            ["policy", "all_plans", "model_used_plans"],
            [
                [
                    policy,
                    _safe_dict(runtime_policies.get("policy_counts_all_plans")).get(policy, 0),
                    count,
                ]
                for policy, count in sorted(
                    _safe_dict(runtime_policies.get("policy_counts")).items(),
                    key=lambda kv: (-kv[1], kv[0]),
                )
            ],
        )
    )
    lines.append("### runtime policy coverage")
    lines.append(
        _md_table(
            ["metric", "value"],
            [
                ["all_plans_with_any_policy", runtime_policies.get("all_plans_with_any_policy", 0)],
                ["model_used_plans_with_any_policy", runtime_policies.get("plans_with_any_policy", 0)],
            ],
        )
    )
    lines.append("### runtime policy meta sources")
    lines.append(
        _md_table(
            ["source", "all_plans_with_any_policy", "model_used_plans_with_any_policy"],
            [
                [
                    source,
                    _safe_dict(runtime_policies.get("source_counts_all_plans")).get(source, 0),
                    count,
                ]
                for source, count in sorted(
                    _safe_dict(runtime_policies.get("source_counts")).items(),
                    key=lambda kv: (-kv[1], kv[0]),
                )
            ],
        )
    )
    lines.append("### runtime policy coverage by decision")
    lines.append(
        _md_table(
            ["decision", "plans_with_any_policy"],
            [
                [decision, count]
                for decision, count in sorted(
                    _safe_dict(runtime_policies.get("all_plans_with_any_policy_by_decision")).items(),
                    key=lambda kv: (-kv[1], kv[0]),
                )
            ],
        )
    )
    lines.append("### runtime policies by category")
    lines.append(
        _md_table(
            ["category", "policy_counts"],
            [
                [
                    cat,
                    ", ".join(
                        f"{policy}:{count}"
                        for policy, count in sorted(_safe_dict(row).items(), key=lambda kv: (-kv[1], kv[0]))
                    )
                    or "-",
                ]
                for cat, row in sorted(_safe_dict(runtime_policies.get("by_category")).items())
            ],
        )
    )
    lines.append("### model-used outcome by runtime policy")
    lines.append(
        _md_table(
            [
                "category",
                "runtime_policy",
                "plans",
                "step_exposed",
                "step_completed",
                "step_completion_rate_pct",
                "offer_redeem_rate_pct",
            ],
            [
                [
                    row.get("category"),
                    row.get("runtime_policy"),
                    row.get("plans", 0),
                    row.get("step_exposed", 0),
                    row.get("step_completed", 0),
                    f"{_safe_number(row.get('step_completion_rate')) * 100.0:.2f}",
                    f"{_safe_number(row.get('offer_redeem_rate')) * 100.0:.2f}",
                ]
                for row in _safe_list(runtime_policies.get("model_used_outcomes_by_policy"))[:15]
            ],
        )
    )
    shadow = _safe_dict(runtime.get("shadow"))
    shadow_top1 = _safe_dict(shadow.get("top1_comparison"))
    lines.append("### shadow comparison")
    lines.append(
        _md_table(
            ["metric", "value"],
            [
                ["plans_with_shadow_meta", shadow.get("plans_with_shadow_meta", 0)],
                ["shadow_enabled_plans", shadow.get("shadow_enabled_plans", 0)],
                ["eligible_top1_comparisons", shadow_top1.get("eligible_plans", 0)],
                ["same_top1_plans", shadow_top1.get("same_top1_plans", 0)],
                ["different_top1_plans", shadow_top1.get("different_top1_plans", 0)],
                ["agreement_rate_pct", f"{_safe_number(shadow_top1.get('agreement_rate')) * 100.0:.2f}"],
            ],
        )
    )
    lines.append("### shadow reasons")
    lines.append(
        _md_table(
            ["reason", "count"],
            [[k, v] for k, v in sorted(_safe_dict(shadow.get("reason_counts")).items(), key=lambda kv: (-kv[1], kv[0]))],
        )
    )
    lines.append("### shadow model versions")
    lines.append(
        _md_table(
            ["model_version", "count"],
            [[k, v] for k, v in sorted(_safe_dict(shadow.get("model_version_counts")).items(), key=lambda kv: (-kv[1], kv[0]))],
        )
    )
    lines.append("### shadow by category")
    lines.append(
        _md_table(
            [
                "category",
                "plans_with_shadow_meta",
                "shadow_enabled_plans",
                "eligible_top1_comparisons",
                "same_top1_plans",
                "different_top1_plans",
                "agreement_rate_pct",
            ],
            [
                [
                    cat,
                    row.get("plans_with_shadow_meta", 0),
                    row.get("shadow_enabled_plans", 0),
                    row.get("eligible_plans", 0),
                    row.get("same_top1_plans", 0),
                    row.get("different_top1_plans", 0),
                    f"{_safe_number(row.get('agreement_rate')) * 100.0:.2f}",
                ]
                for cat, row in sorted(_safe_dict(shadow.get("by_category")).items())
            ],
        )
    )
    lines.append("### top shadow swaps")
    lines.append(
        _md_table(
            ["category", "active_top1", "shadow_top1", "plans"],
            [
                [
                    row.get("category"),
                    row.get("active_top1"),
                    row.get("shadow_top1"),
                    row.get("plans"),
                ]
                for row in _safe_list(shadow.get("top_swaps"))[:15]
            ],
        )
    )
    shadow_outcome = _safe_dict(shadow.get("outcome_comparison"))
    lines.append("### shadow vs outcome")
    lines.append(
        _md_table(
            ["metric", "value"],
            [
                ["eligible_plans", shadow_outcome.get("eligible_plans", 0)],
                ["active_hits", shadow_outcome.get("active_hits", 0)],
                ["shadow_hits", shadow_outcome.get("shadow_hits", 0)],
                ["both_hits", shadow_outcome.get("both_hits", 0)],
                ["active_only_hits", shadow_outcome.get("active_only_hits", 0)],
                ["shadow_only_hits", shadow_outcome.get("shadow_only_hits", 0)],
                ["neither_hits", shadow_outcome.get("neither_hits", 0)],
                ["active_hit_rate_pct", f"{_safe_number(shadow_outcome.get('active_hit_rate')) * 100.0:.2f}"],
                ["shadow_hit_rate_pct", f"{_safe_number(shadow_outcome.get('shadow_hit_rate')) * 100.0:.2f}"],
                ["shadow_delta_vs_active_pp", f"{_safe_number(shadow_outcome.get('shadow_delta_vs_active')) * 100.0:.2f}"],
            ],
        )
    )
    lines.append("### shadow vs outcome by category")
    lines.append(
        _md_table(
            [
                "category",
                "eligible_plans",
                "active_hits",
                "shadow_hits",
                "active_only_hits",
                "shadow_only_hits",
                "active_hit_rate_pct",
                "shadow_hit_rate_pct",
                "shadow_delta_vs_active_pp",
            ],
            [
                [
                    cat,
                    row.get("eligible_plans", 0),
                    row.get("active_hits", 0),
                    row.get("shadow_hits", 0),
                    row.get("active_only_hits", 0),
                    row.get("shadow_only_hits", 0),
                    f"{_safe_number(row.get('active_hit_rate')) * 100.0:.2f}",
                    f"{_safe_number(row.get('shadow_hit_rate')) * 100.0:.2f}",
                    f"{_safe_number(row.get('shadow_delta_vs_active')) * 100.0:.2f}",
                ]
                for cat, row in sorted(_safe_dict(shadow_outcome.get("by_category")).items())
            ],
        )
    )
    lines.append("### shadow outcome by predicted top1")
    lines.append(
        _md_table(
            [
                "category",
                "shadow_top1",
                "eligible_plans",
                "dominant_actual_outcome",
                "top_actual_outcomes",
                "active_hit_rate_pct",
                "shadow_hit_rate_pct",
                "shadow_delta_vs_active_pp",
            ],
            [
                [
                    row.get("category"),
                    row.get("shadow_top1"),
                    row.get("eligible_plans", 0),
                    row.get("dominant_actual_outcome") or "-",
                    row.get("top_actual_outcomes_text") or "-",
                    f"{_safe_number(row.get('active_hit_rate')) * 100.0:.2f}",
                    f"{_safe_number(row.get('shadow_hit_rate')) * 100.0:.2f}",
                    f"{_safe_number(row.get('shadow_delta_vs_active')) * 100.0:.2f}",
                ]
                for row in _safe_list(shadow.get("outcome_by_shadow_top1"))[:15]
            ],
        )
    )
    lines.append("### shadow outcome by swap pair")
    lines.append(
        _md_table(
            [
                "category",
                "active_top1",
                "shadow_top1",
                "eligible_plans",
                "dominant_actual_outcome",
                "top_actual_outcomes",
                "active_only_hits",
                "shadow_only_hits",
                "active_hit_rate_pct",
                "shadow_hit_rate_pct",
                "shadow_delta_vs_active_pp",
            ],
            [
                [
                    row.get("category"),
                    row.get("active_top1"),
                    row.get("shadow_top1"),
                    row.get("eligible_plans", 0),
                    row.get("dominant_actual_outcome") or "-",
                    row.get("top_actual_outcomes_text") or "-",
                    row.get("active_only_hits", 0),
                    row.get("shadow_only_hits", 0),
                    f"{_safe_number(row.get('active_hit_rate')) * 100.0:.2f}",
                    f"{_safe_number(row.get('shadow_hit_rate')) * 100.0:.2f}",
                    f"{_safe_number(row.get('shadow_delta_vs_active')) * 100.0:.2f}",
                ]
                for row in _safe_list(shadow.get("outcome_by_swap_pair"))[:15]
            ],
        )
    )
    candidate_compare = _safe_dict(runtime.get("candidate_path_compare"))
    if candidate_compare:
        candidate_top1 = _safe_dict(candidate_compare.get("top1_comparison"))
        lines.append("### candidate path compare")
        lines.append(
            _md_table(
                ["metric", "value"],
                [
                    ["model_version", candidate_compare.get("model_version") or "-"],
                    ["selected_feature_set", candidate_compare.get("selected_feature_set") or "-"],
                    ["plans_scanned", candidate_compare.get("plans_scanned", 0)],
                    ["predicted_plans", candidate_compare.get("predicted_plans", 0)],
                    ["eligible_top1_comparisons", candidate_top1.get("eligible_plans", 0)],
                    ["same_top1_plans", candidate_top1.get("same_top1_plans", 0)],
                    ["different_top1_plans", candidate_top1.get("different_top1_plans", 0)],
                    ["agreement_rate_pct", f"{_safe_number(candidate_top1.get('agreement_rate')) * 100.0:.2f}"],
                ],
            )
        )
        lines.append("### candidate path skipped reasons")
        lines.append(
            _md_table(
                ["reason", "count"],
                [
                    [k, v]
                    for k, v in sorted(_safe_dict(candidate_compare.get("skipped_counts")).items(), key=lambda kv: (-kv[1], kv[0]))
                ],
            )
        )
        lines.append("### candidate path by category")
        lines.append(
            _md_table(
                [
                    "category",
                    "eligible_top1_comparisons",
                    "same_top1_plans",
                    "different_top1_plans",
                    "agreement_rate_pct",
                ],
                [
                    [
                        cat,
                        row.get("eligible_plans", 0),
                        row.get("same_top1_plans", 0),
                        row.get("different_top1_plans", 0),
                        f"{_safe_number(row.get('agreement_rate')) * 100.0:.2f}",
                    ]
                    for cat, row in sorted(_safe_dict(candidate_compare.get("by_category")).items())
                ],
            )
        )
        lines.append("### top candidate-path swaps")
        lines.append(
            _md_table(
                ["category", "active_top1", "candidate_top1", "plans"],
                [
                    [
                        row.get("category"),
                        row.get("active_top1"),
                        row.get("candidate_top1"),
                        row.get("plans"),
                    ]
                    for row in _safe_list(candidate_compare.get("top_swaps"))[:15]
                ],
            )
        )
        candidate_outcome = _safe_dict(candidate_compare.get("outcome_comparison"))
        lines.append("### candidate path vs outcome")
        lines.append(
            _md_table(
                ["metric", "value"],
                [
                    ["eligible_plans", candidate_outcome.get("eligible_plans", 0)],
                    ["active_hits", candidate_outcome.get("active_hits", 0)],
                    ["candidate_hits", candidate_outcome.get("candidate_hits", 0)],
                    ["both_hits", candidate_outcome.get("both_hits", 0)],
                    ["active_only_hits", candidate_outcome.get("active_only_hits", 0)],
                    ["candidate_only_hits", candidate_outcome.get("candidate_only_hits", 0)],
                    ["neither_hits", candidate_outcome.get("neither_hits", 0)],
                    ["active_hit_rate_pct", f"{_safe_number(candidate_outcome.get('active_hit_rate')) * 100.0:.2f}"],
                    ["candidate_hit_rate_pct", f"{_safe_number(candidate_outcome.get('candidate_hit_rate')) * 100.0:.2f}"],
                    [
                        "candidate_delta_vs_active_pp",
                        f"{_safe_number(candidate_outcome.get('candidate_delta_vs_active')) * 100.0:.2f}",
                    ],
                ],
            )
        )
        lines.append("### candidate path vs outcome by category")
        lines.append(
            _md_table(
                [
                    "category",
                    "eligible_plans",
                    "active_hits",
                    "candidate_hits",
                    "active_only_hits",
                    "candidate_only_hits",
                    "active_hit_rate_pct",
                    "candidate_hit_rate_pct",
                    "candidate_delta_vs_active_pp",
                ],
                [
                    [
                        cat,
                        row.get("eligible_plans", 0),
                        row.get("active_hits", 0),
                        row.get("candidate_hits", 0),
                        row.get("active_only_hits", 0),
                        row.get("candidate_only_hits", 0),
                        f"{_safe_number(row.get('active_hit_rate')) * 100.0:.2f}",
                        f"{_safe_number(row.get('candidate_hit_rate')) * 100.0:.2f}",
                        f"{_safe_number(row.get('candidate_delta_vs_active')) * 100.0:.2f}",
                    ]
                    for cat, row in sorted(_safe_dict(candidate_outcome.get("by_category")).items())
                ],
            )
        )
        lines.append("### candidate path outcome by predicted top1")
        lines.append(
            _md_table(
                [
                    "category",
                    "candidate_top1",
                    "eligible_plans",
                    "dominant_actual_outcome",
                    "top_actual_outcomes",
                    "active_hit_rate_pct",
                    "candidate_hit_rate_pct",
                    "candidate_delta_vs_active_pp",
                ],
                [
                    [
                        row.get("category"),
                        row.get("candidate_top1"),
                        row.get("eligible_plans", 0),
                        row.get("dominant_actual_outcome") or "-",
                        row.get("top_actual_outcomes_text") or "-",
                        f"{_safe_number(row.get('active_hit_rate')) * 100.0:.2f}",
                        f"{_safe_number(row.get('candidate_hit_rate')) * 100.0:.2f}",
                        f"{_safe_number(row.get('candidate_delta_vs_active')) * 100.0:.2f}",
                    ]
                    for row in _safe_list(candidate_compare.get("outcome_by_candidate_top1"))[:15]
                ],
            )
        )
        lines.append("### candidate path outcome by swap pair")
        lines.append(
            _md_table(
                [
                    "category",
                    "active_top1",
                    "candidate_top1",
                    "eligible_plans",
                    "dominant_actual_outcome",
                    "top_actual_outcomes",
                    "active_only_hits",
                    "candidate_only_hits",
                    "active_hit_rate_pct",
                    "candidate_hit_rate_pct",
                    "candidate_delta_vs_active_pp",
                ],
                [
                    [
                        row.get("category"),
                        row.get("active_top1"),
                        row.get("candidate_top1"),
                        row.get("eligible_plans", 0),
                        row.get("dominant_actual_outcome") or "-",
                        row.get("top_actual_outcomes_text") or "-",
                        row.get("active_only_hits", 0),
                        row.get("candidate_only_hits", 0),
                        f"{_safe_number(row.get('active_hit_rate')) * 100.0:.2f}",
                        f"{_safe_number(row.get('candidate_hit_rate')) * 100.0:.2f}",
                        f"{_safe_number(row.get('candidate_delta_vs_active')) * 100.0:.2f}",
                    ]
                    for row in _safe_list(candidate_compare.get("outcome_by_swap_pair"))[:15]
                ],
            )
        )
    lines.append("")

    lines.append("## 8) Model artifacts")
    lines.append(
        _md_table(
            [
                "family",
                "slot",
                "model_version",
                "feature_set",
                "exists",
                "ndcg@5",
                "recall@1",
                "guard_passed",
                "model_path",
            ],
            _artifact_rows(payload),
        )
    )
    lines.append("")

    lines.append("## 9) Unattributed / excluded")
    lines.append(
        _md_table(
            ["bucket", "count"],
            [[k, v] for k, v in sorted(unattributed.items(), key=lambda kv: kv[0])],
        )
    )
    lines.append("")

    lines.append("## 10) Notes")
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
        parser.add_argument("--refresh-caller", type=str, default="all")
        parser.add_argument("--planned-target-product-types", type=str, default="all")
        parser.add_argument("--skip-event-metrics", action="store_true", default=False)
        parser.add_argument("--format", type=str, default="both", choices=FORMAT_CHOICES)
        parser.add_argument("--out", type=str, default=None)
        parser.add_argument("--min-sample", type=int, default=30)
        parser.add_argument("--cohort-mode", type=str, default="fresh", choices=COHORT_MODE_CHOICES)
        parser.add_argument("--control", type=str, default="non_model", choices=CONTROL_CHOICES)
        parser.add_argument("--nextstep-candidate-model-path", type=str, default=None)
        parser.add_argument("--nextstep-candidate-teacher-model-path", type=str, default=None)
        parser.add_argument("--nextstep-candidate-teacher-weight", type=float, default=0.25)
        parser.add_argument(
            "--nextstep-candidate-compare-cohort",
            type=str,
            default="model_used",
            choices=NEXTSTEP_CANDIDATE_COMPARE_COHORT_CHOICES,
        )
        parser.add_argument("--planner-candidate-model-path", type=str, default=None)

    def handle(self, *args, **options):
        days = int(options["days"] or 7)
        if days <= 0:
            raise CommandError("--days must be > 0")

        include_ga = bool(options["include_ga"])
        categories = _parse_categories(options.get("categories"))
        refresh_caller_filter = str(options.get("refresh_caller") or "all").strip().lower()
        planned_target_filter = _normalized_token_set(options.get("planned_target_product_types"))
        if "all" in planned_target_filter:
            planned_target_filter = set()
        skip_event_metrics = bool(options.get("skip_event_metrics"))
        out_format = str(options["format"] or "both").strip().lower()
        out_raw = options.get("out")
        min_sample = int(options["min_sample"] or 30)
        if min_sample <= 0:
            raise CommandError("--min-sample must be > 0")
        cohort_mode = str(options["cohort_mode"] or "fresh").strip().lower()
        control = str(options["control"] or "non_model").strip().lower()
        nextstep_candidate_model_path = str(options.get("nextstep_candidate_model_path") or "").strip()
        nextstep_candidate_teacher_model_path = str(
            options.get("nextstep_candidate_teacher_model_path") or ""
        ).strip()
        nextstep_candidate_teacher_weight = max(
            0.0,
            float(options.get("nextstep_candidate_teacher_weight") or 0.0),
        )
        nextstep_candidate_compare_cohort = str(
            options.get("nextstep_candidate_compare_cohort") or "model_used"
        ).strip().lower()
        planner_candidate_model_path = str(options.get("planner_candidate_model_path") or "").strip()

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
        plan_model_slot: dict[int, str] = {}
        plan_planned_target_product_type: dict[int, str] = {}
        plan_planned_target_step_index: dict[int, int] = {}
        plan_refresh_caller: dict[int, str] = {}
        plan_context_product_ids: dict[int, list[int]] = {}
        plan_runtime_policies: dict[int, list[str]] = {}
        plan_runtime_policy_meta_source: dict[int, str] = {}
        plan_model_version: dict[int, str] = {}
        cohort_by_plan: dict[int, str] = {}
        decision_counts: Counter[str] = Counter()
        fallback_reason_counts: Counter[str] = Counter()
        disabled_reason_counts: Counter[str] = Counter()
        mode_counts: Counter[str] = Counter()
        served_model_slot_counts: Counter[str] = Counter()
        served_model_version_counts: Counter[str] = Counter()
        served_model_slot_by_category: dict[str, Counter[str]] = defaultdict(Counter)
        runtime_policy_counts: Counter[str] = Counter()
        runtime_policy_source_counts: Counter[str] = Counter()
        runtime_policy_by_category: dict[str, Counter[str]] = defaultdict(Counter)
        runtime_policy_counts_all_plans: Counter[str] = Counter()
        runtime_policy_source_counts_all_plans: Counter[str] = Counter()
        runtime_policy_all_plan_ids: set[int] = set()
        runtime_policy_all_plan_counts_by_decision: Counter[str] = Counter()
        shadow_reason_counts: Counter[str] = Counter()
        shadow_model_version_counts: Counter[str] = Counter()
        shadow_meta_plan_ids: set[int] = set()
        shadow_enabled_plan_ids: set[int] = set()
        shadow_top1_eligible_plan_ids: set[int] = set()
        shadow_top1_same_plan_ids: set[int] = set()
        shadow_top1_diff_plan_ids: set[int] = set()
        shadow_by_category: dict[str, Counter[str]] = defaultdict(Counter)
        shadow_swap_counts: Counter[tuple[str, str, str]] = Counter()
        active_top1_by_plan: dict[int, str] = {}
        shadow_top1_by_plan: dict[int, str] = {}
        shadow_enabled_by_plan: dict[int, bool] = {}

        for row in plan_rows:
            pid = int(row["id"])
            cat = str(row["category"] or "")
            user_id = int(row["user_id"])
            meta = _safe_dict(row.get("meta"))
            ml = _safe_dict(meta.get("ml"))
            shadow = _safe_dict(ml.get("shadow"))
            decision = _decision_from_meta(meta)
            context_meta = _safe_dict(meta.get("context"))
            refresh_caller = str(context_meta.get("refresh_caller") or "").strip().lower()

            if refresh_caller_filter not in {"", "all"} and refresh_caller != refresh_caller_filter:
                continue

            planned_target_product_type = (
                str(ml.get("planned_target_product_type") or "").strip().lower() or "__none__"
            )
            if planned_target_filter and planned_target_product_type not in planned_target_filter:
                continue

            all_scope_plan_ids.add(pid)
            plan_category[pid] = cat
            plan_user[pid] = user_id
            plan_updated[pid] = row["updated_at"]
            plan_decision[pid] = decision
            plan_model_slot[pid] = str(ml.get("model_slot") or "active").strip().lower() or "active"
            active_top1_by_plan[pid] = str(_top_prediction_token(ml.get("predictions")) or "")
            plan_planned_target_product_type[pid] = planned_target_product_type
            plan_planned_target_step_index[pid] = int(_to_int(ml.get("planned_target_step_index")) or 0)
            plan_refresh_caller[pid] = refresh_caller
            plan_context_product_ids[pid] = _normalize_product_ids(context_meta.get("post_ctx_product_ids"))
            plan_runtime_policies[pid] = _normalize_runtime_policies(ml.get("runtime_policies"))
            plan_runtime_policy_meta_source[pid] = (
                str(ml.get("runtime_policy_meta_source") or "").strip().lower() or "__missing__"
            )
            plan_model_version[pid] = (
                str(ml.get("model_version") or "__missing_model_version__").strip()
                or "__missing_model_version__"
            )

            decision_counts[decision] += 1
            if decision == "fallback":
                fallback_reason_counts[str(ml.get("fallback_reason") or "__missing_reason__")] += 1
            elif decision == "disabled":
                disabled_reason_counts[str(ml.get("disabled_reason") or "__missing_reason__")] += 1
            mode_counts[str(ml.get("mode") or "none")] += 1
            if decision == "model_used":
                served_model_slot_counts[plan_model_slot[pid]] += 1
                served_model_version_counts[plan_model_version[pid]] += 1
                served_model_slot_by_category[cat][plan_model_slot[pid]] += 1
                runtime_policies_for_plan = plan_runtime_policies.get(pid) or []
                if runtime_policies_for_plan:
                    runtime_policy_source_counts[str(plan_runtime_policy_meta_source.get(pid) or "__missing__")] += 1
                for runtime_policy in runtime_policies_for_plan:
                    runtime_policy_counts[str(runtime_policy)] += 1
                    runtime_policy_by_category[cat][str(runtime_policy)] += 1
            runtime_policies_for_any_plan = plan_runtime_policies.get(pid) or []
            if runtime_policies_for_any_plan:
                runtime_policy_all_plan_ids.add(pid)
                runtime_policy_source_counts_all_plans[
                    str(plan_runtime_policy_meta_source.get(pid) or "__missing__")
                ] += 1
                runtime_policy_all_plan_counts_by_decision[str(decision)] += 1
            for runtime_policy in runtime_policies_for_any_plan:
                runtime_policy_counts_all_plans[str(runtime_policy)] += 1

            if shadow:
                shadow_top1_by_plan[pid] = str(_top_prediction_token(shadow.get("predictions")) or "")
                shadow_enabled_by_plan[pid] = bool(shadow.get("enabled"))
                shadow_meta_plan_ids.add(pid)
                shadow_by_category[cat]["plans_with_shadow_meta"] += 1
                shadow_reason_counts[str(shadow.get("reason") or "__missing_reason__")] += 1
                shadow_model_version_counts[str(shadow.get("model_version") or "__missing_model_version__")] += 1
                if bool(shadow.get("enabled")):
                    shadow_enabled_plan_ids.add(pid)
                    shadow_by_category[cat]["shadow_enabled_plans"] += 1
                    active_top1 = _top_prediction_token(ml.get("predictions"))
                    shadow_top1 = _top_prediction_token(shadow.get("predictions"))
                    if active_top1 and shadow_top1:
                        shadow_top1_eligible_plan_ids.add(pid)
                        shadow_by_category[cat]["eligible_plans"] += 1
                        if active_top1 == shadow_top1:
                            shadow_top1_same_plan_ids.add(pid)
                            shadow_by_category[cat]["same_top1_plans"] += 1
                        else:
                            shadow_top1_diff_plan_ids.add(pid)
                            shadow_by_category[cat]["different_top1_plans"] += 1
                            shadow_swap_counts[(cat or "__unknown__", active_top1, shadow_top1)] += 1

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
        plan_step_candidate_types: dict[int, list[str]] = defaultdict(list)
        plan_step_index_buckets: dict[int, set[str]] = defaultdict(set)
        step_meta: dict[int, dict[str, Any]] = {}
        seen_step_types_by_plan: dict[int, set[str]] = defaultdict(set)
        if analysis_plan_ids:
            step_rows = (
                RoadmapStep.objects.filter(plan_id__in=analysis_plan_ids)
                .order_by("plan_id", "step_index", "id")
                .values(
                "id",
                "plan_id",
                "step_index",
                "product_type",
                "status",
                )
            )
            for row in step_rows:
                sid = int(row["id"])
                pid = int(row["plan_id"])
                product_type = str(row["product_type"] or "").strip().lower() or "__unknown__"
                step_index = int(_to_int(row.get("step_index")) or 0)
                status = str(row.get("status") or "").strip()
                idx_bucket = _step_index_bucket(step_index)
                plan_step_product_types[pid].add(product_type)
                plan_step_index_buckets[pid].add(idx_bucket)
                if product_type and product_type not in seen_step_types_by_plan[pid]:
                    seen_step_types_by_plan[pid].add(product_type)
                    plan_step_candidate_types[pid].append(product_type)
                planned_target_product_type = str(plan_planned_target_product_type.get(pid) or "").strip().lower()
                if status in {RoadmapStep.Status.MISSING, RoadmapStep.Status.RECOMMENDED}:
                    if planned_target_product_type in {"", "__none__"}:
                        plan_planned_target_product_type[pid] = product_type
                        plan_planned_target_step_index[pid] = step_index
                    elif plan_planned_target_step_index.get(pid, 0) <= 0 and planned_target_product_type == product_type:
                        plan_planned_target_step_index[pid] = step_index
                elif plan_planned_target_step_index.get(pid, 0) <= 0 and planned_target_product_type == product_type:
                    plan_planned_target_step_index[pid] = step_index
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

        latest_plan_refresh_key: dict[int, tuple[Any, int]] = {}
        completion_events_by_plan: dict[int, list[tuple[Any, int, str, str]]] = defaultdict(list)
        if analysis_plan_ids and not skip_event_metrics:
            refresh_qs = RoadmapEvent.objects.filter(
                created_at__gte=since,
                created_at__lte=now_utc,
                event_type=RoadmapEvent.Type.PLAN_REFRESHED,
                plan_id__in=analysis_plan_ids,
            )
            if not include_ga:
                refresh_qs = refresh_qs.exclude(user__username__startswith="ga_")
            for row in refresh_qs.values("plan_id", "created_at", "id"):
                pid = int(row["plan_id"])
                key = _event_key(row["created_at"], _to_int(row.get("id")))
                prev = latest_plan_refresh_key.get(pid)
                if prev is None or key > prev:
                    latest_plan_refresh_key[pid] = key

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

        if analysis_plan_ids and not skip_event_metrics:
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
            for row in interaction_qs.values("id", "step_id", "event_type", "created_at", "context", "_effective_plan_id"):
                pid = int(row["_effective_plan_id"])
                sid = _to_int(row.get("step_id"))
                metric = event_map.get(str(row.get("event_type") or ""))
                if not metric:
                    continue
                _increment_step_metric(pid=pid, sid=sid, metric_key=metric, source=None)
                if metric == "step_completed":
                    ctx = _safe_dict(row.get("context"))
                    step_info = _safe_dict(step_meta.get(int(sid or 0)))
                    product_type = str(
                        step_info.get("product_type")
                        or ctx.get("product_type")
                        or ""
                    ).strip().lower()
                    completion_events_by_plan[pid].append(
                        (
                            row.get("created_at"),
                            int(row.get("id") or 0),
                            product_type,
                            str(ctx.get("matched_by") or "").strip().lower(),
                        )
                    )
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

        offer_unattributed_event_counts: Counter[str] = Counter()
        if not skip_event_metrics:
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

        projected_partial_slot_plan_ids: dict[str, dict[str, set[int]]] = defaultdict(lambda: defaultdict(set))
        projected_partial_slot_counts: Counter[str] = Counter()
        projected_partial_slot_by_category: dict[str, Counter[str]] = defaultdict(Counter)
        projected_partial_slot_target_counts: Counter[tuple[str, str, str]] = Counter()
        for pid in sorted(analysis_plan_ids):
            cat = str(plan_category.get(pid) or "")
            projected_slot = _projected_partial_slot_for_plan(
                user_id=int(plan_user.get(pid) or 0),
                category=cat,
                planned_target_product_type=str(plan_planned_target_product_type.get(pid) or ""),
                planned_target_step_index=int(plan_planned_target_step_index.get(pid) or 0),
                refresh_caller=str(plan_refresh_caller.get(pid) or ""),
            )
            if not projected_slot or projected_slot == "active":
                continue
            planned_target = str(plan_planned_target_product_type.get(pid) or "__none__")
            projected_partial_slot_plan_ids[cat][projected_slot].add(pid)
            projected_partial_slot_counts[projected_slot] += 1
            projected_partial_slot_by_category[cat][projected_slot] += 1
            projected_partial_slot_target_counts[(cat, projected_slot, planned_target)] += 1

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
            forced_model_plan_ids: set[int] | None = None,
        ) -> dict[str, Any]:
            category_partial_allow = category_partial_allow or {}
            forced_model_plan_ids = forced_model_plan_ids or set()
            model_counts: dict[str, float] = defaultdict(float)
            control_counts: dict[str, float] = defaultdict(float)
            model_plans = 0
            control_plans = 0

            for pid in sorted(analysis_plan_ids):
                cat = str(plan_category.get(pid) or "__unknown__")
                actual = str(cohort_by_plan.get(pid) or "")
                use_model = actual == "model_used" or pid in forced_model_plan_ids
                allowed_subset = category_partial_allow.get(cat)
                if allowed_subset is not None and actual == "model_used" and pid not in forced_model_plan_ids:
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

        haircare_leavein_projected_plan_ids = {
            pid
            for pid in analysis_plan_ids
            if str(plan_category.get(pid) or "") == "haircare"
            and "haircare_leavein_rerank" in set(plan_runtime_policies.get(pid) or [])
        }
        configured_partial_mix_plan_ids = {
            pid
            for slot_map in projected_partial_slot_plan_ids.values()
            for ids in slot_map.values()
            for pid in ids
        }
        configured_partial_candidate_plan_ids = {
            pid
            for slot_map in projected_partial_slot_plan_ids.values()
            for pid in slot_map.get("partial_candidate", set())
        }
        configured_partial_active_override_plan_ids = {
            pid
            for slot_map in projected_partial_slot_plan_ids.values()
            for pid in slot_map.get("partial_active_override", set())
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
            _simulate_policy(
                policy="Policy D - haircare partial",
                category_partial_allow={"haircare": set(partial_candidate_plan_ids.get("haircare", set()))},
            ),
            _simulate_policy(
                policy="Policy E - haircare leave_in projected",
                forced_model_plan_ids=haircare_leavein_projected_plan_ids,
            ),
            _simulate_policy(
                policy="Policy F - configured partial candidate only",
                forced_model_plan_ids=configured_partial_candidate_plan_ids,
            ),
            _simulate_policy(
                policy="Policy G - configured partial active override only",
                forced_model_plan_ids=configured_partial_active_override_plan_ids,
            ),
            _simulate_policy(
                policy="Policy H - configured partial mix canary",
                forced_model_plan_ids=configured_partial_mix_plan_ids,
            ),
        ]

        served_model_slot_outcomes: dict[tuple[str, str], dict[str, Any]] = defaultdict(_new_bucket)
        served_model_slot_target_outcomes: dict[tuple[str, str, str], dict[str, Any]] = defaultdict(_new_bucket)
        projected_partial_slot_outcomes: dict[tuple[str, str], dict[str, Any]] = defaultdict(_new_bucket)
        projected_partial_slot_target_outcomes: dict[tuple[str, str, str], dict[str, Any]] = defaultdict(_new_bucket)
        runtime_policy_outcomes: dict[tuple[str, str], dict[str, Any]] = defaultdict(_new_bucket)
        runtime_policy_plan_ids: set[int] = set()
        for cat, slot_map in projected_partial_slot_plan_ids.items():
            for model_slot, plan_ids in slot_map.items():
                for pid in sorted(plan_ids):
                    uid = int(plan_user.get(pid))
                    planned_target = str(plan_planned_target_product_type.get(pid) or "__none__")
                    slot_bucket = projected_partial_slot_outcomes[(str(cat), str(model_slot))]
                    slot_bucket["plans"].add(pid)
                    slot_bucket["users"].add(uid)
                    target_bucket = projected_partial_slot_target_outcomes[(str(cat), str(model_slot), planned_target)]
                    target_bucket["plans"].add(pid)
                    target_bucket["users"].add(uid)
                    for metric_key, metric_value in (plan_metrics.get(pid) or {}).items():
                        numeric_value = float(metric_value or 0.0)
                        slot_bucket[str(metric_key)] += numeric_value
                        target_bucket[str(metric_key)] += numeric_value
        for pid in sorted(model_plan_ids):
            cat = str(plan_category.get(pid) or "__unknown__")
            uid = int(plan_user.get(pid))
            model_slot = str(plan_model_slot.get(pid) or "active")
            planned_target = str(plan_planned_target_product_type.get(pid) or "__none__")

            slot_bucket = served_model_slot_outcomes[(cat, model_slot)]
            slot_bucket["plans"].add(pid)
            slot_bucket["users"].add(uid)

            target_bucket = served_model_slot_target_outcomes[(cat, model_slot, planned_target)]
            target_bucket["plans"].add(pid)
            target_bucket["users"].add(uid)

            for metric_key, metric_value in (plan_metrics.get(pid) or {}).items():
                numeric_value = float(metric_value or 0.0)
                slot_bucket[str(metric_key)] += numeric_value
                target_bucket[str(metric_key)] += numeric_value
            runtime_policies_for_plan = plan_runtime_policies.get(pid) or []
            if runtime_policies_for_plan:
                runtime_policy_plan_ids.add(pid)
            for runtime_policy in runtime_policies_for_plan:
                policy_bucket = runtime_policy_outcomes[(cat, str(runtime_policy))]
                policy_bucket["plans"].add(pid)
                policy_bucket["users"].add(uid)
                for metric_key, metric_value in (plan_metrics.get(pid) or {}).items():
                    policy_bucket[str(metric_key)] += float(metric_value or 0.0)

        served_model_slot_outcome_rows = [
            {
                "category": str(category),
                "model_slot": str(model_slot),
                **_serialize_bucket(bucket),
            }
            for (category, model_slot), bucket in sorted(
                served_model_slot_outcomes.items(),
                key=lambda kv: (-int(len(kv[1]["plans"])), kv[0][0], kv[0][1]),
            )
        ]
        served_model_slot_target_outcome_rows = [
            {
                "category": str(category),
                "model_slot": str(model_slot),
                "planned_target_product_type": str(planned_target),
                **_serialize_bucket(bucket),
            }
            for (category, model_slot, planned_target), bucket in sorted(
                served_model_slot_target_outcomes.items(),
                key=lambda kv: (-int(len(kv[1]["plans"])), kv[0][0], kv[0][1], kv[0][2]),
            )
        ]
        projected_partial_slot_outcome_rows = [
            {
                "category": str(category),
                "model_slot": str(model_slot),
                **_serialize_bucket(bucket),
            }
            for (category, model_slot), bucket in sorted(
                projected_partial_slot_outcomes.items(),
                key=lambda kv: (-int(len(kv[1]["plans"])), kv[0][0], kv[0][1]),
            )
        ]
        projected_partial_slot_target_outcome_rows = [
            {
                "category": str(category),
                "model_slot": str(model_slot),
                "planned_target_product_type": str(planned_target),
                **_serialize_bucket(bucket),
            }
            for (category, model_slot, planned_target), bucket in sorted(
                projected_partial_slot_target_outcomes.items(),
                key=lambda kv: (-int(len(kv[1]["plans"])), kv[0][0], kv[0][1], kv[0][2]),
            )
        ]
        runtime_policy_outcome_rows = [
            {
                "category": str(category),
                "runtime_policy": str(runtime_policy),
                **_serialize_bucket(bucket),
            }
            for (category, runtime_policy), bucket in sorted(
                runtime_policy_outcomes.items(),
                key=lambda kv: (-int(len(kv[1]["plans"])), kv[0][0], kv[0][1]),
            )
        ]
        served_model_slot_by_category_summary = {
            str(cat): {str(slot): int(count) for slot, count in sorted(counter.items(), key=lambda kv: (-kv[1], kv[0]))}
            for cat, counter in sorted(served_model_slot_by_category.items(), key=lambda kv: kv[0])
        }
        projected_partial_slot_by_category_summary = {
            str(cat): {str(slot): int(count) for slot, count in sorted(counter.items(), key=lambda kv: (-kv[1], kv[0]))}
            for cat, counter in sorted(projected_partial_slot_by_category.items(), key=lambda kv: kv[0])
        }
        projected_partial_slot_target_rows = [
            {
                "category": str(category),
                "model_slot": str(model_slot),
                "planned_target_product_type": str(planned_target),
                "plans": int(count),
            }
            for (category, model_slot, planned_target), count in sorted(
                projected_partial_slot_target_counts.items(),
                key=lambda kv: (-int(kv[1]), kv[0][0], kv[0][1], kv[0][2]),
            )
        ]
        runtime_policy_by_category_summary = {
            str(cat): {
                str(policy): int(count)
                for policy, count in sorted(counter.items(), key=lambda kv: (-kv[1], kv[0]))
            }
            for cat, counter in sorted(runtime_policy_by_category.items(), key=lambda kv: kv[0])
        }

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

        nextstep_active_artifact = nextstep_model_artifact_summary()
        nextstep_candidate_artifact = (
            nextstep_model_artifact_summary(nextstep_candidate_model_path)
            if nextstep_candidate_model_path
            else None
        )
        nextstep_candidate_teacher_artifact = (
            nextstep_model_artifact_summary(nextstep_candidate_teacher_model_path)
            if nextstep_candidate_teacher_model_path
            else None
        )
        planner_active_artifact = planner_model_artifact_summary()
        planner_candidate_artifact = (
            planner_model_artifact_summary(planner_candidate_model_path)
            if planner_candidate_model_path
            else None
        )
        candidate_path_compare: dict[str, Any] | None = None
        if nextstep_candidate_model_path and bool(_safe_dict(nextstep_candidate_artifact).get("exists")):
            candidate_top1_by_plan: dict[int, str] = {}
            candidate_skipped_counts: Counter[str] = Counter()
            candidate_by_category: dict[str, Counter[str]] = defaultdict(Counter)
            candidate_swap_counts: Counter[tuple[str, str, str]] = Counter()
            candidate_top1_same_plan_ids: set[int] = set()
            candidate_top1_diff_plan_ids: set[int] = set()
            candidate_baseline_source_counts: Counter[str] = Counter()
            candidate_compare_plan_ids = (
                set(analysis_plan_ids)
                if nextstep_candidate_compare_cohort == "analysis"
                else set(model_plan_ids)
            )

            for pid in candidate_compare_plan_ids:
                active_top1 = str(active_top1_by_plan.get(pid) or "").strip().lower()
                baseline_source = "active_prediction"
                if (
                    not active_top1
                    and nextstep_candidate_compare_cohort == "analysis"
                ):
                    planned_target_baseline = str(plan_planned_target_product_type.get(pid) or "").strip().lower()
                    if planned_target_baseline not in {"", "__none__"}:
                        active_top1 = planned_target_baseline
                        baseline_source = "planned_target"
                if not active_top1:
                    candidate_skipped_counts["missing_active_top1"] += 1
                    continue
                candidate_baseline_source_counts[baseline_source] += 1
                candidate_types = list(plan_step_candidate_types.get(pid) or [])
                if not candidate_types:
                    candidate_skipped_counts["no_candidate_types"] += 1
                    continue
                cat = str(plan_category.get(pid) or "").strip().lower() or "__unknown__"
                planned_target_product_type = str(plan_planned_target_product_type.get(pid) or "").strip().lower()
                planned_target_step_index = int(plan_planned_target_step_index.get(pid) or 0)
                predictions = predict_next_product_types_for_model_path(
                    nextstep_candidate_model_path,
                    user=int(plan_user.get(pid) or 0),
                    context_product_ids=list(plan_context_product_ids.get(pid) or []),
                    category=cat,
                    planned_target_product_type=(
                        None if planned_target_product_type in {"", "__none__"} else planned_target_product_type
                    ),
                    planned_target_step_index=(None if planned_target_step_index <= 0 else planned_target_step_index),
                    candidate_types=candidate_types,
                )
                if (
                    nextstep_candidate_teacher_model_path
                    and bool(_safe_dict(nextstep_candidate_teacher_artifact).get("exists"))
                    and nextstep_candidate_teacher_weight > 0.0
                ):
                    teacher_predictions = predict_next_product_types_for_model_path(
                        nextstep_candidate_teacher_model_path,
                        user=int(plan_user.get(pid) or 0),
                        context_product_ids=list(plan_context_product_ids.get(pid) or []),
                        category=cat,
                        planned_target_product_type=(
                            None
                            if planned_target_product_type in {"", "__none__"}
                            else planned_target_product_type
                        ),
                        planned_target_step_index=(
                            None if planned_target_step_index <= 0 else planned_target_step_index
                        ),
                        candidate_types=candidate_types,
                    )
                    predictions = blend_prediction_rows(
                        predictions,
                        teacher_predictions,
                        overlay_weight=nextstep_candidate_teacher_weight,
                        overlay_label="teacher",
                    )
                candidate_top1 = str(_top_prediction_token(predictions) or "").strip().lower()
                if not candidate_top1:
                    candidate_skipped_counts["no_predictions"] += 1
                    continue
                candidate_top1_by_plan[pid] = candidate_top1
                candidate_by_category[cat]["eligible_plans"] += 1
                if active_top1 == candidate_top1:
                    candidate_top1_same_plan_ids.add(pid)
                    candidate_by_category[cat]["same_top1_plans"] += 1
                else:
                    candidate_top1_diff_plan_ids.add(pid)
                    candidate_by_category[cat]["different_top1_plans"] += 1
                    candidate_swap_counts[(cat, active_top1, candidate_top1)] += 1

            candidate_by_category_summary = {
                str(cat): {
                    "eligible_plans": int(counter.get("eligible_plans", 0)),
                    "same_top1_plans": int(counter.get("same_top1_plans", 0)),
                    "different_top1_plans": int(counter.get("different_top1_plans", 0)),
                    "agreement_rate": _round_or_none(
                        _rate(counter.get("same_top1_plans", 0), counter.get("eligible_plans", 0))
                    ),
                }
                for cat, counter in sorted(candidate_by_category.items(), key=lambda kv: kv[0])
            }
            candidate_top_swaps = [
                {
                    "category": str(category),
                    "active_top1": str(active_top1),
                    "candidate_top1": str(candidate_top1),
                    "plans": int(count),
                }
                for (category, active_top1, candidate_top1), count in sorted(
                    candidate_swap_counts.items(),
                    key=lambda kv: (-kv[1], kv[0][0], kv[0][1], kv[0][2]),
                )
            ]

            def _actual_outcome_for_plan(plan_id: int) -> str | None:
                start_key = latest_plan_refresh_key.get(plan_id)
                if start_key is None:
                    start_key = _event_key(plan_updated.get(plan_id), 0)
                completion_rows = sorted(
                    completion_events_by_plan.get(plan_id, []),
                    key=lambda item: _event_key(item[0], item[1]),
                )
                for created_at, event_id, product_type, _matched_by in completion_rows:
                    if _event_key(created_at, event_id) < start_key:
                        continue
                    if str(product_type or "").strip():
                        return str(product_type).strip().lower()
                return None

            candidate_outcome_counts: Counter[str] = Counter()
            candidate_outcome_by_category: dict[str, Counter[str]] = defaultdict(Counter)
            candidate_outcome_by_top1: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
            candidate_outcome_actuals_by_top1: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
            candidate_outcome_by_swap_pair: dict[tuple[str, str, str], Counter[str]] = defaultdict(Counter)
            candidate_outcome_actuals_by_swap_pair: dict[tuple[str, str, str], Counter[str]] = defaultdict(Counter)
            for pid, candidate_top1 in candidate_top1_by_plan.items():
                active_top1 = str(active_top1_by_plan.get(pid) or "").strip().lower()
                if not active_top1:
                    continue
                actual_outcome = _actual_outcome_for_plan(pid)
                if not actual_outcome:
                    continue
                cat = str(plan_category.get(pid) or "__unknown__")
                candidate_outcome_counts["eligible_plans"] += 1
                candidate_outcome_by_category[cat]["eligible_plans"] += 1
                top1_key = (cat, candidate_top1)
                candidate_outcome_by_top1[top1_key]["eligible_plans"] += 1
                candidate_outcome_actuals_by_top1[top1_key][actual_outcome] += 1
                active_hit = active_top1 == actual_outcome
                candidate_hit = candidate_top1 == actual_outcome
                if active_hit:
                    candidate_outcome_counts["active_hits"] += 1
                    candidate_outcome_by_category[cat]["active_hits"] += 1
                    candidate_outcome_by_top1[top1_key]["active_hits"] += 1
                if candidate_hit:
                    candidate_outcome_counts["candidate_hits"] += 1
                    candidate_outcome_by_category[cat]["candidate_hits"] += 1
                    candidate_outcome_by_top1[top1_key]["candidate_hits"] += 1
                if active_hit and candidate_hit:
                    candidate_outcome_counts["both_hits"] += 1
                    candidate_outcome_by_category[cat]["both_hits"] += 1
                    candidate_outcome_by_top1[top1_key]["both_hits"] += 1
                elif active_hit and not candidate_hit:
                    candidate_outcome_counts["active_only_hits"] += 1
                    candidate_outcome_by_category[cat]["active_only_hits"] += 1
                    candidate_outcome_by_top1[top1_key]["active_only_hits"] += 1
                elif candidate_hit and not active_hit:
                    candidate_outcome_counts["candidate_only_hits"] += 1
                    candidate_outcome_by_category[cat]["candidate_only_hits"] += 1
                    candidate_outcome_by_top1[top1_key]["candidate_only_hits"] += 1
                else:
                    candidate_outcome_counts["neither_hits"] += 1
                    candidate_outcome_by_category[cat]["neither_hits"] += 1
                    candidate_outcome_by_top1[top1_key]["neither_hits"] += 1
                if active_top1 != candidate_top1:
                    pair_key = (cat, active_top1, candidate_top1)
                    candidate_outcome_by_swap_pair[pair_key]["eligible_plans"] += 1
                    candidate_outcome_actuals_by_swap_pair[pair_key][actual_outcome] += 1
                    if active_hit:
                        candidate_outcome_by_swap_pair[pair_key]["active_hits"] += 1
                    if candidate_hit:
                        candidate_outcome_by_swap_pair[pair_key]["candidate_hits"] += 1
                    if active_hit and candidate_hit:
                        candidate_outcome_by_swap_pair[pair_key]["both_hits"] += 1
                    elif active_hit and not candidate_hit:
                        candidate_outcome_by_swap_pair[pair_key]["active_only_hits"] += 1
                    elif candidate_hit and not active_hit:
                        candidate_outcome_by_swap_pair[pair_key]["candidate_only_hits"] += 1
                    else:
                        candidate_outcome_by_swap_pair[pair_key]["neither_hits"] += 1

            candidate_outcome_summary = _challenger_outcome_detail(
                candidate_outcome_counts,
                challenger_label="candidate",
            )
            candidate_outcome_summary["by_category"] = {
                str(cat): _challenger_outcome_detail(counter, challenger_label="candidate")
                for cat, counter in sorted(candidate_outcome_by_category.items(), key=lambda kv: kv[0])
            }
            candidate_outcome_by_top1_rows = [
                {
                    "category": str(category),
                    "candidate_top1": str(candidate_top1),
                    **_challenger_outcome_detail(
                        counter,
                        challenger_label="candidate",
                        actual_outcomes=candidate_outcome_actuals_by_top1.get((category, candidate_top1)),
                    ),
                }
                for (category, candidate_top1), counter in sorted(
                    candidate_outcome_by_top1.items(),
                    key=lambda kv: (-int(kv[1].get("eligible_plans", 0)), kv[0][0], kv[0][1]),
                )
            ]
            candidate_outcome_by_swap_pair_rows = [
                {
                    "category": str(category),
                    "active_top1": str(active_top1),
                    "candidate_top1": str(candidate_top1),
                    **_challenger_outcome_detail(
                        counter,
                        challenger_label="candidate",
                        actual_outcomes=candidate_outcome_actuals_by_swap_pair.get(
                            (category, active_top1, candidate_top1)
                        ),
                    ),
                }
                for (category, active_top1, candidate_top1), counter in sorted(
                    candidate_outcome_by_swap_pair.items(),
                    key=lambda kv: (-int(kv[1].get("eligible_plans", 0)), kv[0][0], kv[0][1], kv[0][2]),
                )
            ]

            candidate_path_compare = {
                "model_path": str(_safe_dict(nextstep_candidate_artifact).get("model_path") or nextstep_candidate_model_path),
                "model_version": str(_safe_dict(nextstep_candidate_artifact).get("model_version") or ""),
                "selected_feature_set": str(_safe_dict(nextstep_candidate_artifact).get("selected_feature_set") or ""),
                "teacher_model_path": str(
                    _safe_dict(nextstep_candidate_teacher_artifact).get("model_path")
                    or nextstep_candidate_teacher_model_path
                    or ""
                ),
                "teacher_model_version": str(
                    _safe_dict(nextstep_candidate_teacher_artifact).get("model_version") or ""
                ),
                "teacher_weight": (
                    float(nextstep_candidate_teacher_weight)
                    if nextstep_candidate_teacher_model_path
                    else None
                ),
                "blend_mode": (
                    "weighted_probability_sum" if nextstep_candidate_teacher_model_path else None
                ),
                "compare_cohort": nextstep_candidate_compare_cohort,
                "plans_scanned": int(len(candidate_compare_plan_ids)),
                "predicted_plans": int(len(candidate_top1_by_plan)),
                "skipped_counts": {
                    str(k): int(v)
                    for k, v in sorted(candidate_skipped_counts.items(), key=lambda kv: (-kv[1], kv[0]))
                },
                "baseline_source_counts": {
                    str(k): int(v)
                    for k, v in sorted(candidate_baseline_source_counts.items(), key=lambda kv: (-kv[1], kv[0]))
                },
                "top1_comparison": {
                    "eligible_plans": int(len(candidate_top1_by_plan)),
                    "same_top1_plans": int(len(candidate_top1_same_plan_ids)),
                    "different_top1_plans": int(len(candidate_top1_diff_plan_ids)),
                    "agreement_rate": _round_or_none(
                        _rate(len(candidate_top1_same_plan_ids), len(candidate_top1_by_plan))
                    ),
                },
                "by_category": candidate_by_category_summary,
                "top_swaps": candidate_top_swaps[:25],
                "outcome_comparison": candidate_outcome_summary,
                "outcome_by_candidate_top1": candidate_outcome_by_top1_rows[:25],
                "outcome_by_swap_pair": candidate_outcome_by_swap_pair_rows[:25],
            }
        shadow_by_category_summary = {
            str(cat): {
                "plans_with_shadow_meta": int(counter.get("plans_with_shadow_meta", 0)),
                "shadow_enabled_plans": int(counter.get("shadow_enabled_plans", 0)),
                "eligible_plans": int(counter.get("eligible_plans", 0)),
                "same_top1_plans": int(counter.get("same_top1_plans", 0)),
                "different_top1_plans": int(counter.get("different_top1_plans", 0)),
                "agreement_rate": _round_or_none(
                    _rate(counter.get("same_top1_plans", 0), counter.get("eligible_plans", 0))
                ),
            }
            for cat, counter in sorted(shadow_by_category.items(), key=lambda kv: kv[0])
        }
        shadow_top_swaps = [
            {
                "category": str(category),
                "active_top1": str(active_top1),
                "shadow_top1": str(shadow_top1),
                "plans": int(count),
            }
            for (category, active_top1, shadow_top1), count in sorted(
                shadow_swap_counts.items(),
                key=lambda kv: (-kv[1], kv[0][0], kv[0][1], kv[0][2]),
            )
        ]
        def _actual_outcome_for_plan(plan_id: int) -> str | None:
            start_key = latest_plan_refresh_key.get(plan_id)
            if start_key is None:
                start_key = _event_key(plan_updated.get(plan_id), 0)
            completion_rows = sorted(
                completion_events_by_plan.get(plan_id, []),
                key=lambda item: _event_key(item[0], item[1]),
            )
            for created_at, event_id, product_type, _matched_by in completion_rows:
                if _event_key(created_at, event_id) < start_key:
                    continue
                if str(product_type or "").strip():
                    return str(product_type).strip().lower()
            return None

        shadow_outcome_counts: Counter[str] = Counter()
        shadow_outcome_by_category: dict[str, Counter[str]] = defaultdict(Counter)
        shadow_outcome_by_shadow_top1: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
        shadow_outcome_actuals_by_shadow_top1: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
        shadow_outcome_by_swap_pair: dict[tuple[str, str, str], Counter[str]] = defaultdict(Counter)
        shadow_outcome_actuals_by_swap_pair: dict[tuple[str, str, str], Counter[str]] = defaultdict(Counter)
        for pid in analysis_plan_ids:
            if not bool(shadow_enabled_by_plan.get(pid)):
                continue
            active_top1 = str(active_top1_by_plan.get(pid) or "").strip().lower()
            shadow_top1 = str(shadow_top1_by_plan.get(pid) or "").strip().lower()
            if not active_top1 or not shadow_top1:
                continue
            actual_outcome = _actual_outcome_for_plan(pid)
            if not actual_outcome:
                continue
            cat = str(plan_category.get(pid) or "__unknown__")
            shadow_outcome_counts["eligible_plans"] += 1
            shadow_outcome_by_category[cat]["eligible_plans"] += 1
            shadow_key = (cat, shadow_top1)
            shadow_outcome_by_shadow_top1[shadow_key]["eligible_plans"] += 1
            shadow_outcome_actuals_by_shadow_top1[shadow_key][actual_outcome] += 1
            active_hit = active_top1 == actual_outcome
            shadow_hit = shadow_top1 == actual_outcome
            if active_hit:
                shadow_outcome_counts["active_hits"] += 1
                shadow_outcome_by_category[cat]["active_hits"] += 1
                shadow_outcome_by_shadow_top1[shadow_key]["active_hits"] += 1
            if shadow_hit:
                shadow_outcome_counts["shadow_hits"] += 1
                shadow_outcome_by_category[cat]["shadow_hits"] += 1
                shadow_outcome_by_shadow_top1[shadow_key]["shadow_hits"] += 1
            if active_hit and shadow_hit:
                shadow_outcome_counts["both_hits"] += 1
                shadow_outcome_by_category[cat]["both_hits"] += 1
                shadow_outcome_by_shadow_top1[shadow_key]["both_hits"] += 1
            elif active_hit and not shadow_hit:
                shadow_outcome_counts["active_only_hits"] += 1
                shadow_outcome_by_category[cat]["active_only_hits"] += 1
                shadow_outcome_by_shadow_top1[shadow_key]["active_only_hits"] += 1
            elif shadow_hit and not active_hit:
                shadow_outcome_counts["shadow_only_hits"] += 1
                shadow_outcome_by_category[cat]["shadow_only_hits"] += 1
                shadow_outcome_by_shadow_top1[shadow_key]["shadow_only_hits"] += 1
            else:
                shadow_outcome_counts["neither_hits"] += 1
                shadow_outcome_by_category[cat]["neither_hits"] += 1
                shadow_outcome_by_shadow_top1[shadow_key]["neither_hits"] += 1
            if active_top1 != shadow_top1:
                pair_key = (cat, active_top1, shadow_top1)
                shadow_outcome_by_swap_pair[pair_key]["eligible_plans"] += 1
                shadow_outcome_actuals_by_swap_pair[pair_key][actual_outcome] += 1
                if active_hit:
                    shadow_outcome_by_swap_pair[pair_key]["active_hits"] += 1
                if shadow_hit:
                    shadow_outcome_by_swap_pair[pair_key]["shadow_hits"] += 1
                if active_hit and shadow_hit:
                    shadow_outcome_by_swap_pair[pair_key]["both_hits"] += 1
                elif active_hit and not shadow_hit:
                    shadow_outcome_by_swap_pair[pair_key]["active_only_hits"] += 1
                elif shadow_hit and not active_hit:
                    shadow_outcome_by_swap_pair[pair_key]["shadow_only_hits"] += 1
                else:
                    shadow_outcome_by_swap_pair[pair_key]["neither_hits"] += 1
        shadow_outcome_summary = _shadow_outcome_detail(shadow_outcome_counts)
        shadow_outcome_summary["by_category"] = {
            str(cat): _shadow_outcome_detail(counter)
            for cat, counter in sorted(shadow_outcome_by_category.items(), key=lambda kv: kv[0])
        }
        shadow_outcome_by_shadow_top1_rows = [
            {
                "category": str(category),
                "shadow_top1": str(shadow_top1),
                **_shadow_outcome_detail(
                    counter,
                    actual_outcomes=shadow_outcome_actuals_by_shadow_top1.get((category, shadow_top1)),
                ),
            }
            for (category, shadow_top1), counter in sorted(
                shadow_outcome_by_shadow_top1.items(),
                key=lambda kv: (-int(kv[1].get("eligible_plans", 0)), kv[0][0], kv[0][1]),
            )
        ]
        shadow_outcome_by_swap_pair_rows = [
            {
                "category": str(category),
                "active_top1": str(active_top1),
                "shadow_top1": str(shadow_top1),
                **_shadow_outcome_detail(
                    counter,
                    actual_outcomes=shadow_outcome_actuals_by_swap_pair.get((category, active_top1, shadow_top1)),
                ),
            }
            for (category, active_top1, shadow_top1), counter in sorted(
                shadow_outcome_by_swap_pair.items(),
                key=lambda kv: (-int(kv[1].get("eligible_plans", 0)), kv[0][0], kv[0][1], kv[0][2]),
            )
        ]

        payload: dict[str, Any] = {
            "generated_at_utc": now_utc.isoformat(),
            "window_start_utc": since.isoformat(),
            "window_end_utc": now_utc.isoformat(),
            "params": {
                "days": days,
                "include_ga": include_ga,
                "categories": categories,
                "refresh_caller": refresh_caller_filter or "all",
                "planned_target_product_types": sorted(planned_target_filter) or ["all"],
                "skip_event_metrics": skip_event_metrics,
                "format": out_format,
                "cohort_mode": cohort_mode,
                "control": control,
                "min_sample": min_sample,
                "nextstep_candidate_model_path": nextstep_candidate_model_path or None,
                "nextstep_candidate_teacher_model_path": nextstep_candidate_teacher_model_path or None,
                "nextstep_candidate_teacher_weight": (
                    nextstep_candidate_teacher_weight if nextstep_candidate_teacher_model_path else None
                ),
                "nextstep_candidate_compare_cohort": nextstep_candidate_compare_cohort,
                "planner_candidate_model_path": planner_candidate_model_path or None,
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
                "served_model_slots": {
                    "slot_counts": {
                        str(k): int(v)
                        for k, v in sorted(served_model_slot_counts.items(), key=lambda kv: (-kv[1], kv[0]))
                    },
                    "model_version_counts": {
                        str(k): int(v)
                        for k, v in sorted(served_model_version_counts.items(), key=lambda kv: (-kv[1], kv[0]))
                    },
                    "by_category": served_model_slot_by_category_summary,
                    "model_used_outcomes_by_slot": served_model_slot_outcome_rows[:25],
                    "model_used_outcomes_by_slot_and_planned_target": served_model_slot_target_outcome_rows[:50],
                },
                "projected_partial_slots": {
                    "slot_counts": {
                        str(k): int(v)
                        for k, v in sorted(projected_partial_slot_counts.items(), key=lambda kv: (-kv[1], kv[0]))
                    },
                    "by_category": projected_partial_slot_by_category_summary,
                    "by_slot_and_planned_target": projected_partial_slot_target_rows[:50],
                    "projected_outcomes_by_slot": projected_partial_slot_outcome_rows[:25],
                    "projected_outcomes_by_slot_and_planned_target": projected_partial_slot_target_outcome_rows[:50],
                },
                "runtime_policies": {
                    "all_plans_with_any_policy": int(len(runtime_policy_all_plan_ids)),
                    "plans_with_any_policy": int(len(runtime_policy_plan_ids)),
                    "all_plans_with_any_policy_by_decision": {
                        str(k): int(v)
                        for k, v in sorted(runtime_policy_all_plan_counts_by_decision.items(), key=lambda kv: (-kv[1], kv[0]))
                    },
                    "policy_counts_all_plans": {
                        str(k): int(v)
                        for k, v in sorted(runtime_policy_counts_all_plans.items(), key=lambda kv: (-kv[1], kv[0]))
                    },
                    "policy_counts": {
                        str(k): int(v)
                        for k, v in sorted(runtime_policy_counts.items(), key=lambda kv: (-kv[1], kv[0]))
                    },
                    "source_counts_all_plans": {
                        str(k): int(v)
                        for k, v in sorted(runtime_policy_source_counts_all_plans.items(), key=lambda kv: (-kv[1], kv[0]))
                    },
                    "source_counts": {
                        str(k): int(v)
                        for k, v in sorted(runtime_policy_source_counts.items(), key=lambda kv: (-kv[1], kv[0]))
                    },
                    "by_category": runtime_policy_by_category_summary,
                    "model_used_outcomes_by_policy": runtime_policy_outcome_rows[:50],
                },
                "shadow": {
                    "plans_with_shadow_meta": int(len(shadow_meta_plan_ids)),
                    "shadow_enabled_plans": int(len(shadow_enabled_plan_ids)),
                    "reason_counts": {
                        str(k): int(v)
                        for k, v in sorted(shadow_reason_counts.items(), key=lambda kv: (-kv[1], kv[0]))
                    },
                    "model_version_counts": {
                        str(k): int(v)
                        for k, v in sorted(shadow_model_version_counts.items(), key=lambda kv: (-kv[1], kv[0]))
                    },
                    "top1_comparison": {
                        "eligible_plans": int(len(shadow_top1_eligible_plan_ids)),
                        "same_top1_plans": int(len(shadow_top1_same_plan_ids)),
                        "different_top1_plans": int(len(shadow_top1_diff_plan_ids)),
                        "agreement_rate": _round_or_none(
                            _rate(len(shadow_top1_same_plan_ids), len(shadow_top1_eligible_plan_ids))
                        ),
                    },
                    "by_category": shadow_by_category_summary,
                    "top_swaps": shadow_top_swaps[:25],
                    "outcome_comparison": shadow_outcome_summary,
                    "outcome_by_shadow_top1": shadow_outcome_by_shadow_top1_rows[:25],
                    "outcome_by_swap_pair": shadow_outcome_by_swap_pair_rows[:25],
                },
                "candidate_path_compare": candidate_path_compare,
            },
            "artifacts": {
                "nextstep": {
                    "active": nextstep_active_artifact,
                    "candidate": nextstep_candidate_artifact,
                    "candidate_teacher": nextstep_candidate_teacher_artifact,
                },
                "planner": {
                    "active": planner_active_artifact,
                    "candidate": planner_candidate_artifact,
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
                "Artifact inventory shows active runtime model summaries plus optional candidate paths passed via CLI.",
                "Runtime policy buckets are multi-attributed: one model-used plan can contribute to multiple runtime_policies when multiple policies fire.",
                "Shadow comparison uses stored active and shadow prediction lists from RoadmapPlan.meta.ml.",
                "Shadow vs outcome compares top-1 active/shadow predictions to the first STEP_COMPLETED product_type after the latest PLAN_REFRESHED for that plan.",
                "Candidate-path compare is a read-only simulation against --nextstep-candidate-model-path and does not write shadow metadata.",
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
