from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from datetime import timedelta
from pathlib import Path
from typing import Any

from django.core.management.base import BaseCommand, CommandError
from django.db.models import Count
from django.db.models.functions import Coalesce
from django.utils import timezone

from offers.models import OfferAssignment, OfferEvent
from roadmap_app.ml_next_step import (
    v4_category_rollout_status,
    v4_category_uplift_guard_status_from_report,
    v4_runtime_uplift_report_path,
)
from roadmap_app.models import RoadmapEvent, RoadmapPlan, RoadmapStep


CATEGORY_CHOICES = ["all", "skincare", "haircare", "makeup", "fragrance"]
FORMAT_CHOICES = ["md", "json", "both"]
COHORT_MODE_CHOICES = ["fresh", "all"]
CONTROL_CHOICES = ["disabled", "fallback", "non_model"]
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


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value or "").strip().lower() in {"1", "true", "yes", "y", "on"}


def _rate(numerator: int, denominator: int) -> float | None:
    if denominator <= 0:
        return None
    return float(numerator) / float(denominator)


def _pct(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value * 100.0:.2f}%"


def _round_or_none(value: float | None, ndigits: int = 6) -> float | None:
    if value is None:
        return None
    return round(float(value), ndigits)


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
        "plans_total": 0,
        "user_ids": set(),
        "category_counts": Counter(),
        "steps_total": 0,
        "step_status": Counter(),
        "step_events": Counter(),
        "exposed_plan_ids": set(),
        "exposed_step_ids": set(),
        "offers": Counter(),
    }


def _new_source_bucket() -> dict[str, Any]:
    return {
        "step_events": Counter(),
        "exposed_plan_ids": set(),
        "exposed_step_ids": set(),
    }


def _decision_from_meta(meta: dict[str, Any]) -> str:
    ml = _safe_dict(meta.get("ml"))
    decision = str(ml.get("decision") or "").strip().lower()
    if decision in VALID_DECISIONS:
        return decision
    return "missing_ml_meta"


def _decision_to_cohort(
    decision: str,
    *,
    cohort_mode: str,
    control_decisions: set[str],
) -> str | None:
    if cohort_mode == "fresh" and decision == "missing_ml_meta":
        return None
    if decision == "model_used":
        return "model_used"
    if decision in control_decisions:
        return "control"
    return None


def _freshness_labels(updated_at, *, now_utc, since):
    labels: list[str] = ["window"]
    if updated_at >= (now_utc - timedelta(days=3)):
        labels.append("3d")
    if updated_at >= (now_utc - timedelta(days=1)):
        labels.append("1d")
    if updated_at < since:
        return []
    return labels


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


def _is_shortcut_target(target: dict[str, Any]) -> bool:
    picked_via = str(target.get("picked_via") or "").strip().lower()
    return picked_via.startswith("roadmap_shortcut")


def _wilson_ci(success: int, total: int, z: float = 1.959963984540054) -> dict[str, float | None]:
    if total <= 0:
        return {"low": None, "high": None}
    p_hat = float(success) / float(total)
    z2 = z * z
    denom = 1.0 + (z2 / float(total))
    center = (p_hat + (z2 / (2.0 * float(total)))) / denom
    margin = (z / denom) * math.sqrt(
        (p_hat * (1.0 - p_hat) / float(total)) + (z2 / (4.0 * float(total) * float(total)))
    )
    low = max(0.0, center - margin)
    high = min(1.0, center + margin)
    return {"low": _round_or_none(low), "high": _round_or_none(high)}


def _diff_ci_and_z_test(
    *,
    success_a: int,
    total_a: int,
    success_b: int,
    total_b: int,
    z_crit: float = 1.959963984540054,
) -> dict[str, float | None]:
    if total_a <= 0 or total_b <= 0:
        return {
            "diff_ci95": {"low": None, "high": None},
            "z_stat": None,
            "p_value": None,
        }

    p_a = float(success_a) / float(total_a)
    p_b = float(success_b) / float(total_b)
    diff = p_a - p_b

    se_diff = math.sqrt(
        (p_a * (1.0 - p_a) / float(total_a)) + (p_b * (1.0 - p_b) / float(total_b))
    )
    if se_diff > 0.0:
        ci_low = diff - (z_crit * se_diff)
        ci_high = diff + (z_crit * se_diff)
    else:
        ci_low = diff
        ci_high = diff

    pooled = float(success_a + success_b) / float(total_a + total_b)
    se_pooled = math.sqrt(
        pooled * (1.0 - pooled) * ((1.0 / float(total_a)) + (1.0 / float(total_b)))
    )
    if se_pooled > 0.0:
        z_stat = diff / se_pooled
        p_value = math.erfc(abs(z_stat) / math.sqrt(2.0))
    else:
        z_stat = None
        p_value = None

    return {
        "diff_ci95": {
            "low": _round_or_none(ci_low),
            "high": _round_or_none(ci_high),
        },
        "z_stat": _round_or_none(z_stat),
        "p_value": _round_or_none(p_value),
    }


def _build_uplift_metric(
    *,
    metric: str,
    model_success: int,
    model_total: int,
    control_success: int,
    control_total: int,
    model_plans: int,
    control_plans: int,
    min_plans: int,
) -> dict[str, Any]:
    model_rate = _rate(model_success, model_total)
    control_rate = _rate(control_success, control_total)
    abs_lift = None
    rel_lift = None
    if model_rate is not None and control_rate is not None:
        abs_lift = model_rate - control_rate
        if control_rate > 0.0:
            rel_lift = abs_lift / control_rate

    stat_block = _diff_ci_and_z_test(
        success_a=model_success,
        total_a=model_total,
        success_b=control_success,
        total_b=control_total,
    )
    low_sample = (
        model_plans < min_plans
        or control_plans < min_plans
        or model_total < min_plans
        or control_total < min_plans
    )

    return {
        "metric": metric,
        "model": {
            "success": int(model_success),
            "total": int(model_total),
            "rate": _round_or_none(model_rate),
            "wilson_ci95": _wilson_ci(model_success, model_total),
        },
        "control": {
            "success": int(control_success),
            "total": int(control_total),
            "rate": _round_or_none(control_rate),
            "wilson_ci95": _wilson_ci(control_success, control_total),
        },
        "abs_lift": _round_or_none(abs_lift),
        "rel_lift": _round_or_none(rel_lift),
        "diff_ci95": stat_block["diff_ci95"],
        "z_stat": stat_block["z_stat"],
        "p_value": stat_block["p_value"],
        "low_sample": bool(low_sample),
    }


def _finalize_bucket(bucket: dict[str, Any]) -> dict[str, Any]:
    plans_total = int(bucket["plans_total"])
    users_total = len(bucket["user_ids"])
    steps_total = int(bucket["steps_total"])

    step_exposed = int(bucket["step_events"].get("exposed", 0))
    step_clicked = int(bucket["step_events"].get("clicked", 0))
    step_completed = int(bucket["step_events"].get("completed", 0))
    step_skipped = int(bucket["step_events"].get("skipped", 0))

    offers_assigned_total = int(bucket["offers"].get("assigned_total", 0))
    offers_with_roadmap_shortcut = int(bucket["offers"].get("shortcut_total", 0))
    offer_exposed = int(bucket["offers"].get("exposed", 0))
    offer_clicked = int(bucket["offers"].get("clicked", 0))
    offer_redeemed = int(bucket["offers"].get("redeemed", 0))

    return {
        "plans_total": plans_total,
        "users_total": users_total,
        "categories": {
            str(k): int(v)
            for k, v in sorted(bucket["category_counts"].items(), key=lambda kv: (-kv[1], kv[0]))
        },
        "steps_total": steps_total,
        "steps_recommended": int(bucket["step_status"].get(RoadmapStep.Status.RECOMMENDED, 0)),
        "steps_owned": int(bucket["step_status"].get(RoadmapStep.Status.OWNED, 0)),
        "steps_completed": int(bucket["step_status"].get(RoadmapStep.Status.COMPLETED, 0)),
        "steps_missing": int(bucket["step_status"].get(RoadmapStep.Status.MISSING, 0)),
        "steps_skipped": int(bucket["step_status"].get(RoadmapStep.Status.SKIPPED, 0)),
        "roadmap_step_exposed": step_exposed,
        "roadmap_step_clicked": step_clicked,
        "roadmap_step_completed": step_completed,
        "roadmap_step_skipped": step_skipped,
        "updated_plans_with_exposed": len(bucket["exposed_plan_ids"]),
        "updated_steps_with_exposed": len(bucket["exposed_step_ids"]),
        "exposed_plan_rate": _round_or_none(_rate(len(bucket["exposed_plan_ids"]), plans_total)),
        "exposed_step_rate": _round_or_none(_rate(len(bucket["exposed_step_ids"]), steps_total)),
        "step_ctr": _round_or_none(_rate(step_clicked, step_exposed)),
        "step_completion_rate": _round_or_none(_rate(step_completed, step_exposed)),
        "skip_rate": _round_or_none(_rate(step_skipped, step_exposed)),
        "offers_assigned_total": offers_assigned_total,
        "offers_with_roadmap_shortcut": offers_with_roadmap_shortcut,
        "offer_exposed": offer_exposed,
        "offer_clicked": offer_clicked,
        "offer_redeemed": offer_redeemed,
        "offer_ctr": _round_or_none(_rate(offer_clicked, offer_exposed)),
        "offer_redeem_rate": _round_or_none(_rate(offer_redeemed, offer_exposed)),
    }


def _finalize_source_bucket(bucket: dict[str, Any]) -> dict[str, Any]:
    exposed = int(bucket["step_events"].get("exposed", 0))
    clicked = int(bucket["step_events"].get("clicked", 0))
    completed = int(bucket["step_events"].get("completed", 0))
    skipped = int(bucket["step_events"].get("skipped", 0))
    return {
        "roadmap_step_exposed": exposed,
        "roadmap_step_clicked": clicked,
        "roadmap_step_completed": completed,
        "roadmap_step_skipped": skipped,
        "plans_with_exposed": len(bucket["exposed_plan_ids"]),
        "steps_with_exposed": len(bucket["exposed_step_ids"]),
        "step_ctr": _round_or_none(_rate(clicked, exposed)),
        "step_completion_rate": _round_or_none(_rate(completed, exposed)),
        "skip_rate": _round_or_none(_rate(skipped, exposed)),
    }


def _resolve_out_stem(*, out: str | None, days: int, category: str) -> Path:
    if out:
        p = Path(out)
        if p.suffix.lower() in {".md", ".json"}:
            return p.with_suffix("")
        return p
    return Path("reports") / f"roadmap_ml_uplift_{days}d_{category}"


def _runtime_window_label(days: int) -> str | None:
    if int(days) == 7:
        return "7d"
    if int(days) == 30:
        return "30d"
    return None


def _render_metric_row(metric_payload: dict[str, Any]) -> list[str]:
    model_rate = metric_payload["model"]["rate"]
    control_rate = metric_payload["control"]["rate"]
    abs_lift = metric_payload.get("abs_lift")
    rel_lift = metric_payload.get("rel_lift")
    low_sample = "yes" if metric_payload.get("low_sample") else "no"
    return [
        str(metric_payload.get("metric")),
        _pct(model_rate),
        _pct(control_rate),
        _pct(abs_lift),
        _pct(rel_lift),
        low_sample,
    ]


def _build_markdown(payload: dict[str, Any]) -> str:
    params = _safe_dict(payload.get("params"))
    overall = _safe_dict(payload.get("overall"))
    cohorts = _safe_dict(payload.get("cohorts"))
    model = _safe_dict(cohorts.get("model_used"))
    control = _safe_dict(cohorts.get("control"))
    uplift = _safe_dict(payload.get("uplift"))
    partial_makeup = _safe_dict(payload.get("partial_makeup_uplift"))
    runtime = _safe_dict(payload.get("runtime_observability"))
    rollout_reco = _safe_dict(payload.get("rollout_recommendation_by_category"))
    unattributed = _safe_dict(payload.get("unattributed_excluded"))

    lines: list[str] = []
    lines.append("# Roadmap ML Uplift Report")
    lines.append("")
    lines.append(f"- Generated at (UTC): `{payload.get('generated_at_utc')}`")
    lines.append(
        f"- Window: `{payload.get('window_start_utc')}` .. `{payload.get('window_end_utc')}` "
        f"(days={params.get('days')})"
    )
    lines.append(
        f"- Cohort mode: `{params.get('cohort_mode')}` | Control: `{params.get('control')}` | "
        f"Category: `{params.get('category')}` | include_ga: `{params.get('include_ga')}`"
    )
    lines.append(
        f"- Note: uplift is computed on cohort-mode `{params.get('cohort_mode')}` "
        "(default fresh excludes missing_ml_meta from active comparison)."
    )
    lines.append("")

    lines.append("## 1) Cohort counts")
    lines.append(
        _md_table(
            [
                "cohort",
                "plans_total",
                "users_total",
                "steps_total",
                "steps_recommended",
                "steps_owned",
                "steps_completed",
                "steps_missing",
                "steps_skipped",
            ],
            [
                [
                    "model_used",
                    model.get("plans_total", 0),
                    model.get("users_total", 0),
                    model.get("steps_total", 0),
                    model.get("steps_recommended", 0),
                    model.get("steps_owned", 0),
                    model.get("steps_completed", 0),
                    model.get("steps_missing", 0),
                    model.get("steps_skipped", 0),
                ],
                [
                    f"control ({params.get('control')})",
                    control.get("plans_total", 0),
                    control.get("users_total", 0),
                    control.get("steps_total", 0),
                    control.get("steps_recommended", 0),
                    control.get("steps_owned", 0),
                    control.get("steps_completed", 0),
                    control.get("steps_missing", 0),
                    control.get("steps_skipped", 0),
                ],
            ],
        )
    )
    lines.append("")

    step_uplift = _safe_dict(_safe_dict(uplift.get("overall")).get("step_funnel"))
    lines.append("## 2) Step funnel uplift")
    lines.append(
        _md_table(
            ["metric", "model_used", "control", "abs_lift", "rel_lift", "low_sample"],
            [
                _render_metric_row(_safe_dict(step_uplift.get("step_ctr"))),
                _render_metric_row(_safe_dict(step_uplift.get("step_completion_rate"))),
                _render_metric_row(_safe_dict(step_uplift.get("skip_rate"))),
            ],
        )
    )
    lines.append("")

    offer_uplift = _safe_dict(_safe_dict(uplift.get("overall")).get("offer_funnel"))
    lines.append("## 3) Offer funnel uplift")
    lines.append(
        _md_table(
            ["metric", "model_used", "control", "abs_lift", "rel_lift", "low_sample"],
            [
                _render_metric_row(_safe_dict(offer_uplift.get("offer_ctr"))),
                _render_metric_row(_safe_dict(offer_uplift.get("offer_redeem_rate"))),
            ],
        )
    )
    lines.append("")

    lines.append("## 4) Category breakdown")
    by_cat = _safe_dict(uplift.get("by_category"))
    cat_rows: list[list[Any]] = []
    for cat, block in sorted(by_cat.items()):
        step_block = _safe_dict(_safe_dict(block).get("step_funnel"))
        ctr = _safe_dict(step_block.get("step_ctr"))
        comp = _safe_dict(step_block.get("step_completion_rate"))
        cat_rows.append(
            [
                cat,
                _pct(_safe_dict(ctr.get("model")).get("rate")),
                _pct(_safe_dict(ctr.get("control")).get("rate")),
                _pct(ctr.get("abs_lift")),
                _pct(_safe_dict(comp.get("model")).get("rate")),
                _pct(_safe_dict(comp.get("control")).get("rate")),
                _pct(comp.get("abs_lift")),
                "yes" if (ctr.get("low_sample") or comp.get("low_sample")) else "no",
            ]
        )
    lines.append(
        _md_table(
            [
                "category",
                "step_ctr_model",
                "step_ctr_control",
                "step_ctr_abs_lift",
                "step_completion_model",
                "step_completion_control",
                "step_completion_abs_lift",
                "low_sample",
            ],
            cat_rows,
        )
    )
    lines.append("")

    lines.append("### Breakdown by expose source")
    by_source = _safe_dict(uplift.get("by_expose_source"))
    src_rows: list[list[Any]] = []
    for src, block in sorted(by_source.items()):
        step_block = _safe_dict(_safe_dict(block).get("step_funnel"))
        ctr = _safe_dict(step_block.get("step_ctr"))
        comp = _safe_dict(step_block.get("step_completion_rate"))
        src_rows.append(
            [
                src,
                _pct(_safe_dict(ctr.get("model")).get("rate")),
                _pct(_safe_dict(ctr.get("control")).get("rate")),
                _pct(ctr.get("abs_lift")),
                _pct(_safe_dict(comp.get("model")).get("rate")),
                _pct(_safe_dict(comp.get("control")).get("rate")),
                _pct(comp.get("abs_lift")),
                "yes" if (ctr.get("low_sample") or comp.get("low_sample")) else "no",
            ]
        )
    lines.append(
        _md_table(
            [
                "source",
                "step_ctr_model",
                "step_ctr_control",
                "step_ctr_abs_lift",
                "step_completion_model",
                "step_completion_control",
                "step_completion_abs_lift",
                "low_sample",
            ],
            src_rows,
        )
    )
    lines.append("")

    lines.append("### Breakdown by freshness")
    by_fresh = _safe_dict(uplift.get("by_freshness"))
    fresh_rows: list[list[Any]] = []
    for key in ["1d", "3d", "window"]:
        block = _safe_dict(by_fresh.get(key))
        step_block = _safe_dict(_safe_dict(block).get("step_funnel"))
        ctr = _safe_dict(step_block.get("step_ctr"))
        fresh_rows.append(
            [
                key,
                _pct(_safe_dict(ctr.get("model")).get("rate")),
                _pct(_safe_dict(ctr.get("control")).get("rate")),
                _pct(ctr.get("abs_lift")),
                "yes" if ctr.get("low_sample") else "no",
            ]
        )
    lines.append(
        _md_table(
            ["freshness", "step_ctr_model", "step_ctr_control", "step_ctr_abs_lift", "low_sample"],
            fresh_rows,
        )
    )
    lines.append("")

    lines.append("## 5) Makeup partial canary vs non-model control")
    partial_overall = _safe_dict(partial_makeup.get("overall"))
    partial_overall_uplift = _safe_dict(partial_overall.get("uplift"))
    partial_step = _safe_dict(partial_overall_uplift.get("step_funnel"))
    partial_offer = _safe_dict(partial_overall_uplift.get("offer_funnel"))
    lines.append(
        _md_table(
            ["metric", "canary", "control", "abs_lift", "rel_lift", "low_sample"],
            [
                _render_metric_row(_safe_dict(partial_step.get("step_ctr"))),
                _render_metric_row(_safe_dict(partial_step.get("step_completion_rate"))),
                _render_metric_row(_safe_dict(partial_offer.get("offer_ctr"))),
                _render_metric_row(_safe_dict(partial_offer.get("offer_redeem_rate"))),
            ],
        )
    )
    lines.append("")
    lines.append("### Makeup partial by product_type")
    partial_pt_rows: list[list[Any]] = []
    for pt, block in sorted(_safe_dict(partial_makeup.get("by_product_type")).items()):
        metric = _safe_dict(_safe_dict(_safe_dict(block).get("uplift")).get("step_funnel")).get("step_completion_rate")
        metric_payload = _safe_dict(metric)
        partial_pt_rows.append(
            [
                pt,
                _pct(_safe_dict(metric_payload.get("model")).get("rate")),
                _pct(_safe_dict(metric_payload.get("control")).get("rate")),
                _pct(metric_payload.get("abs_lift")),
                "yes" if metric_payload.get("low_sample") else "no",
            ]
        )
    lines.append(
        _md_table(
            [
                "product_type",
                "step_completion_model",
                "step_completion_control",
                "step_completion_abs_lift",
                "low_sample",
            ],
            partial_pt_rows,
        )
    )
    lines.append("")
    lines.append("### Makeup partial by step_index")
    partial_idx_rows: list[list[Any]] = []
    for idx_bucket, block in sorted(_safe_dict(partial_makeup.get("by_step_index")).items()):
        metric = _safe_dict(_safe_dict(_safe_dict(block).get("uplift")).get("step_funnel")).get("step_completion_rate")
        metric_payload = _safe_dict(metric)
        partial_idx_rows.append(
            [
                idx_bucket,
                _pct(_safe_dict(metric_payload.get("model")).get("rate")),
                _pct(_safe_dict(metric_payload.get("control")).get("rate")),
                _pct(metric_payload.get("abs_lift")),
                "yes" if metric_payload.get("low_sample") else "no",
            ]
        )
    lines.append(
        _md_table(
            [
                "step_index",
                "step_completion_model",
                "step_completion_control",
                "step_completion_abs_lift",
                "low_sample",
            ],
            partial_idx_rows,
        )
    )
    lines.append("")

    lines.append("## 6) Runtime observability")
    decisions = _safe_dict(runtime.get("decision_counts"))
    lines.append(
        _md_table(
            ["decision", "count"],
            [[k, int(decisions.get(k, 0))] for k in ["model_used", "fallback", "disabled", "missing_ml_meta"]],
        )
    )
    lines.append("")
    lines.append("### Rollout mode distribution")
    lines.append(
        _md_table(
            ["rollout_mode", "count"],
            [[k, v] for k, v in sorted(_safe_dict(runtime.get("rollout_mode_distribution")).items(), key=lambda kv: (-kv[1], kv[0]))],
        )
    )
    lines.append("")
    lines.append("### Fallback reasons")
    lines.append(
        _md_table(
            ["reason", "count"],
            [[k, v] for k, v in sorted(_safe_dict(runtime.get("fallback_reasons")).items(), key=lambda kv: (-kv[1], kv[0]))],
        )
    )
    lines.append("")
    lines.append("### Disabled reasons")
    lines.append(
        _md_table(
            ["reason", "count"],
            [[k, v] for k, v in sorted(_safe_dict(runtime.get("disabled_reasons")).items(), key=lambda kv: (-kv[1], kv[0]))],
        )
    )
    lines.append("")
    lines.append("### Mode distribution")
    lines.append(
        _md_table(
            ["mode", "count"],
            [[k, v] for k, v in sorted(_safe_dict(runtime.get("mode_distribution")).items(), key=lambda kv: (-kv[1], kv[0]))],
        )
    )
    lines.append("")
    lines.append("### Top model_path values")
    lines.append(
        _md_table(
            ["model_path", "count"],
            [[x.get("model_path"), x.get("count")] for x in _safe_list(runtime.get("top_model_paths"))],
        )
    )
    lines.append("")

    lines.append("## 7) Rollout recommendation by category")
    lines.append(
        _md_table(
            [
                "category",
                "recommendation",
                "rollout_reason",
                "guard_reason",
                "primary_passed",
                "secondary_passed",
                "sample_model",
                "sample_control",
            ],
            [
                [
                    cat,
                    _safe_dict(row).get("recommendation"),
                    _safe_dict(_safe_dict(row).get("rollout")).get("reason"),
                    _safe_dict(_safe_dict(row).get("guard")).get("reason"),
                    _safe_dict(_safe_dict(row).get("guard")).get("primary_passed"),
                    _safe_dict(_safe_dict(row).get("guard")).get("secondary_passed"),
                    _safe_dict(_safe_dict(row).get("guard")).get("sample_size_model"),
                    _safe_dict(_safe_dict(row).get("guard")).get("sample_size_control"),
                ]
                for cat, row in sorted(rollout_reco.items())
            ],
        )
    )
    lines.append("")

    lines.append("## 8) Unattributed / excluded counts")
    lines.append(
        _md_table(
            ["bucket", "count"],
            [[k, v] for k, v in sorted(unattributed.items(), key=lambda kv: kv[0])],
        )
    )
    lines.append("")

    lines.append("## 9) Notes / caveats")
    for note in _safe_list(payload.get("notes")):
        lines.append(f"- {note}")
    lines.append("")

    lines.append("### Summary")
    lines.append(
        f"- model_used plans: {model.get('plans_total', 0)} | control plans: {control.get('plans_total', 0)} "
        f"| analysis plans: {overall.get('analysis_plans_total', 0)}"
    )
    lines.append(
        f"- step_ctr uplift: {_pct(_safe_dict(step_uplift.get('step_ctr')).get('abs_lift'))} "
        f"| offer_ctr uplift: {_pct(_safe_dict(offer_uplift.get('offer_ctr')).get('abs_lift'))}"
    )
    return "\n".join(lines)


class Command(BaseCommand):
    help = "Read-only uplift report for Roadmap ML runtime cohorts (model_used vs control)."

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
        parser.add_argument("--cohort-mode", type=str, default="fresh", choices=COHORT_MODE_CHOICES)
        parser.add_argument("--control", type=str, default="non_model", choices=CONTROL_CHOICES)
        parser.add_argument("--min-plans", type=int, default=30)
        parser.add_argument(
            "--sync-runtime-artifact",
            action="store_true",
            default=False,
            help="Write JSON payload to the model-owned runtime uplift artifact path for 7d/30d windows.",
        )

    def handle(self, *args, **options):
        days = int(options["days"] or 7)
        if days <= 0:
            raise CommandError("--days must be > 0")

        category = str(options["category"] or "all").strip().lower()
        include_ga = bool(options["include_ga"])
        out_raw = options.get("out")
        out_format = str(options["format"] or "both").strip().lower()
        cohort_mode = str(options["cohort_mode"] or "fresh").strip().lower()
        control = str(options["control"] or "non_model").strip().lower()
        min_plans = int(options["min_plans"] or 30)
        sync_runtime_artifact = bool(options.get("sync_runtime_artifact"))
        if min_plans <= 0:
            raise CommandError("--min-plans must be > 0")
        runtime_window = _runtime_window_label(days)
        if sync_runtime_artifact:
            if runtime_window is None:
                raise CommandError("--sync-runtime-artifact requires --days 7 or --days 30")
            if category != "all":
                raise CommandError("--sync-runtime-artifact requires --category all")
            if cohort_mode != "fresh":
                raise CommandError("--sync-runtime-artifact requires --cohort-mode fresh")
            if control != "non_model":
                raise CommandError("--sync-runtime-artifact requires --control non_model")

        now_utc = timezone.now()
        since = now_utc - timedelta(days=days)

        if control == "disabled":
            control_decisions = {"disabled"}
        elif control == "fallback":
            control_decisions = {"fallback"}
        else:
            control_decisions = {"disabled", "fallback"}

        plan_qs = RoadmapPlan.objects.filter(updated_at__gte=since, updated_at__lte=now_utc)
        if category != "all":
            plan_qs = plan_qs.filter(category=category)
        if not include_ga:
            plan_qs = plan_qs.exclude(user__username__startswith="ga_")

        plan_rows = list(
            plan_qs.values(
                "id",
                "user_id",
                "category",
                "updated_at",
                "meta",
            )
        )

        plan_category: dict[int, str] = {}
        plan_user: dict[int, int] = {}
        plan_updated: dict[int, Any] = {}
        plan_decision: dict[int, str] = {}
        plan_segments: dict[int, list[tuple[str, str]]] = {}
        decision_counts: Counter[str] = Counter()
        fallback_reason_counts: Counter[str] = Counter()
        disabled_reason_counts: Counter[str] = Counter()
        mode_counts: Counter[str] = Counter()
        model_path_counts: Counter[str] = Counter()
        rollout_mode_counts: Counter[str] = Counter()
        rollout_selected_counts: Counter[str] = Counter()
        plans_by_user_scope: dict[int, list[dict[str, Any]]] = defaultdict(list)
        plan_rollout_mode: dict[int, str] = {}
        plan_rollout_selected: dict[int, bool] = {}

        for row in plan_rows:
            pid = int(row["id"])
            user_id = int(row["user_id"])
            cat = str(row["category"] or "")
            updated_at = row["updated_at"]
            meta = _safe_dict(row.get("meta"))
            ml = _safe_dict(meta.get("ml"))

            decision = _decision_from_meta(meta)
            decision_counts[decision] += 1

            if decision == "fallback":
                reason = str(ml.get("fallback_reason") or "").strip() or "__missing_reason__"
                fallback_reason_counts[reason] += 1
            elif decision == "disabled":
                reason = str(ml.get("disabled_reason") or ml.get("fallback_reason") or "").strip() or "__missing_reason__"
                disabled_reason_counts[reason] += 1

            mode = str(ml.get("mode") or "none").strip() or "none"
            model_path = str(ml.get("model_path") or "__none__").strip() or "__none__"
            mode_counts[mode] += 1
            model_path_counts[model_path] += 1
            rollout_mode = str(ml.get("rollout_mode") or "none").strip().lower() or "none"
            if rollout_mode not in {"full", "partial", "none"}:
                rollout_mode = "none"
            rollout_selected = _as_bool(ml.get("rollout_selected"))
            if rollout_mode == "full":
                rollout_selected = True
            elif rollout_mode == "none":
                rollout_selected = False
            rollout_mode_counts[rollout_mode] += 1
            rollout_selected_counts["selected" if rollout_selected else "not_selected"] += 1

            plan_category[pid] = cat
            plan_user[pid] = user_id
            plan_updated[pid] = updated_at
            plan_decision[pid] = decision
            plan_rollout_mode[pid] = rollout_mode
            plan_rollout_selected[pid] = rollout_selected
            plans_by_user_scope[user_id].append(
                {
                    "id": pid,
                    "category": cat,
                    "updated_at": updated_at,
                }
            )

        all_scope_plan_ids = set(plan_category.keys())
        if cohort_mode == "fresh":
            cohort_scope_plan_ids = {pid for pid in all_scope_plan_ids if plan_decision.get(pid) != "missing_ml_meta"}
        else:
            cohort_scope_plan_ids = set(all_scope_plan_ids)

        fresh_excluded_missing = len(all_scope_plan_ids - cohort_scope_plan_ids)

        cohort_by_plan_id: dict[int, str] = {}
        for pid in cohort_scope_plan_ids:
            cohort = _decision_to_cohort(
                plan_decision.get(pid, "missing_ml_meta"),
                cohort_mode=cohort_mode,
                control_decisions=control_decisions,
            )
            if cohort:
                cohort_by_plan_id[pid] = cohort

        model_plan_ids = {pid for pid, c in cohort_by_plan_id.items() if c == "model_used"}
        control_plan_ids = {pid for pid, c in cohort_by_plan_id.items() if c == "control"}
        analysis_plan_ids = set(cohort_by_plan_id.keys())
        excluded_non_selected = len(cohort_scope_plan_ids - analysis_plan_ids)
        makeup_cohort_scope_plan_ids = {
            pid for pid in cohort_scope_plan_ids if str(plan_category.get(pid) or "") == "makeup"
        }
        partial_makeup_canary_plan_ids = {
            pid
            for pid in makeup_cohort_scope_plan_ids
            if str(plan_rollout_mode.get(pid) or "none") == "partial" and bool(plan_rollout_selected.get(pid))
        }
        partial_makeup_control_plan_ids = {
            pid
            for pid in makeup_cohort_scope_plan_ids
            if str(plan_decision.get(pid) or "") in {"fallback", "disabled"}
            and pid not in partial_makeup_canary_plan_ids
        }
        partial_makeup_analysis_plan_ids = partial_makeup_canary_plan_ids | partial_makeup_control_plan_ids

        overall_buckets = {"model_used": _new_bucket(), "control": _new_bucket()}
        by_category_buckets: dict[str, dict[str, Any]] = defaultdict(
            lambda: {"model_used": _new_bucket(), "control": _new_bucket()}
        )
        by_freshness_buckets: dict[str, dict[str, Any]] = {
            "1d": {"model_used": _new_bucket(), "control": _new_bucket()},
            "3d": {"model_used": _new_bucket(), "control": _new_bucket()},
            "window": {"model_used": _new_bucket(), "control": _new_bucket()},
        }
        by_source_buckets: dict[str, dict[str, Any]] = {
            "offers": {"model_used": _new_source_bucket(), "control": _new_source_bucket()},
            "roadmap_api": {"model_used": _new_source_bucket(), "control": _new_source_bucket()},
        }
        plan_metrics: dict[int, Counter[str]] = defaultdict(Counter)
        plan_exposed_step_ids: dict[int, set[int]] = defaultdict(set)

        for user_id, rows in plans_by_user_scope.items():
            rows.sort(key=lambda x: (x["updated_at"], x["id"]), reverse=True)
            plans_by_user_scope[user_id] = rows

        for pid in analysis_plan_ids:
            cohort = cohort_by_plan_id[pid]
            user_id = plan_user[pid]
            cat = plan_category[pid]
            updated_at = plan_updated[pid]
            labels = _freshness_labels(updated_at, now_utc=now_utc, since=since)
            segment_keys: list[tuple[str, str]] = [("overall", "all"), ("category", cat)]
            segment_keys.extend([("freshness", x) for x in labels])
            plan_segments[pid] = segment_keys

            for section, key in segment_keys:
                if section == "overall":
                    bucket = overall_buckets[cohort]
                elif section == "category":
                    bucket = by_category_buckets[key][cohort]
                else:
                    bucket = by_freshness_buckets[key][cohort]
                bucket["plans_total"] += 1
                bucket["user_ids"].add(user_id)
                bucket["category_counts"][cat] += 1

        plan_step_types: dict[int, set[str]] = defaultdict(set)
        plan_step_index_buckets: dict[int, set[str]] = defaultdict(set)
        if all_scope_plan_ids:
            for row in (
                RoadmapStep.objects.filter(plan_id__in=all_scope_plan_ids)
                .values("plan_id", "product_type", "step_index")
                .distinct()
            ):
                pid = int(row["plan_id"])
                pt = str(row["product_type"] or "").strip()
                if pt:
                    plan_step_types[pid].add(pt)
                idx_bucket = _step_index_bucket(_to_int(row.get("step_index")))
                plan_step_index_buckets[pid].add(idx_bucket)

        plan_step_status_counts: dict[int, Counter[str]] = defaultdict(Counter)
        plan_steps_total: dict[int, int] = defaultdict(int)
        if cohort_scope_plan_ids:
            for row in (
                RoadmapStep.objects.filter(plan_id__in=cohort_scope_plan_ids)
                .values("plan_id", "status")
                .annotate(c=Count("id"))
            ):
                pid = int(row["plan_id"])
                status = str(row["status"] or "")
                cnt = int(row["c"] or 0)
                plan_step_status_counts[pid][status] += cnt
                plan_steps_total[pid] += cnt

        for pid in analysis_plan_ids:
            cohort = cohort_by_plan_id[pid]
            status_counts = plan_step_status_counts.get(pid, Counter())
            total_steps = int(plan_steps_total.get(pid, 0))
            for section, key in plan_segments.get(pid, []):
                if section == "overall":
                    bucket = overall_buckets[cohort]
                elif section == "category":
                    bucket = by_category_buckets[key][cohort]
                else:
                    bucket = by_freshness_buckets[key][cohort]
                bucket["steps_total"] += total_steps
                bucket["step_status"].update(status_counts)

        step_sources_by_cohort_step: dict[tuple[str, int], set[str]] = defaultdict(set)
        if cohort_scope_plan_ids:
            exposed_qs = (
                RoadmapEvent.objects.filter(
                    created_at__gte=since,
                    created_at__lte=now_utc,
                    event_type=RoadmapEvent.Type.STEP_EXPOSED,
                )
                .annotate(_effective_plan_id=Coalesce("plan_id", "step__plan_id"))
                .filter(_effective_plan_id__in=cohort_scope_plan_ids)
            )
            if not include_ga:
                exposed_qs = exposed_qs.exclude(user__username__startswith="ga_")

            for row in exposed_qs.values("step_id", "context", "_effective_plan_id"):
                pid = int(row["_effective_plan_id"])
                step_id = _to_int(row.get("step_id"))
                plan_metrics[pid]["roadmap_step_exposed"] += 1
                if step_id is not None:
                    plan_exposed_step_ids[pid].add(step_id)
                cohort = cohort_by_plan_id.get(pid)
                if cohort not in {"model_used", "control"}:
                    continue
                source = _source_from_expose_context(_safe_dict(row.get("context")))

                for section, key in plan_segments.get(pid, []):
                    if section == "overall":
                        bucket = overall_buckets[cohort]
                    elif section == "category":
                        bucket = by_category_buckets[key][cohort]
                    else:
                        bucket = by_freshness_buckets[key][cohort]
                    bucket["step_events"]["exposed"] += 1
                    bucket["exposed_plan_ids"].add(pid)
                    if step_id is not None:
                        bucket["exposed_step_ids"].add(step_id)

                source_bucket = by_source_buckets[source][cohort]
                source_bucket["step_events"]["exposed"] += 1
                source_bucket["exposed_plan_ids"].add(pid)
                if step_id is not None:
                    source_bucket["exposed_step_ids"].add(step_id)
                    step_sources_by_cohort_step[(cohort, step_id)].add(source)

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
                .filter(_effective_plan_id__in=cohort_scope_plan_ids)
            )
            if not include_ga:
                interaction_qs = interaction_qs.exclude(user__username__startswith="ga_")

            event_type_map = {
                RoadmapEvent.Type.STEP_CLICKED: "clicked",
                RoadmapEvent.Type.STEP_COMPLETED: "completed",
                RoadmapEvent.Type.STEP_SKIPPED: "skipped",
            }
            for row in interaction_qs.values("step_id", "event_type", "_effective_plan_id"):
                pid = int(row["_effective_plan_id"])
                step_id = _to_int(row.get("step_id"))
                metric_key = event_type_map.get(str(row.get("event_type") or ""))
                if not metric_key:
                    continue
                plan_metrics[pid][f"roadmap_step_{metric_key}"] += 1
                cohort = cohort_by_plan_id.get(pid)
                if cohort not in {"model_used", "control"}:
                    continue

                for section, key in plan_segments.get(pid, []):
                    if section == "overall":
                        bucket = overall_buckets[cohort]
                    elif section == "category":
                        bucket = by_category_buckets[key][cohort]
                    else:
                        bucket = by_freshness_buckets[key][cohort]
                    bucket["step_events"][metric_key] += 1

                if step_id is not None:
                    for source in step_sources_by_cohort_step.get((cohort, step_id), set()):
                        by_source_buckets[source][cohort]["step_events"][metric_key] += 1

        roadmap_assignment_total = 0
        roadmap_assignment_attributed = 0
        roadmap_assignment_unattributed = 0
        roadmap_assignment_out_of_scope = 0
        roadmap_assignment_excluded_non_cohort = 0
        roadmap_assignment_ids: set[int] = set()
        assignment_state: dict[int, dict[str, Any]] = {}

        assign_qs = OfferAssignment.objects.filter(assigned_at__gte=since, assigned_at__lte=now_utc)
        if not include_ga:
            assign_qs = assign_qs.exclude(user__username__startswith="ga_")

        for row in assign_qs.values("id", "user_id", "assigned_at", "reason", "target"):
            assignment_id = int(row["id"])
            user_id = int(row["user_id"])
            assigned_at = row["assigned_at"]
            reason = _safe_dict(row.get("reason"))
            target = _safe_dict(row.get("target"))

            if not _is_roadmap_related_assignment(reason=reason, target=target):
                continue

            roadmap_assignment_total += 1
            roadmap_assignment_ids.add(assignment_id)

            roadmap_reason = _safe_dict(reason.get("roadmap"))
            explicit_plan_id = _to_int(roadmap_reason.get("plan_id"))
            attributed_plan_id: int | None = None
            attribution_kind = "none"

            if explicit_plan_id is not None:
                if explicit_plan_id in all_scope_plan_ids:
                    attributed_plan_id = explicit_plan_id
                    attribution_kind = "explicit_plan_id"
                else:
                    roadmap_assignment_out_of_scope += 1
                    assignment_state[assignment_id] = {"state": "out_of_scope"}
                    continue
            else:
                category_hint = str(
                    roadmap_reason.get("category")
                    or _safe_dict(reason.get("roadmap_ctx")).get("category")
                    or target.get("category")
                    or ""
                ).strip()
                product_type_hint = str(
                    roadmap_reason.get("next_product_type")
                    or _safe_dict(reason.get("roadmap_ctx")).get("next_product_type")
                    or target.get("product_type")
                    or ""
                ).strip()

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
                                abs((assigned_at - c["updated_at"]).total_seconds()),
                                int(c["id"]),
                            )
                            for c in candidates
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

            attributed_plan_id_int = int(attributed_plan_id)
            shortcut = _is_shortcut_target(target)
            if attributed_plan_id_int in cohort_scope_plan_ids:
                plan_metrics[attributed_plan_id_int]["offers_assigned_total"] += 1
                if shortcut:
                    plan_metrics[attributed_plan_id_int]["offers_with_roadmap_shortcut"] += 1

            cohort = cohort_by_plan_id.get(attributed_plan_id_int)
            if cohort not in {"model_used", "control"}:
                roadmap_assignment_excluded_non_cohort += 1
                assignment_state[assignment_id] = {
                    "state": "non_cohort",
                    "plan_id": attributed_plan_id_int,
                    "attribution_kind": attribution_kind,
                }
                continue

            roadmap_assignment_attributed += 1
            assignment_state[assignment_id] = {
                "state": "cohort",
                "cohort": cohort,
                "plan_id": attributed_plan_id_int,
                "attribution_kind": attribution_kind,
            }
            for section, key in plan_segments.get(attributed_plan_id_int, []):
                if section == "overall":
                    bucket = overall_buckets[cohort]
                elif section == "category":
                    bucket = by_category_buckets[key][cohort]
                else:
                    bucket = by_freshness_buckets[key][cohort]
                bucket["offers"]["assigned_total"] += 1
                if shortcut:
                    bucket["offers"]["shortcut_total"] += 1

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
            OfferEvent.Type.EXPOSED: "exposed",
            OfferEvent.Type.CLICKED: "clicked",
            OfferEvent.Type.REDEEMED: "redeemed",
        }
        for row in offer_qs.values("assignment_id", "event_type"):
            assignment_id = _to_int(row.get("assignment_id"))
            if assignment_id is None or assignment_id not in roadmap_assignment_ids:
                continue
            metric = offer_event_map.get(str(row.get("event_type") or ""))
            if not metric:
                continue
            state = assignment_state.get(assignment_id) or {}
            plan_id_for_metrics = _to_int(state.get("plan_id"))
            if plan_id_for_metrics is not None and int(plan_id_for_metrics) in cohort_scope_plan_ids:
                plan_metrics[int(plan_id_for_metrics)][f"offer_{metric}"] += 1
            if state.get("state") == "cohort":
                cohort = str(state.get("cohort"))
                plan_id = int(state.get("plan_id"))
                for section, key in plan_segments.get(plan_id, []):
                    if section == "overall":
                        bucket = overall_buckets[cohort]
                    elif section == "category":
                        bucket = by_category_buckets[key][cohort]
                    else:
                        bucket = by_freshness_buckets[key][cohort]
                    bucket["offers"][metric] += 1
            else:
                offer_unattributed_event_counts[metric] += 1

        model_final = _finalize_bucket(overall_buckets["model_used"])
        control_final = _finalize_bucket(overall_buckets["control"])

        by_category_final: dict[str, dict[str, Any]] = {}
        for cat in sorted(by_category_buckets.keys()):
            by_category_final[cat] = {
                "model_used": _finalize_bucket(by_category_buckets[cat]["model_used"]),
                "control": _finalize_bucket(by_category_buckets[cat]["control"]),
            }

        by_freshness_final = {
            key: {
                "model_used": _finalize_bucket(by_freshness_buckets[key]["model_used"]),
                "control": _finalize_bucket(by_freshness_buckets[key]["control"]),
            }
            for key in ["1d", "3d", "window"]
        }

        by_source_final = {
            source: {
                "model_used": _finalize_source_bucket(by_source_buckets[source]["model_used"]),
                "control": _finalize_source_bucket(by_source_buckets[source]["control"]),
            }
            for source in ["offers", "roadmap_api"]
        }

        def _bucket_from_plan_ids(plan_ids: set[int]) -> dict[str, Any]:
            bucket = _new_bucket()
            for pid in sorted(plan_ids):
                if pid not in cohort_scope_plan_ids:
                    continue
                uid = _to_int(plan_user.get(pid))
                cat = str(plan_category.get(pid) or "__unknown__")
                bucket["plans_total"] += 1
                if uid is not None:
                    bucket["user_ids"].add(uid)
                bucket["category_counts"][cat] += 1
                bucket["steps_total"] += int(plan_steps_total.get(pid, 0))
                bucket["step_status"].update(plan_step_status_counts.get(pid, Counter()))

                metrics = plan_metrics.get(pid, Counter())
                exposed = int(metrics.get("roadmap_step_exposed", 0))
                clicked = int(metrics.get("roadmap_step_clicked", 0))
                completed = int(metrics.get("roadmap_step_completed", 0))
                skipped = int(metrics.get("roadmap_step_skipped", 0))
                bucket["step_events"]["exposed"] += exposed
                bucket["step_events"]["clicked"] += clicked
                bucket["step_events"]["completed"] += completed
                bucket["step_events"]["skipped"] += skipped
                if exposed > 0:
                    bucket["exposed_plan_ids"].add(pid)
                bucket["exposed_step_ids"].update(plan_exposed_step_ids.get(pid, set()))

                bucket["offers"]["assigned_total"] += int(metrics.get("offers_assigned_total", 0))
                bucket["offers"]["shortcut_total"] += int(metrics.get("offers_with_roadmap_shortcut", 0))
                bucket["offers"]["exposed"] += int(metrics.get("offer_exposed", 0))
                bucket["offers"]["clicked"] += int(metrics.get("offer_clicked", 0))
                bucket["offers"]["redeemed"] += int(metrics.get("offer_redeemed", 0))
            return bucket

        def _uplift_from_pair(model_block: dict[str, Any], control_block: dict[str, Any]) -> dict[str, Any]:
            model_plans = int(model_block.get("plans_total", 0))
            control_plans = int(control_block.get("plans_total", 0))
            step_funnel = {
                "step_ctr": _build_uplift_metric(
                    metric="step_ctr",
                    model_success=int(model_block.get("roadmap_step_clicked", 0)),
                    model_total=int(model_block.get("roadmap_step_exposed", 0)),
                    control_success=int(control_block.get("roadmap_step_clicked", 0)),
                    control_total=int(control_block.get("roadmap_step_exposed", 0)),
                    model_plans=model_plans,
                    control_plans=control_plans,
                    min_plans=min_plans,
                ),
                "step_completion_rate": _build_uplift_metric(
                    metric="step_completion_rate",
                    model_success=int(model_block.get("roadmap_step_completed", 0)),
                    model_total=int(model_block.get("roadmap_step_exposed", 0)),
                    control_success=int(control_block.get("roadmap_step_completed", 0)),
                    control_total=int(control_block.get("roadmap_step_exposed", 0)),
                    model_plans=model_plans,
                    control_plans=control_plans,
                    min_plans=min_plans,
                ),
                "skip_rate": _build_uplift_metric(
                    metric="skip_rate",
                    model_success=int(model_block.get("roadmap_step_skipped", 0)),
                    model_total=int(model_block.get("roadmap_step_exposed", 0)),
                    control_success=int(control_block.get("roadmap_step_skipped", 0)),
                    control_total=int(control_block.get("roadmap_step_exposed", 0)),
                    model_plans=model_plans,
                    control_plans=control_plans,
                    min_plans=min_plans,
                ),
            }
            offer_funnel = {
                "offer_ctr": _build_uplift_metric(
                    metric="offer_ctr",
                    model_success=int(model_block.get("offer_clicked", 0)),
                    model_total=int(model_block.get("offer_exposed", 0)),
                    control_success=int(control_block.get("offer_clicked", 0)),
                    control_total=int(control_block.get("offer_exposed", 0)),
                    model_plans=model_plans,
                    control_plans=control_plans,
                    min_plans=min_plans,
                ),
                "offer_redeem_rate": _build_uplift_metric(
                    metric="offer_redeem_rate",
                    model_success=int(model_block.get("offer_redeemed", 0)),
                    model_total=int(model_block.get("offer_exposed", 0)),
                    control_success=int(control_block.get("offer_redeemed", 0)),
                    control_total=int(control_block.get("offer_exposed", 0)),
                    model_plans=model_plans,
                    control_plans=control_plans,
                    min_plans=min_plans,
                ),
            }
            return {"step_funnel": step_funnel, "offer_funnel": offer_funnel}

        overall_uplift = _uplift_from_pair(model_final, control_final)
        by_category_uplift: dict[str, Any] = {}
        for cat, block in by_category_final.items():
            by_category_uplift[cat] = _uplift_from_pair(
                _safe_dict(block.get("model_used")),
                _safe_dict(block.get("control")),
            )

        by_freshness_uplift: dict[str, Any] = {}
        for key in ["1d", "3d", "window"]:
            block = _safe_dict(by_freshness_final.get(key))
            by_freshness_uplift[key] = _uplift_from_pair(
                _safe_dict(block.get("model_used")),
                _safe_dict(block.get("control")),
            )

        by_source_uplift: dict[str, Any] = {}
        for source in ["offers", "roadmap_api"]:
            block = _safe_dict(by_source_final.get(source))
            model_block = _safe_dict(block.get("model_used"))
            control_block = _safe_dict(block.get("control"))
            model_plans = int(model_final.get("plans_total", 0))
            control_plans = int(control_final.get("plans_total", 0))
            by_source_uplift[source] = {
                "step_funnel": {
                    "step_ctr": _build_uplift_metric(
                        metric="step_ctr",
                        model_success=int(model_block.get("roadmap_step_clicked", 0)),
                        model_total=int(model_block.get("roadmap_step_exposed", 0)),
                        control_success=int(control_block.get("roadmap_step_clicked", 0)),
                        control_total=int(control_block.get("roadmap_step_exposed", 0)),
                        model_plans=model_plans,
                        control_plans=control_plans,
                        min_plans=min_plans,
                    ),
                    "step_completion_rate": _build_uplift_metric(
                        metric="step_completion_rate",
                        model_success=int(model_block.get("roadmap_step_completed", 0)),
                        model_total=int(model_block.get("roadmap_step_exposed", 0)),
                        control_success=int(control_block.get("roadmap_step_completed", 0)),
                        control_total=int(control_block.get("roadmap_step_exposed", 0)),
                        model_plans=model_plans,
                        control_plans=control_plans,
                        min_plans=min_plans,
                    ),
                    "skip_rate": _build_uplift_metric(
                        metric="skip_rate",
                        model_success=int(model_block.get("roadmap_step_skipped", 0)),
                        model_total=int(model_block.get("roadmap_step_exposed", 0)),
                        control_success=int(control_block.get("roadmap_step_skipped", 0)),
                        control_total=int(control_block.get("roadmap_step_exposed", 0)),
                        model_plans=model_plans,
                        control_plans=control_plans,
                        min_plans=min_plans,
                    ),
                },
                "offer_funnel": {
                    "offer_ctr": {
                        "metric": "offer_ctr",
                        "note": "not segmented by expose source",
                    },
                    "offer_redeem_rate": {
                        "metric": "offer_redeem_rate",
                        "note": "not segmented by expose source",
                    },
                },
            }

        partial_makeup_canary_final = _finalize_bucket(
            _bucket_from_plan_ids(partial_makeup_canary_plan_ids)
        )
        partial_makeup_control_final = _finalize_bucket(
            _bucket_from_plan_ids(partial_makeup_control_plan_ids)
        )
        partial_makeup_overall_uplift = _uplift_from_pair(
            partial_makeup_canary_final,
            partial_makeup_control_final,
        )
        partial_makeup_product_types = sorted(
            {
                pt
                for pid in partial_makeup_analysis_plan_ids
                for pt in plan_step_types.get(pid, set())
            }
        )
        partial_makeup_step_indexes = sorted(
            {
                idx
                for pid in partial_makeup_analysis_plan_ids
                for idx in plan_step_index_buckets.get(pid, set())
            }
        )

        partial_makeup_by_product_type: dict[str, Any] = {}
        for pt in partial_makeup_product_types:
            canary_ids = {
                pid
                for pid in partial_makeup_canary_plan_ids
                if pt in plan_step_types.get(pid, set())
            }
            control_ids = {
                pid
                for pid in partial_makeup_control_plan_ids
                if pt in plan_step_types.get(pid, set())
            }
            canary_block = _finalize_bucket(_bucket_from_plan_ids(canary_ids))
            control_block = _finalize_bucket(_bucket_from_plan_ids(control_ids))
            partial_makeup_by_product_type[pt] = {
                "canary": canary_block,
                "control": control_block,
                "uplift": _uplift_from_pair(canary_block, control_block),
            }

        partial_makeup_by_step_index: dict[str, Any] = {}
        for idx_bucket in partial_makeup_step_indexes:
            canary_ids = {
                pid
                for pid in partial_makeup_canary_plan_ids
                if idx_bucket in plan_step_index_buckets.get(pid, set())
            }
            control_ids = {
                pid
                for pid in partial_makeup_control_plan_ids
                if idx_bucket in plan_step_index_buckets.get(pid, set())
            }
            canary_block = _finalize_bucket(_bucket_from_plan_ids(canary_ids))
            control_block = _finalize_bucket(_bucket_from_plan_ids(control_ids))
            partial_makeup_by_step_index[idx_bucket] = {
                "canary": canary_block,
                "control": control_block,
                "uplift": _uplift_from_pair(canary_block, control_block),
            }

        partial_makeup_uplift = {
            "definition": {
                "canary": "category=makeup and rollout_mode=partial and rollout_selected=true",
                "control": "category=makeup and decision in {fallback,disabled} and not in canary",
            },
            "overall": {
                "canary": partial_makeup_canary_final,
                "control": partial_makeup_control_final,
                "uplift": partial_makeup_overall_uplift,
            },
            "by_product_type": partial_makeup_by_product_type,
            "by_step_index": partial_makeup_by_step_index,
        }

        runtime_observability = {
            "decision_counts": {
                "model_used": int(decision_counts.get("model_used", 0)),
                "fallback": int(decision_counts.get("fallback", 0)),
                "disabled": int(decision_counts.get("disabled", 0)),
                "missing_ml_meta": int(decision_counts.get("missing_ml_meta", 0)),
            },
            "rollout_mode_distribution": {
                str(k): int(v)
                for k, v in sorted(rollout_mode_counts.items(), key=lambda kv: (-kv[1], kv[0]))
            },
            "rollout_selected_distribution": {
                str(k): int(v)
                for k, v in sorted(rollout_selected_counts.items(), key=lambda kv: (-kv[1], kv[0]))
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
            "top_model_paths": [
                {"model_path": str(k), "count": int(v)}
                for k, v in model_path_counts.most_common(10)
            ],
        }

        guard_eval_report = {
            "params": {
                "cohort_mode": "fresh",
                "control": "non_model",
            },
            "breakdowns": {
                "by_category": by_category_final,
            },
            "uplift": {
                "by_category": by_category_uplift,
            },
        }
        rollout_categories = ["skincare", "haircare", "makeup", "fragrance"]
        if category != "all":
            rollout_categories = [category]
        else:
            rollout_categories = sorted(set(rollout_categories) | set(by_category_final.keys()))
        rollout_recommendation_by_category: dict[str, Any] = {}
        for cat in rollout_categories:
            rollout = v4_category_rollout_status(cat)
            guard = v4_category_uplift_guard_status_from_report(
                cat,
                guard_eval_report,
                report_path=f"inline://report_roadmap_ml_uplift?days={days}&category={category}",
            )
            if not bool(rollout.get("passed")):
                recommendation = "DISABLE"
            elif bool(guard.get("passed")):
                recommendation = "ENABLE"
            else:
                recommendation = "HOLD"
            rollout_recommendation_by_category[cat] = {
                "recommendation": recommendation,
                "rollout": rollout,
                "guard": guard,
            }

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
                "control": control,
                "min_plans": min_plans,
                "sync_runtime_artifact": sync_runtime_artifact,
            },
            "overall": {
                "plans_total_in_scope": len(all_scope_plan_ids),
                "plans_total_after_cohort_mode": len(cohort_scope_plan_ids),
                "analysis_plans_total": len(analysis_plan_ids),
                "model_used_plans_total": len(model_plan_ids),
                "control_plans_total": len(control_plan_ids),
            },
            "cohorts": {
                "model_used": model_final,
                "control": control_final,
            },
            "uplift": {
                "overall": overall_uplift,
                "by_category": by_category_uplift,
                "by_freshness": by_freshness_uplift,
                "by_expose_source": by_source_uplift,
            },
            "partial_makeup_uplift": partial_makeup_uplift,
            "breakdowns": {
                "by_category": by_category_final,
                "by_freshness": by_freshness_final,
                "by_expose_source": by_source_final,
            },
            "runtime_observability": runtime_observability,
            "rollout_recommendation_by_category": rollout_recommendation_by_category,
            "unattributed_excluded": {
                "fresh_mode_excluded_missing_ml_meta_plans": int(fresh_excluded_missing),
                "cohort_scope_excluded_non_selected_plans": int(excluded_non_selected),
                "roadmap_assignments_total": int(roadmap_assignment_total),
                "roadmap_assignments_attributed_to_cohorts": int(roadmap_assignment_attributed),
                "roadmap_assignments_unattributed": int(roadmap_assignment_unattributed),
                "roadmap_assignments_out_of_scope_plan": int(roadmap_assignment_out_of_scope),
                "roadmap_assignments_excluded_non_cohort": int(roadmap_assignment_excluded_non_cohort),
                "roadmap_offer_events_unattributed_exposed": int(offer_unattributed_event_counts.get("exposed", 0)),
                "roadmap_offer_events_unattributed_clicked": int(offer_unattributed_event_counts.get("clicked", 0)),
                "roadmap_offer_events_unattributed_redeemed": int(offer_unattributed_event_counts.get("redeemed", 0)),
            },
            "notes": [
                "Read-only wrt DB: no DB writes; file outputs happen only via --out or --sync-runtime-artifact.",
                "By default cohort-mode=fresh excludes missing_ml_meta from active uplift comparison.",
                "Historical tail is still visible in runtime_observability decision counts.",
                "Offer attribution is conservative: explicit roadmap.plan_id first, fallback only when match is reliable.",
                "Unattributed roadmap assignments/events are reported separately and never forced into cohorts.",
                "Source breakdown uses expose-source tagging; interactions are attributed by exposed step source mapping.",
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

        if sync_runtime_artifact and runtime_window is not None:
            runtime_json_path = v4_runtime_uplift_report_path(runtime_window)
            runtime_json_path.parent.mkdir(parents=True, exist_ok=True)
            runtime_json_path.write_text(json_text, encoding="utf-8")
            self.stderr.write(f"[report_roadmap_ml_uplift] synced runtime artifact: {runtime_json_path}")

        for p in wrote_paths:
            self.stdout.write(f"[report_roadmap_ml_uplift] wrote: {p}")
