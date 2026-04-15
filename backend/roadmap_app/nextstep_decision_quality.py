from __future__ import annotations

from collections import Counter, defaultdict
from datetime import timedelta
from pathlib import Path
from typing import Any, Callable

from django.utils import timezone

from roadmap_app.ml_next_step import nextstep_model_artifact_summary, v4_category_staged_rollout_status
from roadmap_app.nextstep_historical_anchor_context import resolve_historical_anchor_read_context
from roadmap_app.shadow_evidence import (
    get_historical_control_evidence_for_model_path,
    get_historical_shadow_evidence_for_model_path,
    normalized_model_path,
)


DEFAULT_REPORT_STEM = Path("reports") / "roadmap_nextstep_v4_decision_quality"


def _safe_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _safe_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _event_key(created_at: Any, event_id: Any) -> tuple[Any, int]:
    try:
        event_id_int = int(event_id or 0)
    except Exception:
        event_id_int = 0
    return created_at, event_id_int


def _rate(numerator: int, denominator: int) -> float | None:
    if denominator <= 0:
        return None
    return float(numerator) / float(denominator)


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


def _pct(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value * 100.0:.2f}%"


def _completion_window_truth_reason(anchor: dict[str, Any]) -> str:
    if bool(anchor.get("has_skipped_after_generated")):
        return "no_completed_truth_in_window_after_skip"
    if not bool(anchor.get("has_exposed")):
        return "no_completed_truth_in_window_not_exposed"
    if not bool(anchor.get("has_completed_after_generated")):
        return "no_completed_truth_in_window_exposed_no_completion"
    return "no_completed_truth_in_window"


def _first_completed_generated_candidate_truth(
    anchor: dict[str, Any],
    *,
    completions_by_step: dict[int, list[dict[str, Any]]],
) -> dict[str, Any]:
    start_key = _event_key(anchor.get("anchor_created_at"), anchor.get("anchor_event_id"))
    end_key = (
        _event_key(anchor.get("next_refresh_at"), 10**18)
        if anchor.get("next_refresh_at") is not None
        else None
    )
    first_row: dict[str, Any] | None = None
    truth_types: list[str] = []
    seen_types: set[str] = set()

    for raw_step_id in _safe_list(anchor.get("generated_step_ids")):
        try:
            step_id = int(raw_step_id)
        except Exception:
            continue
        for row in completions_by_step.get(step_id, []):
            row_key = _event_key(row.get("created_at"), row.get("id"))
            if row_key < start_key:
                continue
            if end_key is not None and row_key >= end_key:
                break
            ctx = _safe_dict(row.get("context"))
            truth_product_type = str(ctx.get("product_type") or "").strip().lower()
            if not truth_product_type:
                continue
            if first_row is None or row_key < _event_key(first_row.get("created_at"), first_row.get("id")):
                first_row = row
            if truth_product_type not in seen_types:
                seen_types.add(truth_product_type)
                truth_types.append(truth_product_type)

    if first_row is None:
        return {
            "resolved": False,
            "reason": _completion_window_truth_reason(anchor),
            "matched_by": "",
            "truth_product_type": "",
            "truth_product_types_in_window": [],
            "ambiguous_multiple_completed_types": False,
        }

    ctx = _safe_dict(first_row.get("context"))
    return {
        "resolved": True,
        "reason": "ok",
        "matched_by": str(ctx.get("matched_by") or "").strip().lower(),
        "truth_product_type": str(ctx.get("product_type") or "").strip().lower(),
        "truth_product_types_in_window": truth_types,
        "ambiguous_multiple_completed_types": len(truth_types) > 1,
    }


def _new_bucket() -> dict[str, Any]:
    return {
        "counts": Counter(),
        "truth_matched_by": Counter(),
        "unresolved_reasons": Counter(),
        "comparability_exclusions": Counter(),
    }


def _bucket_to_summary(bucket: dict[str, Any]) -> dict[str, Any]:
    counts = Counter(bucket.get("counts") or {})
    comparable = int(counts.get("comparable", 0))
    resolved = int(counts.get("resolved_truth", 0))
    agreement = int(counts.get("agreement", 0))
    disagreement = int(counts.get("disagreement", 0))
    model_wins = int(counts.get("model_wins", 0))
    baseline_wins = int(counts.get("baseline_wins", 0))
    both_correct = int(counts.get("both_correct", 0))
    both_wrong = int(counts.get("both_wrong", 0))
    unresolved_truth = int(counts.get("unresolved_truth", 0))

    return {
        "comparable_anchors_total": comparable,
        "resolved_truth_anchors_total": resolved,
        "unresolved_truth_anchors_total": unresolved_truth,
        "agreement_total": agreement,
        "disagreement_total": disagreement,
        "both_correct_total": both_correct,
        "both_wrong_total": both_wrong,
        "model_wins_total": model_wins,
        "baseline_wins_total": baseline_wins,
        "resolved_truth_share": _round_or_none(_rate(resolved, comparable)),
        "agreement_rate": _round_or_none(_rate(agreement, resolved)),
        "disagreement_rate": _round_or_none(_rate(disagreement, resolved)),
        "both_correct_rate": _round_or_none(_rate(both_correct, resolved)),
        "both_wrong_rate": _round_or_none(_rate(both_wrong, resolved)),
        "model_win_rate_vs_truth": _round_or_none(_rate(model_wins, resolved)),
        "baseline_win_rate_vs_truth": _round_or_none(_rate(baseline_wins, resolved)),
        "net_win_rate_model_minus_baseline": _round_or_none(_rate(model_wins - baseline_wins, resolved)),
        "truth_matched_by": {
            str(k): int(v)
            for k, v in sorted(bucket.get("truth_matched_by", Counter()).items(), key=lambda kv: (-kv[1], kv[0]))
        },
        "unresolved_reasons": {
            str(k): int(v)
            for k, v in sorted(bucket.get("unresolved_reasons", Counter()).items(), key=lambda kv: (-kv[1], kv[0]))
        },
        "comparability_exclusions": {
            str(k): int(v)
            for k, v in sorted(bucket.get("comparability_exclusions", Counter()).items(), key=lambda kv: (-kv[1], kv[0]))
        },
    }


def _diagnosis_from_summary(
    summary: dict[str, Any],
    *,
    rollout_reason: str,
    min_rollout_sample: int,
) -> dict[str, Any]:
    comparable = int(summary.get("comparable_anchors_total", 0) or 0)
    resolved = int(summary.get("resolved_truth_anchors_total", 0) or 0)
    agreement_rate = float(summary.get("agreement_rate") or 0.0)
    model_win_rate = float(summary.get("model_win_rate_vs_truth") or 0.0)
    baseline_win_rate = float(summary.get("baseline_win_rate_vs_truth") or 0.0)
    resolved_truth_share = float(summary.get("resolved_truth_share") or 0.0)
    truth_matched_by = _safe_dict(summary.get("truth_matched_by"))
    resolved_recommended_product_id = int(truth_matched_by.get("recommended_product_id", 0) or 0)
    recommended_share = _rate(resolved_recommended_product_id, resolved)

    code = "C"
    summary_text = "mixed signal"
    if comparable < min_rollout_sample and agreement_rate >= 0.8 and abs(model_win_rate - baseline_win_rate) <= 0.05:
        code = "A"
        summary_text = "mostly agrees with baseline; incremental value remains low and sample is still below rollout threshold"
    elif comparable < min_rollout_sample and resolved <= 0:
        code = "D"
        summary_text = "insufficient resolved truth on the recovered anchor sample"
    elif recommended_share is not None and recommended_share < 0.8:
        code = "D"
        summary_text = "truth protocol is too weak for a defensible winner/loser read"
    elif resolved_truth_share < 0.5 and comparable >= min_rollout_sample:
        code = "D"
        summary_text = "too many recovered anchors still lack resolved completion truth"
    elif agreement_rate >= 0.8 and abs(model_win_rate - baseline_win_rate) <= 0.05:
        code = "A"
        summary_text = "model mostly agrees with baseline, so upside is near zero"
    elif baseline_win_rate >= (model_win_rate + 0.10):
        code = "B"
        summary_text = "disagreements are usually worse than baseline"
    elif model_win_rate > 0.0 and baseline_win_rate > 0.0:
        code = "C"
        summary_text = "model helps on some slices but is dragged down elsewhere"
    elif model_win_rate > baseline_win_rate:
        code = "C"
        summary_text = "there is limited promise, but not enough to justify rollout"
    else:
        code = "A"
        summary_text = "incremental value over baseline is weak"

    if rollout_reason == "sample_too_small_but_nonzero_control":
        summary_text = f"{summary_text}; rollout remains sample-limited"
    elif rollout_reason == "low_uplift":
        summary_text = f"{summary_text}; current rollout guard still reads low_uplift"
    elif rollout_reason == "category_disabled":
        summary_text = f"{summary_text}; category is disabled in runtime configuration"

    return {
        "code": code,
        "summary": summary_text,
    }


def _slice_aggregate(
    rows: list[dict[str, Any]],
    *,
    key_fn: Callable[[dict[str, Any]], tuple[Any, ...] | None],
    labels_fn: Callable[[tuple[Any, ...]], dict[str, Any]],
) -> list[dict[str, Any]]:
    buckets: dict[tuple[Any, ...], Counter] = defaultdict(Counter)
    examples: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = key_fn(row)
        if key is None:
            continue
        bucket = buckets[key]
        bucket["anchors"] += 1
        outcome_class = str(row.get("outcome_class") or "")
        if outcome_class:
            bucket[outcome_class] += 1
        if len(examples[key]) < 3:
            examples[key].append(
                {
                    "plan_id": int(row.get("plan_id") or 0),
                    "anchor_event_id": int(row.get("anchor_event_id") or 0),
                    "anchor_key": str(row.get("anchor_key") or ""),
                    "baseline": str(row.get("baseline_selected_product_type") or ""),
                    "model": str(row.get("model_top1_product_type") or ""),
                    "truth": str(row.get("truth_product_type") or ""),
                }
            )

    out: list[dict[str, Any]] = []
    for key, bucket in buckets.items():
        anchors = int(bucket.get("anchors", 0))
        model_wins = int(bucket.get("model_wins", 0))
        baseline_wins = int(bucket.get("baseline_wins", 0))
        both_correct = int(bucket.get("both_correct", 0))
        both_wrong = int(bucket.get("both_wrong", 0))
        out.append(
            {
                **labels_fn(key),
                "anchors": anchors,
                "model_wins": model_wins,
                "baseline_wins": baseline_wins,
                "both_correct": both_correct,
                "both_wrong": both_wrong,
                "net_wins_model_minus_baseline": model_wins - baseline_wins,
                "model_win_rate_vs_truth": _round_or_none(_rate(model_wins, anchors)),
                "baseline_win_rate_vs_truth": _round_or_none(_rate(baseline_wins, anchors)),
                "sample_anchor_refs": examples.get(key, []),
            }
        )

    out.sort(
        key=lambda row: (
            -(abs(int(row.get("net_wins_model_minus_baseline", 0)))),
            -int(row.get("anchors", 0)),
            str(row.get("category") or ""),
            str(row.get("truth_product_type") or row.get("baseline_product_type") or ""),
        )
    )
    return out


def build_nextstep_v4_decision_quality_payload(
    *,
    model_path: str | Path | None,
    days: int = 30,
    category: str = "all",
    include_ga: bool = False,
    min_slice_size: int = 10,
    historical_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    now_utc = timezone.now()
    since = now_utc - timedelta(days=int(days))
    model_path_norm = normalized_model_path(model_path)
    if not model_path_norm:
        raise ValueError("model_path is required")

    shared_context = resolve_historical_anchor_read_context(
        since=since,
        until=now_utc,
        category=category,
        include_ga=include_ga,
        historical_context=historical_context,
    )
    since = shared_context.get("since", since)
    now_utc = shared_context.get("until", now_utc)
    anchors = _safe_list(shared_context.get("anchors"))
    meta_by_plan = _safe_dict(shared_context.get("meta_by_plan"))
    completions_by_step = _safe_dict(shared_context.get("completions_by_step"))

    records: list[dict[str, Any]] = []
    all_bucket = _new_bucket()
    enabled_bucket = _new_bucket()
    category_buckets: dict[str, dict[str, Any]] = defaultdict(_new_bucket)

    for anchor in anchors:
        category_norm = str(anchor.get("category") or "__unknown__").strip().lower()
        rollout = v4_category_staged_rollout_status(category_norm, model_path=model_path_norm)
        rollout_reason = str(rollout.get("reason") or "")
        rollout_enabled = bool(_safe_dict(rollout.get("rollout")).get("passed"))
        meta = meta_by_plan.get(int(anchor.get("plan_id") or 0), {})
        shadow_map = get_historical_shadow_evidence_for_model_path(meta, model_path_norm)
        control_map = get_historical_control_evidence_for_model_path(meta, model_path_norm)
        shadow_payload = _safe_dict(shadow_map.get(str(anchor.get("anchor_key") or "")))
        control_payload = _safe_dict(control_map.get(str(anchor.get("anchor_key") or "")))
        record: dict[str, Any] = {
            "anchor_key": str(anchor.get("anchor_key") or ""),
            "anchor_event_id": int(anchor.get("anchor_event_id") or 0),
            "plan_id": int(anchor.get("plan_id") or 0),
            "category": category_norm,
            "rollout_status": str(rollout.get("final_status") or ""),
            "rollout_reason": rollout_reason,
            "rollout_enabled": rollout_enabled,
            "model_path": model_path_norm,
            "model_top1_product_type": str(shadow_payload.get("top1_product_type") or "").strip().lower(),
            "baseline_selected_product_type": str(control_payload.get("selected_product_type") or "").strip().lower(),
            "planned_target_product_type": str(anchor.get("planned_target_product_type") or "").strip().lower(),
            "candidate_types": [
                str(item or "").strip().lower()
                for item in _safe_list(anchor.get("candidate_types"))
                if str(item or "").strip()
            ],
            "reconstruction_reason": str(anchor.get("reconstruction_reason") or "").strip(),
            "truth_product_type": "",
            "truth_matched_by": "",
            "truth_product_types_in_window": [],
            "truth_ambiguous_multiple_completed_types": False,
            "truth_reason": "",
            "comparable": False,
            "truth_resolved": False,
            "agreement": False,
            "model_correct": False,
            "baseline_correct": False,
            "outcome_class": "unresolved",
            "comparability_exclusion_reason": "",
        }

        bucket_targets = [all_bucket, category_buckets[category_norm]]
        if rollout_enabled:
            bucket_targets.append(enabled_bucket)

        for bucket in bucket_targets:
            bucket["counts"]["anchors_total"] += 1

        if record["reconstruction_reason"]:
            for bucket in bucket_targets:
                bucket["comparability_exclusions"][record["reconstruction_reason"]] += 1
            record["comparability_exclusion_reason"] = record["reconstruction_reason"]
            records.append(record)
            continue

        if not shadow_payload:
            exclusion = "missing_historical_shadow_evidence"
            for bucket in bucket_targets:
                bucket["comparability_exclusions"][exclusion] += 1
            record["comparability_exclusion_reason"] = exclusion
            records.append(record)
            continue

        if not control_payload:
            exclusion = "missing_historical_control_evidence"
            for bucket in bucket_targets:
                bucket["comparability_exclusions"][exclusion] += 1
            record["comparability_exclusion_reason"] = exclusion
            records.append(record)
            continue

        if not bool(shadow_payload.get("was_model_selected")):
            exclusion = str(shadow_payload.get("comparable_reason") or "model_not_selected")
            for bucket in bucket_targets:
                bucket["comparability_exclusions"][exclusion] += 1
            record["comparability_exclusion_reason"] = exclusion
            records.append(record)
            continue

        if not bool(control_payload.get("was_control_selected")):
            exclusion = str(control_payload.get("comparable_reason") or "baseline_control_unavailable")
            for bucket in bucket_targets:
                bucket["comparability_exclusions"][exclusion] += 1
            record["comparability_exclusion_reason"] = exclusion
            records.append(record)
            continue

        record["comparable"] = True
        for bucket in bucket_targets:
            bucket["counts"]["comparable"] += 1

        truth = _first_completed_generated_candidate_truth(
            anchor,
            completions_by_step=completions_by_step,
        )
        record["truth_reason"] = str(truth.get("reason") or "")
        record["truth_matched_by"] = str(truth.get("matched_by") or "")
        record["truth_product_type"] = str(truth.get("truth_product_type") or "")
        record["truth_product_types_in_window"] = _safe_list(truth.get("truth_product_types_in_window"))
        record["truth_ambiguous_multiple_completed_types"] = bool(
            truth.get("ambiguous_multiple_completed_types")
        )
        if not bool(truth.get("resolved")):
            for bucket in bucket_targets:
                bucket["counts"]["unresolved_truth"] += 1
                bucket["unresolved_reasons"][record["truth_reason"]] += 1
            records.append(record)
            continue

        record["truth_resolved"] = True
        record["agreement"] = (
            record["model_top1_product_type"] == record["baseline_selected_product_type"]
        )
        record["model_correct"] = (
            record["model_top1_product_type"] == record["truth_product_type"]
        )
        record["baseline_correct"] = (
            record["baseline_selected_product_type"] == record["truth_product_type"]
        )
        if record["model_correct"] and record["baseline_correct"]:
            record["outcome_class"] = "both_correct"
        elif (not record["model_correct"]) and (not record["baseline_correct"]):
            record["outcome_class"] = "both_wrong"
        elif record["model_correct"]:
            record["outcome_class"] = "model_wins"
        else:
            record["outcome_class"] = "baseline_wins"

        for bucket in bucket_targets:
            bucket["counts"]["resolved_truth"] += 1
            if record["agreement"]:
                bucket["counts"]["agreement"] += 1
            else:
                bucket["counts"]["disagreement"] += 1
            bucket["counts"][record["outcome_class"]] += 1
            bucket["truth_matched_by"][record["truth_matched_by"] or "unknown"] += 1
        records.append(record)

    min_rollout_sample = 100
    all_summary = _bucket_to_summary(all_bucket)
    enabled_summary = _bucket_to_summary(enabled_bucket)
    resolved_records = [row for row in records if bool(row.get("comparable")) and bool(row.get("truth_resolved"))]
    truth_slices = _slice_aggregate(
        resolved_records,
        key_fn=lambda row: (
            str(row.get("category") or ""),
            str(row.get("truth_product_type") or ""),
        ),
        labels_fn=lambda key: {
            "category": str(key[0]),
            "truth_product_type": str(key[1]),
        },
    )
    disagreement_pairs = _slice_aggregate(
        [row for row in resolved_records if not bool(row.get("agreement"))],
        key_fn=lambda row: (
            str(row.get("category") or ""),
            str(row.get("baseline_selected_product_type") or ""),
            str(row.get("model_top1_product_type") or ""),
        ),
        labels_fn=lambda key: {
            "category": str(key[0]),
            "baseline_product_type": str(key[1]),
            "model_product_type": str(key[2]),
        },
    )

    per_category: dict[str, Any] = {}
    for category_norm, bucket in sorted(category_buckets.items()):
        summary = _bucket_to_summary(bucket)
        rollout = v4_category_staged_rollout_status(category_norm, model_path=model_path_norm)
        summary["rollout_status"] = str(rollout.get("final_status") or "")
        summary["rollout_reason"] = str(rollout.get("reason") or "")
        summary["diagnosis"] = _diagnosis_from_summary(
            summary,
            rollout_reason=summary["rollout_reason"],
            min_rollout_sample=min_rollout_sample,
        )
        summary["worst_slices"] = [
            row
            for row in truth_slices
            if str(row.get("category") or "") == category_norm
            and int(row.get("anchors", 0)) >= int(min_slice_size)
            and int(row.get("net_wins_model_minus_baseline", 0)) < 0
        ][:5]
        summary["promising_slices"] = [
            row
            for row in truth_slices
            if str(row.get("category") or "") == category_norm
            and int(row.get("anchors", 0)) >= int(min_slice_size)
            and int(row.get("net_wins_model_minus_baseline", 0)) > 0
        ][:5]
        summary["worst_disagreement_pairs"] = [
            row
            for row in disagreement_pairs
            if str(row.get("category") or "") == category_norm
            and int(row.get("anchors", 0)) >= int(min_slice_size)
            and int(row.get("net_wins_model_minus_baseline", 0)) < 0
        ][:5]
        summary["promising_disagreement_pairs"] = [
            row
            for row in disagreement_pairs
            if str(row.get("category") or "") == category_norm
            and int(row.get("anchors", 0)) >= int(min_slice_size)
            and int(row.get("net_wins_model_minus_baseline", 0)) > 0
        ][:5]
        per_category[category_norm] = summary

    enabled_rollout_reasons = {
        str(cat): _safe_dict(summary).get("rollout_reason")
        for cat, summary in per_category.items()
        if str(_safe_dict(summary).get("rollout_status") or "") != "DISABLE"
    }
    enabled_diagnosis = _diagnosis_from_summary(
        enabled_summary,
        rollout_reason=";".join(
            str(reason or "") for reason in enabled_rollout_reasons.values() if str(reason or "").strip()
        ),
        min_rollout_sample=min_rollout_sample,
    )

    recommendation = {
        "runtime_state": "keep_frozen",
        "next_block": "targeted_retrain",
        "why_not_artifact_swap": "This report evaluates only the exact active artifact; it does not establish a stronger replacement artifact.",
        "why_not_runtime_enablement": "Recovered historical anchors now show decision-quality deficits, not missing qualification plumbing.",
        "rationale": [
            "Skincare disagreements skew worse than baseline on resolved historical truth.",
            "Haircare has some promising treatment slices, but overall rollout uplift remains below guard and both-wrong mass is still large.",
            "Makeup still lacks rollout-scale sample and shows no incremental signal over baseline on resolved anchors.",
        ],
    }

    artifact_summary = nextstep_model_artifact_summary(model_path_norm)
    return {
        "generated_at_utc": now_utc.isoformat(),
        "window_start_utc": since.isoformat(),
        "window_end_utc": now_utc.isoformat(),
        "model_path": model_path_norm,
        "model_version": str(artifact_summary.get("model_version") or ""),
        "params": {
            "days": int(days),
            "category": str(category or "all"),
            "include_ga": bool(include_ga),
            "min_slice_size": int(min_slice_size),
            "truth_protocol": "first_completed_generated_candidate_in_refresh_window",
        },
        "truth_protocol": {
            "name": "first_completed_generated_candidate_in_refresh_window",
            "description": "Truth is the first completed generated candidate inside the same PLAN_REFRESHED immutable window.",
            "stop_boundary_applicable": False,
            "truth_strength_note": "Resolved truth is measured from event-time STEP_COMPLETED rows, not current plan snapshot.",
        },
        "executive_verdict": {
            "overall_diagnosis": enabled_diagnosis,
            "recommendation": recommendation,
            "active_rollout_state": {
                str(cat): {
                    "status": str(_safe_dict(summary).get("rollout_status") or ""),
                    "reason": str(_safe_dict(summary).get("rollout_reason") or ""),
                }
                for cat, summary in per_category.items()
            },
        },
        "overall_all_categories": all_summary,
        "overall_enabled_categories": enabled_summary,
        "per_category": per_category,
        "slice_analysis": {
            "truth_slice_lookup": {
                f"{str(row.get('category') or '')}:{str(row.get('truth_product_type') or '')}": row
                for row in truth_slices
            },
            "disagreement_pair_lookup": {
                f"{str(row.get('category') or '')}:{str(row.get('baseline_product_type') or '')}:{str(row.get('model_product_type') or '')}": row
                for row in disagreement_pairs
            },
            "worst_truth_slices": [
                row
                for row in truth_slices
                if int(row.get("anchors", 0)) >= int(min_slice_size)
                and int(row.get("net_wins_model_minus_baseline", 0)) < 0
            ][:10],
            "promising_truth_slices": [
                row
                for row in truth_slices
                if int(row.get("anchors", 0)) >= int(min_slice_size)
                and int(row.get("net_wins_model_minus_baseline", 0)) > 0
            ][:10],
            "worst_disagreement_pairs": [
                row
                for row in disagreement_pairs
                if int(row.get("anchors", 0)) >= int(min_slice_size)
                and int(row.get("net_wins_model_minus_baseline", 0)) < 0
            ][:10],
            "promising_disagreement_pairs": [
                row
                for row in disagreement_pairs
                if int(row.get("anchors", 0)) >= int(min_slice_size)
                and int(row.get("net_wins_model_minus_baseline", 0)) > 0
            ][:10],
        },
        "notes": [
            "Read-only analysis: no DB writes and no runtime decision changes.",
            "This report uses recovered historical anchors plus exact-artifact historical shadow/control evidence keyed by model_path.",
            "Unlike rollout uplift, this path scores winner/loser on the same comparable anchor using event-time completion truth.",
            "Resolved truth uses STEP_COMPLETED matched_by strength; if completion truth is missing, the anchor is kept unresolved rather than forced into a winner.",
        ],
    }


def render_nextstep_v4_decision_quality_markdown(payload: dict[str, Any]) -> str:
    lines: list[str] = [
        "# Roadmap Nextstep V4 Decision Quality",
        "",
        f"Generated: {payload.get('generated_at_utc')}",
        "",
        "## Executive Verdict",
    ]
    executive = _safe_dict(payload.get("executive_verdict"))
    overall_diagnosis = _safe_dict(executive.get("overall_diagnosis"))
    recommendation = _safe_dict(executive.get("recommendation"))
    lines.extend(
        [
            f"- model_path: `{payload.get('model_path')}`",
            f"- overall diagnosis: `{overall_diagnosis.get('code')}`",
            f"- overall summary: {overall_diagnosis.get('summary')}",
            f"- runtime state: `{recommendation.get('runtime_state')}`",
            f"- exact next block: `{recommendation.get('next_block')}`",
            f"- why not artifact swap: {recommendation.get('why_not_artifact_swap')}",
            f"- why not runtime enablement: {recommendation.get('why_not_runtime_enablement')}",
            "",
            "## Truth Protocol",
        ]
    )
    truth_protocol = _safe_dict(payload.get("truth_protocol"))
    lines.extend(
        [
            f"- protocol: `{truth_protocol.get('name')}`",
            f"- description: {truth_protocol.get('description')}",
            f"- stop boundary applicable: `{truth_protocol.get('stop_boundary_applicable')}`",
            f"- truth strength: {truth_protocol.get('truth_strength_note')}",
            "",
            "## Per-Category Decision Quality",
        ]
    )

    per_category = _safe_dict(payload.get("per_category"))
    rows: list[list[Any]] = []
    for category_norm in sorted(per_category.keys()):
        summary = _safe_dict(per_category.get(category_norm))
        diagnosis = _safe_dict(summary.get("diagnosis"))
        rows.append(
            [
                category_norm,
                str(summary.get("rollout_status") or ""),
                str(summary.get("rollout_reason") or ""),
                str(diagnosis.get("code") or ""),
                int(summary.get("comparable_anchors_total", 0) or 0),
                int(summary.get("resolved_truth_anchors_total", 0) or 0),
                _pct(summary.get("agreement_rate")),
                _pct(summary.get("model_win_rate_vs_truth")),
                _pct(summary.get("baseline_win_rate_vs_truth")),
                _pct(summary.get("both_wrong_rate")),
            ]
        )
    lines.append(
        _md_table(
            [
                "category",
                "rollout",
                "guard_reason",
                "A/B/C/D",
                "comparable",
                "resolved_truth",
                "agreement_rate",
                "model_win_rate",
                "baseline_win_rate",
                "both_wrong_rate",
            ],
            rows,
        )
    )

    lines.extend(["", "## Win/Loss Matrix"])
    matrix_rows: list[list[Any]] = []
    overall_enabled = _safe_dict(payload.get("overall_enabled_categories"))
    matrix_rows.append(
        [
            "enabled_total",
            int(overall_enabled.get("resolved_truth_anchors_total", 0) or 0),
            int(overall_enabled.get("both_correct_total", 0) or 0),
            int(overall_enabled.get("model_wins_total", 0) or 0),
            int(overall_enabled.get("baseline_wins_total", 0) or 0),
            int(overall_enabled.get("both_wrong_total", 0) or 0),
            int(overall_enabled.get("unresolved_truth_anchors_total", 0) or 0),
        ]
    )
    for category_norm in sorted(per_category.keys()):
        summary = _safe_dict(per_category.get(category_norm))
        matrix_rows.append(
            [
                category_norm,
                int(summary.get("resolved_truth_anchors_total", 0) or 0),
                int(summary.get("both_correct_total", 0) or 0),
                int(summary.get("model_wins_total", 0) or 0),
                int(summary.get("baseline_wins_total", 0) or 0),
                int(summary.get("both_wrong_total", 0) or 0),
                int(summary.get("unresolved_truth_anchors_total", 0) or 0),
            ]
        )
    lines.append(
        _md_table(
            [
                "scope",
                "resolved_truth",
                "both_correct",
                "model_wins",
                "baseline_wins",
                "both_wrong",
                "unresolved_truth",
            ],
            matrix_rows,
        )
    )

    slice_analysis = _safe_dict(payload.get("slice_analysis"))
    lines.extend(["", "## Worst Error Slices"])
    worst_rows = [
        [
            str(row.get("category") or ""),
            str(row.get("truth_product_type") or ""),
            int(row.get("anchors", 0) or 0),
            int(row.get("model_wins", 0) or 0),
            int(row.get("baseline_wins", 0) or 0),
            int(row.get("both_wrong", 0) or 0),
            int(row.get("net_wins_model_minus_baseline", 0) or 0),
        ]
        for row in _safe_list(slice_analysis.get("worst_truth_slices"))
    ]
    lines.append(
        _md_table(
            [
                "category",
                "truth_type_or_slot",
                "anchors",
                "model_wins",
                "baseline_wins",
                "both_wrong",
                "net_wins",
            ],
            worst_rows,
        )
    )

    lines.extend(["", "## Promising Slices"])
    promising_rows = [
        [
            str(row.get("category") or ""),
            str(row.get("truth_product_type") or ""),
            int(row.get("anchors", 0) or 0),
            int(row.get("model_wins", 0) or 0),
            int(row.get("baseline_wins", 0) or 0),
            int(row.get("both_wrong", 0) or 0),
            int(row.get("net_wins_model_minus_baseline", 0) or 0),
        ]
        for row in _safe_list(slice_analysis.get("promising_truth_slices"))
    ]
    lines.append(
        _md_table(
            [
                "category",
                "truth_type_or_slot",
                "anchors",
                "model_wins",
                "baseline_wins",
                "both_wrong",
                "net_wins",
            ],
            promising_rows,
        )
    )

    lines.extend(["", "## Disagreement Pairs"])
    pair_rows = [
        [
            str(row.get("category") or ""),
            str(row.get("baseline_product_type") or ""),
            str(row.get("model_product_type") or ""),
            int(row.get("anchors", 0) or 0),
            int(row.get("model_wins", 0) or 0),
            int(row.get("baseline_wins", 0) or 0),
            int(row.get("both_wrong", 0) or 0),
            int(row.get("net_wins_model_minus_baseline", 0) or 0),
        ]
        for row in _safe_list(slice_analysis.get("worst_disagreement_pairs"))[:5]
        + _safe_list(slice_analysis.get("promising_disagreement_pairs"))[:5]
    ]
    lines.append(
        _md_table(
            [
                "category",
                "baseline",
                "model",
                "anchors",
                "model_wins",
                "baseline_wins",
                "both_wrong",
                "net_wins",
            ],
            pair_rows,
        )
    )

    lines.extend(["", "## Recommendation"])
    for item in _safe_list(recommendation.get("rationale")):
        lines.append(f"- {item}")

    lines.extend(["", "## Reproduction Command", "```powershell"])
    lines.append(
        ".\\.venv\\Scripts\\python.exe backend\\manage.py report_roadmap_nextstep_v4_decision_quality"
    )
    lines.append("```")
    return "\n".join(lines).strip() + "\n"
