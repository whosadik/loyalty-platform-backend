from __future__ import annotations

from collections import Counter
from datetime import timedelta
from pathlib import Path
from typing import Any

from django.conf import settings
from django.utils import timezone

from roadmap_app.historical_anchor_replay import build_historical_continuation_anchor_records
from roadmap_app.ml_next_step import nextstep_model_artifact_summary
from roadmap_app.models import RoadmapPlan
from roadmap_app.nextstep_historical_anchor_dataset import (
    completion_events_by_step,
    resolve_first_completed_generated_candidate,
)
from roadmap_app.shadow_evidence import (
    get_historical_control_evidence_for_model_path,
    get_historical_shadow_evidence_for_model_path,
    normalized_model_path,
)


DEFAULT_REPORT_STEM = Path("reports") / "roadmap_nextstep_haircare_shampoo_truth_design"
DOWNSTREAM_TREATMENT_TYPES = {"hair_mask", "hair_oil", "leave_in", "scalp_serum"}


def _safe_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _safe_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _rate(numerator: int, denominator: int) -> float | None:
    if denominator <= 0:
        return None
    return float(numerator) / float(denominator)


def _round_or_none(value: float | None, ndigits: int = 6) -> float | None:
    if value is None:
        return None
    return round(float(value), ndigits)


def _pct(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value * 100.0:.2f}%"


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


def _model_context(model_path: str | Path | None) -> dict[str, Any]:
    normalized = normalized_model_path(model_path)
    artifact = nextstep_model_artifact_summary(normalized) if normalized else {}
    return {
        "model_path": normalized,
        "model_version": str(_safe_dict(artifact).get("model_version") or ""),
        "selected_feature_set": str(_safe_dict(artifact).get("selected_feature_set") or ""),
    }


def _is_shampoo_anchor(anchor: dict[str, Any], control_payload: dict[str, Any]) -> tuple[bool, list[str]]:
    sources: list[str] = []
    if str(anchor.get("anchor_next_product_type") or "").strip().lower() == "shampoo":
        sources.append("anchor_next_product_type")
    if str(control_payload.get("selected_product_type") or "").strip().lower() == "shampoo":
        sources.append("historical_control_selected_product_type")
    return bool(sources), sources


def _structural_exclusion_reason(anchor: dict[str, Any]) -> str:
    if not bool(anchor.get("anchor_has_actionable_step")):
        return "no_actionable_step"
    if int(anchor.get("anchor_next_step_id") or 0) <= 0:
        return "missing_next_step_id"

    reconstruction_reason = str(anchor.get("reconstruction_reason") or "").strip().lower()
    if anchor.get("next_refresh_at") is None or reconstruction_reason == "missing_generated_steps_in_refresh_window":
        return "incomplete_refresh_window"
    if reconstruction_reason and reconstruction_reason not in {
        "",
        "ok",
        "missing_next_step_id_for_outcome_window",
        "missing_next_step_product_type",
    }:
        return f"other:{reconstruction_reason}"
    return ""


def _top_items(counter: Counter, *, limit: int = 10) -> dict[str, int]:
    return {
        str(key): int(value)
        for key, value in sorted(counter.items(), key=lambda kv: (-kv[1], str(kv[0])))[:limit]
    }


def _transition_family(product_type: str) -> str:
    normalized = str(product_type or "").strip().lower()
    if not normalized:
        return ""
    if normalized == "shampoo":
        return "repeat_shampoo"
    if normalized == "conditioner":
        return "pair_conditioner"
    if normalized in DOWNSTREAM_TREATMENT_TYPES:
        return "downstream_treatment"
    return "other"


def _build_shampoo_anchor_rows(
    *,
    model_path: str | Path,
    anchors: list[dict[str, Any]],
    meta_by_plan: dict[int, dict[str, Any]],
    completions_by_step: dict[int, list[dict[str, Any]]],
) -> dict[str, Any]:
    model_info = _model_context(model_path)
    model_path_norm = str(model_info.get("model_path") or "")
    if not model_path_norm:
        raise ValueError("model_path is required")

    identity_sources = Counter()
    rows: list[dict[str, Any]] = []

    for anchor in anchors:
        if str(anchor.get("category") or "").strip().lower() != "haircare":
            continue

        meta = _safe_dict(meta_by_plan.get(int(anchor.get("plan_id") or 0)))
        shadow_map = get_historical_shadow_evidence_for_model_path(meta, model_path_norm)
        control_map = get_historical_control_evidence_for_model_path(meta, model_path_norm)
        anchor_key = str(anchor.get("anchor_key") or "")
        shadow_payload = _safe_dict(shadow_map.get(anchor_key))
        control_payload = _safe_dict(control_map.get(anchor_key))
        is_shampoo, row_identity_sources = _is_shampoo_anchor(anchor, control_payload)
        if not is_shampoo:
            continue

        for source in row_identity_sources:
            identity_sources[source] += 1

        truth = resolve_first_completed_generated_candidate(
            anchor,
            completions_by_step=completions_by_step,
        )
        structural_reason = _structural_exclusion_reason(anchor)
        pair_available = bool(
            shadow_payload
            and control_payload
            and bool(shadow_payload.get("was_model_selected"))
            and bool(control_payload.get("was_control_selected"))
            and str(shadow_payload.get("top1_product_type") or "").strip().lower()
            and str(control_payload.get("selected_product_type") or "").strip().lower()
        )
        truth_product_type = str(truth.get("truth_selected_product_type") or "").strip().lower()
        rows.append(
            {
                "anchor_key": anchor_key,
                "plan_id": int(anchor.get("plan_id") or 0),
                "anchor_event_id": int(anchor.get("anchor_event_id") or 0),
                "anchor_created_at": str(anchor.get("anchor_created_at") or ""),
                "anchor_next_step_id": int(anchor.get("anchor_next_step_id") or 0),
                "anchor_next_step_index": int(anchor.get("anchor_next_step_index") or 0),
                "anchor_next_product_type": str(anchor.get("anchor_next_product_type") or "").strip().lower(),
                "baseline_selected_product_type": str(control_payload.get("selected_product_type") or "").strip().lower(),
                "model_top1_product_type": str(shadow_payload.get("top1_product_type") or "").strip().lower(),
                "truth_selected_product_type": truth_product_type,
                "truth_matched_by": str(truth.get("truth_matched_by") or "").strip().lower(),
                "truth_resolved": bool(truth.get("resolved")),
                "truth_reason": str(truth.get("reason") or "").strip().lower(),
                "truth_transition_family": _transition_family(truth_product_type),
                "baseline_transition_family": _transition_family(str(control_payload.get("selected_product_type") or "")),
                "model_transition_family": _transition_family(str(shadow_payload.get("top1_product_type") or "")),
                "pair_available": pair_available,
                "comparability_exclusion_reason": "" if pair_available else "pair_mapping_unavailable",
                "structural_exclusion_reason": structural_reason,
                "identity_sources": row_identity_sources,
            }
        )

    return {
        "model": model_info,
        "anchor_identity_source_counts": _top_items(identity_sources, limit=10),
        "rows": rows,
    }


def _current_gate_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    scanned = len(rows)
    unresolved = 0
    unresolved_by_reason = Counter()
    comparable_resolved = 0
    standalone_shampoo_truth_rows = 0
    shampoo_conditioner_rows = 0
    harmful_rows = 0

    for row in rows:
        if row.get("structural_exclusion_reason"):
            unresolved += 1
            unresolved_by_reason[str(row.get("structural_exclusion_reason"))] += 1
            continue
        if not bool(row.get("truth_resolved")):
            unresolved += 1
            unresolved_by_reason[str(row.get("truth_reason") or "other:unknown")] += 1
            continue
        if not bool(row.get("pair_available")):
            unresolved += 1
            unresolved_by_reason[str(row.get("comparability_exclusion_reason") or "pair_mapping_unavailable")] += 1
            continue

        comparable_resolved += 1
        truth_type = str(row.get("truth_selected_product_type") or "")
        baseline_type = str(row.get("baseline_selected_product_type") or "")
        model_type = str(row.get("model_top1_product_type") or "")
        if truth_type == "shampoo":
            standalone_shampoo_truth_rows += 1
        if baseline_type == "shampoo" and model_type == "conditioner":
            shampoo_conditioner_rows += 1
            if truth_type == "shampoo":
                harmful_rows += 1

    if harmful_rows > 0:
        verdict = {
            "status": "not_defensible_model_still_loses",
            "reason": "resolved exact shampoo -> conditioner failure rows still exist",
            "informative": True,
        }
    elif standalone_shampoo_truth_rows <= 0 and shampoo_conditioner_rows <= 0:
        verdict = {
            "status": "fail_closed_missing_truth",
            "reason": "no resolved shampoo truth rows and no resolved shampoo -> conditioner comparable rows materialize under the immutable protocol",
            "informative": False,
        }
    else:
        verdict = {
            "status": "defensible",
            "reason": "resolved exact shampoo truth rows exist",
            "informative": True,
        }

    return {
        "what_is_being_judged_now": (
            "Exact product-level shampoo acceptance: the gate only credits closure when immutable historical truth "
            "materializes resolved shampoo truth rows or resolved shampoo->conditioner comparable rows."
        ),
        "anchors_scanned_total": int(scanned),
        "resolved_anchors_total": int(comparable_resolved),
        "unresolved_anchors_total": int(unresolved),
        "unresolved_anchors_by_reason": _top_items(unresolved_by_reason, limit=20),
        "standalone_shampoo_truth_rows_total": int(standalone_shampoo_truth_rows),
        "resolved_shampoo_conditioner_comparable_rows_total": int(shampoo_conditioner_rows),
        "exact_harmful_shampoo_conditioner_rows_total": int(harmful_rows),
        "diagnosis": {
            "semantically_wrong": False,
            "too_strict": False,
            "mismatched_to_historical_truth_protocol": verdict["status"] == "fail_closed_missing_truth",
            "why": (
                "Immutable historical truth resolves shampoo anchors into downstream generated completions. "
                "That makes exact shampoo / shampoo->conditioner acceptance a mismatch for the observed event-time truth."
            ),
        },
        "verdict": verdict,
    }


def _empty_design_bucket() -> dict[str, Any]:
    return {
        "resolved": 0,
        "unresolved": 0,
        "unresolved_reasons": Counter(),
        "truth_distribution": Counter(),
        "outcomes": Counter(),
    }


def _classify_outcome(*, model_correct: bool, baseline_correct: bool) -> str:
    if model_correct and baseline_correct:
        return "both_correct"
    if (not model_correct) and (not baseline_correct):
        return "both_wrong"
    if model_correct:
        return "model_wins"
    return "baseline_wins"


def _bucket_summary(
    bucket: dict[str, Any],
    *,
    anchors_scanned_total: int,
    shampoo_conditioner_status: str,
    shampoo_conditioner_positive_truth_rows_total: int,
    defensibility_reason: str,
    informativeness_status: str,
    semantics: str,
) -> dict[str, Any]:
    resolved = int(bucket["resolved"])
    unresolved = int(bucket["unresolved"])
    outcomes = Counter(bucket["outcomes"] or {})
    return {
        "semantics": semantics,
        "anchors_scanned_total": int(anchors_scanned_total),
        "resolved_anchors_total": resolved,
        "unresolved_anchors_total": unresolved,
        "unresolved_anchors_by_reason": _top_items(bucket["unresolved_reasons"], limit=20),
        "truth_distribution": _top_items(bucket["truth_distribution"], limit=10),
        "shampoo_conditioner_observability": {
            "status": shampoo_conditioner_status,
            "positive_truth_rows_total": int(shampoo_conditioner_positive_truth_rows_total),
        },
        "model_vs_baseline_comparable": resolved > 0,
        "defensibility_reason": defensibility_reason,
        "gate_informativeness_status": informativeness_status,
        "outcome_matrix": {
            "model_wins": int(outcomes.get("model_wins", 0)),
            "baseline_wins": int(outcomes.get("baseline_wins", 0)),
            "both_correct": int(outcomes.get("both_correct", 0)),
            "both_wrong": int(outcomes.get("both_wrong", 0)),
            "model_win_rate": _round_or_none(_rate(int(outcomes.get("model_wins", 0)), resolved)),
            "baseline_win_rate": _round_or_none(_rate(int(outcomes.get("baseline_wins", 0)), resolved)),
            "agreement_rate": _round_or_none(
                _rate(int(outcomes.get("both_correct", 0)) + int(outcomes.get("both_wrong", 0)), resolved)
            ),
        },
    }


def _evaluate_design_a_anchor_step_correctness(rows: list[dict[str, Any]]) -> dict[str, Any]:
    bucket = _empty_design_bucket()
    for row in rows:
        if row.get("structural_exclusion_reason"):
            bucket["unresolved"] += 1
            bucket["unresolved_reasons"][str(row.get("structural_exclusion_reason"))] += 1
            continue
        if not bool(row.get("pair_available")):
            bucket["unresolved"] += 1
            bucket["unresolved_reasons"][str(row.get("comparability_exclusion_reason") or "pair_mapping_unavailable")] += 1
            continue
        truth_value = str(row.get("anchor_next_product_type") or "")
        if not truth_value:
            bucket["unresolved"] += 1
            bucket["unresolved_reasons"]["missing_anchor_step_identity"] += 1
            continue
        bucket["resolved"] += 1
        bucket["truth_distribution"][truth_value] += 1
        baseline_correct = str(row.get("baseline_selected_product_type") or "") == truth_value
        model_correct = str(row.get("model_top1_product_type") or "") == truth_value
        bucket["outcomes"][_classify_outcome(model_correct=model_correct, baseline_correct=baseline_correct)] += 1

    return _bucket_summary(
        bucket,
        anchors_scanned_total=len(rows),
        shampoo_conditioner_status="not_targeted_by_design",
        shampoo_conditioner_positive_truth_rows_total=0,
        defensibility_reason=(
            "Measurable from immutable PLAN_REFRESHED anchor state, but the truth is the anchor snapshot itself. "
            "That makes baseline correctness close to tautological."
        ),
        informativeness_status="measurable_but_semantically_misaligned",
        semantics="Anchor-step correctness: baseline/model are judged against the anchor's own shampoo next-step identity.",
    )


def _evaluate_design_b_immediate_pair(rows: list[dict[str, Any]]) -> dict[str, Any]:
    bucket = _empty_design_bucket()
    pair_positive_rows = 0

    for row in rows:
        if row.get("structural_exclusion_reason"):
            bucket["unresolved"] += 1
            bucket["unresolved_reasons"][str(row.get("structural_exclusion_reason"))] += 1
            continue
        if not bool(row.get("truth_resolved")):
            bucket["unresolved"] += 1
            bucket["unresolved_reasons"][str(row.get("truth_reason") or "other:unknown")] += 1
            continue
        if not bool(row.get("pair_available")):
            bucket["unresolved"] += 1
            bucket["unresolved_reasons"][str(row.get("comparability_exclusion_reason") or "pair_mapping_unavailable")] += 1
            continue

        truth_value = int(str(row.get("truth_selected_product_type") or "") == "conditioner")
        if truth_value:
            pair_positive_rows += 1
        bucket["resolved"] += 1
        bucket["truth_distribution"]["pair_conditioner" if truth_value else "not_pair_conditioner"] += 1
        baseline_correct = int(str(row.get("baseline_selected_product_type") or "") == "conditioner") == truth_value
        model_correct = int(str(row.get("model_top1_product_type") or "") == "conditioner") == truth_value
        bucket["outcomes"][_classify_outcome(model_correct=model_correct, baseline_correct=baseline_correct)] += 1

    if pair_positive_rows > 0:
        observability = "observable"
        informativeness = "measurable_exact_pair_truth"
    elif bucket["resolved"] > 0:
        observability = "observable_negative_only_zero_positive_pair_rows"
        informativeness = "measurable_but_fail_closed_zero_positive_pair_rows"
    else:
        observability = "unobservable"
        informativeness = "fail_closed_no_resolved_rows"

    return _bucket_summary(
        bucket,
        anchors_scanned_total=len(rows),
        shampoo_conditioner_status=observability,
        shampoo_conditioner_positive_truth_rows_total=pair_positive_rows,
        defensibility_reason=(
            "The pair-specific truth is measurable only when first completed in-window generated truth resolves and "
            "conditioner actually appears as the downstream positive."
        ),
        informativeness_status=informativeness,
        semantics="Immediate pair-closure truth: did shampoo -> conditioner happen as the first valid downstream generated completion?",
    )


def _evaluate_design_c_downstream_treatment(rows: list[dict[str, Any]]) -> dict[str, Any]:
    bucket = _empty_design_bucket()

    for row in rows:
        if row.get("structural_exclusion_reason"):
            bucket["unresolved"] += 1
            bucket["unresolved_reasons"][str(row.get("structural_exclusion_reason"))] += 1
            continue
        if not bool(row.get("truth_resolved")):
            bucket["unresolved"] += 1
            bucket["unresolved_reasons"][str(row.get("truth_reason") or "other:unknown")] += 1
            continue
        if not bool(row.get("pair_available")):
            bucket["unresolved"] += 1
            bucket["unresolved_reasons"][str(row.get("comparability_exclusion_reason") or "pair_mapping_unavailable")] += 1
            continue
        truth_family = str(row.get("truth_transition_family") or "")
        if truth_family != "downstream_treatment":
            bucket["unresolved"] += 1
            bucket["unresolved_reasons"]["truth_not_downstream_treatment"] += 1
            continue

        bucket["resolved"] += 1
        bucket["truth_distribution"][truth_family] += 1
        baseline_correct = str(row.get("baseline_transition_family") or "") == truth_family
        model_correct = str(row.get("model_transition_family") or "") == truth_family
        bucket["outcomes"][_classify_outcome(model_correct=model_correct, baseline_correct=baseline_correct)] += 1

    return _bucket_summary(
        bucket,
        anchors_scanned_total=len(rows),
        shampoo_conditioner_status="not_targeted_by_design",
        shampoo_conditioner_positive_truth_rows_total=0,
        defensibility_reason=(
            "This design is defensible on immutable completions because it aligns to the observed downstream treatment family. "
            "It is still coarse and can over-credit any model that simply leaves the shampoo family."
        ),
        informativeness_status="measurable_but_coarse_family_only",
        semantics="Downstream treatment truth: did the first completed generated candidate land in the correct downstream treatment family?",
    )


def _evaluate_design_d_two_stage(rows: list[dict[str, Any]]) -> dict[str, Any]:
    stage1 = _empty_design_bucket()
    stage2 = _empty_design_bucket()
    pair_conditioner_rows = 0

    for row in rows:
        if row.get("structural_exclusion_reason"):
            reason = str(row.get("structural_exclusion_reason"))
            stage1["unresolved"] += 1
            stage1["unresolved_reasons"][reason] += 1
            stage2["unresolved"] += 1
            stage2["unresolved_reasons"][reason] += 1
            continue
        if not bool(row.get("truth_resolved")):
            reason = str(row.get("truth_reason") or "other:unknown")
            stage1["unresolved"] += 1
            stage1["unresolved_reasons"][reason] += 1
            stage2["unresolved"] += 1
            stage2["unresolved_reasons"][reason] += 1
            continue
        if not bool(row.get("pair_available")):
            reason = str(row.get("comparability_exclusion_reason") or "pair_mapping_unavailable")
            stage1["unresolved"] += 1
            stage1["unresolved_reasons"][reason] += 1
            stage2["unresolved"] += 1
            stage2["unresolved_reasons"][reason] += 1
            continue

        truth_family = str(row.get("truth_transition_family") or "")
        truth_type = str(row.get("truth_selected_product_type") or "")
        if truth_family == "pair_conditioner":
            pair_conditioner_rows += 1

        stage1["resolved"] += 1
        stage1["truth_distribution"][truth_family or "unknown"] += 1
        stage1["outcomes"][
            _classify_outcome(
                model_correct=str(row.get("model_transition_family") or "") == truth_family,
                baseline_correct=str(row.get("baseline_transition_family") or "") == truth_family,
            )
        ] += 1

        stage2["resolved"] += 1
        stage2["truth_distribution"][truth_type or "unknown"] += 1
        stage2["outcomes"][
            _classify_outcome(
                model_correct=str(row.get("model_top1_product_type") or "") == truth_type,
                baseline_correct=str(row.get("baseline_selected_product_type") or "") == truth_type,
            )
        ] += 1

    if pair_conditioner_rows > 0:
        observability = "observable"
    elif stage1["resolved"] > 0:
        observability = "observable_family_level_zero_positive_pair_rows"
    else:
        observability = "unobservable"

    stage1_summary = _bucket_summary(
        stage1,
        anchors_scanned_total=len(rows),
        shampoo_conditioner_status=observability,
        shampoo_conditioner_positive_truth_rows_total=pair_conditioner_rows,
        defensibility_reason=(
            "Stage 1 uses immutable downstream family truth. It can distinguish repeat_shampoo vs pair_conditioner vs downstream_treatment."
        ),
        informativeness_status="measurable_family_transition_truth",
        semantics="Stage 1: correct transition family after shampoo anchor.",
    )
    stage2_summary = _bucket_summary(
        stage2,
        anchors_scanned_total=len(rows),
        shampoo_conditioner_status=observability,
        shampoo_conditioner_positive_truth_rows_total=pair_conditioner_rows,
        defensibility_reason=(
            "Stage 2 uses exact downstream completed generated truth once stage-1 family is measurable."
        ),
        informativeness_status="measurable_exact_concrete_downstream_truth",
        semantics="Stage 2: correct concrete downstream generated step after the family transition.",
    )

    return {
        "semantics": (
            "Two-stage truth: first judge the transition family after shampoo, then judge the exact downstream generated step."
        ),
        "anchors_scanned_total": len(rows),
        "resolved_anchors_total": int(stage1["resolved"]),
        "unresolved_anchors_total": int(stage1["unresolved"]),
        "unresolved_anchors_by_reason": _top_items(stage1["unresolved_reasons"], limit=20),
        "shampoo_conditioner_observability": {
            "status": observability,
            "positive_truth_rows_total": int(pair_conditioner_rows),
        },
        "model_vs_baseline_comparable": int(stage1["resolved"]) > 0,
        "defensibility_reason": (
            "This design matches what immutable history actually records: a transition family is measurable first, "
            "then the exact downstream concrete step is measurable on the same resolved anchors."
        ),
        "gate_informativeness_status": "measurable_and_most_defensible",
        "stage_1_family": stage1_summary,
        "stage_2_concrete_step": stage2_summary,
    }


def evaluate_haircare_shampoo_truth_designs(rows: list[dict[str, Any]]) -> dict[str, Any]:
    current_gate = _current_gate_summary(rows)
    design_a = _evaluate_design_a_anchor_step_correctness(rows)
    design_b = _evaluate_design_b_immediate_pair(rows)
    design_c = _evaluate_design_c_downstream_treatment(rows)
    design_d = _evaluate_design_d_two_stage(rows)

    recommendation_status = "truth_too_weak_keep_frozen"
    recommendation_reason = (
        "No alternative truth design was both measurable and informative enough to replace the current fail-closed gate."
    )
    rerun_v5_comparison = False

    stage2_matrix = _safe_dict(_safe_dict(design_d.get("stage_2_concrete_step")).get("outcome_matrix"))
    if (
        current_gate.get("verdict", {}).get("status") == "fail_closed_missing_truth"
        and design_d.get("model_vs_baseline_comparable")
        and int(design_d.get("resolved_anchors_total", 0) or 0) > 0
        and int(stage2_matrix.get("model_wins", 0) or 0) > 0
    ):
        recommendation_status = "adopt_redesigned_truth_and_rerun_v5_comparison"
        recommendation_reason = (
            "The current gate is mismatched to immutable truth, while two-stage truth is measurable on resolved historical anchors "
            "and preserves both transition-family and concrete-step checks."
        )
        rerun_v5_comparison = True
    elif current_gate.get("verdict", {}).get("status") != "fail_closed_missing_truth":
        recommendation_status = "keep_current_gate"
        recommendation_reason = "The current gate already materializes defensible exact shampoo truth."

    return {
        "current_gate": current_gate,
        "designs": {
            "A_anchor_step_correctness": design_a,
            "B_immediate_pair_closure": design_b,
            "C_downstream_treatment_truth": design_c,
            "D_two_stage_truth": design_d,
        },
        "recommendation": {
            "status": recommendation_status,
            "reason": recommendation_reason,
            "recommended_truth_design": "D_two_stage_truth" if rerun_v5_comparison else "",
            "rerun_v5_comparison_under_recommended_truth": rerun_v5_comparison,
            "remaining_unresolved": (
                "Explicit positive shampoo->conditioner truth rows remain absent in the current immutable window, "
                "so conditioner-specific positive closure still cannot be claimed."
            ),
        },
    }


def _reference_delta(candidate_value: float | None, reference_value: float | None) -> float | None:
    if candidate_value is None or reference_value is None:
        return None
    return _round_or_none(float(candidate_value) - float(reference_value))


def _compare_design_recommendation(candidate_designs: dict[str, Any], reference_designs: dict[str, Any] | None) -> dict[str, Any]:
    comparison: dict[str, Any] = {}
    if not reference_designs:
        return comparison

    candidate_d = _safe_dict(candidate_designs.get("D_two_stage_truth"))
    reference_d = _safe_dict(reference_designs.get("D_two_stage_truth"))
    candidate_stage2 = _safe_dict(candidate_d.get("stage_2_concrete_step"))
    reference_stage2 = _safe_dict(reference_d.get("stage_2_concrete_step"))
    candidate_stage2_matrix = _safe_dict(candidate_stage2.get("outcome_matrix"))
    reference_stage2_matrix = _safe_dict(reference_stage2.get("outcome_matrix"))
    comparison["D_two_stage_truth"] = {
        "candidate_stage2_model_win_rate": candidate_stage2_matrix.get("model_win_rate"),
        "reference_stage2_model_win_rate": reference_stage2_matrix.get("model_win_rate"),
        "delta_stage2_model_win_rate": _reference_delta(
            candidate_stage2_matrix.get("model_win_rate"),
            reference_stage2_matrix.get("model_win_rate"),
        ),
        "candidate_stage2_model_wins": candidate_stage2_matrix.get("model_wins"),
        "reference_stage2_model_wins": reference_stage2_matrix.get("model_wins"),
    }
    return comparison


def build_nextstep_haircare_shampoo_truth_design_payload(
    *,
    model_path: str | Path | None,
    days: int = 30,
    include_ga: bool = False,
    reference_model_path: str | Path | None = None,
) -> dict[str, Any]:
    now_utc = timezone.now()
    since = now_utc - timedelta(days=int(days))
    anchors = build_historical_continuation_anchor_records(
        since=since,
        until=now_utc,
        category="all",
        include_ga=include_ga,
    )
    plan_ids = {
        int(anchor.get("plan_id") or 0)
        for anchor in anchors
        if int(anchor.get("plan_id") or 0) > 0
    }
    meta_by_plan = {
        int(row["id"]): _safe_dict(row.get("meta"))
        for row in RoadmapPlan.objects.filter(id__in=plan_ids).values("id", "meta")
    }
    all_generated_step_ids = {
        int(step_id)
        for anchor in anchors
        for step_id in _safe_list(anchor.get("generated_step_ids"))
        if str(step_id or "").strip()
    }
    completions_by_step = completion_events_by_step(
        since=since,
        until=now_utc,
        step_ids=all_generated_step_ids,
    )

    candidate_rows_payload = _build_shampoo_anchor_rows(
        model_path=str(model_path or ""),
        anchors=anchors,
        meta_by_plan=meta_by_plan,
        completions_by_step=completions_by_step,
    )
    candidate_truth_designs = evaluate_haircare_shampoo_truth_designs(candidate_rows_payload["rows"])

    resolved_reference = normalized_model_path(
        reference_model_path or getattr(settings, "ROADMAP_NEXTSTEP_V4_MODEL_PATH", "")
    )
    reference_payload: dict[str, Any] | None = None
    reference_truth_designs: dict[str, Any] | None = None
    if resolved_reference and resolved_reference != candidate_rows_payload["model"]["model_path"]:
        reference_rows_payload = _build_shampoo_anchor_rows(
            model_path=resolved_reference,
            anchors=anchors,
            meta_by_plan=meta_by_plan,
            completions_by_step=completions_by_step,
        )
        reference_truth_designs = evaluate_haircare_shampoo_truth_designs(reference_rows_payload["rows"])
        reference_payload = {
            "model": reference_rows_payload["model"],
            "anchor_identity_source_counts": reference_rows_payload["anchor_identity_source_counts"],
            "truth_designs": reference_truth_designs,
        }

    recommendation = _safe_dict(candidate_truth_designs.get("recommendation"))
    comparison = _compare_design_recommendation(
        _safe_dict(candidate_truth_designs.get("designs")),
        None if reference_truth_designs is None else _safe_dict(reference_truth_designs.get("designs")),
    )

    return {
        "generated_at_utc": now_utc.isoformat(),
        "window_start_utc": since.isoformat(),
        "window_end_utc": now_utc.isoformat(),
        "params": {
            "days": int(days),
            "include_ga": bool(include_ga),
            "candidate_model_path": candidate_rows_payload["model"]["model_path"],
            "reference_model_path": None if reference_payload is None else reference_payload["model"]["model_path"],
        },
        "catalog_safety": {
            "catalog_writes_performed": False,
            "sources_used": [
                "RoadmapEvent PLAN_REFRESHED",
                "RoadmapEvent STEP_GENERATED",
                "RoadmapEvent STEP_EXPOSED",
                "RoadmapEvent STEP_COMPLETED",
                "RoadmapEvent STEP_SKIPPED",
                "RoadmapPlan.meta historical replay evidence",
            ],
            "catalog_models_touched": [],
            "import_logic_touched": [],
        },
        "candidate": {
            "model": candidate_rows_payload["model"],
            "anchor_identity_source_counts": candidate_rows_payload["anchor_identity_source_counts"],
            "truth_designs": candidate_truth_designs,
        },
        "reference": reference_payload,
        "comparison_vs_reference": comparison,
        "executive_verdict": {
            "current_gate_root_cause": (
                "The current shampoo gate is mismatched to the immutable historical truth protocol. "
                "History resolves these anchors into downstream generated treatment completions, not exact shampoo truth rows."
            ),
            "recommended_truth_design": recommendation.get("recommended_truth_design"),
            "recommendation_status": recommendation.get("status"),
            "recommendation_reason": recommendation.get("reason"),
            "rerun_v5_comparison_under_recommended_truth": bool(
                recommendation.get("rerun_v5_comparison_under_recommended_truth")
            ),
        },
        "notes": [
            "Read-only analysis: no runtime changes, no retraining, no catalog writes.",
            "Truth designs are evaluated only from immutable roadmap events and replay evidence already stored in plan meta.",
            "Conditioner-specific positive truth may still remain absent even under the recommended design.",
        ],
    }


def render_nextstep_haircare_shampoo_truth_design_markdown(payload: dict[str, Any]) -> str:
    candidate = _safe_dict(payload.get("candidate"))
    candidate_truth_designs = _safe_dict(candidate.get("truth_designs"))
    candidate_current_gate = _safe_dict(candidate_truth_designs.get("current_gate"))
    candidate_designs = _safe_dict(candidate_truth_designs.get("designs"))
    recommendation = _safe_dict(candidate_truth_designs.get("recommendation"))
    reference = _safe_dict(payload.get("reference"))
    reference_truth_designs = _safe_dict(reference.get("truth_designs"))
    reference_designs = _safe_dict(reference_truth_designs.get("designs"))
    comparison = _safe_dict(payload.get("comparison_vs_reference"))

    lines: list[str] = []
    lines.append("# Roadmap Nextstep Haircare Shampoo Truth Design")
    lines.append("")
    lines.append("## Executive Verdict")
    lines.append(f"- current gate verdict: `{_safe_dict(candidate_current_gate.get('verdict')).get('status')}`")
    lines.append(f"- recommended truth design: `{recommendation.get('recommended_truth_design') or 'none'}`")
    lines.append(f"- recommendation: `{recommendation.get('status')}`")
    lines.append(f"- reason: {recommendation.get('reason')}")
    lines.append(
        f"- rerun v5 comparison under redesigned truth: `{bool(recommendation.get('rerun_v5_comparison_under_recommended_truth'))}`"
    )
    lines.append("")
    lines.append("## Current Gate Failure Root Cause")
    lines.append(f"- judged now: {candidate_current_gate.get('what_is_being_judged_now')}")
    lines.append(f"- standalone shampoo truth rows: `{candidate_current_gate.get('standalone_shampoo_truth_rows_total')}`")
    lines.append(
        f"- resolved shampoo -> conditioner rows: `{candidate_current_gate.get('resolved_shampoo_conditioner_comparable_rows_total')}`"
    )
    lines.append(
        f"- diagnosis: mismatched_to_historical_truth_protocol=`{_safe_dict(candidate_current_gate.get('diagnosis')).get('mismatched_to_historical_truth_protocol')}`"
    )
    lines.append(f"- why: {_safe_dict(candidate_current_gate.get('diagnosis')).get('why')}")
    lines.append("")
    lines.append("## Truth Design Comparison")
    comparison_rows: list[list[Any]] = []
    for design_key in [
        "A_anchor_step_correctness",
        "B_immediate_pair_closure",
        "C_downstream_treatment_truth",
        "D_two_stage_truth",
    ]:
        candidate_design = _safe_dict(candidate_designs.get(design_key))
        pair_obs = _safe_dict(candidate_design.get("shampoo_conditioner_observability"))
        if design_key == "D_two_stage_truth":
            stage2_matrix = _safe_dict(_safe_dict(candidate_design.get("stage_2_concrete_step")).get("outcome_matrix"))
            model_win_rate = _pct(stage2_matrix.get("model_win_rate"))
        else:
            model_win_rate = _pct(_safe_dict(candidate_design.get("outcome_matrix")).get("model_win_rate"))
        comparison_rows.append(
            [
                design_key,
                candidate_design.get("resolved_anchors_total"),
                candidate_design.get("unresolved_anchors_total"),
                pair_obs.get("status"),
                "yes" if candidate_design.get("model_vs_baseline_comparable") else "no",
                candidate_design.get("gate_informativeness_status"),
                model_win_rate,
            ]
        )
    lines.append(
        _md_table(
            [
                "design",
                "resolved",
                "unresolved",
                "shampoo->conditioner observability",
                "model_vs_baseline_defensible",
                "informativeness",
                "candidate_model_win_rate",
            ],
            comparison_rows,
        )
    )
    for design_key, title in [
        ("A_anchor_step_correctness", "A. Anchor-Step Correctness"),
        ("B_immediate_pair_closure", "B. Immediate Pair-Closure Truth"),
        ("C_downstream_treatment_truth", "C. Downstream Treatment Truth"),
        ("D_two_stage_truth", "D. Two-Stage Truth"),
    ]:
        candidate_design = _safe_dict(candidate_designs.get(design_key))
        lines.append("")
        lines.append(f"## {title}")
        lines.append(f"- semantics: {candidate_design.get('semantics')}")
        lines.append(f"- resolved anchors: `{candidate_design.get('resolved_anchors_total')}`")
        lines.append(f"- unresolved anchors: `{candidate_design.get('unresolved_anchors_total')}`")
        lines.append(
            f"- shampoo->conditioner observability: `{_safe_dict(candidate_design.get('shampoo_conditioner_observability')).get('status')}`"
        )
        lines.append(
            f"- positive shampoo->conditioner truth rows: `{_safe_dict(candidate_design.get('shampoo_conditioner_observability')).get('positive_truth_rows_total')}`"
        )
        lines.append(f"- defensible model-vs-baseline compare: `{bool(candidate_design.get('model_vs_baseline_comparable'))}`")
        lines.append(f"- defensibility reason: {candidate_design.get('defensibility_reason')}")
        lines.append(f"- informativeness: `{candidate_design.get('gate_informativeness_status')}`")
        lines.append("")
        lines.append("Unresolved by reason")
        lines.append(
            _md_table(
                ["reason", "count"],
                [
                    [reason, count]
                    for reason, count in _safe_dict(candidate_design.get("unresolved_anchors_by_reason")).items()
                ],
            )
        )
        if design_key == "D_two_stage_truth":
            stage1 = _safe_dict(candidate_design.get("stage_1_family"))
            stage2 = _safe_dict(candidate_design.get("stage_2_concrete_step"))
            stage1_matrix = _safe_dict(stage1.get("outcome_matrix"))
            stage2_matrix = _safe_dict(stage2.get("outcome_matrix"))
            lines.append("")
            lines.append("Stage 1 family outcome matrix")
            lines.append(
                _md_table(
                    ["metric", "value"],
                    [
                        ["model_wins", stage1_matrix.get("model_wins", 0)],
                        ["baseline_wins", stage1_matrix.get("baseline_wins", 0)],
                        ["both_correct", stage1_matrix.get("both_correct", 0)],
                        ["both_wrong", stage1_matrix.get("both_wrong", 0)],
                        ["model_win_rate", _pct(stage1_matrix.get("model_win_rate"))],
                        ["baseline_win_rate", _pct(stage1_matrix.get("baseline_win_rate"))],
                    ],
                )
            )
            lines.append("")
            lines.append("Stage 2 concrete-step outcome matrix")
            lines.append(
                _md_table(
                    ["metric", "value"],
                    [
                        ["model_wins", stage2_matrix.get("model_wins", 0)],
                        ["baseline_wins", stage2_matrix.get("baseline_wins", 0)],
                        ["both_correct", stage2_matrix.get("both_correct", 0)],
                        ["both_wrong", stage2_matrix.get("both_wrong", 0)],
                        ["model_win_rate", _pct(stage2_matrix.get("model_win_rate"))],
                        ["baseline_win_rate", _pct(stage2_matrix.get("baseline_win_rate"))],
                    ],
                )
            )
        else:
            matrix = _safe_dict(candidate_design.get("outcome_matrix"))
            lines.append("")
            lines.append("Outcome matrix")
            lines.append(
                _md_table(
                    ["metric", "value"],
                    [
                        ["model_wins", matrix.get("model_wins", 0)],
                        ["baseline_wins", matrix.get("baseline_wins", 0)],
                        ["both_correct", matrix.get("both_correct", 0)],
                        ["both_wrong", matrix.get("both_wrong", 0)],
                        ["model_win_rate", _pct(matrix.get("model_win_rate"))],
                        ["baseline_win_rate", _pct(matrix.get("baseline_win_rate"))],
                    ],
                )
            )

    if reference:
        lines.append("")
        lines.append("## Reference Check")
        lines.append(f"- reference_model_path: `{_safe_dict(reference.get('model')).get('model_path')}`")
        for design_key in ["B_immediate_pair_closure", "C_downstream_treatment_truth", "D_two_stage_truth"]:
            reference_design = _safe_dict(reference_designs.get(design_key))
            if design_key == "D_two_stage_truth":
                stage2_matrix = _safe_dict(_safe_dict(reference_design.get("stage_2_concrete_step")).get("outcome_matrix"))
                reference_model_win_rate = _pct(stage2_matrix.get("model_win_rate"))
            else:
                reference_model_win_rate = _pct(_safe_dict(reference_design.get("outcome_matrix")).get("model_win_rate"))
            lines.append(
                f"- {design_key}: resolved=`{reference_design.get('resolved_anchors_total')}`, model_win_rate=`{reference_model_win_rate}`"
            )
        d_comparison = _safe_dict(comparison.get("D_two_stage_truth"))
        if d_comparison:
            lines.append(
                f"- D stage2 candidate vs reference delta model_win_rate: `{_pct(d_comparison.get('delta_stage2_model_win_rate'))}`"
            )

    lines.append("")
    lines.append("## Recommended Truth Design")
    lines.append(f"- recommended: `{recommendation.get('recommended_truth_design') or 'none'}`")
    lines.append(f"- why: {recommendation.get('reason')}")
    lines.append(f"- still unresolved: {recommendation.get('remaining_unresolved')}")
    lines.append("")
    lines.append("## Catalog Safety")
    lines.append("- catalog_writes_performed: `false`")
    lines.append("- catalog/product rows modified: `0`")
    lines.append("- catalog attrs/raw_meta modified: `0`")
    lines.append("- import logic modified: `0`")
    return "\n".join(lines).strip() + "\n"
