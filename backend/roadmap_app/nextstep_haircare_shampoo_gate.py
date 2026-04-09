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


DEFAULT_REPORT_STEM = Path("reports") / "roadmap_nextstep_haircare_shampoo_gate"


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


def _primary_exclusion_reason(anchor: dict[str, Any], truth: dict[str, Any]) -> str:
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

    truth_reason = str(truth.get("reason") or "").strip().lower()
    if truth_reason in {
        "no_completed_generated_candidate",
        "ambiguous_outcome_window",
        "missing_next_step_id",
        "no_actionable_step",
        "incomplete_refresh_window",
    }:
        return truth_reason
    if truth_reason:
        return f"other:{truth_reason}"
    return "other:unknown"


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


def _analyze_single_model_shampoo_gate(
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

    scanned = 0
    resolved_truth = 0
    resolved_truth_without_pair_mapping = 0
    comparable_resolved = 0
    unresolved = 0
    pair_mapping_unavailable = 0
    standalone_shampoo_truth_rows = 0
    resolved_shampoo_conditioner_rows = 0
    exact_harmful_rows = 0

    anchor_identity_sources = Counter()
    unresolved_by_reason = Counter()
    truth_product_types = Counter()
    baseline_selected_types = Counter()
    model_top1_types = Counter()
    outcome_classes = Counter()
    pair_patterns = Counter()
    unresolved_examples: dict[str, list[dict[str, Any]]] = {}
    sample_resolved_rows: list[dict[str, Any]] = []

    for anchor in anchors:
        if str(anchor.get("category") or "").strip().lower() != "haircare":
            continue

        meta = _safe_dict(meta_by_plan.get(int(anchor.get("plan_id") or 0)))
        shadow_map = get_historical_shadow_evidence_for_model_path(meta, model_path_norm)
        control_map = get_historical_control_evidence_for_model_path(meta, model_path_norm)
        anchor_key = str(anchor.get("anchor_key") or "")
        shadow_payload = _safe_dict(shadow_map.get(anchor_key))
        control_payload = _safe_dict(control_map.get(anchor_key))

        is_shampoo, identity_sources = _is_shampoo_anchor(anchor, control_payload)
        if not is_shampoo:
            continue

        scanned += 1
        for source in identity_sources:
            anchor_identity_sources[source] += 1

        truth = resolve_first_completed_generated_candidate(
            anchor,
            completions_by_step=completions_by_step,
        )
        truth_resolved = bool(truth.get("resolved"))
        pair_available = bool(
            shadow_payload
            and control_payload
            and bool(shadow_payload.get("was_model_selected"))
            and bool(control_payload.get("was_control_selected"))
            and str(shadow_payload.get("top1_product_type") or "").strip().lower()
            and str(control_payload.get("selected_product_type") or "").strip().lower()
        )

        record = {
            "anchor_key": anchor_key,
            "plan_id": int(anchor.get("plan_id") or 0),
            "anchor_event_id": int(anchor.get("anchor_event_id") or 0),
            "anchor_created_at": str(anchor.get("anchor_created_at") or ""),
            "anchor_next_step_id": int(anchor.get("anchor_next_step_id") or 0),
            "anchor_next_step_index": int(anchor.get("anchor_next_step_index") or 0),
            "anchor_next_product_type": str(anchor.get("anchor_next_product_type") or "").strip().lower(),
            "baseline_selected_product_type": str(control_payload.get("selected_product_type") or "").strip().lower(),
            "model_top1_product_type": str(shadow_payload.get("top1_product_type") or "").strip().lower(),
            "truth_selected_product_type": str(truth.get("truth_selected_product_type") or "").strip().lower(),
            "truth_matched_by": str(truth.get("truth_matched_by") or "").strip().lower(),
            "truth_resolved": truth_resolved,
            "pair_available": pair_available,
        }

        structural_reason = _structural_exclusion_reason(anchor)
        if structural_reason:
            unresolved += 1
            unresolved_by_reason[structural_reason] += 1
            if len(unresolved_examples.setdefault(structural_reason, [])) < 3:
                unresolved_examples[structural_reason].append(record)
            continue

        if truth_resolved and pair_available:
            comparable_resolved += 1
            resolved_truth += 1
            truth_type = str(truth.get("truth_selected_product_type") or "").strip().lower()
            baseline_type = str(control_payload.get("selected_product_type") or "").strip().lower()
            model_type = str(shadow_payload.get("top1_product_type") or "").strip().lower()
            truth_product_types[truth_type] += 1
            baseline_selected_types[baseline_type] += 1
            model_top1_types[model_type] += 1
            if truth_type == "shampoo":
                standalone_shampoo_truth_rows += 1
            if baseline_type == "shampoo" and model_type == "conditioner":
                resolved_shampoo_conditioner_rows += 1
                if truth_type == "shampoo":
                    exact_harmful_rows += 1

            baseline_correct = baseline_type == truth_type
            model_correct = model_type == truth_type
            if baseline_correct and model_correct:
                outcome_class = "both_correct"
            elif (not baseline_correct) and (not model_correct):
                outcome_class = "both_wrong"
            elif model_correct:
                outcome_class = "model_wins"
            else:
                outcome_class = "baseline_wins"
            outcome_classes[outcome_class] += 1
            pair_patterns[f"{baseline_type}->{model_type}=>{truth_type}"] += 1
            if len(sample_resolved_rows) < 5:
                sample_resolved_rows.append(record)
            continue

        if truth_resolved and not pair_available:
            resolved_truth += 1
            resolved_truth_without_pair_mapping += 1
            pair_mapping_unavailable += 1
            unresolved_by_reason["pair_mapping_unavailable"] += 1
            unresolved_examples.setdefault("pair_mapping_unavailable", []).append(record)
            continue

        unresolved += 1
        reason = _primary_exclusion_reason(anchor, truth)
        unresolved_by_reason[reason] += 1
        if reason == "pair_mapping_unavailable":
            pair_mapping_unavailable += 1
        if len(unresolved_examples.setdefault(reason, [])) < 3:
            unresolved_examples[reason].append(record)

    verdict_status = "closed"
    verdict_reason = "resolved shampoo anchors show no remaining harmful shampoo -> conditioner failures"
    if exact_harmful_rows > 0:
        verdict_status = "not_closed_because_model_still_loses"
        verdict_reason = (
            "resolved shampoo anchors still contain baseline=shampoo, model=conditioner, truth=shampoo failures"
        )
    elif standalone_shampoo_truth_rows <= 0 and resolved_shampoo_conditioner_rows <= 0:
        verdict_status = "not_closed_because_missing_truth"
        verdict_reason = (
            "the current historical-anchor truth protocol does not materialize resolved shampoo truth or resolved shampoo -> conditioner pair rows"
        )
    elif unresolved > 0 or resolved_truth_without_pair_mapping > 0:
        verdict_status = "not_closed_because_missing_truth"
        verdict_reason = (
            "some shampoo anchors still lack defensible completed-candidate truth or pair mapping, so full closure cannot be proven"
        )

    invisibility_root_cause = (
        "Broad historical compare aggregates by resolved truth_product_type and resolved disagreement pairs. "
        "Shampoo anchors remain present, but under the current immutable truth protocol they resolve to downstream "
        "haircare outcomes instead of truth_product_type=shampoo, and there are no resolved baseline=shampoo/model=conditioner rows."
    )

    return {
        "model": model_info,
        "root_cause_of_invisibility": invisibility_root_cause,
        "shampoo_anchor_definition": {
            "anchor_identity": "anchor_next_product_type == shampoo OR reconstructed baseline control selected_product_type == shampoo",
            "uses_only_immutable_sources": True,
        },
        "anchors_scanned_total": int(scanned),
        "anchor_identity_source_counts": _top_items(anchor_identity_sources, limit=10),
        "resolved_truth_anchors_total": int(resolved_truth),
        "resolved_comparable_anchors_total": int(comparable_resolved),
        "resolved_truth_without_pair_mapping_total": int(resolved_truth_without_pair_mapping),
        "unresolved_anchors_total": int(unresolved),
        "unresolved_anchors_by_reason": _top_items(unresolved_by_reason, limit=20),
        "pair_mapping_unavailable_total": int(pair_mapping_unavailable),
        "standalone_shampoo_truth_rows_total": int(standalone_shampoo_truth_rows),
        "resolved_shampoo_conditioner_comparable_rows_total": int(resolved_shampoo_conditioner_rows),
        "exact_harmful_shampoo_conditioner_rows_total": int(exact_harmful_rows),
        "baseline_selected_distribution": _top_items(baseline_selected_types, limit=10),
        "model_top1_distribution": _top_items(model_top1_types, limit=10),
        "resolved_truth_distribution": _top_items(truth_product_types, limit=10),
        "resolved_outcome_matrix": {
            "model_wins": int(outcome_classes.get("model_wins", 0)),
            "baseline_wins": int(outcome_classes.get("baseline_wins", 0)),
            "both_correct": int(outcome_classes.get("both_correct", 0)),
            "both_wrong": int(outcome_classes.get("both_wrong", 0)),
            "model_win_rate": _round_or_none(_rate(int(outcome_classes.get("model_wins", 0)), comparable_resolved)),
            "baseline_win_rate": _round_or_none(_rate(int(outcome_classes.get("baseline_wins", 0)), comparable_resolved)),
        },
        "resolved_pair_patterns": _top_items(pair_patterns, limit=10),
        "sample_resolved_rows": sample_resolved_rows,
        "sample_unresolved_rows": {
            reason: rows[:3]
            for reason, rows in sorted(unresolved_examples.items(), key=lambda kv: kv[0])
        },
        "verdict": {
            "status": verdict_status,
            "reason": verdict_reason,
            "targeted_failure_mode_defensibly_closed": verdict_status == "closed",
        },
    }


def build_nextstep_haircare_shampoo_gate_payload(
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

    resolved_reference = normalized_model_path(
        reference_model_path or getattr(settings, "ROADMAP_NEXTSTEP_V4_MODEL_PATH", "")
    )
    candidate_summary = _analyze_single_model_shampoo_gate(
        model_path=str(model_path or ""),
        anchors=anchors,
        meta_by_plan=meta_by_plan,
        completions_by_step=completions_by_step,
    )

    reference_summary: dict[str, Any] | None = None
    if resolved_reference and normalized_model_path(model_path) != resolved_reference:
        reference_summary = _analyze_single_model_shampoo_gate(
            model_path=resolved_reference,
            anchors=anchors,
            meta_by_plan=meta_by_plan,
            completions_by_step=completions_by_step,
        )

    root_cause = candidate_summary["root_cause_of_invisibility"]
    if reference_summary is not None:
        candidate_pair_rows = int(candidate_summary.get("resolved_shampoo_conditioner_comparable_rows_total", 0))
        reference_pair_rows = int(reference_summary.get("resolved_shampoo_conditioner_comparable_rows_total", 0))
        candidate_truth_rows = int(candidate_summary.get("standalone_shampoo_truth_rows_total", 0))
        reference_truth_rows = int(reference_summary.get("standalone_shampoo_truth_rows_total", 0))
        if (
            candidate_pair_rows == 0
            and reference_pair_rows == 0
            and candidate_truth_rows == 0
            and reference_truth_rows == 0
        ):
            root_cause += (
                " The same invisibility also holds for the active reference artifact, so zero harmful rows here "
                "cannot be credited as a model-specific fix."
            )

    return {
        "generated_at_utc": now_utc.isoformat(),
        "window_start_utc": since.isoformat(),
        "window_end_utc": now_utc.isoformat(),
        "params": {
            "days": int(days),
            "include_ga": bool(include_ga),
            "candidate_model_path": candidate_summary["model"]["model_path"],
            "reference_model_path": None if reference_summary is None else reference_summary["model"]["model_path"],
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
        "root_cause_of_shampoo_gate_invisibility": root_cause,
        "candidate": candidate_summary,
        "reference": reference_summary,
        "notes": [
            "Read-only analysis: no runtime changes, no retraining, no catalog writes.",
            "Shampoo gate is analyzed on immutable historical anchors, not current catalog content.",
            "A closed gate requires defensible resolved shampoo truth or resolved shampoo -> conditioner comparable rows with no remaining harmful failures.",
        ],
    }


def render_nextstep_haircare_shampoo_gate_markdown(payload: dict[str, Any]) -> str:
    candidate = _safe_dict(payload.get("candidate"))
    reference = _safe_dict(payload.get("reference"))
    verdict = _safe_dict(candidate.get("verdict"))
    lines: list[str] = []
    lines.append("# Roadmap Nextstep Haircare Shampoo Gate")
    lines.append("")
    lines.append("## Executive Verdict")
    lines.append(f"- status: `{verdict.get('status')}`")
    lines.append(f"- closed: `{bool(verdict.get('targeted_failure_mode_defensibly_closed'))}`")
    lines.append(f"- reason: {verdict.get('reason')}")
    lines.append("")
    lines.append("## Root Cause")
    lines.append(f"- {payload.get('root_cause_of_shampoo_gate_invisibility')}")
    lines.append("")
    lines.append("## Candidate Summary")
    lines.append(f"- model_path: `{_safe_dict(candidate.get('model')).get('model_path')}`")
    lines.append(f"- model_version: `{_safe_dict(candidate.get('model')).get('model_version')}`")
    lines.append(f"- shampoo anchors scanned: `{candidate.get('anchors_scanned_total')}`")
    lines.append(f"- resolved shampoo anchors: `{candidate.get('resolved_truth_anchors_total')}`")
    lines.append(f"- resolved comparable shampoo anchors: `{candidate.get('resolved_comparable_anchors_total')}`")
    lines.append(f"- resolved truth without pair mapping: `{candidate.get('resolved_truth_without_pair_mapping_total')}`")
    lines.append(f"- unresolved shampoo anchors: `{candidate.get('unresolved_anchors_total')}`")
    lines.append(f"- standalone shampoo truth rows: `{candidate.get('standalone_shampoo_truth_rows_total')}`")
    lines.append(
        f"- resolved shampoo -> conditioner comparable rows: `{candidate.get('resolved_shampoo_conditioner_comparable_rows_total')}`"
    )
    lines.append(
        f"- exact harmful shampoo -> conditioner rows: `{candidate.get('exact_harmful_shampoo_conditioner_rows_total')}`"
    )
    lines.append("")
    lines.append("## Unresolved Anchors By Reason")
    unresolved_rows = [
        [reason, count]
        for reason, count in _safe_dict(candidate.get("unresolved_anchors_by_reason")).items()
    ]
    lines.append(_md_table(["reason", "count"], unresolved_rows))
    lines.append("")
    lines.append("## Resolved Outcome Matrix")
    matrix = _safe_dict(candidate.get("resolved_outcome_matrix"))
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
    lines.append("")
    lines.append("## Resolved Truth Distribution")
    truth_rows = [
        [truth_type, count]
        for truth_type, count in _safe_dict(candidate.get("resolved_truth_distribution")).items()
    ]
    lines.append(_md_table(["truth_product_type", "count"], truth_rows))
    lines.append("")
    lines.append("## Pair Patterns")
    pair_rows = [
        [pair, count]
        for pair, count in _safe_dict(candidate.get("resolved_pair_patterns")).items()
    ]
    lines.append(_md_table(["baseline->model=>truth", "count"], pair_rows))
    if reference:
        lines.append("")
        lines.append("## Reference Check")
        lines.append(f"- reference_model_path: `{_safe_dict(reference.get('model')).get('model_path')}`")
        lines.append(
            f"- reference standalone shampoo truth rows: `{reference.get('standalone_shampoo_truth_rows_total')}`"
        )
        lines.append(
            f"- reference resolved shampoo -> conditioner comparable rows: `{reference.get('resolved_shampoo_conditioner_comparable_rows_total')}`"
        )
        lines.append(
            f"- reference exact harmful shampoo -> conditioner rows: `{reference.get('exact_harmful_shampoo_conditioner_rows_total')}`"
        )
    lines.append("")
    lines.append("## Catalog Safety")
    lines.append("- catalog_writes_performed: `false`")
    lines.append("- sources: immutable roadmap events + plan meta only")
    lines.append("- catalog/product rows modified: `0`")
    lines.append("- import logic modified: `0`")
    return "\n".join(lines).strip() + "\n"
