from __future__ import annotations

from datetime import datetime, timezone as dt_timezone
from pathlib import Path
from typing import Any

from roadmap_app.nextstep_targeted_retrain import (
    DEFAULT_HISTORICAL_ANCHOR_COMPARE_REPORT_JSON,
    SOURCE_PREFERENCE_CHOICES,
    materialize_historical_anchor_candidate_comparison_payload,
)


DEFAULT_SKINCARE_FREEZE_QUALIFICATION_REPORT_STEM = (
    Path("reports") / "roadmap_nextstep_v5_skincare_freeze_qualification"
)


def _safe_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _safe_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _pct(value: Any) -> str:
    try:
        if value is None:
            return "n/a"
        return f"{float(value) * 100.0:.2f}%"
    except Exception:
        return "n/a"


def _slice_rows(payload: dict[str, Any], *, category: str, key: str) -> list[dict[str, Any]]:
    return [
        _safe_dict(row)
        for row in _safe_list(payload.get(key))
        if str(_safe_dict(row).get("category") or "") == category
    ]


def _acceptance_gate_lookup(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(_safe_dict(gate).get("name") or ""): _safe_dict(gate)
        for gate in _safe_list(_safe_dict(payload.get("acceptance_gates")).get("gates"))
    }


def _lane_rationale(comparison_payload: dict[str, Any]) -> list[str]:
    broader = _safe_dict(comparison_payload.get("broader_qualification"))
    broader_global = _safe_dict(broader.get("global"))
    per_category = _safe_dict(broader.get("per_category"))
    skincare = _safe_dict(per_category.get("skincare"))
    haircare = _safe_dict(per_category.get("haircare"))
    makeup = _safe_dict(per_category.get("makeup"))
    fragrance = _safe_dict(per_category.get("fragrance"))
    next_focus = [str(category) for category in _safe_list(broader_global.get("next_stage_focus_categories"))]
    return [
        (
            "Skincare is the only current next-stage focus category under recommendation `C`."
            if next_focus == ["skincare"]
            else f"Current next-stage focus categories under freeze: `{', '.join(next_focus) or 'none'}`."
        ),
        f"Skincare direct answer: {skincare.get('direct_answer')}.",
        f"Haircare remains blocked: {haircare.get('direct_answer')}.",
        f"Makeup remains out of scope for next stage: {makeup.get('direct_answer')}.",
        f"Fragrance stays analysis-only: {fragrance.get('direct_answer')}.",
    ]


def _canonical_commands(
    *,
    candidate_model_path: str,
    cached_comparison_json_path: str,
) -> list[str]:
    candidate_arg = f"--candidate-model-path {candidate_model_path}"
    cached_arg = f"--cached-comparison-json {cached_comparison_json_path}"
    return [
        ".\\.venv\\Scripts\\python.exe backend\\manage.py report_roadmap_nextstep_v5_broader_qualification_rerun "
        f"--source-preference fresh_db {candidate_arg} --format both",
        ".\\.venv\\Scripts\\python.exe backend\\manage.py report_roadmap_nextstep_v5_skincare_freeze_qualification "
        f"--source-preference auto {candidate_arg} {cached_arg} --format both",
        ".\\.venv\\Scripts\\python.exe backend\\manage.py test "
        "roadmap_app.tests.RoadmapNextstepHistoricalAnchorComparisonTests "
        "roadmap_app.tests.RoadmapNextstepSkincareFreezeQualificationTests --verbosity 2",
    ]


def build_v5_skincare_freeze_qualification_payload(
    *,
    active_model_path: str | Path,
    retrain_v1_model_path: str | Path,
    candidate_model_path: str | Path,
    days: int = 30,
    source_preference: str = "auto",
    cached_comparison_json_path: str | Path | None = None,
) -> dict[str, Any]:
    comparison_payload = materialize_historical_anchor_candidate_comparison_payload(
        active_model_path=active_model_path,
        retrain_v1_model_path=retrain_v1_model_path,
        candidate_model_path=candidate_model_path,
        days=days,
        source_preference=source_preference,
        cached_comparison_json_path=cached_comparison_json_path or DEFAULT_HISTORICAL_ANCHOR_COMPARE_REPORT_JSON,
    )
    broader = _safe_dict(comparison_payload.get("broader_qualification"))
    broader_global = _safe_dict(broader.get("global"))
    per_category = _safe_dict(broader.get("per_category"))
    skincare = _safe_dict(per_category.get("skincare"))
    haircare = _safe_dict(per_category.get("haircare"))
    makeup = _safe_dict(per_category.get("makeup"))
    fragrance = _safe_dict(per_category.get("fragrance"))
    artifacts = _safe_dict(comparison_payload.get("artifacts"))
    acceptance_lookup = _acceptance_gate_lookup(comparison_payload)
    runtime_guardrails = _safe_dict(comparison_payload.get("runtime_guardrails"))
    runtime_after = _safe_dict(runtime_guardrails.get("after"))
    catalog_safety = _safe_dict(comparison_payload.get("catalog_safety"))
    candidate_model_path_str = str(candidate_model_path)
    cached_path_str = str(
        Path(str(cached_comparison_json_path or DEFAULT_HISTORICAL_ANCHOR_COMPARE_REPORT_JSON)).expanduser().resolve()
    )
    skincare_ready = bool(skincare.get("candidate_for_next_stage_under_freeze"))
    next_focus = [str(category) for category in _safe_list(broader_global.get("next_stage_focus_categories"))]

    return {
        "generated_at_utc": datetime.now(dt_timezone.utc).isoformat(),
        "executive_verdict": {
            "status": (
                "candidate_for_next_freeze_qualification_stage"
                if skincare_ready
                else str(skincare.get("status") or "hold")
            ),
            "recommendation_code": str(broader_global.get("recommendation_code") or ""),
            "recommendation_label": str(broader_global.get("recommendation_label") or ""),
            "lane_category": "skincare",
            "continue_under_freeze": bool(skincare_ready),
            "runtime_enablement_allowed": False,
            "runtime_still_frozen": bool(runtime_after.get("runtime_freeze_ml")),
            "active_runtime_artifact_unchanged": not bool(runtime_guardrails.get("runtime_config_changed")),
            "current_direct_answer": str(skincare.get("direct_answer") or ""),
            "next_stage_focus_categories": next_focus,
            "haircare_blocker_still_present": bool(haircare.get("blocks_next_qualification_phase")),
            "exact_blocker": str(broader_global.get("exact_blocker") or ""),
        },
        "provenance": _safe_dict(comparison_payload.get("report_provenance")),
        "artifact_roles": {
            "active_runtime_continuation_artifact": _safe_dict(artifacts.get("active")),
            "retrain_v1_reference_artifact": _safe_dict(artifacts.get("retrain_v1")),
            "freeze_only_candidate_under_evaluation": _safe_dict(artifacts.get("v5_historical_anchor")),
        },
        "lane_rationale": _lane_rationale(comparison_payload),
        "lane_summary": {
            "skincare": skincare,
            "haircare": haircare,
            "makeup": makeup,
            "fragrance": fragrance,
        },
        "skincare_acceptance_context": {
            "overall_decision_quality_not_worse_than_active": _safe_dict(
                acceptance_lookup.get("overall_decision_quality_not_worse_than_active")
            ),
            "offline_eval_not_materially_worse_than_active": _safe_dict(
                acceptance_lookup.get("offline_eval_not_materially_worse_than_active")
            ),
            "protected_slices_non_regression": _safe_dict(
                acceptance_lookup.get("protected_slices_non_regression")
            ),
        },
        "skincare_targeted_truth_slices": _slice_rows(
            comparison_payload,
            category="skincare",
            key="targeted_truth_slices",
        ),
        "skincare_protected_truth_slices": _slice_rows(
            comparison_payload,
            category="skincare",
            key="protected_truth_slices",
        ),
        "remaining_blockers": _safe_list(broader_global.get("remaining_blockers")),
        "read_only_guards": {
            "catalog_writes_performed": bool(catalog_safety.get("catalog_writes_performed")),
            "runtime_config_changed": bool(runtime_guardrails.get("runtime_config_changed")),
            "runtime_enablement_allowed": False,
        },
        "canonical_commands": _canonical_commands(
            candidate_model_path=candidate_model_path_str,
            cached_comparison_json_path=cached_path_str,
        ),
        "report_paths": {
            "comparison_json": str(DEFAULT_HISTORICAL_ANCHOR_COMPARE_REPORT_JSON.resolve()),
            "skincare_freeze_qualification_report_stem": str(
                DEFAULT_SKINCARE_FREEZE_QUALIFICATION_REPORT_STEM.resolve()
            ),
        },
    }


def render_v5_skincare_freeze_qualification_markdown(payload: dict[str, Any]) -> str:
    executive = _safe_dict(payload.get("executive_verdict"))
    provenance = _safe_dict(payload.get("provenance"))
    roles = _safe_dict(payload.get("artifact_roles"))
    active = _safe_dict(roles.get("active_runtime_continuation_artifact"))
    retrain = _safe_dict(roles.get("retrain_v1_reference_artifact"))
    candidate = _safe_dict(roles.get("freeze_only_candidate_under_evaluation"))
    lane_summary = _safe_dict(payload.get("lane_summary"))
    skincare = _safe_dict(lane_summary.get("skincare"))
    haircare = _safe_dict(lane_summary.get("haircare"))
    acceptance = _safe_dict(payload.get("skincare_acceptance_context"))
    read_only = _safe_dict(payload.get("read_only_guards"))

    lines = [
        "# Roadmap Nextstep v5 Skincare Freeze Qualification",
        "",
        "## Executive Verdict",
        f"- status: `{executive.get('status')}`",
        f"- recommendation: `{executive.get('recommendation_code')}` {executive.get('recommendation_label')}",
        f"- lane category: `{executive.get('lane_category')}`",
        f"- continue under freeze: `{executive.get('continue_under_freeze')}`",
        f"- runtime still frozen: `{executive.get('runtime_still_frozen')}`",
        f"- runtime enablement allowed: `{executive.get('runtime_enablement_allowed')}`",
        f"- active runtime artifact unchanged: `{executive.get('active_runtime_artifact_unchanged')}`",
        f"- current direct answer: {executive.get('current_direct_answer')}",
        f"- next stage focus categories: `{', '.join(_safe_list(executive.get('next_stage_focus_categories'))) or 'none'}`",
        f"- haircare blocker still present: `{executive.get('haircare_blocker_still_present')}`",
        f"- exact blocker: `{executive.get('exact_blocker')}`",
        "",
        "## Artifact Roles",
        f"- active runtime continuation artifact: `{active.get('model_path')}`",
        f"- retrain_v1 reference artifact: `{retrain.get('model_path')}`",
        f"- freeze-only candidate under evaluation: `{candidate.get('model_path')}`",
        "",
        "## Provenance",
        f"- report_materialization: `{provenance.get('report_materialization')}`",
        f"- source_of_truth: `{provenance.get('source_of_truth')}`",
        f"- generated_from: `{provenance.get('generated_from')}`",
        f"- fresh_db_attempted: `{provenance.get('fresh_db_attempted')}`",
        f"- fresh_db_succeeded: `{provenance.get('fresh_db_succeeded')}`",
        f"- cached_artifact_path: `{provenance.get('cached_artifact_path')}`",
        f"- fresh_db_error: `{provenance.get('fresh_db_error')}`",
        "",
        "## Why Skincare Is The Current Next-Phase Lane",
    ]
    for item in _safe_list(payload.get("lane_rationale")):
        lines.append(f"- {item}")

    lines.extend(
        [
            "",
            "## Lane Summary",
            f"- skincare status: `{skincare.get('status')}`",
            f"- skincare rollout reason: `{skincare.get('current_rollout_reason')}`",
            f"- skincare diagnosis: `{_safe_dict(skincare.get('candidate_diagnosis')).get('code')}`",
            f"- skincare vs active model-win delta: `{_pct(_safe_dict(skincare.get('comparison_vs_active')).get('model_win_rate_delta'))}`",
            f"- skincare vs retrain_v1 model-win delta: `{_pct(_safe_dict(skincare.get('comparison_vs_retrain_v1')).get('model_win_rate_delta'))}`",
            f"- haircare status outside this lane: `{haircare.get('status')}`",
            f"- haircare direct answer: {haircare.get('direct_answer')}",
            "",
            "## Skincare Acceptance Context",
        ]
    )
    for gate_name, gate in acceptance.items():
        gate_dict = _safe_dict(gate)
        lines.append(
            f"- {gate_name}: passed=`{gate_dict.get('passed')}` reason=`{gate_dict.get('reason')}`"
        )

    lines.extend(["", "## Skincare Targeted Slices"])
    targeted_rows = _safe_list(payload.get("skincare_targeted_truth_slices"))
    if not targeted_rows:
        lines.append("- none")
    else:
        lines.append("| slice | active_model_win | retrain_v1_model_win | v5_model_win | v5_net |")
        lines.append("| --- | --- | --- | --- | --- |")
        for row in targeted_rows:
            row_dict = _safe_dict(row)
            active_row = _safe_dict(row_dict.get("active"))
            retrain_row = _safe_dict(row_dict.get("retrain_v1"))
            candidate_row = _safe_dict(row_dict.get("v5_historical_anchor"))
            lines.append(
                f"| {row_dict.get('category')}/{row_dict.get('truth_product_type')} | "
                f"{_pct(active_row.get('model_win_rate_vs_truth'))} | "
                f"{_pct(retrain_row.get('model_win_rate_vs_truth'))} | "
                f"{_pct(candidate_row.get('model_win_rate_vs_truth'))} | "
                f"{candidate_row.get('net_wins_model_minus_baseline')} |"
            )

    lines.extend(["", "## Skincare Protected Slices"])
    protected_rows = _safe_list(payload.get("skincare_protected_truth_slices"))
    if not protected_rows:
        lines.append("- none")
    else:
        lines.append("| slice | active_model_win | retrain_v1_model_win | v5_model_win | v5_net |")
        lines.append("| --- | --- | --- | --- | --- |")
        for row in protected_rows:
            row_dict = _safe_dict(row)
            active_row = _safe_dict(row_dict.get("active"))
            retrain_row = _safe_dict(row_dict.get("retrain_v1"))
            candidate_row = _safe_dict(row_dict.get("v5_historical_anchor"))
            lines.append(
                f"| {row_dict.get('category')}/{row_dict.get('truth_product_type')} | "
                f"{_pct(active_row.get('model_win_rate_vs_truth'))} | "
                f"{_pct(retrain_row.get('model_win_rate_vs_truth'))} | "
                f"{_pct(candidate_row.get('model_win_rate_vs_truth'))} | "
                f"{candidate_row.get('net_wins_model_minus_baseline')} |"
            )

    lines.extend(["", "## Remaining Blockers"])
    blockers = _safe_list(payload.get("remaining_blockers"))
    if not blockers:
        lines.append("- none")
    else:
        for blocker in blockers:
            lines.append(f"- {blocker}")

    lines.extend(
        [
            "",
            "## Read-Only Guards",
            f"- catalog writes performed: `{read_only.get('catalog_writes_performed')}`",
            f"- runtime config changed: `{read_only.get('runtime_config_changed')}`",
            f"- runtime enablement allowed: `{read_only.get('runtime_enablement_allowed')}`",
            "",
            "## Canonical Commands",
            "```powershell",
            *[str(command) for command in _safe_list(payload.get("canonical_commands"))],
            "```",
        ]
    )
    return "\n".join(lines)
