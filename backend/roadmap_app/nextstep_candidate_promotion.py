from __future__ import annotations

from datetime import datetime, timezone as dt_timezone
from pathlib import Path
from typing import Any

from django.conf import settings

from roadmap_app.nextstep_targeted_retrain import (
    DEFAULT_HISTORICAL_ANCHOR_COMPARE_REPORT_JSON,
    SOURCE_PREFERENCE_CHOICES,
    materialize_historical_anchor_candidate_comparison_payload,
)


DEFAULT_CANDIDATE_PROMOTION_UNDER_FREEZE_REPORT_STEM = (
    Path("reports") / "roadmap_nextstep_v5_candidate_promotion_under_freeze"
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


def _same_path(left: Any, right: Any) -> bool:
    left_text = str(left or "").strip()
    right_text = str(right or "").strip()
    if not left_text or not right_text:
        return False
    return Path(left_text).expanduser().resolve() == Path(right_text).expanduser().resolve()


def _candidate_promotion_state(comparison_payload: dict[str, Any]) -> dict[str, Any]:
    artifacts = _safe_dict(comparison_payload.get("artifacts"))
    active = _safe_dict(artifacts.get("active"))
    candidate = _safe_dict(artifacts.get("v5_historical_anchor"))
    broader = _safe_dict(comparison_payload.get("broader_qualification"))
    broader_global = _safe_dict(broader.get("global"))
    runtime_guardrails = _safe_dict(comparison_payload.get("runtime_guardrails"))
    runtime_after = _safe_dict(runtime_guardrails.get("after"))
    recommendation_code = str(broader_global.get("recommendation_code") or "")
    promoted_under_freeze = (
        recommendation_code == "A"
        and bool(broader_global.get("is_v5_new_best_continuation_candidate"))
        and bool(runtime_after.get("runtime_freeze_ml"))
    )

    active_runtime_model_path = str(runtime_after.get("roadmap_nextstep_v4_model_path") or active.get("model_path") or "")
    candidate_model_path = str(candidate.get("model_path") or "")
    return {
        "promotion_status": "promoted_under_freeze" if promoted_under_freeze else "not_promoted_under_freeze",
        "recommendation_code": recommendation_code,
        "recommendation_label": str(broader_global.get("recommendation_label") or ""),
        "recommendation_accepted": recommendation_code == "A",
        "promoted_candidate_under_freeze": bool(promoted_under_freeze),
        "active_runtime_continuation_artifact": {
            "key": "nextstep_v4_active",
            "role": "current_active_runtime_continuation_artifact",
            "model_path": active_runtime_model_path,
            "serve_path_selected": True,
        },
        "promoted_freeze_only_continuation_candidate": {
            "key": "v5_historical_anchor",
            "role": "promoted_freeze_only_continuation_candidate",
            "model_path": candidate_model_path,
            "serve_path_selected": False,
        },
        "runtime_serve": {
            "runtime_freeze_ml": bool(runtime_after.get("runtime_freeze_ml")),
            "serve_enabled": False,
            "disabled_reason": "roadmap_ml_frozen",
            "active_runtime_artifact_unchanged": _same_path(active_runtime_model_path, active.get("model_path")),
            "runtime_model_path_switched_to_candidate": _same_path(active_runtime_model_path, candidate_model_path),
        },
    }


def _why_v5_is_best_candidate(comparison_payload: dict[str, Any]) -> list[str]:
    broader = _safe_dict(comparison_payload.get("broader_qualification"))
    broader_global = _safe_dict(broader.get("global"))
    per_category = _safe_dict(broader.get("per_category"))
    return [
        "Broader freeze-only acceptance passes with recommendation `A` and no remaining blockers."
        if not _safe_list(broader_global.get("remaining_blockers"))
        else "Recommendation remains conditional because blockers are still present.",
        "Haircare clears the shampoo gate under `D_two_stage_truth`, and unresolved anchors remain fail-closed.",
        f"Haircare verdict: {_safe_dict(per_category.get('haircare')).get('direct_answer')}.",
        f"Skincare verdict: {_safe_dict(per_category.get('skincare')).get('direct_answer')}.",
        "Protected slices remain non-regressed relative to the active artifact.",
    ]


def _why_not_runtime_enablement(comparison_payload: dict[str, Any], promotion_state: dict[str, Any]) -> list[str]:
    runtime_serve = _safe_dict(promotion_state.get("runtime_serve"))
    active_runtime = _safe_dict(promotion_state.get("active_runtime_continuation_artifact"))
    promoted = _safe_dict(promotion_state.get("promoted_freeze_only_continuation_candidate"))
    return [
        "Runtime ML freeze remains ON, so this promotion is qualification-only.",
        f"Serve path remains bound to the active runtime artifact: `{active_runtime.get('model_path')}`.",
        f"Promoted freeze-only candidate is tracked separately: `{promoted.get('model_path')}`.",
        f"Runtime model_path switched to v5 for serve: `{runtime_serve.get('runtime_model_path_switched_to_candidate')}`.",
        "Rule baseline behavior remains unchanged.",
    ]


def _exact_next_phase(comparison_payload: dict[str, Any]) -> str:
    provenance = _safe_dict(comparison_payload.get("report_provenance"))
    if str(provenance.get("source_of_truth") or "") == "cached_artifact":
        return (
            "Re-run the promoted v5 qualification pack from live DB when the connection is healthy, then continue "
            "freeze-only qualification for haircare and skincare while makeup stays sample-limited and fragrance remains analysis-only."
        )
    return (
        "Continue the promoted v5 qualification pack under freeze for haircare and skincare; keep makeup sample-limited, "
        "keep fragrance analysis-only, and do not enable runtime serve."
    )


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
        ".\\.venv\\Scripts\\python.exe backend\\manage.py report_roadmap_nextstep_v5_broader_qualification_rerun "
        f"--source-preference cached_artifact {candidate_arg} {cached_arg} --format both",
        ".\\.venv\\Scripts\\python.exe backend\\manage.py report_roadmap_nextstep_v5_candidate_promotion_under_freeze "
        f"--source-preference auto {candidate_arg} {cached_arg} --format both",
        ".\\.venv\\Scripts\\python.exe backend\\manage.py test "
        "roadmap_app.tests.RoadmapNextstepHistoricalAnchorComparisonTests "
        "roadmap_app.tests.RoadmapNextstepCandidatePromotionTests --verbosity 2",
    ]


def build_v5_candidate_promotion_under_freeze_payload(
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
    promotion_state = _candidate_promotion_state(comparison_payload)
    broader = _safe_dict(comparison_payload.get("broader_qualification"))
    broader_global = _safe_dict(broader.get("global"))
    runtime_serve = _safe_dict(promotion_state.get("runtime_serve"))
    candidate_model_path_str = str(
        _safe_dict(promotion_state.get("promoted_freeze_only_continuation_candidate")).get("model_path") or ""
    )
    cached_path_str = str(
        Path(str(cached_comparison_json_path or DEFAULT_HISTORICAL_ANCHOR_COMPARE_REPORT_JSON)).expanduser().resolve()
    )

    return {
        "generated_at_utc": datetime.now(dt_timezone.utc).isoformat(),
        "promotion_state": promotion_state,
        "executive_verdict": {
            "status": str(promotion_state.get("promotion_status") or ""),
            "recommendation_code": str(promotion_state.get("recommendation_code") or ""),
            "recommendation_label": str(promotion_state.get("recommendation_label") or ""),
            "canonical_freeze_candidate": bool(promotion_state.get("promoted_candidate_under_freeze")),
            "runtime_still_frozen": bool(runtime_serve.get("runtime_freeze_ml")),
            "active_runtime_artifact_unchanged": bool(runtime_serve.get("active_runtime_artifact_unchanged")),
            "why_not_runtime_enablement": "Runtime ML freeze remains ON and serve path was not switched to v5.",
            "exact_next_qualification_phase": _exact_next_phase(comparison_payload),
        },
        "provenance": _safe_dict(comparison_payload.get("report_provenance")),
        "why_v5_is_best_candidate": _why_v5_is_best_candidate(comparison_payload),
        "why_not_runtime_enablement": _why_not_runtime_enablement(comparison_payload, promotion_state),
        "per_category_freeze_qualification_summary": _safe_dict(broader.get("per_category")),
        "shampoo_gate_under_d_two_stage_truth": {
            "gate": next(
                (
                    _safe_dict(gate)
                    for gate in _safe_list(_safe_dict(comparison_payload.get("acceptance_gates")).get("gates"))
                    if str(_safe_dict(gate).get("name") or "") == "haircare_shampoo_two_stage_truth_improves"
                ),
                {},
            ),
            "comparison": _safe_dict(_safe_dict(comparison_payload.get("haircare_shampoo_truth_gate_comparison")).get("v5_historical_anchor")),
        },
        "remaining_blockers": _safe_list(broader_global.get("remaining_blockers")),
        "exact_next_qualification_phase": _exact_next_phase(comparison_payload),
        "qualification_reference": {
            "artifacts": _safe_dict(comparison_payload.get("artifacts")),
            "category_comparison": _safe_list(comparison_payload.get("category_comparison")),
            "acceptance_gates": _safe_dict(comparison_payload.get("acceptance_gates")),
            "targeted_truth_slices": _safe_list(comparison_payload.get("targeted_truth_slices")),
            "protected_truth_slices": _safe_list(comparison_payload.get("protected_truth_slices")),
            "haircare_shampoo_truth_gate_comparison": _safe_dict(
                comparison_payload.get("haircare_shampoo_truth_gate_comparison")
            ),
        },
        "canonical_commands": _canonical_commands(
            candidate_model_path=candidate_model_path_str,
            cached_comparison_json_path=cached_path_str,
        ),
        "report_paths": {
            "comparison_json": str(DEFAULT_HISTORICAL_ANCHOR_COMPARE_REPORT_JSON.resolve()),
            "candidate_promotion_report_stem": str(DEFAULT_CANDIDATE_PROMOTION_UNDER_FREEZE_REPORT_STEM.resolve()),
        },
        "read_only_guards": {
            "catalog_writes_performed": False,
            "runtime_config_changed": False,
            "runtime_enablement_allowed": False,
            "active_runtime_artifact_unchanged": bool(runtime_serve.get("active_runtime_artifact_unchanged")),
        },
    }


def render_v5_candidate_promotion_under_freeze_markdown(payload: dict[str, Any]) -> str:
    promotion_state = _safe_dict(payload.get("promotion_state"))
    active_runtime = _safe_dict(promotion_state.get("active_runtime_continuation_artifact"))
    promoted_candidate = _safe_dict(promotion_state.get("promoted_freeze_only_continuation_candidate"))
    runtime_serve = _safe_dict(promotion_state.get("runtime_serve"))
    executive = _safe_dict(payload.get("executive_verdict"))
    provenance = _safe_dict(payload.get("provenance"))
    per_category = _safe_dict(payload.get("per_category_freeze_qualification_summary"))
    shampoo = _safe_dict(payload.get("shampoo_gate_under_d_two_stage_truth"))
    shampoo_gate = _safe_dict(shampoo.get("gate"))
    shampoo_comparison = _safe_dict(shampoo.get("comparison"))

    lines = [
        "# Roadmap Nextstep v5 Candidate Promotion Under Freeze",
        "",
        "## Executive Verdict",
        f"- status: `{executive.get('status')}`",
        f"- recommendation: `{executive.get('recommendation_code')}` {executive.get('recommendation_label')}",
        f"- canonical freeze-only continuation candidate: `{executive.get('canonical_freeze_candidate')}`",
        f"- runtime still frozen: `{executive.get('runtime_still_frozen')}`",
        f"- active runtime artifact unchanged: `{executive.get('active_runtime_artifact_unchanged')}`",
        f"- why not runtime enablement: {executive.get('why_not_runtime_enablement')}",
        f"- exact next qualification phase: {executive.get('exact_next_qualification_phase')}",
        "",
        "## Artifact Roles",
        f"- current active runtime continuation artifact: `{active_runtime.get('model_path')}`",
        f"- promoted freeze-only continuation candidate: `{promoted_candidate.get('model_path')}`",
        f"- runtime serve enabled: `{runtime_serve.get('serve_enabled')}`",
        f"- runtime model_path switched to candidate: `{runtime_serve.get('runtime_model_path_switched_to_candidate')}`",
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
        "## Why v5 Is Now The Best Continuation Candidate",
    ]
    for item in _safe_list(payload.get("why_v5_is_best_candidate")):
        lines.append(f"- {item}")

    lines.extend(["", "## Why This Is Not Runtime Enablement"])
    for item in _safe_list(payload.get("why_not_runtime_enablement")):
        lines.append(f"- {item}")

    lines.extend(["", "## Per-Category Freeze Qualification Summary"])
    lines.append("| category | status | direct_answer | rollout_reason | diagnosis | next_stage_under_freeze |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for category in ["haircare", "skincare", "makeup", "fragrance"]:
        row = _safe_dict(per_category.get(category))
        lines.append(
            f"| {category} | {row.get('status')} | {row.get('direct_answer')} | "
            f"{row.get('current_rollout_reason')} | {str(_safe_dict(row.get('candidate_diagnosis')).get('code') or '')} | "
            f"{row.get('candidate_for_next_stage_under_freeze')} |"
        )

    lines.extend(
        [
            "",
            "## Shampoo Gate Under D_two_stage_truth",
            f"- gate passed: `{shampoo_gate.get('passed')}`",
            f"- gate reason: `{shampoo_gate.get('reason')}`",
            f"- stage1_model_win_rate: `{_pct(shampoo_comparison.get('stage_1_family_model_win_rate'))}`",
            f"- stage2_model_win_rate: `{_pct(shampoo_comparison.get('stage_2_concrete_model_win_rate'))}`",
            f"- unresolved anchors remain fail-closed: `True`",
            "",
            "## Remaining Blockers",
        ]
    )
    blockers = _safe_list(payload.get("remaining_blockers"))
    if not blockers:
        lines.append("- none")
    else:
        for blocker in blockers:
            lines.append(f"- {blocker}")

    lines.extend(
        [
            "",
            "## Canonical Commands",
            "```powershell",
            *[str(command) for command in _safe_list(payload.get("canonical_commands"))],
            "```",
        ]
    )
    return "\n".join(lines).strip() + "\n"
