from __future__ import annotations

from datetime import datetime, timezone as dt_timezone
from pathlib import Path
from typing import Any

from django.conf import settings
from django.db.utils import DatabaseError, OperationalError

from roadmap_app.integrity import (
    active_fragrance_runtime_integrity_counts,
    legacy_bad_fragrance_completion_details,
)
from roadmap_app.nextstep_candidate_promotion import (
    build_v5_candidate_promotion_under_freeze_payload,
)
from roadmap_app.ml_artifact_proof import (
    PROOF_FILE_EVAL,
    PROOF_FILE_METADATA,
    PROOF_FILE_SHADOW,
    PROOF_FILE_UPLIFT_30D,
    PROOF_FILE_UPLIFT_7D,
    artifact_dir_for_model_path,
    artifact_file_path,
    load_json_file,
    proof_bundle_status,
)
from roadmap_app.ml_next_step import (
    nextstep_model_artifact_summary,
    v4_category_staged_rollout_status,
    v4_min_lift_guard_status,
)
from roadmap_app.ml_planner import (
    planner_model_artifact_summary,
    planner_runtime_guard_status,
)

RUNTIME_CATEGORIES = ["skincare", "haircare", "makeup", "fragrance"]
PASS = "PASS"
HOLD = "HOLD"
DISABLE = "DISABLE"
DEFAULT_RETRAIN_V1_MODEL_PATH = "models/roadmap_next_step_v4_targeted_retrain_v1/model.pkl"
DEFAULT_V5_PROMOTED_CANDIDATE_MODEL_PATH = "models/roadmap_next_step_v5_historical_anchor_targeted_v1/model.pkl"


def _safe_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _safe_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _normalized_category_set(value: Any) -> set[str]:
    out: set[str] = set()
    if isinstance(value, str):
        for token in value.split(","):
            normalized = str(token or "").strip().lower()
            if normalized:
                out.add(normalized)
        return out
    if isinstance(value, (list, tuple, set)):
        for token in value:
            normalized = str(token or "").strip().lower()
            if normalized:
                out.add(normalized)
    return out


def _artifact_expected_model_version(model_path: str | Path | None) -> str | None:
    metadata = load_json_file(artifact_file_path(model_path, PROOF_FILE_METADATA))
    version = str(_safe_dict(metadata).get("model_version") or "").strip()
    return version or None


def _generic_artifact_proof(
    *,
    model_path: str | Path | None,
    required_files: list[str],
    optional_files: list[str] | None = None,
) -> dict[str, Any]:
    return proof_bundle_status(
        model_path=model_path,
        required_files=required_files,
        optional_files=optional_files or [],
        expected_model_version=_artifact_expected_model_version(model_path),
    )


def _artifact_entry(
    *,
    key: str,
    lane: str,
    role: str,
    model_path: str | Path | None,
    required_files: list[str],
    categories: list[str] | None = None,
    settings_keys: list[str] | None = None,
    enabled: bool = True,
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    raw_path = str(model_path or "").strip()
    proof = _generic_artifact_proof(
        model_path=raw_path,
        required_files=required_files,
    )
    summary = {}
    if lane == "planner_v1":
        summary = planner_model_artifact_summary(raw_path)
    elif lane.startswith("nextstep_v4"):
        summary = nextstep_model_artifact_summary(raw_path)
    return {
        "key": key,
        "lane": lane,
        "role": role,
        "enabled": bool(enabled),
        "model_path": raw_path,
        "artifact_dir": str(artifact_dir_for_model_path(raw_path)) if raw_path else "",
        "categories": list(categories or []),
        "settings_keys": list(settings_keys or []),
        "required_files": list(required_files),
        "proof_bundle": proof,
        "artifact_summary": summary,
        "details": dict(details or {}),
    }


def configured_runtime_artifacts() -> list[dict[str, Any]]:
    planner_categories = sorted(
        _normalized_category_set(getattr(settings, "ROADMAP_PLANNER_V1_ENABLED_CATEGORIES", []))
        or set(RUNTIME_CATEGORIES)
    )
    nextstep_enabled_categories = sorted(
        _normalized_category_set(getattr(settings, "ROADMAP_NEXTSTEP_V4_ENABLED_CATEGORIES", []))
        or set(RUNTIME_CATEGORIES)
    )

    entries: list[dict[str, Any]] = [
        _artifact_entry(
            key="planner_v1",
            lane="planner_v1",
            role="runtime_lane",
            model_path=str(getattr(settings, "ROADMAP_PLANNER_V1_MODEL_PATH", "") or "").strip(),
            required_files=[PROOF_FILE_METADATA, PROOF_FILE_EVAL, PROOF_FILE_SHADOW],
            categories=planner_categories,
            settings_keys=[
                "ROADMAP_PLANNER_V1_MODEL_PATH",
                "ROADMAP_PLANNER_V1_MODE",
                "ROADMAP_PLANNER_V1_ENABLED_CATEGORIES",
            ],
            enabled=str(getattr(settings, "ROADMAP_PLANNER_V1_MODE", "off") or "off").strip().lower() in {"shadow", "serve"},
        ),
        _artifact_entry(
            key="nextstep_v4_active",
            lane="nextstep_v4_active",
            role="serve_candidate",
            model_path=str(getattr(settings, "ROADMAP_NEXTSTEP_V4_MODEL_PATH", "") or "").strip(),
            required_files=[PROOF_FILE_METADATA, PROOF_FILE_EVAL, PROOF_FILE_UPLIFT_7D, PROOF_FILE_UPLIFT_30D],
            categories=nextstep_enabled_categories,
            settings_keys=[
                "ROADMAP_NEXTSTEP_V4_MODEL_PATH",
                "ROADMAP_NEXTSTEP_V4_ENABLED",
                "ROADMAP_NEXTSTEP_V4_ENABLED_CATEGORIES",
                "ROADMAP_NEXTSTEP_V4_DISABLED_CATEGORIES",
            ],
            enabled=bool(getattr(settings, "ROADMAP_NEXTSTEP_V4_ENABLED", False)),
        ),
    ]

    shadow_model_path = str(getattr(settings, "ROADMAP_NEXTSTEP_V4_SHADOW_MODEL_PATH", "") or "").strip()
    if shadow_model_path:
        entries.append(
            _artifact_entry(
                key="nextstep_v4_shadow",
                lane="nextstep_v4_shadow",
                role="shadow_candidate",
                model_path=shadow_model_path,
                required_files=[PROOF_FILE_METADATA, PROOF_FILE_EVAL, PROOF_FILE_SHADOW],
                categories=list(RUNTIME_CATEGORIES),
                settings_keys=["ROADMAP_NEXTSTEP_V4_SHADOW_MODEL_PATH"],
                enabled=True,
            )
        )

    partial_model_path = str(getattr(settings, "ROADMAP_NEXTSTEP_V4_PARTIAL_MODEL_PATH", "") or "").strip()
    if partial_model_path:
        entries.append(
            _artifact_entry(
                key="nextstep_v4_partial_default",
                lane="nextstep_v4_partial_default",
                role="partial_candidate",
                model_path=partial_model_path,
                required_files=[PROOF_FILE_METADATA, PROOF_FILE_EVAL, PROOF_FILE_UPLIFT_7D, PROOF_FILE_UPLIFT_30D],
                categories=sorted(
                    _normalized_category_set(
                        getattr(settings, "ROADMAP_NEXTSTEP_V4_PARTIAL_ENABLED_CATEGORIES", [])
                    )
                ),
                settings_keys=[
                    "ROADMAP_NEXTSTEP_V4_PARTIAL_MODEL_PATH",
                    "ROADMAP_NEXTSTEP_V4_PARTIAL_ENABLED_CATEGORIES",
                ],
                enabled=True,
            )
        )

    partial_haircare_model_path = str(
        getattr(settings, "ROADMAP_NEXTSTEP_V4_PARTIAL_HAIRCARE_MODEL_PATH", "") or ""
    ).strip()
    if partial_haircare_model_path:
        entries.append(
            _artifact_entry(
                key="nextstep_v4_partial_haircare",
                lane="nextstep_v4_partial_haircare",
                role="partial_candidate",
                model_path=partial_haircare_model_path,
                required_files=[PROOF_FILE_METADATA, PROOF_FILE_EVAL, PROOF_FILE_UPLIFT_7D, PROOF_FILE_UPLIFT_30D],
                categories=["haircare"],
                settings_keys=[
                    "ROADMAP_NEXTSTEP_V4_PARTIAL_HAIRCARE_MODEL_PATH",
                    "ROADMAP_NEXTSTEP_V4_PARTIAL_ENABLED_CATEGORIES",
                ],
                enabled=True,
            )
        )

    teacher_corechain_enabled = bool(
        getattr(settings, "ROADMAP_NEXTSTEP_V4_HAIRCARE_CORECHAIN_TEACHER_RERANK_ENABLED", False)
    )
    teacher_corechain_model_path = str(
        getattr(settings, "ROADMAP_NEXTSTEP_V4_HAIRCARE_CORECHAIN_TEACHER_MODEL_PATH", "") or ""
    ).strip()
    if teacher_corechain_enabled and teacher_corechain_model_path:
        entries.append(
            _artifact_entry(
                key="nextstep_v4_haircare_corechain_teacher",
                lane="nextstep_v4_haircare_corechain_teacher",
                role="runtime_overlay",
                model_path=teacher_corechain_model_path,
                required_files=[PROOF_FILE_METADATA, PROOF_FILE_EVAL],
                categories=["haircare"],
                settings_keys=[
                    "ROADMAP_NEXTSTEP_V4_HAIRCARE_CORECHAIN_TEACHER_MODEL_PATH",
                    "ROADMAP_NEXTSTEP_V4_HAIRCARE_CORECHAIN_TEACHER_RERANK_ENABLED",
                ],
                enabled=True,
            )
        )

    teacher_scalp_enabled = bool(
        getattr(settings, "ROADMAP_NEXTSTEP_V4_HAIRCARE_SCALP_TEACHER_RERANK_ENABLED", False)
    )
    teacher_scalp_model_path = str(
        getattr(settings, "ROADMAP_NEXTSTEP_V4_HAIRCARE_SCALP_TEACHER_MODEL_PATH", "") or ""
    ).strip()
    if teacher_scalp_enabled and teacher_scalp_model_path:
        entries.append(
            _artifact_entry(
                key="nextstep_v4_haircare_scalp_teacher",
                lane="nextstep_v4_haircare_scalp_teacher",
                role="runtime_overlay",
                model_path=teacher_scalp_model_path,
                required_files=[PROOF_FILE_METADATA, PROOF_FILE_EVAL],
                categories=["haircare"],
                settings_keys=[
                    "ROADMAP_NEXTSTEP_V4_HAIRCARE_SCALP_TEACHER_MODEL_PATH",
                    "ROADMAP_NEXTSTEP_V4_HAIRCARE_SCALP_TEACHER_RERANK_ENABLED",
                ],
                enabled=True,
            )
        )

    return entries


def planner_pass_fail_manifest() -> dict[str, Any]:
    by_category: dict[str, Any] = {}
    for category in RUNTIME_CATEGORIES:
        guard = planner_runtime_guard_status(category, require_shadow_report=True)
        if not bool(guard.get("enabled_for_category")):
            status = DISABLE
        elif not bool(_safe_dict(guard.get("proof_bundle")).get("required_complete")):
            status = HOLD
        elif not bool(_safe_dict(guard.get("eval_guard")).get("passed")):
            status = HOLD
        else:
            status = PASS
        by_category[category] = {
            "status": status,
            "reason": str(guard.get("reason") or ""),
            "model_path": str(guard.get("model_path") or ""),
            "proof_reason": str(_safe_dict(guard.get("proof_bundle")).get("reason") or ""),
            "eval_reason": str(_safe_dict(guard.get("eval_guard")).get("reason") or ""),
        }
    return {
        "lane": "planner_v1",
        "model_path": str(getattr(settings, "ROADMAP_PLANNER_V1_MODEL_PATH", "") or "").strip(),
        "by_category": by_category,
    }


def nextstep_pass_fail_manifest(model_path: str | Path | None = None) -> dict[str, Any]:
    resolved_model_path = str(model_path or getattr(settings, "ROADMAP_NEXTSTEP_V4_MODEL_PATH", "") or "").strip()
    lift_guard = v4_min_lift_guard_status(resolved_model_path)
    by_category: dict[str, Any] = {}
    for category in RUNTIME_CATEGORIES:
        staged = v4_category_staged_rollout_status(category, model_path=resolved_model_path)
        rollout = _safe_dict(staged.get("rollout"))
        if not bool(rollout.get("passed")):
            status = DISABLE
            reason = str(staged.get("reason") or rollout.get("reason") or "category_disabled")
        elif not bool(lift_guard.get("passed")):
            status = HOLD
            reason = str(lift_guard.get("reason") or "missing_eval_report")
        else:
            final_status = str(staged.get("final_status") or HOLD).upper()
            status = PASS if final_status == "ENABLE" else HOLD
            reason = str(staged.get("hold_reason") or staged.get("reason") or final_status.lower())
        by_category[category] = {
            "status": status,
            "reason": reason,
            "model_path": resolved_model_path,
            "guard_eval_path": str(lift_guard.get("eval_path") or ""),
            "guard_report_7d": str(staged.get("source_report_path_7d") or ""),
            "guard_report_30d": str(staged.get("source_report_path_30d") or ""),
        }
    return {
        "lane": "nextstep_v4",
        "model_path": resolved_model_path,
        "by_category": by_category,
        "lift_guard": lift_guard,
    }


def _shadow_evidence_count(payload: dict[str, Any] | None) -> int | None:
    report = _safe_dict(payload)
    overall = _safe_dict(report.get("overall"))
    if overall.get("eligible_plans") is not None:
        try:
            return int(overall.get("eligible_plans") or 0)
        except Exception:
            return 0
    top1 = _safe_dict(report.get("top1_comparison"))
    if top1.get("eligible_plans") is not None:
        try:
            return int(top1.get("eligible_plans") or 0)
        except Exception:
            return 0
    shadow = _safe_dict(report.get("shadow"))
    shadow_top1 = _safe_dict(shadow.get("top1_comparison"))
    if shadow_top1.get("eligible_plans") is not None:
        try:
            return int(shadow_top1.get("eligible_plans") or 0)
        except Exception:
            return 0
    return None


def _shadow_consideration_rows(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for entry in entries:
        required_files = set(entry.get("required_files") or [])
        if PROOF_FILE_SHADOW not in required_files:
            continue
        proof = _safe_dict(entry.get("proof_bundle"))
        shadow_payload = load_json_file(artifact_file_path(entry.get("model_path"), PROOF_FILE_SHADOW))
        evidence_count = _shadow_evidence_count(shadow_payload)
        eligible = bool(proof.get("required_complete")) and (evidence_count is None or evidence_count > 0)
        reason = "proof_complete"
        if not bool(proof.get("required_complete")):
            reason = str(proof.get("reason") or "missing_shadow_proof")
        elif evidence_count == 0:
            reason = "shadow_report_has_no_eligible_plans"
        out.append(
            {
                "key": str(entry.get("key") or ""),
                "lane": str(entry.get("lane") or ""),
                "role": str(entry.get("role") or ""),
                "model_path": str(entry.get("model_path") or ""),
                "eligible": eligible,
                "reason": reason,
                "eligible_plans": evidence_count,
            }
        )
    return out


def _missing_proof_items(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for entry in entries:
        proof = _safe_dict(entry.get("proof_bundle"))
        files = _safe_dict(proof.get("files"))
        for filename in entry.get("required_files") or []:
            file_state = _safe_dict(files.get(filename))
            if str(file_state.get("status") or "missing") == "ok":
                continue
            rows.append(
                {
                    "artifact_key": str(entry.get("key") or ""),
                    "lane": str(entry.get("lane") or ""),
                    "role": str(entry.get("role") or ""),
                    "model_path": str(entry.get("model_path") or ""),
                    "file": str(filename),
                    "status": str(file_state.get("status") or "missing"),
                    "reason": str(file_state.get("reason") or ""),
                    "path": str(file_state.get("path") or ""),
                }
            )
    rows.sort(key=lambda row: (row["artifact_key"], row["file"]))
    return rows


def freeze_candidate_promotion_manifest() -> dict[str, Any]:
    active_model_path = str(getattr(settings, "ROADMAP_NEXTSTEP_V4_MODEL_PATH", "") or "").strip()
    retrain_v1_model_path = DEFAULT_RETRAIN_V1_MODEL_PATH
    candidate_model_path = DEFAULT_V5_PROMOTED_CANDIDATE_MODEL_PATH
    try:
        payload = build_v5_candidate_promotion_under_freeze_payload(
            active_model_path=active_model_path,
            retrain_v1_model_path=retrain_v1_model_path,
            candidate_model_path=candidate_model_path,
            source_preference="auto",
        )
        promotion_state = _safe_dict(payload.get("promotion_state"))
        executive = _safe_dict(payload.get("executive_verdict"))
        provenance = _safe_dict(payload.get("provenance"))
        return {
            "status": "available",
            "promotion_state": promotion_state,
            "executive_verdict": executive,
            "provenance": provenance,
            "report_paths": _safe_dict(payload.get("report_paths")),
            "read_only_guards": _safe_dict(payload.get("read_only_guards")),
        }
    except Exception as exc:
        return {
            "status": "unavailable",
            "reason": f"{type(exc).__module__}.{type(exc).__name__}: {exc}",
            "promotion_state": {
                "active_runtime_continuation_artifact": {
                    "model_path": active_model_path,
                },
                "promoted_freeze_only_continuation_candidate": {
                    "model_path": candidate_model_path,
                },
                "runtime_serve": {
                    "runtime_freeze_ml": bool(getattr(settings, "ROADMAP_RUNTIME_FREEZE_ML", True)),
                    "serve_enabled": False,
                },
            },
            "executive_verdict": {
                "status": "promotion_status_unavailable",
                "canonical_freeze_candidate": False,
                "runtime_still_frozen": bool(getattr(settings, "ROADMAP_RUNTIME_FREEZE_ML", True)),
                "active_runtime_artifact_unchanged": True,
            },
            "provenance": {
                "source_of_truth": "unavailable",
                "report_materialization": "unavailable",
                "generated_from": "unavailable",
            },
            "report_paths": {},
            "read_only_guards": {
                "catalog_writes_performed": False,
                "runtime_config_changed": False,
                "runtime_enablement_allowed": False,
            },
        }


def _safe_fragrance_slot_qualification() -> dict[str, Any]:
    try:
        return {
            "runtime": active_fragrance_runtime_integrity_counts(),
            "legacy": legacy_bad_fragrance_completion_details(recent_days=30),
            "source_of_truth": "live_db",
            "db_error": "",
        }
    except (OperationalError, DatabaseError) as exc:
        error = f"{type(exc).__module__}.{type(exc).__name__}: {exc}"
        return {
            "runtime": {"status": "unavailable", "reason": "db_unavailable", "error": error},
            "legacy": {"status": "unavailable", "reason": "db_unavailable", "error": error},
            "source_of_truth": "db_unavailable",
            "db_error": error,
        }


def build_roadmap_ml_artifact_qualification_payload() -> dict[str, Any]:
    artifacts = configured_runtime_artifacts()
    planner_manifest = planner_pass_fail_manifest()
    nextstep_manifest = nextstep_pass_fail_manifest()
    candidate_promotion = freeze_candidate_promotion_manifest()
    shadow_rows = _shadow_consideration_rows(artifacts)
    missing_items = _missing_proof_items(artifacts)
    fragrance = _safe_fragrance_slot_qualification()

    any_future_shadow_eligible = any(bool(row.get("eligible")) for row in shadow_rows)
    any_future_serve_eligible = any(
        str(row.get("status") or "") == PASS
        for row in list(_safe_dict(planner_manifest.get("by_category")).values())
        + list(_safe_dict(nextstep_manifest.get("by_category")).values())
    )

    continuation_model_root = (
        Path(__file__).resolve().parents[2]
        / "models"
        / "roadmap_continuation_planner_v2_after_runtime_patch"
    ).resolve()

    return {
        "generated_at_utc": datetime.now(dt_timezone.utc).isoformat(),
        "runtime_freeze_ml": bool(getattr(settings, "ROADMAP_RUNTIME_FREEZE_ML", True)),
        "configured_artifacts": artifacts,
        "per_category_manifest": {
            "planner_v1": planner_manifest,
            "nextstep_v4": nextstep_manifest,
        },
        "freeze_candidate_promotion": candidate_promotion,
        "missing_proof_items": missing_items,
        "future_consideration": {
            "shadow_candidates": shadow_rows,
            "any_future_shadow_eligible": any_future_shadow_eligible,
            "any_future_serve_eligible": any_future_serve_eligible,
        },
        "fragrance_slot_qualification": fragrance,
        "continuation_runtime_candidate": {
            "model_root": str(continuation_model_root),
            "shadow_report_json": str((continuation_model_root / "shadow_report.json").resolve()),
            "shadow_report_md": str((continuation_model_root / "shadow_report.md").resolve()),
        },
        "canonical_commands": [
            ".\\.venv\\Scripts\\python.exe backend\\manage.py report_roadmap_planner_v1_shadow",
            ".\\.venv\\Scripts\\python.exe backend\\manage.py report_roadmap_nextstep_v4_artifact_eval --model-path models\\roadmap_next_step_v4\\model.pkl",
            ".\\.venv\\Scripts\\python.exe backend\\manage.py report_roadmap_rollout_decision",
            ".\\.venv\\Scripts\\python.exe backend\\manage.py backfill_roadmap_shadow_meta --days 30 --category all --replay-mode historical_anchors --model-path models\\roadmap_next_step_v4\\model.pkl --write",
            ".\\.venv\\Scripts\\python.exe backend\\manage.py backfill_roadmap_shadow_meta --days 30 --category all --replay-mode historical_anchors --model-path models\\roadmap_next_step_v4_semantic_v4\\model.pkl --write",
            ".\\.venv\\Scripts\\python.exe backend\\manage.py report_roadmap_ml_uplift --days 7 --category all --format both --evidence-source historical_replay --model-path models\\roadmap_next_step_v4\\model.pkl --out models\\roadmap_next_step_v4\\uplift_report_7d",
            ".\\.venv\\Scripts\\python.exe backend\\manage.py report_roadmap_ml_uplift --days 30 --category all --format both --evidence-source historical_replay --model-path models\\roadmap_next_step_v4\\model.pkl --out models\\roadmap_next_step_v4\\uplift_report_30d",
            ".\\.venv\\Scripts\\python.exe backend\\manage.py report_roadmap_ml_uplift --days 7 --category all --format both --evidence-source historical_replay --model-path models\\roadmap_next_step_v4_semantic_v4\\model.pkl --out models\\roadmap_next_step_v4_semantic_v4\\uplift_report_7d",
            ".\\.venv\\Scripts\\python.exe backend\\manage.py report_roadmap_ml_uplift --days 30 --category all --format both --evidence-source historical_replay --model-path models\\roadmap_next_step_v4_semantic_v4\\model.pkl --out models\\roadmap_next_step_v4_semantic_v4\\uplift_report_30d",
            ".\\.venv\\Scripts\\python.exe backend\\manage.py report_roadmap_ml_diagnostics --out models\\roadmap_next_step_v4_semantic_v4\\shadow_report --format both",
            ".\\.venv\\Scripts\\python.exe backend\\manage.py report_roadmap_continuation_shadow_diff --model-root models\\roadmap_continuation_planner_v2_after_runtime_patch --report-json models\\roadmap_continuation_planner_v2_after_runtime_patch\\shadow_report.json --report-md models\\roadmap_continuation_planner_v2_after_runtime_patch\\shadow_report.md",
            ".\\.venv\\Scripts\\python.exe backend\\manage.py report_roadmap_nextstep_v5_broader_qualification_rerun --source-preference fresh_db --candidate-model-path models\\roadmap_next_step_v5_historical_anchor_targeted_v1\\model.pkl --format both",
            ".\\.venv\\Scripts\\python.exe backend\\manage.py report_roadmap_nextstep_v5_broader_qualification_rerun --source-preference cached_artifact --candidate-model-path models\\roadmap_next_step_v5_historical_anchor_targeted_v1\\model.pkl --cached-comparison-json reports\\roadmap_nextstep_v5_historical_anchor_targeted_v1_comparison.json --format both",
            ".\\.venv\\Scripts\\python.exe backend\\manage.py report_roadmap_nextstep_v5_candidate_promotion_under_freeze --source-preference auto --candidate-model-path models\\roadmap_next_step_v5_historical_anchor_targeted_v1\\model.pkl --cached-comparison-json reports\\roadmap_nextstep_v5_historical_anchor_targeted_v1_comparison.json --format both",
            ".\\.venv\\Scripts\\python.exe backend\\manage.py report_roadmap_nextstep_v5_skincare_freeze_qualification --source-preference auto --candidate-model-path models\\roadmap_next_step_v5_historical_anchor_targeted_v1\\model.pkl --cached-comparison-json reports\\roadmap_nextstep_v5_historical_anchor_targeted_v1_comparison.json --format both",
            ".\\.venv\\Scripts\\python.exe backend\\manage.py report_roadmap_runtime_integrity --output-json reports\\roadmap_fragrance_slot_qualification.json --output-md reports\\roadmap_fragrance_slot_qualification.md",
            ".\\.venv\\Scripts\\python.exe backend\\manage.py report_roadmap_ml_artifact_qualification",
        ],
    }


def _proof_line(entry: dict[str, Any]) -> str:
    proof = _safe_dict(entry.get("proof_bundle"))
    missing = list(proof.get("missing_required") or [])
    invalid = list(proof.get("invalid_required") or [])
    stale = list(proof.get("stale_required") or [])
    status = "complete" if bool(proof.get("required_complete")) else "incomplete"
    detail = ", ".join(missing + invalid + stale) if (missing or invalid or stale) else "all required files present"
    return f"- `{entry.get('key')}` [{entry.get('role')}] `{status}`: {detail}"


def render_roadmap_ml_artifact_qualification_markdown(payload: dict[str, Any]) -> str:
    lines: list[str] = [
        "# Roadmap ML Artifact Qualification",
        "",
        f"Generated: {payload.get('generated_at_utc')}",
        "",
        "## Configured Artifacts",
    ]
    for entry in payload.get("configured_artifacts") or []:
        lines.append(
            f"- `{entry.get('key')}` lane=`{entry.get('lane')}` role=`{entry.get('role')}` "
            f"path=`{entry.get('model_path') or '__none__'}` categories=`{entry.get('categories')}`"
        )

    lines.extend(
        [
            "",
            "## Proof Bundle Completeness",
        ]
    )
    for entry in payload.get("configured_artifacts") or []:
        lines.append(_proof_line(_safe_dict(entry)))

    lines.extend(
        [
            "",
            "## Per-Category Pass/Fail Manifest",
        ]
    )
    manifests = _safe_dict(payload.get("per_category_manifest"))
    for lane, lane_payload in manifests.items():
        lines.append("")
        lines.append(f"### {lane}")
        lines.append("| category | status | reason | model_path |")
        lines.append("| --- | --- | --- | --- |")
        for category, row in sorted(_safe_dict(_safe_dict(lane_payload).get("by_category")).items()):
            row_dict = _safe_dict(row)
            lines.append(
                f"| {category} | {row_dict.get('status') or ''} | {row_dict.get('reason') or ''} | "
                f"{row_dict.get('model_path') or ''} |"
            )

    promotion = _safe_dict(payload.get("freeze_candidate_promotion"))
    promotion_state = _safe_dict(promotion.get("promotion_state"))
    promotion_exec = _safe_dict(promotion.get("executive_verdict"))
    promotion_provenance = _safe_dict(promotion.get("provenance"))
    active_runtime = _safe_dict(promotion_state.get("active_runtime_continuation_artifact"))
    promoted_candidate = _safe_dict(promotion_state.get("promoted_freeze_only_continuation_candidate"))
    runtime_serve = _safe_dict(promotion_state.get("runtime_serve"))
    lines.extend(
        [
            "",
            "## Freeze Candidate Promotion",
            f"- status: `{promotion.get('status')}`",
            f"- promoted candidate under freeze: `{promotion_exec.get('canonical_freeze_candidate')}`",
            f"- active runtime artifact: `{active_runtime.get('model_path')}`",
            f"- promoted freeze-only candidate: `{promoted_candidate.get('model_path')}`",
            f"- runtime still frozen: `{promotion_exec.get('runtime_still_frozen')}`",
            f"- active runtime artifact unchanged: `{promotion_exec.get('active_runtime_artifact_unchanged')}`",
            f"- runtime serve enabled: `{runtime_serve.get('serve_enabled')}`",
            f"- promotion recommendation: `{promotion_exec.get('recommendation_code')}` {promotion_exec.get('recommendation_label')}",
            f"- provenance: `{promotion_provenance.get('report_materialization')}` / `{promotion_provenance.get('source_of_truth')}` / `{promotion_provenance.get('generated_from')}`",
        ]
    )

    lines.extend(
        [
            "",
            "## Missing Proof Items",
        ]
    )
    missing_rows = payload.get("missing_proof_items") or []
    if not missing_rows:
        lines.append("- none")
    else:
        for row in missing_rows:
            row_dict = _safe_dict(row)
            lines.append(
                f"- `{row_dict.get('artifact_key')}` missing `{row_dict.get('file')}` "
                f"status=`{row_dict.get('status')}` reason=`{row_dict.get('reason')}`"
            )

    future = _safe_dict(payload.get("future_consideration"))
    lines.extend(
        [
            "",
            "## Future Consideration",
            f"- any_future_shadow_eligible: `{str(future.get('any_future_shadow_eligible')).lower()}`",
            f"- any_future_serve_eligible: `{str(future.get('any_future_serve_eligible')).lower()}`",
        ]
    )
    for row in future.get("shadow_candidates") or []:
        row_dict = _safe_dict(row)
        lines.append(
            f"- `{row_dict.get('key')}` shadow eligible=`{str(row_dict.get('eligible')).lower()}` "
            f"reason=`{row_dict.get('reason')}` eligible_plans=`{row_dict.get('eligible_plans')}`"
        )

    fragrance = _safe_dict(payload.get("fragrance_slot_qualification"))
    lines.extend(
        [
            "",
            "## Fragrance Slot Qualification",
            f"- source_of_truth: `{fragrance.get('source_of_truth')}`",
            f"- db_error: `{fragrance.get('db_error')}`",
            f"- runtime: `{_safe_dict(fragrance.get('runtime'))}`",
            f"- legacy: `{_safe_dict(fragrance.get('legacy'))}`",
        ]
    )

    continuation = _safe_dict(payload.get("continuation_runtime_candidate"))
    lines.extend(
        [
            "",
            "## Continuation Candidate",
            f"- model_root: `{continuation.get('model_root')}`",
            f"- shadow_report_json: `{continuation.get('shadow_report_json')}`",
            f"- shadow_report_md: `{continuation.get('shadow_report_md')}`",
        ]
    )

    lines.extend(
        [
            "",
            "## Reproduction Commands",
            "```powershell",
            *[str(command) for command in payload.get("canonical_commands") or []],
            "```",
        ]
    )
    return "\n".join(lines)
