from __future__ import annotations

from datetime import datetime, timedelta, timezone as dt_timezone
from pathlib import Path
from typing import Any

from django.conf import settings
from django.utils import timezone

from roadmap_app.nextstep_historical_anchor_context import probe_historical_anchor_read_context


DEFAULT_V5_DB_RERUN_READINESS_REPORT_STEM = (
    Path("reports") / "roadmap_nextstep_v5_db_rerun_readiness"
)


def _safe_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _runtime_state_snapshot() -> dict[str, Any]:
    return {
        "runtime_freeze_ml": bool(getattr(settings, "ROADMAP_RUNTIME_FREEZE_ML", True)),
        "active_runtime_model_path": str(getattr(settings, "ROADMAP_NEXTSTEP_V4_MODEL_PATH", "") or ""),
        "runtime_serve_enabled": False,
    }


def _canonical_commands(*, candidate_model_path: str) -> list[str]:
    return [
        ".\\.venv\\Scripts\\python.exe backend\\manage.py report_roadmap_nextstep_v5_db_rerun_readiness --format both",
        ".\\.venv\\Scripts\\python.exe backend\\manage.py report_roadmap_nextstep_v5_broader_qualification_rerun "
        f"--source-preference fresh_db --candidate-model-path {candidate_model_path} --format both",
        ".\\.venv\\Scripts\\python.exe backend\\manage.py report_roadmap_nextstep_v5_broader_qualification_rerun "
        f"--source-preference cached_artifact --candidate-model-path {candidate_model_path} "
        "--cached-comparison-json reports\\roadmap_nextstep_v5_historical_anchor_targeted_v1_comparison.json --format both",
    ]


def build_v5_db_rerun_readiness_payload(
    *,
    days: int = 30,
    category: str = "all",
    include_ga: bool = False,
    candidate_model_path: str | Path = "models/roadmap_next_step_v5_historical_anchor_targeted_v1/model.pkl",
) -> dict[str, Any]:
    now_utc = timezone.now()
    since = now_utc - timedelta(days=int(days))
    probe = probe_historical_anchor_read_context(
        since=since,
        until=now_utc,
        category=category,
        include_ga=include_ga,
    )
    runtime_state = _runtime_state_snapshot()
    blocked = str(probe.get("status") or "") != "ready"
    return {
        "generated_at_utc": datetime.now(dt_timezone.utc).isoformat(),
        "probe_scope": {
            "days": int(days),
            "category": str(category or "all"),
            "include_ga": bool(include_ga),
            "candidate_model_path": str(Path(str(candidate_model_path)).expanduser().resolve()),
            "read_only": True,
        },
        "runtime_state": runtime_state,
        "probe": probe,
        "executive_verdict": {
            "status": "blocked" if blocked else "ready_for_fresh_db_rerun",
            "source_of_truth": "live_db_probe",
            "read_only": True,
            "runtime_still_frozen": bool(runtime_state.get("runtime_freeze_ml")),
            "active_runtime_artifact_unchanged": True,
            "catalog_writes_performed": False,
            "runtime_config_changed": False,
            "failure_stage": str(probe.get("failure_stage") or ""),
            "failure_operation": str(probe.get("failure_operation") or ""),
            "failure_error": str(probe.get("failure_error") or ""),
            "exact_next_step": (
                "Restore DB connectivity for the fresh rerun path, then re-run the broader qualification from live DB."
                if blocked
                else "Run the fresh DB-backed broader qualification rerun for v5."
            ),
        },
        "canonical_commands": _canonical_commands(
            candidate_model_path=str(Path(str(candidate_model_path)).expanduser().resolve())
        ),
    }


def render_v5_db_rerun_readiness_markdown(payload: dict[str, Any]) -> str:
    verdict = _safe_dict(payload.get("executive_verdict"))
    runtime_state = _safe_dict(payload.get("runtime_state"))
    probe_scope = _safe_dict(payload.get("probe_scope"))
    probe = _safe_dict(payload.get("probe"))
    lines = [
        "# Roadmap Nextstep v5 DB Rerun Readiness",
        "",
        "## Executive Verdict",
        f"- status: `{verdict.get('status')}`",
        f"- source_of_truth: `{verdict.get('source_of_truth')}`",
        f"- read_only: `{verdict.get('read_only')}`",
        f"- runtime still frozen: `{verdict.get('runtime_still_frozen')}`",
        f"- active runtime artifact unchanged: `{verdict.get('active_runtime_artifact_unchanged')}`",
        f"- catalog writes performed: `{verdict.get('catalog_writes_performed')}`",
        f"- runtime config changed: `{verdict.get('runtime_config_changed')}`",
        f"- failure_stage: `{verdict.get('failure_stage')}`",
        f"- failure_operation: `{verdict.get('failure_operation')}`",
        f"- failure_error: `{verdict.get('failure_error')}`",
        f"- exact next step: {verdict.get('exact_next_step')}",
        "",
        "## Probe Scope",
        f"- days: `{probe_scope.get('days')}`",
        f"- category: `{probe_scope.get('category')}`",
        f"- include_ga: `{probe_scope.get('include_ga')}`",
        f"- candidate_model_path: `{probe_scope.get('candidate_model_path')}`",
        "",
        "## Runtime State",
        f"- runtime_freeze_ml: `{runtime_state.get('runtime_freeze_ml')}`",
        f"- active_runtime_model_path: `{runtime_state.get('active_runtime_model_path')}`",
        f"- runtime_serve_enabled: `{runtime_state.get('runtime_serve_enabled')}`",
        "",
        "## DB Probe",
        f"- db_connect_ok: `{probe.get('db_connect_ok')}`",
        f"- historical_anchor_query_ok: `{probe.get('historical_anchor_query_ok')}`",
        f"- plan_meta_query_ok: `{probe.get('plan_meta_query_ok')}`",
        f"- completion_query_ok: `{probe.get('completion_query_ok')}`",
        f"- anchors_total: `{probe.get('anchors_total')}`",
        f"- plan_ids_total: `{probe.get('plan_ids_total')}`",
        f"- generated_step_ids_total: `{probe.get('generated_step_ids_total')}`",
        "",
        "## Canonical Commands",
        "```powershell",
        *[str(command) for command in payload.get("canonical_commands") or []],
        "```",
    ]
    return "\n".join(lines).strip() + "\n"
