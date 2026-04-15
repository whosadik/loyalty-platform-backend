from __future__ import annotations

from typing import Any

from django.db import connections
from django.db.utils import DatabaseError, OperationalError

from roadmap_app.historical_anchor_replay import build_historical_continuation_anchor_records
from roadmap_app.models import RoadmapPlan
from roadmap_app.nextstep_historical_anchor_dataset import completion_events_by_step


class HistoricalAnchorReadError(RuntimeError):
    def __init__(self, *, stage: str, operation: str, exc: Exception):
        self.stage = str(stage or "unknown_stage")
        self.operation = str(operation or "unknown_operation")
        self.original_exception = exc
        error_text = f"{type(exc).__module__}.{type(exc).__name__}: {exc}"
        self.error_text = error_text
        super().__init__(f"{self.stage}:{self.operation}: {error_text}")


def _safe_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _safe_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _int_keyed_dict(value: Any) -> dict[int, Any]:
    out: dict[int, Any] = {}
    for raw_key, raw_value in _safe_dict(value).items():
        try:
            key = int(raw_key)
        except Exception:
            continue
        out[key] = raw_value
    return out


def _wrap_db_stage(*, stage: str, operation: str, fn):
    try:
        return fn()
    except HistoricalAnchorReadError:
        raise
    except (OperationalError, DatabaseError) as exc:
        raise HistoricalAnchorReadError(stage=stage, operation=operation, exc=exc) from exc


def _probe_default_db_connection() -> None:
    connection = connections["default"]
    with connection.cursor() as cursor:
        cursor.execute("SELECT 1")
        cursor.fetchone()


def probe_historical_anchor_read_context(
    *,
    since,
    until,
    category: str = "all",
    include_ga: bool = False,
) -> dict[str, Any]:
    category_norm = str(category or "all").strip().lower() or "all"
    result: dict[str, Any] = {
        "status": "ready",
        "source_of_truth": "live_db_probe",
        "read_only": True,
        "category": category_norm,
        "include_ga": bool(include_ga),
        "db_connect_ok": False,
        "historical_anchor_query_ok": False,
        "plan_meta_query_ok": False,
        "completion_query_ok": False,
        "anchors_total": 0,
        "plan_ids_total": 0,
        "generated_step_ids_total": 0,
        "failure_stage": "",
        "failure_operation": "",
        "failure_error": "",
    }
    try:
        _wrap_db_stage(
            stage="db_connect",
            operation="select_1_preflight",
            fn=_probe_default_db_connection,
        )
        result["db_connect_ok"] = True
        anchors = _wrap_db_stage(
            stage="historical_anchor_query",
            operation="build_historical_continuation_anchor_records",
            fn=lambda: build_historical_continuation_anchor_records(
                since=since,
                until=until,
                category=category_norm,
                include_ga=include_ga,
            ),
        )
        result["historical_anchor_query_ok"] = True
        result["anchors_total"] = int(len(anchors))
        plan_ids = {
            int(anchor.get("plan_id") or 0)
            for anchor in anchors
            if int(anchor.get("plan_id") or 0) > 0
        }
        result["plan_ids_total"] = int(len(plan_ids))
        _wrap_db_stage(
            stage="plan_meta_query",
            operation="RoadmapPlan.values(id,meta)",
            fn=lambda: {
                int(row["id"]): _safe_dict(row.get("meta"))
                for row in RoadmapPlan.objects.filter(id__in=plan_ids).values("id", "meta")
            },
        )
        result["plan_meta_query_ok"] = True
        generated_step_ids = {
            int(step_id)
            for anchor in anchors
            for step_id in _safe_list(anchor.get("generated_step_ids"))
            if str(step_id or "").strip()
        }
        result["generated_step_ids_total"] = int(len(generated_step_ids))
        _wrap_db_stage(
            stage="completion_query",
            operation="completion_events_by_step",
            fn=lambda: completion_events_by_step(
                since=since,
                until=until,
                step_ids=generated_step_ids,
            ),
        )
        result["completion_query_ok"] = True
    except HistoricalAnchorReadError as exc:
        result["status"] = "blocked"
        result["failure_stage"] = str(exc.stage)
        result["failure_operation"] = str(exc.operation)
        result["failure_error"] = str(exc.error_text)
    return result


def build_historical_anchor_read_context(
    *,
    since,
    until,
    category: str = "all",
    include_ga: bool = False,
) -> dict[str, Any]:
    category_norm = str(category or "all").strip().lower() or "all"
    _wrap_db_stage(
        stage="db_connect",
        operation="select_1_preflight",
        fn=_probe_default_db_connection,
    )
    anchors = _wrap_db_stage(
        stage="historical_anchor_query",
        operation="build_historical_continuation_anchor_records",
        fn=lambda: build_historical_continuation_anchor_records(
            since=since,
            until=until,
            category=category_norm,
            include_ga=include_ga,
        ),
    )
    plan_ids = {
        int(anchor.get("plan_id") or 0)
        for anchor in anchors
        if int(anchor.get("plan_id") or 0) > 0
    }
    meta_by_plan = _wrap_db_stage(
        stage="plan_meta_query",
        operation="RoadmapPlan.values(id,meta)",
        fn=lambda: {
            int(row["id"]): _safe_dict(row.get("meta"))
            for row in RoadmapPlan.objects.filter(id__in=plan_ids).values("id", "meta")
        },
    )
    all_generated_step_ids = {
        int(step_id)
        for anchor in anchors
        for step_id in _safe_list(anchor.get("generated_step_ids"))
        if str(step_id or "").strip()
    }
    return {
        "since": since,
        "until": until,
        "category": category_norm,
        "include_ga": bool(include_ga),
        "anchors": anchors,
        "meta_by_plan": meta_by_plan,
        "completions_by_step": _wrap_db_stage(
            stage="completion_query",
            operation="completion_events_by_step",
            fn=lambda: completion_events_by_step(
                since=since,
                until=until,
                step_ids=all_generated_step_ids,
            ),
        ),
        "read_only": True,
    }


def resolve_historical_anchor_read_context(
    *,
    since,
    until,
    category: str = "all",
    include_ga: bool = False,
    historical_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if historical_context is None:
        return build_historical_anchor_read_context(
            since=since,
            until=until,
            category=category,
            include_ga=include_ga,
        )

    category_norm = str(category or "all").strip().lower() or "all"
    provided_category = str(historical_context.get("category") or category_norm).strip().lower() or "all"
    if provided_category != category_norm:
        raise ValueError(
            f"historical_context category mismatch: expected {category_norm}, got {provided_category}"
        )

    provided_include_ga = bool(historical_context.get("include_ga"))
    if provided_include_ga != bool(include_ga):
        raise ValueError(
            f"historical_context include_ga mismatch: expected {bool(include_ga)}, got {provided_include_ga}"
        )

    meta_by_plan = {
        plan_id: _safe_dict(meta)
        for plan_id, meta in _int_keyed_dict(historical_context.get("meta_by_plan")).items()
    }
    completions_by_step = {
        step_id: _safe_list(rows)
        for step_id, rows in _int_keyed_dict(historical_context.get("completions_by_step")).items()
    }
    return {
        "since": historical_context.get("since", since),
        "until": historical_context.get("until", until),
        "category": provided_category,
        "include_ga": provided_include_ga,
        "anchors": _safe_list(historical_context.get("anchors")),
        "meta_by_plan": meta_by_plan,
        "completions_by_step": completions_by_step,
        "read_only": bool(historical_context.get("read_only", True)),
    }
