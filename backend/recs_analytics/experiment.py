from __future__ import annotations

from typing import Any


EXPERIMENT_CONTEXT_KEYS = (
    "experiment_id",
    "experiment_variant",
    "algo_requested",
    "algo_used",
    "model_version",
    "algo_source",
    "ab_bucket",
    "guardrail_forced",
    "guardrail_reason",
)


def _compact(d: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in d.items():
        if v is None:
            continue
        if isinstance(v, str) and not v.strip():
            continue
        out[k] = v
    return out


def extract_experiment_context(raw: dict[str, Any] | None) -> dict[str, Any]:
    src = raw if isinstance(raw, dict) else {}
    return _compact({k: src.get(k) for k in EXPERIMENT_CONTEXT_KEYS})


def build_event_experiment_context(
    *,
    algo_requested: str | None,
    algo_used: str | None,
    model_version: str | None,
    routing: dict[str, Any] | None,
) -> dict[str, Any]:
    route = routing if isinstance(routing, dict) else {}
    out: dict[str, Any] = {
        "algo_requested": algo_requested,
        "algo_used": algo_used,
        "model_version": model_version,
        "algo_source": route.get("source"),
    }
    if route.get("experiment_id"):
        out["experiment_id"] = route.get("experiment_id")
    if route.get("ab_variant"):
        out["experiment_variant"] = route.get("ab_variant")
    if route.get("ab_bucket") is not None:
        out["ab_bucket"] = route.get("ab_bucket")
    if "guardrail_forced" in route:
        out["guardrail_forced"] = bool(route.get("guardrail_forced"))
    if route.get("guardrail_reason"):
        out["guardrail_reason"] = route.get("guardrail_reason")
    return _compact(out)
