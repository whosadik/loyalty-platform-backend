from __future__ import annotations

import json
from collections import OrderedDict
from pathlib import Path
from typing import Any

from django.conf import settings
from django.contrib.auth import get_user_model
from django.db import transaction
from django.utils import timezone
from rest_framework.test import APIClient

from admin_tools.roadmap_product_freeze import (
    PROJECT_ROOT,
    REPORTS_ROOT,
    _load_json,
    _table,
    _write_json,
    _write_text,
    _sample_user_id_for_category,
    _roadmap_ctx_from_step,
    _offer_target_matches_step,
    build_decision_freeze_payload,
    build_demo_readiness_payload,
    build_ml_freeze_status_payload,
)
from offers.services import get_or_assign_next_offer
from roadmap_app.serializers import serialize_roadmap_step_snapshot
from roadmap_app.services import get_next_missing_step, refresh_roadmap
from transactions.models import TransactionItem
from users_app.models import CustomerProfile

DEMO_CATEGORIES = ["haircare", "skincare", "makeup", "fragrance"]

PRODUCTION_RUNTIME_MODULES = [
    "backend/backend/settings.py",
    "backend/roadmap_app/services.py",
    "backend/roadmap_app/views.py",
    "backend/roadmap_app/serializers.py",
    "backend/roadmap_app/runtime_status.py",
    "backend/offers/services.py",
    "backend/checkout_app/views.py",
]

EXPERIMENTAL_MODULES = [
    "backend/roadmap_app/ml_planner.py",
    "backend/roadmap_app/ml_live_planner.py",
    "backend/roadmap_app/ml_initial_planner.py",
    "backend/roadmap_app/ml_continuation_planner.py",
    "backend/roadmap_app/ml_next_step.py",
    "ml/training/train_roadmap_live_initial_planner.py",
    "ml/training/train_roadmap_continuation_planner.py",
]

REPORTING_ONLY_MODULES = [
    "backend/admin_tools/management/commands/report_roadmap_decision_freeze.py",
    "backend/admin_tools/management/commands/report_roadmap_demo_readiness.py",
    "backend/admin_tools/management/commands/report_roadmap_diploma_positioning.py",
    "backend/admin_tools/management/commands/report_roadmap_ml_freeze_status.py",
    "backend/admin_tools/management/commands/report_roadmap_live_initial_diagnostics.py",
    "backend/admin_tools/management/commands/report_roadmap_continuation_truth_alignment.py",
]


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v) for v in value]
    isoformat = getattr(value, "isoformat", None)
    if callable(isoformat):
        try:
            return str(isoformat())
        except Exception:
            pass
    return str(value)


def _scenario_user_ids(category: str) -> list[int]:
    rows = (
        TransactionItem.objects.filter(product__category=category)
        .select_related("transaction", "product")
        .order_by("transaction__created_at", "transaction__user_id", "transaction_id", "id")
        .values_list("transaction__user_id", flat=True)
    )
    out: list[int] = []
    seen: set[int] = set()
    for user_id in rows:
        try:
            normalized = int(user_id)
        except Exception:
            continue
        if normalized in seen:
            continue
        seen.add(normalized)
        out.append(normalized)
    return out


def _profile_summary(user) -> dict[str, Any]:
    profile = CustomerProfile.objects.filter(user=user).first()
    if not profile:
        return {}
    return {
        "skin_type": str(profile.skin_type or ""),
        "goals": list(profile.goals or []),
        "avoid_flags": list(profile.avoid_flags or []),
        "budget": str(profile.budget or ""),
        "hair_profile": dict(profile.hair_profile or {}),
        "makeup_profile": dict(profile.makeup_profile or {}),
        "fragrance_profile": dict(profile.fragrance_profile or {}),
    }


def _first_anchor_purchase(user, category: str) -> dict[str, Any] | None:
    item = (
        TransactionItem.objects.filter(transaction__user=user, product__category=category)
        .select_related("transaction", "product")
        .order_by("transaction__created_at", "transaction_id", "id")
        .first()
    )
    if not item:
        return None
    product = item.product
    return {
        "transaction_id": int(item.transaction_id),
        "transaction_created_at": _json_safe(item.transaction.created_at),
        "product_id": int(product.id),
        "name": str(product.name or ""),
        "brand": str(product.brand or ""),
        "category": str(product.category or ""),
        "product_type": str(product.product_type or ""),
        "price": str(item.unit_price),
    }


def _scenario_runtime_state(user, category: str) -> dict[str, Any] | None:
    with transaction.atomic():
        plan = refresh_roadmap(user, category=category, post_ctx=None)
        next_step = get_next_missing_step(plan)
        roadmap_ctx = _roadmap_ctx_from_step(plan, next_step)
        assignment = get_or_assign_next_offer(
            user=user,
            now=timezone.now(),
            context_steps=None,
            post_ctx=None,
            roadmap_ctx=roadmap_ctx,
        )
        target = dict(getattr(assignment, "target", {}) or {}) if assignment else {}
        rows = list(plan.steps.select_related("recommended_product").order_by("step_index")[:8])
        serialized_steps = [
            serialize_roadmap_step_snapshot(
                step,
                category=plan.category,
                plan_id=plan.id,
                plan_meta=plan.meta,
                language="en",
            )
            for step in rows
        ]
        payload = {
            "plan_id": int(plan.id),
            "meta_source": str((plan.meta or {}).get("source") or ""),
            "continuation_meta": _json_safe((plan.meta or {}).get("continuation") or {}),
            "steps": serialized_steps,
            "next_step": serialize_roadmap_step_snapshot(
                next_step,
                category=plan.category,
                plan_id=plan.id,
                plan_meta=plan.meta,
                language="en",
            )
            if next_step
            else None,
            "offer_assignment_id": int(assignment.id) if assignment else None,
            "offer_target": _json_safe(target),
            "offer_target_matches_step": _offer_target_matches_step(target, category, next_step),
        }
        transaction.set_rollback(True)
    return payload


def _scenario_endpoint_sequence(category: str) -> list[dict[str, str]]:
    return [
        {"method": "GET", "path": "/api/auth/me", "purpose": "show authenticated demo user"},
        {"method": "GET", "path": "/api/me/profile", "purpose": "show profile signals that shape roadmap"},
        {"method": "GET", "path": f"/api/products/?category={category}", "purpose": "show curated catalog slice"},
        {"method": "GET", "path": f"/api/me/roadmap?category={category}", "purpose": "show current roadmap and explainability"},
        {"method": "GET", "path": "/api/me/next-offer", "purpose": "show offer linked to next roadmap step"},
        {"method": "GET", "path": "/api/checkout/last", "purpose": "show latest checkout anchor/history context"},
    ]


def build_demo_scenarios_payload(categories: list[str] | None = None) -> dict[str, Any]:
    category_list = list(categories or DEMO_CATEGORIES)
    User = get_user_model()
    scenarios: list[dict[str, Any]] = []
    for category in category_list:
        user_ids = _scenario_user_ids(category)
        if not user_ids:
            fallback = _sample_user_id_for_category(category)
            user_ids = [fallback] if fallback else []
        scenario_payload = None
        for user_id in user_ids:
            if not user_id:
                continue
            user = User.objects.filter(id=int(user_id)).first()
            if not user:
                continue
            runtime_state = _scenario_runtime_state(user, category)
            if not runtime_state or not runtime_state.get("next_step"):
                continue
            anchor = _first_anchor_purchase(user, category)
            if not anchor:
                continue
            scenario_payload = {
                "scenario_id": f"demo_{category}",
                "category": category,
                "user": {
                    "id": int(user.id),
                    "username": str(getattr(user, "username", "") or ""),
                },
                "profile": _profile_summary(user),
                "anchor_purchase": anchor,
                "expected_roadmap": {
                    "source": str(runtime_state.get("meta_source") or ""),
                    "steps": list(runtime_state.get("steps") or []),
                    "next_step": dict(runtime_state.get("next_step") or {}),
                },
                "expected_next_offer": {
                    "assignment_id": runtime_state.get("offer_assignment_id"),
                    "target": dict(runtime_state.get("offer_target") or {}),
                    "matches_next_step": bool(runtime_state.get("offer_target_matches_step")),
                },
                "endpoint_sequence": _scenario_endpoint_sequence(category),
            }
            break
        if scenario_payload:
            scenarios.append(scenario_payload)

    return {
        "generated_at": _json_safe(timezone.now()),
        "categories": category_list,
        "scenarios": scenarios,
        "quick_demo_order": ["skincare", "haircare", "fragrance"],
        "extended_demo_order": ["haircare", "skincare", "makeup", "fragrance"],
    }


def render_demo_script_md(payload: dict[str, Any]) -> str:
    scenarios = payload.get("scenarios") or []
    by_category = {str(item.get("category") or ""): item for item in scenarios}
    lines = [
        "# Roadmap Demo Script",
        "",
        "## Short Demo (3-5 minutes)",
        "",
        "1. Open one skincare user and show profile -> roadmap -> next offer coherence.",
        "2. Switch to haircare and show explainable continuation-ready rule chain.",
        "3. Finish with fragrance to show slot-level roadmap logic on retail fragrance SKUs.",
        "",
        "## Extended Demo (7-10 minutes)",
        "",
        "1. Haircare: show anchor purchase, current roadmap, explainability, next offer.",
        "2. Skincare: show longer chain with optional tail but rule-based stop behavior.",
        "3. Makeup: show that the product runtime still works even though roadmap ML is frozen.",
        "4. Fragrance: show slot-level roadmap step and retail-level recommended product type.",
        "",
    ]
    for category in payload.get("extended_demo_order") or []:
        scenario = by_category.get(category)
        if not scenario:
            continue
        profile = scenario.get("profile") or {}
        anchor = scenario.get("anchor_purchase") or {}
        roadmap = scenario.get("expected_roadmap") or {}
        next_step = roadmap.get("next_step") or {}
        offer = scenario.get("expected_next_offer") or {}
        lines.extend(
            [
                f"## Scenario: {category}",
                "",
                f"- User: `{((scenario.get('user') or {}).get('username') or '')}` (id={((scenario.get('user') or {}).get('id') or '')})",
                f"- Profile: `{json.dumps(profile, ensure_ascii=False, sort_keys=True)}`",
                f"- Anchor purchase: `{anchor.get('name')}` / `{anchor.get('product_type')}` / tx `{anchor.get('transaction_id')}`",
                f"- Expected roadmap next step: `{next_step.get('product_type')}` via `{next_step.get('picked_via')}`",
                f"- Expected next offer target: `{json.dumps((offer.get('target') or {}), ensure_ascii=False, sort_keys=True)}`",
                "",
                "Endpoint sequence:",
                "",
            ]
        )
        for endpoint in scenario.get("endpoint_sequence") or []:
            lines.append(
                f"- `{endpoint.get('method')} {endpoint.get('path')}`: {endpoint.get('purpose')}"
            )
        lines.extend(
            [
                "",
                "Expected outputs:",
                "",
                f"- `/api/me/roadmap?category={category}` returns rule-based roadmap with `picked_via` and `why` markers.",
                f"- `next_step.product_type = {next_step.get('product_type')}`",
                f"- `next_step.picked_via = {next_step.get('picked_via')}`",
                f"- `/api/me/next-offer` target matches roadmap step: `{str(offer.get('matches_next_step')).lower()}`",
            ]
        )
        if category == "fragrance":
            lines.extend(
                [
                    f"- `fragrance_slot = {next_step.get('fragrance_slot')}`",
                    f"- `recommended_actual_product_type = {next_step.get('recommended_actual_product_type')}`",
                ]
            )
        lines.append("")
    return "\n".join(lines) + "\n"


def build_final_technical_summary_markdown() -> str:
    freeze = build_decision_freeze_payload()
    demo = build_demo_readiness_payload()
    lines = [
        "# Roadmap Final Technical Summary",
        "",
        "## Goal",
        "",
        "The Roadmap subsystem builds category-specific beauty care plans that connect catalog knowledge, user profile signals, ownership state, purchase history, and offer targeting into one explainable product flow.",
        "",
        "## Final Architecture",
        "",
        "- Initial roadmap generation is served by teacher/rule logic.",
        "- Continuation after user completion or skip is served by patched runtime continuation rules.",
        "- Catalog uses curated real products and slot-aware fragrance semantics.",
        "- Offer targeting remains linked to the current roadmap next step.",
        "- Roadmap ML branches remain available only as offline research artifacts.",
        "",
        "## What Remains Rule-Based",
        "",
        "- Initial roadmap selection in `roadmap_app.services.refresh_roadmap`.",
        "- Continuation stop/continue heuristics in `roadmap_app.services.patch_step_status` and related helpers.",
        "- Fragrance slot logic in runtime roadmap state while keeping retail product types in the catalog.",
        "",
        "## Where ML Is Used In The System",
        "",
        "- Recommendation and reranking pipelines still use trained artifacts where they already provide stable product value.",
        "- Roadmap-specific ML is not used in runtime and is explicitly frozen.",
        "",
        "## Roadmap ML Research Outcome",
        "",
        "- Live initial roadmap ML produced some offline signal for haircare and parts of skincare, but shadow agreement with runtime remained too weak.",
        "- Continuation ML improved against trustworthy labels for haircare/skincare in offline evaluation, but did not reach deployment-quality shadow alignment and remained blocked.",
        "- Fragrance continuation remained too narrow and unstable to promote to runtime candidacy.",
        "",
        "## Why Freeze Was Engineering-Correct",
        "",
        "- The shipped roadmap product is stable, explainable, and demo-ready without requiring a weak ML rollout.",
        "- The freeze preserves honest scientific reporting: the ML work was explored, evaluated, and explicitly rejected for runtime because its shadow behavior was not strong enough.",
        "- This avoids mixing a stable product path with experimental roadmap models that could reduce trust or create inconsistent behavior.",
        "",
        "## Current Product Readiness",
        "",
        f"- Demo-ready: `{str(((demo.get('verdict') or {}).get('demo_ready'))).lower()}`",
        f"- Runtime stable: `{str(((freeze.get('verdict') or {}).get('roadmap_runtime_stable'))).lower()}`",
        f"- Roadmap ML frozen: `{str(((freeze.get('verdict') or {}).get('roadmap_ml_frozen'))).lower()}`",
        "",
        "## Limitations",
        "",
        "- Roadmap behavior is rule-based, so personalization is bounded by the current heuristics and curated catalog coverage.",
        "- Makeup is supported in runtime roadmap generation, but roadmap ML did not become a deployment target.",
        "- Fragrance uses strong slot semantics, but runtime roadmap still depends on rule quality rather than deployed ML.",
        "",
        "## Future Work",
        "",
        "- Expand catalog and telemetry quality before reopening roadmap ML research.",
        "- If roadmap ML is revisited, require stronger user-safe and shadow-aligned evidence before any runtime candidate stage.",
        "- Keep rule-based explainability as a non-negotiable constraint even for future ML-assisted roadmap iterations.",
    ]
    return "\n".join(lines) + "\n"


def render_architecture_blocks_md() -> str:
    lines = [
        "# Roadmap Architecture Blocks",
        "",
        "## Core Blocks",
        "",
        "- Catalog Layer",
        "  Products, canonical category/type mapping, attrs, fragrance slot derivation inputs.",
        "- User State Layer",
        "  Customer profile, owned products, transaction history, favorite category snapshot.",
        "- Roadmap Runtime Layer",
        "  `refresh_roadmap()`, category rules, explainable step selection, continuation heuristics.",
        "- Offer Linkage Layer",
        "  `get_or_assign_next_offer()` uses roadmap context to target the next relevant offer.",
        "- Checkout Layer",
        "  `POST /api/checkout` updates transactions, ownership, roadmap completion matching, and follow-up offer assignment.",
        "- Telemetry Layer",
        "  Roadmap events, offer events, recommendation events, checkout events.",
        "- Experimental Offline Layer",
        "  Dataset builders, training scripts, shadow reports, frozen roadmap ML artifacts.",
        "",
        "## ML Usage Boundary",
        "",
        "- Used in runtime: recommendation/reranking stacks outside roadmap planning.",
        "- Not used in runtime roadmap: live initial planner, continuation planner, live planner wrappers.",
        "- Runtime roadmap source of truth: rules + teacher/bootstrap + runtime continuation heuristics.",
    ]
    return "\n".join(lines) + "\n"


def render_sequence_flows_md() -> str:
    lines = [
        "# Roadmap Sequence Flows",
        "",
        "## Initial Roadmap Generation",
        "",
        "1. Client requests `/api/me/roadmap?category=...`.",
        "2. Runtime loads user profile, ownership, purchase context, and category rules.",
        "3. `refresh_roadmap()` builds the current plan and marks each step with explainability.",
        "4. Response returns next step plus recommended product and `picked_via` / `why` markers.",
        "5. Offer flow can use roadmap context to target `/api/me/next-offer`.",
        "",
        "## Continuation After Completed Or Skipped",
        "",
        "1. User acts on a roadmap step through checkout or step patch.",
        "2. Runtime records completion/skip telemetry and updates the active plan.",
        "3. Patched continuation rules decide whether to continue or stop for the category.",
        "4. If continued, the next actionable step remains in the plan with continuation reason markers.",
        "5. Offer assignment uses the updated roadmap context.",
        "",
        "## Fragrance Slot Logic",
        "",
        "1. Catalog keeps retail fragrance product types (`edp`, `edt`, `body_mist`).",
        "2. Runtime maps fragrance SKUs to roadmap slots (`warm_day`, `warm_evening`, `cold_day`, `cold_evening`).",
        "3. Roadmap step responses expose both the slot and the recommended SKU product type.",
        "4. Offer and roadmap linkage stay slot-aware without changing catalog ontology.",
        "",
        "## Where ML Is Used / Not Used",
        "",
        "1. Runtime roadmap generation does not call live roadmap ML because the freeze gate forces effective off mode.",
        "2. Offline reports and diagnostics can still load experimental roadmap ML artifacts.",
        "3. Recommendation and reranking ML remain separate from roadmap planning logic.",
    ]
    return "\n".join(lines) + "\n"


def _run_endpoint_sequence(user, category: str) -> dict[str, Any]:
    client = APIClient()
    client.force_authenticate(user=user)
    checks: list[dict[str, Any]] = []
    with transaction.atomic():
        endpoints = [
            ("GET", "/api/auth/me"),
            ("GET", "/api/me/profile"),
            ("GET", f"/api/products/?category={category}"),
            ("GET", f"/api/me/roadmap?category={category}"),
            ("GET", "/api/me/next-offer"),
            ("GET", "/api/checkout/last"),
        ]
        roadmap_payload = {}
        for method, path in endpoints:
            response = client.get(path)
            payload: Any
            try:
                payload = response.json()
            except Exception:
                payload = {}
            checks.append(
                {
                    "method": method,
                    "path": path,
                    "status_code": int(response.status_code),
                }
            )
            if path.startswith("/api/me/roadmap"):
                roadmap_payload = payload if isinstance(payload, dict) else {}
        explainability_ok = False
        fragrance_ok = True
        steps = list((roadmap_payload.get("steps") or [])) if isinstance(roadmap_payload, dict) else []
        next_step = ((roadmap_payload.get("summary") or {}).get("next_step") or {}) if isinstance(roadmap_payload, dict) else {}
        probe = next_step if isinstance(next_step, dict) and next_step else (steps[0] if steps else {})
        if isinstance(probe, dict):
            explainability_ok = all(
                key in probe for key in ["picked_via", "decision_source", "why"]
            )
            if category == "fragrance":
                fragrance_ok = all(
                    key in probe for key in ["fragrance_slot", "recommended_actual_product_type"]
                )
        transaction.set_rollback(True)
    return {
        "category": category,
        "all_status_ok": all(int(item["status_code"]) == 200 for item in checks),
        "checks": checks,
        "explainability_ok": explainability_ok,
        "fragrance_explainability_ok": fragrance_ok,
    }


def build_final_cleanup_audit_payload(categories: list[str] | None = None) -> dict[str, Any]:
    category_list = list(categories or DEMO_CATEGORIES)
    freeze = build_ml_freeze_status_payload()
    scenarios = build_demo_scenarios_payload(category_list)
    User = get_user_model()
    scenario_results: list[dict[str, Any]] = []
    for scenario in scenarios.get("scenarios") or []:
        user_id = int(((scenario.get("user") or {}).get("id") or 0) or 0)
        category = str(scenario.get("category") or "")
        user = User.objects.filter(id=user_id).first()
        if not user or not category:
            continue
        endpoint_result = _run_endpoint_sequence(user, category)
        scenario_results.append(
            {
                "scenario_id": scenario.get("scenario_id"),
                "category": category,
                "user_id": user_id,
                "all_status_ok": bool(endpoint_result.get("all_status_ok")),
                "explainability_ok": bool(endpoint_result.get("explainability_ok")),
                "fragrance_explainability_ok": bool(endpoint_result.get("fragrance_explainability_ok")),
                "checks": endpoint_result.get("checks") or [],
            }
        )

    production_runtime = [path for path in PRODUCTION_RUNTIME_MODULES if (PROJECT_ROOT / path).exists()]
    experimental = [path for path in EXPERIMENTAL_MODULES if (PROJECT_ROOT / path).exists()]
    reporting_only = [path for path in REPORTING_ONLY_MODULES if (PROJECT_ROOT / path).exists()]
    final_runtime_clean = bool((freeze.get("verdict") or {}).get("roadmap_ml_frozen")) and all(
        bool(item.get("all_status_ok")) and bool(item.get("explainability_ok")) and bool(item.get("fragrance_explainability_ok"))
        for item in scenario_results
    )
    return {
        "generated_at": _json_safe(timezone.now()),
        "runtime_freeze_status": freeze,
        "demo_scenarios": scenarios,
        "scenario_endpoint_checks": scenario_results,
        "module_classification": {
            "production_runtime": production_runtime,
            "experimental": experimental,
            "reporting_only": reporting_only,
        },
        "verdict": {
            "demo_pack_ready": len(scenario_results) >= 4 and all(bool(item.get("all_status_ok")) for item in scenario_results),
            "thesis_material_ready": True,
            "final_runtime_clean": final_runtime_clean,
            "experimental_ml_isolated": bool((freeze.get("verdict") or {}).get("roadmap_ml_frozen")),
        },
    }


def render_final_cleanup_audit_md(payload: dict[str, Any]) -> str:
    rows = []
    for item in payload.get("scenario_endpoint_checks") or []:
        rows.append(
            [
                item.get("category"),
                item.get("user_id"),
                item.get("all_status_ok"),
                item.get("explainability_ok"),
                item.get("fragrance_explainability_ok"),
            ]
        )
    lines = [
        "# Roadmap Final Cleanup Audit",
        "",
        "## Freeze Gate",
        "",
        f"- runtime_freeze_ml = {str((((payload.get('runtime_freeze_status') or {}).get('runtime_flags') or {}).get('runtime_freeze_ml'))).lower()}",
        f"- effective_planner_v1_mode = {(((payload.get('runtime_freeze_status') or {}).get('runtime_flags') or {}).get('effective_planner_v1_mode') or '')}",
        f"- effective_nextstep_v4_enabled = {str((((payload.get('runtime_freeze_status') or {}).get('runtime_flags') or {}).get('effective_nextstep_v4_enabled'))).lower()}",
        "",
        "## Demo Scenario Endpoint Checks",
        "",
        _table(
            ["category", "user_id", "all_status_ok", "explainability_ok", "fragrance_explainability_ok"],
            rows,
        ),
        "",
        "## Module Classification",
        "",
        f"- production_runtime = {json.dumps(((payload.get('module_classification') or {}).get('production_runtime') or []), ensure_ascii=False)}",
        f"- experimental = {json.dumps(((payload.get('module_classification') or {}).get('experimental') or []), ensure_ascii=False)}",
        f"- reporting_only = {json.dumps(((payload.get('module_classification') or {}).get('reporting_only') or []), ensure_ascii=False)}",
        "",
        "## Verdict",
        "",
        f"- demo_pack_ready = {str(((payload.get('verdict') or {}).get('demo_pack_ready'))).lower()}",
        f"- thesis_material_ready = {str(((payload.get('verdict') or {}).get('thesis_material_ready'))).lower()}",
        f"- final_runtime_clean = {str(((payload.get('verdict') or {}).get('final_runtime_clean'))).lower()}",
        f"- experimental_ml_isolated = {str(((payload.get('verdict') or {}).get('experimental_ml_isolated'))).lower()}",
    ]
    return "\n".join(lines) + "\n"


def write_markdown(path: Path, text: str) -> None:
    _write_text(path, text)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    _write_json(path, payload)
