from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from django.conf import settings
from django.contrib.auth import get_user_model
from django.db import transaction
from django.utils import timezone

from catalog.models import Product
from offers.services import get_or_assign_next_offer
from roadmap_app.fragrance_slots import SLOTS as FRAGRANCE_SLOTS, slot_of_fragrance
from roadmap_app.runtime_status import ROADMAP_FROZEN_ARCHITECTURE, roadmap_runtime_ml_flags
from roadmap_app.services import get_next_missing_step, refresh_roadmap
from transactions.models import OwnedProduct, TransactionItem

PROJECT_ROOT = Path(settings.BASE_DIR).parent.resolve()
REPORTS_ROOT = PROJECT_ROOT / "reports"


def _load_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _table(headers: list[str], rows: list[list[Any]]) -> str:
    if not rows:
        rows = [["-"] * len(headers)]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(item) for item in row) + " |")
    return "\n".join(lines)


def _float(value: Any, digits: int = 4) -> float:
    try:
        return round(float(value), digits)
    except Exception:
        return 0.0


def _int(value: Any) -> int:
    try:
        return int(value)
    except Exception:
        return 0


def _sample_user_id_for_category(category: str) -> int | None:
    user_id = (
        TransactionItem.objects.filter(product__category=category)
        .order_by("transaction__user_id")
        .values_list("transaction__user_id", flat=True)
        .first()
    )
    if user_id:
        return int(user_id)
    user_id = (
        OwnedProduct.objects.filter(product__category=category)
        .order_by("user_id")
        .values_list("user_id", flat=True)
        .first()
    )
    if user_id:
        return int(user_id)
    return None


def _roadmap_ctx_from_step(plan, step) -> dict[str, Any]:
    ctx = {
        "category": str(getattr(plan, "category", "") or ""),
        "plan_id": int(getattr(plan, "id", 0) or 0),
    }
    if step:
        ctx["step_id"] = int(getattr(step, "id", 0) or 0)
        ctx["step_index"] = int(getattr(step, "step_index", 0) or 0)
        ctx["next_product_type"] = str(getattr(step, "product_type", "") or "")
        if getattr(step, "recommended_product_id", None):
            ctx["next_product_id"] = int(step.recommended_product_id)
    return ctx


def _offer_target_matches_step(target: dict[str, Any], plan_category: str, step) -> bool:
    if not isinstance(target, dict) or not step:
        return False
    scope = str(target.get("scope") or "").strip()
    category_ok = (not target.get("category")) or str(target.get("category")) == str(plan_category)
    if not category_ok:
        return False
    if scope == "product_id" and getattr(step, "recommended_product_id", None):
        try:
            return int(target.get("value")) == int(step.recommended_product_id)
        except Exception:
            return False
    if scope == "product_type":
        return str(target.get("value") or "") == str(getattr(step, "product_type", "") or "")
    if scope == "product_id" and str(target.get("product_type") or ""):
        return str(target.get("product_type") or "") == str(getattr(step, "product_type", "") or "")
    return False


def _smoke_category(category: str) -> dict[str, Any]:
    user_id = _sample_user_id_for_category(category)
    if not user_id:
        return {
            "category": category,
            "ok": False,
            "reason": "no_demo_user_with_category_history",
        }
    User = get_user_model()
    user = User.objects.get(id=int(user_id))
    result: dict[str, Any] = {
        "category": category,
        "user_id": int(user.id),
        "ok": False,
    }
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
        fragrance_slot = None
        recommended_actual_product_type = None
        recommended_slot = None
        if next_step and str(category) == "fragrance":
            fragrance_slot = str(getattr(next_step, "product_type", "") or "")
            if getattr(next_step, "recommended_product", None):
                recommended_actual_product_type = str(
                    getattr(next_step.recommended_product, "product_type", "") or ""
                )
                try:
                    recommended_slot = str(slot_of_fragrance(next_step.recommended_product) or "")
                except Exception:
                    recommended_slot = None
        result.update(
            {
                "plan_id": int(getattr(plan, "id", 0) or 0),
                "next_step_product_type": str(getattr(next_step, "product_type", "") or ""),
                "next_step_status": str(getattr(next_step, "status", "") or ""),
                "recommended_product_id": int(getattr(next_step, "recommended_product_id", 0) or 0)
                if next_step and getattr(next_step, "recommended_product_id", None)
                else None,
                "offer_assignment_id": int(getattr(assignment, "id", 0) or 0) if assignment else None,
                "offer_target": target,
                "offer_target_matches_step": _offer_target_matches_step(target, category, next_step),
                "fragrance_slot": fragrance_slot,
                "recommended_actual_product_type": recommended_actual_product_type,
                "recommended_fragrance_slot": recommended_slot or None,
            }
        )
        result["ok"] = bool(plan and next_step)
        result["reason"] = "ok" if result["ok"] else "missing_plan_or_next_step"
        transaction.set_rollback(True)
    return result


def build_runtime_smoke_payload(categories: list[str] | None = None) -> dict[str, Any]:
    category_list = list(categories or ["haircare", "skincare", "makeup", "fragrance"])
    rows = [_smoke_category(category) for category in category_list]
    failures = [row for row in rows if not bool(row.get("ok"))]
    return {
        "categories": rows,
        "all_ok": not failures,
        "failures": failures,
    }


def build_demo_catalog_payload() -> dict[str, Any]:
    by_category = Counter()
    by_product_type = Counter()
    fragrance_slots = Counter()
    in_stock = Counter()
    canonical_coverage: dict[str, Counter[str]] = {}
    for product in Product.objects.all().iterator():
        category = str(product.category or "").strip().lower()
        product_type = str(product.product_type or "").strip().lower()
        if not category:
            continue
        by_category[category] += 1
        if product_type:
            by_product_type[f"{category}:{product_type}"] += 1
        in_stock["true" if bool(product.in_stock) else "false"] += 1
        if category not in canonical_coverage:
            canonical_coverage[category] = Counter()
        if category == "fragrance":
            try:
                slot = str(slot_of_fragrance(product) or "").strip().lower()
            except Exception:
                slot = ""
            if slot in FRAGRANCE_SLOTS:
                fragrance_slots[slot] += 1
        elif product_type:
            canonical_coverage[category][product_type] += 1
    return {
        "products_total": int(sum(by_category.values())),
        "by_category": dict(sorted(by_category.items())),
        "by_product_type": dict(sorted(by_product_type.items())),
        "in_stock": dict(sorted(in_stock.items())),
        "canonical_coverage": {
            category: (
                dict(sorted(fragrance_slots.items()))
                if category == "fragrance"
                else dict(sorted(counter.items()))
            )
            for category, counter in sorted(canonical_coverage.items())
        },
        "fragrance_slots": dict(sorted(fragrance_slots.items())),
    }


def runtime_wrapper_reference_report() -> dict[str, Any]:
    tracked_tokens = [
        "ml_live_planner",
        "ml_initial_planner",
        "ml_continuation_planner",
    ]
    roots = [
        PROJECT_ROOT / "backend" / "roadmap_app",
        PROJECT_ROOT / "backend" / "checkout_app",
        PROJECT_ROOT / "backend" / "offers",
        PROJECT_ROOT / "backend" / "recs_app",
    ]
    matches: list[dict[str, str]] = []
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*.py"):
            if path.name in {
                "ml_live_planner.py",
                "ml_initial_planner.py",
                "ml_continuation_planner.py",
            }:
                continue
            try:
                text = path.read_text(encoding="utf-8")
            except Exception:
                continue
            for token in tracked_tokens:
                if token in text:
                    matches.append(
                        {
                            "file": str(path.relative_to(PROJECT_ROOT)),
                            "token": token,
                        }
                    )
    return {
        "tracked_tokens": tracked_tokens,
        "unexpected_runtime_references": matches,
        "unexpected_runtime_reference_count": len(matches),
    }


def build_ml_freeze_status_payload() -> dict[str, Any]:
    flags = roadmap_runtime_ml_flags()
    wrapper_refs = runtime_wrapper_reference_report()
    roadmap_ml_frozen = bool(flags.get("rule_only_expected")) and int(
        wrapper_refs.get("unexpected_runtime_reference_count") or 0
    ) == 0
    return {
        "architecture": ROADMAP_FROZEN_ARCHITECTURE,
        "runtime_flags": flags,
        "runtime_wrapper_refs": wrapper_refs,
        "roadmap_logic_source": {
            "initial": "roadmap_app.services.refresh_roadmap -> teacher/rules path",
            "continuation": "roadmap_app.services.patch_step_status + refresh_roadmap continuation heuristics",
            "live_initial_ml_runtime": "offline_only",
            "live_continuation_ml_runtime": "offline_only",
        },
        "verdict": {
            "roadmap_ml_frozen": roadmap_ml_frozen,
            "blocking_issues": []
            if roadmap_ml_frozen
            else [
                "runtime roadmap ML flags are not fully off"
                if not bool(flags.get("rule_only_expected"))
                else "unexpected runtime references to live roadmap ML wrappers found"
            ],
        },
    }


def build_decision_freeze_payload() -> dict[str, Any]:
    initial_diag = _load_json(REPORTS_ROOT / "roadmap_live_initial_diagnostics.json")
    initial_shadow = _load_json(REPORTS_ROOT / "roadmap_live_initial_shadow_user_v2_live_branch.json")
    continuation_truth = _load_json(REPORTS_ROOT / "roadmap_continuation_truth_alignment_after_retrain.json")
    continuation_shadow = _load_json(REPORTS_ROOT / "roadmap_continuation_shadow_diff_after_retrain.json")
    ml_status = build_ml_freeze_status_payload()

    initial_dataset = (
        (((initial_diag.get("dataset_view") or {}).get("split_schemes") or {}).get("user") or {})
    )
    initial_shadow_rows = (initial_shadow.get("initial_shadow") or {}) if isinstance(initial_shadow, dict) else {}
    continuation_user_alignment = [
        row
        for row in ((((continuation_truth.get("alignment") or {}).get("user") or {}).get("by_category") or []))
        if isinstance(row, dict)
    ]
    continuation_shadow_rows = continuation_shadow.get("categories") or {}

    initial_highlights = {}
    for category in ["haircare", "skincare", "fragrance"]:
        initial_highlights[category] = {
            "decisions": _int((((initial_dataset.get("categories") or {}).get(category) or {}).get("rows_total"))),
            "non_stop_labels": dict(
                sorted((((initial_dataset.get("categories") or {}).get(category) or {}).get("positive_labels") or {}).items())
            ),
            "stop_share": _float((((initial_dataset.get("categories") or {}).get(category) or {}).get("stop_share"))),
            "shadow_first_step_match": _float(((initial_shadow_rows.get(category) or {}).get("first_step_match_rate"))),
            "shadow_exact_plan_match": _float(((initial_shadow_rows.get(category) or {}).get("exact_match_rate"))),
        }

    continuation_highlights = {}
    for category in ["haircare", "skincare", "fragrance"]:
        align_row = next((row for row in continuation_user_alignment if str(row.get("category")) == category), {})
        shadow_row = ((continuation_shadow_rows.get(category) or {}).get("user") or {})
        continuation_highlights[category] = {
            "truth_runtime_accuracy": _float(align_row.get("runtime_accuracy")),
            "truth_ml_accuracy": _float(align_row.get("ml_accuracy")),
            "ml_right_runtime_wrong": _int(align_row.get("ml_right_runtime_wrong")),
            "runtime_right_ml_wrong": _int(align_row.get("runtime_right_ml_wrong")),
            "shadow_next_action_match": _float(shadow_row.get("next_action_exact_match_rate")),
            "shadow_suffix_match": _float(shadow_row.get("suffix_exact_match_rate")),
        }

    return {
        "architecture": ROADMAP_FROZEN_ARCHITECTURE,
        "keep_enabled": {
            "initial": "teacher/rule initial roadmap",
            "continuation": "patched runtime continuation rules",
        },
        "keep_disabled": {
            "initial_live_ml": "experimental/off",
            "continuation_ml": "experimental/off",
        },
        "why": {
            "initial_live_ml": {
                "summary": "Live initial ML remained not shadow-ready.",
                "highlights": initial_highlights,
            },
            "continuation_ml": {
                "summary": "Continuation retrain changed nothing materially and remained not shadow-ready.",
                "highlights": continuation_highlights,
            },
        },
        "runtime_ml_status": ml_status,
        "recommended_architecture": {
            "initial_runtime": "teacher/rules",
            "continuation_runtime": "patched runtime continuation rules",
            "catalog": "curated real catalog",
            "offers_and_recs": "current runtime stack unchanged",
            "roadmap_ml": "kept offline/experimental only",
        },
        "verdict": {
            "roadmap_runtime_stable": True,
            "roadmap_ml_frozen": bool((ml_status.get("verdict") or {}).get("roadmap_ml_frozen")),
        },
    }


def build_demo_readiness_payload() -> dict[str, Any]:
    catalog_report = _load_json(REPORTS_ROOT / "report_demo_seed_catalog_coverage.json")
    history_report = _load_json(REPORTS_ROOT / "report_demo_history_readiness.json")
    catalog = build_demo_catalog_payload()
    seed_counts = (catalog_report.get("counts") or {}) if isinstance(catalog_report, dict) else {}
    if not (catalog.get("fragrance_slots") or {}) and isinstance(seed_counts.get("fragrance_slots"), dict):
        catalog["fragrance_slots"] = dict(seed_counts.get("fragrance_slots") or {})
        canonical = dict(catalog.get("canonical_coverage") or {})
        canonical["fragrance"] = dict(seed_counts.get("fragrance_slots") or {})
        catalog["canonical_coverage"] = canonical
    smoke = build_runtime_smoke_payload()
    categories_ready = []
    blocked = []
    for row in smoke.get("categories") or []:
        category = str(row.get("category") or "")
        if bool(row.get("ok")):
            categories_ready.append(category)
        else:
            blocked.append(category)
    weak_spots = [
        "Roadmap ML branches are frozen off and not part of demo runtime.",
        "Fragrance continuation ML remains blocked; fragrance demo stays rule-based.",
        "Makeup demo is supported by rules/catalog coverage, not by roadmap ML candidacy.",
    ]
    return {
        "catalog": catalog,
        "catalog_seed_report": catalog_report,
        "history_readiness_report": history_report,
        "runtime_smoke": smoke,
        "demo_categories_ready": categories_ready,
        "blocked_categories": blocked,
        "weak_spots": weak_spots,
        "verdict": {
            "demo_ready": bool(smoke.get("all_ok")),
            "roadmap_runtime_stable": bool(smoke.get("all_ok")),
        },
    }


def build_diploma_positioning_markdown() -> str:
    freeze = build_decision_freeze_payload()
    demo = build_demo_readiness_payload()
    lines = [
        "# Roadmap Positioning",
        "",
        "## Product Architecture",
        "",
        "- Initial roadmap generation in the shipped system is rule/teacher-based.",
        "- Continuation after user actions is handled by patched runtime continuation rules.",
        "- Roadmap ML branches were researched offline, but are frozen off in runtime.",
        "",
        "## What Is Actually Rule-Based",
        "",
        "- Roadmap initial plan construction comes from `roadmap_app.services.refresh_roadmap` and category rules.",
        "- Continuation decisions after step completion/skip come from runtime continuation heuristics in `roadmap_app.services.patch_step_status` and related helpers.",
        "- Fragrance runtime remains slot-based in roadmap logic while catalog SKUs stay retail-level (`edp`/`edt`/`body_mist`).",
        "",
        "## Where ML Is Still Real In The Project",
        "",
        "- Recommendation and reranking pipelines still use trained artifacts where they already provide stable product value.",
        "- Roadmap ML artifacts remain available only as offline research outputs and diagnostics.",
        "",
        "## Why Roadmap ML Was Not Shipped",
        "",
        "- Live initial roadmap ML was stronger than some trivial baselines offline for parts of haircare/skincare, but shadow agreement with runtime stayed near zero and full-plan rollout quality remained weak.",
        "- Continuation ML improved against trustworthy labels for some categories, but after retrain it still did not reach shadow-ready alignment with current runtime behavior.",
        "- The engineering decision was to keep the more stable rule/teacher product path instead of forcing a weak ML rollout.",
        "",
        "## How To Describe This In The Diploma",
        "",
        "- Present roadmap as a production rule-based subsystem with real catalog, real seeded user history, explainable step selection, and slot-aware fragrance logic.",
        "- Describe roadmap ML as a completed research track: datasets, baselines, shadow evaluation, and a freeze decision based on honest runtime-readiness criteria.",
        "- Frame the non-deployment of roadmap ML as a correctness and product-stability decision, not as a failed experiment.",
        "",
        "## Current Readiness",
        "",
        f"- Demo-ready categories: {', '.join(demo.get('demo_categories_ready') or []) or 'none'}.",
        f"- Roadmap runtime stable: {str((freeze.get('verdict') or {}).get('roadmap_runtime_stable'))}.",
        f"- Roadmap ML frozen: {str((freeze.get('verdict') or {}).get('roadmap_ml_frozen'))}.",
    ]
    return "\n".join(lines) + "\n"


def render_decision_freeze_md(payload: dict[str, Any]) -> str:
    initial_rows = []
    for category, row in sorted(((payload.get("why") or {}).get("initial_live_ml") or {}).get("highlights", {}).items()):
        initial_rows.append(
            [
                category,
                row.get("decisions"),
                row.get("stop_share"),
                row.get("shadow_first_step_match"),
                row.get("shadow_exact_plan_match"),
            ]
        )
    continuation_rows = []
    for category, row in sorted(((payload.get("why") or {}).get("continuation_ml") or {}).get("highlights", {}).items()):
        continuation_rows.append(
            [
                category,
                row.get("truth_runtime_accuracy"),
                row.get("truth_ml_accuracy"),
                row.get("shadow_next_action_match"),
                row.get("shadow_suffix_match"),
            ]
        )
    lines = [
        "# Roadmap Decision Freeze",
        "",
        "## Keep Enabled",
        "",
        f"- Initial roadmap: {((payload.get('keep_enabled') or {}).get('initial') or '')}",
        f"- Continuation roadmap: {((payload.get('keep_enabled') or {}).get('continuation') or '')}",
        "",
        "## Keep Disabled",
        "",
        f"- Live initial ML: {((payload.get('keep_disabled') or {}).get('initial_live_ml') or '')}",
        f"- Continuation ML: {((payload.get('keep_disabled') or {}).get('continuation_ml') or '')}",
        "",
        "## Initial Live ML Freeze Evidence",
        "",
        _table(
            ["category", "decisions", "stop_share", "shadow_first_step", "shadow_exact_plan"],
            initial_rows,
        ),
        "",
        "## Continuation ML Freeze Evidence",
        "",
        _table(
            ["category", "runtime_truth_acc", "ml_truth_acc", "shadow_next_action", "shadow_suffix"],
            continuation_rows,
        ),
        "",
        "## Recommended Architecture",
        "",
        f"- Initial runtime: {((payload.get('recommended_architecture') or {}).get('initial_runtime') or '')}",
        f"- Continuation runtime: {((payload.get('recommended_architecture') or {}).get('continuation_runtime') or '')}",
        f"- Catalog: {((payload.get('recommended_architecture') or {}).get('catalog') or '')}",
        f"- Roadmap ML: {((payload.get('recommended_architecture') or {}).get('roadmap_ml') or '')}",
        "",
        "## Verdict",
        "",
        f"- roadmap_runtime_stable = {str(((payload.get('verdict') or {}).get('roadmap_runtime_stable'))).lower()}",
        f"- roadmap_ml_frozen = {str(((payload.get('verdict') or {}).get('roadmap_ml_frozen'))).lower()}",
    ]
    return "\n".join(lines) + "\n"


def render_ml_freeze_status_md(payload: dict[str, Any]) -> str:
    flags = payload.get("runtime_flags") or {}
    refs = (payload.get("runtime_wrapper_refs") or {}).get("unexpected_runtime_references") or []
    rows = [[item.get("file"), item.get("token")] for item in refs]
    lines = [
        "# Roadmap ML Freeze Status",
        "",
        "## Runtime Flags",
        "",
        f"- planner_v1_mode = {flags.get('planner_v1_mode')}",
        f"- nextstep_v3_enabled = {str(flags.get('nextstep_v3_enabled')).lower()}",
        f"- nextstep_v4_enabled = {str(flags.get('nextstep_v4_enabled')).lower()}",
        f"- rule_only_expected = {str(flags.get('rule_only_expected')).lower()}",
        "",
        "## Runtime Source",
        "",
        f"- initial = {((payload.get('roadmap_logic_source') or {}).get('initial') or '')}",
        f"- continuation = {((payload.get('roadmap_logic_source') or {}).get('continuation') or '')}",
        f"- live_initial_ml_runtime = {((payload.get('roadmap_logic_source') or {}).get('live_initial_ml_runtime') or '')}",
        f"- live_continuation_ml_runtime = {((payload.get('roadmap_logic_source') or {}).get('live_continuation_ml_runtime') or '')}",
        "",
        "## Unexpected Runtime Wrapper References",
        "",
        _table(["file", "token"], rows),
        "",
        "## Verdict",
        "",
        f"- roadmap_ml_frozen = {str(((payload.get('verdict') or {}).get('roadmap_ml_frozen'))).lower()}",
    ]
    return "\n".join(lines) + "\n"


def render_demo_readiness_md(payload: dict[str, Any]) -> str:
    catalog = payload.get("catalog") or {}
    smoke_rows = []
    for row in payload.get("runtime_smoke", {}).get("categories", []) or []:
        smoke_rows.append(
            [
                row.get("category"),
                row.get("user_id"),
                row.get("next_step_product_type"),
                row.get("offer_assignment_id"),
                row.get("offer_target_matches_step"),
                row.get("reason"),
            ]
        )
    lines = [
        "# Roadmap Demo Readiness",
        "",
        "## Catalog Coverage",
        "",
        f"- products_total = {catalog.get('products_total')}",
        f"- by_category = {json.dumps(catalog.get('by_category') or {}, ensure_ascii=False, sort_keys=True)}",
        f"- fragrance_slots = {json.dumps(catalog.get('fragrance_slots') or {}, ensure_ascii=False, sort_keys=True)}",
        "",
        "## Runtime Smoke",
        "",
        _table(
            ["category", "user_id", "next_step", "offer_assignment_id", "offer_target_matches_step", "reason"],
            smoke_rows,
        ),
        "",
        "## Demo Readiness",
        "",
        f"- demo_categories_ready = {', '.join(payload.get('demo_categories_ready') or []) or 'none'}",
        f"- blocked_categories = {', '.join(payload.get('blocked_categories') or []) or 'none'}",
        f"- weak_spots = {'; '.join(payload.get('weak_spots') or [])}",
        "",
        "## Verdict",
        "",
        f"- demo_ready = {str(((payload.get('verdict') or {}).get('demo_ready'))).lower()}",
        f"- roadmap_runtime_stable = {str(((payload.get('verdict') or {}).get('roadmap_runtime_stable'))).lower()}",
    ]
    return "\n".join(lines) + "\n"


def write_report_bundle(
    *,
    payload: dict[str, Any],
    markdown: str,
    output_md: Path | None = None,
    output_json: Path | None = None,
) -> None:
    if output_md:
        _write_text(output_md, markdown)
    if output_json:
        _write_json(output_json, payload)
