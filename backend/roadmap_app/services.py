from __future__ import annotations

from collections import defaultdict
from typing import Any

from django.conf import settings
from django.db import transaction as db_tx
from django.db.models import Q
from django.utils import timezone

from catalog.models import Product
from ml_logic.recommender import recommend as rec_recommend
from offers.services import _build_rec_profile, _cooccurrence_90d, _load_products_for_recs
from roadmap_app.fragrance_slots import SLOTS as FRAGRANCE_SLOTS, slot_of_fragrance
from roadmap_app.ml_next_step import (
    predict_next_product_types,
    v4_category_staged_rollout_status,
    v4_min_lift_guard_status,
)
from roadmap_app.models import RoadmapPlan, RoadmapStep
from transactions.models import OwnedProduct, TransactionItem
from users_app.models import CustomerProfile


CATEGORY_RULES: dict[str, dict[str, Any]] = {
    "haircare": {
        "base": ["shampoo", "conditioner", "hair_mask", "hair_oil"],
        "optional": ["scalp_serum", "leave_in"],
        "min_steps": 4,
        "max_steps": 6,
    },
    "skincare": {
        "base": ["cleanser", "serum", "moisturizer", "spf"],
        "optional": ["toner", "mask", "eye_cream", "essence"],
        "min_steps": 4,
        "max_steps": 8,
    },
    "makeup": {
        "base": ["foundation", "mascara", "blush"],
        "optional": ["lipstick", "eyeshadow", "primer", "setting_spray"],
        "min_steps": 3,
        "max_steps": 7,
    },
    "fragrance": {
        "base": ["edp", "body_mist"],
        "optional": ["edt", "perfume_oil"],
        "min_steps": 2,
        "max_steps": 4,
    },
}


CADENCE_BY_TYPE: dict[str, str] = {
    "cleanser": RoadmapStep.Cadence.DAILY,
    "serum": RoadmapStep.Cadence.DAILY,
    "moisturizer": RoadmapStep.Cadence.DAILY,
    "spf": RoadmapStep.Cadence.DAILY,
    "toner": RoadmapStep.Cadence.DAILY,
    "mask": RoadmapStep.Cadence.WEEKLY,
    "shampoo": RoadmapStep.Cadence.WEEKLY,
    "conditioner": RoadmapStep.Cadence.WEEKLY,
    "hair_mask": RoadmapStep.Cadence.WEEKLY,
    "hair_oil": RoadmapStep.Cadence.OPTIONAL,
    "scalp_serum": RoadmapStep.Cadence.OPTIONAL,
}

FRAGRANCE_DEFAULT_CHAIN = ["warm_day", "warm_evening", "cold_day", "cold_evening"]


def _unique(values: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for v in values:
        s = str(v or "").strip()
        if not s:
            continue
        if s in seen:
            continue
        seen.add(s)
        out.append(s)
    return out


def _default_ml_mode_and_path() -> tuple[str, str]:
    use_v4 = bool(getattr(settings, "ROADMAP_NEXTSTEP_V4_ENABLED", False))
    use_v3 = bool(getattr(settings, "ROADMAP_NEXTSTEP_V3_ENABLED", False)) and not use_v4
    if use_v4:
        return "v4_ranking", str(getattr(settings, "ROADMAP_NEXTSTEP_V4_MODEL_PATH", "") or "")
    if use_v3:
        return "v3_multiclass", str(getattr(settings, "ROADMAP_NEXTSTEP_V3_MODEL_PATH", "") or "")
    return "legacy", str(getattr(settings, "ROADMAP_NEXTSTEP_MODEL_PATH", "") or "")


def _default_ml_threshold(mode: str) -> float:
    if str(mode) == "v4_ranking":
        return float(
            getattr(
                settings,
                "ROADMAP_NEXTSTEP_V4_CONFIDENCE_THRESHOLD",
                getattr(settings, "ROADMAP_NEXTSTEP_CONFIDENCE_THRESHOLD", 0.35),
            )
        )
    return float(getattr(settings, "ROADMAP_NEXTSTEP_CONFIDENCE_THRESHOLD", 0.35))


def _normalize_plan_meta(meta: dict[str, Any] | None) -> dict[str, Any]:
    out = dict(meta) if isinstance(meta, dict) else {}
    ml = out.get("ml")
    ml_out: dict[str, Any] = dict(ml) if isinstance(ml, dict) else {}
    default_mode, default_path = _default_ml_mode_and_path()
    mode = str(ml_out.get("mode") or default_mode)
    model_path = str(ml_out.get("model_path") or default_path)
    ml_out["mode"] = mode
    ml_out["model_path"] = model_path
    if "threshold" not in ml_out:
        ml_out["threshold"] = _default_ml_threshold(mode)
    if "predictions" not in ml_out:
        ml_out["predictions"] = []
    if "used" not in ml_out:
        ml_out["used"] = False
    if "category_guard" not in ml_out:
        ml_out["category_guard"] = None
    if "guard" not in ml_out:
        ml_out["guard"] = None

    decision_raw = str(ml_out.get("decision") or "").strip().lower()
    if decision_raw not in {"model_used", "fallback", "disabled"}:
        if bool(ml_out.get("used")):
            decision_raw = "model_used"
        elif str(ml_out.get("fallback_reason") or "").strip():
            decision_raw = "fallback"
        else:
            decision_raw = "disabled"
    ml_out["decision"] = decision_raw
    fallback_reason = str(ml_out.get("fallback_reason") or "").strip()
    disabled_reason = str(ml_out.get("disabled_reason") or "").strip()
    if decision_raw == "fallback":
        ml_out["disabled_reason"] = None
        ml_out["fallback_reason"] = fallback_reason or None
    elif decision_raw == "disabled":
        if not disabled_reason:
            disabled_reason = fallback_reason or "ml_disabled"
        ml_out["disabled_reason"] = disabled_reason
        ml_out["fallback_reason"] = None
    else:
        ml_out["disabled_reason"] = None
        ml_out["fallback_reason"] = None

    out["ml"] = ml_out
    return out


def _resolve_categories_from_post_ctx(post_ctx: dict[str, Any] | None) -> list[str]:
    if not post_ctx:
        return []
    raw = [str(x) for x in (post_ctx.get("categories") or []) if str(x)]
    categories = [x for x in raw if x in CATEGORY_RULES]
    if categories:
        return _unique(categories)

    product_ids = [int(x) for x in (post_ctx.get("product_ids") or []) if str(x).strip()]
    if not product_ids:
        return []

    rows = Product.objects.filter(id__in=product_ids).values("id", "category")
    by_id = {int(r["id"]): str(r["category"]) for r in rows}
    inferred: list[str] = []
    for pid in product_ids:
        cat = by_id.get(int(pid))
        if cat in CATEGORY_RULES:
            inferred.append(cat)
    return _unique(inferred)


def _post_ctx_types_by_category(post_ctx: dict[str, Any] | None) -> dict[str, list[str]]:
    out: dict[str, list[str]] = defaultdict(list)
    if not post_ctx:
        return out

    product_ids = [int(x) for x in (post_ctx.get("product_ids") or []) if str(x).strip()]
    if product_ids:
        rows = Product.objects.filter(id__in=product_ids).values("id", "category", "product_type")
        by_id = {
            int(r["id"]): {"category": str(r["category"]), "product_type": str(r["product_type"])}
            for r in rows
        }
        for pid in product_ids:
            row = by_id.get(int(pid))
            if not row:
                continue
            cat = row["category"]
            pt = row["product_type"]
            if cat in CATEGORY_RULES and pt and pt not in out[cat]:
                out[cat].append(pt)
        return out

    categories = [str(x) for x in (post_ctx.get("categories") or [])]
    product_types = _unique([str(x) for x in (post_ctx.get("product_types") or [])])
    if len(categories) == 1 and categories[0] in CATEGORY_RULES:
        out[categories[0]] = product_types
    return out


def _context_product_ids(user, post_ctx: dict[str, Any] | None, limit: int = 50) -> list[int]:
    seed = [int(x) for x in (post_ctx or {}).get("product_ids", []) if str(x).strip()]
    recent = list(
        TransactionItem.objects.filter(transaction__user=user)
        .order_by("-transaction__created_at", "-id")
        .values_list("product_id", flat=True)[: max(200, limit * 10)]
    )
    merged = seed + [int(x) for x in recent]
    out: list[int] = []
    seen: set[int] = set()
    for pid in merged:
        if pid in seen:
            continue
        seen.add(pid)
        out.append(int(pid))
        if len(out) >= int(limit):
            break
    return out


def _distinct_catalog_types(category: str, *, exclude: set[str] | None = None, limit: int = 30) -> list[str]:
    exclude = exclude or set()
    rows = (
        Product.objects.filter(category=category, in_stock=True)
        .exclude(product_type__in=list(exclude))
        .values_list("product_type", flat=True)
        .distinct()[: int(limit)]
    )
    return _unique([str(x) for x in rows])


def _fragrance_slots_from_products_qs(rows: list[dict[str, Any]]) -> list[str]:
    slots: list[str] = []
    for row in rows:
        slot = slot_of_fragrance(
            row.get("attrs") or {},
            raw_meta=row.get("raw_meta") if isinstance(row.get("raw_meta"), dict) else None,
        )
        if slot in FRAGRANCE_SLOTS and slot not in slots:
            slots.append(slot)
    return slots


def _owned_fragrance_slots(user) -> list[str]:
    rows = list(
        OwnedProduct.objects.filter(user=user, is_active=True, product__category="fragrance")
        .select_related("product")
        .values("product__attrs", "product__raw_meta")
    )
    normalized = [{"attrs": r.get("product__attrs") or {}, "raw_meta": r.get("product__raw_meta") or {}} for r in rows]
    return _fragrance_slots_from_products_qs(normalized)


def _purchased_fragrance_slots(post_ctx: dict[str, Any] | None) -> list[str]:
    product_ids = [int(x) for x in (post_ctx or {}).get("product_ids", []) if str(x).strip()]
    if not product_ids:
        return []
    rows = list(
        Product.objects.filter(id__in=product_ids, category="fragrance")
        .values("attrs", "raw_meta")
    )
    return _fragrance_slots_from_products_qs(rows)


def _category_owned(user, category: str) -> tuple[list[OwnedProduct], set[int], list[str], set[str]]:
    owned_rows = list(
        OwnedProduct.objects.filter(user=user, is_active=True, product__category=category)
        .select_related("product")
        .order_by("-last_acquired_at", "-id")
    )
    owned_product_ids = {int(row.product_id) for row in owned_rows}
    owned_types_ordered = _unique([str(row.product.product_type) for row in owned_rows if row.product_id])
    owned_types_set = set(owned_types_ordered)
    return owned_rows, owned_product_ids, owned_types_ordered, owned_types_set


def _build_chain(
    *,
    user,
    category: str,
    purchased_types: list[str],
    owned_types_ordered: list[str],
    context_product_ids: list[int],
) -> tuple[list[str], dict[str, dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    rules = CATEGORY_RULES[category]
    min_steps = int(rules["min_steps"])
    max_steps = int(rules["max_steps"])

    if category == "fragrance":
        purchased_slots = [x for x in purchased_types if x in FRAGRANCE_SLOTS]
        owned_slots = [x for x in owned_types_ordered if x in FRAGRANCE_SLOTS]
        chain = _unique(purchased_slots + FRAGRANCE_DEFAULT_CHAIN + owned_slots)
        target_len = max(min_steps, min(max_steps, len(chain)))
        chain = chain[:target_len]
    else:
        chain = _unique(purchased_types + rules["base"] + rules["optional"] + owned_types_ordered)
        recent_types = list(
            TransactionItem.objects.filter(transaction__user=user, product__category=category)
            .order_by("-transaction__created_at", "-id")
            .values_list("product__product_type", flat=True)[:40]
        )
        chain = _unique(chain + [str(x) for x in recent_types])
        chain = _unique(chain + _distinct_catalog_types(category, exclude=set(chain), limit=30))
        owned_signal = min(max_steps - min_steps, len(set(owned_types_ordered)))
        target_len = min_steps + max(0, owned_signal)
        target_len = max(min_steps, min(max_steps, target_len))
        if len(chain) < target_len:
            chain = _unique(chain + _distinct_catalog_types(category, exclude=set(chain), limit=40))
        chain = chain[: min(max_steps, target_len)]

    source_by_type: dict[str, dict[str, Any]] = {pt: {"source": "rules", "score": None} for pt in chain}
    ml_predictions: list[dict[str, Any]] = []
    if not chain:
        return chain, source_by_type, ml_predictions, {
            "decision": "disabled",
            "fallback_reason": None,
            "disabled_reason": "no_candidate_types",
            "guard": None,
            "category_guard": None,
        }

    use_v4 = bool(getattr(settings, "ROADMAP_NEXTSTEP_V4_ENABLED", False))
    use_v3 = bool(getattr(settings, "ROADMAP_NEXTSTEP_V3_ENABLED", False)) and not use_v4
    ml_runtime: dict[str, Any] = {
        "decision": "disabled",
        "fallback_reason": None,
        "disabled_reason": "ml_disabled",
        "guard": None,
        "category_guard": None,
    }
    if not use_v4 and not use_v3:
        return chain, source_by_type, ml_predictions, ml_runtime

    if use_v4:
        threshold = float(
            getattr(
                settings,
                "ROADMAP_NEXTSTEP_V4_CONFIDENCE_THRESHOLD",
                getattr(settings, "ROADMAP_NEXTSTEP_CONFIDENCE_THRESHOLD", 0.35),
            )
        )
        category_guard_thresholds = {
            "min_plans": int(getattr(settings, "ROADMAP_NEXTSTEP_V4_CATEGORY_MIN_PLANS", 100)),
            "min_step_completion_lift": float(
                getattr(settings, "ROADMAP_NEXTSTEP_V4_MIN_STEP_COMPLETION_LIFT", 0.01)
            ),
            "min_offer_redeem_lift": float(
                getattr(settings, "ROADMAP_NEXTSTEP_V4_MIN_OFFER_REDEEM_LIFT", 0.005)
            ),
            "max_negative_step_ctr_lift_soft": float(
                getattr(
                    settings,
                    "ROADMAP_NEXTSTEP_V4_MAX_NEGATIVE_STEP_CTR_LIFT_SOFT",
                    getattr(settings, "ROADMAP_NEXTSTEP_V4_MAX_NEGATIVE_STEP_CTR_LIFT", -0.02),
                )
            ),
            "max_negative_offer_ctr_lift_soft": float(
                getattr(
                    settings,
                    "ROADMAP_NEXTSTEP_V4_MAX_NEGATIVE_OFFER_CTR_LIFT_SOFT",
                    getattr(settings, "ROADMAP_NEXTSTEP_V4_MAX_NEGATIVE_OFFER_CTR_LIFT", -0.03),
                )
            ),
            "allow_primary_win_despite_soft_ctr_drop": bool(
                getattr(settings, "ROADMAP_NEXTSTEP_V4_ALLOW_PRIMARY_WIN_DESPITE_SOFT_CTR_DROP", True)
            ),
        }
        staged = v4_category_staged_rollout_status(category)
        rollout = staged.get("rollout") if isinstance(staged.get("rollout"), dict) else {}
        final_status = str(staged.get("final_status") or "HOLD").upper()
        category_guard = {
            "passed": final_status == "ENABLE",
            "reason": str(staged.get("reason") or ""),
            "hold_reason": staged.get("hold_reason"),
            "final_status": final_status,
            "current_decision": str(staged.get("current_decision") or final_status),
            "recommendation_7d": str(staged.get("recommendation_7d") or ""),
            "recommendation_30d": str(staged.get("recommendation_30d") or ""),
            "stability_gate_failures": list(staged.get("stability_gate_failures") or []),
            "source_report_path_7d": str(staged.get("source_report_path_7d") or ""),
            "source_report_path_30d": str(staged.get("source_report_path_30d") or ""),
            "category": str(category),
            "cohort_mode": "fresh",
            "control": "non_model",
            "thresholds": category_guard_thresholds,
            "guard_7d": staged.get("guard_7d"),
            "guard_30d": staged.get("guard_30d"),
        }

        if final_status == "DISABLE":
            rollout_reason = str(staged.get("reason") or rollout.get("reason") or "category_disabled")
            disabled_reason = "ml_disabled" if rollout_reason == "ml_disabled" else "category_disabled"
            category_guard["passed"] = False
            category_guard["reason"] = rollout_reason
            ml_runtime = {
                "decision": "disabled",
                "fallback_reason": None,
                "disabled_reason": disabled_reason,
                "guard": None,
                "category_guard": category_guard,
            }
            return chain, source_by_type, ml_predictions, ml_runtime

        if final_status != "ENABLE":
            ml_runtime = {
                "decision": "fallback",
                "fallback_reason": "category_guard_failed",
                "disabled_reason": None,
                "guard": None,
                "category_guard": category_guard,
            }
            return chain, source_by_type, ml_predictions, ml_runtime

        guard = v4_min_lift_guard_status()
        ml_runtime = {
            "decision": "fallback",
            "fallback_reason": None,
            "disabled_reason": None,
            "guard": guard,
            "category_guard": category_guard,
        }
        if not bool(guard.get("passed")):
            ml_runtime["fallback_reason"] = str(guard.get("reason") or "min_lift_guard_blocked")
            return chain, source_by_type, ml_predictions, ml_runtime
        ml_predictions = predict_next_product_types(
            user,
            context_product_ids,
            category,
            candidate_types=chain,
        )
        if not ml_predictions:
            ml_runtime["fallback_reason"] = "no_predictions"
            return chain, source_by_type, ml_predictions, ml_runtime

        top = None
        for row in ml_predictions:
            pt = str(row.get("product_type") or row.get("candidate_type") or "").strip()
            if pt in chain:
                top = {"product_type": pt, "score": float(row.get("score", 0.0))}
                break
        if not top:
            ml_runtime["fallback_reason"] = "top_outside_guardrails"
            return chain, source_by_type, ml_predictions, ml_runtime
        if float(top["score"]) < threshold:
            ml_runtime["fallback_reason"] = "low_confidence"
            return chain, source_by_type, ml_predictions, ml_runtime

        top_pt = str(top["product_type"])
        top_score = float(top["score"])
        anchor = min(len(chain), max(0, len(purchased_types)))
        if top_pt in chain:
            old_idx = int(chain.index(top_pt))
            if old_idx >= anchor and old_idx != anchor:
                chain = list(chain)
                chain.pop(old_idx)
                chain.insert(anchor, top_pt)
        source_by_type[top_pt] = {"source": "ml_next_step", "score": top_score}
        ml_runtime["decision"] = "model_used"
        ml_runtime["fallback_reason"] = None
        ml_runtime["disabled_reason"] = None
        return chain, source_by_type, ml_predictions, ml_runtime

    threshold = float(getattr(settings, "ROADMAP_NEXTSTEP_CONFIDENCE_THRESHOLD", 0.35))
    ml_runtime = {
        "decision": "fallback",
        "fallback_reason": None,
        "disabled_reason": None,
        "guard": None,
        "category_guard": None,
    }
    ml_predictions = predict_next_product_types(
        user,
        context_product_ids,
        category,
        candidate_types=chain,
    )
    if not ml_predictions:
        ml_runtime["fallback_reason"] = "no_predictions"
        return chain, source_by_type, ml_predictions, ml_runtime

    score_by_type: dict[str, float] = {}
    for row in ml_predictions:
        pt = str(row.get("product_type") or row.get("candidate_type") or "").strip()
        if not pt or pt not in chain:
            continue
        score = float(row.get("score", 0.0))
        if score < threshold:
            continue
        prev = score_by_type.get(pt)
        if prev is None or score > prev:
            score_by_type[pt] = score

    if not score_by_type:
        ml_runtime["fallback_reason"] = "low_confidence"
        return chain, source_by_type, ml_predictions, ml_runtime

    anchor = min(len(chain), max(0, len(purchased_types)))
    prefix = list(chain[:anchor])
    suffix = list(chain[anchor:])
    suffix_pos = {pt: idx for idx, pt in enumerate(suffix)}
    suffix_sorted = sorted(
        suffix,
        key=lambda pt: (
            0 if pt in score_by_type else 1,
            -float(score_by_type.get(pt, 0.0)),
            int(suffix_pos.get(pt, 0)),
        ),
    )
    chain = prefix + suffix_sorted
    for pt, score in score_by_type.items():
        source_by_type[pt] = {"source": "ml_next_step", "score": float(score)}
    ml_runtime["decision"] = "model_used"
    ml_runtime["disabled_reason"] = None
    return chain, source_by_type, ml_predictions, ml_runtime


def _recommend_candidates_for_type(
    *,
    user,
    category: str,
    product_type: str,
    context_product_ids: list[int],
    owned_product_ids: set[int],
    used_recommended_ids: set[int],
    prof,
    products_for_recs: list[dict[str, Any]],
    co_map: dict[int, dict[int, int]],
) -> list[dict[str, Any]]:
    max_suggestions = max(5, int(getattr(settings, "ROADMAP_SUGGESTIONS_LIMIT", 10)))
    if category == "fragrance" and product_type in FRAGRANCE_SLOTS:
        # Soft slot layer: try slot-filtered candidates first, then fallback to regular fragrance picks.
        recs = rec_recommend(
            prof=prof,
            products=products_for_recs,
            owned_active_ids=sorted(list(owned_product_ids)),
            context_product_ids=context_product_ids[:50],
            category=category,
            product_type=None,
            limit=max(40, max_suggestions * 4),
            co=co_map,
        )
        slot_filtered: list[dict[str, Any]] = []
        for row in recs:
            product = row.get("product") or {}
            slot = slot_of_fragrance(product.get("attrs") or {}, raw_meta=product.get("raw_meta") or {})
            if slot == product_type:
                slot_filtered.append(row)
        if slot_filtered:
            recs = slot_filtered
    else:
        recs = rec_recommend(
            prof=prof,
            products=products_for_recs,
            owned_active_ids=sorted(list(owned_product_ids)),
            context_product_ids=context_product_ids[:50],
            category=category,
            product_type=product_type,
            limit=max(20, max_suggestions * 2),
            co=co_map,
        )

    filtered: list[dict[str, Any]] = []
    blocked = set(owned_product_ids) | set(used_recommended_ids)
    for row in recs:
        product = row.get("product") or {}
        pid_raw = product.get("id")
        try:
            pid = int(pid_raw)
        except Exception:
            continue
        if pid in blocked:
            continue
        if product.get("in_stock") is False:
            continue
        filtered.append(row)
        if len(filtered) >= max_suggestions:
            break

    if filtered:
        return filtered

    db_qs = Product.objects.filter(category=category, in_stock=True).exclude(id__in=list(blocked))
    if not (category == "fragrance" and product_type in FRAGRANCE_SLOTS):
        db_qs = db_qs.filter(product_type=product_type)
    fallback_rows = list(db_qs.values("id", "category", "product_type", "brand", "attrs", "raw_meta").order_by("id")[: max_suggestions * 2])
    out: list[dict[str, Any]] = []
    for row in fallback_rows:
        if category == "fragrance" and product_type in FRAGRANCE_SLOTS:
            slot = slot_of_fragrance(row.get("attrs") or {}, raw_meta=row.get("raw_meta") or {})
            if slot != product_type:
                continue
        out.append(
            {
                "product": row,
                "score": 0.0,
                "components": {"mode": "fallback_db"},
                "why": ["fallback catalog match"],
            }
        )
        if len(out) >= max_suggestions:
            break

    if out:
        return out

    # Ultimate fallback for slot mode: do not enforce slot filter to avoid empty roadmap recommendations.
    if category == "fragrance" and product_type in FRAGRANCE_SLOTS:
        fallback_rows = list(
            Product.objects.filter(category=category, in_stock=True)
            .exclude(id__in=list(blocked))
            .values("id", "category", "product_type", "brand", "attrs")
            .order_by("id")[:max_suggestions]
        )
        for row in fallback_rows:
            out.append(
                {
                    "product": row,
                    "score": 0.0,
                    "components": {"mode": "fallback_db_no_slot"},
                    "why": ["fallback catalog match (slot relaxed)"],
                }
            )
            if len(out) >= max_suggestions:
                break
    return out


def _status_for_type(product_type: str, owned_types_set: set[str], purchased_types_set: set[str]) -> str:
    if product_type in purchased_types_set:
        return RoadmapStep.Status.COMPLETED
    if product_type in owned_types_set:
        return RoadmapStep.Status.OWNED
    return RoadmapStep.Status.MISSING


def _cadence_for_type(product_type: str) -> str:
    return CADENCE_BY_TYPE.get(product_type, RoadmapStep.Cadence.OPTIONAL)


def _upsert_plan_with_steps(
    *,
    user,
    category: str,
    meta: dict[str, Any],
    step_payloads: list[dict[str, Any]],
) -> RoadmapPlan:
    with db_tx.atomic():
        active_plans = list(
            RoadmapPlan.objects.select_for_update()
            .filter(user=user, category=category, is_active=True)
            .order_by("-updated_at", "-id")
        )

        plan: RoadmapPlan
        if active_plans:
            plan = active_plans[0]
            stale_ids = [p.id for p in active_plans[1:]]
            if stale_ids:
                RoadmapPlan.objects.filter(id__in=stale_ids).update(is_active=False)
        else:
            plan = RoadmapPlan.objects.create(user=user, category=category, is_active=True, meta={})

        plan.meta = _normalize_plan_meta(meta)
        plan.save(update_fields=["meta", "updated_at"])

        keep_indexes: list[int] = []
        for idx, payload in enumerate(step_payloads, start=1):
            keep_indexes.append(idx)
            RoadmapStep.objects.update_or_create(
                plan=plan,
                step_index=idx,
                defaults=payload,
            )

        RoadmapStep.objects.filter(plan=plan).exclude(step_index__in=keep_indexes).delete()

    return (
        RoadmapPlan.objects.filter(id=plan.id)
        .prefetch_related("steps", "steps__recommended_product")
        .first()
    ) or plan


def refresh_roadmap(user, category: str, post_ctx: dict[str, Any] | None = None) -> RoadmapPlan:
    category = str(category or "").strip()
    if category not in CATEGORY_RULES:
        raise ValueError(f"Unsupported roadmap category: {category}")

    now = timezone.now()
    _, owned_product_ids, owned_types_ordered, owned_types_set = _category_owned(user, category)
    purchased_by_category = _post_ctx_types_by_category(post_ctx)
    purchased_types = _unique(purchased_by_category.get(category, []))
    if category == "fragrance":
        owned_types_ordered = _unique(_owned_fragrance_slots(user))
        owned_types_set = set(owned_types_ordered)
        purchased_types = _unique(_purchased_fragrance_slots(post_ctx))
    purchased_types_set = set(purchased_types)
    context_product_ids = _context_product_ids(user, post_ctx, limit=50)

    chain, source_by_type, ml_predictions, ml_runtime = _build_chain(
        user=user,
        category=category,
        purchased_types=purchased_types,
        owned_types_ordered=owned_types_ordered,
        context_product_ids=context_product_ids,
    )

    cp, _ = CustomerProfile.objects.get_or_create(user=user)
    prof = _build_rec_profile(cp)
    products_for_recs = _load_products_for_recs()
    co_map = _cooccurrence_90d(now)
    used_recommended_ids: set[int] = set()

    step_payloads: list[dict[str, Any]] = []
    for product_type in chain:
        source_meta = source_by_type.get(product_type, {"source": "rules", "score": None})
        source_is_ml = source_meta.get("source") == "ml_next_step"
        score_val = source_meta.get("score")

        status = _status_for_type(product_type, owned_types_set, purchased_types_set)
        suggestions: list[int] = []
        recommended_product_id: int | None = None
        rec_top = None

        if status in {RoadmapStep.Status.MISSING, RoadmapStep.Status.RECOMMENDED}:
            recs = _recommend_candidates_for_type(
                user=user,
                category=category,
                product_type=product_type,
                context_product_ids=context_product_ids,
                owned_product_ids=owned_product_ids,
                used_recommended_ids=used_recommended_ids,
                prof=prof,
                products_for_recs=products_for_recs,
                co_map=co_map,
            )
            suggestions = []
            for row in recs:
                product = row.get("product") or {}
                try:
                    pid = int(product.get("id"))
                except Exception:
                    continue
                suggestions.append(pid)
            suggestions = _unique([str(x) for x in suggestions])
            suggestions = [int(x) for x in suggestions][: max(5, int(getattr(settings, "ROADMAP_SUGGESTIONS_LIMIT", 10)))]

            if suggestions:
                recommended_product_id = int(suggestions[0])
                used_recommended_ids.add(recommended_product_id)
                status = RoadmapStep.Status.RECOMMENDED
                rec_top = recs[0] if recs else None
            else:
                status = RoadmapStep.Status.MISSING

        why: list[str] = []
        why.append("picked via ML next_step" if source_is_ml else "picked via rules")
        if status in {RoadmapStep.Status.OWNED, RoadmapStep.Status.COMPLETED}:
            why.append("already owned")
        elif recommended_product_id:
            why.append("recommended via reranker/cooc")
            match_why = (rec_top or {}).get("why") or []
            for item in match_why[:4]:
                why.append(str(item))
        else:
            why.append("recommended via reranker/cooc: no suitable candidates")

        step_payloads.append(
            {
                "product_type": product_type,
                "status": status,
                "recommended_product_id": recommended_product_id,
                "suggestions": suggestions,
                "score": float((rec_top or {}).get("score", score_val))
                if (rec_top is not None or score_val is not None)
                else None,
                "confidence": float(score_val) if score_val is not None else None,
                "why": why,
                "cadence": _cadence_for_type(product_type),
            }
        )

    use_v4 = bool(getattr(settings, "ROADMAP_NEXTSTEP_V4_ENABLED", False))
    use_v3 = bool(getattr(settings, "ROADMAP_NEXTSTEP_V3_ENABLED", False))
    threshold = float(
        getattr(
            settings,
            "ROADMAP_NEXTSTEP_V4_CONFIDENCE_THRESHOLD",
            getattr(settings, "ROADMAP_NEXTSTEP_CONFIDENCE_THRESHOLD", 0.35),
        )
    ) if use_v4 else float(getattr(settings, "ROADMAP_NEXTSTEP_CONFIDENCE_THRESHOLD", 0.35))
    if use_v4:
        model_path = str(getattr(settings, "ROADMAP_NEXTSTEP_V4_MODEL_PATH", "") or "")
        ml_mode = "v4_ranking"
    elif use_v3:
        model_path = str(getattr(settings, "ROADMAP_NEXTSTEP_V3_MODEL_PATH", "") or "")
        ml_mode = "v3_multiclass"
    else:
        model_path = str(getattr(settings, "ROADMAP_NEXTSTEP_MODEL_PATH", "") or "")
        ml_mode = "legacy"
    meta = {
        "generation_state": "ok",
        "source": "roadmap_v1",
        "category": category,
        "generated_at": now.isoformat(),
        "ml": {
            "threshold": threshold,
            "predictions": ml_predictions[:10],
            "used": any((x.get("source") == "ml_next_step") for x in source_by_type.values()),
            "decision": str(ml_runtime.get("decision") or "fallback"),
            "fallback_reason": ml_runtime.get("fallback_reason"),
            "disabled_reason": ml_runtime.get("disabled_reason"),
            "guard": ml_runtime.get("guard"),
            "category_guard": ml_runtime.get("category_guard"),
            "mode": ml_mode,
            "model_path": model_path,
        },
        "context": {
            "refresh_caller": (
                "update_roadmap_from_purchase"
                if (post_ctx and ((post_ctx.get("product_ids") or []) or (post_ctx.get("categories") or [])))
                else "refresh_roadmap"
            ),
            "post_ctx_categories": (post_ctx or {}).get("categories") or [],
            "post_ctx_product_ids": (post_ctx or {}).get("product_ids") or [],
            "context_product_ids_count": len(context_product_ids),
            "owned_product_types_count": len(owned_types_set),
            "purchased_types_count": len(purchased_types_set),
        },
    }

    return _upsert_plan_with_steps(user=user, category=category, meta=meta, step_payloads=step_payloads)


def get_active_plan(user, category: str) -> RoadmapPlan | None:
    category = str(category or "").strip()
    if not category:
        return (
            RoadmapPlan.objects.filter(user=user, is_active=True)
            .prefetch_related("steps", "steps__recommended_product")
            .order_by("-updated_at", "-id")
            .first()
        )
    return (
        RoadmapPlan.objects.filter(user=user, category=category, is_active=True)
        .prefetch_related("steps", "steps__recommended_product")
        .order_by("-updated_at", "-id")
        .first()
    )


def get_next_missing_step(plan: RoadmapPlan | None) -> RoadmapStep | None:
    if not plan:
        return None
    return (
        plan.steps.filter(status__in=[RoadmapStep.Status.MISSING, RoadmapStep.Status.RECOMMENDED])
        .order_by("step_index")
        .first()
    )


def build_plan_summary(plan: RoadmapPlan | None) -> dict[str, Any]:
    if not plan:
        return {
            "next_step": None,
            "missing_steps_count": 0,
            "total_steps": 0,
        }
    next_step = get_next_missing_step(plan)
    missing_count = plan.steps.filter(
        status__in=[RoadmapStep.Status.MISSING, RoadmapStep.Status.RECOMMENDED]
    ).count()
    total_steps = plan.steps.count()
    return {
        "next_step": {
            "id": next_step.id,
            "step_index": next_step.step_index,
            "product_type": next_step.product_type,
            "status": next_step.status,
            "recommended_product_id": next_step.recommended_product_id,
        }
        if next_step
        else None,
        "missing_steps_count": int(missing_count),
        "total_steps": int(total_steps),
    }


def update_roadmap_from_purchase(user, post_ctx: dict[str, Any] | None) -> dict[str, Any] | None:
    categories = _resolve_categories_from_post_ctx(post_ctx)
    if not categories:
        return None

    plans_by_category: dict[str, RoadmapPlan] = {}
    for category in categories:
        plans_by_category[category] = refresh_roadmap(user, category=category, post_ctx=post_ctx)

    selected_category = categories[0]
    selected_plan = plans_by_category[selected_category]
    next_step = get_next_missing_step(selected_plan)

    roadmap_ctx: dict[str, Any] = {
        "category": selected_category,
        "plan_id": selected_plan.id,
    }
    if next_step:
        roadmap_ctx["next_product_type"] = next_step.product_type
        if next_step.recommended_product_id:
            roadmap_ctx["next_product_id"] = int(next_step.recommended_product_id)

    return {
        "category": selected_category,
        "plan": selected_plan,
        "next_missing_step": next_step,
        "roadmap_ctx": roadmap_ctx,
    }


def match_completed_steps_for_purchase(user, post_ctx: dict[str, Any] | None) -> list[dict[str, Any]]:
    categories = _resolve_categories_from_post_ctx(post_ctx)
    if not categories:
        return []

    purchased_ids = {
        int(x) for x in (post_ctx or {}).get("product_ids", []) if str(x).strip()
    }
    purchased_by_category = _post_ctx_types_by_category(post_ctx)
    purchased_fragrance_slots = set(_purchased_fragrance_slots(post_ctx))

    out: list[dict[str, Any]] = []
    for category in categories:
        plan = get_active_plan(user, category=category)
        if not plan:
            continue
        step = get_next_missing_step(plan)
        if not step:
            continue

        matched_by = None
        if step.recommended_product_id and int(step.recommended_product_id) in purchased_ids:
            matched_by = "recommended_product_id"
        elif category == "fragrance":
            if step.product_type in purchased_fragrance_slots:
                matched_by = "fragrance_slot"
        else:
            if step.product_type in set(purchased_by_category.get(category, [])):
                matched_by = "product_type"

        if matched_by:
            out.append(
                {
                    "category": category,
                    "plan": plan,
                    "step": step,
                    "matched_by": matched_by,
                }
            )
    return out


def patch_step_status(*, user, step_id: int, status: str) -> RoadmapStep:
    allowed = {choice for choice, _ in RoadmapStep.Status.choices}
    if status not in allowed:
        raise ValueError("Unsupported status")
    step = RoadmapStep.objects.select_related("plan").get(
        Q(id=step_id),
        Q(plan__user=user),
        Q(plan__is_active=True),
    )
    step.status = status
    step.save(update_fields=["status", "updated_at"])
    return step
