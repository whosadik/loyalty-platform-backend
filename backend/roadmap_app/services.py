from __future__ import annotations

import hashlib
from collections import defaultdict
from typing import Any

from django.conf import settings
from django.db import transaction as db_tx
from django.db.models import Q
from django.utils import timezone

from catalog.models import Product
from ml_logic.recommender import recommend as rec_recommend
from offers.services import _build_rec_profile, _cooccurrence_90d, _load_products_for_recs
from roadmap_app.content_features import product_signature, profile_signature
from roadmap_app.events import emit_plan_refresh_events
from roadmap_app.fragrance_slots import SLOTS as FRAGRANCE_SLOTS, slot_of_fragrance
from roadmap_app.ml_planner import generate_planner_chain, planner_runtime_mode
from roadmap_app.ml_next_step import (
    nextstep_model_artifact_summary,
    predict_next_product_types,
    predict_next_product_types_for_model_path,
    v4_category_staged_rollout_status,
    v4_min_lift_guard_status,
)
from roadmap_app.models import RoadmapPlan, RoadmapStep
from transactions.models import OwnedProduct, TransactionItem
from users_app.models import CustomerProfile
from users_app.services import favorite_category_snapshot


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


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value or "").strip().lower() in {"1", "true", "yes", "y", "on"}


def _to_int_or_none(value: Any) -> int | None:
    try:
        if value is None:
            return None
        return int(value)
    except Exception:
        return None


def _normalized_token_set(value: Any) -> set[str]:
    out: set[str] = set()
    if isinstance(value, str):
        values = value.split(",")
    elif isinstance(value, (list, tuple, set)):
        values = list(value)
    else:
        values = []
    for raw in values:
        token = str(raw or "").strip().lower()
        if token:
            out.add(token)
    return out


def _normalized_int_set(value: Any) -> set[int]:
    out: set[int] = set()
    if isinstance(value, str):
        values = value.split(",")
    elif isinstance(value, (list, tuple, set)):
        values = list(value)
    else:
        values = []
    for raw in values:
        val = _to_int_or_none(raw)
        if val is None:
            continue
        if int(val) <= 0:
            continue
        out.add(int(val))
    return out


def _default_rollout_meta() -> dict[str, Any]:
    return {
        "rollout_mode": "none",
        "rollout_selected": False,
        "rollout_reason": "not_applicable",
        "rollout_bucket": None,
        "rollout_percent": None,
        "partial_match_product_type": None,
        "partial_match_step_index": None,
    }


def _stable_rollout_bucket(user, *, category: str, salt: str) -> int:
    user_key = getattr(user, "pk", None)
    if user_key in (None, ""):
        user_key = getattr(user, "id", None)
    if user_key in (None, ""):
        user_key = getattr(user, "username", "anon")
    raw = f"{str(salt)}:{str(category)}:{str(user_key)}".encode("utf-8")
    digest = hashlib.sha1(raw).hexdigest()[:8]
    return int(digest, 16) % 100


def _partial_rollout_allowlists(category: str) -> tuple[set[str], set[int]]:
    category_norm = str(category or "").strip().lower()
    if not category_norm:
        return set(), set()
    product_setting = f"ROADMAP_NEXTSTEP_V4_PARTIAL_{category_norm.upper()}_PRODUCT_TYPES"
    step_setting = f"ROADMAP_NEXTSTEP_V4_PARTIAL_{category_norm.upper()}_STEP_INDEXES"
    allow_product_types = _normalized_token_set(getattr(settings, product_setting, []))
    allow_step_indexes = _normalized_int_set(getattr(settings, step_setting, []))
    return allow_product_types, allow_step_indexes


def _category_partial_rollout_state(
    *,
    user,
    category: str,
    chain: list[str],
    purchased_types: list[str],
) -> dict[str, Any]:
    state = _default_rollout_meta()
    category_norm = str(category or "").strip().lower()
    if not category_norm:
        return state

    enabled_categories = _normalized_token_set(
        getattr(settings, "ROADMAP_NEXTSTEP_V4_PARTIAL_ENABLED_CATEGORIES", [])
    )
    if category_norm not in enabled_categories:
        state["rollout_reason"] = "partial_disabled_for_category"
        return state

    state["rollout_mode"] = "partial"
    percent = max(
        0,
        min(
            100,
            int(getattr(settings, "ROADMAP_NEXTSTEP_V4_PARTIAL_PERCENT", 0) or 0),
        ),
    )
    salt = str(
        getattr(settings, "ROADMAP_NEXTSTEP_V4_PARTIAL_SALT", "roadmap_nextstep_v4_partial_v1")
        or "roadmap_nextstep_v4_partial_v1"
    ).strip()
    state["rollout_percent"] = percent
    state["rollout_bucket"] = _stable_rollout_bucket(
        user,
        category=category_norm,
        salt=salt,
    )

    allow_product_types, allow_step_indexes = _partial_rollout_allowlists(category_norm)

    if chain:
        anchor = min(len(chain), max(0, len(purchased_types)))
        step_index = min(max(1, anchor + 1), len(chain))
        step_product_type = str(chain[step_index - 1] or "").strip().lower()
    else:
        step_index = None
        step_product_type = ""

    if allow_product_types and step_product_type and step_product_type in allow_product_types:
        state["partial_match_product_type"] = step_product_type
    if allow_step_indexes and step_index is not None and int(step_index) in allow_step_indexes:
        state["partial_match_step_index"] = int(step_index)

    if allow_product_types and allow_step_indexes:
        allowlist_passed = bool(
            state["partial_match_product_type"] or state["partial_match_step_index"] is not None
        )
    elif allow_product_types:
        allowlist_passed = bool(state["partial_match_product_type"])
    elif allow_step_indexes:
        allowlist_passed = bool(state["partial_match_step_index"] is not None)
    else:
        allowlist_passed = False

    if not allowlist_passed:
        if not allow_product_types and not allow_step_indexes:
            state["rollout_reason"] = "allowlist_empty"
        else:
            state["rollout_reason"] = "allowlist_no_match"
        return state

    if int(state["rollout_bucket"]) >= int(percent):
        state["rollout_reason"] = "bucket_out_of_range"
        return state

    state["rollout_selected"] = True
    state["rollout_reason"] = "selected"
    return state


def _partial_candidate_model_path(category: str | None = None) -> str:
    category_norm = str(category or "").strip().lower()
    if category_norm:
        category_setting = f"ROADMAP_NEXTSTEP_V4_PARTIAL_{category_norm.upper()}_MODEL_PATH"
        category_value = str(getattr(settings, category_setting, "") or "").strip()
        if category_value:
            return category_value
    return str(getattr(settings, "ROADMAP_NEXTSTEP_V4_PARTIAL_MODEL_PATH", "") or "").strip()


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

    rollout_mode = str(ml_out.get("rollout_mode") or "").strip().lower()
    if rollout_mode not in {"full", "partial", "none"}:
        rollout_mode = "none"
    ml_out["rollout_mode"] = rollout_mode
    rollout_selected = _as_bool(ml_out.get("rollout_selected"))
    if rollout_mode == "full":
        rollout_selected = True
    elif rollout_mode == "none":
        rollout_selected = False
    ml_out["rollout_selected"] = rollout_selected
    rollout_reason = str(ml_out.get("rollout_reason") or "").strip()
    ml_out["rollout_reason"] = rollout_reason or "not_applicable"

    rollout_bucket = _to_int_or_none(ml_out.get("rollout_bucket"))
    if rollout_bucket is None or rollout_bucket < 0 or rollout_bucket >= 100:
        rollout_bucket = None
    ml_out["rollout_bucket"] = rollout_bucket

    rollout_percent = _to_int_or_none(ml_out.get("rollout_percent"))
    if rollout_percent is not None:
        rollout_percent = max(0, min(100, int(rollout_percent)))
    ml_out["rollout_percent"] = rollout_percent

    partial_match_product_type = str(ml_out.get("partial_match_product_type") or "").strip().lower()
    ml_out["partial_match_product_type"] = partial_match_product_type or None
    partial_match_step_index = _to_int_or_none(ml_out.get("partial_match_step_index"))
    if partial_match_step_index is not None and partial_match_step_index <= 0:
        partial_match_step_index = None
    ml_out["partial_match_step_index"] = partial_match_step_index

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


def _post_ctx_products(post_ctx: dict[str, Any] | None) -> list[dict[str, Any]]:
    product_ids = [int(x) for x in (post_ctx or {}).get("product_ids", []) if str(x).strip()]
    if not product_ids:
        return []
    rows = list(
        Product.objects.filter(id__in=product_ids).values(
            "id",
            "category",
            "product_type",
            "concerns",
            "actives",
            "flags",
            "supported_skin_types",
            "attrs",
            "ingredients_inci",
            "raw_meta",
        )
    )
    by_id = {int(row["id"]): row for row in rows}
    out: list[dict[str, Any]] = []
    for pid in product_ids:
        row = by_id.get(int(pid))
        if row:
            out.append(row)
    return out


def _profile_match_reasons(profile_sig: dict[str, Any], product_sig: dict[str, Any]) -> list[str]:
    category = str(product_sig.get("category") or "")
    reasons: list[str] = []

    if category == "skincare":
        supported = set(product_sig.get("supported_skin_types") or [])
        skin_type = str(profile_sig.get("skin_type") or "__none__")
        if supported and skin_type != "__none__" and skin_type in supported:
            reasons.append("skin_type")
        if set(product_sig.get("concerns") or []).intersection(set(profile_sig.get("goals") or [])):
            reasons.append("goals")
        return reasons

    if category == "haircare":
        for key in ["hair_type", "scalp_type", "hair_thickness"]:
            if str(profile_sig.get(key) or "__none__") != "__none__" and str(profile_sig.get(key)) == str(product_sig.get(key)):
                reasons.append(key)
        if set(product_sig.get("concerns") or []).intersection(set(profile_sig.get("hair_concerns") or [])):
            reasons.append("hair_concerns")
        return reasons

    if category == "makeup":
        if str(product_sig.get("finish") or "__none__") in set(profile_sig.get("makeup_finish_pref") or []):
            reasons.append("finish")
        if str(product_sig.get("coverage") or "__none__") in set(profile_sig.get("makeup_coverage_pref") or []):
            reasons.append("coverage")
        for key, profile_key in [
            ("undertone", "makeup_undertone"),
            ("tone_family", "makeup_tone_family"),
        ]:
            if str(profile_sig.get(profile_key) or "__none__") != "__none__" and str(profile_sig.get(profile_key)) == str(product_sig.get(key)):
                reasons.append(key)
        if set(product_sig.get("concerns") or []).intersection(set(profile_sig.get("makeup_concerns") or [])):
            reasons.append("makeup_concerns")
        return reasons

    if category == "fragrance":
        if str(product_sig.get("scent_family") or "__none__") in set(profile_sig.get("fragrance_liked_families") or []):
            reasons.append("scent_family")
        if str(product_sig.get("intensity") or "__none__") == str(profile_sig.get("fragrance_intensity_pref") or "__none__"):
            if str(product_sig.get("intensity") or "__none__") != "__none__":
                reasons.append("intensity")
        if set(product_sig.get("notes") or []).intersection(set(profile_sig.get("fragrance_liked_notes") or [])):
            reasons.append("notes")
    return reasons


def _semantic_completion_match_meta(
    *,
    user,
    step: RoadmapStep,
    purchased_products: list[dict[str, Any]],
) -> dict[str, Any] | None:
    if not step.recommended_product_id or not purchased_products:
        return None

    recommended_row = (
        Product.objects.filter(id=step.recommended_product_id)
        .values(
            "id",
            "category",
            "product_type",
            "concerns",
            "actives",
            "flags",
            "supported_skin_types",
            "attrs",
            "ingredients_inci",
            "raw_meta",
        )
        .first()
    )
    if not recommended_row:
        return None

    profile_sig = profile_signature(CustomerProfile.objects.filter(user=user).first())
    recommended_sig = product_signature(recommended_row)
    step_product_type = str(step.product_type or "").strip().lower()

    best_meta: dict[str, Any] | None = None
    best_score = 0.0

    for product in purchased_products:
        purchased_sig = product_signature(product)
        if str(purchased_sig.get("category") or "") != str(recommended_sig.get("category") or ""):
            continue
        if step_product_type and str(purchased_sig.get("product_type") or "") != step_product_type:
            continue

        shared_concerns = sorted(
            set(recommended_sig.get("concerns") or []).intersection(set(purchased_sig.get("concerns") or []))
        )[:6]
        shared_actives = sorted(
            set(recommended_sig.get("actives") or []).intersection(set(purchased_sig.get("actives") or []))
        )[:6]
        shared_inci_tokens = sorted(
            set(recommended_sig.get("inci_tokens") or []).intersection(set(purchased_sig.get("inci_tokens") or []))
        )[:6]
        attr_matches = [
            field
            for field in [
                "hair_type",
                "scalp_type",
                "hair_thickness",
                "finish",
                "coverage",
                "undertone",
                "tone_family",
                "scent_family",
                "intensity",
            ]
            if str(recommended_sig.get(field) or "__none__") != "__none__"
            and str(recommended_sig.get(field) or "") == str(purchased_sig.get(field) or "")
        ]
        profile_reasons = _profile_match_reasons(profile_sig, purchased_sig)
        evidence_count = (
            len(shared_concerns)
            + len(shared_actives)
            + len(shared_inci_tokens)
            + len(attr_matches)
            + len(profile_reasons)
        )
        if evidence_count <= 0:
            continue

        score = (
            1.0
            + 0.35 * len(shared_concerns)
            + 0.25 * len(shared_actives)
            + 0.05 * len(shared_inci_tokens)
            + 0.20 * len(attr_matches)
            + 0.20 * len(profile_reasons)
        )
        if score <= best_score:
            continue
        best_score = float(score)
        best_meta = {
            "recommended_product_id": int(recommended_row["id"]),
            "purchased_product_id": int(product["id"]),
            "purchased_product_type": str(product.get("product_type") or ""),
            "semantic_score": round(float(score), 4),
            "shared_concerns": shared_concerns,
            "shared_actives": shared_actives,
            "shared_inci_tokens": shared_inci_tokens,
            "attribute_matches": attr_matches,
            "profile_match_reasons": profile_reasons,
        }

    if best_meta and float(best_meta.get("semantic_score") or 0.0) >= 1.25:
        return best_meta
    return None


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


def _planner_candidate_universe(
    *,
    category: str,
    purchased_types: list[str],
    owned_types_ordered: list[str],
) -> list[str]:
    if category == "fragrance":
        purchased_slots = [x for x in purchased_types if x in FRAGRANCE_SLOTS]
        owned_slots = [x for x in owned_types_ordered if x in FRAGRANCE_SLOTS]
        return _unique(purchased_slots + FRAGRANCE_DEFAULT_CHAIN + owned_slots)
    rules = CATEGORY_RULES[category]
    return _unique(purchased_types + owned_types_ordered + rules["base"] + rules["optional"])


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
    default_rollout_meta = _default_rollout_meta()
    if not chain:
        return chain, source_by_type, ml_predictions, {
            "decision": "disabled",
            "fallback_reason": None,
            "disabled_reason": "no_candidate_types",
            "guard": None,
            "category_guard": None,
            **default_rollout_meta,
        }

    anchor = min(len(chain), max(0, len(purchased_types)))
    planned_target_product_type = str(chain[anchor]) if anchor < len(chain) else ""
    planned_target_step_index = int(anchor + 1) if anchor < len(chain) else 0

    use_v4 = bool(getattr(settings, "ROADMAP_NEXTSTEP_V4_ENABLED", False))
    use_v3 = bool(getattr(settings, "ROADMAP_NEXTSTEP_V3_ENABLED", False)) and not use_v4
    ml_runtime: dict[str, Any] = {
        "decision": "disabled",
        "fallback_reason": None,
        "disabled_reason": "ml_disabled",
        "guard": None,
        "category_guard": None,
        **default_rollout_meta,
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
        rollout_meta = _default_rollout_meta()
        rollout_meta = _category_partial_rollout_state(
            user=user,
            category=category,
            chain=chain,
            purchased_types=purchased_types,
        )
        active_model_path = str(getattr(settings, "ROADMAP_NEXTSTEP_V4_MODEL_PATH", "") or "")
        partial_model_path = _partial_candidate_model_path(category)
        partial_selected = bool(rollout_meta.get("rollout_selected"))
        partial_active = str(rollout_meta.get("rollout_mode") or "") == "partial"
        served_model_path = active_model_path
        served_model_slot = "active"
        if partial_active and partial_selected and partial_model_path:
            served_model_path = partial_model_path
            served_model_slot = "partial_candidate"
        served_artifact = nextstep_model_artifact_summary(served_model_path)
        served_model_meta = {
            "model_path": served_model_path,
            "model_version": str((served_artifact or {}).get("model_version") or ""),
            "selected_feature_set": str((served_artifact or {}).get("selected_feature_set") or ""),
            "model_slot": served_model_slot,
        }

        if final_status == "DISABLE":
            rollout_reason = str(staged.get("reason") or rollout.get("reason") or "category_disabled")
            disabled_reason = "ml_disabled" if rollout_reason == "ml_disabled" else "category_disabled"
            category_guard["passed"] = False
            category_guard["reason"] = rollout_reason
            rollout_meta_disabled = {**rollout_meta, "rollout_selected": False}
            if str(rollout_meta_disabled.get("rollout_mode") or "") == "partial":
                rollout_meta_disabled["rollout_reason"] = rollout_reason
            ml_runtime = {
                "decision": "disabled",
                "fallback_reason": None,
                "disabled_reason": disabled_reason,
                "guard": None,
                "category_guard": category_guard,
                **served_model_meta,
                **rollout_meta_disabled,
            }
            return chain, source_by_type, ml_predictions, ml_runtime

        if final_status == "ENABLE":
            rollout_meta = {
                **_default_rollout_meta(),
                "rollout_mode": "full",
                "rollout_selected": True,
                "rollout_reason": "category_enabled",
                "rollout_percent": 100,
            }
        elif partial_active and partial_selected:
            rollout_meta = {
                **rollout_meta,
                "rollout_mode": "partial",
                "rollout_selected": True,
                "rollout_reason": str(rollout_meta.get("rollout_reason") or "selected"),
            }
        elif partial_active and not partial_selected:
            ml_runtime = {
                "decision": "fallback",
                "fallback_reason": "partial_rollout_not_selected",
                "disabled_reason": None,
                "guard": None,
                "category_guard": category_guard,
                **served_model_meta,
                **rollout_meta,
            }
            return chain, source_by_type, ml_predictions, ml_runtime
        elif final_status != "ENABLE":
            ml_runtime = {
                "decision": "fallback",
                "fallback_reason": "category_guard_failed",
                "disabled_reason": None,
                "guard": None,
                "category_guard": category_guard,
                **served_model_meta,
                **rollout_meta,
            }
            return chain, source_by_type, ml_predictions, ml_runtime

        guard = v4_min_lift_guard_status()
        ml_runtime = {
            "decision": "fallback",
            "fallback_reason": None,
            "disabled_reason": None,
            "guard": guard,
            "category_guard": category_guard,
            "planned_target_product_type": planned_target_product_type,
            "planned_target_step_index": planned_target_step_index,
            **served_model_meta,
            **rollout_meta,
        }
        if not bool(guard.get("passed")):
            ml_runtime["fallback_reason"] = str(guard.get("reason") or "min_lift_guard_blocked")
            return chain, source_by_type, ml_predictions, ml_runtime
        if served_model_slot == "partial_candidate":
            ml_predictions = predict_next_product_types_for_model_path(
                served_model_path,
                user=user,
                context_product_ids=context_product_ids,
                category=category,
                planned_target_product_type=planned_target_product_type,
                planned_target_step_index=planned_target_step_index,
                candidate_types=chain,
            )
        else:
            ml_predictions = predict_next_product_types(
                user,
                context_product_ids,
                category,
                planned_target_product_type=planned_target_product_type,
                planned_target_step_index=planned_target_step_index,
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
        "planned_target_product_type": planned_target_product_type,
        "planned_target_step_index": planned_target_step_index,
        **_default_rollout_meta(),
    }
    ml_predictions = predict_next_product_types(
        user,
        context_product_ids,
        category,
        planned_target_product_type=planned_target_product_type,
        planned_target_step_index=planned_target_step_index,
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
    refresh_caller = (
        "update_roadmap_from_purchase"
        if (post_ctx and ((post_ctx.get("product_ids") or []) or (post_ctx.get("categories") or [])))
        else "refresh_roadmap"
    )

    planner_mode = planner_runtime_mode()
    planner_result: dict[str, Any] | None = None
    planner_served = False
    if planner_mode in {"shadow", "serve"}:
        rules = CATEGORY_RULES[category]
        planner_result = generate_planner_chain(
            user=user,
            category=category,
            candidate_types=_planner_candidate_universe(
                category=category,
                purchased_types=purchased_types,
                owned_types_ordered=owned_types_ordered,
            ),
            purchased_types=purchased_types,
            owned_types_ordered=owned_types_ordered,
            min_steps=int(rules["min_steps"]),
            max_steps=int(rules["max_steps"]),
            refresh_caller=refresh_caller,
        )

    if planner_mode == "serve" and planner_result and planner_result.get("decision") == "model_used":
        chain = list(planner_result.get("chain") or [])
        source_by_type = dict(planner_result.get("source_by_type") or {})
        ml_predictions = []
        ml_runtime = {
            "decision": "disabled",
            "fallback_reason": None,
            "disabled_reason": "planner_enabled",
            "guard": None,
            "category_guard": None,
            **_default_rollout_meta(),
        }
        planner_served = True
    else:
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
        source_key = str(source_meta.get("source") or "")
        source_is_ml = source_key == "ml_next_step"
        source_is_planner = source_key == "ml_planner"
        source_is_planner_fallback = source_key == "planner_fallback"
        source_is_state_prefix = source_key == "state_prefix"
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
        if source_is_planner:
            why.append("picked via ML planner")
        elif source_is_planner_fallback:
            why.append("picked via planner fallback")
        elif source_is_state_prefix:
            why.append("picked via user state")
        else:
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
    served_model_path = str(ml_runtime.get("model_path") or model_path or "")
    served_model_version = str(ml_runtime.get("model_version") or "")
    served_feature_set = str(ml_runtime.get("selected_feature_set") or "")
    served_model_slot = str(ml_runtime.get("model_slot") or "active")
    nextstep_artifact = nextstep_model_artifact_summary(served_model_path)
    if not served_model_version:
        served_model_version = str((nextstep_artifact or {}).get("model_version") or "")
    if not served_feature_set:
        served_feature_set = str((nextstep_artifact or {}).get("selected_feature_set") or "")
    shadow_model_path = str(getattr(settings, "ROADMAP_NEXTSTEP_V4_SHADOW_MODEL_PATH", "") or "").strip()
    shadow_enabled = bool(use_v4 and shadow_model_path and shadow_model_path != served_model_path)
    shadow_reason = "disabled"
    shadow_predictions: list[dict[str, Any]] = []
    shadow_artifact = None
    shadow_planned_target_product_type = str(ml_runtime.get("planned_target_product_type") or "")
    shadow_planned_target_step_index = int(ml_runtime.get("planned_target_step_index") or 0)
    if shadow_enabled:
        shadow_reason = "ok"
        shadow_artifact = nextstep_model_artifact_summary(shadow_model_path)
        shadow_predictions = predict_next_product_types_for_model_path(
            shadow_model_path,
            user=user,
            context_product_ids=context_product_ids,
            category=category,
            planned_target_product_type=shadow_planned_target_product_type,
            planned_target_step_index=shadow_planned_target_step_index,
            candidate_types=chain,
        )
        if not shadow_predictions:
            shadow_reason = "no_predictions_or_model_unavailable"
    elif not use_v4:
        shadow_reason = "shadow_supported_for_v4_only"
    elif not shadow_model_path:
        shadow_reason = "shadow_not_configured"
    else:
        shadow_reason = "shadow_same_as_active"

    planner_meta = {
        "mode": planner_mode,
        "served": bool(planner_served),
        "decision": str((planner_result or {}).get("decision") or "disabled"),
        "fallback_reason": (planner_result or {}).get("fallback_reason"),
        "disabled_reason": (planner_result or {}).get("disabled_reason"),
        "model_path": str((planner_result or {}).get("model_path") or ""),
        "model_version": str((planner_result or {}).get("model_version") or ""),
        "selected_feature_set": str((planner_result or {}).get("selected_feature_set") or ""),
        "chain": list((planner_result or {}).get("chain") or []),
        "trace": list((planner_result or {}).get("trace") or [])[:10],
    }
    meta = {
        "generation_state": "ok",
        "source": "roadmap_planner_v1" if planner_served else "roadmap_v1",
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
            "rollout_mode": str(ml_runtime.get("rollout_mode") or "none"),
            "rollout_selected": bool(ml_runtime.get("rollout_selected")),
            "rollout_reason": str(ml_runtime.get("rollout_reason") or "not_applicable"),
            "rollout_bucket": ml_runtime.get("rollout_bucket"),
            "rollout_percent": ml_runtime.get("rollout_percent"),
            "partial_match_product_type": ml_runtime.get("partial_match_product_type"),
            "partial_match_step_index": ml_runtime.get("partial_match_step_index"),
            "mode": ml_mode,
            "model_path": served_model_path,
            "model_version": served_model_version,
            "selected_feature_set": served_feature_set,
            "model_slot": served_model_slot,
            "planned_target_product_type": shadow_planned_target_product_type,
            "planned_target_step_index": shadow_planned_target_step_index,
            "shadow": {
                "enabled": bool(shadow_enabled),
                "reason": shadow_reason,
                "model_path": shadow_model_path,
                "model_version": str((shadow_artifact or {}).get("model_version") or ""),
                "selected_feature_set": str((shadow_artifact or {}).get("selected_feature_set") or ""),
                "planned_target_product_type": shadow_planned_target_product_type,
                "planned_target_step_index": shadow_planned_target_step_index,
                "predictions": shadow_predictions[:10],
            },
        },
        "planner": planner_meta,
        "context": {
            "refresh_caller": refresh_caller,
            "post_ctx_categories": (post_ctx or {}).get("categories") or [],
            "post_ctx_product_ids": (post_ctx or {}).get("product_ids") or [],
            "context_product_ids_count": len(context_product_ids),
            "owned_product_types_count": len(owned_types_set),
            "purchased_types_count": len(purchased_types_set),
        },
    }

    plan = _upsert_plan_with_steps(user=user, category=category, meta=meta, step_payloads=step_payloads)
    emit_plan_refresh_events(user=user, plan=plan, request_id=None)
    return plan


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


def resolve_primary_roadmap_category(user) -> str:
    active_plan = get_active_plan(user, category="")
    if active_plan and str(active_plan.category or "").strip() in CATEGORY_RULES:
        return str(active_plan.category)

    favorite_category = str(
        (favorite_category_snapshot(user) or {}).get("favorite_category") or ""
    ).strip()
    if favorite_category in CATEGORY_RULES:
        return favorite_category

    return RoadmapPlan.Category.SKINCARE


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
        roadmap_ctx["step_id"] = int(next_step.id)
        roadmap_ctx["step_index"] = int(next_step.step_index)
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
    purchased_products = _post_ctx_products(post_ctx)
    purchased_products_by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for product in purchased_products:
        category_key = str(product.get("category") or "").strip().lower()
        if category_key:
            purchased_products_by_category[category_key].append(product)

    out: list[dict[str, Any]] = []
    for category in categories:
        plan = get_active_plan(user, category=category)
        if not plan:
            continue
        step = get_next_missing_step(plan)
        if not step:
            continue

        matched_by = None
        match_meta: dict[str, Any] = {}
        if step.recommended_product_id and int(step.recommended_product_id) in purchased_ids:
            matched_by = "recommended_product_id"
            match_meta = {
                "recommended_product_id": int(step.recommended_product_id),
                "purchased_product_id": int(step.recommended_product_id),
                "purchased_product_type": str(step.product_type or ""),
            }
        elif category == "fragrance":
            if step.product_type in purchased_fragrance_slots:
                matched_by = "fragrance_slot"
                match_meta = {"purchased_slot": str(step.product_type or "")}
        else:
            if step.product_type in set(purchased_by_category.get(category, [])):
                semantic_meta = _semantic_completion_match_meta(
                    user=user,
                    step=step,
                    purchased_products=purchased_products_by_category.get(category, []),
                )
                if semantic_meta:
                    matched_by = "semantic_content_match"
                    match_meta = semantic_meta
                else:
                    matched_by = "product_type"
                    purchased_product = next(
                        (
                            product
                            for product in purchased_products_by_category.get(category, [])
                            if str(product.get("product_type") or "").strip().lower()
                            == str(step.product_type or "").strip().lower()
                        ),
                        None,
                    )
                    if purchased_product:
                        match_meta = {
                            "purchased_product_id": int(purchased_product["id"]),
                            "purchased_product_type": str(purchased_product.get("product_type") or ""),
                        }

        if matched_by:
            out.append(
                {
                    "category": category,
                    "plan": plan,
                    "step": step,
                    "matched_by": matched_by,
                    "match_meta": match_meta,
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
