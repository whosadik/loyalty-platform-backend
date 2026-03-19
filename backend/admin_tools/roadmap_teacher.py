from __future__ import annotations

import hashlib
import json
from bisect import bisect_left
from collections import Counter, defaultdict
from datetime import timedelta, timezone as dt_timezone
from typing import Any

from django.utils import timezone

from catalog.models import Product
from roadmap_app.content_features import (
    SCALP_OBJECTIVE_TOKENS,
    build_candidate_catalog_summaries,
    effective_nextstep_rules_chain,
    normalize_tokens,
    product_signature,
    profile_signature,
    slug_token,
)
from roadmap_app.fragrance_slots import SLOTS, slot_of_fragrance
from roadmap_app.services import CATEGORY_RULES, FRAGRANCE_DEFAULT_CHAIN, ROADMAP_OWNED_FRESHNESS_DEFAULTS
from transactions.models import TransactionItem
from users_app.models import CustomerProfile

TARGET_CATEGORIES = {"skincare", "haircare", "makeup", "fragrance"}
STOP_TOKEN = "__stop__"
CANDIDATE_SPACE_BY_CATEGORY: dict[str, list[str]] = {
    "haircare": ["shampoo", "conditioner", "hair_mask", "hair_oil", "scalp_serum", "leave_in"],
    "skincare": ["cleanser", "serum", "moisturizer", "spf", "toner", "mask", "eye_cream", "essence"],
    "makeup": ["foundation", "mascara", "blush", "lipstick", "eyeshadow", "primer", "setting_spray"],
    "fragrance": list(SLOTS),
}


def _safe_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _safe_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _to_int_or_none(value: Any) -> int | None:
    try:
        if value is None:
            return None
        return int(value)
    except Exception:
        return None


def _unique(values: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for raw in values:
        token = str(raw or "").strip().lower()
        if not token or token in seen:
            continue
        seen.add(token)
        out.append(token)
    return out


def _position_key(ts, tx_id: int, item_id: int) -> tuple[Any, int, int]:
    return (ts.astimezone(dt_timezone.utc), int(tx_id), int(item_id))


def _attr_token(product: dict[str, Any], *keys: str) -> str:
    attrs = _safe_dict(product.get("attrs"))
    raw_meta = _safe_dict(product.get("raw_meta"))
    for key in keys:
        if key in attrs:
            token = slug_token(attrs.get(key), default="")
            if token:
                return token
        if key in raw_meta:
            token = slug_token(raw_meta.get(key), default="")
            if token:
                return token
    return "__none__"


def _attr_tokens(product: dict[str, Any], *keys: str) -> list[str]:
    attrs = _safe_dict(product.get("attrs"))
    raw_meta = _safe_dict(product.get("raw_meta"))
    for key in keys:
        if key in attrs:
            values = normalize_tokens(attrs.get(key))
            if values:
                return values
        if key in raw_meta:
            values = normalize_tokens(raw_meta.get(key))
            if values:
                return values
    return []


def _candidate_types(category: str) -> list[str]:
    values = list(CANDIDATE_SPACE_BY_CATEGORY.get(str(category or "").strip().lower()) or [])
    if STOP_TOKEN not in values:
        values.append(STOP_TOKEN)
    return values


def _max_target_steps(category: str) -> int:
    if category == "fragrance":
        return 4
    return int(CATEGORY_RULES[category]["max_steps"])


def _estimate_current_owned_tokens_before_anchor(*, category: str, prior_items: list[dict[str, Any]], anchor_ts) -> list[str]:
    token_last_seen: dict[str, Any] = {}
    for item in prior_items:
        if str(item.get("category") or "") != category:
            continue
        token = str(item.get("history_token") or "").strip().lower()
        if not token:
            continue
        token_last_seen[token] = item["ts"]

    out: list[str] = []
    freshness_lookup = ROADMAP_OWNED_FRESHNESS_DEFAULTS.get(category) or {}
    for token, seen_at in token_last_seen.items():
        freshness_days = None
        if category == "fragrance":
            freshness_days = 365
        else:
            freshness_days = _to_int_or_none(freshness_lookup.get(token))
        if freshness_days is None:
            out.append(token)
            continue
        if int((anchor_ts - seen_at).days) <= int(freshness_days):
            out.append(token)
    return _unique(out)


def _seed_action_token(category: str, product: dict[str, Any]) -> str:
    category_norm = str(category or "").strip().lower()
    if category_norm == "fragrance":
        slot = slot_of_fragrance(_safe_dict(product.get("attrs")), raw_meta=_safe_dict(product.get("raw_meta")))
        if slot in SLOTS:
            return str(slot)
        return "__none__"
    return str(product.get("product_type") or "").strip().lower() or "__none__"


def _teacher_haircare_sequence(*, seed_sig: dict[str, Any], profile_sig: dict[str, Any]) -> dict[str, Any]:
    base = ["shampoo", "conditioner", "hair_mask", "hair_oil"]
    trace: list[str] = ["base_core_chain"]
    optional: list[str] = []
    profile_tokens = set(normalize_tokens(profile_sig.get("goals"))) | set(normalize_tokens(profile_sig.get("hair_concerns")))
    anchor_tokens = set(normalize_tokens(seed_sig.get("concerns"))) | set(normalize_tokens(seed_sig.get("actives")))
    hair_type = str(profile_sig.get("hair_type") or seed_sig.get("hair_type") or "__none__")
    scalp_type = str(profile_sig.get("scalp_type") or seed_sig.get("scalp_type") or "__none__")
    has_scalp_focus = scalp_type in {"oily", "sensitive"} or bool(profile_tokens & SCALP_OBJECTIVE_TOKENS) or bool(anchor_tokens & SCALP_OBJECTIVE_TOKENS)
    if str(seed_sig.get("product_type") or "") == "scalp_serum":
        has_scalp_focus = True
    if has_scalp_focus:
        optional.append("scalp_serum")
        trace.append("optional_scalp_serum")

    leave_in_tokens = {"frizz", "curl_definition", "definition", "heat_protection", "damage", "dryness", "smoothing"}
    has_leave_in_need = hair_type in {"wavy", "curly", "coily"} or bool(profile_tokens & leave_in_tokens) or bool(anchor_tokens & leave_in_tokens)
    if str(seed_sig.get("product_type") or "") == "leave_in":
        has_leave_in_need = True
    if has_leave_in_need:
        optional.append("leave_in")
        trace.append("optional_leave_in")

    chain = _unique(base + optional)
    chain = effective_nextstep_rules_chain(
        category="haircare",
        rules_chain=chain,
        planned_target_product_type="scalp_serum" if "scalp_serum" in chain else None,
        profile_sig=profile_sig,
        anchor_product_type=str(seed_sig.get("product_type") or ""),
    )
    if "leave_in" in chain:
        chain = [token for token in chain if token != "leave_in"]
        insert_at = chain.index("hair_mask") + 1 if "hair_mask" in chain else min(len(chain), 2)
        chain.insert(insert_at, "leave_in")
    return {"sequence": chain[:6], "trace": trace}


def _teacher_skincare_sequence(*, seed_sig: dict[str, Any], profile_sig: dict[str, Any]) -> dict[str, Any]:
    trace: list[str] = ["base_core_chain"]
    goals = set(normalize_tokens(profile_sig.get("goals")))
    avoid = set(normalize_tokens(profile_sig.get("avoid_flags")))
    anchor_concerns = set(normalize_tokens(seed_sig.get("concerns")))
    skin_type = str(profile_sig.get("skin_type") or "__none__")
    include = {"cleanser", "serum", "moisturizer", "spf"}
    if skin_type in {"oily", "combination"} or bool(anchor_concerns & {"oiliness", "pores", "texture", "acne"}):
        include.add("toner")
        trace.append("optional_toner")
    if skin_type in {"dry", "sensitive"} or len(goals) >= 2 or int(len(seed_sig.get("actives") or [])) >= 2:
        include.add("essence")
        trace.append("optional_essence")
    if bool(goals & {"dark_circles", "wrinkles", "dehydration", "firming"}) or "eye" in set(seed_sig.get("concerns") or []):
        include.add("eye_cream")
        trace.append("optional_eye_cream")
    if bool(goals & {"hydration", "repair", "brightening", "calming", "acne"}) or bool(anchor_concerns & {"hydration", "repair", "brightening", "acne"}):
        include.add("mask")
        trace.append("optional_mask")
    if "spf" in avoid:
        include.discard("spf")
        trace.append("avoid_spf_removed")
    ordered = ["cleanser", "toner", "essence", "serum", "eye_cream", "moisturizer", "spf", "mask"]
    chain = [token for token in ordered if token in include]
    return {"sequence": chain[:8], "trace": trace}


def _teacher_makeup_sequence(*, seed_sig: dict[str, Any], profile_sig: dict[str, Any]) -> dict[str, Any]:
    trace: list[str] = ["base_core_chain"]
    include = {"foundation", "mascara", "blush"}
    anchor_type = str(seed_sig.get("product_type") or "")
    finish_pref = set(normalize_tokens(profile_sig.get("makeup_finish_pref")))
    coverage_pref = set(normalize_tokens(profile_sig.get("makeup_coverage_pref")))
    makeup_concerns = set(normalize_tokens(profile_sig.get("makeup_concerns"))) | set(normalize_tokens(seed_sig.get("concerns")))
    if anchor_type in {"primer", "foundation"} or bool(finish_pref) or bool(coverage_pref) or bool(makeup_concerns & {"long_wear", "oil_control", "pores"}):
        include.add("primer")
        trace.append("optional_primer")
    if anchor_type in {"mascara", "eyeshadow"} or bool(makeup_concerns & {"smoky", "eye_definition", "sensitive_eyes"}):
        include.add("eyeshadow")
        trace.append("optional_eyeshadow")
    if anchor_type == "lipstick" or str(profile_sig.get("makeup_tone_family") or "__none__") != "__none__":
        include.add("lipstick")
        trace.append("optional_lipstick")
    if anchor_type == "setting_spray" or bool(makeup_concerns & {"long_wear", "waterproof", "humidity"}):
        include.add("setting_spray")
        trace.append("optional_setting_spray")
    ordered = ["primer", "foundation", "blush", "mascara", "eyeshadow", "lipstick", "setting_spray"]
    chain = [token for token in ordered if token in include]
    return {"sequence": chain[:7], "trace": trace}


def _fragrance_order_for_seed(seed_slot: str) -> list[str]:
    seed_slot = str(seed_slot or "").strip().lower()
    if seed_slot == "warm_day":
        return ["warm_day", "warm_evening", "cold_day", "cold_evening"]
    if seed_slot == "warm_evening":
        return ["warm_evening", "warm_day", "cold_evening", "cold_day"]
    if seed_slot == "cold_day":
        return ["cold_day", "cold_evening", "warm_day", "warm_evening"]
    if seed_slot == "cold_evening":
        return ["cold_evening", "cold_day", "warm_evening", "warm_day"]
    return list(FRAGRANCE_DEFAULT_CHAIN)


def _teacher_fragrance_sequence(*, seed_sig: dict[str, Any], profile_sig: dict[str, Any], seed_token: str) -> dict[str, Any]:
    trace: list[str] = ["fragrance_slot_chain"]
    base_chain = _fragrance_order_for_seed(seed_token)
    richness = int(len(profile_sig.get("fragrance_liked_families") or [])) + int(len(profile_sig.get("fragrance_liked_notes") or []))
    if str(profile_sig.get("fragrance_intensity_pref") or "__none__") != "__none__":
        richness += 1
    richness += int(len(seed_sig.get("notes") or []) >= 2)
    if richness >= 3:
        target_len = 4
        trace.append("length_full_slot_coverage")
    elif richness >= 1:
        target_len = 3
        trace.append("length_three_slots")
    else:
        target_len = 2
        trace.append("length_two_slots")
    return {"sequence": base_chain[:target_len], "trace": trace}


def build_teacher_policy(
    *,
    category: str,
    seed_product: dict[str, Any],
    profile_sig: dict[str, Any],
    prior_state: dict[str, Any],
) -> dict[str, Any]:
    category_norm = str(category or "").strip().lower()
    seed_sig = product_signature(seed_product)
    seed_token = _seed_action_token(category_norm, seed_product)
    if category_norm == "haircare":
        result = _teacher_haircare_sequence(seed_sig=seed_sig, profile_sig=profile_sig)
    elif category_norm == "skincare":
        result = _teacher_skincare_sequence(seed_sig=seed_sig, profile_sig=profile_sig)
    elif category_norm == "makeup":
        result = _teacher_makeup_sequence(seed_sig=seed_sig, profile_sig=profile_sig)
    elif category_norm == "fragrance":
        result = _teacher_fragrance_sequence(seed_sig=seed_sig, profile_sig=profile_sig, seed_token=seed_token)
    else:
        raise ValueError(f"Unsupported teacher category: {category_norm}")

    sequence = _unique(list(result.get("sequence") or []))
    if seed_token in CANDIDATE_SPACE_BY_CATEGORY.get(category_norm, []) and seed_token not in sequence and category_norm == "fragrance":
        sequence = _unique([seed_token] + sequence)
    max_steps = _max_target_steps(category_norm)
    sequence = [token for token in sequence if token in set(CANDIDATE_SPACE_BY_CATEGORY.get(category_norm) or [])][:max_steps]
    return {
        "policy_version": "teacher_policy_v1",
        "target_source": "teacher_policy_v1",
        "sequence": sequence,
        "target_length": int(len(sequence)),
        "seed_action_token": seed_token,
        "seed_in_target": int(seed_token in sequence),
        "seed_target_position": int(sequence.index(seed_token) + 1) if seed_token in sequence else 0,
        "trace": list(result.get("trace") or []),
        "prior_state": dict(prior_state or {}),
    }


def load_teacher_source_data(*, days: int, include_ga: bool) -> dict[str, Any]:
    now_utc = timezone.now().astimezone(dt_timezone.utc)
    since = now_utc - timedelta(days=days)
    anchor_qs = (
        TransactionItem.objects.filter(
            transaction__created_at__gte=since,
            transaction__created_at__lte=now_utc,
            product__category__in=sorted(TARGET_CATEGORIES),
        )
        .order_by("transaction__user_id", "product__category", "transaction__created_at", "transaction__id", "id")
        .values(
            "id",
            "transaction__user_id",
            "transaction__id",
            "transaction__created_at",
            "product_id",
            "product__category",
            "product__product_type",
            "product__brand",
            "product__price",
            "product__concerns",
            "product__actives",
            "product__flags",
            "product__supported_skin_types",
            "product__attrs",
            "product__ingredients_inci",
            "product__raw_meta",
            "transaction__user__username",
        )
    )
    anchor_rows: list[dict[str, Any]] = []
    excluded_counts: Counter[str] = Counter()
    seen_anchor_keys: set[tuple[int, str]] = set()
    for row in anchor_qs.iterator(chunk_size=5000):
        username = str(row.get("transaction__user__username") or "")
        if not include_ga and username.startswith("ga_"):
            excluded_counts["ga_user"] += 1
            continue
        user_id = int(row["transaction__user_id"])
        category = str(row["product__category"] or "").strip().lower()
        if category not in TARGET_CATEGORIES:
            excluded_counts["unsupported_category"] += 1
            continue
        key = (user_id, category)
        if key in seen_anchor_keys:
            excluded_counts["duplicate_later_purchase"] += 1
            continue
        seen_anchor_keys.add(key)
        anchor_rows.append(
            {
                "planning_id": int(len(anchor_rows) + 1),
                "user_id": user_id,
                "category": category,
                "anchor_ts": row["transaction__created_at"].astimezone(dt_timezone.utc),
                "anchor_tx_id": int(row["transaction__id"]),
                "anchor_item_id": int(row["id"]),
                "seed_product": {
                    "id": int(row["product_id"]),
                    "category": category,
                    "product_type": str(row["product__product_type"] or "").strip().lower(),
                    "brand": str(row.get("product__brand") or "").strip().lower(),
                    "price": float(row.get("product__price") or 0.0),
                    "concerns": row.get("product__concerns") if isinstance(row.get("product__concerns"), list) else [],
                    "actives": row.get("product__actives") if isinstance(row.get("product__actives"), list) else [],
                    "flags": row.get("product__flags") if isinstance(row.get("product__flags"), list) else [],
                    "supported_skin_types": row.get("product__supported_skin_types")
                    if isinstance(row.get("product__supported_skin_types"), list)
                    else [],
                    "attrs": row.get("product__attrs") if isinstance(row.get("product__attrs"), dict) else {},
                    "ingredients_inci": str(row.get("product__ingredients_inci") or ""),
                    "raw_meta": row.get("product__raw_meta") if isinstance(row.get("product__raw_meta"), dict) else {},
                },
            }
        )

    users = sorted({int(row["user_id"]) for row in anchor_rows})
    profile_map = {
        int(profile.user_id): profile_signature(profile)
        for profile in CustomerProfile.objects.filter(user_id__in=users)
    }

    tx_qs = TransactionItem.objects.filter(transaction__user_id__in=users).values(
        "id",
        "transaction__user_id",
        "transaction__id",
        "transaction__created_at",
        "transaction__total_amount",
        "product_id",
        "quantity",
        "product__category",
        "product__product_type",
        "product__brand",
        "product__price",
        "product__concerns",
        "product__actives",
        "product__flags",
        "product__supported_skin_types",
        "product__attrs",
        "product__ingredients_inci",
        "product__raw_meta",
    )

    user_items: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in tx_qs.iterator(chunk_size=5000):
        user_id = int(row["transaction__user_id"])
        category = str(row.get("product__category") or "").strip().lower()
        product = {
            "id": int(row["product_id"]),
            "category": category,
            "product_type": str(row.get("product__product_type") or "").strip().lower(),
            "brand": str(row.get("product__brand") or "").strip().lower(),
            "price": float(row.get("product__price") or 0.0),
            "concerns": row.get("product__concerns") if isinstance(row.get("product__concerns"), list) else [],
            "actives": row.get("product__actives") if isinstance(row.get("product__actives"), list) else [],
            "flags": row.get("product__flags") if isinstance(row.get("product__flags"), list) else [],
            "supported_skin_types": row.get("product__supported_skin_types")
            if isinstance(row.get("product__supported_skin_types"), list)
            else [],
            "attrs": row.get("product__attrs") if isinstance(row.get("product__attrs"), dict) else {},
            "ingredients_inci": str(row.get("product__ingredients_inci") or ""),
            "raw_meta": row.get("product__raw_meta") if isinstance(row.get("product__raw_meta"), dict) else {},
        }
        ts = row["transaction__created_at"].astimezone(dt_timezone.utc)
        user_items[user_id].append(
            {
                "position": _position_key(ts, int(row["transaction__id"]), int(row["id"])),
                "ts": ts,
                "tx_id": int(row["transaction__id"]),
                "item_id": int(row["id"]),
                "quantity": max(1, int(row.get("quantity") or 0)),
                "tx_total": float(row.get("transaction__total_amount") or 0.0),
                "category": category,
                "brand": str(product.get("brand") or ""),
                "product": product,
                "history_token": _seed_action_token(category, product),
            }
        )

    for rows in user_items.values():
        rows.sort(key=lambda row: row["position"])

    catalog_rows: list[dict[str, Any]] = []
    for row in Product.objects.filter(category__in=sorted(TARGET_CATEGORIES)).values(
        "category",
        "product_type",
        "concerns",
        "actives",
        "flags",
        "supported_skin_types",
        "attrs",
        "ingredients_inci",
        "raw_meta",
    ):
        product_row = dict(row)
        category = str(product_row.get("category") or "").strip().lower()
        if category == "fragrance":
            slot = slot_of_fragrance(_safe_dict(product_row.get("attrs")), raw_meta=_safe_dict(product_row.get("raw_meta")))
            if slot not in SLOTS:
                continue
            product_row["product_type"] = str(slot)
        catalog_rows.append(product_row)

    return {
        "now_utc": now_utc,
        "since": since,
        "anchor_rows": anchor_rows,
        "profile_map": profile_map,
        "user_items": user_items,
        "candidate_catalog_summaries": build_candidate_catalog_summaries(catalog_rows),
        "excluded_counts": dict(sorted(excluded_counts.items())),
    }


def _stable_user_split(user_id: int, *, seed: int) -> str:
    digest = hashlib.sha1(f"{seed}:{int(user_id)}".encode("utf-8")).hexdigest()[:8]
    bucket = int(digest, 16) % 100
    if bucket < 70:
        return "train"
    if bucket < 85:
        return "val"
    return "test"


def build_teacher_examples(source_data: dict[str, Any], *, seed: int) -> dict[str, Any]:
    anchor_rows = list(source_data.get("anchor_rows") or [])
    profile_map = dict(source_data.get("profile_map") or {})
    user_items = dict(source_data.get("user_items") or {})
    examples: list[dict[str, Any]] = []
    edge_counts: Counter[str] = Counter()

    for anchor in anchor_rows:
        user_id = int(anchor["user_id"])
        category = str(anchor["category"])
        seed_product = dict(anchor.get("seed_product") or {})
        seed_sig = product_signature(seed_product)
        profile_sig = dict(profile_map.get(user_id) or {})
        rows = user_items.get(user_id) or []
        positions = [row["position"] for row in rows]
        anchor_position = _position_key(anchor["anchor_ts"], int(anchor["anchor_tx_id"]), int(anchor["anchor_item_id"]))
        pivot = bisect_left(positions, anchor_position)
        prior_items = rows[:pivot]
        prior_category_items = [row for row in prior_items if str(row.get("category") or "") == category]
        prior_category_tokens = [str(row.get("history_token") or "") for row in prior_category_items if str(row.get("history_token") or "")]
        prior_current_owned_tokens = _estimate_current_owned_tokens_before_anchor(
            category=category,
            prior_items=prior_category_items,
            anchor_ts=anchor["anchor_ts"],
        )
        prior_brand_counter = Counter(str(row.get("brand") or "") for row in prior_items if str(row.get("brand") or ""))
        prior_category_brand_counter = Counter(
            str(row.get("brand") or "") for row in prior_category_items if str(row.get("brand") or "")
        )
        prior_category_prices = [float((row.get("product") or {}).get("price") or 0.0) for row in prior_category_items if float((row.get("product") or {}).get("price") or 0.0) > 0]
        prior_days_since_category_purchase = -1
        if prior_category_items:
            prior_days_since_category_purchase = int((anchor["anchor_ts"].date() - prior_category_items[-1]["ts"].date()).days)

        prior_state = {
            "prior_category_purchase_total": int(sum(row.get("quantity") or 0 for row in prior_category_items)),
            "prior_category_distinct_token_count": int(len(set(prior_category_tokens))),
            "prior_current_owned_token_count": int(len(prior_current_owned_tokens)),
            "prior_current_owned_tokens": list(prior_current_owned_tokens),
            "prior_days_since_category_purchase": int(prior_days_since_category_purchase),
            "favorite_brand_overall": prior_brand_counter.most_common(1)[0][0] if prior_brand_counter else "__none__",
            "favorite_brand_in_category": prior_category_brand_counter.most_common(1)[0][0] if prior_category_brand_counter else "__none__",
            "avg_price_in_category_before_anchor": round(float(sum(prior_category_prices) / max(1, len(prior_category_prices))), 4) if prior_category_prices else 0.0,
            "prior_total_purchases_all": int(sum(row.get("quantity") or 0 for row in prior_items)),
            "prior_distinct_categories_count": int(len({str(row.get('category') or '') for row in prior_items if str(row.get('category') or '')})),
        }
        teacher = build_teacher_policy(
            category=category,
            seed_product=seed_product,
            profile_sig=profile_sig,
            prior_state=prior_state,
        )
        sequence = list(teacher.get("sequence") or [])
        if not sequence:
            edge_counts["empty_teacher_sequence"] += 1
            continue
        split = _stable_user_split(user_id, seed=seed)
        examples.append(
            {
                "planning_id": int(anchor["planning_id"]),
                "user_id": user_id,
                "split": split,
                "category": category,
                "planning_t0_utc": anchor["anchor_ts"].isoformat().replace("+00:00", "Z"),
                "seed_transaction_id": int(anchor["anchor_tx_id"]),
                "seed_product_id": int(seed_product["id"]),
                "seed_product_type": str(seed_product.get("product_type") or "__none__"),
                "seed_action_token": str(teacher.get("seed_action_token") or "__none__"),
                "seed_slot": str(teacher.get("seed_action_token") or "__none__") if category == "fragrance" else "__none__",
                "seed_brand": str(seed_product.get("brand") or "__none__"),
                "seed_price": round(float(seed_product.get("price") or 0.0), 4),
                "seed_scent_family": str(seed_sig.get("scent_family") or "__none__"),
                "seed_intensity": str(seed_sig.get("intensity") or "__none__"),
                "seed_notes_count": int(len(seed_sig.get("notes") or [])),
                "seed_hair_type": str(seed_sig.get("hair_type") or "__none__"),
                "seed_scalp_type": str(seed_sig.get("scalp_type") or "__none__"),
                "seed_hair_thickness": str(seed_sig.get("hair_thickness") or "__none__"),
                "seed_finish": str(seed_sig.get("finish") or "__none__"),
                "seed_coverage": str(seed_sig.get("coverage") or "__none__"),
                "seed_undertone": str(seed_sig.get("undertone") or "__none__"),
                "seed_tone_family": str(seed_sig.get("tone_family") or "__none__"),
                "seed_area": _attr_token(seed_product, "area"),
                "seed_spf_signal": _attr_token(seed_product, "spf", "spf_level"),
                "seed_effect": _attr_token(seed_product, "effect"),
                "seed_waterproof": _attr_token(seed_product, "waterproof"),
                "seed_concerns_count": int(len(seed_sig.get("concerns") or [])),
                "seed_actives_count": int(len(seed_sig.get("actives") or [])),
                "seed_supported_skin_types_count": int(len(seed_sig.get("supported_skin_types") or [])),
                "seed_inci_token_count": int(len(seed_sig.get("inci_tokens") or [])),
                "target_sequence_json": json.dumps(sequence, ensure_ascii=False),
                "target_length": int(len(sequence)),
                "teacher_policy_version": str(teacher.get("policy_version") or "teacher_policy_v1"),
                "target_source": str(teacher.get("target_source") or "teacher_policy_v1"),
                "teacher_policy_trace_json": json.dumps(list(teacher.get("trace") or []), ensure_ascii=False),
                "teacher_seed_in_target": int(teacher.get("seed_in_target") or 0),
                "teacher_seed_target_position": int(teacher.get("seed_target_position") or 0),
                "prior_category_purchase_total": int(prior_state["prior_category_purchase_total"]),
                "prior_category_distinct_token_count": int(prior_state["prior_category_distinct_token_count"]),
                "prior_current_owned_token_count": int(prior_state["prior_current_owned_token_count"]),
                "prior_current_owned_tokens_json": json.dumps(list(prior_state["prior_current_owned_tokens"]), ensure_ascii=False),
                "prior_days_since_category_purchase": int(prior_state["prior_days_since_category_purchase"]),
                "favorite_brand_overall_before_anchor": str(prior_state["favorite_brand_overall"] or "__none__"),
                "favorite_brand_in_category_before_anchor": str(prior_state["favorite_brand_in_category"] or "__none__"),
                "avg_price_in_category_before_anchor": round(float(prior_state["avg_price_in_category_before_anchor"]), 4),
                "prior_total_purchases_all": int(prior_state["prior_total_purchases_all"]),
                "prior_distinct_categories_count": int(prior_state["prior_distinct_categories_count"]),
                "profile_signature": dict(profile_sig),
                "seed_product": dict(seed_product),
                "seed_signature": dict(seed_sig),
            }
        )

    split_counts = Counter(str(row.get("split") or "train") for row in examples)
    split_users: dict[str, set[int]] = defaultdict(set)
    for row in examples:
        split_users[str(row["split"])].add(int(row["user_id"]))
    return {
        "examples": examples,
        "edge_counts": dict(sorted(edge_counts.items())),
        "split_counts": dict(sorted(split_counts.items())),
        "split_user_overlap_counts": {
            "train_val": int(len(split_users["train"].intersection(split_users["val"]))),
            "train_test": int(len(split_users["train"].intersection(split_users["test"]))),
            "val_test": int(len(split_users["val"].intersection(split_users["test"]))),
        },
    }
