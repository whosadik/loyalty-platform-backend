from __future__ import annotations

from collections import defaultdict
from datetime import timedelta
from typing import Any

from django.conf import settings
from django.core.cache import cache
from django.utils import timezone

from ml_logic.routine_rules import CONFLICT_PAIRS
from roadmap_app.content_features import (
    SCALP_ACTIVE_TOKENS,
    SCALP_OBJECTIVE_TOKENS,
    product_signature,
    profile_signature,
)


_CONFLICT_LOOKUP = {tuple(sorted(pair)) for pair in CONFLICT_PAIRS}

_LIGHTWEIGHT_GOALS = {"volume", "lightweight_care", "flatness"}
_CURL_GOALS = {"definition", "frizz", "frizz_control", "dryness", "hydration"}
_REPAIR_GOALS = {"repair", "damage", "split_ends", "smoothness", "shine", "detangling"}

_LIGHT_FINISHES = {"airy", "fresh"}
_DEFINED_FINISHES = {"defined", "soft", "sleek"}
_RICH_FINISHES = {"rich", "glossy"}

_LIGHTWEIGHT_ACTIVES = {"rice_protein", "panthenol", "aloe", "amino_acids"}
_CURL_LEAVEIN_ACTIVES = {
    "glycerin",
    "aloe",
    "linseed",
    "panthenol",
    "ceramide",
    "ceramide_np",
    "amino_acids",
}
_RICH_OIL_ACTIVES = {"argan_oil", "jojoba_oil", "squalane", "shea_butter", "coconut_oil"}
_SENSITIVE_SCALP_SUPPORT = {"aloe", "panthenol", "niacinamide"}
_OILY_SCALP_SUPPORT = {"salicylic_acid", "zinc_pca", "tea_tree", "niacinamide"}


def _to_int(value: Any) -> int | None:
    try:
        if value is None:
            return None
        return int(value)
    except Exception:
        return None


def _safe_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _safe_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return float(default)
        return float(value)
    except Exception:
        return float(default)


def _event_key(created_at: Any, event_id: int) -> tuple[Any, int]:
    return created_at, int(event_id or 0)


def _overlap(left: list[str] | set[str], right: list[str] | set[str]) -> list[str]:
    right_set = {str(item) for item in right if str(item)}
    return [str(item) for item in left if str(item) in right_set]


def _conflict_count(candidate_actives: list[str], anchor_actives: list[str]) -> int:
    count = 0
    for candidate_active in candidate_actives:
        for anchor_active in anchor_actives:
            if tuple(sorted((str(candidate_active), str(anchor_active)))) in _CONFLICT_LOOKUP:
                count += 1
    return count


def _anchor_signature_for_category(
    *,
    category: str,
    context_products: list[dict[str, Any]] | None,
) -> dict[str, Any]:
    for product in context_products or []:
        sig = product_signature(product)
        if str(sig.get("category") or "") == str(category or ""):
            return sig
    return {}


def _sku_health_penalty_enabled(*, category: str, product_type: str) -> bool:
    if not bool(getattr(settings, "ROADMAP_SKU_HEALTH_PENALTY_ENABLED", False)):
        return False
    enabled_categories = {
        str(item).strip().lower()
        for item in list(getattr(settings, "ROADMAP_SKU_HEALTH_PENALTY_CATEGORIES", []) or [])
        if str(item).strip()
    }
    enabled_product_types = {
        str(item).strip().lower()
        for item in list(getattr(settings, "ROADMAP_SKU_HEALTH_PENALTY_PRODUCT_TYPES", []) or [])
        if str(item).strip()
    }
    category_norm = str(category or "").strip().lower()
    product_type_norm = str(product_type or "").strip().lower()
    if enabled_categories and category_norm not in enabled_categories:
        return False
    if enabled_product_types and product_type_norm not in enabled_product_types:
        return False
    return True


def _recent_recommended_product_health_map(
    *,
    category: str,
    product_type: str,
) -> dict[int, dict[str, Any]]:
    category_norm = str(category or "").strip().lower()
    product_type_norm = str(product_type or "").strip().lower()
    if not _sku_health_penalty_enabled(category=category_norm, product_type=product_type_norm):
        return {}

    window_days = int(getattr(settings, "ROADMAP_SKU_HEALTH_PENALTY_WINDOW_DAYS", 60) or 60)
    include_ga = bool(getattr(settings, "ROADMAP_SKU_HEALTH_PENALTY_INCLUDE_GA", False))
    semantic_weight = _safe_float(
        getattr(settings, "ROADMAP_SKU_HEALTH_PENALTY_SEMANTIC_WEIGHT", 0.25),
        default=0.25,
    )
    cache_ttl = int(getattr(settings, "ROADMAP_SKU_HEALTH_PENALTY_CACHE_TTL_SECONDS", 1800) or 1800)
    cache_key = (
        f"roadmap:sku_health:v1:{category_norm}:{product_type_norm}:"
        f"{window_days}:{int(include_ga)}:{semantic_weight:.4f}"
    )
    cached = cache.get(cache_key)
    if isinstance(cached, dict):
        return cached

    from roadmap_app.models import RoadmapEvent

    now_utc = timezone.now()
    since = now_utc - timedelta(days=window_days)
    event_qs = RoadmapEvent.objects.filter(
        created_at__gte=since,
        created_at__lte=now_utc,
        event_type__in=[
            RoadmapEvent.Type.PLAN_REFRESHED,
            RoadmapEvent.Type.STEP_GENERATED,
            RoadmapEvent.Type.STEP_COMPLETED,
        ],
    )
    if not include_ga:
        event_qs = event_qs.exclude(user__username__startswith="ga_")

    rows = list(
        event_qs.order_by("user_id", "step_id", "created_at", "id").values(
            "id",
            "user_id",
            "plan_id",
            "step_id",
            "event_type",
            "created_at",
            "context",
        )
    )

    plan_refreshes: dict[int, list[dict[str, Any]]] = defaultdict(list)
    generated_by_key: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    generated_by_plan_step: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    completions_by_key: dict[tuple[int, int], list[tuple[Any, int, str, int | None]]] = defaultdict(list)

    for row in rows:
        ctx = _safe_dict(row.get("context"))
        event_type = str(row.get("event_type") or "")
        user_id = _to_int(row.get("user_id"))
        event_id = _to_int(row.get("id")) or 0
        created_at = row.get("created_at")
        if user_id is None or created_at is None:
            continue

        if event_type == RoadmapEvent.Type.PLAN_REFRESHED:
            plan_id = _to_int(row.get("plan_id")) or _to_int(ctx.get("plan_id"))
            if plan_id is None:
                continue
            event_category = str(ctx.get("category") or "").strip().lower() or "__unknown__"
            if event_category != category_norm:
                continue
            plan_refreshes[int(plan_id)].append(
                {
                    "event_id": int(event_id),
                    "plan_id": int(plan_id),
                    "generated_at": created_at,
                    "next_step_id": _to_int(ctx.get("next_step_id")),
                }
            )
            continue

        step_id = _to_int(row.get("step_id")) or _to_int(ctx.get("step_id"))
        if step_id is None:
            continue
        key = (int(user_id), int(step_id))

        if event_type == RoadmapEvent.Type.STEP_GENERATED:
            event_category = str(ctx.get("category") or "").strip().lower() or "__unknown__"
            step_product_type = str(ctx.get("product_type") or "").strip().lower() or "__unknown__"
            if event_category != category_norm or step_product_type != product_type_norm:
                continue
            plan_id = _to_int(row.get("plan_id")) or _to_int(ctx.get("plan_id"))
            recommended_product_id = _to_int(ctx.get("recommended_product_id"))
            if plan_id is None or recommended_product_id is None:
                continue
            generated = {
                "generated_event_id": int(event_id),
                "user_id": int(user_id),
                "plan_id": int(plan_id),
                "step_id": int(step_id),
                "generated_at": created_at,
                "recommended_product_id": int(recommended_product_id),
                "has_recommendation": bool(ctx.get("has_recommendation")) or recommended_product_id is not None,
            }
            generated_by_key[key].append(generated)
            generated_by_plan_step[(int(plan_id), int(step_id))].append(generated)
            continue

        if event_type == RoadmapEvent.Type.STEP_COMPLETED:
            match_meta = _safe_dict(ctx.get("match_meta"))
            completions_by_key[key].append(
                (
                    created_at,
                    int(event_id),
                    str(ctx.get("matched_by") or "").strip().lower(),
                    _to_int(match_meta.get("recommended_product_id")) or _to_int(ctx.get("recommended_product_id")),
                )
            )

    next_step_generated_event_ids: set[int] = set()
    for items in generated_by_plan_step.values():
        items.sort(key=lambda item: _event_key(item.get("generated_at"), int(item.get("generated_event_id") or 0)))
    for refreshes in plan_refreshes.values():
        refreshes.sort(key=lambda item: _event_key(item.get("generated_at"), int(item.get("event_id") or 0)))
        for idx, refresh in enumerate(refreshes):
            next_step_id = _to_int(refresh.get("next_step_id"))
            if next_step_id is None:
                continue
            refresh_key = _event_key(refresh.get("generated_at"), int(refresh.get("event_id") or 0))
            next_refresh_key = (
                _event_key(refreshes[idx + 1].get("generated_at"), int(refreshes[idx + 1].get("event_id") or 0))
                if idx + 1 < len(refreshes)
                else None
            )
            for generated in generated_by_plan_step.get((int(refresh["plan_id"]), int(next_step_id)), []):
                generated_key = _event_key(generated.get("generated_at"), int(generated.get("generated_event_id") or 0))
                if generated_key < refresh_key:
                    continue
                if next_refresh_key is not None and generated_key >= next_refresh_key:
                    break
                next_step_generated_event_ids.add(int(generated.get("generated_event_id") or 0))
                break

    stats_by_product: dict[int, dict[str, Any]] = {}
    for key, items in generated_by_key.items():
        items.sort(key=lambda item: _event_key(item["generated_at"], int(item.get("generated_event_id") or 0)))
        for idx, item in enumerate(items):
            generated_event_id = int(item.get("generated_event_id") or 0)
            if generated_event_id not in next_step_generated_event_ids:
                continue
            if not bool(item.get("has_recommendation")) or _to_int(item.get("recommended_product_id")) is None:
                continue

            start_key = _event_key(item["generated_at"], generated_event_id)
            next_generated_key = (
                _event_key(items[idx + 1]["generated_at"], int(items[idx + 1].get("generated_event_id") or 0))
                if idx + 1 < len(items)
                else None
            )
            completions = [
                completion
                for completion in completions_by_key.get(key, [])
                if _event_key(completion[0], completion[1]) >= start_key
                and (next_generated_key is None or _event_key(completion[0], completion[1]) < next_generated_key)
            ]
            recommended_product_id = int(item["recommended_product_id"])
            exact_match = any(
                str(completion[2] or "") == "recommended_product_id"
                and (completion[3] is None or int(completion[3]) == recommended_product_id)
                for completion in completions
            )
            semantic_match = any(
                str(completion[2] or "") == "semantic_content_match"
                and (completion[3] is None or int(completion[3]) == recommended_product_id)
                for completion in completions
            )
            product_type_match = any(str(completion[2] or "") == "product_type" for completion in completions)

            bucket = stats_by_product.setdefault(
                recommended_product_id,
                {
                    "recommended_steps": 0,
                    "exact_recommended_product_checkout": 0,
                    "semantic_alternative_checkout": 0,
                    "product_type_checkout": 0,
                },
            )
            bucket["recommended_steps"] += 1
            bucket["exact_recommended_product_checkout"] += int(bool(exact_match))
            bucket["semantic_alternative_checkout"] += int(bool(semantic_match))
            bucket["product_type_checkout"] += int(bool(product_type_match))

    finalized: dict[int, dict[str, Any]] = {}
    for product_id, bucket in stats_by_product.items():
        recommended_steps = int(bucket.get("recommended_steps") or 0)
        exact = int(bucket.get("exact_recommended_product_checkout") or 0)
        semantic = int(bucket.get("semantic_alternative_checkout") or 0)
        product_type_checkout = int(bucket.get("product_type_checkout") or 0)
        exact_rate = (float(exact) / float(recommended_steps)) if recommended_steps > 0 else 0.0
        semantic_rate = (float(semantic) / float(recommended_steps)) if recommended_steps > 0 else 0.0
        product_type_rate = (
            (float(product_type_checkout) / float(recommended_steps))
            if recommended_steps > 0
            else 0.0
        )
        effective_rate = (
            (float(exact) + (semantic_weight * float(semantic))) / float(recommended_steps)
            if recommended_steps > 0
            else 0.0
        )
        finalized[int(product_id)] = {
            "recommended_steps": recommended_steps,
            "exact_recommended_product_checkout": exact,
            "semantic_alternative_checkout": semantic,
            "product_type_checkout": product_type_checkout,
            "exact_adoption_rate": round(exact_rate, 6),
            "semantic_alternative_rate": round(semantic_rate, 6),
            "product_type_match_rate": round(product_type_rate, 6),
            "effective_adoption_rate": round(effective_rate, 6),
        }

    cache.set(cache_key, finalized, timeout=cache_ttl)
    return finalized


def _sku_health_penalty(
    *,
    category: str,
    product_type: str,
    product_id: int | None,
    health_map: dict[int, dict[str, Any]] | None,
) -> tuple[float, dict[str, Any]]:
    if not _sku_health_penalty_enabled(category=category, product_type=product_type):
        return 0.0, {}
    product_id_int = _to_int(product_id)
    if product_id_int is None:
        return 0.0, {}

    stats = _safe_dict((health_map or {}).get(int(product_id_int)))
    if not stats:
        return 0.0, {}

    min_recommended_steps = int(getattr(settings, "ROADMAP_SKU_HEALTH_PENALTY_MIN_RECOMMENDED_STEPS", 20) or 20)
    max_exact_rate = _safe_float(getattr(settings, "ROADMAP_SKU_HEALTH_PENALTY_MAX_EXACT_ADOPTION_RATE", 0.01), default=0.01)
    max_effective_rate = _safe_float(
        getattr(settings, "ROADMAP_SKU_HEALTH_PENALTY_MAX_EFFECTIVE_ADOPTION_RATE", 0.03),
        default=0.03,
    )
    max_penalty = _safe_float(getattr(settings, "ROADMAP_SKU_HEALTH_PENALTY_MAX_VALUE", 0.65), default=0.65)

    recommended_steps = int(stats.get("recommended_steps") or 0)
    exact_rate = _safe_float(stats.get("exact_adoption_rate"), default=0.0)
    semantic_rate = _safe_float(stats.get("semantic_alternative_rate"), default=0.0)
    effective_rate = _safe_float(stats.get("effective_adoption_rate"), default=0.0)

    meta = {
        "enabled": True,
        "recommended_steps": recommended_steps,
        "exact_adoption_rate": round(exact_rate, 6),
        "semantic_alternative_rate": round(semantic_rate, 6),
        "effective_adoption_rate": round(effective_rate, 6),
        "min_recommended_steps": int(min_recommended_steps),
        "max_exact_adoption_rate": round(max_exact_rate, 6),
        "max_effective_adoption_rate": round(max_effective_rate, 6),
        "penalty": 0.0,
        "eligible": False,
    }

    if recommended_steps < min_recommended_steps:
        return 0.0, meta
    meta["eligible"] = True
    if exact_rate > max_exact_rate or effective_rate > max_effective_rate:
        return 0.0, meta

    safe_exact = max(max_exact_rate, 1e-9)
    safe_effective = max(max_effective_rate, 1e-9)
    exact_gap = max(0.0, max_exact_rate - exact_rate) / safe_exact
    effective_gap = max(0.0, max_effective_rate - effective_rate) / safe_effective
    severity = max(exact_gap, effective_gap)
    volume_factor = min(1.0, float(recommended_steps) / float(max(min_recommended_steps * 3, 1)))
    penalty = min(max_penalty, max_penalty * severity * volume_factor)
    meta["penalty"] = round(-penalty, 6)
    return -penalty, meta


def _direct_profile_bonus(
    *,
    category: str,
    product_type: str,
    profile_sig: dict[str, Any],
    candidate_sig: dict[str, Any],
) -> tuple[float, list[str], dict[str, float]]:
    bonus = 0.0
    why: list[str] = []
    components: dict[str, float] = {}

    avoid_conflicts = _overlap(candidate_sig.get("flags") or [], profile_sig.get("avoid_flags") or [])
    if avoid_conflicts:
        penalty = min(2.5, 1.2 + 0.35 * len(avoid_conflicts))
        bonus -= penalty
        components["avoid_conflict_penalty"] = round(-penalty, 6)
        why.append(f"profile avoid flags conflict: {', '.join(avoid_conflicts[:3])}")

    if category == "haircare":
        if candidate_sig.get("hair_type") not in {"", "__none__"} and candidate_sig.get("hair_type") == profile_sig.get("hair_type"):
            bonus += 0.35
            components["hair_type_match"] = 0.35
            why.append(f"matches hair_type={candidate_sig['hair_type']}")
        if candidate_sig.get("scalp_type") not in {"", "__none__"} and candidate_sig.get("scalp_type") == profile_sig.get("scalp_type"):
            bonus += 0.3
            components["scalp_type_match"] = 0.3
            why.append(f"matches scalp_type={candidate_sig['scalp_type']}")
        if (
            candidate_sig.get("hair_thickness") not in {"", "__none__"}
            and candidate_sig.get("hair_thickness") == profile_sig.get("hair_thickness")
        ):
            bonus += 0.2
            components["hair_thickness_match"] = 0.2
            why.append(f"matches hair_thickness={candidate_sig['hair_thickness']}")

        target_concerns = list(profile_sig.get("hair_concerns") or []) + list(profile_sig.get("goals") or [])
        concern_overlap = _overlap(candidate_sig.get("concerns") or [], target_concerns)
        if concern_overlap:
            concern_bonus = min(0.7, 0.18 * len(set(concern_overlap)))
            bonus += concern_bonus
            components["hair_concern_overlap"] = round(concern_bonus, 6)
            why.append(f"matches profile concerns: {', '.join(sorted(set(concern_overlap))[:3])}")

        profile_goals = set(profile_sig.get("goals") or [])
        profile_hair_concerns = set(profile_sig.get("hair_concerns") or [])
        profile_scalp_type = str(profile_sig.get("scalp_type") or "")
        profile_hair_type = str(profile_sig.get("hair_type") or "")
        profile_thickness = str(profile_sig.get("hair_thickness") or "")
        finish = str(candidate_sig.get("finish") or "")
        actives = set(candidate_sig.get("actives") or [])
        flags = set(candidate_sig.get("flags") or [])
        concerns = set(candidate_sig.get("concerns") or [])

        if product_type == "leave_in":
            if bool((_LIGHTWEIGHT_GOALS & profile_goals) or (_LIGHTWEIGHT_GOALS & profile_hair_concerns) or profile_thickness == "fine"):
                if finish in _LIGHT_FINISHES:
                    bonus += 0.4
                    components["leavein_light_finish"] = 0.4
                    why.append(f"lightweight finish={finish}")
                if finish in _RICH_FINISHES:
                    bonus -= 0.25
                    components["leavein_rich_finish_penalty"] = -0.25
                    why.append(f"rich finish penalty={finish}")
                if "heavy_oils" in flags:
                    bonus -= 0.8
                    components["leavein_heavy_oils_penalty"] = -0.8
                    why.append("heavy oils penalized for lightweight profile")
                light_overlap = actives & _LIGHTWEIGHT_ACTIVES
                if light_overlap:
                    light_bonus = min(0.3, 0.12 * len(light_overlap))
                    bonus += light_bonus
                    components["leavein_light_actives"] = round(light_bonus, 6)
                    why.append(f"lightweight actives: {', '.join(sorted(light_overlap)[:3])}")
            if bool((_CURL_GOALS & profile_goals) or (_CURL_GOALS & profile_hair_concerns) or profile_hair_type in {"curly", "coily"}):
                if finish in _DEFINED_FINISHES:
                    bonus += 0.35
                    components["leavein_defined_finish"] = 0.35
                    why.append(f"curl definition finish={finish}")
                curl_overlap = actives & _CURL_LEAVEIN_ACTIVES
                if curl_overlap:
                    curl_bonus = min(0.35, 0.1 * len(curl_overlap))
                    bonus += curl_bonus
                    components["leavein_curl_actives"] = round(curl_bonus, 6)
                    why.append(f"curl-support actives: {', '.join(sorted(curl_overlap)[:3])}")

        elif product_type == "hair_oil":
            if bool((_LIGHTWEIGHT_GOALS & profile_goals) or (_LIGHTWEIGHT_GOALS & profile_hair_concerns) or profile_thickness == "fine"):
                penalty = 0.55
                if "heavy_oils" in flags or finish in _RICH_FINISHES:
                    penalty += 0.35
                bonus -= penalty
                components["hair_oil_lightweight_penalty"] = round(-penalty, 6)
                why.append("hair oil penalized for lightweight profile")
            if bool((_REPAIR_GOALS & profile_goals) or (_REPAIR_GOALS & profile_hair_concerns) or profile_thickness == "thick"):
                repair_overlap = actives & _RICH_OIL_ACTIVES
                repair_bonus = 0.2 + min(0.25, 0.08 * len(repair_overlap))
                bonus += repair_bonus
                components["hair_oil_repair_bonus"] = round(repair_bonus, 6)
                why.append("repair profile supports richer oil")

        elif product_type == "scalp_serum":
            scalp_overlap = concerns & SCALP_OBJECTIVE_TOKENS
            if scalp_overlap:
                scalp_bonus = min(0.6, 0.18 * len(scalp_overlap))
                bonus += scalp_bonus
                components["scalp_concern_bonus"] = round(scalp_bonus, 6)
                why.append(f"scalp concerns: {', '.join(sorted(scalp_overlap)[:3])}")
            if profile_scalp_type == "oily":
                oily_overlap = actives & _OILY_SCALP_SUPPORT
                if oily_overlap:
                    oily_bonus = 0.25 + min(0.25, 0.08 * len(oily_overlap))
                    bonus += oily_bonus
                    components["scalp_oily_support"] = round(oily_bonus, 6)
                    why.append(f"oil-control actives: {', '.join(sorted(oily_overlap)[:3])}")
            if profile_scalp_type == "sensitive":
                sensitive_overlap = actives & _SENSITIVE_SCALP_SUPPORT
                if sensitive_overlap:
                    sensitive_bonus = 0.2 + min(0.2, 0.08 * len(sensitive_overlap))
                    bonus += sensitive_bonus
                    components["scalp_sensitive_support"] = round(sensitive_bonus, 6)
                    why.append(f"sensitive-scalp actives: {', '.join(sorted(sensitive_overlap)[:3])}")

        elif product_type in {"conditioner", "hair_mask"}:
            repair_overlap = concerns & (_REPAIR_GOALS | _CURL_GOALS)
            if repair_overlap:
                repair_bonus = min(0.5, 0.14 * len(repair_overlap))
                bonus += repair_bonus
                components["repair_overlap"] = round(repair_bonus, 6)
                why.append(f"repair/hydration concerns: {', '.join(sorted(repair_overlap)[:3])}")

    elif category == "fragrance":
        liked_families = set(profile_sig.get("fragrance_liked_families") or [])
        liked_notes = set(profile_sig.get("fragrance_liked_notes") or [])
        if str(candidate_sig.get("scent_family") or "") in liked_families:
            bonus += 0.45
            components["fragrance_family_match"] = 0.45
        note_overlap = set(candidate_sig.get("notes") or []) & liked_notes
        if note_overlap:
            note_bonus = min(0.3, 0.08 * len(note_overlap))
            bonus += note_bonus
            components["fragrance_note_overlap"] = round(note_bonus, 6)

    return bonus, why, components


def _anchor_bonus(
    *,
    candidate_sig: dict[str, Any],
    anchor_sig: dict[str, Any],
) -> tuple[float, list[str], dict[str, float]]:
    if not anchor_sig:
        return 0.0, [], {}

    bonus = 0.0
    why: list[str] = []
    components: dict[str, float] = {}

    shared_concerns = _overlap(candidate_sig.get("concerns") or [], anchor_sig.get("concerns") or [])
    if shared_concerns:
        concern_bonus = min(0.25, 0.08 * len(set(shared_concerns)))
        bonus += concern_bonus
        components["anchor_shared_concerns"] = round(concern_bonus, 6)
        why.append(f"aligns with anchor concerns: {', '.join(sorted(set(shared_concerns))[:3])}")

    shared_actives = _overlap(candidate_sig.get("actives") or [], anchor_sig.get("actives") or [])
    if shared_actives:
        actives_bonus = min(0.2, 0.06 * len(set(shared_actives)))
        bonus += actives_bonus
        components["anchor_shared_actives"] = round(actives_bonus, 6)
        why.append(f"aligns with anchor actives: {', '.join(sorted(set(shared_actives))[:3])}")

    shared_inci = _overlap(candidate_sig.get("inci_tokens") or [], anchor_sig.get("inci_tokens") or [])
    if shared_inci:
        inci_bonus = min(0.12, 0.03 * len(set(shared_inci)))
        bonus += inci_bonus
        components["anchor_shared_inci"] = round(inci_bonus, 6)

    active_conflicts = _conflict_count(
        candidate_sig.get("actives") or [],
        anchor_sig.get("actives") or [],
    )
    if active_conflicts:
        penalty = min(0.8, 0.22 * active_conflicts)
        bonus -= penalty
        components["anchor_active_conflict_penalty"] = round(-penalty, 6)
        why.append("active conflict penalty")

    return bonus, why, components


def rerank_roadmap_candidate_rows(
    *,
    category: str,
    product_type: str,
    profile: Any,
    context_products: list[dict[str, Any]] | None,
    rows: list[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    category_norm = str(category or "").strip().lower()
    product_type_norm = str(product_type or "").strip().lower()
    if not rows:
        return []

    profile_sig = profile_signature(profile)
    anchor_sig = _anchor_signature_for_category(category=category_norm, context_products=context_products)
    health_map = _recent_recommended_product_health_map(category=category_norm, product_type=product_type_norm)

    reranked: list[dict[str, Any]] = []
    for row in rows:
        product = _safe_dict(_safe_dict(row).get("product"))
        candidate_sig = product_signature(product)
        base_score = _safe_float(_safe_dict(row).get("score"), default=0.0)

        profile_bonus, profile_why, profile_components = _direct_profile_bonus(
            category=category_norm,
            product_type=product_type_norm,
            profile_sig=profile_sig,
            candidate_sig=candidate_sig,
        )
        anchor_bonus, anchor_why, anchor_components = _anchor_bonus(
            candidate_sig=candidate_sig,
            anchor_sig=anchor_sig,
        )
        health_penalty, health_meta = _sku_health_penalty(
            category=category_norm,
            product_type=product_type_norm,
            product_id=_to_int(product.get("id")),
            health_map=health_map,
        )
        rerank_delta = profile_bonus + anchor_bonus + health_penalty
        final_score = base_score + rerank_delta

        merged_why = [str(item) for item in _safe_list(_safe_dict(row).get("why"))]
        for reason in [*profile_why, *anchor_why]:
            if reason not in merged_why:
                merged_why.append(reason)
        if health_penalty < 0:
            reason = "low recent SKU adoption signal"
            if reason not in merged_why:
                merged_why.append(reason)

        components = dict(_safe_dict(_safe_dict(row).get("components")))
        components["roadmap_rerank"] = {
            "base_score": round(base_score, 6),
            "rerank_delta": round(rerank_delta, 6),
            "final_score": round(final_score, 6),
            "profile": profile_components,
            "anchor": anchor_components,
            "sku_health": health_meta,
        }

        reranked.append(
            {
                **row,
                "score": round(final_score, 6),
                "why": merged_why,
                "components": components,
                "_roadmap_rerank_delta": round(rerank_delta, 6),
            }
        )

    reranked.sort(
        key=lambda item: (
            -_safe_float(item.get("score"), default=0.0),
            -_safe_float(item.get("_roadmap_rerank_delta"), default=0.0),
            -_safe_float(_safe_dict(item.get("components")).get("cooccurrence"), default=0.0),
            int(_safe_dict(item.get("product")).get("id") or 0),
        )
    )
    for item in reranked:
        item.pop("_roadmap_rerank_delta", None)
    return reranked
