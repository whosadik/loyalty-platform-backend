from __future__ import annotations

import re
from collections import Counter
from typing import Any

from ml_logic.routine_rules import CONFLICT_PAIRS


_TOKEN_RE = re.compile(r"[^a-z0-9_]+")
_LIST_SPLIT_RE = re.compile(r"[,;/|\n\r]+")

_CONFLICT_LOOKUP = {tuple(sorted(pair)) for pair in CONFLICT_PAIRS}

BASE_CATEGORICAL_FEATURES = [
    "profile_skin_type",
    "profile_budget",
    "profile_hair_type",
    "profile_scalp_type",
    "profile_hair_thickness",
    "profile_makeup_finish_pref_primary",
    "profile_makeup_coverage_pref_primary",
    "profile_makeup_undertone",
    "profile_makeup_tone_family",
    "profile_fragrance_intensity_pref",
    "anchor_product_type",
    "anchor_hair_type",
    "anchor_scalp_type",
    "anchor_hair_thickness",
    "anchor_finish",
    "anchor_coverage",
    "anchor_undertone",
    "anchor_tone_family",
    "anchor_scent_family",
    "anchor_intensity",
]

BASE_NUMERIC_FEATURES = [
    "profile_goals_count",
    "profile_avoid_flags_count",
    "profile_hair_concerns_count",
    "profile_makeup_finish_pref_count",
    "profile_makeup_coverage_pref_count",
    "profile_makeup_concerns_count",
    "profile_fragrance_liked_families_count",
    "profile_fragrance_liked_notes_count",
    "anchor_concerns_count",
    "anchor_actives_count",
    "anchor_supported_skin_types_count",
    "anchor_notes_count",
    "anchor_inci_token_count",
]

CANDIDATE_CATEGORICAL_FEATURES = [
    "candidate_dominant_hair_type",
    "candidate_dominant_scalp_type",
    "candidate_dominant_hair_thickness",
    "candidate_dominant_finish",
    "candidate_dominant_coverage",
    "candidate_dominant_undertone",
    "candidate_dominant_tone_family",
    "candidate_dominant_scent_family",
    "candidate_dominant_intensity",
]

CANDIDATE_NUMERIC_FEATURES = [
    "candidate_catalog_product_count",
    "candidate_catalog_avg_concerns_count",
    "candidate_catalog_avg_actives_count",
    "candidate_catalog_avg_notes_count",
    "candidate_catalog_avg_inci_token_count",
    "candidate_profile_avoid_conflict_rate",
    "candidate_profile_goal_match_rate",
    "candidate_profile_skin_type_match_rate",
    "candidate_profile_hair_concern_match_rate",
    "candidate_profile_hair_type_match_rate",
    "candidate_profile_scalp_type_match_rate",
    "candidate_profile_hair_thickness_match_rate",
    "candidate_profile_makeup_finish_match_rate",
    "candidate_profile_makeup_coverage_match_rate",
    "candidate_profile_makeup_undertone_match_rate",
    "candidate_profile_makeup_tone_family_match_rate",
    "candidate_profile_makeup_concern_match_rate",
    "candidate_profile_fragrance_family_match_rate",
    "candidate_profile_fragrance_note_match_rate",
    "candidate_profile_fragrance_intensity_match_rate",
    "candidate_anchor_shared_concern_rate",
    "candidate_anchor_shared_active_rate",
    "candidate_anchor_shared_inci_rate",
    "candidate_anchor_active_conflict_rate",
]

CHAIN_TRANSITION_NUMERIC_FEATURES = [
    "anchor_position_in_chain",
    "last1_position_in_chain",
    "last2_position_in_chain",
    "candidate_distance_from_anchor",
    "candidate_distance_from_last1",
    "candidate_distance_from_last2",
    "candidate_abs_distance_from_anchor",
    "candidate_abs_distance_from_last1",
    "candidate_abs_distance_from_last2",
    "candidate_is_same_as_anchor",
    "candidate_is_after_anchor",
    "candidate_is_before_anchor",
    "candidate_is_immediate_followup_to_anchor",
    "candidate_is_immediate_predecessor_to_anchor",
    "candidate_is_same_as_last1",
    "candidate_is_after_last1",
    "candidate_is_before_last1",
    "candidate_is_immediate_followup_to_last1",
    "candidate_is_immediate_predecessor_to_last1",
    "last1_after_last2_in_chain",
    "last1_before_last2_in_chain",
    "candidate_continues_last_transition_direction",
    "candidate_reverses_last_transition_direction",
]

NEXTSTEP_PLAN_STATE_CATEGORICAL_FEATURES = [
    "planned_target_product_type",
]

NEXTSTEP_PLAN_STATE_NUMERIC_FEATURES = [
    "planned_target_step_index",
    "planned_target_position_in_chain",
    "candidate_matches_planned_target",
    "candidate_distance_from_planned_target",
    "candidate_abs_distance_from_planned_target",
    "candidate_is_after_planned_target",
    "candidate_is_before_planned_target",
    "candidate_is_immediate_followup_to_planned_target",
    "candidate_is_immediate_predecessor_to_planned_target",
]

ALL_CATEGORICAL_FEATURES = [*BASE_CATEGORICAL_FEATURES, *CANDIDATE_CATEGORICAL_FEATURES]
ALL_NUMERIC_FEATURES = [*BASE_NUMERIC_FEATURES, *CANDIDATE_NUMERIC_FEATURES]


def _safe_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def slug_token(raw: Any, *, default: str = "__none__") -> str:
    text = str(raw or "").strip().lower()
    if not text:
        return default
    text = text.replace("-", "_").replace(" ", "_")
    text = _TOKEN_RE.sub("_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text or default


def normalize_tokens(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        raw_values = list(value)
    elif isinstance(value, str):
        raw_values = [item for item in _LIST_SPLIT_RE.split(value) if item.strip()]
    else:
        raw_values = [value]

    out: list[str] = []
    seen: set[str] = set()
    for item in raw_values:
        token = slug_token(item, default="")
        if not token or token in seen:
            continue
        seen.add(token)
        out.append(token)
    return out


def tokenize_inci(value: Any, *, limit: int = 24) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for item in normalize_tokens(value):
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
        if len(out) >= limit:
            break
    return out


def _get_attr_token(attrs: dict[str, Any], raw_meta: dict[str, Any], *keys: str) -> str:
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


def _get_attr_tokens(attrs: dict[str, Any], raw_meta: dict[str, Any], *keys: str) -> list[str]:
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


def product_signature(product: dict[str, Any] | None) -> dict[str, Any]:
    product = product or {}
    attrs = _safe_dict(product.get("attrs"))
    raw_meta = _safe_dict(product.get("raw_meta"))
    return {
        "category": slug_token(product.get("category"), default=""),
        "product_type": slug_token(product.get("product_type")),
        "concerns": normalize_tokens(product.get("concerns")),
        "actives": normalize_tokens(product.get("actives")),
        "flags": normalize_tokens(product.get("flags")),
        "supported_skin_types": normalize_tokens(product.get("supported_skin_types")),
        "hair_type": _get_attr_token(attrs, raw_meta, "hair_type"),
        "scalp_type": _get_attr_token(attrs, raw_meta, "scalp_type"),
        "hair_thickness": _get_attr_token(attrs, raw_meta, "hair_thickness"),
        "finish": _get_attr_token(attrs, raw_meta, "finish"),
        "coverage": _get_attr_token(attrs, raw_meta, "coverage"),
        "undertone": _get_attr_token(attrs, raw_meta, "undertone"),
        "tone_family": _get_attr_token(attrs, raw_meta, "tone_family"),
        "scent_family": _get_attr_token(attrs, raw_meta, "scent_family"),
        "intensity": _get_attr_token(attrs, raw_meta, "intensity"),
        "notes": _get_attr_tokens(attrs, raw_meta, "notes"),
        "inci_tokens": tokenize_inci(product.get("ingredients_inci")),
    }


def _field_value(source: Any, name: str) -> Any:
    if source is None:
        return None
    if isinstance(source, dict):
        return source.get(name)
    return getattr(source, name, None)


def profile_signature(profile: Any | None) -> dict[str, Any]:
    hair = _safe_dict(_field_value(profile, "hair_profile"))
    makeup = _safe_dict(_field_value(profile, "makeup_profile"))
    fragrance = _safe_dict(_field_value(profile, "fragrance_profile"))

    return {
        "skin_type": slug_token(_field_value(profile, "skin_type")),
        "goals": normalize_tokens(_field_value(profile, "goals")),
        "avoid_flags": normalize_tokens(_field_value(profile, "avoid_flags")),
        "budget": slug_token(_field_value(profile, "budget")),
        "hair_type": slug_token(hair.get("hair_type")),
        "scalp_type": slug_token(hair.get("scalp_type")),
        "hair_thickness": slug_token(hair.get("hair_thickness")),
        "hair_concerns": normalize_tokens(hair.get("concerns")),
        "makeup_finish_pref": normalize_tokens(makeup.get("finish_pref")),
        "makeup_coverage_pref": normalize_tokens(makeup.get("coverage_pref")),
        "makeup_undertone": slug_token(makeup.get("undertone")),
        "makeup_tone_family": slug_token(makeup.get("tone_family")),
        "makeup_concerns": normalize_tokens(makeup.get("concerns")),
        "fragrance_liked_families": normalize_tokens(fragrance.get("liked_families")),
        "fragrance_liked_notes": normalize_tokens(fragrance.get("liked_notes")),
        "fragrance_intensity_pref": slug_token(fragrance.get("intensity_pref")),
    }


def _empty_summary() -> dict[str, Any]:
    return {
        "product_count": 0,
        "supports_all_skin_types_count": 0,
        "concerns_len_sum": 0,
        "actives_len_sum": 0,
        "notes_len_sum": 0,
        "inci_len_sum": 0,
        "concerns": Counter(),
        "actives": Counter(),
        "flags": Counter(),
        "supported_skin_types": Counter(),
        "hair_type": Counter(),
        "scalp_type": Counter(),
        "hair_thickness": Counter(),
        "finish": Counter(),
        "coverage": Counter(),
        "undertone": Counter(),
        "tone_family": Counter(),
        "scent_family": Counter(),
        "intensity": Counter(),
        "notes": Counter(),
        "inci_tokens": Counter(),
    }


def build_candidate_catalog_summaries(products: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    out: dict[tuple[str, str], dict[str, Any]] = {}
    for product in products:
        signature = product_signature(product)
        category = str(signature.get("category") or "")
        product_type = str(signature.get("product_type") or "")
        if not category or not product_type:
            continue

        key = (category, product_type)
        summary = out.setdefault(key, _empty_summary())
        summary["product_count"] += 1
        summary["concerns_len_sum"] += len(signature["concerns"])
        summary["actives_len_sum"] += len(signature["actives"])
        summary["notes_len_sum"] += len(signature["notes"])
        summary["inci_len_sum"] += len(signature["inci_tokens"])
        summary["concerns"].update(set(signature["concerns"]))
        summary["actives"].update(set(signature["actives"]))
        summary["flags"].update(set(signature["flags"]))
        summary["notes"].update(set(signature["notes"]))
        summary["inci_tokens"].update(set(signature["inci_tokens"]))
        if signature["supported_skin_types"]:
            summary["supported_skin_types"].update(set(signature["supported_skin_types"]))
        else:
            summary["supports_all_skin_types_count"] += 1

        for name in [
            "hair_type",
            "scalp_type",
            "hair_thickness",
            "finish",
            "coverage",
            "undertone",
            "tone_family",
            "scent_family",
            "intensity",
        ]:
            token = str(signature.get(name) or "__none__")
            if token != "__none__":
                summary[name].update([token])
    return out


def _dominant(counter: Counter[str]) -> str:
    if not counter:
        return "__none__"
    return str(counter.most_common(1)[0][0] or "__none__")


def _avg(sum_value: Any, total: int) -> float:
    if total <= 0:
        return 0.0
    return round(float(sum_value) / float(total), 6)


def _rate(counter: Counter[str], tokens: list[str], total: int) -> float:
    if total <= 0 or not tokens:
        return 0.0
    matched = sum(int(counter.get(token, 0)) for token in set(tokens))
    return round(min(1.0, float(matched) / float(total)), 6)


def _skin_type_rate(summary: dict[str, Any], profile_sig: dict[str, Any]) -> float:
    total = int(summary.get("product_count") or 0)
    skin_type = str(profile_sig.get("skin_type") or "__none__")
    if total <= 0 or skin_type == "__none__":
        return 0.0
    matched = int(summary.get("supports_all_skin_types_count") or 0) + int(
        (summary.get("supported_skin_types") or Counter()).get(skin_type, 0)
    )
    return round(min(1.0, float(matched) / float(total)), 6)


def _conflict_rate(summary: dict[str, Any], anchor_sig: dict[str, Any]) -> float:
    total = int(summary.get("product_count") or 0)
    anchor_actives = [str(x) for x in (anchor_sig.get("actives") or []) if str(x)]
    if total <= 0 or not anchor_actives:
        return 0.0
    candidate_actives = summary.get("actives") or Counter()
    conflicting_tokens: set[str] = set()
    for anchor_active in anchor_actives:
        for candidate_active in candidate_actives.keys():
            if tuple(sorted((anchor_active, candidate_active))) in _CONFLICT_LOOKUP:
                conflicting_tokens.add(str(candidate_active))
    if not conflicting_tokens:
        return 0.0
    matched = sum(int(candidate_actives.get(token, 0)) for token in conflicting_tokens)
    return round(min(1.0, float(matched) / float(total)), 6)


def build_base_content_features(profile_sig: dict[str, Any] | None, anchor_sig: dict[str, Any] | None) -> dict[str, Any]:
    profile_sig = profile_sig or {}
    anchor_sig = anchor_sig or {}
    out = {col: "__none__" for col in BASE_CATEGORICAL_FEATURES}
    out.update({col: 0.0 for col in BASE_NUMERIC_FEATURES})

    out["profile_skin_type"] = str(profile_sig.get("skin_type") or "__none__")
    out["profile_budget"] = str(profile_sig.get("budget") or "__none__")
    out["profile_hair_type"] = str(profile_sig.get("hair_type") or "__none__")
    out["profile_scalp_type"] = str(profile_sig.get("scalp_type") or "__none__")
    out["profile_hair_thickness"] = str(profile_sig.get("hair_thickness") or "__none__")
    out["profile_makeup_finish_pref_primary"] = (
        str((profile_sig.get("makeup_finish_pref") or ["__none__"])[0] or "__none__")
    )
    out["profile_makeup_coverage_pref_primary"] = (
        str((profile_sig.get("makeup_coverage_pref") or ["__none__"])[0] or "__none__")
    )
    out["profile_makeup_undertone"] = str(profile_sig.get("makeup_undertone") or "__none__")
    out["profile_makeup_tone_family"] = str(profile_sig.get("makeup_tone_family") or "__none__")
    out["profile_fragrance_intensity_pref"] = str(
        profile_sig.get("fragrance_intensity_pref") or "__none__"
    )

    out["profile_goals_count"] = int(len(profile_sig.get("goals") or []))
    out["profile_avoid_flags_count"] = int(len(profile_sig.get("avoid_flags") or []))
    out["profile_hair_concerns_count"] = int(len(profile_sig.get("hair_concerns") or []))
    out["profile_makeup_finish_pref_count"] = int(len(profile_sig.get("makeup_finish_pref") or []))
    out["profile_makeup_coverage_pref_count"] = int(len(profile_sig.get("makeup_coverage_pref") or []))
    out["profile_makeup_concerns_count"] = int(len(profile_sig.get("makeup_concerns") or []))
    out["profile_fragrance_liked_families_count"] = int(len(profile_sig.get("fragrance_liked_families") or []))
    out["profile_fragrance_liked_notes_count"] = int(len(profile_sig.get("fragrance_liked_notes") or []))

    out["anchor_product_type"] = str(anchor_sig.get("product_type") or "__none__")
    out["anchor_hair_type"] = str(anchor_sig.get("hair_type") or "__none__")
    out["anchor_scalp_type"] = str(anchor_sig.get("scalp_type") or "__none__")
    out["anchor_hair_thickness"] = str(anchor_sig.get("hair_thickness") or "__none__")
    out["anchor_finish"] = str(anchor_sig.get("finish") or "__none__")
    out["anchor_coverage"] = str(anchor_sig.get("coverage") or "__none__")
    out["anchor_undertone"] = str(anchor_sig.get("undertone") or "__none__")
    out["anchor_tone_family"] = str(anchor_sig.get("tone_family") or "__none__")
    out["anchor_scent_family"] = str(anchor_sig.get("scent_family") or "__none__")
    out["anchor_intensity"] = str(anchor_sig.get("intensity") or "__none__")
    out["anchor_concerns_count"] = int(len(anchor_sig.get("concerns") or []))
    out["anchor_actives_count"] = int(len(anchor_sig.get("actives") or []))
    out["anchor_supported_skin_types_count"] = int(len(anchor_sig.get("supported_skin_types") or []))
    out["anchor_notes_count"] = int(len(anchor_sig.get("notes") or []))
    out["anchor_inci_token_count"] = int(len(anchor_sig.get("inci_tokens") or []))
    return out


def build_candidate_content_features(
    candidate_summary: dict[str, Any] | None,
    profile_sig: dict[str, Any] | None,
    anchor_sig: dict[str, Any] | None,
) -> dict[str, Any]:
    profile_sig = profile_sig or {}
    anchor_sig = anchor_sig or {}
    summary = candidate_summary or _empty_summary()
    total = int(summary.get("product_count") or 0)

    out = {col: "__none__" for col in CANDIDATE_CATEGORICAL_FEATURES}
    out.update({col: 0.0 for col in CANDIDATE_NUMERIC_FEATURES})

    out["candidate_dominant_hair_type"] = _dominant(summary.get("hair_type") or Counter())
    out["candidate_dominant_scalp_type"] = _dominant(summary.get("scalp_type") or Counter())
    out["candidate_dominant_hair_thickness"] = _dominant(summary.get("hair_thickness") or Counter())
    out["candidate_dominant_finish"] = _dominant(summary.get("finish") or Counter())
    out["candidate_dominant_coverage"] = _dominant(summary.get("coverage") or Counter())
    out["candidate_dominant_undertone"] = _dominant(summary.get("undertone") or Counter())
    out["candidate_dominant_tone_family"] = _dominant(summary.get("tone_family") or Counter())
    out["candidate_dominant_scent_family"] = _dominant(summary.get("scent_family") or Counter())
    out["candidate_dominant_intensity"] = _dominant(summary.get("intensity") or Counter())

    out["candidate_catalog_product_count"] = int(total)
    out["candidate_catalog_avg_concerns_count"] = _avg(summary.get("concerns_len_sum"), total)
    out["candidate_catalog_avg_actives_count"] = _avg(summary.get("actives_len_sum"), total)
    out["candidate_catalog_avg_notes_count"] = _avg(summary.get("notes_len_sum"), total)
    out["candidate_catalog_avg_inci_token_count"] = _avg(summary.get("inci_len_sum"), total)
    out["candidate_profile_avoid_conflict_rate"] = _rate(
        summary.get("flags") or Counter(),
        list(profile_sig.get("avoid_flags") or []),
        total,
    )
    out["candidate_profile_goal_match_rate"] = _rate(
        summary.get("concerns") or Counter(),
        list(profile_sig.get("goals") or []),
        total,
    )
    out["candidate_profile_skin_type_match_rate"] = _skin_type_rate(summary, profile_sig)
    out["candidate_profile_hair_concern_match_rate"] = _rate(
        summary.get("concerns") or Counter(),
        list(profile_sig.get("hair_concerns") or []),
        total,
    )
    out["candidate_profile_hair_type_match_rate"] = _rate(
        summary.get("hair_type") or Counter(),
        [str(profile_sig.get("hair_type") or "__none__")],
        total,
    )
    out["candidate_profile_scalp_type_match_rate"] = _rate(
        summary.get("scalp_type") or Counter(),
        [str(profile_sig.get("scalp_type") or "__none__")],
        total,
    )
    out["candidate_profile_hair_thickness_match_rate"] = _rate(
        summary.get("hair_thickness") or Counter(),
        [str(profile_sig.get("hair_thickness") or "__none__")],
        total,
    )
    out["candidate_profile_makeup_finish_match_rate"] = _rate(
        summary.get("finish") or Counter(),
        list(profile_sig.get("makeup_finish_pref") or []),
        total,
    )
    out["candidate_profile_makeup_coverage_match_rate"] = _rate(
        summary.get("coverage") or Counter(),
        list(profile_sig.get("makeup_coverage_pref") or []),
        total,
    )
    out["candidate_profile_makeup_undertone_match_rate"] = _rate(
        summary.get("undertone") or Counter(),
        [str(profile_sig.get("makeup_undertone") or "__none__")],
        total,
    )
    out["candidate_profile_makeup_tone_family_match_rate"] = _rate(
        summary.get("tone_family") or Counter(),
        [str(profile_sig.get("makeup_tone_family") or "__none__")],
        total,
    )
    out["candidate_profile_makeup_concern_match_rate"] = _rate(
        summary.get("concerns") or Counter(),
        list(profile_sig.get("makeup_concerns") or []),
        total,
    )
    out["candidate_profile_fragrance_family_match_rate"] = _rate(
        summary.get("scent_family") or Counter(),
        list(profile_sig.get("fragrance_liked_families") or []),
        total,
    )
    out["candidate_profile_fragrance_note_match_rate"] = _rate(
        summary.get("notes") or Counter(),
        list(profile_sig.get("fragrance_liked_notes") or []),
        total,
    )
    out["candidate_profile_fragrance_intensity_match_rate"] = _rate(
        summary.get("intensity") or Counter(),
        [str(profile_sig.get("fragrance_intensity_pref") or "__none__")],
        total,
    )
    out["candidate_anchor_shared_concern_rate"] = _rate(
        summary.get("concerns") or Counter(),
        list(anchor_sig.get("concerns") or []),
        total,
    )
    out["candidate_anchor_shared_active_rate"] = _rate(
        summary.get("actives") or Counter(),
        list(anchor_sig.get("actives") or []),
        total,
    )
    out["candidate_anchor_shared_inci_rate"] = _rate(
        summary.get("inci_tokens") or Counter(),
        list(anchor_sig.get("inci_tokens") or []),
        total,
    )
    out["candidate_anchor_active_conflict_rate"] = _conflict_rate(summary, anchor_sig)
    return out


def build_chain_transition_features(
    *,
    rules_chain: list[str] | tuple[str, ...] | None,
    candidate_type: str | None,
    anchor_product_type: str | None,
    last1_product_type: str | None,
    last2_product_type: str | None = None,
) -> dict[str, Any]:
    chain = [slug_token(item, default="") for item in (rules_chain or []) if slug_token(item, default="")]
    pos_map = {token: idx for idx, token in enumerate(chain)}

    def _pos(token: str | None) -> int:
        normalized = slug_token(token, default="")
        if not normalized:
            return -1
        return int(pos_map.get(normalized, -1))

    candidate = slug_token(candidate_type, default="")
    candidate_pos = _pos(candidate)
    anchor_pos = _pos(anchor_product_type)
    last1_pos = _pos(last1_product_type)
    last2_pos = _pos(last2_product_type)

    def _distance(from_pos: int, to_pos: int) -> int:
        if from_pos < 0 or to_pos < 0:
            return -99
        return int(to_pos - from_pos)

    dist_anchor = _distance(anchor_pos, candidate_pos)
    dist_last1 = _distance(last1_pos, candidate_pos)
    dist_last2 = _distance(last2_pos, candidate_pos)
    last_progression = _distance(last2_pos, last1_pos)

    out = {col: 0 for col in CHAIN_TRANSITION_NUMERIC_FEATURES}
    out["anchor_position_in_chain"] = int(anchor_pos)
    out["last1_position_in_chain"] = int(last1_pos)
    out["last2_position_in_chain"] = int(last2_pos)
    out["candidate_distance_from_anchor"] = int(dist_anchor)
    out["candidate_distance_from_last1"] = int(dist_last1)
    out["candidate_distance_from_last2"] = int(dist_last2)
    out["candidate_abs_distance_from_anchor"] = int(abs(dist_anchor)) if dist_anchor != -99 else 99
    out["candidate_abs_distance_from_last1"] = int(abs(dist_last1)) if dist_last1 != -99 else 99
    out["candidate_abs_distance_from_last2"] = int(abs(dist_last2)) if dist_last2 != -99 else 99
    out["candidate_is_same_as_anchor"] = int(dist_anchor == 0)
    out["candidate_is_after_anchor"] = int(dist_anchor > 0)
    out["candidate_is_before_anchor"] = int(dist_anchor < 0 and dist_anchor != -99)
    out["candidate_is_immediate_followup_to_anchor"] = int(dist_anchor == 1)
    out["candidate_is_immediate_predecessor_to_anchor"] = int(dist_anchor == -1)
    out["candidate_is_same_as_last1"] = int(dist_last1 == 0)
    out["candidate_is_after_last1"] = int(dist_last1 > 0)
    out["candidate_is_before_last1"] = int(dist_last1 < 0 and dist_last1 != -99)
    out["candidate_is_immediate_followup_to_last1"] = int(dist_last1 == 1)
    out["candidate_is_immediate_predecessor_to_last1"] = int(dist_last1 == -1)
    out["last1_after_last2_in_chain"] = int(last_progression > 0)
    out["last1_before_last2_in_chain"] = int(last_progression < 0 and last_progression != -99)

    if last_progression > 0 and dist_last1 > 0:
        out["candidate_continues_last_transition_direction"] = 1
    elif last_progression < 0 and dist_last1 < 0 and dist_last1 != -99:
        out["candidate_continues_last_transition_direction"] = 1
    if last_progression > 0 and dist_last1 < 0 and dist_last1 != -99:
        out["candidate_reverses_last_transition_direction"] = 1
    elif last_progression < 0 and dist_last1 > 0:
        out["candidate_reverses_last_transition_direction"] = 1
    return out


def build_nextstep_plan_state_features(
    *,
    rules_chain: list[str] | tuple[str, ...] | None,
    candidate_type: str | None,
    planned_target_product_type: str | None,
    planned_target_step_index: int | None = None,
) -> dict[str, Any]:
    chain = [slug_token(item, default="") for item in (rules_chain or []) if slug_token(item, default="")]
    pos_map = {token: idx for idx, token in enumerate(chain)}

    candidate = slug_token(candidate_type, default="")
    planned_target = slug_token(planned_target_product_type, default="")

    candidate_pos = int(pos_map.get(candidate, -1)) if candidate else -1
    target_pos = int(pos_map.get(planned_target, -1)) if planned_target else -1

    if target_pos >= 0 and candidate_pos >= 0:
        dist = int(candidate_pos - target_pos)
    else:
        dist = -99

    out = {col: "__none__" for col in NEXTSTEP_PLAN_STATE_CATEGORICAL_FEATURES}
    out.update({col: 0 for col in NEXTSTEP_PLAN_STATE_NUMERIC_FEATURES})

    out["planned_target_product_type"] = planned_target or "__none__"
    out["planned_target_step_index"] = int(planned_target_step_index or 0)
    out["planned_target_position_in_chain"] = int(target_pos)
    out["candidate_matches_planned_target"] = int(dist == 0)
    out["candidate_distance_from_planned_target"] = int(dist)
    out["candidate_abs_distance_from_planned_target"] = int(abs(dist)) if dist != -99 else 99
    out["candidate_is_after_planned_target"] = int(dist > 0)
    out["candidate_is_before_planned_target"] = int(dist < 0 and dist != -99)
    out["candidate_is_immediate_followup_to_planned_target"] = int(dist == 1)
    out["candidate_is_immediate_predecessor_to_planned_target"] = int(dist == -1)
    return out
