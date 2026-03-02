from __future__ import annotations

import re
from typing import Any


SLOTS = ["warm_day", "warm_evening", "cold_day", "cold_evening"]

_WARM_FAMILIES = {"citrus", "aquatic", "fresh", "green", "floral"}
_COLD_FAMILIES = {"oriental", "amber", "woody", "spicy", "leather", "gourmand", "chypre"}

_WARM_NOTES = {
    "bergamot",
    "lemon",
    "citrus",
    "neroli",
    "orange_blossom",
    "mint",
    "tea",
    "marine",
    "aquatic",
}
_COLD_NOTES = {
    "vanilla",
    "tonka",
    "oud",
    "patchouli",
    "amber",
    "incense",
    "leather",
    "spice",
    "cinnamon",
    "clove",
    "tobacco",
}

_SOFT_INTENSITY = {"soft", "light", "mild", "subtle", "gentle", "airy"}
_MODERATE_INTENSITY = {"moderate", "medium", "balanced", "normal", "everyday"}
_STRONG_INTENSITY = {"strong", "intense", "heavy", "bold", "rich", "powerful"}


def _norm_token(raw: Any) -> str:
    s = str(raw or "").strip().lower()
    if not s:
        return ""
    s = s.replace("-", "_").replace(" ", "_")
    s = re.sub(r"[^a-z0-9_]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s


def normalize_intensity(raw: Any) -> str:
    token = _norm_token(raw)
    if token in _SOFT_INTENSITY:
        return "soft"
    if token in _STRONG_INTENSITY:
        return "strong"
    if token in _MODERATE_INTENSITY:
        return "moderate"
    if "strong" in token or "intense" in token:
        return "strong"
    if "light" in token or "soft" in token:
        return "soft"
    return "moderate"


def normalize_scent_family(raw: Any) -> str:
    token = _norm_token(raw)
    if not token:
        return ""
    if token in _WARM_FAMILIES or token in _COLD_FAMILIES:
        return token
    # Common aliases
    aliases = {
        "ambery": "amber",
        "wood": "woody",
        "fresh_aquatic": "aquatic",
        "fresh_citrus": "citrus",
        "oriental_spicy": "oriental",
    }
    return aliases.get(token, token)


def normalize_notes(raw_notes: Any) -> set[str]:
    if raw_notes is None:
        return set()
    if isinstance(raw_notes, str):
        chunks = re.split(r"[;,/|]+", raw_notes)
        raw_list = chunks if chunks else [raw_notes]
    elif isinstance(raw_notes, (list, tuple, set)):
        raw_list = list(raw_notes)
    else:
        raw_list = [raw_notes]

    out: set[str] = set()
    for x in raw_list:
        token = _norm_token(x)
        if token:
            out.add(token)
    return out


def _temperature_score(family: str, notes: set[str]) -> int:
    score = 0
    if family in _WARM_FAMILIES:
        score += 2
    elif family in _COLD_FAMILIES:
        score -= 2

    for n in notes:
        if n in _WARM_NOTES:
            score += 1
        if n in _COLD_NOTES:
            score -= 1
    return score


def slot_of_fragrance(attrs: dict, raw_meta: dict | None = None) -> str:
    attrs = attrs or {}
    raw_meta = raw_meta or {}

    family = normalize_scent_family(attrs.get("scent_family") or raw_meta.get("scent_family"))
    notes = normalize_notes(attrs.get("notes") or raw_meta.get("notes") or raw_meta.get("note_list"))
    intensity = normalize_intensity(attrs.get("intensity") or raw_meta.get("intensity"))

    temp_score = _temperature_score(family, notes)
    temp = "warm" if temp_score >= 0 else "cold"

    if intensity == "strong":
        daypart = "evening"
    elif intensity == "soft":
        daypart = "day"
    else:
        daypart = "evening" if temp_score <= -1 else "day"

    slot = f"{temp}_{daypart}"
    if slot not in SLOTS:
        return "warm_day"
    return slot


def slot_features(attrs: dict) -> dict[str, Any]:
    attrs = attrs or {}
    family = normalize_scent_family(attrs.get("scent_family"))
    notes = normalize_notes(attrs.get("notes"))
    intensity = normalize_intensity(attrs.get("intensity"))
    temp_score = _temperature_score(family, notes)
    slot = slot_of_fragrance(attrs)
    temp, daypart = slot.split("_", 1)
    return {
        "slot": slot,
        "temp": temp,
        "daypart": daypart,
        "temp_score": temp_score,
        "intensity": intensity,
        "scent_family": family,
    }
