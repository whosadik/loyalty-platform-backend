from __future__ import annotations

from dataclasses import dataclass
from typing import Any


# Консервативные конфликты для MVP
CONFLICT_PAIRS = {
    ("retinoid", "aha"),
    ("retinoid", "bha"),
    ("benzoyl_peroxide", "retinoid"),
    ("vitamin_c", "aha"),
    ("vitamin_c", "bha"),
}

ACTIVE_GROUP_STRONG = {"aha", "bha", "retinoid", "vitamin_c", "benzoyl_peroxide"}


def _pairs(actives: list[str]) -> set[tuple[str, str]]:
    s = set(actives)
    out = set()
    for a in s:
        for b in s:
            if a == b:
                continue
            out.add(tuple(sorted((a, b))))
    return out


def detect_conflicts(all_actives: list[str]) -> list[dict[str, Any]]:
    """
    Возвращает список конфликтов по активам, если они есть.
    """
    conflicts = []
    for a, b in sorted(_pairs(all_actives)):
        if (a, b) in {tuple(sorted(p)) for p in CONFLICT_PAIRS}:
            conflicts.append(
                {
                    "type": "active_conflict",
                    "pair": [a, b],
                    "message": f"Potential conflict between actives: {a} + {b} in the same routine.",
                }
            )

    # Доп. правило: не больше одного "сильного" актива за раз (для PM)
    strong = [a for a in set(all_actives) if a in ACTIVE_GROUP_STRONG]
    if len(strong) >= 2:
        conflicts.append(
            {
                "type": "too_many_strong_actives",
                "actives": strong,
                "message": "Too many strong actives in one routine. Consider using only one per evening.",
            }
        )

    return conflicts
