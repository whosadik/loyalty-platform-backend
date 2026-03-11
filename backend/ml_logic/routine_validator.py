from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .routine_builder import Profile, _fits_profile
from .routine_rules import detect_conflicts


def _matches_step(product: dict[str, Any], step: str) -> bool:
    return product.get("product_type") == step or product.get("step") == step


def validate_routine(
    profile: Profile,
    products: list[dict[str, Any]],
    routine: dict[str, Any],
    top_k: int = 3,
) -> dict[str, Any]:
    """
    routine:
    {
      "am": [{"step":"cleanser","product_id":1}, ...],
      "pm": [{"step":"serum","product_id":5}, ...]
    }
    """
    by_id = {p["id"]: p for p in products}

    def get_actives(item: dict[str, Any]) -> list[str]:
        pid = item.get("product_id")
        if not pid:
            return []
        p = by_id.get(pid)
        if not p:
            return []
        return p.get("actives") or []

    # собираем активы в PM (там основное)
    pm_actives: list[str] = []
    for item in routine.get("pm", []):
        pm_actives.extend(get_actives(item))

    conflicts = detect_conflicts(pm_actives)

    # если конфликт — предлагаем замены для проблемных шагов (чаще всего serum)
    suggestions: list[dict[str, Any]] = []
    if conflicts:
        for item in routine.get("pm", []):
            step = item.get("step")
            pid = item.get("product_id")
            if not pid:
                continue

            current = by_id.get(pid)
            if not current:
                continue

            # на MVP предлагаем замены для serum/treatment
            if step not in {"serum", "toner", "mask"}:
                continue

            candidates = [p for p in products if _matches_step(p, step) and _fits_profile(p, profile)]
            candidates.sort(key=lambda x: (x.get("price") is None, x.get("price", 0)))

            # убираем кандидатов, которые сохраняют конфликт (упрощённо: убираем те же активы)
            current_actives = set(current.get("actives") or [])
            filtered = []
            for c in candidates:
                c_actives = set(c.get("actives") or [])
                # замена должна уменьшать "сильные" активы
                if len(c_actives.intersection(current_actives)) > 0 and len(current_actives) > 0:
                    continue
                filtered.append(c)

            suggestions.append(
                {
                    "step": step,
                    "current_product_id": pid,
                    "alternatives": [c["id"] for c in filtered[:top_k]],
                }
            )

    return {
        "conflicts": conflicts,
        "suggestions": suggestions,
        "is_valid": len(conflicts) == 0,
    }
