from dataclasses import dataclass
from typing import Any


AM_STEPS = ["cleanser", "moisturizer", "spf"]
PM_STEPS = ["cleanser", "serum", "moisturizer"]  # serum = treatment/active на MVP


@dataclass
class Profile:
    skin_type: str
    goals: list[str]
    avoid_flags: list[str]
    budget: str


def _fits_profile(product: dict[str, Any], profile: Profile) -> bool:
    # skin type filter
    if profile.skin_type and profile.skin_type not in (product.get("supported_skin_types") or []):
        return False

    # avoid flags
    p_flags = set(product.get("flags") or [])
    if p_flags.intersection(profile.avoid_flags):
        return False

    # stock
    if product.get("in_stock") is False:
        return False

    return True


def build_routine(
    profile: Profile,
    products: list[dict[str, Any]],
    top_k: int = 3,
) -> dict[str, Any]:
    # группируем по шагам
    by_step: dict[str, list[dict[str, Any]]] = {}
    for p in products:
        by_step.setdefault(p["step"], []).append(p)

    def pick_for_step(step: str) -> dict[str, Any]:
        candidates = [p for p in by_step.get(step, []) if _fits_profile(p, profile)]
        # простая сортировка: дешевле выше (потом улучшим)
        candidates.sort(key=lambda x: (x.get("price") is None, x.get("price", 0)))

        if candidates:
            chosen = candidates[0]
            return {
                "step": step,
                "status": "filled",
                "product": chosen,
                "why": [
                    f"matches skin_type={profile.skin_type}",
                    "no avoided ingredients/flags",
                ],
                "suggestions": [c["id"] for c in candidates[:top_k]],
            }

        # если не нашли — предложим что есть (без фильтра skin_type) но с avoid_flags
        fallback = [p for p in by_step.get(step, []) if not set(p.get("flags") or []).intersection(profile.avoid_flags)]
        fallback.sort(key=lambda x: (x.get("price") is None, x.get("price", 0)))

        return {
            "step": step,
            "status": "missing",
            "product": None,
            "why": [f"no products found for skin_type={profile.skin_type} with current constraints"],
            "suggestions": [c["id"] for c in fallback[:top_k]],
        }

    am = [pick_for_step(s) for s in AM_STEPS]
    pm = [pick_for_step(s) for s in PM_STEPS]

    notes = []
    # простое правило: если в PM есть активы, напомнить про SPF
    pm_actives = []
    for item in pm:
        prod = item.get("product") or {}
        pm_actives.extend(prod.get("actives") or [])
    if any(a in pm_actives for a in ["aha", "bha", "retinoid", "vitamin_c"]):
        notes.append("Consider SPF in the morning when using active ingredients.")

    return {"am": am, "pm": pm, "notes": notes}
