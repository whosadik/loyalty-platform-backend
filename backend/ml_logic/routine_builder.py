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
    # skin type filter:
    # пустой supported_skin_types => подходит всем
    supported = product.get("supported_skin_types") or []
    if supported and profile.skin_type and profile.skin_type not in supported:
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
    owned_product_ids: list[int] | None = None,
) -> dict[str, Any]:
    # routine строим только из skincare
    products = [p for p in products if p.get("category") == "skincare"]

    # группируем по product_type (fallback на step для старых данных)
    by_step: dict[str, list[dict[str, Any]]] = {}
    for p in products:
        key = p.get("product_type") or p.get("step")
        if not key:
            continue
        by_step.setdefault(key, []).append(p)

    def pick_for_step(step: str) -> dict[str, Any]:
        candidates = [p for p in by_step.get(step, []) if _fits_profile(p, profile)]

        owned_set = set(owned_product_ids or [])
        owned_candidates = [p for p in candidates if p.get("id") in owned_set]
        if owned_candidates:
            chosen = owned_candidates[0]
            return {
                "step": step,
                "status": "filled",
                "source": "owned",
                "product": chosen,
                "why": [
                    "already owned by user",
                    f"matches skin_type={profile.skin_type}",
                    "no avoided ingredients/flags",
                ],
                "suggestions": [c["id"] for c in candidates[:top_k]],
            }

        # простая сортировка: дешевле выше (потом улучшим)
        candidates.sort(key=lambda x: (x.get("price") is None, x.get("price", 0)))

        if candidates:
            chosen = candidates[0]
            return {
                "step": step,
                "status": "filled",
                "source": "recommended",
                "product": chosen,
                "why": [
                    f"matches skin_type={profile.skin_type}",
                    "no avoided ingredients/flags",
                ],
                "suggestions": [c["id"] for c in candidates[:top_k]],
            }

        fallback = [
            p for p in by_step.get(step, [])
            if not set(p.get("flags") or []).intersection(profile.avoid_flags)
        ]
        fallback.sort(key=lambda x: (x.get("price") is None, x.get("price", 0)))

        return {
            "step": step,
            "status": "missing",
            "source": "recommended",
            "product": None,
            "why": [f"no products found for skin_type={profile.skin_type} with current constraints"],
            "suggestions": [c["id"] for c in fallback[:top_k]],
        }

    am = [pick_for_step(s) for s in AM_STEPS]
    pm = [pick_for_step(s) for s in PM_STEPS]

    notes = []
    pm_actives = []
    for item in pm:
        prod = item.get("product") or {}
        pm_actives.extend(prod.get("actives") or [])
    if any(a in pm_actives for a in ["aha", "bha", "retinoid", "vitamin_c"]):
        notes.append("Consider SPF in the morning when using active ingredients.")

    return {"am": am, "pm": pm, "notes": notes}
