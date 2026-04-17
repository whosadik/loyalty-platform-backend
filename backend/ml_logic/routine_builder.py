from dataclasses import asdict, dataclass
from typing import Any

from . import routine_scorer


AM_STEPS = ["cleanser", "moisturizer", "spf"]
PM_STEPS = ["cleanser", "serum", "moisturizer"]  # serum = treatment/active на MVP


@dataclass
class Profile:
    skin_type: str
    goals: list[str]
    avoid_flags: list[str]
    budget: str


def _profile_dict(profile: Profile) -> dict[str, Any]:
    return asdict(profile)


def _rule_score(product: dict[str, Any], profile: Profile) -> float:
    """Score a product by how well it matches the user profile. Higher is better."""
    score = 0.0

    # Primary differentiator: product concerns matching user goals
    p_concerns = set(product.get("concerns") or [])
    for goal in (profile.goals or []):
        if goal in p_concerns:
            score += 10.0

    # Bonus for strength level matching active-ingredient goals
    strength = str(product.get("strength") or "").lower()
    active_goals = {"anti_aging", "brightening", "anti_acne", "acne", "wrinkles", "hyperpigmentation"}
    if strength == "strong" and set(profile.goals or []) & active_goals:
        score += 3.0
    elif strength == "medium":
        score += 1.0

    # Small tiebreaker: prefer cheaper products
    price = product.get("price")
    if price is not None:
        score -= float(price) / 100_000.0

    return score


def _score_candidates(
    candidates: list[dict[str, Any]],
    *,
    profile: Profile,
    step: str,
    context: dict[str, Any] | None = None,
) -> tuple[list[float], str]:
    """Return (scores, source). `source` is 'ml' when the ML ranker was used,
    otherwise 'rules'. Falls back to rule-based scoring automatically.
    """
    if candidates:
        ml_scores = routine_scorer.score_candidates(
            profile=_profile_dict(profile),
            step=step,
            candidates=candidates,
            context=context,
        )
        if ml_scores is not None and len(ml_scores) == len(candidates):
            return ml_scores, "ml"

    return [_rule_score(p, profile) for p in candidates], "rules"


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


def _sort_by_scores(
    items: list[dict[str, Any]],
    scores: list[float],
) -> list[dict[str, Any]]:
    pairs = list(zip(items, scores))
    pairs.sort(key=lambda pair: pair[1], reverse=True)
    return [item for item, _ in pairs]


def build_routine(
    profile: Profile,
    products: list[dict[str, Any]],
    top_k: int = 3,
    owned_product_ids: list[int] | None = None,
    ml_context: dict[str, Any] | None = None,
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

        if candidates:
            scores, scorer = _score_candidates(
                candidates, profile=profile, step=step, context=ml_context
            )
            ordered = _sort_by_scores(candidates, scores)

            owned_set = set(owned_product_ids or [])
            owned_ordered = [p for p in ordered if p.get("id") in owned_set]

            if owned_ordered:
                chosen = owned_ordered[0]
                return {
                    "step": step,
                    "status": "filled",
                    "source": "owned",
                    "scorer": scorer,
                    "product": chosen,
                    "why": [
                        "already owned by user",
                        f"matches skin_type={profile.skin_type}",
                        "no avoided ingredients/flags",
                    ],
                    "suggestions": [c["id"] for c in ordered[:top_k]],
                }

            chosen = ordered[0]
            return {
                "step": step,
                "status": "filled",
                "source": "recommended",
                "scorer": scorer,
                "product": chosen,
                "why": [
                    f"matches skin_type={profile.skin_type}",
                    "no avoided ingredients/flags",
                ],
                "suggestions": [c["id"] for c in ordered[:top_k]],
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
            "scorer": "rules",
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
