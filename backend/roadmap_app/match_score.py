"""Honest match-percent for a recommended product vs. a user profile.

The score is a weighted average of category-specific facets, normalised to
0..100. Avoid-flag hits are a hard zero. Missing data on either side drops the
facet out of the average instead of penalising the user for an empty field.
"""
from __future__ import annotations

from decimal import Decimal, InvalidOperation
from typing import Iterable

from catalog.models import Product
from users_app.models import CustomerProfile


Facet = tuple[float, float | None]


def _norm_set(values: Iterable | None) -> set[str]:
    if not values:
        return set()
    out: set[str] = set()
    for v in values:
        if v is None:
            continue
        s = str(v).strip().lower()
        if s:
            out.add(s)
    return out


def _norm_str(value) -> str:
    if value is None:
        return ""
    return str(value).strip().lower()


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = a & b
    union = a | b
    return len(inter) / len(union) if union else 0.0


def _overlap_share(profile_set: set[str], product_set: set[str]) -> float:
    """How much of the user's wishlist the product covers."""
    if not profile_set or not product_set:
        return 0.0
    inter = profile_set & product_set
    return len(inter) / len(profile_set)


def _budget_band(budget: str) -> tuple[Decimal | None, Decimal | None]:
    b = _norm_str(budget)
    if b == CustomerProfile.Budget.LOW:
        return Decimal("0"), Decimal("20000")
    if b == CustomerProfile.Budget.MEDIUM:
        return Decimal("20000"), Decimal("50000")
    if b == CustomerProfile.Budget.HIGH:
        return Decimal("50000"), None
    return None, None


def _budget_facet(profile: CustomerProfile, product: Product) -> Facet:
    price = product.price
    if price is None:
        return (0.15, None)
    try:
        price_dec = Decimal(price) if not isinstance(price, Decimal) else price
    except (InvalidOperation, TypeError):
        return (0.15, None)

    low, high = _budget_band(profile.budget)
    if low is None and high is None:
        return (0.15, None)

    if (low is None or price_dec >= low) and (high is None or price_dec <= high):
        return (0.15, 1.0)

    # Off-band: scale by how far we miss.
    if high is not None and price_dec > high:
        overshoot = float(price_dec - high) / float(high if high > 0 else Decimal("1"))
        score = max(0.0, 1.0 - overshoot)
        return (0.15, score)
    if low is not None and price_dec < low:
        # Cheaper than band — soft positive, not a real miss.
        return (0.15, 0.8)
    return (0.15, 0.5)


def _skincare_facets(profile: CustomerProfile, product: Product) -> list[Facet]:
    facets: list[Facet] = []

    supported = _norm_set(product.supported_skin_types)
    skin = _norm_str(profile.skin_type)
    if not supported or "all" in supported:
        facets.append((0.30, 1.0))
    elif skin and skin in supported:
        facets.append((0.30, 1.0))
    elif skin:
        facets.append((0.30, 0.3))
    else:
        facets.append((0.30, None))

    goals = _norm_set(profile.goals)
    concerns = _norm_set(product.concerns)
    if goals and concerns:
        facets.append((0.40, _overlap_share(goals, concerns)))
    else:
        facets.append((0.40, None))

    return facets


def _haircare_facets(profile: CustomerProfile, product: Product) -> list[Facet]:
    facets: list[Facet] = []
    attrs = product.attrs or {}
    hair = profile.hair_profile or {}

    matches: list[float] = []
    for key in ("hair_type", "scalp_type", "hair_thickness"):
        p_val = _norm_str(hair.get(key))
        a_val = _norm_str(attrs.get(key))
        if p_val and a_val:
            matches.append(1.0 if p_val == a_val else 0.0)
    if matches:
        facets.append((0.45, sum(matches) / len(matches)))
    else:
        facets.append((0.45, None))

    hair_concerns = _norm_set(hair.get("concerns"))
    product_concerns = _norm_set(product.concerns)
    if hair_concerns and product_concerns:
        facets.append((0.40, _overlap_share(hair_concerns, product_concerns)))
    else:
        facets.append((0.40, None))

    return facets


def _makeup_facets(profile: CustomerProfile, product: Product) -> list[Facet]:
    facets: list[Facet] = []
    attrs = product.attrs or {}
    makeup = profile.makeup_profile or {}

    finish_pref = _norm_set(makeup.get("finish_pref"))
    finish = _norm_str(attrs.get("finish"))
    if finish_pref and finish:
        facets.append((0.35, 1.0 if finish in finish_pref else 0.2))
    else:
        facets.append((0.35, None))

    coverage_pref = _norm_set(makeup.get("coverage_pref"))
    coverage = _norm_str(attrs.get("coverage"))
    if coverage_pref and coverage:
        facets.append((0.30, 1.0 if coverage in coverage_pref else 0.2))
    else:
        facets.append((0.30, None))

    undertone_pref = _norm_str(makeup.get("undertone"))
    undertone = _norm_str(attrs.get("tone_family")) or _norm_str(attrs.get("undertone"))
    if undertone_pref and undertone:
        facets.append((0.20, 1.0 if undertone == undertone_pref else 0.3))
    else:
        facets.append((0.20, None))

    return facets


def _fragrance_facets(profile: CustomerProfile, product: Product) -> list[Facet]:
    facets: list[Facet] = []
    attrs = product.attrs or {}
    fragrance = profile.fragrance_profile or {}

    liked_families = _norm_set(fragrance.get("liked_families"))
    family = _norm_str(attrs.get("scent_family"))
    if liked_families and family:
        facets.append((0.40, 1.0 if family in liked_families else 0.0))
    else:
        facets.append((0.40, None))

    liked_notes = _norm_set(fragrance.get("liked_notes"))
    notes = _norm_set(attrs.get("notes"))
    if liked_notes and notes:
        facets.append((0.30, _overlap_share(liked_notes, notes)))
    else:
        facets.append((0.30, None))

    intensity_pref = _norm_str(fragrance.get("intensity_pref"))
    intensity = _norm_str(attrs.get("intensity"))
    if intensity_pref and intensity:
        facets.append((0.15, 1.0 if intensity == intensity_pref else 0.4))
    else:
        facets.append((0.15, None))

    return facets


def compute_match_percent(
    profile: CustomerProfile | None,
    product: Product | None,
    *,
    category: str | None = None,
) -> int | None:
    """Return 0..100 match-percent for *product* against *profile*.

    ``None`` means we cannot compute a meaningful score (no profile, no product,
    or no overlapping fields). The caller decides what to render.
    """
    if profile is None or product is None:
        return None

    avoid = _norm_set(profile.avoid_flags)
    product_flags = _norm_set(product.flags)
    if avoid and product_flags and avoid & product_flags:
        return 0

    resolved_category = _norm_str(category) or _norm_str(getattr(product, "category", None)) or "skincare"

    if resolved_category == "haircare":
        facets = _haircare_facets(profile, product)
    elif resolved_category == "makeup":
        facets = _makeup_facets(profile, product)
    elif resolved_category == "fragrance":
        facets = _fragrance_facets(profile, product)
    else:
        facets = _skincare_facets(profile, product)

    facets.append(_budget_facet(profile, product))

    total_weight = 0.0
    weighted_sum = 0.0
    for weight, score in facets:
        if score is None:
            continue
        total_weight += weight
        weighted_sum += weight * score

    if total_weight <= 0:
        return None

    return max(0, min(100, round(weighted_sum / total_weight * 100)))
