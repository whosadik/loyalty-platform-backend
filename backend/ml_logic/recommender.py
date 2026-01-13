from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from collections import defaultdict


@dataclass
class UserProfile:
    skin_type: str
    goals: list[str]
    avoid_flags: list[str]
    budget: str
    hair: dict[str, Any]
    makeup: dict[str, Any]
    fragrance: dict[str, Any]


def _budget_price_cap(budget: str) -> float | None:
    # MVP caps: можно менять
    if budget == "low":
        return 15.0
    if budget == "medium":
        return 30.0
    return None  # high = без капа


def _passes_global_filters(p: dict[str, Any], prof: UserProfile) -> bool:
    if p.get("in_stock") is False:
        return False

    # avoid flags
    pf = set(p.get("flags") or [])
    if pf.intersection(set(prof.avoid_flags or [])):
        return False

    # budget cap (если цена есть)
    cap = _budget_price_cap(prof.budget)
    price = p.get("price")
    if cap is not None and price is not None:
        try:
            if float(price) > cap:
                return False
        except Exception:
            pass

    return True


def _skincare_score(p: dict[str, Any], prof: UserProfile) -> tuple[float, list[str]]:
    score = 0.0
    why = []

    # skin type match: пустой список = подходит всем
    supported = p.get("supported_skin_types") or []
    if supported:
        if prof.skin_type and prof.skin_type in supported:
            score += 1.0
            why.append(f"matches skin_type={prof.skin_type}")
        else:
            return -1e9, ["skin_type mismatch"]

    # concerns vs goals
    concerns = set(p.get("concerns") or [])
    goals = set(prof.goals or [])
    inter = concerns.intersection(goals)
    if inter:
        score += 0.6 * len(inter)
        why.append(f"matches goals: {sorted(list(inter))}")

    # actives preference (минимально)
    actives = set(p.get("actives") or [])
    if actives:
        score += 0.1 * len(actives)
        why.append("has actives")

    return score, why


def _haircare_score(p: dict[str, Any], prof: UserProfile) -> tuple[float, list[str]]:
    score = 0.0
    why = []
    hp = prof.hair or {}
    attrs = p.get("attrs") or {}

    if hp.get("hair_type") and attrs.get("hair_type") == hp["hair_type"]:
        score += 1.0
        why.append(f"matches hair_type={hp['hair_type']}")
    if hp.get("scalp_type") and attrs.get("scalp_type") == hp["scalp_type"]:
        score += 0.8
        why.append(f"matches scalp_type={hp['scalp_type']}")
    if hp.get("hair_thickness") and attrs.get("hair_thickness") == hp["hair_thickness"]:
        score += 0.5
        why.append(f"matches hair_thickness={hp['hair_thickness']}")

    concerns = set(p.get("concerns") or [])
    user_concerns = set((hp.get("concerns") or []))
    inter = concerns.intersection(user_concerns)
    if inter:
        score += 0.8 * len(inter)
        why.append(f"matches hair concerns: {sorted(list(inter))}")

    return score, why


def _makeup_score(p: dict[str, Any], prof: UserProfile) -> tuple[float, list[str]]:
    score = 0.0
    why = []
    mp = prof.makeup or {}
    attrs = p.get("attrs") or {}

    finish_pref = set(mp.get("finish_pref") or [])
    coverage_pref = set(mp.get("coverage_pref") or [])

    if finish_pref and attrs.get("finish") and attrs["finish"] in finish_pref:
        score += 0.8
        why.append(f"matches finish={attrs['finish']}")
    if coverage_pref and attrs.get("coverage") and attrs["coverage"] in coverage_pref:
        score += 0.8
        why.append(f"matches coverage={attrs['coverage']}")

    if mp.get("undertone") and attrs.get("undertone") and attrs["undertone"] == mp["undertone"]:
        score += 0.6
        why.append(f"matches undertone={mp['undertone']}")
    if mp.get("tone_family") and attrs.get("tone_family") and attrs["tone_family"] == mp["tone_family"]:
        score += 0.6
        why.append(f"matches tone_family={mp['tone_family']}")

    # concerns: long_wear/waterproof etc.
    concerns = set(p.get("concerns") or [])
    user_concerns = set(mp.get("concerns") or [])
    inter = concerns.intersection(user_concerns)
    if inter:
        score += 0.6 * len(inter)
        why.append(f"matches makeup concerns: {sorted(list(inter))}")

    # sensitive eyes tweak
    if mp.get("sensitive_eyes") and p.get("product_type") == "mascara":
        score += 0.2
        why.append("user has sensitive_eyes (mascara boosted)")

    return score, why


def _fragrance_score(p: dict[str, Any], prof: UserProfile) -> tuple[float, list[str]]:
    score = 0.0
    why = []
    fp = prof.fragrance or {}
    attrs = p.get("attrs") or {}

    liked_families = set(fp.get("liked_families") or [])
    disliked_families = set(fp.get("disliked_families") or [])
    family = attrs.get("scent_family")

    if family:
        if family in disliked_families:
            return -1e9, [f"disliked scent_family={family}"]
        if family in liked_families:
            score += 1.2
            why.append(f"matches scent_family={family}")

    if fp.get("intensity_pref") and attrs.get("intensity") == fp["intensity_pref"]:
        score += 0.6
        why.append(f"matches intensity={fp['intensity_pref']}")

    notes = set(attrs.get("notes") or [])
    liked_notes = set(fp.get("liked_notes") or [])
    avoid_notes = set(fp.get("avoid_notes") or [])
    if notes.intersection(avoid_notes):
        return -1e9, ["contains avoided notes"]

    inter = notes.intersection(liked_notes)
    if inter:
        score += 0.3 * len(inter)
        why.append(f"matches notes: {sorted(list(inter))}")

    return score, why


def content_score(p: dict[str, Any], prof: UserProfile) -> tuple[float, list[str]]:
    cat = p.get("category")
    if cat == "skincare":
        return _skincare_score(p, prof)
    if cat == "haircare":
        return _haircare_score(p, prof)
    if cat == "makeup":
        return _makeup_score(p, prof)
    if cat == "fragrance":
        return _fragrance_score(p, prof)
    return 0.0, []


def build_cooccurrence(txn_product_lists: list[list[int]]) -> dict[int, dict[int, int]]:
    co = defaultdict(lambda: defaultdict(int))
    for items in txn_product_lists:
        uniq = list(dict.fromkeys(items))  # unique, stable
        n = len(uniq)
        for i in range(n):
            a = uniq[i]
            for j in range(i + 1, n):
                b = uniq[j]
                co[a][b] += 1
                co[b][a] += 1
    return co


def cooc_score(candidate_id: int, context_ids: list[int], co: dict[int, dict[int, int]]) -> tuple[float, int]:
    s = 0
    for cid in context_ids:
        s += int(co.get(cid, {}).get(candidate_id, 0))
    return float(s), int(s)


def recommend(
    prof: UserProfile,
    products: list[dict[str, Any]],
    owned_active_ids: list[int],
    context_product_ids: list[int],
    category: str,
    product_type: str | None,
    limit: int = 10,
    co: dict[int, dict[int, int]] | None = None,
) -> list[dict[str, Any]]:
    owned_set = set(owned_active_ids or [])
    co = co or {}

    candidates = []
    for p in products:
        if p.get("category") != category:
            continue
        if product_type and p.get("product_type") != product_type:
            continue

        if p["id"] in owned_set:
            continue

        if not _passes_global_filters(p, prof):
            continue

        c_score, why_c = content_score(p, prof)
        if c_score < -1e8:
            continue

        co_score, co_raw = cooc_score(p["id"], context_product_ids or [], co)

        # normalisation / weights
        # content dominates; cooccurrence adds “signal”
        total = (0.6 * c_score) + (0.4 * min(co_score, 10.0) / 10.0)

        why = []
        if why_c:
            why.extend(why_c)
        if co_raw > 0:
            why.append(f"often bought together with your items (count={co_raw})")

        candidates.append(
            {
                "product": p,
                "score": round(total, 6),
                "components": {
                    "content": round(c_score, 6),
                    "cooccurrence": co_raw,
                },
                "why": why[:6],
            }
        )

    candidates.sort(key=lambda x: x["score"], reverse=True)
    return candidates[:limit]


def bundle(
    products: list[dict[str, Any]],
    base_product_id: int,
    owned_active_ids: list[int],
    prof: UserProfile,
    co: dict[int, dict[int, int]],
    limit: int = 10,
) -> list[dict[str, Any]]:
    owned_set = set(owned_active_ids or [])
    by_id = {p["id"]: p for p in products}

    related = co.get(base_product_id, {})
    ranked = sorted(related.items(), key=lambda kv: kv[1], reverse=True)

    out = []
    for pid, cnt in ranked:
        if pid == base_product_id:
            continue
        if pid in owned_set:
            continue
        p = by_id.get(pid)
        if not p:
            continue
        if not _passes_global_filters(p, prof):
            continue

        out.append(
            {
                "product": p,
                "score": float(cnt),
                "components": {"cooccurrence": int(cnt)},
                "why": [f"frequently purchased with product_id={base_product_id} (count={cnt})"],
            }
        )
        if len(out) >= limit:
            break
    return out
