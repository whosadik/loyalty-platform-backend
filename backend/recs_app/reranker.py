from __future__ import annotations

import math
from functools import lru_cache
from pathlib import Path
from typing import Any

from django.conf import settings

from ml_logic.recommender import UserProfile, recommend as heuristic_recommend


def _to_int(v: Any) -> int | None:
    try:
        return int(v)
    except Exception:
        return None


def _to_float(v: Any, default: float = 0.0) -> float:
    try:
        if v is None:
            return default
        return float(v)
    except Exception:
        return default


@lru_cache(maxsize=8)
def _load_model_cached(path_str: str, mtime_ns: int):
    del mtime_ns
    import joblib

    return joblib.load(path_str)


def _load_reranker_model():
    try:
        import joblib  # noqa: F401
    except Exception:
        return None, "joblib_unavailable"

    path = Path(str(getattr(settings, "RECS_RERANKER_MODEL_PATH", "") or "")).expanduser()
    if not path.exists():
        return None, "model_not_found"
    if not path.is_file():
        return None, "model_not_file"

    try:
        model = _load_model_cached(str(path.resolve()), int(path.stat().st_mtime_ns))
        return model, None
    except Exception:
        return None, "model_load_error"


def _context_ids(context_product_ids: list[int] | None, max_k: int) -> list[int]:
    vals: list[int] = []
    for raw in context_product_ids or []:
        v = _to_int(raw)
        if v is None:
            continue
        vals.append(v)
    dedup = list(dict.fromkeys(vals))
    k = max(1, int(max_k or 1))
    return dedup[-k:]


def _aggregate_transition_counts(context_ids: list[int], co: dict[int, dict[int, int]]) -> dict[int, float]:
    if not context_ids:
        return {}

    recency_weights = [1.0, 0.7, 0.5, 0.3]
    agg: dict[int, float] = {}

    for idx, ctx in enumerate(reversed(context_ids)):
        w = recency_weights[idx] if idx < len(recency_weights) else 0.2
        for cand, cnt in (co.get(int(ctx), {}) or {}).items():
            cid = _to_int(cand)
            if cid is None:
                continue
            agg[cid] = float(agg.get(cid, 0.0)) + (float(cnt or 0.0) * float(w))

    return agg


def _co_popularity(co: dict[int, dict[int, int]]) -> dict[int, float]:
    pop: dict[int, float] = {}
    for src, related in (co or {}).items():
        sid = _to_int(src)
        if sid is None:
            continue
        total = 0.0
        for _, cnt in (related or {}).items():
            total += float(cnt or 0.0)
        pop[sid] = total
    return pop


def recommend_with_algo(
    *,
    prof: UserProfile,
    products: list[dict[str, Any]],
    owned_active_ids: list[int],
    context_product_ids: list[int],
    category: str | None,
    product_type: str | None,
    limit: int,
    co: dict[int, dict[int, int]] | None,
    algo_requested: str | None,
) -> tuple[list[dict[str, Any]], str]:
    algo_req = (algo_requested or getattr(settings, "RECS_ALGO_DEFAULT", "cooc") or "cooc").strip().lower()
    if algo_req not in {"cooc", "reranker"}:
        algo_req = "cooc"

    top_m = max(int(getattr(settings, "RECS_RERANKER_TOP_M", 200) or 200), int(limit))
    pool_limit = max(
        int(getattr(settings, "RECS_RERANKER_HEURISTIC_POOL", 500) or 500),
        top_m,
        int(limit),
    )

    heuristic_results = heuristic_recommend(
        prof=prof,
        products=products,
        owned_active_ids=owned_active_ids,
        context_product_ids=context_product_ids,
        category=category,
        product_type=product_type,
        limit=pool_limit,
        co=co,
    )

    if algo_req != "reranker":
        return heuristic_results[:limit], "cooc"

    model, err = _load_reranker_model()
    if model is None:
        return heuristic_results[:limit], f"cooc_fallback:{err}"

    ctx_ids = _context_ids(context_product_ids, int(getattr(settings, "RECS_RERANKER_CONTEXT_K", 3) or 3))
    if not ctx_ids:
        return heuristic_results[:limit], "cooc_fallback:no_context"

    ctx_item = ctx_ids[-1]
    by_id = {}
    for p in products:
        pid = _to_int(p.get("id"))
        if pid is not None:
            by_id[pid] = p

    ctx = by_id.get(ctx_item)
    if not ctx:
        return heuristic_results[:limit], "cooc_fallback:no_context_item"

    transitions = _aggregate_transition_counts(ctx_ids, co or {})
    ranked_trans = sorted(transitions.items(), key=lambda x: (-x[1], x[0]))
    candidate_ids = [int(cid) for cid, _ in ranked_trans[:top_m]]

    heuristic_map: dict[int, dict[str, Any]] = {}
    for r in heuristic_results:
        pid = _to_int((r.get("product") or {}).get("id"))
        if pid is None:
            continue
        heuristic_map[pid] = r

    if len(candidate_ids) < top_m:
        for pid in heuristic_map.keys():
            if pid in candidate_ids:
                continue
            candidate_ids.append(pid)
            if len(candidate_ids) >= top_m:
                break

    candidate_ids = [pid for pid in candidate_ids if pid in heuristic_map]
    if not candidate_ids:
        return heuristic_results[:limit], "cooc_fallback:no_candidates"

    ctx_cat = ctx.get("category")
    ctx_brand = str(ctx.get("brand") or "").strip().lower()
    ctx_price = _to_float(ctx.get("price"), 0.0)
    pop_map = _co_popularity(co or {})
    rank_map = {pid: idx + 1 for idx, pid in enumerate(candidate_ids)}

    feats: list[list[float]] = []
    valid_ids: list[int] = []

    for pid in candidate_ids:
        cand = by_id.get(pid)
        if not cand:
            continue

        cand_cat = cand.get("category")
        cand_brand = str(cand.get("brand") or "").strip().lower()
        cand_price = _to_float(cand.get("price"), 0.0)
        trans = float(transitions.get(pid, 0.0))
        rank = int(rank_map.get(pid, 9999))
        rank_inv = 1.0 / float(rank + 1)
        same_cat = 1.0 if (ctx_cat and cand_cat and cand_cat == ctx_cat) else 0.0
        same_brand = 1.0 if (ctx_brand and cand_brand and cand_brand == ctx_brand) else 0.0
        price_diff = abs(cand_price - ctx_price)
        popularity = float(pop_map.get(pid, 0.0))

        feats.append([trans, rank_inv, same_cat, same_brand, price_diff, math.log1p(popularity)])
        valid_ids.append(pid)

    if not feats:
        return heuristic_results[:limit], "cooc_fallback:no_features"

    try:
        import numpy as np

        X = np.asarray(feats, dtype=float)
        probs = model.predict_proba(X)[:, 1]
    except Exception:
        return heuristic_results[:limit], "cooc_fallback:predict_error"

    scored = sorted(zip(valid_ids, probs), key=lambda x: (-float(x[1]), x[0]))

    out: list[dict[str, Any]] = []
    for pid, prob in scored:
        base = heuristic_map.get(pid)
        if not base:
            continue
        row = dict(base)
        comps = dict(row.get("components") or {})
        row["score"] = round(float(prob), 6)
        row["components"] = {
            **comps,
            "mode": "reranker",
            "reranker_prob": round(float(prob), 6),
            "heuristic_score": round(float(base.get("score") or 0.0), 6),
            "transition_count": round(float(transitions.get(pid, 0.0)), 6),
            "candidate_rank": int(rank_map.get(pid, 9999)),
        }
        out.append(row)
        if len(out) >= int(limit):
            break

    return out, "reranker"
