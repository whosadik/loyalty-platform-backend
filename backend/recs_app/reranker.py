from __future__ import annotations

import json
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


def _normalize_algo(algo_requested: str | None) -> str:
    algo_req = (algo_requested or getattr(settings, "RECS_ALGO_DEFAULT", "cooc") or "cooc").strip().lower()
    if algo_req not in {"cooc", "reranker"}:
        return "cooc"
    return algo_req


def _model_path() -> Path:
    return Path(str(getattr(settings, "RECS_RERANKER_MODEL_PATH", "") or "")).expanduser()


@lru_cache(maxsize=8)
def _load_model_cached(path_str: str, mtime_ns: int):
    del mtime_ns
    import joblib

    return joblib.load(path_str)


@lru_cache(maxsize=8)
def _load_metadata_cached(path_str: str, mtime_ns: int):
    del mtime_ns
    p = Path(path_str)
    meta_path = p.parent / "metadata.json"
    if not meta_path.exists():
        return {}
    try:
        return json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _load_reranker_model() -> tuple[Any | None, str | None, str | None]:
    try:
        import joblib  # noqa: F401
    except Exception:
        return None, None, "joblib_unavailable"

    path = _model_path()
    if not path.exists():
        return None, None, "model_not_found"
    if not path.is_file():
        return None, None, "model_not_file"

    try:
        path_resolved = str(path.resolve())
        mtime_ns = int(path.stat().st_mtime_ns)
        model = _load_model_cached(path_resolved, mtime_ns)
        meta = _load_metadata_cached(path_resolved, mtime_ns) or {}
        version = str(meta.get("model_version") or path.stem)
        return model, version, None
    except Exception:
        return None, None, "model_load_error"


def get_reranker_model_version() -> str | None:
    _, version, err = _load_reranker_model()
    if err:
        return None
    return version


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


def _build_feature_rows(
    *,
    context_product: dict[str, Any],
    candidate_ids: list[int],
    by_id: dict[int, dict[str, Any]],
    transitions: dict[int, float],
    rank_map: dict[int, int],
    pop_map: dict[int, float],
) -> tuple[list[int], list[list[float]]]:
    ctx_cat = context_product.get("category")
    ctx_brand = str(context_product.get("brand") or "").strip().lower()
    ctx_price = _to_float(context_product.get("price"), 0.0)

    valid_ids: list[int] = []
    feats: list[list[float]] = []

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

    return valid_ids, feats


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
) -> tuple[list[dict[str, Any]], str, str | None]:
    algo_req = _normalize_algo(algo_requested)

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
        return heuristic_results[:limit], "cooc", None

    model, model_version, err = _load_reranker_model()
    if model is None:
        return heuristic_results[:limit], f"cooc_fallback:{err}", None

    ctx_ids = _context_ids(context_product_ids, int(getattr(settings, "RECS_RERANKER_CONTEXT_K", 3) or 3))
    if not ctx_ids:
        return heuristic_results[:limit], "cooc_fallback:no_context", None

    by_id: dict[int, dict[str, Any]] = {}
    for p in products:
        pid = _to_int(p.get("id"))
        if pid is not None:
            by_id[pid] = p

    ctx_item = ctx_ids[-1]
    ctx = by_id.get(ctx_item)
    if not ctx:
        return heuristic_results[:limit], "cooc_fallback:no_context_item", None

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
        return heuristic_results[:limit], "cooc_fallback:no_candidates", None

    pop_map = _co_popularity(co or {})
    rank_map = {pid: idx + 1 for idx, pid in enumerate(candidate_ids)}
    valid_ids, feats = _build_feature_rows(
        context_product=ctx,
        candidate_ids=candidate_ids,
        by_id=by_id,
        transitions=transitions,
        rank_map=rank_map,
        pop_map=pop_map,
    )
    if not feats:
        return heuristic_results[:limit], "cooc_fallback:no_features", None

    try:
        import numpy as np

        X = np.asarray(feats, dtype=float)
        probs = model.predict_proba(X)[:, 1]
    except Exception:
        return heuristic_results[:limit], "cooc_fallback:predict_error", None

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
            "model_version": model_version,
            "reranker_prob": round(float(prob), 6),
            "heuristic_score": round(float(base.get("score") or 0.0), 6),
            "transition_count": round(float(transitions.get(pid, 0.0)), 6),
            "candidate_rank": int(rank_map.get(pid, 9999)),
        }
        out.append(row)
        if len(out) >= int(limit):
            break

    return out, "reranker", model_version


def rerank_bundle_with_algo(
    *,
    base_product_id: int,
    bundle_results: list[dict[str, Any]],
    products: list[dict[str, Any]],
    co: dict[int, dict[int, int]] | None,
    limit: int,
    algo_requested: str | None,
) -> tuple[list[dict[str, Any]], str, str | None]:
    algo_req = _normalize_algo(algo_requested)
    if algo_req != "reranker":
        return bundle_results[:limit], "cooc", None

    model, model_version, err = _load_reranker_model()
    if model is None:
        return bundle_results[:limit], f"cooc_fallback:{err}", None

    by_id: dict[int, dict[str, Any]] = {}
    for p in products:
        pid = _to_int(p.get("id"))
        if pid is not None:
            by_id[pid] = p

    base_id = int(base_product_id)
    ctx = by_id.get(base_id)
    if not ctx:
        return bundle_results[:limit], "cooc_fallback:no_context_item", None

    top_m = max(int(getattr(settings, "RECS_RERANKER_TOP_M", 200) or 200), int(limit))
    candidate_ids: list[int] = []
    transitions: dict[int, float] = {}
    row_map: dict[int, dict[str, Any]] = {}

    for idx, row in enumerate(bundle_results):
        pid = _to_int((row.get("product") or {}).get("id"))
        if pid is None:
            continue
        if pid in row_map:
            continue
        row_map[pid] = row
        candidate_ids.append(pid)

        comps = row.get("components") or {}
        trans = _to_float(comps.get("cooccurrence"), None)
        if trans is None:
            trans = _to_float(((co or {}).get(base_id, {}) or {}).get(pid), 0.0)
        transitions[pid] = float(trans)

        if len(candidate_ids) >= top_m:
            break

    if not candidate_ids:
        return bundle_results[:limit], "cooc_fallback:no_candidates", None

    pop_map = _co_popularity(co or {})
    rank_map = {pid: idx + 1 for idx, pid in enumerate(candidate_ids)}

    valid_ids, feats = _build_feature_rows(
        context_product=ctx,
        candidate_ids=candidate_ids,
        by_id=by_id,
        transitions=transitions,
        rank_map=rank_map,
        pop_map=pop_map,
    )
    if not feats:
        return bundle_results[:limit], "cooc_fallback:no_features", None

    try:
        import numpy as np

        X = np.asarray(feats, dtype=float)
        probs = model.predict_proba(X)[:, 1]
    except Exception:
        return bundle_results[:limit], "cooc_fallback:predict_error", None

    scored = sorted(zip(valid_ids, probs), key=lambda x: (-float(x[1]), x[0]))

    out: list[dict[str, Any]] = []
    for pid, prob in scored:
        base_row = row_map.get(pid)
        if not base_row:
            continue
        row = dict(base_row)
        comps = dict(row.get("components") or {})
        row["score"] = round(float(prob), 6)
        row["components"] = {
            **comps,
            "mode": "reranker",
            "model_version": model_version,
            "reranker_prob": round(float(prob), 6),
            "heuristic_score": round(float(base_row.get("score") or 0.0), 6),
            "transition_count": round(float(transitions.get(pid, 0.0)), 6),
            "candidate_rank": int(rank_map.get(pid, 9999)),
            "base_mode": str(comps.get("mode") or ""),
        }
        out.append(row)
        if len(out) >= int(limit):
            break

    return out, "reranker", model_version
