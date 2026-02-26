from __future__ import annotations

import hashlib
import json
import math
from datetime import timedelta
from functools import lru_cache
from pathlib import Path
from typing import Any

from django.conf import settings
from django.core.cache import cache
from django.db.models import Q
from django.utils import timezone

from ml_logic.recommender import UserProfile, recommend as heuristic_recommend
from recs_analytics.models import RecommendationEvent


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


def _normalize_algo(algo_requested: str | None) -> str | None:
    if algo_requested is None:
        return None
    s = str(algo_requested).strip().lower()
    if not s:
        return None
    if s == "auto":
        return None
    if s in {"cooc", "reranker"}:
        return s
    return None


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


def _ab_bucket(user_id: int | str, salt: str) -> int:
    raw = f"{salt}:{user_id}".encode("utf-8")
    digest = hashlib.sha1(raw).hexdigest()[:8]
    return int(digest, 16) % 100


def _algo_filter(algo_name: str):
    s = str(algo_name or "").strip().lower()
    if s == "reranker":
        return Q(algo_mode__startswith="reranker")
    if s == "cooc":
        return (
            Q(algo_mode__startswith="cooc")
            | Q(algo_mode__in=["cooccurrence", "fallback", "recommend"])
        )
    return Q(algo_mode=s)


def _guardrail_state() -> dict[str, Any]:
    enabled = bool(getattr(settings, "RECS_GUARDRAIL_ENABLED", False))
    if not enabled:
        return {"enabled": False, "force_cooc": False}

    window_days = int(getattr(settings, "RECS_GUARDRAIL_WINDOW_DAYS", 7) or 7)
    min_imp = int(getattr(settings, "RECS_GUARDRAIL_MIN_IMPRESSIONS", 200) or 200)
    min_delta = float(getattr(settings, "RECS_GUARDRAIL_MIN_DELTA_CR", -0.002) or -0.002)
    control_algo = str(getattr(settings, "RECS_GUARDRAIL_CONTROL_ALGO", "cooc") or "cooc").strip().lower()
    test_algo = str(getattr(settings, "RECS_GUARDRAIL_TEST_ALGO", "reranker") or "reranker").strip().lower()
    ttl = int(getattr(settings, "RECS_GUARDRAIL_CACHE_TTL_SECONDS", 60) or 60)

    cache_key = (
        f"recs:guardrail:v1:{window_days}:{min_imp}:{min_delta}:{control_algo}:{test_algo}"
    )
    if ttl > 0:
        cached = cache.get(cache_key)
        if cached is not None:
            return cached

    since = timezone.now() - timedelta(days=window_days)

    def stats_for(algo_name: str) -> tuple[int, int, float]:
        q = _algo_filter(algo_name)
        base = RecommendationEvent.objects.filter(created_at__gte=since).filter(q)
        imp = base.filter(action=RecommendationEvent.Action.IMPRESSION).count()
        pur = base.filter(action=RecommendationEvent.Action.PURCHASE_ATTRIBUTED).count()
        cr = (pur / imp) if imp else 0.0
        return int(imp), int(pur), float(cr)

    imp_ctrl, pur_ctrl, cr_ctrl = stats_for(control_algo)
    imp_test, pur_test, cr_test = stats_for(test_algo)

    force_cooc = False
    reason = "ok"
    if imp_ctrl < min_imp or imp_test < min_imp:
        reason = "insufficient_impressions"
    else:
        delta = cr_test - cr_ctrl
        if delta < min_delta:
            force_cooc = True
            reason = "cr_drop_below_threshold"
        else:
            reason = "healthy"

    out = {
        "enabled": True,
        "force_cooc": force_cooc,
        "reason": reason,
        "window_days": window_days,
        "min_impressions": min_imp,
        "min_delta_cr": min_delta,
        "control_algo": control_algo,
        "test_algo": test_algo,
        "control": {
            "impressions": imp_ctrl,
            "purchases": pur_ctrl,
            "cr": round(cr_ctrl, 6),
        },
        "test": {
            "impressions": imp_test,
            "purchases": pur_test,
            "cr": round(cr_test, 6),
        },
    }
    if ttl > 0:
        cache.set(cache_key, out, timeout=ttl)
    return out


def _resolve_algo_for_user(user_id: int | str | None, algo_requested: str | None) -> tuple[str, dict[str, Any]]:
    requested = _normalize_algo(algo_requested)
    if requested in {"cooc", "reranker"}:
        return requested, {"source": "explicit", "requested": requested}

    if bool(getattr(settings, "RECS_AB_ENABLED", False)) and user_id is not None:
        pct = max(0, min(100, int(getattr(settings, "RECS_AB_RERANKER_PERCENT", 50) or 50)))
        salt = str(getattr(settings, "RECS_AB_SALT", "recs_ab_v1") or "recs_ab_v1")
        experiment_id = str(
            getattr(settings, "RECS_AB_EXPERIMENT_ID", "recs_algo_ab_v1") or "recs_algo_ab_v1"
        ).strip()
        bucket = _ab_bucket(user_id, salt)
        variant = "test" if bucket < pct else "control"
        algo = "reranker" if variant == "test" else "cooc"
        return algo, {
            "source": "ab",
            "requested": algo_requested,
            "experiment_id": experiment_id,
            "ab_bucket": bucket,
            "ab_variant": variant,
            "ab_reranker_percent": pct,
        }

    default_algo = str(getattr(settings, "RECS_ALGO_DEFAULT", "cooc") or "cooc").strip().lower()
    if default_algo not in {"cooc", "reranker"}:
        default_algo = "cooc"
    return default_algo, {"source": "default", "requested": algo_requested}


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
    user_id: int | str | None,
    prof: UserProfile,
    products: list[dict[str, Any]],
    owned_active_ids: list[int],
    context_product_ids: list[int],
    category: str | None,
    product_type: str | None,
    limit: int,
    co: dict[int, dict[int, int]] | None,
    algo_requested: str | None,
) -> tuple[list[dict[str, Any]], str, str | None, dict[str, Any]]:
    target_algo, route_meta = _resolve_algo_for_user(user_id, algo_requested)

    if target_algo == "reranker":
        guard = _guardrail_state()
        route_meta["guardrail"] = guard
        if guard.get("force_cooc"):
            target_algo = "cooc"
            route_meta["guardrail_forced"] = True
            route_meta["guardrail_reason"] = guard.get("reason")
        else:
            route_meta["guardrail_forced"] = False

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
    heuristic_rows: list[dict[str, Any]] = []
    for r in heuristic_results:
        row = dict(r)
        comps = dict(row.get("components") or {})
        comps.setdefault("mode", "cooc")
        row["components"] = comps
        heuristic_rows.append(row)

    if target_algo != "reranker":
        return heuristic_rows[:limit], "cooc", None, route_meta

    model, model_version, err = _load_reranker_model()
    if model is None:
        route_meta["fallback_reason"] = err
        return heuristic_rows[:limit], f"cooc_fallback:{err}", None, route_meta

    ctx_ids = _context_ids(context_product_ids, int(getattr(settings, "RECS_RERANKER_CONTEXT_K", 3) or 3))
    if not ctx_ids:
        route_meta["fallback_reason"] = "no_context"
        return heuristic_rows[:limit], "cooc_fallback:no_context", None, route_meta

    by_id: dict[int, dict[str, Any]] = {}
    for p in products:
        pid = _to_int(p.get("id"))
        if pid is not None:
            by_id[pid] = p

    ctx_item = ctx_ids[-1]
    ctx = by_id.get(ctx_item)
    if not ctx:
        route_meta["fallback_reason"] = "no_context_item"
        return heuristic_rows[:limit], "cooc_fallback:no_context_item", None, route_meta

    transitions = _aggregate_transition_counts(ctx_ids, co or {})
    ranked_trans = sorted(transitions.items(), key=lambda x: (-x[1], x[0]))
    candidate_ids = [int(cid) for cid, _ in ranked_trans[:top_m]]

    heuristic_map: dict[int, dict[str, Any]] = {}
    for r in heuristic_rows:
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
        route_meta["fallback_reason"] = "no_candidates"
        return heuristic_rows[:limit], "cooc_fallback:no_candidates", None, route_meta

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
        route_meta["fallback_reason"] = "no_features"
        return heuristic_rows[:limit], "cooc_fallback:no_features", None, route_meta

    try:
        import numpy as np

        X = np.asarray(feats, dtype=float)
        probs = model.predict_proba(X)[:, 1]
    except Exception:
        route_meta["fallback_reason"] = "predict_error"
        return heuristic_rows[:limit], "cooc_fallback:predict_error", None, route_meta

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

    return out, "reranker", model_version, route_meta


def rerank_bundle_with_algo(
    *,
    user_id: int | str | None,
    base_product_id: int,
    bundle_results: list[dict[str, Any]],
    products: list[dict[str, Any]],
    co: dict[int, dict[int, int]] | None,
    limit: int,
    algo_requested: str | None,
) -> tuple[list[dict[str, Any]], str, str | None, dict[str, Any]]:
    target_algo, route_meta = _resolve_algo_for_user(user_id, algo_requested)

    if target_algo == "reranker":
        guard = _guardrail_state()
        route_meta["guardrail"] = guard
        if guard.get("force_cooc"):
            target_algo = "cooc"
            route_meta["guardrail_forced"] = True
            route_meta["guardrail_reason"] = guard.get("reason")
        else:
            route_meta["guardrail_forced"] = False

    if target_algo != "reranker":
        out = []
        for r in bundle_results[:limit]:
            row = dict(r)
            comps = dict(row.get("components") or {})
            comps.setdefault("mode", "cooc")
            row["components"] = comps
            out.append(row)
        return out, "cooc", None, route_meta

    model, model_version, err = _load_reranker_model()
    if model is None:
        route_meta["fallback_reason"] = err
        out = []
        for r in bundle_results[:limit]:
            row = dict(r)
            comps = dict(row.get("components") or {})
            comps.setdefault("mode", "cooc")
            row["components"] = comps
            out.append(row)
        return out, f"cooc_fallback:{err}", None, route_meta

    by_id: dict[int, dict[str, Any]] = {}
    for p in products:
        pid = _to_int(p.get("id"))
        if pid is not None:
            by_id[pid] = p

    base_id = int(base_product_id)
    ctx = by_id.get(base_id)
    if not ctx:
        route_meta["fallback_reason"] = "no_context_item"
        return bundle_results[:limit], "cooc_fallback:no_context_item", None, route_meta

    top_m = max(int(getattr(settings, "RECS_RERANKER_TOP_M", 200) or 200), int(limit))
    candidate_ids: list[int] = []
    transitions: dict[int, float] = {}
    row_map: dict[int, dict[str, Any]] = {}

    for row in bundle_results:
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
        route_meta["fallback_reason"] = "no_candidates"
        return bundle_results[:limit], "cooc_fallback:no_candidates", None, route_meta

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
        route_meta["fallback_reason"] = "no_features"
        return bundle_results[:limit], "cooc_fallback:no_features", None, route_meta

    try:
        import numpy as np

        X = np.asarray(feats, dtype=float)
        probs = model.predict_proba(X)[:, 1]
    except Exception:
        route_meta["fallback_reason"] = "predict_error"
        return bundle_results[:limit], "cooc_fallback:predict_error", None, route_meta

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

    return out, "reranker", model_version, route_meta
