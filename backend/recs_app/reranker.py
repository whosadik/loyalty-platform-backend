from __future__ import annotations

import hashlib
import json
import math
import pickle
from datetime import timedelta
from functools import lru_cache
from pathlib import Path
from typing import Any
from collections import defaultdict

from django.conf import settings
from django.core.cache import cache
from django.db import connection
from django.db.models import Q, Sum
from django.utils import timezone

from ml_logic.recommender import UserProfile, recommend as heuristic_recommend
from recs_analytics.models import RecommendationEvent
from transactions.models import TransactionItem


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


def _co_path_candidates() -> list[Path]:
    configured = str(getattr(settings, "RECS_CO_MAP_PATH", "") or "").strip()
    out: list[Path] = []
    if configured:
        out.append(Path(configured).expanduser())
    model_dir = _model_path().expanduser().parent
    out.extend(
        [
            model_dir / "co_map.pkl",
            model_dir / "co_map.json",
            model_dir / "cooc.pkl",
            model_dir / "cooc.json",
        ]
    )
    uniq: list[Path] = []
    seen: set[str] = set()
    for p in out:
        k = str(p)
        if k in seen:
            continue
        seen.add(k)
        uniq.append(p)
    return uniq


def _normalize_co_map(raw: Any) -> dict[int, dict[int, int]]:
    if not isinstance(raw, dict):
        return {}
    out: dict[int, dict[int, int]] = {}
    for src, rel in raw.items():
        s = _to_int(src)
        if s is None or not isinstance(rel, dict):
            continue
        inner: dict[int, int] = {}
        for dst, cnt in rel.items():
            d = _to_int(dst)
            if d is None:
                continue
            try:
                c = int(cnt)
            except Exception:
                continue
            if c <= 0:
                continue
            inner[d] = c
        if inner:
            out[s] = inner
    return out


def _has_nonempty_co_map(co: dict[int, dict[int, int]] | None) -> bool:
    if not isinstance(co, dict) or not co:
        return False
    for rel in co.values():
        if isinstance(rel, dict) and rel:
            return True
    return False


@lru_cache(maxsize=8)
def _load_co_cached(path_str: str, mtime_ns: int) -> dict[int, dict[int, int]]:
    del mtime_ns
    p = Path(path_str)
    if not p.exists() or not p.is_file():
        return {}
    try:
        if p.suffix.lower() == ".json":
            raw = json.loads(p.read_text(encoding="utf-8"))
        else:
            try:
                import joblib

                raw = joblib.load(str(p))
            except Exception:
                with p.open("rb") as f:
                    raw = pickle.load(f)
    except Exception:
        return {}
    return _normalize_co_map(raw)


def _load_runtime_co_from_artifact() -> tuple[dict[int, dict[int, int]], str]:
    empty_source: str | None = None
    for p in _co_path_candidates():
        if not p.exists() or not p.is_file():
            continue
        try:
            co = _load_co_cached(str(p.resolve()), int(p.stat().st_mtime_ns))
        except Exception:
            continue
        if _has_nonempty_co_map(co):
            return co, f"artifact:{p.name}"
        if empty_source is None:
            empty_source = f"artifact:{p.name}_empty"
    if empty_source is not None:
        return {}, empty_source
    return {}, "artifact:none"


def _events_co_map(days: int) -> dict[int, dict[int, int]]:
    ttl = max(
        0,
        int(getattr(settings, "RECS_RUNTIME_CO_EVENTS_CACHE_TTL_SECONDS", 600) or 600),
    )
    db_name = str(connection.settings_dict.get("NAME", "default"))
    key = f"recs:runtime:co_map:events:v1:{db_name}:days:{int(days)}"
    if ttl > 0:
        cached = cache.get(key)
        if cached is not None:
            return cached

    qs = RecommendationEvent.objects.filter(
        action__in=[
            RecommendationEvent.Action.CLICK,
            RecommendationEvent.Action.ADD_TO_CART,
            RecommendationEvent.Action.PURCHASE_ATTRIBUTED,
        ],
        product_id__isnull=False,
    ).values("user_id", "product_id", "created_at", "id")
    if int(days) > 0:
        since = timezone.now() - timedelta(days=int(days))
        qs = qs.filter(created_at__gte=since)
    qs = qs.order_by("user_id", "created_at", "id")

    co: dict[int, dict[int, int]] = defaultdict(lambda: defaultdict(int))
    prev_by_user: dict[int, int] = {}

    for row in qs.iterator():
        uid = _to_int(row.get("user_id"))
        pid = _to_int(row.get("product_id"))
        if uid is None or pid is None:
            continue
        prev = prev_by_user.get(uid)
        if prev is not None and prev != pid:
            co[int(prev)][int(pid)] += 1
        prev_by_user[uid] = int(pid)

    out = {
        int(src): {int(dst): int(cnt) for dst, cnt in dict(rel).items() if int(cnt) > 0}
        for src, rel in dict(co).items()
        if rel
    }
    if ttl > 0:
        cache.set(key, out, timeout=ttl)
    return out


def _normalize_co_source(source: str | None, co_dict: dict[int, dict[int, int]]) -> str:
    s = str(source or "").strip() or "none"
    if _has_nonempty_co_map(co_dict):
        return s
    if s in {"none", "none:empty"}:
        return "none:empty"
    if s.endswith("_empty"):
        return s
    return f"{s}_empty"


def _cooc_source_label(co_source: str) -> str:
    s = str(co_source or "").strip().lower()
    if s.startswith("events_") or "event" in s:
        return "cooc_events"
    return "cooc_purchase"


def _db_co_map(days: int) -> dict[int, dict[int, int]]:
    ttl = max(0, int(getattr(settings, "RECS_RUNTIME_CO_CACHE_TTL_SECONDS", 600) or 600))
    db_name = str(connection.settings_dict.get("NAME", "default"))
    key = f"recs:runtime:co_map:v1:{db_name}:days:{int(days)}"
    if ttl > 0:
        cached = cache.get(key)
        if cached is not None:
            return cached

    qs = TransactionItem.objects.all().values("transaction_id", "product_id")
    if int(days) > 0:
        since = timezone.now() - timedelta(days=int(days))
        qs = qs.filter(transaction__created_at__gte=since)

    txn_map: dict[int, list[int]] = {}
    for row in qs.iterator():
        tx_id = _to_int(row.get("transaction_id"))
        pid = _to_int(row.get("product_id"))
        if tx_id is None or pid is None:
            continue
        txn_map.setdefault(tx_id, []).append(pid)

    co: dict[int, dict[int, int]] = defaultdict(lambda: defaultdict(int))
    for items in txn_map.values():
        uniq = list(dict.fromkeys(int(x) for x in items if _to_int(x) is not None))
        n = len(uniq)
        if n < 2:
            continue
        for i in range(n):
            a = int(uniq[i])
            for j in range(i + 1, n):
                b = int(uniq[j])
                co[a][b] += 1
                co[b][a] += 1

    out = {int(k): {int(kk): int(vv) for kk, vv in dict(v).items()} for k, v in dict(co).items()}
    if ttl > 0:
        cache.set(key, out, timeout=ttl)
    return out


def get_runtime_co_map() -> tuple[dict[int, dict[int, int]], str]:
    co_artifact, source_artifact = _load_runtime_co_from_artifact()
    if _has_nonempty_co_map(co_artifact):
        return co_artifact, source_artifact

    events_days = max(0, int(getattr(settings, "RECS_RUNTIME_CO_EVENTS_DAYS", 60) or 60))
    co_events = _events_co_map(events_days)
    if _has_nonempty_co_map(co_events):
        return co_events, f"events_{events_days}d"

    db_days = max(0, int(getattr(settings, "RECS_RUNTIME_CO_DB_DAYS", 90) or 90))
    co_db = _db_co_map(db_days)
    if _has_nonempty_co_map(co_db):
        return co_db, f"db_{db_days}d"

    co_all = _db_co_map(0)
    if _has_nonempty_co_map(co_all):
        return co_all, "db_all_time"

    if source_artifact != "artifact:none":
        return {}, source_artifact
    return {}, "db_all_time_empty"


def get_db_popularity_map(days: int = 30) -> tuple[dict[int, float], str]:
    ttl = max(0, int(getattr(settings, "RECS_RUNTIME_POPULARITY_CACHE_TTL_SECONDS", 600) or 600))
    db_name = str(connection.settings_dict.get("NAME", "default"))
    d = max(0, int(days or 0))
    key = f"recs:runtime:popularity:v1:{db_name}:days:{d}"
    if ttl > 0:
        cached = cache.get(key)
        if cached is not None:
            return cached, f"db_{d}d"

    qs = TransactionItem.objects.all()
    if d > 0:
        since = timezone.now() - timedelta(days=d)
        qs = qs.filter(transaction__created_at__gte=since)

    agg = (
        qs.values("product_id")
        .annotate(pop=Sum("quantity"))
        .order_by("-pop")
    )
    pop = {
        int(row["product_id"]): float(row["pop"] or 0.0)
        for row in agg
        if _to_int(row.get("product_id")) is not None and float(row.get("pop") or 0.0) > 0.0
    }
    if ttl > 0:
        cache.set(key, pop, timeout=ttl)
    return pop, f"db_{d}d"


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
    pop: dict[int, float] = defaultdict(float)
    for src, related in (co or {}).items():
        sid = _to_int(src)
        if sid is None:
            continue
        total = 0.0
        for _, cnt in (related or {}).items():
            total += float(cnt or 0.0)
        pop[sid] += total
        for dst, cnt in (related or {}).items():
            did = _to_int(dst)
            if did is None:
                continue
            pop[did] += float(cnt or 0.0)
    return dict(pop)


def _retrieval_top_m(limit: int) -> int:
    configured = int(getattr(settings, "RECS_RETRIEVAL_TOP_M", 0) or 0)
    if configured <= 0:
        configured = int(getattr(settings, "RECS_RERANKER_TOP_M", 500) or 500)
    return max(configured, int(limit))


def _retrieval_context_k() -> int:
    configured = int(getattr(settings, "RECS_RETRIEVAL_CONTEXT_K", 0) or 0)
    if configured <= 0:
        configured = int(getattr(settings, "RECS_RERANKER_CONTEXT_K", 5) or 5)
    return max(1, configured)


def _retrieval_category_fallback_topn() -> int:
    return max(0, int(getattr(settings, "RECS_RETRIEVAL_CATEGORY_FALLBACK_TOPN", 100) or 100))


def _retrieval_product_type_fallback_topn() -> int:
    return max(0, int(getattr(settings, "RECS_RETRIEVAL_PRODUCT_TYPE_FALLBACK_TOPN", 100) or 100))


def _retrieval_global_fallback_topn() -> int:
    return max(0, int(getattr(settings, "RECS_RETRIEVAL_GLOBAL_FALLBACK_TOPN", 100) or 100))


def _retrieval_brand_fallback_topn() -> int:
    return max(0, int(getattr(settings, "RECS_RETRIEVAL_BRAND_FALLBACK_TOPN", 100) or 100))


def _popularity_indexes(
    by_id: dict[int, dict[str, Any]],
    pop_map: dict[int, float],
) -> tuple[list[int], dict[str, list[int]], dict[str, list[int]], dict[str, list[int]]]:
    known_ids = [pid for pid in pop_map.keys() if pid in by_id]
    global_rank = sorted(known_ids, key=lambda x: (-float(pop_map.get(x, 0.0)), x))
    by_cat_raw: dict[str, list[int]] = defaultdict(list)
    by_type_raw: dict[str, list[int]] = defaultdict(list)
    by_brand_raw: dict[str, list[int]] = defaultdict(list)
    for pid in global_rank:
        cat = str((by_id.get(pid) or {}).get("category") or "")
        ptype = str((by_id.get(pid) or {}).get("product_type") or "").strip().lower()
        if not cat:
            cat = ""
        brand = str((by_id.get(pid) or {}).get("brand") or "").strip().lower()
        if cat:
            by_cat_raw[cat].append(int(pid))
        if ptype:
            by_type_raw[ptype].append(int(pid))
        if brand:
            by_brand_raw[brand].append(int(pid))
    return global_rank, dict(by_cat_raw), dict(by_type_raw), dict(by_brand_raw)


def _category_fallback_ids(
    context_ids: list[int],
    *,
    by_id: dict[int, dict[str, Any]],
    by_cat_pop: dict[str, list[int]],
    per_cat_topn: int,
) -> list[int]:
    if per_cat_topn <= 0 or not context_ids:
        return []
    out: list[int] = []
    seen: set[int] = set()
    seen_cat: set[str] = set()
    for ctx in reversed(context_ids):
        cat = str((by_id.get(int(ctx)) or {}).get("category") or "")
        if not cat or cat in seen_cat:
            continue
        seen_cat.add(cat)
        for pid in by_cat_pop.get(cat, [])[:per_cat_topn]:
            ipid = int(pid)
            if ipid in seen:
                continue
            seen.add(ipid)
            out.append(ipid)
    return out


def _product_type_fallback_ids(
    context_ids: list[int],
    *,
    by_id: dict[int, dict[str, Any]],
    by_type_pop: dict[str, list[int]],
    per_type_topn: int,
) -> list[int]:
    if per_type_topn <= 0 or not context_ids:
        return []
    out: list[int] = []
    seen: set[int] = set()
    seen_types: set[str] = set()
    for ctx in reversed(context_ids):
        ptype = str((by_id.get(int(ctx)) or {}).get("product_type") or "").strip().lower()
        if not ptype or ptype in seen_types:
            continue
        seen_types.add(ptype)
        for pid in by_type_pop.get(ptype, [])[:per_type_topn]:
            ipid = int(pid)
            if ipid in seen:
                continue
            seen.add(ipid)
            out.append(ipid)
    return out


def _brand_fallback_ids(
    context_ids: list[int],
    *,
    by_id: dict[int, dict[str, Any]],
    by_brand_pop: dict[str, list[int]],
    per_brand_topn: int,
) -> list[int]:
    if per_brand_topn <= 0 or not context_ids:
        return []
    out: list[int] = []
    seen: set[int] = set()
    seen_brand: set[str] = set()
    for ctx in reversed(context_ids):
        brand = str((by_id.get(int(ctx)) or {}).get("brand") or "").strip().lower()
        if not brand or brand in seen_brand:
            continue
        seen_brand.add(brand)
        for pid in by_brand_pop.get(brand, [])[:per_brand_topn]:
            ipid = int(pid)
            if ipid in seen:
                continue
            seen.add(ipid)
            out.append(ipid)
    return out


USER_FEATURE_NAMES = [
    "u_tx_count_90d",
    "u_avg_price_90d_log",
    "u_top_cat_match",
    "u_top_brand_match",
    "u_top_ptype_match",
    "u_cat_affinity",
    "u_price_fit",
]


def _reranker_model_features() -> list[str]:
    """Return the feature list recorded in metadata.json alongside the model.

    If metadata is missing or unreadable, returns an empty list so callers
    fall back to the legacy 6-feature path.
    """
    path = _model_path()
    if not path.exists() or not path.is_file():
        return []
    try:
        mtime_ns = int(path.stat().st_mtime_ns)
        meta = _load_metadata_cached(str(path.resolve()), mtime_ns) or {}
    except Exception:
        return []
    return list(meta.get("features") or [])


def _build_runtime_user_context(
    user_id: int | str | None,
    by_id: dict[int, dict[str, Any]],
    *,
    window_days: int = 90,
) -> dict[str, Any] | None:
    """Build the user-context dict consumed by the user-features reranker.

    Derives top category / brand / product_type and average price from the
    user's purchases in the last `window_days`. Returns None when we have
    no user_id or no qualifying purchases — callers should then skip
    appending user features and fall back to the 6-feature path.
    """
    uid = _to_int(user_id)
    if uid is None:
        return None

    since = timezone.now() - timedelta(days=int(window_days))
    qs = (
        TransactionItem.objects.filter(
            transaction__user_id=uid,
            transaction__created_at__gte=since,
        )
        .values("product_id", "unit_price")
    )
    rows = list(qs.iterator())
    if not rows:
        return None

    cat_counts: dict[str, int] = defaultdict(int)
    brand_counts: dict[str, int] = defaultdict(int)
    ptype_counts: dict[str, int] = defaultdict(int)
    prices: list[float] = []
    total = 0

    for row in rows:
        pid = _to_int(row.get("product_id"))
        if pid is None:
            continue
        prod = by_id.get(pid)
        if prod:
            cat = str(prod.get("category") or "").strip().lower()
            brand = str(prod.get("brand") or "").strip().lower()
            ptype = str(prod.get("product_type") or "").strip().lower()
            if cat:
                cat_counts[cat] += 1
            if brand:
                brand_counts[brand] += 1
            if ptype:
                ptype_counts[ptype] += 1
        price = _to_float(row.get("unit_price"), 0.0)
        if price > 0:
            prices.append(price)
        total += 1

    if total == 0:
        return None

    avg_price = (sum(prices) / len(prices)) if prices else 0.0
    top_cat = max(cat_counts, key=cat_counts.get) if cat_counts else None
    top_brand = max(brand_counts, key=brand_counts.get) if brand_counts else None
    top_ptype = max(ptype_counts, key=ptype_counts.get) if ptype_counts else None
    cat_total = sum(cat_counts.values()) or 1
    cat_share = {k: float(v) / float(cat_total) for k, v in cat_counts.items()}

    return {
        "u_tx_count_90d": float(total),
        "u_avg_price_90d": float(avg_price),
        "u_top_cat": top_cat,
        "u_top_brand": top_brand,
        "u_top_ptype": top_ptype,
        "u_cat_share": cat_share,
    }


def _user_feature_row(
    *,
    cand_cat: Any,
    cand_brand: str,
    cand_ptype: Any,
    cand_price: float,
    user_ctx: dict[str, Any] | None,
) -> list[float]:
    if not user_ctx:
        return [0.0] * len(USER_FEATURE_NAMES)

    tx = float(user_ctx.get("u_tx_count_90d") or 0.0)
    avg_price = float(user_ctx.get("u_avg_price_90d") or 0.0)
    top_cat = user_ctx.get("u_top_cat")
    top_brand = user_ctx.get("u_top_brand")
    top_ptype = user_ctx.get("u_top_ptype")
    cat_share = user_ctx.get("u_cat_share") or {}

    cc = str(cand_cat or "").strip().lower()
    cb = (cand_brand or "").strip().lower()
    cp = str(cand_ptype or "").strip().lower()

    top_cat_match = 1.0 if (top_cat and cc and cc == top_cat) else 0.0
    top_brand_match = 1.0 if (top_brand and cb and cb == top_brand) else 0.0
    top_ptype_match = 1.0 if (top_ptype and cp and cp == top_ptype) else 0.0
    cat_affinity = float(cat_share.get(cc, 0.0)) if cc else 0.0
    if avg_price > 0 and cand_price > 0:
        price_fit = 1.0 / (1.0 + abs(cand_price - avg_price) / max(avg_price, 1.0))
    else:
        price_fit = 0.0

    return [
        tx,
        math.log1p(avg_price),
        top_cat_match,
        top_brand_match,
        top_ptype_match,
        cat_affinity,
        price_fit,
    ]


def _build_feature_rows(
    *,
    context_product: dict[str, Any],
    candidate_ids: list[int],
    by_id: dict[int, dict[str, Any]],
    transitions: dict[int, float],
    rank_map: dict[int, int],
    pop_map: dict[int, float],
    user_context: dict[str, Any] | None = None,
) -> tuple[list[int], list[list[float]]]:
    ctx_cat = context_product.get("category")
    ctx_brand = str(context_product.get("brand") or "").strip().lower()
    ctx_price = _to_float(context_product.get("price"), 0.0)

    valid_ids: list[int] = []
    feats: list[list[float]] = []
    use_user_ctx = user_context is not None

    for pid in candidate_ids:
        cand = by_id.get(pid)
        if not cand:
            continue

        cand_cat = cand.get("category")
        cand_brand = str(cand.get("brand") or "").strip().lower()
        cand_ptype = cand.get("product_type")
        cand_price = _to_float(cand.get("price"), 0.0)
        trans = float(transitions.get(pid, 0.0))
        rank = int(rank_map.get(pid, 9999))
        rank_inv = 1.0 / float(rank + 1)
        same_cat = 1.0 if (ctx_cat and cand_cat and cand_cat == ctx_cat) else 0.0
        same_brand = 1.0 if (ctx_brand and cand_brand and cand_brand == ctx_brand) else 0.0
        price_diff = abs(cand_price - ctx_price)
        popularity = float(pop_map.get(pid, 0.0))

        row = [trans, rank_inv, same_cat, same_brand, price_diff, math.log1p(popularity)]
        if use_user_ctx:
            row.extend(
                _user_feature_row(
                    cand_cat=cand_cat,
                    cand_brand=cand_brand,
                    cand_ptype=cand_ptype,
                    cand_price=cand_price,
                    user_ctx=user_context,
                )
            )
        feats.append(row)
        valid_ids.append(pid)

    return valid_ids, feats


def _passes_runtime_candidate_filters(
    product: dict[str, Any] | None,
    *,
    owned_set: set[int],
    category: str | None,
    product_type: str | None,
) -> bool:
    if not product:
        return False
    pid = _to_int(product.get("id"))
    if pid is None:
        return False
    if pid in owned_set:
        return False
    if product.get("in_stock") is False:
        return False
    if category:
        if str(product.get("category") or "").strip().lower() != str(category).strip().lower():
            return False
    if product_type:
        if str(product.get("product_type") or "").strip().lower() != str(product_type).strip().lower():
            return False
    return True


def _cold_start_rows(
    *,
    products: list[dict[str, Any]],
    owned_active_ids: list[int],
    category: str | None,
    product_type: str | None,
    pop_map: dict[int, float],
    limit: int,
) -> list[dict[str, Any]]:
    lim = max(1, int(limit or 1))
    owned_set = {int(x) for x in owned_active_ids if _to_int(x) is not None}
    cat_filter = str(category or "").strip().lower() or None
    ptype_filter = str(product_type or "").strip().lower() or None

    seen: set[int] = set()
    out: list[dict[str, Any]] = []

    def append_scope(*, scope_label: str, cat: str | None, ptype: str | None) -> None:
        candidates: list[tuple[int, float, dict[str, Any]]] = []
        for p in products:
            pid = _to_int(p.get("id"))
            if pid is None or pid in owned_set:
                continue
            if p.get("in_stock") is False:
                continue
            p_cat = str(p.get("category") or "").strip().lower()
            p_type = str(p.get("product_type") or "").strip().lower()
            if cat and p_cat != cat:
                continue
            if ptype and p_type != ptype:
                continue
            pop = float(pop_map.get(pid, 0.0))
            candidates.append((int(pid), pop, p))

        candidates.sort(key=lambda x: (-x[1], x[0]))
        for pid, pop, product_row in candidates:
            if pid in seen:
                continue
            seen.add(pid)
            out.append(
                {
                    "product": product_row,
                    "score": round(float(pop), 6),
                    "components": {
                        "mode": "cold_start",
                        "source": "trending_popularity",
                        "popularity": round(float(pop), 6),
                    },
                    "why": [f"cold start: top-selling {scope_label}"],
                }
            )
            if len(out) >= lim:
                return

    if ptype_filter:
        append_scope(
            scope_label=f"for product_type={ptype_filter}",
            cat=cat_filter,
            ptype=ptype_filter,
        )
    if len(out) < lim and cat_filter:
        append_scope(scope_label=f"in category={cat_filter}", cat=cat_filter, ptype=None)
    if len(out) < lim:
        append_scope(scope_label="globally", cat=None, ptype=None)

    return out[:lim]


def _cold_start_rows_with_relaxation(
    *,
    products: list[dict[str, Any]],
    owned_active_ids: list[int],
    category: str | None,
    product_type: str | None,
    pop_map: dict[int, float],
    limit: int,
) -> list[dict[str, Any]]:
    rows = _cold_start_rows(
        products=products,
        owned_active_ids=owned_active_ids,
        category=category,
        product_type=product_type,
        pop_map=pop_map,
        limit=limit,
    )
    if rows:
        return rows

    if category or product_type:
        relaxed = _cold_start_rows(
            products=products,
            owned_active_ids=owned_active_ids,
            category=None,
            product_type=None,
            pop_map=pop_map,
            limit=limit,
        )
        if relaxed:
            for row in relaxed:
                why = list(row.get("why") or [])
                why.append("fallback: relaxed category/product_type filters due to sparse candidates")
                row["why"] = why
            return relaxed

    return rows


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
    co_source: str | None = None,
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

    top_m = _retrieval_top_m(limit)
    ctx_k = _retrieval_context_k()
    category_fb_topn = _retrieval_category_fallback_topn()
    product_type_fb_topn = _retrieval_product_type_fallback_topn()
    brand_fb_topn = _retrieval_brand_fallback_topn()
    global_fb_topn = _retrieval_global_fallback_topn()

    co_dict = co or {}
    co_source_resolved = _normalize_co_source(co_source or ("provided" if co_dict else "none"), co_dict)
    co_present = _has_nonempty_co_map(co_dict)
    pop_days = max(0, int(getattr(settings, "RECS_RUNTIME_POPULARITY_DAYS", 30) or 30))
    db_pop_map, db_pop_source = get_db_popularity_map(pop_days)
    if db_pop_map:
        pop_map = db_pop_map
        pop_source = db_pop_source
    else:
        pop_map = _co_popularity(co_dict)
        pop_source = "co_map" if pop_map else "none"

    retrieval_diag: dict[str, Any] = {
        "top_m": int(top_m),
        "context_k": int(ctx_k),
        "products_count": int(len(products)),
        "heuristic_pool_size": 0,
        "co_present": bool(co_present),
        "co_keys": int(len(co_dict)),
        "co_source": co_source_resolved,
        "pop_source": pop_source,
        "pop_nonzero_topn": 0,
        "ctx_ids_len": 0,
        "transitions_len": 0,
    }

    pool_limit = max(
        int(getattr(settings, "RECS_RERANKER_HEURISTIC_POOL", 1000) or 1000),
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
        co=co_dict,
    )
    heuristic_rows: list[dict[str, Any]] = []
    for r in heuristic_results:
        row = dict(r)
        comps = dict(row.get("components") or {})
        comps.setdefault("mode", "cooc")
        row["components"] = comps
        heuristic_rows.append(row)
    retrieval_diag["heuristic_pool_size"] = int(len(heuristic_rows))

    if target_algo != "reranker":
        route_meta["retrieval"] = retrieval_diag
        return heuristic_rows[:limit], "cooc", None, route_meta

    model, model_version, err = _load_reranker_model()
    if model is None:
        route_meta["fallback_reason"] = err
        route_meta["retrieval"] = retrieval_diag
        return heuristic_rows[:limit], f"cooc_fallback:{err}", None, route_meta

    by_id: dict[int, dict[str, Any]] = {}
    for p in products:
        pid = _to_int(p.get("id"))
        if pid is not None:
            by_id[pid] = p

    ctx_ids = _context_ids(context_product_ids, ctx_k)
    retrieval_diag["ctx_ids_len"] = int(len(ctx_ids))
    if not ctx_ids:
        route_meta["fallback_reason"] = "no_context"
        cold_rows = _cold_start_rows_with_relaxation(
            products=products,
            owned_active_ids=owned_active_ids,
            category=category,
            product_type=product_type,
            pop_map=pop_map,
            limit=limit,
        )
        if cold_rows:
            retrieval_diag.update(
                {
                    "sources": ["cold_start_trending"],
                    "final_candidates": int(len(cold_rows)),
                    "pop_nonzero_topn": int(sum(1 for v in pop_map.values() if float(v) > 0.0)),
                }
            )
            route_meta["retrieval"] = retrieval_diag
            return cold_rows, "cold_start:trending", None, route_meta
        route_meta["retrieval"] = retrieval_diag
        return heuristic_rows[:limit], "cooc_fallback:no_context", None, route_meta

    ctx_item = ctx_ids[-1]
    ctx = by_id.get(ctx_item)
    if not ctx:
        route_meta["fallback_reason"] = "no_context_item"
        cold_rows = _cold_start_rows_with_relaxation(
            products=products,
            owned_active_ids=owned_active_ids,
            category=category,
            product_type=product_type,
            pop_map=pop_map,
            limit=limit,
        )
        if cold_rows:
            retrieval_diag.update(
                {
                    "sources": ["cold_start_trending"],
                    "final_candidates": int(len(cold_rows)),
                    "pop_nonzero_topn": int(sum(1 for v in pop_map.values() if float(v) > 0.0)),
                }
            )
            route_meta["retrieval"] = retrieval_diag
            return cold_rows, "cold_start:trending", None, route_meta
        route_meta["retrieval"] = retrieval_diag
        return heuristic_rows[:limit], "cooc_fallback:no_context_item", None, route_meta

    transitions = _aggregate_transition_counts(ctx_ids, co_dict)
    retrieval_diag["transitions_len"] = int(len(transitions))
    ranked_trans = sorted(transitions.items(), key=lambda x: (-x[1], x[0]))
    cooc_ids = [int(cid) for cid, _ in ranked_trans[:top_m]]

    heuristic_map: dict[int, dict[str, Any]] = {}
    for r in heuristic_rows:
        pid = _to_int((r.get("product") or {}).get("id"))
        if pid is None:
            continue
        heuristic_map[pid] = r

    global_pop_ids, by_cat_pop, by_type_pop, by_brand_pop = _popularity_indexes(by_id, pop_map)
    retrieval_diag["pop_nonzero_topn"] = int(sum(1 for pid in global_pop_ids[:top_m] if float(pop_map.get(pid, 0.0)) > 0.0))
    product_type_fb_ids = _product_type_fallback_ids(
        ctx_ids,
        by_id=by_id,
        by_type_pop=by_type_pop,
        per_type_topn=product_type_fb_topn,
    )
    category_fb_ids = _category_fallback_ids(
        ctx_ids,
        by_id=by_id,
        by_cat_pop=by_cat_pop,
        per_cat_topn=category_fb_topn,
    )
    brand_fb_ids = _brand_fallback_ids(
        ctx_ids,
        by_id=by_id,
        by_brand_pop=by_brand_pop,
        per_brand_topn=brand_fb_topn,
    )
    global_fb_ids = [int(pid) for pid in global_pop_ids[:global_fb_topn]]
    profile_ids = [int(pid) for pid in heuristic_map.keys()]

    candidate_ids: list[int] = []
    seen: set[int] = set()
    for source_ids in (
        cooc_ids,
        product_type_fb_ids,
        category_fb_ids,
        brand_fb_ids,
        global_fb_ids,
        profile_ids,
    ):
        for pid in source_ids:
            ipid = int(pid)
            if ipid in seen:
                continue
            seen.add(ipid)
            candidate_ids.append(ipid)
            if len(candidate_ids) >= top_m:
                break
        if len(candidate_ids) >= top_m:
            break
    if len(candidate_ids) < top_m:
        for pid in global_pop_ids:
            ipid = int(pid)
            if ipid in seen:
                continue
            seen.add(ipid)
            candidate_ids.append(ipid)
            if len(candidate_ids) >= top_m:
                break

    owned_set = {int(x) for x in owned_active_ids if _to_int(x) is not None}
    candidate_ids = [
        pid
        for pid in candidate_ids
        if _passes_runtime_candidate_filters(
            by_id.get(pid),
            owned_set=owned_set,
            category=category,
            product_type=product_type,
        )
    ][:top_m]
    if not candidate_ids:
        route_meta["fallback_reason"] = "no_candidates"
        cold_rows = _cold_start_rows_with_relaxation(
            products=products,
            owned_active_ids=owned_active_ids,
            category=category,
            product_type=product_type,
            pop_map=pop_map,
            limit=limit,
        )
        retrieval_diag.update(
            {
                "sources": ["cold_start_trending"],
                "cooc_candidates": int(len(cooc_ids)),
                "product_type_fallback_candidates": int(len(product_type_fb_ids)),
                "category_fallback_candidates": int(len(category_fb_ids)),
                "brand_fallback_candidates": int(len(brand_fb_ids)),
                "global_fallback_candidates": int(len(global_fb_ids)),
                "profile_candidates": int(len(profile_ids)),
                "final_candidates": int(len(cold_rows)),
            }
        )
        route_meta["retrieval"] = retrieval_diag
        if cold_rows:
            return cold_rows, "cold_start:trending", None, route_meta
        return heuristic_rows[:limit], "cooc_fallback:no_candidates", None, route_meta

    sources: list[str] = []
    if len(cooc_ids) > 0:
        sources.append(_cooc_source_label(co_source_resolved))
    if len(product_type_fb_ids) > 0:
        sources.append("product_type_fallback")
    if len(category_fb_ids) > 0:
        sources.append("category_fallback")
    if len(brand_fb_ids) > 0:
        sources.append("brand_fallback")
    if len(global_fb_ids) > 0:
        sources.append("global_fallback")
    if len(profile_ids) > 0:
        sources.append("profile_content")

    retrieval_diag.update(
        {
            "sources": sources,
            "cooc_candidates": int(len(cooc_ids)),
            "product_type_fallback_candidates": int(len(product_type_fb_ids)),
            "category_fallback_candidates": int(len(category_fb_ids)),
            "brand_fallback_candidates": int(len(brand_fb_ids)),
            "global_fallback_candidates": int(len(global_fb_ids)),
            "profile_candidates": int(len(profile_ids)),
            "final_candidates": int(len(candidate_ids)),
        }
    )
    route_meta["retrieval"] = retrieval_diag

    rank_map = {pid: idx + 1 for idx, pid in enumerate(candidate_ids)}
    model_features = _reranker_model_features()
    needs_user_ctx = any(f in USER_FEATURE_NAMES for f in model_features)
    user_ctx = _build_runtime_user_context(user_id, by_id) if needs_user_ctx else None
    route_meta["user_context_used"] = bool(user_ctx is not None)
    valid_ids, feats = _build_feature_rows(
        context_product=ctx,
        candidate_ids=candidate_ids,
        by_id=by_id,
        transitions=transitions,
        rank_map=rank_map,
        pop_map=pop_map,
        user_context=user_ctx if needs_user_ctx else None,
    )
    if not feats:
        route_meta["fallback_reason"] = "no_features"
        cold_rows = _cold_start_rows_with_relaxation(
            products=products,
            owned_active_ids=owned_active_ids,
            category=category,
            product_type=product_type,
            pop_map=pop_map,
            limit=limit,
        )
        if cold_rows:
            return cold_rows, "cold_start:trending", None, route_meta
        return heuristic_rows[:limit], "cooc_fallback:no_features", None, route_meta

    try:
        import numpy as np

        X = np.asarray(feats, dtype=float)
        probs = model.predict_proba(X)[:, 1]
    except Exception:
        route_meta["fallback_reason"] = "predict_error"
        cold_rows = _cold_start_rows_with_relaxation(
            products=products,
            owned_active_ids=owned_active_ids,
            category=category,
            product_type=product_type,
            pop_map=pop_map,
            limit=limit,
        )
        if cold_rows:
            return cold_rows, "cold_start:trending", None, route_meta
        return heuristic_rows[:limit], "cooc_fallback:predict_error", None, route_meta

    scored = sorted(zip(valid_ids, probs), key=lambda x: (-float(x[1]), x[0]))

    out: list[dict[str, Any]] = []
    for pid, prob in scored:
        base = heuristic_map.get(pid)
        if base:
            row = dict(base)
            comps = dict(row.get("components") or {})
            heuristic_score = round(float(base.get("score") or 0.0), 6)
        else:
            product_row = by_id.get(pid)
            if not product_row:
                continue
            pop = float(pop_map.get(pid, 0.0))
            trans = float(transitions.get(pid, 0.0))
            heuristic_score = round(float(trans + math.log1p(pop)), 6)
            row = {
                "product": product_row,
                "score": heuristic_score,
                "components": {
                    "mode": "retrieval_fallback",
                    "transition_count": round(float(trans), 6),
                    "popularity": round(float(pop), 6),
                },
                "why": ["candidate from runtime retrieval fallback"],
            }
            comps = dict(row.get("components") or {})
        row["score"] = round(float(prob), 6)
        row["components"] = {
            **comps,
            "mode": "reranker",
            "model_version": model_version,
            "reranker_prob": round(float(prob), 6),
            "heuristic_score": heuristic_score,
            "transition_count": round(float(transitions.get(pid, 0.0)), 6),
            "candidate_rank": int(rank_map.get(pid, 9999)),
        }
        out.append(row)
        if len(out) >= int(limit):
            break

    if not out:
        route_meta["fallback_reason"] = "reranker_empty_after_score"
        cold_rows = _cold_start_rows_with_relaxation(
            products=products,
            owned_active_ids=owned_active_ids,
            category=category,
            product_type=product_type,
            pop_map=pop_map,
            limit=limit,
        )
        if cold_rows:
            return cold_rows, "cold_start:trending", None, route_meta
        return heuristic_rows[:limit], "cooc_fallback:reranker_empty_after_score", None, route_meta

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
    owned_active_ids: list[int] | None = None,
    co_source: str | None = None,
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

    co_dict = co or {}
    co_source_resolved = _normalize_co_source(co_source or ("provided" if co_dict else "none"), co_dict)
    co_present = _has_nonempty_co_map(co_dict)
    pop_days = max(0, int(getattr(settings, "RECS_RUNTIME_POPULARITY_DAYS", 30) or 30))
    db_pop_map, db_pop_source = get_db_popularity_map(pop_days)
    pop_map = db_pop_map or _co_popularity(co_dict)
    pop_source = db_pop_source if db_pop_map else ("co_map" if pop_map else "none")

    if target_algo != "reranker":
        out = []
        for r in bundle_results[:limit]:
            row = dict(r)
            comps = dict(row.get("components") or {})
            comps.setdefault("mode", "cooc")
            row["components"] = comps
            out.append(row)
        route_meta["retrieval"] = {
            "products_count": int(len(products)),
            "co_present": bool(co_present),
            "co_keys": int(len(co_dict)),
            "co_source": co_source_resolved,
            "pop_source": pop_source,
            "pop_nonzero_topn": int(sum(1 for v in pop_map.values() if float(v) > 0.0)),
            "sources": [
                _cooc_source_label(co_source_resolved),
            ] if co_present else [],
            "final_candidates": int(len(out)),
        }
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
        cold_rows = _cold_start_rows_with_relaxation(
            products=products,
            owned_active_ids=list(owned_active_ids or []),
            category=None,
            product_type=None,
            pop_map=pop_map,
            limit=limit,
        )
        if cold_rows:
            return cold_rows, "cold_start:trending", None, route_meta
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
            trans = _to_float((co_dict.get(base_id, {}) or {}).get(pid), 0.0)
        transitions[pid] = float(trans)

        if len(candidate_ids) >= top_m:
            break

    if not candidate_ids:
        route_meta["fallback_reason"] = "no_candidates"
        cold_rows = _cold_start_rows_with_relaxation(
            products=products,
            owned_active_ids=list(owned_active_ids or []),
            category=str(ctx.get("category") or "") or None,
            product_type=str(ctx.get("product_type") or "") or None,
            pop_map=pop_map,
            limit=limit,
        )
        if cold_rows:
            return cold_rows, "cold_start:trending", None, route_meta
        return bundle_results[:limit], "cooc_fallback:no_candidates", None, route_meta

    rank_map = {pid: idx + 1 for idx, pid in enumerate(candidate_ids)}
    model_features = _reranker_model_features()
    needs_user_ctx = any(f in USER_FEATURE_NAMES for f in model_features)
    user_ctx = _build_runtime_user_context(user_id, by_id) if needs_user_ctx else None
    route_meta["user_context_used"] = bool(user_ctx is not None)
    valid_ids, feats = _build_feature_rows(
        context_product=ctx,
        candidate_ids=candidate_ids,
        by_id=by_id,
        transitions=transitions,
        rank_map=rank_map,
        pop_map=pop_map,
        user_context=user_ctx if needs_user_ctx else None,
    )
    if not feats:
        route_meta["fallback_reason"] = "no_features"
        cold_rows = _cold_start_rows_with_relaxation(
            products=products,
            owned_active_ids=list(owned_active_ids or []),
            category=str(ctx.get("category") or "") or None,
            product_type=str(ctx.get("product_type") or "") or None,
            pop_map=pop_map,
            limit=limit,
        )
        if cold_rows:
            return cold_rows, "cold_start:trending", None, route_meta
        return bundle_results[:limit], "cooc_fallback:no_features", None, route_meta

    try:
        import numpy as np

        X = np.asarray(feats, dtype=float)
        probs = model.predict_proba(X)[:, 1]
    except Exception:
        route_meta["fallback_reason"] = "predict_error"
        cold_rows = _cold_start_rows_with_relaxation(
            products=products,
            owned_active_ids=list(owned_active_ids or []),
            category=str(ctx.get("category") or "") or None,
            product_type=str(ctx.get("product_type") or "") or None,
            pop_map=pop_map,
            limit=limit,
        )
        if cold_rows:
            return cold_rows, "cold_start:trending", None, route_meta
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

    route_meta["retrieval"] = {
        "products_count": int(len(products)),
        "co_present": bool(co_present),
        "co_keys": int(len(co_dict)),
        "co_source": co_source_resolved,
        "pop_source": pop_source,
        "pop_nonzero_topn": int(sum(1 for v in pop_map.values() if float(v) > 0.0)),
        "transitions_len": int(len(transitions)),
        "sources": [
            _cooc_source_label(co_source_resolved),
        ] if len(transitions) > 0 else [],
        "final_candidates": int(len(candidate_ids)),
    }
    return out, "reranker", model_version, route_meta
