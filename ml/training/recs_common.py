from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def normalize_user_id(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    return str(value)


def to_item_id(value: Any) -> int | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    try:
        return int(value)
    except Exception:
        try:
            return int(float(value))
        except Exception:
            return None


def read_user_ids(path: str) -> set[str]:
    p = Path(path)
    if not p.exists():
        return set()
    out: set[str] = set()
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            v = line.strip()
            if v:
                out.add(v)
    return out


def write_user_ids(path: str, user_ids: list[str] | set[str]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    vals = sorted({str(x) for x in user_ids if str(x)})
    with p.open("w", encoding="utf-8") as f:
        for u in vals:
            f.write(u + "\n")


def build_next_item_map(purchases: pd.DataFrame, top_m: int = 50) -> dict[int, list[tuple[int, int]]]:
    """
    Sequential baseline:
    counts transitions item_t -> item_{t+1} per user timeline.
    """
    if purchases.empty:
        return {}

    work = purchases[["user_id", "item_id", "ts"]].copy()
    work["user_id"] = work["user_id"].map(normalize_user_id)
    work["item_id"] = work["item_id"].map(to_item_id)
    work = work[(work["user_id"] != "") & work["item_id"].notna()].copy()
    if work.empty:
        return {}

    work = work.sort_values(["user_id", "ts"])

    nxt = defaultdict(lambda: defaultdict(int))
    for _, grp in work.groupby("user_id", sort=False):
        seq = [int(x) for x in grp["item_id"].tolist()]
        if len(seq) < 2:
            continue
        for a, b in zip(seq[:-1], seq[1:]):
            nxt[a][b] += 1

    top: dict[int, list[tuple[int, int]]] = {}
    for a, m in nxt.items():
        top[a] = sorted(m.items(), key=lambda x: (-x[1], x[0]))[:top_m]
    return top


def merge_top_maps(
    primary: dict[int, list[tuple[int, int]]],
    secondary: dict[int, list[tuple[int, int]]],
    *,
    secondary_weight: float = 0.3,
    top_m: int = 200,
) -> dict[int, list[tuple[int, int]]]:
    if not primary and not secondary:
        return {}
    if not secondary or secondary_weight <= 0:
        return {k: list(v[:top_m]) for k, v in primary.items()}

    agg: dict[int, dict[int, float]] = defaultdict(lambda: defaultdict(float))
    for src, pairs in (primary or {}).items():
        for dst, cnt in pairs:
            agg[int(src)][int(dst)] += float(cnt or 0.0)
    w = float(secondary_weight)
    for src, pairs in (secondary or {}).items():
        for dst, cnt in pairs:
            agg[int(src)][int(dst)] += float(cnt or 0.0) * w

    out: dict[int, list[tuple[int, int]]] = {}
    for src, rel in agg.items():
        ranked = sorted(rel.items(), key=lambda x: (-x[1], x[0]))[: int(top_m)]
        # keep int counts interface (downstream expects ints)
        out[int(src)] = [(int(dst), int(round(float(score)))) for dst, score in ranked]
    return out


def build_behavior_next_item_map(
    interactions: pd.DataFrame,
    *,
    event_types: list[str] | set[str],
    top_m: int = 200,
) -> dict[int, list[tuple[int, int]]]:
    if interactions.empty:
        return {}

    ev_set = {str(x).strip().lower() for x in (event_types or []) if str(x).strip()}
    if not ev_set:
        return {}

    work = interactions[["user_id", "item_id", "ts", "event_type"]].copy()
    work["user_id"] = work["user_id"].map(normalize_user_id)
    work["item_id"] = work["item_id"].map(to_item_id)
    work["event_type"] = work["event_type"].astype(str).str.strip().str.lower()
    work = work[
        (work["user_id"] != "")
        & work["item_id"].notna()
        & work["event_type"].isin(ev_set)
    ].copy()
    if work.empty:
        return {}

    work["item_id"] = work["item_id"].astype(int)
    work = work.sort_values(["user_id", "ts"])

    nxt = defaultdict(lambda: defaultdict(int))
    for _, grp in work.groupby("user_id", sort=False):
        seq = [int(x) for x in grp["item_id"].tolist()]
        if len(seq) < 2:
            continue
        for a, b in zip(seq[:-1], seq[1:]):
            if a == b:
                continue
            nxt[a][b] += 1

    top: dict[int, list[tuple[int, int]]] = {}
    for a, m in nxt.items():
        top[int(a)] = sorted(m.items(), key=lambda x: (-x[1], x[0]))[: int(top_m)]
    return top


def parse_context_items(raw: Any, fallback_last_item: Any = None, max_k: int = 3) -> list[int]:
    vals: list[int] = []

    def _push(v: Any):
        iid = to_item_id(v)
        if iid is not None:
            vals.append(int(iid))

    if isinstance(raw, list):
        for x in raw:
            _push(x)
    elif isinstance(raw, tuple):
        for x in list(raw):
            _push(x)
    elif isinstance(raw, str) and raw.strip():
        s = raw.strip()
        try:
            arr = json.loads(s)
            if isinstance(arr, list):
                for x in arr:
                    _push(x)
        except Exception:
            _push(raw)
    elif raw is not None:
        _push(raw)

    if not vals and fallback_last_item is not None:
        _push(fallback_last_item)

    # keep unique, preserve order and only last max_k
    dedup = list(dict.fromkeys(vals))
    return dedup[-max(1, int(max_k)) :]


def build_context_candidates(
    context_items: list[int],
    top_map: dict[int, list[tuple[int, int]]],
    top_m: int = 200,
    fallback_items: list[int] | None = None,
    product_type_fallback_items: list[int] | None = None,
    category_fallback_items: list[int] | None = None,
    brand_fallback_items: list[int] | None = None,
) -> tuple[list[int], dict[int, float], dict[int, int]]:
    if not context_items:
        base: list[int] = []
        if category_fallback_items:
            base = [int(x) for x in category_fallback_items[:top_m]]
        if fallback_items:
            seen = set(base)
            for fi in fallback_items:
                f = int(fi)
                if f in seen:
                    continue
                base.append(f)
                seen.add(f)
                if len(base) >= top_m:
                    break
        scores = {int(i): 0.0 for i in base}
        ranks = {int(i): int(pos + 1) for pos, i in enumerate(base)}
        return base, scores, ranks

    # context_items expected oldest -> newest; newest gets largest weight
    recency_weights = [1.0, 0.7, 0.5, 0.3]
    agg: dict[int, float] = defaultdict(float)
    for idx, ctx in enumerate(reversed(context_items)):
        w = recency_weights[idx] if idx < len(recency_weights) else 0.2
        for cand, cnt in top_map.get(int(ctx), []):
            agg[int(cand)] += float(cnt) * float(w)

    ranked_pairs = sorted(agg.items(), key=lambda x: (-x[1], x[0]))[:top_m]
    cands = [int(i) for i, _ in ranked_pairs]
    scores = {int(i): float(s) for i, s in ranked_pairs}
    if product_type_fallback_items and len(cands) < top_m:
        seen = set(cands)
        for ti in product_type_fallback_items:
            t = int(ti)
            if t in seen:
                continue
            cands.append(t)
            scores[t] = 0.0
            seen.add(t)
            if len(cands) >= top_m:
                break
    if category_fallback_items and len(cands) < top_m:
        seen = set(cands)
        for ci in category_fallback_items:
            c = int(ci)
            if c in seen:
                continue
            cands.append(c)
            scores[c] = 0.0
            seen.add(c)
            if len(cands) >= top_m:
                break
    if brand_fallback_items and len(cands) < top_m:
        seen = set(cands)
        for bi in brand_fallback_items:
            b = int(bi)
            if b in seen:
                continue
            cands.append(b)
            scores[b] = 0.0
            seen.add(b)
            if len(cands) >= top_m:
                break
    if fallback_items and len(cands) < top_m:
        seen = set(cands)
        for fi in fallback_items:
            f = int(fi)
            if f in seen:
                continue
            cands.append(f)
            scores[f] = 0.0
            seen.add(f)
            if len(cands) >= top_m:
                break
    cands = cands[:top_m]
    ranks = {int(i): int(pos + 1) for pos, (i, _) in enumerate(ranked_pairs)}
    for pos, i in enumerate(cands):
        if int(i) not in ranks:
            ranks[int(i)] = int(pos + 1)
    return cands, scores, ranks


def build_category_popularity_map(
    purchases: pd.DataFrame,
    items_lookup: pd.DataFrame,
    *,
    top_n: int = 300,
) -> dict[str, list[int]]:
    k = max(0, int(top_n or 0))
    if k == 0 or purchases.empty or items_lookup.empty:
        return {}

    cat_map = items_lookup["category"].to_dict() if "category" in items_lookup.columns else {}
    work = purchases[["item_id"]].copy()
    work["item_id"] = work["item_id"].map(to_item_id)
    work = work[work["item_id"].notna()].copy()
    if work.empty:
        return {}
    work["item_id"] = work["item_id"].astype(int)
    work["category"] = work["item_id"].map(cat_map)
    work = work[work["category"].notna()].copy()
    if work.empty:
        return {}

    agg = (
        work.groupby(["category", "item_id"], as_index=False)
        .size()
        .rename(columns={"size": "cnt"})
        .sort_values(["category", "cnt", "item_id"], ascending=[True, False, True])
    )
    out: dict[str, list[int]] = {}
    for cat, grp in agg.groupby("category", sort=False):
        out[str(cat)] = [int(x) for x in grp["item_id"].tolist()[:k]]
    return out


def build_product_type_popularity_map(
    purchases: pd.DataFrame,
    items_lookup: pd.DataFrame,
    *,
    top_n: int = 300,
) -> dict[str, list[int]]:
    k = max(0, int(top_n or 0))
    if k == 0 or purchases.empty or items_lookup.empty:
        return {}

    type_map = (
        items_lookup["product_type"].to_dict()
        if "product_type" in items_lookup.columns
        else {}
    )
    work = purchases[["item_id"]].copy()
    work["item_id"] = work["item_id"].map(to_item_id)
    work = work[work["item_id"].notna()].copy()
    if work.empty:
        return {}
    work["item_id"] = work["item_id"].astype(int)
    work["product_type"] = work["item_id"].map(type_map)
    work = work[work["product_type"].notna()].copy()
    work["product_type"] = work["product_type"].astype(str).str.strip().str.lower()
    work = work[(work["product_type"] != "") & (work["product_type"] != "nan")].copy()
    if work.empty:
        return {}

    agg = (
        work.groupby(["product_type", "item_id"], as_index=False)
        .size()
        .rename(columns={"size": "cnt"})
        .sort_values(["product_type", "cnt", "item_id"], ascending=[True, False, True])
    )
    out: dict[str, list[int]] = {}
    for ptype, grp in agg.groupby("product_type", sort=False):
        out[str(ptype)] = [int(x) for x in grp["item_id"].tolist()[:k]]
    return out


def category_fallback_for_context(
    context_items: list[int],
    items_lookup: pd.DataFrame,
    category_popularity: dict[str, list[int]],
    *,
    max_items: int = 300,
) -> list[int]:
    lim = max(0, int(max_items or 0))
    if lim == 0 or not context_items or items_lookup.empty or not category_popularity:
        return []

    cat_map = items_lookup["category"].to_dict() if "category" in items_lookup.columns else {}
    ordered_cats: list[str] = []
    seen_cats: set[str] = set()
    for ctx in reversed(context_items):
        c = cat_map.get(int(ctx))
        if c is None:
            continue
        cs = str(c)
        if cs in seen_cats:
            continue
        seen_cats.add(cs)
        ordered_cats.append(cs)

    out: list[int] = []
    seen_items: set[int] = set()
    for cat in ordered_cats:
        for item_id in category_popularity.get(cat, []):
            iid = int(item_id)
            if iid in seen_items:
                continue
            out.append(iid)
            seen_items.add(iid)
            if len(out) >= lim:
                return out
    return out


def product_type_fallback_for_context(
    context_items: list[int],
    items_lookup: pd.DataFrame,
    product_type_popularity: dict[str, list[int]],
    *,
    max_items: int = 300,
) -> list[int]:
    lim = max(0, int(max_items or 0))
    if lim == 0 or not context_items or items_lookup.empty or not product_type_popularity:
        return []

    type_map = (
        items_lookup["product_type"].to_dict()
        if "product_type" in items_lookup.columns
        else {}
    )
    ordered_types: list[str] = []
    seen_types: set[str] = set()
    for ctx in reversed(context_items):
        ptype = type_map.get(int(ctx))
        if ptype is None:
            continue
        ps = str(ptype).strip().lower()
        if not ps or ps == "nan" or ps in seen_types:
            continue
        seen_types.add(ps)
        ordered_types.append(ps)

    out: list[int] = []
    seen_items: set[int] = set()
    for ptype in ordered_types:
        for item_id in product_type_popularity.get(ptype, []):
            iid = int(item_id)
            if iid in seen_items:
                continue
            out.append(iid)
            seen_items.add(iid)
            if len(out) >= lim:
                return out
    return out


def build_brand_popularity_map(
    purchases: pd.DataFrame,
    items_lookup: pd.DataFrame,
    *,
    top_n: int = 150,
) -> dict[str, list[int]]:
    k = max(0, int(top_n or 0))
    if k == 0 or purchases.empty or items_lookup.empty:
        return {}

    brand_map = items_lookup["brand"].to_dict() if "brand" in items_lookup.columns else {}
    work = purchases[["item_id"]].copy()
    work["item_id"] = work["item_id"].map(to_item_id)
    work = work[work["item_id"].notna()].copy()
    if work.empty:
        return {}
    work["item_id"] = work["item_id"].astype(int)
    work["brand"] = work["item_id"].map(brand_map)
    work = work[work["brand"].notna()].copy()
    work["brand"] = work["brand"].astype(str).str.strip().str.lower()
    work = work[(work["brand"] != "") & (work["brand"] != "nan")].copy()
    if work.empty:
        return {}

    agg = (
        work.groupby(["brand", "item_id"], as_index=False)
        .size()
        .rename(columns={"size": "cnt"})
        .sort_values(["brand", "cnt", "item_id"], ascending=[True, False, True])
    )
    out: dict[str, list[int]] = {}
    for brand, grp in agg.groupby("brand", sort=False):
        out[str(brand)] = [int(x) for x in grp["item_id"].tolist()[:k]]
    return out


def brand_fallback_for_context(
    context_items: list[int],
    items_lookup: pd.DataFrame,
    brand_popularity: dict[str, list[int]],
    *,
    max_items: int = 150,
) -> list[int]:
    lim = max(0, int(max_items or 0))
    if lim == 0 or not context_items or items_lookup.empty or not brand_popularity:
        return []

    brand_map = items_lookup["brand"].to_dict() if "brand" in items_lookup.columns else {}
    ordered_brands: list[str] = []
    seen_brands: set[str] = set()
    for ctx in reversed(context_items):
        b = brand_map.get(int(ctx))
        if b is None:
            continue
        bs = str(b).strip().lower()
        if not bs or bs == "nan" or bs in seen_brands:
            continue
        seen_brands.add(bs)
        ordered_brands.append(bs)

    out: list[int] = []
    seen_items: set[int] = set()
    for brand in ordered_brands:
        for item_id in brand_popularity.get(brand, []):
            iid = int(item_id)
            if iid in seen_items:
                continue
            out.append(iid)
            seen_items.add(iid)
            if len(out) >= lim:
                return out
    return out


def recall_at_k(ds: pd.DataFrame, ranked_map: dict[int, list[int]], k: int) -> tuple[float, int, int]:
    hit = 0
    total = len(ds)
    for r in ds.itertuples(index=False):
        ctx = to_item_id(getattr(r, "context_last_item", None))
        label = to_item_id(getattr(r, "label_item", None))
        if ctx is None or label is None:
            continue
        cands = ranked_map.get(ctx, [])[:k]
        if label in cands:
            hit += 1
    return (hit / total if total else 0.0), hit, total


def recall_at_k_from_context(
    ds: pd.DataFrame,
    top_map: dict[int, list[tuple[int, int]]],
    k: int,
    *,
    top_m: int,
    context_k: int = 3,
) -> tuple[float, int, int]:
    hit = 0
    total = len(ds)
    for r in ds.itertuples(index=False):
        label = to_item_id(getattr(r, "label_item", None))
        if label is None:
            continue
        ctx_items = parse_context_items(
            getattr(r, "context_items", None),
            fallback_last_item=getattr(r, "context_last_item", None),
            max_k=context_k,
        )
        cands, _, _ = build_context_candidates(ctx_items, top_map, top_m=top_m)
        if label in cands[:k]:
            hit += 1
    return (hit / total if total else 0.0), hit, total


def prepare_items_lookup(items_df: pd.DataFrame) -> pd.DataFrame:
    out = items_df.copy()
    out["item_id"] = out["item_id"].map(to_item_id)
    out = out[out["item_id"].notna()].copy()
    out["item_id"] = out["item_id"].astype(int)
    if "price" in out.columns:
        out["price"] = pd.to_numeric(out["price"], errors="coerce")
    if "category" not in out.columns:
        out["category"] = None
    if "brand" not in out.columns:
        out["brand"] = None
    return out.drop_duplicates("item_id").set_index("item_id")


BASE_FEATURE_NAMES: list[str] = [
    "transition_count",
    "rank_inv",
    "same_category",
    "same_brand",
    "price_diff",
    "log_popularity",
]
USER_FEATURE_NAMES: list[str] = [
    "u_tx_count_90d",
    "u_avg_price_90d_log",
    "u_top_cat_match",
    "u_top_brand_match",
    "u_top_ptype_match",
    "u_cat_affinity",
    "u_price_fit",
]


def build_user_context(
    *,
    user_purchases: pd.DataFrame,
    items_lookup: pd.DataFrame,
    cutoff_ts: Any,
    window_days: int = 90,
) -> dict[str, Any]:
    """Summarize a single user's purchase history up to `cutoff_ts`.

    Returns a plain dict with numeric + categorical summary fields. Used
    by `build_feature_matrix_for_candidates` to derive per-candidate
    user-context features (top-category match, price fit, etc.).

    All values default to neutrals when the user has no qualifying
    history, so the downstream feature matrix never needs to special-case
    cold-start users.
    """
    ctx = {
        "u_tx_count_90d": 0.0,
        "u_avg_price_90d": 0.0,
        "u_top_cat": None,
        "u_top_brand": None,
        "u_top_ptype": None,
        "u_cat_share": {},
    }
    if user_purchases is None or user_purchases.empty:
        return ctx

    try:
        cutoff = pd.Timestamp(cutoff_ts)
    except Exception:
        return ctx
    if cutoff.tzinfo is None:
        cutoff = cutoff.tz_localize("UTC")
    window_start = cutoff - pd.Timedelta(days=int(window_days))

    df = user_purchases.copy()
    if "ts" not in df.columns or "item_id" not in df.columns:
        return ctx
    df["ts"] = pd.to_datetime(df["ts"], errors="coerce", utc=True)
    df = df[df["ts"].notna()].copy()
    df = df[(df["ts"] >= window_start) & (df["ts"] < cutoff)]
    if df.empty:
        return ctx

    df["item_id"] = df["item_id"].map(to_item_id)
    df = df[df["item_id"].notna()].copy()
    df["item_id"] = df["item_id"].astype(int)
    if df.empty:
        return ctx

    joined = df.join(items_lookup, on="item_id", how="left", rsuffix="_i")
    prices = pd.to_numeric(joined.get("price"), errors="coerce")
    prices = prices[prices.notna()]
    avg_price = float(prices.mean()) if not prices.empty else 0.0

    cats = joined["category"].dropna().astype(str).str.lower()
    brands = joined["brand"].dropna().astype(str).str.lower()
    ptypes = (
        joined["product_type"].dropna().astype(str).str.lower()
        if "product_type" in joined.columns else pd.Series(dtype=str)
    )

    top_cat = cats.value_counts().idxmax() if not cats.empty else None
    top_brand = brands.value_counts().idxmax() if not brands.empty else None
    top_ptype = ptypes.value_counts().idxmax() if not ptypes.empty else None

    cat_total = int(cats.count())
    cat_share: dict[str, float] = {}
    if cat_total > 0:
        cat_share = {
            str(k): float(v) / float(cat_total)
            for k, v in cats.value_counts().to_dict().items()
        }

    ctx.update(
        u_tx_count_90d=float(len(df)),
        u_avg_price_90d=avg_price,
        u_top_cat=top_cat,
        u_top_brand=top_brand,
        u_top_ptype=top_ptype,
        u_cat_share=cat_share,
    )
    return ctx


def _user_feature_row(
    *,
    cand_cat: Any,
    cand_brand: Any,
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

    cc = str(cand_cat).lower() if cand_cat is not None else ""
    cb = str(cand_brand).lower() if cand_brand is not None else ""
    cp = str(cand_ptype).lower() if cand_ptype is not None else ""

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
        float(np.log1p(avg_price)),
        top_cat_match,
        top_brand_match,
        top_ptype_match,
        cat_affinity,
        price_fit,
    ]


def build_feature_matrix_for_candidates(
    *,
    context_item: int,
    candidate_items: list[int],
    transition_counts: dict[int, float],
    candidate_ranks: dict[int, int] | None,
    items_lookup: pd.DataFrame,
    popularity_map: dict[int, float],
    user_context: dict[str, Any] | None = None,
) -> np.ndarray:
    """Build the feature matrix for a list of candidate items.

    When `user_context` is None (legacy behavior), returns a 6-column
    matrix with the classic co-occurrence features. When `user_context`
    is provided, appends 7 user-context features so the learned model
    can personalize beyond item→item co-occurrence.

    The column order is `BASE_FEATURE_NAMES + USER_FEATURE_NAMES` and is
    locked — runtime inference matches the model's trained feature list
    via metadata.json.
    """
    ctx_row = items_lookup.loc[context_item] if context_item in items_lookup.index else None
    ctx_cat = (ctx_row.get("category") if ctx_row is not None else None)
    ctx_brand = (ctx_row.get("brand") if ctx_row is not None else None)
    ctx_price = (
        float(ctx_row.get("price"))
        if ctx_row is not None and pd.notna(ctx_row.get("price"))
        else 0.0
    )

    use_user_ctx = user_context is not None
    n_cols = len(BASE_FEATURE_NAMES) + (len(USER_FEATURE_NAMES) if use_user_ctx else 0)

    feats: list[list[float]] = []
    for it in candidate_items:
        row = items_lookup.loc[it] if it in items_lookup.index else None
        cand_cat = (row.get("category") if row is not None else None)
        cand_brand = (row.get("brand") if row is not None else None)
        cand_ptype = (row.get("product_type") if row is not None else None)
        cand_price = (
            float(row.get("price"))
            if row is not None and pd.notna(row.get("price"))
            else 0.0
        )
        same_cat = 1.0 if (cand_cat is not None and ctx_cat is not None and cand_cat == ctx_cat) else 0.0
        same_brand = 1.0 if (cand_brand is not None and ctx_brand is not None and cand_brand == ctx_brand) else 0.0
        price_diff = abs(cand_price - ctx_price)
        popularity = float(popularity_map.get(int(it), 0.0))
        trans = float(transition_counts.get(int(it), 0.0))
        rank = int((candidate_ranks or {}).get(int(it), 9999))
        rank_inv = 1.0 / float(rank + 1)

        base = [trans, rank_inv, same_cat, same_brand, price_diff, np.log1p(popularity)]
        if use_user_ctx:
            base.extend(
                _user_feature_row(
                    cand_cat=cand_cat,
                    cand_brand=cand_brand,
                    cand_ptype=cand_ptype,
                    cand_price=cand_price,
                    user_ctx=user_context,
                )
            )
        feats.append(base)

    if not feats:
        return np.empty((0, n_cols), dtype=float)
    return np.array(feats, dtype=float)


def candidate_coverage(ds: pd.DataFrame, ranked_map: dict[int, list[int]]) -> tuple[float, int, int]:
    covered = 0
    total = len(ds)
    for r in ds.itertuples(index=False):
        ctx = to_item_id(getattr(r, "context_last_item", None))
        label = to_item_id(getattr(r, "label_item", None))
        if ctx is None or label is None:
            continue
        cands = ranked_map.get(ctx, [])
        if label in cands:
            covered += 1
    return (covered / total if total else 0.0), covered, total


def candidate_coverage_from_context(
    ds: pd.DataFrame,
    top_map: dict[int, list[tuple[int, int]]],
    *,
    top_m: int,
    context_k: int = 3,
) -> tuple[float, int, int]:
    covered = 0
    total = len(ds)
    for r in ds.itertuples(index=False):
        label = to_item_id(getattr(r, "label_item", None))
        if label is None:
            continue
        ctx_items = parse_context_items(
            getattr(r, "context_items", None),
            fallback_last_item=getattr(r, "context_last_item", None),
            max_k=context_k,
        )
        cands, _, _ = build_context_candidates(ctx_items, top_map, top_m=top_m)
        if label in cands:
            covered += 1
    return (covered / total if total else 0.0), covered, total
