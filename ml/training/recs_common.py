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
) -> tuple[list[int], dict[int, float], dict[int, int]]:
    if not context_items:
        base: list[int] = []
        if fallback_items:
            base = [int(x) for x in fallback_items[:top_m]]
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


def build_feature_matrix_for_candidates(
    *,
    context_item: int,
    candidate_items: list[int],
    transition_counts: dict[int, float],
    candidate_ranks: dict[int, int] | None,
    items_lookup: pd.DataFrame,
    popularity_map: dict[int, float],
) -> np.ndarray:
    ctx_row = items_lookup.loc[context_item] if context_item in items_lookup.index else None
    ctx_cat = (ctx_row.get("category") if ctx_row is not None else None)
    ctx_brand = (ctx_row.get("brand") if ctx_row is not None else None)
    ctx_price = (
        float(ctx_row.get("price"))
        if ctx_row is not None and pd.notna(ctx_row.get("price"))
        else 0.0
    )

    feats: list[list[float]] = []
    for it in candidate_items:
        row = items_lookup.loc[it] if it in items_lookup.index else None
        cand_cat = (row.get("category") if row is not None else None)
        cand_brand = (row.get("brand") if row is not None else None)
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
        feats.append([trans, rank_inv, same_cat, same_brand, price_diff, np.log1p(popularity)])

    if not feats:
        return np.empty((0, 6), dtype=float)
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
