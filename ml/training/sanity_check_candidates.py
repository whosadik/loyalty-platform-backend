from __future__ import annotations

import argparse
import random
from typing import Any

import pandas as pd

try:
    from ml.training.recs_common import (
        build_behavior_next_item_map,
        build_category_popularity_map,
        build_brand_popularity_map,
        build_product_type_popularity_map,
        build_context_candidates,
        build_next_item_map,
        brand_fallback_for_context,
        category_fallback_for_context,
        merge_top_maps,
        normalize_user_id,
        parse_context_items,
        product_type_fallback_for_context,
        prepare_items_lookup,
        read_user_ids,
        to_item_id,
    )
except ModuleNotFoundError:
    from recs_common import (  # type: ignore
        build_behavior_next_item_map,
        build_category_popularity_map,
        build_brand_popularity_map,
        build_product_type_popularity_map,
        build_context_candidates,
        build_next_item_map,
        brand_fallback_for_context,
        category_fallback_for_context,
        merge_top_maps,
        normalize_user_id,
        parse_context_items,
        product_type_fallback_for_context,
        prepare_items_lookup,
        read_user_ids,
        to_item_id,
    )


def _load(path: str) -> pd.DataFrame:
    return pd.read_parquet(path)


def _bool(v: Any) -> str:
    return "yes" if bool(v) else "no"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--interactions", required=True)
    ap.add_argument("--items", required=True)
    ap.add_argument("--ds", required=True)
    ap.add_argument("--train_users", default="")
    ap.add_argument("--eval_users", default="")
    ap.add_argument("--top_m", type=int, default=1500)
    ap.add_argument("--context_k", type=int, default=10)
    ap.add_argument("--product_type_fallback_topn", type=int, default=400)
    ap.add_argument("--category_fallback_topn", type=int, default=400)
    ap.add_argument("--brand_fallback_topn", type=int, default=400)
    ap.add_argument(
        "--behavior_event_types",
        default="add_to_cart,click,purchase_attributed",
        help="Comma-separated event types for additional behavior transitions",
    )
    ap.add_argument("--behavior_weight", type=float, default=0.25)
    ap.add_argument("--sample_n", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    inter = _load(args.interactions)
    inter["user_id"] = inter["user_id"].map(normalize_user_id)
    inter = inter[inter["user_id"] != ""].copy()
    items = prepare_items_lookup(_load(args.items))
    ds = _load(args.ds).copy()
    ds["user_id"] = ds["user_id"].map(normalize_user_id)
    ds = ds[ds["user_id"] != ""].copy()

    train_users = read_user_ids(args.train_users) if args.train_users else set()
    eval_users = read_user_ids(args.eval_users) if args.eval_users else set()

    pur = inter[inter["event_type"] == "purchase"][["user_id", "item_id", "ts"]].copy()
    pur["user_id"] = pur["user_id"].map(normalize_user_id)
    pur["item_id"] = pur["item_id"].map(to_item_id)
    pur = pur[(pur["user_id"] != "") & pur["item_id"].notna()].copy()
    pur["item_id"] = pur["item_id"].astype(int)
    if train_users:
        pur = pur[pur["user_id"].isin(train_users)].copy()
    if eval_users:
        ds = ds[ds["user_id"].isin(eval_users)].copy()
        inter = inter[inter["user_id"].map(normalize_user_id).isin(train_users or eval_users)].copy()

    behavior_event_types = [
        x.strip().lower()
        for x in str(args.behavior_event_types or "").split(",")
        if x.strip()
    ]

    top_map_purchase = build_next_item_map(pur, top_m=args.top_m)
    top_map_behavior = build_behavior_next_item_map(
        inter,
        event_types=behavior_event_types,
        top_m=args.top_m,
    )
    top_map_mix = merge_top_maps(
        top_map_purchase,
        top_map_behavior,
        secondary_weight=max(0.0, float(args.behavior_weight)),
        top_m=args.top_m,
    )
    fallback_items = [int(x) for x in pur["item_id"].value_counts().index.tolist()]
    category_pop_map = build_category_popularity_map(
        pur,
        items,
        top_n=max(0, int(args.category_fallback_topn)),
    )
    product_type_pop_map = build_product_type_popularity_map(
        pur,
        items,
        top_n=max(0, int(args.product_type_fallback_topn)),
    )
    brand_pop_map = build_brand_popularity_map(
        pur,
        items,
        top_n=max(0, int(args.brand_fallback_topn)),
    )

    item_ids = set(int(x) for x in items.index.tolist()) if not items.empty else set()
    labels_total = 0
    labels_in_items = 0
    labels_in_cooc_only = 0
    labels_in_full = 0

    sample_rows = list(ds.itertuples(index=False))
    random.Random(args.seed).shuffle(sample_rows)
    sample_rows = sample_rows[: max(1, int(args.sample_n))]

    printed = 0
    print("SANITY CHECK: candidates + id consistency")
    print(
        f"rows_eval={len(ds)} top_m={args.top_m} context_k={args.context_k} "
        f"product_type_fallback_topn={args.product_type_fallback_topn} "
        f"category_fallback_topn={args.category_fallback_topn} brand_fallback_topn={args.brand_fallback_topn}"
    )
    for r in ds.itertuples(index=False):
        label = to_item_id(getattr(r, "label_item", None))
        if label is None:
            continue
        labels_total += 1
        if label in item_ids:
            labels_in_items += 1
        ctx_items = parse_context_items(
            getattr(r, "context_items", None),
            fallback_last_item=getattr(r, "context_last_item", None),
            max_k=args.context_k,
        )
        cands_cooc, _, _ = build_context_candidates(
            ctx_items,
            top_map_mix,
            top_m=args.top_m,
            fallback_items=[],
            category_fallback_items=[],
        )
        if label in cands_cooc:
            labels_in_cooc_only += 1
        cat_fb = category_fallback_for_context(
            ctx_items,
            items,
            category_pop_map,
            max_items=max(0, int(args.category_fallback_topn)),
        )
        type_fb = product_type_fallback_for_context(
            ctx_items,
            items,
            product_type_pop_map,
            max_items=max(0, int(args.product_type_fallback_topn)),
        )
        brand_fb = brand_fallback_for_context(
            ctx_items,
            items,
            brand_pop_map,
            max_items=max(0, int(args.brand_fallback_topn)),
        )
        cands_full, _, _ = build_context_candidates(
            ctx_items,
            top_map_mix,
            top_m=args.top_m,
            fallback_items=fallback_items,
            product_type_fallback_items=type_fb,
            category_fallback_items=cat_fb,
            brand_fallback_items=brand_fb,
        )
        if label in cands_full:
            labels_in_full += 1

    for r in sample_rows:
        label = to_item_id(getattr(r, "label_item", None))
        if label is None:
            continue
        ctx_items = parse_context_items(
            getattr(r, "context_items", None),
            fallback_last_item=getattr(r, "context_last_item", None),
            max_k=args.context_k,
        )
        cands_cooc, _, _ = build_context_candidates(
            ctx_items,
            top_map_mix,
            top_m=args.top_m,
            fallback_items=[],
            category_fallback_items=[],
        )
        cat_fb = category_fallback_for_context(
            ctx_items,
            items,
            category_pop_map,
            max_items=max(0, int(args.category_fallback_topn)),
        )
        type_fb = product_type_fallback_for_context(
            ctx_items,
            items,
            product_type_pop_map,
            max_items=max(0, int(args.product_type_fallback_topn)),
        )
        brand_fb = brand_fallback_for_context(
            ctx_items,
            items,
            brand_pop_map,
            max_items=max(0, int(args.brand_fallback_topn)),
        )
        cands_full, _, _ = build_context_candidates(
            ctx_items,
            top_map_mix,
            top_m=args.top_m,
            fallback_items=fallback_items,
            product_type_fallback_items=type_fb,
            category_fallback_items=cat_fb,
            brand_fallback_items=brand_fb,
        )
        print(
            "sample",
            {
                "user_id": str(getattr(r, "user_id", "")),
                "label_item": int(label),
                "label_in_items": _bool(label in item_ids),
                "in_candidates_cooc_only": _bool(label in cands_cooc),
                "in_candidates_full": _bool(label in cands_full),
                "context_len": len(ctx_items),
                "cooc_candidates": len(cands_cooc),
                "full_candidates": len(cands_full),
            },
        )
        printed += 1
        if printed >= max(1, int(args.sample_n)):
            break

    cov_cooc = (labels_in_cooc_only / labels_total) if labels_total else 0.0
    cov_full = (labels_in_full / labels_total) if labels_total else 0.0
    id_match = (labels_in_items / labels_total) if labels_total else 0.0

    print("summary", {
        "labels_total": int(labels_total),
        "labels_in_items": int(labels_in_items),
        "label_item_id_match_rate": round(float(id_match), 4),
        "coverage_cooc_only": round(float(cov_cooc), 4),
        "coverage_with_fallback": round(float(cov_full), 4),
        "coverage_lift": round(float(cov_full - cov_cooc), 4),
    })


if __name__ == "__main__":
    main()
