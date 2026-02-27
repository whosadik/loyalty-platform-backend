from __future__ import annotations

import argparse
import json
import os
from typing import Any

import joblib
import numpy as np
import pandas as pd

try:
    from ml.training.recs_common import (
        build_category_popularity_map,
        build_behavior_next_item_map,
        build_brand_popularity_map,
        build_product_type_popularity_map,
        build_feature_matrix_for_candidates,
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
        build_category_popularity_map,
        build_behavior_next_item_map,
        build_brand_popularity_map,
        build_product_type_popularity_map,
        build_feature_matrix_for_candidates,
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


def _to_float(v: Any) -> float | None:
    try:
        if v is None:
            return None
        if pd.isna(v):
            return None
        return float(v)
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--interactions", required=True)
    ap.add_argument("--items", required=True)
    ap.add_argument("--ds", required=True)
    ap.add_argument("--model", required=True)
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
    ap.add_argument("--train_users", default="")
    ap.add_argument("--eval_users", default="")
    ap.add_argument("--out_report", default="")
    ap.add_argument("--out_json", default="")
    args = ap.parse_args()

    if args.out_report:
        os.makedirs(os.path.dirname(args.out_report), exist_ok=True)
    if args.out_json:
        os.makedirs(os.path.dirname(args.out_json), exist_ok=True)

    inter = pd.read_parquet(args.interactions)
    inter["user_id"] = inter["user_id"].map(normalize_user_id)
    inter = inter[inter["user_id"] != ""].copy()
    items = prepare_items_lookup(pd.read_parquet(args.items))
    ds = pd.read_parquet(args.ds).copy()
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
    if pur.empty:
        raise SystemExit("No purchases available for reranker eval")

    pop = pur["item_id"].value_counts().to_dict()
    fallback_items = [int(x) for x in pur["item_id"].value_counts().index.tolist()]
    model = joblib.load(args.model)
    behavior_event_types = [
        x.strip().lower()
        for x in str(args.behavior_event_types or "").split(",")
        if x.strip()
    ]
    top_map_purchase = build_next_item_map(pur, top_m=args.top_m)
    top_map_behavior = build_behavior_next_item_map(
        inter[inter["user_id"].isin(train_users)] if train_users else inter,
        event_types=behavior_event_types,
        top_m=args.top_m,
    )
    top_map = merge_top_maps(
        top_map_purchase,
        top_map_behavior,
        secondary_weight=max(0.0, float(args.behavior_weight)),
        top_m=args.top_m,
    )
    category_pop_map = build_category_popularity_map(
        pur,
        items,
        top_n=max(0, int(args.category_fallback_topn or 0)),
    )
    product_type_pop_map = build_product_type_popularity_map(
        pur,
        items,
        top_n=max(0, int(args.product_type_fallback_topn or 0)),
    )
    brand_pop_map = build_brand_popularity_map(
        pur,
        items,
        top_n=max(0, int(args.brand_fallback_topn or 0)),
    )
    price_map = items["price"].to_dict() if "price" in items.columns else {}
    popularity_map = {int(k): float(v) for k, v in pop.items()}

    hit10 = 0
    hit20 = 0
    covered = 0
    total = len(ds)

    unique_top10: set[int] = set()
    unique_top20: set[int] = set()
    avg_cands_sum = 0

    rec_price_sum10 = 0.0
    rec_price_cnt10 = 0
    rec_price_sum20 = 0.0
    rec_price_cnt20 = 0

    label_rev_total = 0.0
    label_rev_hit10 = 0.0
    label_rev_hit20 = 0.0

    for r in ds.itertuples(index=False):
        ctx_last = getattr(r, "context_last_item", None)
        ctx_items = parse_context_items(
            getattr(r, "context_items", None),
            fallback_last_item=ctx_last,
            max_k=args.context_k,
        )
        if not ctx_items:
            continue

        category_fallback = category_fallback_for_context(
            ctx_items,
            items,
            category_pop_map,
            max_items=max(0, int(args.category_fallback_topn or 0)),
        )
        product_type_fallback = product_type_fallback_for_context(
            ctx_items,
            items,
            product_type_pop_map,
            max_items=max(0, int(args.product_type_fallback_topn or 0)),
        )
        brand_fallback = brand_fallback_for_context(
            ctx_items,
            items,
            brand_pop_map,
            max_items=max(0, int(args.brand_fallback_topn or 0)),
        )
        candidate_items, counts, ranks = build_context_candidates(
            ctx_items,
            top_map,
            top_m=args.top_m,
            fallback_items=fallback_items,
            product_type_fallback_items=product_type_fallback,
            category_fallback_items=category_fallback,
            brand_fallback_items=brand_fallback,
        )
        if not candidate_items:
            continue
        X = build_feature_matrix_for_candidates(
            context_item=int(ctx_items[-1]),
            candidate_items=candidate_items,
            transition_counts=counts,
            candidate_ranks=ranks,
            items_lookup=items,
            popularity_map=popularity_map,
        )
        if X.shape[0] == 0:
            continue
        p = model.predict_proba(X)[:, 1]
        order = np.argsort(-p)
        ranked = [candidate_items[int(i)] for i in order]

        label = to_item_id(getattr(r, "label_item", None))
        if label is None:
            continue
        avg_cands_sum += len(candidate_items)

        top10 = ranked[:10]
        top20 = ranked[:20]
        unique_top10.update(top10)
        unique_top20.update(top20)

        if label in candidate_items:
            covered += 1
        if label in top10:
            hit10 += 1
        if label in top20:
            hit20 += 1

        for pid in top10:
            p10 = _to_float(price_map.get(int(pid)))
            if p10 is None:
                continue
            rec_price_sum10 += p10
            rec_price_cnt10 += 1
        for pid in top20:
            p20 = _to_float(price_map.get(int(pid)))
            if p20 is None:
                continue
            rec_price_sum20 += p20
            rec_price_cnt20 += 1

        label_price = _to_float(getattr(r, "label_price", None))
        if label_price is not None and label_price > 0:
            label_rev_total += label_price
            if label in top10:
                label_rev_hit10 += label_price
            if label in top20:
                label_rev_hit20 += label_price

    r10 = (hit10 / total) if total else 0.0
    r20 = (hit20 / total) if total else 0.0
    cov = (covered / total) if total else 0.0
    avg_cands = (avg_cands_sum / total) if total else 0.0

    catalog_size = int(len(items.index)) if not items.empty else int(pur["item_id"].nunique())
    catalog_cov10 = (len(unique_top10) / catalog_size) if catalog_size else 0.0
    catalog_cov20 = (len(unique_top20) / catalog_size) if catalog_size else 0.0

    avg_price10 = (rec_price_sum10 / rec_price_cnt10) if rec_price_cnt10 else 0.0
    avg_price20 = (rec_price_sum20 / rec_price_cnt20) if rec_price_cnt20 else 0.0
    rev_recall10 = (label_rev_hit10 / label_rev_total) if label_rev_total > 0 else 0.0
    rev_recall20 = (label_rev_hit20 / label_rev_total) if label_rev_total > 0 else 0.0

    metrics = {
        "model": "reranker",
        "rows": int(total),
        "recall_at_10": round(float(r10), 4),
        "recall_at_20": round(float(r20), 4),
        "hits_at_10": int(hit10),
        "hits_at_20": int(hit20),
        "top_m": int(args.top_m),
        "context_k": int(args.context_k),
        "product_type_fallback_topn": int(args.product_type_fallback_topn),
        "category_fallback_topn": int(args.category_fallback_topn),
        "brand_fallback_topn": int(args.brand_fallback_topn),
        "behavior_event_types": behavior_event_types,
        "behavior_weight": float(args.behavior_weight),
        "candidate_coverage": round(float(cov), 4),
        "covered_users": int(covered),
        "catalog_size": int(catalog_size),
        "catalog_coverage_at_10": round(float(catalog_cov10), 4),
        "catalog_coverage_at_20": round(float(catalog_cov20), 4),
        "avg_candidates_per_user": round(float(avg_cands), 2),
        "avg_recommended_price_at_10": round(float(avg_price10), 4),
        "avg_recommended_price_at_20": round(float(avg_price20), 4),
        "revenue_recall_at_10": round(float(rev_recall10), 4),
        "revenue_recall_at_20": round(float(rev_recall20), 4),
        "train_purchases": int(len(pur)),
        "train_users": int(pur["user_id"].nunique()),
        "eval_users": int(ds["user_id"].nunique()),
    }

    text = (
        "RERANKER\n"
        f"rows={metrics['rows']}\n"
        f"Recall@10={metrics['recall_at_10']:.4f} (hits={metrics['hits_at_10']})\n"
        f"Recall@20={metrics['recall_at_20']:.4f} (hits={metrics['hits_at_20']})\n"
        f"candidate_coverage={metrics['candidate_coverage']:.4f} (covered={metrics['covered_users']}/{metrics['rows']})\n"
        f"top_m={metrics['top_m']}\n"
        f"context_k={metrics['context_k']}\n"
        f"product_type_fallback_topn={metrics['product_type_fallback_topn']}\n"
        f"category_fallback_topn={metrics['category_fallback_topn']}\n"
        f"brand_fallback_topn={metrics['brand_fallback_topn']}\n"
        f"behavior_weight={metrics['behavior_weight']:.3f}\n"
        f"catalog_coverage@10={metrics['catalog_coverage_at_10']:.4f}\n"
        f"catalog_coverage@20={metrics['catalog_coverage_at_20']:.4f}\n"
        f"revenue_recall@10={metrics['revenue_recall_at_10']:.4f}\n"
        f"revenue_recall@20={metrics['revenue_recall_at_20']:.4f}\n"
        f"avg_price@10={metrics['avg_recommended_price_at_10']:.4f}\n"
        f"avg_price@20={metrics['avg_recommended_price_at_20']:.4f}\n"
        f"train_purchases={metrics['train_purchases']}\n"
        f"train_users={metrics['train_users']}\n"
        f"eval_users={metrics['eval_users']}\n"
    )
    print(text)
    if args.out_report:
        with open(args.out_report, "w", encoding="utf-8") as f:
            f.write(text)
        print("saved:", args.out_report)
    if args.out_json:
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
        print("saved:", args.out_json)


if __name__ == "__main__":
    main()
