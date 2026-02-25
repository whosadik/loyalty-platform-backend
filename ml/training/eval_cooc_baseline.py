from __future__ import annotations

import argparse
import os

import pandas as pd

try:
    from ml.training.recs_common import (
        build_context_candidates,
        build_next_item_map,
        normalize_user_id,
        parse_context_items,
        read_user_ids,
        to_item_id,
    )
except ModuleNotFoundError:
    from recs_common import (  # type: ignore
        build_context_candidates,
        build_next_item_map,
        normalize_user_id,
        parse_context_items,
        read_user_ids,
        to_item_id,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--interactions", required=True)
    ap.add_argument("--ds", required=True)
    ap.add_argument("--out_report", default="reports/cooc_baseline.txt")
    ap.add_argument("--top_m", type=int, default=50)
    ap.add_argument("--context_k", type=int, default=3)
    ap.add_argument("--train_users", default="")
    ap.add_argument("--eval_users", default="")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out_report), exist_ok=True)

    inter = pd.read_parquet(args.interactions)
    pur = inter[inter["event_type"] == "purchase"][["user_id", "item_id", "ts"]].copy()
    pur["user_id"] = pur["user_id"].map(normalize_user_id)
    pur = pur[pur["user_id"] != ""].copy()

    ds = pd.read_parquet(args.ds).copy()
    ds["user_id"] = ds["user_id"].map(normalize_user_id)
    ds = ds[ds["user_id"] != ""].copy()

    train_users = read_user_ids(args.train_users) if args.train_users else set()
    eval_users = read_user_ids(args.eval_users) if args.eval_users else set()

    if train_users:
        pur = pur[pur["user_id"].isin(train_users)].copy()
    if eval_users:
        ds = ds[ds["user_id"].isin(eval_users)].copy()

    top_map = build_next_item_map(pur, top_m=args.top_m)
    fallback_items = [int(x) for x in pur["item_id"].value_counts().index.tolist()]
    hit10 = 0
    hit20 = 0
    covered = 0
    total = len(ds)

    for r in ds.itertuples(index=False):
        label = to_item_id(getattr(r, "label_item", None))
        if label is None:
            continue
        ctx_items = parse_context_items(
            getattr(r, "context_items", None),
            fallback_last_item=getattr(r, "context_last_item", None),
            max_k=args.context_k,
        )
        cands, _, _ = build_context_candidates(
            ctx_items,
            top_map,
            top_m=args.top_m,
            fallback_items=fallback_items,
        )
        if label in cands:
            covered += 1
        if label in cands[:10]:
            hit10 += 1
        if label in cands[:20]:
            hit20 += 1

    cov = (covered / total) if total else 0.0
    r10 = (hit10 / total) if total else 0.0
    r20 = (hit20 / total) if total else 0.0

    text = (
        "COOC BASELINE (MULTI-CONTEXT)\n"
        f"rows={total}\n"
        f"Recall@10={r10:.4f} (hits={hit10})\n"
        f"Recall@20={r20:.4f} (hits={hit20})\n"
        f"top_m={args.top_m}\n"
        f"context_k={args.context_k}\n"
        f"candidate_coverage={cov:.4f} (covered={covered}/{total})\n"
        f"train_purchases={len(pur)}\n"
        f"train_users={pur['user_id'].nunique()}\n"
        f"eval_users={ds['user_id'].nunique()}\n"
    )

    with open(args.out_report, "w", encoding="utf-8") as f:
        f.write(text)

    print(text)
    print("saved:", args.out_report)


if __name__ == "__main__":
    main()
