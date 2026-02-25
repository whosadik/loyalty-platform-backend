from __future__ import annotations

import argparse
import os

import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--interactions", required=True)
    ap.add_argument("--items", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--min_purchases", type=int, default=2)
    ap.add_argument("--context_k", type=int, default=3)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    inter = pd.read_parquet(args.interactions)
    items = pd.read_parquet(args.items)

    # Ground truth from purchase events.
    pur = inter[inter["event_type"] == "purchase"].copy()
    pur = pur.sort_values(["user_id", "ts"])

    cnt = pur.groupby("user_id")["item_id"].count()
    good_users = cnt[cnt >= args.min_purchases].index
    pur = pur[pur["user_id"].isin(good_users)]

    # Label = last purchase of user.
    last = pur.groupby("user_id", as_index=False).tail(1)[["user_id", "item_id", "ts"]].rename(
        columns={"item_id": "label_item", "ts": "label_ts"}
    )

    # History before label timestamp.
    pur = pur.merge(last[["user_id", "label_ts"]], on="user_id", how="inner")
    hist = pur[pur["ts"] < pur["label_ts"]].copy()
    hist = hist.sort_values(["user_id", "ts"])

    # Last item context.
    last_hist = (
        hist.groupby("user_id", as_index=False)
        .tail(1)[["user_id", "item_id"]]
        .rename(columns={"item_id": "context_last_item"})
    )

    # Multi-context from last K items in history.
    k = max(1, int(args.context_k))
    ctx_items = (
        hist.groupby("user_id")["item_id"]
        .apply(lambda s: [int(x) for x in s.tail(k).tolist()])
        .reset_index(name="context_items")
    )

    ds = last.merge(last_hist, on="user_id", how="inner")
    ds = ds.merge(ctx_items, on="user_id", how="left")
    ds["context_items"] = ds["context_items"].apply(lambda x: x if isinstance(x, list) else [])
    ds["context_len"] = ds["context_items"].apply(len).astype("int16")

    # Meta features for quick diagnostics.
    ds = ds.merge(items.add_prefix("label_"), left_on="label_item", right_on="label_item_id", how="left")
    ds = ds.merge(items.add_prefix("ctx_"), left_on="context_last_item", right_on="ctx_item_id", how="left")
    ds["same_category"] = (ds["label_category"] == ds["ctx_category"]).astype("int8")

    ds.to_parquet(args.out, index=False)

    print("OK")
    print("rows:", len(ds))
    print("users:", ds["user_id"].nunique())
    print("context_k:", k)
    print("saved:", args.out)


if __name__ == "__main__":
    main()
