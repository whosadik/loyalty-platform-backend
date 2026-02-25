# ml/datasets/convert_cosmetics.py
import argparse
import glob
import os
import pandas as pd

EVENT_MAP = {
    "view": ("impression", 1.0),
    "cart": ("add_to_cart", 3.0),
    "purchase": ("purchase", 5.0),
}

USECOLS = [
    "event_time",
    "event_type",
    "product_id",
    "category_code",
    "brand",
    "price",
    "user_id",
]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_glob", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--limit_rows", type=int, default=0)  # 0 = all
    args = ap.parse_args()

    paths = sorted(glob.glob(args.input_glob))
    if not paths:
        raise SystemExit(f"No files matched: {args.input_glob}")

    os.makedirs(args.out_dir, exist_ok=True)

    parts = []
    for p in paths:
        df = pd.read_csv(p, usecols=USECOLS)

        if args.limit_rows and len(df) > args.limit_rows:
            df = df.sample(args.limit_rows, random_state=42)

        df = df[df["event_type"].isin(EVENT_MAP.keys())].copy()

        df["ts"] = pd.to_datetime(df["event_time"], utc=True, errors="coerce")
        df = df[df["ts"].notna()]

        df["event_type_norm"] = df["event_type"].map(lambda x: EVENT_MAP[x][0])
        df["weight"] = df["event_type"].map(lambda x: EVENT_MAP[x][1])

        df.rename(columns={"product_id": "item_id"}, inplace=True)
        df["dataset"] = "cosmetics"

        parts.append(
            df[
                [
                    "user_id",
                    "item_id",
                    "ts",
                    "event_type_norm",
                    "weight",
                    "dataset",
                    "category_code",
                    "brand",
                    "price",
                ]
            ]
        )

    all_df = pd.concat(parts, ignore_index=True)

    interactions = all_df.rename(columns={"event_type_norm": "event_type"})[
        ["user_id", "item_id", "ts", "event_type", "weight", "dataset"]
    ]

    items = (
        all_df.sort_values("ts")
        .groupby("item_id", as_index=False)
        .tail(1)[["item_id", "category_code", "brand", "price", "dataset"]]
        .rename(columns={"category_code": "category"})
    )
    items["product_type"] = None

    interactions_path = os.path.join(args.out_dir, "interactions.parquet")
    items_path = os.path.join(args.out_dir, "items.parquet")

    interactions.to_parquet(interactions_path, index=False)
    items.to_parquet(items_path, index=False)

    print("OK")
    print("interactions:", len(interactions))
    print("items:", len(items))
    print("saved:", interactions_path)
    print("saved:", items_path)

if __name__ == "__main__":
    main()