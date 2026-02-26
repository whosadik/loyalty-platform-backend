from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd


EVENT_WEIGHT = {
    "view": 1.0,
    "cart": 3.0,
    "purchase": 5.0,
}


def _require_columns(df: pd.DataFrame, required: list[str], name: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{name}: missing columns {missing}")


def _to_int_item(v: Any) -> int | None:
    if pd.isna(v):
        return None
    s = str(v).strip()
    if not s:
        return None
    try:
        return int(float(s))
    except Exception:
        return None


def _read_xlsx(path: Path) -> pd.DataFrame:
    xls = pd.ExcelFile(path)
    return pd.read_excel(path, sheet_name=xls.sheet_names[0])


def _build_tx_features(assign_df: pd.DataFrame, tx_df: pd.DataFrame) -> pd.DataFrame:
    tx_by_user = defaultdict(list)
    for r in tx_df.itertuples(index=False):
        tx_by_user[str(r.user_id)].append((pd.Timestamp(r.created_at), float(r.total_amount)))
    for u in tx_by_user:
        tx_by_user[u].sort(key=lambda x: x[0])

    recency_days: list[int] = []
    freq_90d: list[int] = []
    monetary_90d: list[float] = []
    txn_count_before: list[int] = []
    spend_before: list[float] = []

    for r in assign_df.itertuples(index=False):
        user = str(r.user_id)
        at = pd.Timestamp(r.assigned_at)
        seen = [(t, amt) for (t, amt) in tx_by_user.get(user, []) if t <= at]
        txn_count_before.append(len(seen))
        spend_before.append(float(sum(amt for _, amt in seen)))

        if not seen:
            recency_days.append(9999)
            freq_90d.append(0)
            monetary_90d.append(0.0)
            continue

        last_t = seen[-1][0]
        recency_days.append(max(0, int((at - last_t).days)))
        start_90 = at - pd.Timedelta(days=90)
        in_90 = [(t, amt) for (t, amt) in seen if t >= start_90]
        freq_90d.append(len(in_90))
        monetary_90d.append(float(sum(amt for _, amt in in_90)))

    out = assign_df.copy()
    out["recency_days"] = recency_days
    out["frequency_90d"] = freq_90d
    out["monetary_90d"] = monetary_90d
    out["txn_count_before"] = txn_count_before
    out["spend_before"] = spend_before
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", default="data/goldapple_user_events_excels_deep")
    ap.add_argument("--out_dir", default="data/processed/goldapple_deep")
    ap.add_argument("--catalog_xlsx", default="data/catalog/goldapple_300_products.xlsx")
    args = ap.parse_args()

    input_dir = Path(args.input_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    p_events = input_dir / "events_recs_deep.xlsx"
    p_orders = input_dir / "orders_items_deep.xlsx"
    p_offer_assign = input_dir / "offer_assignments_deep.xlsx"
    p_offer_events = input_dir / "offer_events_deep.xlsx"
    p_catalog = Path(args.catalog_xlsx)

    for p in [p_events, p_orders, p_offer_assign, p_offer_events]:
        if not p.exists():
            raise FileNotFoundError(f"Required file not found: {p}")

    events = _read_xlsx(p_events)
    _require_columns(
        events,
        ["event_time", "event_type", "user_id", "source_product_id", "price", "brand", "category_code"],
        "events_recs_deep.xlsx",
    )
    events = events.copy()
    events["event_type"] = events["event_type"].astype(str).str.strip().str.lower()
    events = events[events["event_type"].isin({"view", "cart", "purchase"})].copy()
    events["ts"] = pd.to_datetime(events["event_time"], errors="coerce", utc=True)
    events["item_id"] = events["source_product_id"].map(_to_int_item)
    events["user_id"] = events["user_id"].astype(str)
    events["weight"] = events["event_type"].map(lambda x: float(EVENT_WEIGHT.get(x, 1.0)))
    events["dataset"] = "goldapple_deep"
    events = events[(events["ts"].notna()) & (events["item_id"].notna())].copy()
    events["item_id"] = events["item_id"].astype("int64")

    interactions = events[["user_id", "item_id", "ts", "event_type", "weight", "dataset"]].copy()
    interactions.to_parquet(out_dir / "interactions.parquet", index=False)

    # Build items from catalog xlsx if available, fallback to events snapshot.
    if p_catalog.exists():
        cat = _read_xlsx(p_catalog)
        required_cat = ["id", "brand", "price", "category", "product_type", "attrs", "concerns", "in_stock"]
        _require_columns(cat, required_cat, "catalog_xlsx")
        cat = cat.copy()
        cat["item_id"] = cat["id"].map(_to_int_item)
        cat = cat[cat["item_id"].notna()].copy()
        cat["item_id"] = cat["item_id"].astype("int64")
        cat["price"] = pd.to_numeric(cat["price"], errors="coerce")
        cat["brand"] = cat["brand"].astype(str)
        cat["category"] = cat["category"].astype(str).str.lower()
        cat["product_type"] = cat["product_type"].astype(str).str.lower()
        cat["in_stock"] = cat["in_stock"].fillna(1).astype(int).astype(bool)
        for c in ["attrs", "concerns"]:
            cat[c] = cat[c].apply(
                lambda x: json.loads(x) if isinstance(x, str) and x.strip().startswith(("{", "[")) else ({} if c == "attrs" else [])
            )
            cat[c] = cat[c].apply(lambda x: x if isinstance(x, (dict, list)) else ({} if c == "attrs" else []))
        items = cat[["item_id", "category", "product_type", "brand", "price", "attrs", "concerns", "in_stock"]].copy()
        items = items.drop_duplicates("item_id")
        items["dataset"] = "goldapple_catalog"
    else:
        items = (
            events.sort_values("ts")
            .groupby("item_id", as_index=False)
            .tail(1)[["item_id", "category_code", "brand", "price"]]
            .rename(columns={"category_code": "category"})
        )
        items["product_type"] = "unknown"
        items["attrs"] = [{} for _ in range(len(items))]
        items["concerns"] = [[] for _ in range(len(items))]
        items["in_stock"] = True
        items["dataset"] = "goldapple_events_fallback"
        items = items[["item_id", "category", "product_type", "brand", "price", "attrs", "concerns", "in_stock", "dataset"]]
    items.to_parquet(out_dir / "items.parquet", index=False)

    # Orders -> transaction rows for feature engineering.
    orders = _read_xlsx(p_orders)
    _require_columns(
        orders,
        ["order_id", "user_id", "created_at", "source_product_id", "quantity", "unit_price", "channel"],
        "orders_items_deep.xlsx",
    )
    orders = orders.copy()
    orders["created_at"] = pd.to_datetime(orders["created_at"], errors="coerce", utc=True)
    orders["quantity"] = pd.to_numeric(orders["quantity"], errors="coerce").fillna(1.0)
    orders["unit_price"] = pd.to_numeric(orders["unit_price"], errors="coerce").fillna(0.0)
    orders["line_total"] = orders["quantity"] * orders["unit_price"]
    tx_df = (
        orders.groupby(["order_id", "user_id", "created_at"], as_index=False)["line_total"]
        .sum()
        .rename(columns={"line_total": "total_amount"})
    )

    # Offer training dataset.
    offer_assign = _read_xlsx(p_offer_assign)
    _require_columns(
        offer_assign,
        [
            "assignment_id",
            "user_id",
            "assigned_at",
            "campaign_name",
            "offer_type",
            "target_scope",
            "offer_value",
            "estimated_cost",
            "cooldown_days",
            "expires_in_days",
            "label_redeemed",
        ],
        "offer_assignments_deep.xlsx",
    )
    offer_assign = offer_assign.copy()
    offer_assign["assigned_at"] = pd.to_datetime(offer_assign["assigned_at"], errors="coerce", utc=True)
    offer_assign = offer_assign[offer_assign["assigned_at"].notna()].copy()
    for c in ["offer_value", "estimated_cost", "cooldown_days", "expires_in_days", "label_redeemed"]:
        offer_assign[c] = pd.to_numeric(offer_assign[c], errors="coerce").fillna(0)

    offer_ev = _read_xlsx(p_offer_events)
    _require_columns(offer_ev, ["assignment_id", "event_type", "created_at"], "offer_events_deep.xlsx")
    offer_ev = offer_ev.copy()
    offer_ev["event_type"] = offer_ev["event_type"].astype(str).str.strip().str.lower()
    offer_ev["is_exposed"] = (offer_ev["event_type"] == "offer_exposed").astype(int)
    offer_ev["is_clicked"] = (offer_ev["event_type"] == "offer_clicked").astype(int)
    offer_ev["is_redeemed_event"] = (offer_ev["event_type"] == "offer_redeemed").astype(int)
    offer_ev_agg = offer_ev.groupby("assignment_id", as_index=False)[["is_exposed", "is_clicked", "is_redeemed_event"]].max()

    offer_train = offer_assign.merge(offer_ev_agg, on="assignment_id", how="left")
    for c in ["is_exposed", "is_clicked", "is_redeemed_event"]:
        offer_train[c] = offer_train[c].fillna(0).astype(int)
    offer_train["label_redeemed"] = offer_train["label_redeemed"].fillna(0).astype(int)
    offer_train = _build_tx_features(offer_train, tx_df)
    offer_train.to_parquet(out_dir / "offer_train.parquet", index=False)

    # Optional normalized orders export.
    orders_norm = orders.copy()
    orders_norm["item_id"] = orders_norm["source_product_id"].map(_to_int_item)
    orders_norm = orders_norm[orders_norm["item_id"].notna()].copy()
    orders_norm["item_id"] = orders_norm["item_id"].astype("int64")
    orders_norm.to_parquet(out_dir / "orders_items.parquet", index=False)

    print("OK")
    print("saved:", out_dir / "interactions.parquet")
    print("saved:", out_dir / "items.parquet")
    print("saved:", out_dir / "offer_train.parquet")
    print("saved:", out_dir / "orders_items.parquet")
    print("stats:")
    print("interactions_rows:", len(interactions))
    print("interaction_users:", interactions["user_id"].nunique())
    print("interaction_items:", interactions["item_id"].nunique())
    print("purchases:", int((interactions["event_type"] == "purchase").sum()))
    print("offer_rows:", len(offer_train))
    print("offer_positive_rate:", round(float(offer_train["label_redeemed"].mean()), 6))


if __name__ == "__main__":
    main()
