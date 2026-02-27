from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from datetime import timedelta
from pathlib import Path
from typing import Any

import pandas as pd


INTERACTIONS_COLUMNS = [
    "user_id",
    "item_id",
    "ts",
    "event_type",
    "weight",
    "source",
    "transaction_id",
    "quantity",
    "unit_price",
    "total_amount",
    "channel",
    "request_id",
    "page",
    "section_key",
    "algo_mode",
    "score",
    "components",
    "context",
]

REC_EVENTS_COLUMNS = [
    "user_id",
    "item_id",
    "ts",
    "action",
    "request_id",
    "page",
    "section_key",
    "algo_mode",
    "score",
    "components",
    "context",
]

TRANSACTION_ITEMS_COLUMNS = [
    "transaction_id",
    "user_id",
    "item_id",
    "created_at",
    "quantity",
    "unit_price",
    "total_amount",
    "channel",
]

OFFER_EVENTS_COLUMNS = [
    "assignment_id",
    "user_id",
    "offer_id",
    "campaign_name",
    "event_type",
    "event_key",
    "event_version",
    "ts",
    "request_id",
    "context",
]

ITEMS_COLUMNS = [
    "item_id",
    "source_product_id",
    "name",
    "brand",
    "price",
    "currency",
    "category",
    "product_type",
    "concerns",
    "attrs",
    "flags",
    "in_stock",
]


def setup_django() -> None:
    root = Path(__file__).resolve().parents[2]
    backend_dir = root / "backend"
    if str(backend_dir) not in sys.path:
        sys.path.insert(0, str(backend_dir))
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "backend.settings")
    import django

    django.setup()


def _empty(columns: list[str]) -> pd.DataFrame:
    return pd.DataFrame(columns=columns)


def _to_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def _json_text(value: Any) -> str | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value)


def _normalize_interactions(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return _empty(INTERACTIONS_COLUMNS)

    out = df.copy()
    out["user_id"] = out["user_id"].astype(str)
    out["item_id"] = pd.to_numeric(out["item_id"], errors="coerce")
    out = out[out["item_id"].notna()].copy()
    out["item_id"] = out["item_id"].astype(int)
    out["ts"] = pd.to_datetime(out["ts"], errors="coerce", utc=True)
    out = out[out["ts"].notna()].copy()

    numeric_cols = ["weight", "quantity", "unit_price", "total_amount", "score"]
    for col in numeric_cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    for col in ["components", "context"]:
        if col in out.columns:
            out[col] = out[col].map(_json_text)

    for col in INTERACTIONS_COLUMNS:
        if col not in out.columns:
            out[col] = None

    out = out[INTERACTIONS_COLUMNS].sort_values(["user_id", "ts", "item_id", "source"], kind="stable")
    return out.reset_index(drop=True)


def _build_transaction_interactions(since) -> tuple[pd.DataFrame, pd.DataFrame]:
    from transactions.models import TransactionItem

    rows = list(
        TransactionItem.objects.filter(transaction__created_at__gte=since).values(
            "transaction_id",
            "transaction__user_id",
            "product_id",
            "transaction__created_at",
            "quantity",
            "unit_price",
            "transaction__total_amount",
            "transaction__channel",
        )
    )
    if not rows:
        return _empty(INTERACTIONS_COLUMNS), _empty(TRANSACTION_ITEMS_COLUMNS)

    tx = pd.DataFrame(rows).rename(
        columns={
            "transaction__user_id": "user_id",
            "product_id": "item_id",
            "transaction__created_at": "created_at",
            "transaction__total_amount": "total_amount",
            "transaction__channel": "channel",
        }
    )
    tx["created_at"] = pd.to_datetime(tx["created_at"], errors="coerce", utc=True)
    tx["quantity"] = pd.to_numeric(tx["quantity"], errors="coerce").fillna(1).clip(lower=1).astype(int)
    tx["unit_price"] = pd.to_numeric(tx["unit_price"], errors="coerce")
    tx["total_amount"] = pd.to_numeric(tx["total_amount"], errors="coerce")

    interactions = tx.rename(columns={"created_at": "ts"}).copy()
    interactions["event_type"] = "purchase"
    interactions["weight"] = interactions["quantity"].astype(float) * 5.0
    interactions["source"] = "transactions"
    interactions["request_id"] = None
    interactions["page"] = None
    interactions["section_key"] = None
    interactions["algo_mode"] = None
    interactions["score"] = None
    interactions["components"] = None
    interactions["context"] = None

    tx_items = tx[TRANSACTION_ITEMS_COLUMNS].copy()
    return _normalize_interactions(interactions), tx_items


def _build_rec_event_interactions(since) -> tuple[pd.DataFrame, pd.DataFrame]:
    from recs_analytics.models import RecommendationEvent

    rows = list(
        RecommendationEvent.objects.filter(created_at__gte=since).values(
            "user_id",
            "product_id",
            "created_at",
            "action",
            "request_id",
            "page",
            "section_key",
            "algo_mode",
            "score",
            "components",
            "context",
        )
    )
    if not rows:
        return _empty(INTERACTIONS_COLUMNS), _empty(REC_EVENTS_COLUMNS)

    rec = pd.DataFrame(rows).rename(
        columns={
            "product_id": "item_id",
            "created_at": "ts",
        }
    )
    rec["ts"] = pd.to_datetime(rec["ts"], errors="coerce", utc=True)

    weight_map = {
        "impression": 1.0,
        "click": 2.0,
        "add_to_cart": 3.0,
        "purchase_attributed": 4.0,
    }
    interactions = rec.copy()
    interactions["event_type"] = interactions["action"].astype(str)
    interactions["weight"] = interactions["event_type"].map(lambda x: float(weight_map.get(x, 1.0)))
    interactions["source"] = "rec_events"
    interactions["transaction_id"] = None
    interactions["quantity"] = None
    interactions["unit_price"] = None
    interactions["total_amount"] = None
    interactions["channel"] = "recs"

    rec_events = rec[REC_EVENTS_COLUMNS].copy()
    rec_events["components"] = rec_events["components"].map(_json_text)
    rec_events["context"] = rec_events["context"].map(_json_text)

    return _normalize_interactions(interactions), rec_events


def _export_items(out_dir: Path) -> pd.DataFrame:
    from catalog.models import Product

    rows = list(
        Product.objects.values(
            "id",
            "source_product_id",
            "name",
            "brand",
            "price",
            "currency",
            "category",
            "product_type",
            "concerns",
            "attrs",
            "flags",
            "in_stock",
        )
    )
    if not rows:
        items = _empty(ITEMS_COLUMNS)
    else:
        items = pd.DataFrame(rows).rename(columns={"id": "item_id"})
        for col in ITEMS_COLUMNS:
            if col not in items.columns:
                items[col] = None
        items = items[ITEMS_COLUMNS].copy()
        items["item_id"] = pd.to_numeric(items["item_id"], errors="coerce")
        items = items[items["item_id"].notna()].copy()
        items["item_id"] = items["item_id"].astype(int)
        items["price"] = pd.to_numeric(items["price"], errors="coerce")
        for col in ["concerns", "attrs", "flags"]:
            items[col] = items[col].map(_json_text)

    _to_parquet(items, out_dir / "items.parquet")
    # Backward compatibility for old scripts/notebooks.
    _to_parquet(items, out_dir / "project_items.parquet")
    return items


def export_recs_data(out_dir: Path, since) -> dict[str, int]:
    tx_interactions, tx_items = _build_transaction_interactions(since)
    rec_interactions, rec_events = _build_rec_event_interactions(since)

    interactions = pd.concat([tx_interactions, rec_interactions], ignore_index=True, sort=False)
    interactions = _normalize_interactions(interactions)

    _to_parquet(interactions, out_dir / "interactions.parquet")
    _to_parquet(rec_interactions, out_dir / "recs_interactions.parquet")
    _to_parquet(rec_events, out_dir / "rec_events.parquet")
    _to_parquet(tx_items, out_dir / "transactions_items.parquet")
    items = _export_items(out_dir)

    return {
        "interactions_rows": int(len(interactions)),
        "items_rows": int(len(items)),
        "transaction_interactions_rows": int(len(tx_interactions)),
        "rec_event_interactions_rows": int(len(rec_interactions)),
        "transaction_items_rows": int(len(tx_items)),
        "rec_events_rows": int(len(rec_events)),
    }


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
        u = str(r.user_id)
        at = pd.Timestamp(r.assigned_at)
        user_txs = tx_by_user.get(u, [])

        seen = [(t, amt) for (t, amt) in user_txs if t <= at]
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


def _export_offer_events(out_dir: Path, since) -> pd.DataFrame:
    from offers.models import OfferEvent

    rows = list(
        OfferEvent.objects.filter(created_at__gte=since).values(
            "assignment_id",
            "user_id",
            "offer_id",
            "campaign_name",
            "event_type",
            "event_key",
            "event_version",
            "created_at",
            "request_id",
            "context",
        )
    )
    if not rows:
        offer_events = _empty(OFFER_EVENTS_COLUMNS)
    else:
        offer_events = pd.DataFrame(rows).rename(columns={"created_at": "ts"})
        for col in OFFER_EVENTS_COLUMNS:
            if col not in offer_events.columns:
                offer_events[col] = None
        offer_events = offer_events[OFFER_EVENTS_COLUMNS].copy()
        offer_events["ts"] = pd.to_datetime(offer_events["ts"], errors="coerce", utc=True)
        offer_events = offer_events[offer_events["ts"].notna()].copy()
        offer_events["context"] = offer_events["context"].map(_json_text)

    _to_parquet(offer_events, out_dir / "offer_events.parquet")
    return offer_events


def export_offer_data(out_dir: Path, since) -> dict[str, int]:
    from offers.models import OfferAssignment, OfferEvent
    from transactions.models import Transaction

    assign_rows = list(
        OfferAssignment.objects.filter(assigned_at__gte=since)
        .select_related("offer", "offer__campaign")
        .values(
            "id",
            "user_id",
            "assigned_at",
            "is_redeemed",
            "target",
            "offer__offer_type",
            "offer__target_scope",
            "offer__value",
            "offer__estimated_cost",
            "offer__cooldown_days",
            "offer__expires_in_days",
            "offer__campaign__name",
        )
    )
    if not assign_rows:
        _to_parquet(
            _empty(["assignment_id", "user_id", "assigned_at", "label_redeemed"]),
            out_dir / "offer_train.parquet",
        )
        offer_events = _export_offer_events(out_dir, since)
        return {
            "offer_assignments_rows": 0,
            "offer_events_rows": int(len(offer_events)),
            "offer_train_rows": 0,
        }

    assign = pd.DataFrame(assign_rows).rename(
        columns={
            "id": "assignment_id",
            "is_redeemed": "label_redeemed",
            "offer__offer_type": "offer_type",
            "offer__target_scope": "target_scope",
            "offer__value": "offer_value",
            "offer__estimated_cost": "estimated_cost",
            "offer__cooldown_days": "cooldown_days",
            "offer__expires_in_days": "expires_in_days",
            "offer__campaign__name": "campaign_name",
        }
    )

    ev_rows = list(
        OfferEvent.objects.filter(created_at__gte=since).values(
            "assignment_id",
            "event_type",
            "created_at",
        )
    )
    ev = pd.DataFrame(ev_rows)
    if not ev.empty:
        ev["is_exposed"] = (ev["event_type"] == "offer_exposed").astype(int)
        ev["is_clicked"] = (ev["event_type"] == "offer_clicked").astype(int)
        ev["is_redeemed_event"] = (ev["event_type"] == "offer_redeemed").astype(int)
        agg = (
            ev.groupby("assignment_id", as_index=False)[["is_exposed", "is_clicked", "is_redeemed_event"]]
            .max()
        )
        assign = assign.merge(agg, on="assignment_id", how="left")
    for col in ["is_exposed", "is_clicked", "is_redeemed_event"]:
        if col not in assign.columns:
            assign[col] = 0
        assign[col] = assign[col].fillna(0).astype(int)

    tx = pd.DataFrame(
        list(
            Transaction.objects.values(
                "user_id",
                "created_at",
                "total_amount",
            )
        )
    )
    if tx.empty:
        assign["recency_days"] = 9999
        assign["frequency_90d"] = 0
        assign["monetary_90d"] = 0.0
        assign["txn_count_before"] = 0
        assign["spend_before"] = 0.0
    else:
        assign = _build_tx_features(assign, tx)

    assign["campaign_name"] = assign["campaign_name"].fillna("none")
    assign["offer_type"] = assign["offer_type"].fillna("unknown")
    assign["target_scope"] = assign["target_scope"].fillna("unknown")
    assign["target"] = assign["target"].map(_json_text)
    assign["offer_value"] = pd.to_numeric(assign["offer_value"], errors="coerce").fillna(0.0)
    assign["estimated_cost"] = pd.to_numeric(assign["estimated_cost"], errors="coerce").fillna(0.0)
    assign["cooldown_days"] = pd.to_numeric(assign["cooldown_days"], errors="coerce").fillna(0).astype(int)
    assign["expires_in_days"] = pd.to_numeric(assign["expires_in_days"], errors="coerce").fillna(0).astype(int)

    _to_parquet(assign, out_dir / "offer_train.parquet")
    _to_parquet(assign, out_dir / "offer_assignments.parquet")
    offer_events = _export_offer_events(out_dir, since)

    return {
        "offer_assignments_rows": int(len(assign)),
        "offer_events_rows": int(len(offer_events)),
        "offer_train_rows": int(len(assign)),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="data/processed/project")
    ap.add_argument("--days", type=int, default=180)
    args = ap.parse_args()

    setup_django()

    from django.utils import timezone

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    since = timezone.now() - timedelta(days=args.days)

    recs_stats = export_recs_data(out_dir, since)
    offer_stats = export_offer_data(out_dir, since)

    print("OK")
    print("saved:", str(out_dir / "interactions.parquet"))
    print("saved:", str(out_dir / "items.parquet"))
    print("saved:", str(out_dir / "transactions_items.parquet"))
    print("saved:", str(out_dir / "rec_events.parquet"))
    print("saved:", str(out_dir / "offer_events.parquet"))
    print("saved:", str(out_dir / "offer_train.parquet"))
    print("stats:", {**recs_stats, **offer_stats})


if __name__ == "__main__":
    main()
