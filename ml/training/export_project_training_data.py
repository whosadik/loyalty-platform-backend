from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict
from datetime import timedelta
from pathlib import Path

import pandas as pd


def setup_django() -> None:
    root = Path(__file__).resolve().parents[2]
    backend_dir = root / "backend"
    if str(backend_dir) not in sys.path:
        sys.path.insert(0, str(backend_dir))
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "backend.settings")
    import django

    django.setup()


def export_recs_data(out_dir: Path, since):
    from catalog.models import Product
    from recs_analytics.models import RecommendationEvent

    rows = list(
        RecommendationEvent.objects.filter(created_at__gte=since).values(
            "user_id",
            "product_id",
            "created_at",
            "action",
            "page",
            "section_key",
            "algo_mode",
            "score",
            "components",
            "context",
        )
    )
    if rows:
        inter = pd.DataFrame(rows)
        inter = inter.rename(
            columns={
                "product_id": "item_id",
                "created_at": "ts",
                "action": "event_type",
            }
        )
        weight_map = {
            "impression": 1.0,
            "click": 2.0,
            "add_to_cart": 3.0,
            "purchase_attributed": 5.0,
        }
        inter["weight"] = inter["event_type"].map(lambda x: float(weight_map.get(x, 1.0)))
        inter["dataset"] = "project_recs"
        inter.to_parquet(out_dir / "recs_interactions.parquet", index=False)
    else:
        pd.DataFrame(
            columns=[
                "user_id",
                "item_id",
                "ts",
                "event_type",
                "weight",
                "page",
                "section_key",
                "algo_mode",
                "score",
                "components",
                "context",
                "dataset",
            ]
        ).to_parquet(out_dir / "recs_interactions.parquet", index=False)

    items = pd.DataFrame(
        list(
            Product.objects.values(
                "id",
                "category",
                "product_type",
                "brand",
                "price",
                "attrs",
                "concerns",
                "in_stock",
            )
        )
    ).rename(columns={"id": "item_id"})
    items["dataset"] = "project_catalog"
    items.to_parquet(out_dir / "project_items.parquet", index=False)


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


def export_offer_data(out_dir: Path, since):
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
        pd.DataFrame(
            columns=[
                "assignment_id",
                "user_id",
                "assigned_at",
                "label_redeemed",
            ]
        ).to_parquet(out_dir / "offer_train.parquet", index=False)
        return

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
    assign["offer_value"] = pd.to_numeric(assign["offer_value"], errors="coerce").fillna(0.0)
    assign["estimated_cost"] = pd.to_numeric(assign["estimated_cost"], errors="coerce").fillna(0.0)
    assign["cooldown_days"] = pd.to_numeric(assign["cooldown_days"], errors="coerce").fillna(0).astype(int)
    assign["expires_in_days"] = pd.to_numeric(assign["expires_in_days"], errors="coerce").fillna(0).astype(int)

    assign.to_parquet(out_dir / "offer_train.parquet", index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="data/processed/project")
    ap.add_argument("--days", type=int, default=180)
    args = ap.parse_args()

    setup_django()

    from django.utils import timezone

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    since = timezone.now() - timedelta(days=args.days)

    export_recs_data(out_dir, since)
    export_offer_data(out_dir, since)

    print("OK")
    print("saved:", str(out_dir / "recs_interactions.parquet"))
    print("saved:", str(out_dir / "project_items.parquet"))
    print("saved:", str(out_dir / "offer_train.parquet"))


if __name__ == "__main__":
    main()
