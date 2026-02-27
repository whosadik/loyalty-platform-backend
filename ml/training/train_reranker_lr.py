from __future__ import annotations

import argparse
import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import train_test_split

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
        to_item_id,
        write_user_ids,
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
        to_item_id,
        write_user_ids,
    )


def _sample_negatives(
    *,
    label: int,
    candidate_items: list[int],
    neg_per_pos: int,
    rng: np.random.Generator,
) -> list[int]:
    neg_pool = [c for c in candidate_items if c != label]
    if not neg_pool:
        return []
    k = min(neg_per_pos, len(neg_pool))
    idx = rng.choice(len(neg_pool), size=k, replace=False)
    return [int(neg_pool[i]) for i in idx]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--interactions", required=True)
    ap.add_argument("--items", required=True)
    ap.add_argument("--ds", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--top_m", type=int, default=1500)
    ap.add_argument("--neg_per_pos", type=int, default=20)
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
    ap.add_argument("--estimator", choices=["lr", "hgb"], default="hgb")
    ap.add_argument("--model_version", default="recs_reranker_v3")
    ap.add_argument("--test_size", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    inter = pd.read_parquet(args.interactions)
    inter["user_id"] = inter["user_id"].map(normalize_user_id)
    inter = inter[inter["user_id"] != ""].copy()
    items_lookup = prepare_items_lookup(pd.read_parquet(args.items))
    ds = pd.read_parquet(args.ds).copy()
    if ds.empty:
        raise SystemExit("Dataset is empty: --ds")

    ds["user_id"] = ds["user_id"].map(normalize_user_id)
    ds = ds[ds["user_id"] != ""].copy()
    if ds.empty:
        raise SystemExit("No valid users in --ds")

    users = sorted(ds["user_id"].unique().tolist())
    if len(users) < 2:
        raise SystemExit("Need at least 2 users in dataset for train/test split")

    train_users, test_users = train_test_split(
        users,
        test_size=args.test_size,
        random_state=args.seed,
    )
    train_users = sorted(set(train_users))
    test_users = sorted(set(test_users))
    write_user_ids(os.path.join(args.out_dir, "train_users.txt"), train_users)
    write_user_ids(os.path.join(args.out_dir, "test_users.txt"), test_users)

    pur = inter[inter["event_type"] == "purchase"][["user_id", "item_id", "ts"]].copy()
    pur["user_id"] = pur["user_id"].map(normalize_user_id)
    pur["item_id"] = pur["item_id"].map(to_item_id)
    pur = pur[(pur["user_id"] != "") & pur["item_id"].notna()].copy()
    pur["item_id"] = pur["item_id"].astype(int)

    pur_train = pur[pur["user_id"].isin(train_users)].copy()
    if pur_train.empty:
        raise SystemExit("No training purchases after user split; check dataset overlap")

    top_map_purchase = build_next_item_map(pur_train, top_m=args.top_m)
    behavior_event_types = [
        x.strip().lower()
        for x in str(args.behavior_event_types or "").split(",")
        if x.strip()
    ]
    inter_train = inter[inter["user_id"].isin(train_users)].copy()
    top_map_behavior = build_behavior_next_item_map(
        inter_train,
        event_types=behavior_event_types,
        top_m=args.top_m,
    )
    top_map = merge_top_maps(
        top_map_purchase,
        top_map_behavior,
        secondary_weight=max(0.0, float(args.behavior_weight)),
        top_m=args.top_m,
    )
    pop = pur_train["item_id"].value_counts().to_dict()
    fallback_items = [int(x) for x in pur_train["item_id"].value_counts().index.tolist()]
    category_pop_map = build_category_popularity_map(
        pur_train,
        items_lookup,
        top_n=max(0, int(args.category_fallback_topn or 0)),
    )
    product_type_pop_map = build_product_type_popularity_map(
        pur_train,
        items_lookup,
        top_n=max(0, int(args.product_type_fallback_topn or 0)),
    )
    brand_pop_map = build_brand_popularity_map(
        pur_train,
        items_lookup,
        top_n=max(0, int(args.brand_fallback_topn or 0)),
    )

    ds_train = ds[ds["user_id"].isin(train_users)].copy()
    rng = np.random.default_rng(args.seed)
    rows: list[tuple[str, int, int, int, float, int]] = []

    for r in ds_train.itertuples(index=False):
        user = str(r.user_id)
        ctx = to_item_id(getattr(r, "context_last_item", None))
        label = to_item_id(getattr(r, "label_item", None))
        if ctx is None or label is None:
            continue

        ctx_items = parse_context_items(
            getattr(r, "context_items", None),
            fallback_last_item=ctx,
            max_k=args.context_k,
        )
        product_type_fallback = product_type_fallback_for_context(
            ctx_items,
            items_lookup,
            product_type_pop_map,
            max_items=max(0, int(args.product_type_fallback_topn or 0)),
        )
        category_fallback = category_fallback_for_context(
            ctx_items,
            items_lookup,
            category_pop_map,
            max_items=max(0, int(args.category_fallback_topn or 0)),
        )
        brand_fallback = brand_fallback_for_context(
            ctx_items,
            items_lookup,
            brand_pop_map,
            max_items=max(0, int(args.brand_fallback_topn or 0)),
        )
        cands, scores, ranks = build_context_candidates(
            ctx_items,
            top_map,
            top_m=args.top_m,
            fallback_items=fallback_items,
            product_type_fallback_items=product_type_fallback,
            category_fallback_items=category_fallback,
            brand_fallback_items=brand_fallback,
        )
        if label not in cands:
            continue
        if len(cands) < 2:
            continue

        rows.append((user, ctx, label, 1, float(scores.get(label, 0.0)), ranks.get(label, 9999)))

        negs = _sample_negatives(
            label=label,
            candidate_items=cands,
            neg_per_pos=max(1, args.neg_per_pos),
            rng=rng,
        )
        for n in negs:
            nn = int(n)
            rows.append((user, ctx, nn, 0, float(scores.get(nn, 0.0)), ranks.get(nn, 9999)))

    if not rows:
        raise SystemExit("No training rows were generated")

    df = pd.DataFrame(
        rows,
        columns=["user_id", "context_item", "candidate_item", "y", "transition_count", "candidate_rank"],
    )

    pop_map = {int(k): float(v) for k, v in pop.items()}
    feat_rows = []
    for rr in df.itertuples(index=False):
        X = build_feature_matrix_for_candidates(
            context_item=int(rr.context_item),
            candidate_items=[int(rr.candidate_item)],
            transition_counts={int(rr.candidate_item): int(rr.transition_count)},
            candidate_ranks={int(rr.candidate_item): int(rr.candidate_rank)},
            items_lookup=items_lookup,
            popularity_map=pop_map,
        )
        if X.shape[0] != 1:
            continue
        feat_rows.append(X[0].tolist())

    if len(feat_rows) != len(df):
        keep = min(len(feat_rows), len(df))
        df = df.iloc[:keep].copy()
        feat_rows = feat_rows[:keep]
    if len(df) < 100:
        raise SystemExit(f"Too few training rows after feature build: {len(df)}")

    X = np.asarray(feat_rows, dtype=float)
    y = df["y"].astype(int).values

    if args.estimator == "lr":
        model = LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            solver="lbfgs",
        )
        model.fit(X, y)
    else:
        # Balance positives for tree model via sample weights.
        pos_rate = float(np.mean(y))
        pos_w = ((1.0 - pos_rate) / max(pos_rate, 1e-6)) if pos_rate > 0 else 1.0
        sample_weight = np.where(y == 1, pos_w, 1.0)
        model = HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_iter=350,
            max_depth=6,
            min_samples_leaf=25,
            l2_regularization=0.05,
            random_state=args.seed,
        )
        model.fit(X, y, sample_weight=sample_weight)

    p_train = model.predict_proba(X)[:, 1]
    auc = roc_auc_score(y, p_train) if len(np.unique(y)) > 1 else 0.0
    ll = log_loss(y, p_train, labels=[0, 1]) if len(np.unique(y)) > 1 else 0.0

    model_path = os.path.join(args.out_dir, "model.pkl")
    joblib.dump(model, model_path)

    meta = {
        "model_version": str(args.model_version),
        "estimator": str(args.estimator),
        "seed": int(args.seed),
        "top_m": int(args.top_m),
        "neg_per_pos": int(args.neg_per_pos),
        "product_type_fallback_topn": int(args.product_type_fallback_topn),
        "category_fallback_topn": int(args.category_fallback_topn),
        "brand_fallback_topn": int(args.brand_fallback_topn),
        "behavior_event_types": behavior_event_types,
        "behavior_weight": float(args.behavior_weight),
        "test_size_users": float(args.test_size),
        "features": [
            "transition_count",
            "rank_inv",
            "same_category",
            "same_brand",
            "price_diff",
            "log_popularity",
        ],
        "train_rows": int(len(df)),
        "train_pos_rate": round(float(df["y"].mean()), 6),
        "train_auc": round(float(auc), 6),
        "train_logloss": round(float(ll), 6),
        "train_users": int(len(train_users)),
        "test_users": int(len(test_users)),
        "dataset": str(args.ds),
        "interactions": str(args.interactions),
        "items": str(args.items),
    }

    with open(os.path.join(args.out_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print("OK")
    print("saved:", model_path)
    print("meta:", meta)


if __name__ == "__main__":
    main()
