from __future__ import annotations

import argparse
import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="Parquet from export_project_training_data.py")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    df = pd.read_parquet(args.dataset).copy()
    if df.empty:
        raise SystemExit("Offer dataset is empty")

    if "label_redeemed" not in df.columns:
        raise SystemExit("Column label_redeemed not found in dataset")

    df["assigned_at"] = pd.to_datetime(df["assigned_at"], errors="coerce")
    df = df[df["assigned_at"].notna()].copy()
    if df.empty:
        raise SystemExit("No rows with valid assigned_at")

    cat_cols = ["campaign_name", "offer_type", "target_scope"]
    num_cols = [
        "offer_value",
        "estimated_cost",
        "cooldown_days",
        "expires_in_days",
        "is_exposed",
        "is_clicked",
        "recency_days",
        "frequency_90d",
        "monetary_90d",
        "txn_count_before",
        "spend_before",
    ]

    for c in cat_cols:
        if c not in df.columns:
            df[c] = "unknown"
        df[c] = df[c].fillna("unknown").astype(str)
    for c in num_cols:
        if c not in df.columns:
            df[c] = 0.0
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    y = df["label_redeemed"].astype(int).values
    if len(np.unique(y)) < 2:
        raise SystemExit("Need both classes in label_redeemed for training")

    # Temporal split for realistic evaluation.
    cut = df["assigned_at"].quantile(0.8)
    train_df = df[df["assigned_at"] <= cut].copy()
    test_df = df[df["assigned_at"] > cut].copy()
    if train_df.empty or test_df.empty:
        raise SystemExit("Temporal split produced empty train or test set")

    X_train = train_df[cat_cols + num_cols]
    y_train = train_df["label_redeemed"].astype(int).values
    X_test = test_df[cat_cols + num_cols]
    y_test = test_df["label_redeemed"].astype(int).values

    prep = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
            ("num", StandardScaler(), num_cols),
        ]
    )
    model = LogisticRegression(
        max_iter=500,
        class_weight="balanced",
        random_state=args.seed,
    )
    pipe = Pipeline(
        steps=[
            ("prep", prep),
            ("model", model),
        ]
    )
    pipe.fit(X_train, y_train)

    p_test = pipe.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, p_test) if len(np.unique(y_test)) > 1 else 0.0
    pr_auc = average_precision_score(y_test, p_test) if len(np.unique(y_test)) > 1 else 0.0
    ll = log_loss(y_test, p_test, labels=[0, 1]) if len(np.unique(y_test)) > 1 else 0.0

    joblib.dump(pipe, os.path.join(args.out_dir, "model.pkl"))
    meta = {
        "model_version": "offer_redemption_lr_v1",
        "dataset": str(args.dataset),
        "seed": int(args.seed),
        "categorical_features": cat_cols,
        "numeric_features": num_cols,
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "positive_rate_train": round(float(train_df["label_redeemed"].mean()), 6),
        "positive_rate_test": round(float(test_df["label_redeemed"].mean()), 6),
        "auc_test": round(float(auc), 6),
        "pr_auc_test": round(float(pr_auc), 6),
        "logloss_test": round(float(ll), 6),
        "split_strategy": "temporal_quantile_0.8",
    }
    with open(os.path.join(args.out_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print("OK")
    print("saved:", os.path.join(args.out_dir, "model.pkl"))
    print("meta:", meta)


if __name__ == "__main__":
    main()
