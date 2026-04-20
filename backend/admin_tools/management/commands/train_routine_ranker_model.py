"""Train the skincare routine ranker.

Reads the dataset produced by build_routine_ranker_dataset and trains a
LightGBM LambdaRank model (falls back to a logistic baseline if LightGBM is
not installed). Saves model.pkl, metadata.json, and eval_report.json into the
target directory.
"""

from __future__ import annotations

import importlib.util
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from django.core.management.base import BaseCommand, CommandError

try:
    import joblib
except Exception:  # pragma: no cover
    joblib = None

try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None

try:
    import pandas as pd
except Exception:  # pragma: no cover
    pd = None


CATEGORICAL_FEATURES = ["skin_type", "budget", "product_type", "strength", "step"]
NUMERIC_FEATURES = [
    "price",
    "has_price",
    "in_stock",
    "skin_type_match",
    "goal_concern_match_count",
    "goals_total",
    "avoid_flag_hit",
    "actives_count",
    "concerns_count",
    "user_tx_count_90d",
    "user_owned_skincare_count",
    "product_popularity",
    "product_in_wishlist",
    "product_roadmap_clicks_30d",
    "product_roadmap_skips_30d",
]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _resolve_existing_dir(raw_path: str) -> Path:
    candidate = Path(str(raw_path)).expanduser()
    if candidate.is_absolute():
        resolved = candidate.resolve()
        if resolved.exists() and resolved.is_dir():
            return resolved
        raise CommandError(f"Directory not found: {raw_path}")
    for base in (Path.cwd(), _repo_root()):
        resolved = (base / candidate).resolve()
        if resolved.exists() and resolved.is_dir():
            return resolved
    raise CommandError(f"Directory not found: {raw_path}")


def _resolve_output_dir(raw_path: str) -> Path:
    candidate = Path(str(raw_path)).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    return (_repo_root() / candidate).resolve()


def _module_available(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except Exception:
        return False


def _load_dataset(data_dir: Path):
    parquet_path = data_dir / "dataset.parquet"
    csv_path = data_dir / "dataset.csv"
    if parquet_path.exists():
        return pd.read_parquet(parquet_path), str(parquet_path)
    if csv_path.exists():
        return pd.read_csv(csv_path), str(csv_path)
    raise CommandError(f"dataset.parquet or dataset.csv not found in {data_dir}")


def _user_split(df, *, test_ratio: float, seed: int):
    users = sorted(df["user_id"].unique().tolist())
    rng = np.random.RandomState(seed)
    rng.shuffle(users)
    cut = max(1, int(round(len(users) * (1.0 - test_ratio))))
    train_users = set(users[:cut])
    test_users = set(users[cut:]) or set(users[-1:])
    train_df = df[df["user_id"].isin(train_users)].reset_index(drop=True)
    test_df = df[df["user_id"].isin(test_users)].reset_index(drop=True)
    return train_df, test_df


def _recall_at_k(test_df, scores, k: int) -> float:
    test_df = test_df.copy()
    test_df["__score"] = scores
    hits = 0
    groups = 0
    for _, group in test_df.groupby("episode_id", sort=False):
        if int(group["y"].sum()) == 0:
            continue
        groups += 1
        ordered = group.sort_values("__score", ascending=False)
        topk = ordered.head(k)
        if int(topk["y"].sum()) > 0:
            hits += 1
    if groups == 0:
        return 0.0
    return hits / groups


def _ndcg_at_k(test_df, scores, k: int) -> float:
    test_df = test_df.copy()
    test_df["__score"] = scores
    total = 0.0
    groups = 0
    for _, group in test_df.groupby("episode_id", sort=False):
        if int(group["y"].sum()) == 0:
            continue
        groups += 1
        ordered = group.sort_values("__score", ascending=False).head(k)
        gains = ordered["y"].astype(float).to_numpy()
        discounts = np.log2(np.arange(2, len(gains) + 2))
        dcg = float((gains / discounts).sum())
        ideal_gains = np.array([1.0] + [0.0] * (len(gains) - 1))[: len(gains)]
        idcg = float((ideal_gains / discounts).sum())
        if idcg > 0:
            total += dcg / idcg
    if groups == 0:
        return 0.0
    return total / groups


def _prepare_matrix(df, *, categorical_maps: dict[str, dict[str, int]] | None = None):
    """Return (X, maps) where X is a numpy array and maps are string->int encodings."""
    maps: dict[str, dict[str, int]] = categorical_maps.copy() if categorical_maps else {}
    columns: list[np.ndarray] = []

    for col in CATEGORICAL_FEATURES:
        series = df[col].astype(str).fillna("")
        if col not in maps:
            unique = sorted(series.unique().tolist())
            maps[col] = {value: idx for idx, value in enumerate(unique)}
        encoded = series.map(lambda v: maps[col].get(v, -1)).astype(np.int32).to_numpy()
        columns.append(encoded)

    for col in NUMERIC_FEATURES:
        series = pd.to_numeric(df[col], errors="coerce").fillna(0.0).astype(np.float32)
        columns.append(series.to_numpy())

    X = np.column_stack(columns).astype(np.float32)
    return X, maps


def _train_lightgbm(train_df, test_df, *, maps, seed: int):
    import lightgbm as lgb

    X_train, maps = _prepare_matrix(train_df, categorical_maps=maps)
    X_test, _ = _prepare_matrix(test_df, categorical_maps=maps)
    y_train = train_df["y"].astype(int).to_numpy()

    group_train = train_df.groupby("episode_id", sort=False).size().astype(int).to_numpy()

    categorical_indices = list(range(len(CATEGORICAL_FEATURES)))

    model = lgb.LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        eval_at=[1, 3, 5],
        learning_rate=0.05,
        n_estimators=300,
        num_leaves=31,
        min_child_samples=10,
        random_state=seed,
        verbose=-1,
    )
    model.fit(
        X_train,
        y_train,
        group=group_train,
        categorical_feature=categorical_indices,
    )
    predictions = model.predict(X_test)
    return model, predictions, maps, "lightgbm_lambdarank"


def _train_logistic(train_df, test_df, *, maps, seed: int):
    from sklearn.linear_model import LogisticRegression

    X_train, maps = _prepare_matrix(train_df, categorical_maps=maps)
    X_test, _ = _prepare_matrix(test_df, categorical_maps=maps)
    y_train = train_df["y"].astype(int).to_numpy()

    model = LogisticRegression(max_iter=500, random_state=seed)
    model.fit(X_train, y_train)
    predictions = model.predict_proba(X_test)[:, 1]
    return model, predictions, maps, "logistic_fallback"


class Command(BaseCommand):
    help = "Train the skincare routine ranker model."

    def add_arguments(self, parser):
        parser.add_argument(
            "--data-dir",
            default="data/ml/routine_ranker_v1",
            help="Directory with dataset.parquet.",
        )
        parser.add_argument(
            "--output-dir",
            default="models/routine_ranker_v1",
            help="Directory to write model.pkl, metadata.json, eval_report.json.",
        )
        parser.add_argument(
            "--estimator",
            default="auto",
            choices=["auto", "lightgbm", "logistic"],
            help="Which backend to train.",
        )
        parser.add_argument("--test-ratio", type=float, default=0.2)
        parser.add_argument("--seed", type=int, default=42)

    def handle(self, *args, **options):
        if pd is None or np is None:
            raise CommandError("pandas and numpy are required.")
        if joblib is None:
            raise CommandError("joblib is required. Install with: pip install joblib")

        data_dir = _resolve_existing_dir(options["data_dir"])
        output_dir = _resolve_output_dir(options["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)

        df, dataset_path = _load_dataset(data_dir)

        missing_columns = {"y", "episode_id", "user_id"} - set(df.columns)
        if missing_columns:
            raise CommandError(f"Dataset missing required columns: {sorted(missing_columns)}")

        for col in CATEGORICAL_FEATURES + NUMERIC_FEATURES:
            if col not in df.columns:
                raise CommandError(f"Dataset missing feature column: {col}")

        train_df, test_df = _user_split(
            df, test_ratio=float(options["test_ratio"]), seed=int(options["seed"])
        )

        if train_df.empty or test_df.empty:
            raise CommandError("Train or test split is empty. Collect more data.")

        train_df = train_df.sort_values("episode_id").reset_index(drop=True)
        test_df = test_df.sort_values("episode_id").reset_index(drop=True)

        requested = str(options["estimator"]).lower()
        if requested == "auto":
            requested = "lightgbm" if _module_available("lightgbm") else "logistic"

        if requested == "lightgbm":
            if not _module_available("lightgbm"):
                raise CommandError("LightGBM not installed. Install with: pip install lightgbm")
            model, predictions, maps, model_type = _train_lightgbm(
                train_df, test_df, maps={}, seed=int(options["seed"])
            )
        else:
            model, predictions, maps, model_type = _train_logistic(
                train_df, test_df, maps={}, seed=int(options["seed"])
            )

        recall_at_1 = _recall_at_k(test_df, predictions, 1)
        recall_at_3 = _recall_at_k(test_df, predictions, 3)
        ndcg_at_5 = _ndcg_at_k(test_df, predictions, 5)

        artifact = {
            "model": model,
            "model_type": model_type,
            "categorical_features": CATEGORICAL_FEATURES,
            "numeric_features": NUMERIC_FEATURES,
            "feature_order": CATEGORICAL_FEATURES + NUMERIC_FEATURES,
            "categorical_maps": maps,
            "version": "routine_ranker_v1",
        }

        model_path = output_dir / "model.pkl"
        joblib.dump(artifact, model_path)

        metadata = {
            "version": "routine_ranker_v1",
            "model_type": model_type,
            "trained_at_utc": datetime.now(timezone.utc).isoformat(),
            "dataset_file": dataset_path,
            "train_rows": int(len(train_df)),
            "test_rows": int(len(test_df)),
            "train_groups": int(train_df["episode_id"].nunique()),
            "test_groups": int(test_df["episode_id"].nunique()),
            "categorical_features": CATEGORICAL_FEATURES,
            "numeric_features": NUMERIC_FEATURES,
            "seed": int(options["seed"]),
            "recall_at_1": float(recall_at_1),
            "recall_at_3": float(recall_at_3),
            "ndcg_at_5": float(ndcg_at_5),
        }
        with (output_dir / "metadata.json").open("w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        eval_report = {
            "recall_at_1": recall_at_1,
            "recall_at_3": recall_at_3,
            "ndcg_at_5": ndcg_at_5,
            "test_groups": int(test_df["episode_id"].nunique()),
            "test_positive_groups": int(
                test_df.groupby("episode_id")["y"].max().astype(int).sum()
            ),
            "model_type": model_type,
        }
        with (output_dir / "eval_report.json").open("w", encoding="utf-8") as f:
            json.dump(eval_report, f, indent=2, ensure_ascii=False)

        self.stdout.write(
            self.style.SUCCESS(
                f"Trained {model_type}. recall@1={recall_at_1:.3f} "
                f"recall@3={recall_at_3:.3f} ndcg@5={ndcg_at_5:.3f}"
            )
        )
        self.stdout.write(f"Saved model to {model_path}")
