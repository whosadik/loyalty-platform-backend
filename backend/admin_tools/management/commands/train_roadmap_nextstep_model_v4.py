from __future__ import annotations

import importlib.util
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

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

from django.core.management.base import BaseCommand, CommandError
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _resolve_existing_dir(raw_path: str) -> Path:
    candidate = Path(str(raw_path)).expanduser()
    candidates: list[Path] = []
    if candidate.is_absolute():
        candidates = [candidate]
    else:
        candidates = [(Path.cwd() / candidate), (_repo_root() / candidate)]
    for item in candidates:
        resolved = item.resolve()
        if resolved.exists() and resolved.is_dir():
            return resolved
    tried = ", ".join(str(x.resolve()) for x in candidates)
    raise CommandError(f"Directory not found: {raw_path}. Tried: {tried}")


def _resolve_output_dir(raw_path: str) -> Path:
    candidate = Path(str(raw_path)).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    cwd_path = (Path.cwd() / candidate).resolve()
    if cwd_path.parent.exists():
        return cwd_path
    return (_repo_root() / candidate).resolve()


def _load_dataset(data_dir: Path) -> tuple["pd.DataFrame", str]:
    parquet_path = data_dir / "dataset.parquet"
    csv_path = data_dir / "dataset.csv"
    if parquet_path.exists():
        return pd.read_parquet(parquet_path), str(parquet_path)
    if csv_path.exists():
        return pd.read_csv(csv_path), str(csv_path)
    raise CommandError(f"dataset.parquet or dataset.csv not found in {data_dir}")


def _prepare_split_set(payload: dict[str, Any], key: str) -> set[int]:
    return {int(value) for value in (payload.get(key) or [])}


def _module_available(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except Exception:
        return False


def _resolve_estimator_name(requested: str, *, allow_fallback: bool) -> str:
    requested = str(requested or "auto").strip().lower()
    if requested == "lightgbm":
        if not _module_available("lightgbm"):
            raise CommandError("LightGBM is not installed. Install with: pip install lightgbm")
        return requested
    if requested == "catboost":
        if not _module_available("catboost"):
            raise CommandError("CatBoost is not installed. Install with: pip install catboost")
        return requested
    if requested in {"logistic", "baseline"}:
        if not allow_fallback:
            raise CommandError(
                "Fallback estimators are disabled by default. Re-run with --allow-fallback to use logistic."
            )
        return "logistic"
    if requested != "auto":
        raise CommandError("--estimator must be one of: auto, lightgbm, catboost, logistic")
    if _module_available("lightgbm"):
        return "lightgbm"
    if _module_available("catboost"):
        return "catboost"
    if allow_fallback:
        return "logistic"
    raise CommandError(
        "No ranker backend is installed. Install lightgbm or catboost, "
        "or re-run with --allow-fallback to enable logistic baseline."
    )


def _param_grid(estimator_name: str) -> list[dict[str, Any]]:
    if estimator_name == "lightgbm":
        return [
            {"learning_rate": 0.05, "n_estimators": 250, "num_leaves": 31, "min_child_samples": 20},
            {"learning_rate": 0.05, "n_estimators": 400, "num_leaves": 63, "min_child_samples": 10},
            {"learning_rate": 0.03, "n_estimators": 600, "num_leaves": 127, "min_child_samples": 20},
        ]
    if estimator_name == "catboost":
        return [
            {"depth": 6, "learning_rate": 0.05, "iterations": 300},
            {"depth": 8, "learning_rate": 0.05, "iterations": 450},
            {"depth": 8, "learning_rate": 0.03, "iterations": 650},
        ]
    return [
        {"C": 0.5, "max_iter": 400},
        {"C": 1.0, "max_iter": 500},
        {"C": 2.0, "max_iter": 600},
    ]


def _baseline_feature_columns(feature_columns: list[str]) -> list[str]:
    preferred = [
        "category",
        "candidate_type",
        "last1_product_type",
        "last1_category",
        "month_of_year",
        "day_of_week",
        "candidate_is_fragrance_slot",
        "candidate_position_in_chain",
        "candidate_popularity_in_train",
    ]
    return [col for col in preferred if col in set(feature_columns)]


def _make_one_hot_encoder():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    except TypeError:  # pragma: no cover
        return OneHotEncoder(handle_unknown="ignore", sparse=True)


def _sort_frame(df: "pd.DataFrame") -> "pd.DataFrame":
    return df.sort_values(["episode_id", "candidate_type"]).reset_index(drop=True)


def _group_sizes(df: "pd.DataFrame") -> list[int]:
    return [int(x) for x in df.groupby("episode_id", sort=False).size().tolist()]


def _positive_group_mask(df: "pd.DataFrame") -> "pd.Series":
    return df.groupby("episode_id")["y"].transform("max").astype(int) > 0


def _assign_popularity_bucket(values: "pd.Series") -> "pd.Series":
    numeric = pd.to_numeric(values, errors="coerce").fillna(0.0).astype(float)
    if len(numeric) <= 2:
        return pd.Series(["mid"] * len(numeric), index=values.index)
    q1, q2 = numeric.quantile([0.33, 0.66]).tolist()

    def _bucket(v: float) -> str:
        if v <= q1:
            return "low"
        if v <= q2:
            return "mid"
        return "high"

    return numeric.map(_bucket)


def _negative_sample_train(
    df: "pd.DataFrame",
    *,
    max_negatives_per_episode: int,
    seed: int,
) -> "pd.DataFrame":
    if max_negatives_per_episode <= 0 or df.empty:
        return _sort_frame(df)

    work = _sort_frame(df).copy()
    work["__bucket"] = _assign_popularity_bucket(work.get("candidate_popularity_in_train", 0.0))
    rng = np.random.RandomState(seed)
    kept_frames: list["pd.DataFrame"] = []
    for _, group in work.groupby("episode_id", sort=False):
        pos = group[group["y"].astype(int) == 1]
        neg = group[group["y"].astype(int) == 0]
        if len(neg) <= max_negatives_per_episode:
            selected_neg = neg
        else:
            quotas = {"high": max_negatives_per_episode // 3, "mid": max_negatives_per_episode // 3}
            quotas["low"] = max_negatives_per_episode - quotas["high"] - quotas["mid"]
            picked_index: list[int] = []
            for bucket_name in ["high", "mid", "low"]:
                part = neg[neg["__bucket"] == bucket_name]
                take = min(len(part), int(quotas[bucket_name]))
                if take > 0:
                    picked = rng.choice(part.index.to_numpy(), size=take, replace=False)
                    picked_index.extend(int(x) for x in picked.tolist())
            remaining_budget = max_negatives_per_episode - len(picked_index)
            if remaining_budget > 0:
                rest = neg.loc[~neg.index.isin(picked_index)]
                if not rest.empty:
                    extra = rng.choice(
                        rest.index.to_numpy(),
                        size=min(len(rest), int(remaining_budget)),
                        replace=False,
                    )
                    picked_index.extend(int(x) for x in extra.tolist())
            selected_neg = neg.loc[sorted(set(picked_index))]
        kept_frames.append(pd.concat([pos, selected_neg], axis=0, ignore_index=False))
    sampled = pd.concat(kept_frames, axis=0).drop(columns=["__bucket"], errors="ignore")
    return _sort_frame(sampled)


def _prepare_features(
    df: "pd.DataFrame",
    *,
    feature_columns: list[str],
    categorical_features: list[str],
    numeric_features: list[str],
    estimator_name: str,
) -> "pd.DataFrame":
    X = df[feature_columns].copy()
    for col in categorical_features:
        if col in X.columns:
            X[col] = X[col].fillna("__none__").astype(str)
            if estimator_name == "lightgbm":
                X[col] = X[col].astype("category")
    for col in numeric_features:
        if col in X.columns:
            X[col] = pd.to_numeric(X[col], errors="coerce").fillna(0.0)
    return X


def _build_logistic_bundle(
    *,
    X_train: "pd.DataFrame",
    y_train: "np.ndarray",
    categorical_features: list[str],
    numeric_features: list[str],
    params: dict[str, Any],
    seed: int,
) -> dict[str, Any]:
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="constant", fill_value="__none__")),
                        ("onehot", _make_one_hot_encoder()),
                    ]
                ),
                categorical_features,
            ),
            (
                "numeric",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="constant", fill_value=0.0)),
                    ]
                ),
                numeric_features,
            ),
        ],
        remainder="drop",
    )
    model = LogisticRegression(
        random_state=seed,
        solver="liblinear",
        C=float(params.get("C", 1.0)),
        max_iter=int(params.get("max_iter", 500)),
    )
    X_train_model = preprocessor.fit_transform(X_train)
    model.fit(X_train_model, y_train)
    return {"model": model, "preprocessor": preprocessor, "model_type": "logistic_pairwise"}


def _dense_group_ids(group_sizes: list[int]) -> list[int]:
    out: list[int] = []
    group_id = 1
    for size in group_sizes:
        out.extend([group_id] * int(size))
        group_id += 1
    return out


def _build_ranker_bundle(
    *,
    estimator_name: str,
    X_train: "pd.DataFrame",
    y_train: "np.ndarray",
    group_train: list[int],
    X_val: "pd.DataFrame",
    y_val: "np.ndarray",
    group_val: list[int],
    categorical_features: list[str],
    params: dict[str, Any],
    seed: int,
) -> dict[str, Any]:
    if estimator_name == "lightgbm":
        from lightgbm import LGBMRanker

        model = LGBMRanker(
            objective="lambdarank",
            metric="ndcg",
            random_state=seed,
            verbosity=-1,
            **params,
        )
        model.fit(
            X_train,
            y_train,
            group=group_train,
            eval_set=[(X_val, y_val)],
            eval_group=[group_val],
            eval_at=[1, 3, 5],
            categorical_feature=[col for col in categorical_features if col in X_train.columns],
        )
        return {"model": model, "preprocessor": None, "model_type": "lightgbm_ranker"}

    from catboost import CatBoostRanker, Pool

    cat_idx = [int(X_train.columns.get_loc(col)) for col in categorical_features if col in X_train.columns]
    train_pool = Pool(X_train, y_train, group_id=_dense_group_ids(group_train), cat_features=cat_idx)
    val_pool = Pool(X_val, y_val, group_id=_dense_group_ids(group_val), cat_features=cat_idx)
    model = CatBoostRanker(
        loss_function="YetiRankPairwise",
        eval_metric="NDCG:top=5",
        random_seed=seed,
        allow_writing_files=False,
        verbose=False,
        **params,
    )
    model.fit(train_pool, eval_set=val_pool, use_best_model=False)
    return {"model": model, "preprocessor": None, "model_type": "catboost_ranker"}


def _predict_raw_scores(bundle: dict[str, Any], X: "pd.DataFrame") -> "np.ndarray":
    model = bundle["model"]
    preprocessor = bundle.get("preprocessor")
    if preprocessor is not None:
        X_model = preprocessor.transform(X)
        if hasattr(model, "decision_function"):
            raw = model.decision_function(X_model)
        elif hasattr(model, "predict_proba"):
            raw = model.predict_proba(X_model)
        else:
            raw = model.predict(X_model)
    else:
        raw = model.predict(X)
    arr = np.asarray(raw)
    if arr.ndim > 1:
        if arr.shape[1] >= 2:
            return np.asarray(arr[:, -1], dtype=float)
        return np.asarray(arr.reshape(-1), dtype=float)
    return np.asarray(arr, dtype=float).reshape(-1)


def _softmax(values: "np.ndarray", temperature: float) -> "np.ndarray":
    t = max(0.05, float(temperature))
    arr = np.asarray(values, dtype=float) / t
    arr = arr - float(arr.max())
    exp = np.exp(arr)
    den = float(exp.sum())
    if den <= 0:
        den = 1.0
    return exp / den


def _temperature_nll(df: "pd.DataFrame", raw_scores: "np.ndarray", temperature: float) -> float:
    y = df["y"].astype(int).to_numpy()
    group_sizes = _group_sizes(df)
    losses: list[float] = []
    offset = 0
    for size in group_sizes:
        group_scores = raw_scores[offset : offset + size]
        group_y = y[offset : offset + size]
        offset += size
        if int(group_y.sum()) <= 0:
            continue
        label_idx = int(np.argmax(group_y))
        probs = _softmax(group_scores, temperature)
        losses.append(float(-math.log(max(float(probs[label_idx]), 1e-12))))
    return float(sum(losses) / len(losses)) if losses else 0.0


def _fit_temperature(df: "pd.DataFrame", raw_scores: "np.ndarray") -> tuple[float, float, float]:
    best_t = 1.0
    best_after = float("inf")
    before = _temperature_nll(df, raw_scores, 1.0)
    for t in [x / 100.0 for x in range(50, 301, 5)]:
        nll = _temperature_nll(df, raw_scores, float(t))
        if nll < best_after:
            best_after = nll
            best_t = float(t)
    return best_t, before, best_after


def _ece_score(confidences: "np.ndarray", correct: "np.ndarray", *, n_bins: int = 10) -> float:
    if len(confidences) == 0:
        return 0.0
    ece = 0.0
    for idx in range(n_bins):
        low = idx / n_bins
        high = (idx + 1) / n_bins
        if idx == 0:
            mask = (confidences >= low) & (confidences <= high)
        else:
            mask = (confidences > low) & (confidences <= high)
        if not np.any(mask):
            continue
        conf_bin = float(confidences[mask].mean())
        acc_bin = float(correct[mask].mean())
        ece += abs(acc_bin - conf_bin) * float(mask.mean())
    return float(ece)


def _evaluate_scores(
    *,
    df: "pd.DataFrame",
    raw_scores: "np.ndarray",
    temperature: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    thresholds = [0.05, 0.10, 0.20, 0.35, 0.50, 0.65, 0.80]
    y = df["y"].astype(int).to_numpy()
    categories = df["category"].astype(str).to_numpy()
    group_sizes = _group_sizes(df)
    total_groups = len(group_sizes)

    positives = 0
    none_episodes = 0
    hits_1 = 0
    hits_3 = 0
    hits_5 = 0
    ndcg_5_sum = 0.0
    confs: list[float] = []
    corrects: list[int] = []
    positive_flags: list[int] = []
    positive_top1_hits: list[int] = []
    per_category: dict[str, dict[str, float]] = defaultdict(
        lambda: {
            "groups": 0,
            "positive_episodes": 0,
            "recall_at_1": 0.0,
            "recall_at_3": 0.0,
            "recall_at_5": 0.0,
            "ndcg_at_5": 0.0,
        }
    )

    offset = 0
    for size in group_sizes:
        group_scores = raw_scores[offset : offset + size]
        group_y = y[offset : offset + size]
        category = str(categories[offset])
        probs = _softmax(group_scores, temperature)
        order = np.argsort(-group_scores, kind="mergesort")
        top_idx = int(order[0])
        top1_correct = int(group_y[top_idx] == 1)
        confs.append(float(probs[top_idx]))
        corrects.append(int(top1_correct))
        per_category[category]["groups"] += 1

        if int(group_y.sum()) > 0:
            positives += 1
            positive_flags.append(1)
            positive_top1_hits.append(int(top1_correct))
            per_category[category]["positive_episodes"] += 1
            label_idx = int(np.argmax(group_y))
            rank = int(np.where(order == label_idx)[0][0] + 1)
            if rank == 1:
                hits_1 += 1
                per_category[category]["recall_at_1"] += 1.0
            if rank <= 3:
                hits_3 += 1
                per_category[category]["recall_at_3"] += 1.0
            if rank <= 5:
                hits_5 += 1
                dcg = float(1.0 / math.log2(rank + 1.0))
                ndcg_5_sum += dcg
                per_category[category]["recall_at_5"] += 1.0
                per_category[category]["ndcg_at_5"] += dcg
        else:
            none_episodes += 1
            positive_flags.append(0)
            positive_top1_hits.append(0)
        offset += size

    conf_arr = np.asarray(confs, dtype=float)
    correct_arr = np.asarray(corrects, dtype=float)
    positive_arr = np.asarray(positive_flags, dtype=bool)
    positive_top1_arr = np.asarray(positive_top1_hits, dtype=float)
    coverage_curve: list[dict[str, Any]] = []
    for threshold in thresholds:
        covered_mask = conf_arr >= float(threshold)
        covered = int(covered_mask.sum())
        positive_covered = covered_mask & positive_arr
        recall_on_covered = float(positive_top1_arr[positive_covered].mean()) if np.any(positive_covered) else 0.0
        top1_on_covered = float(correct_arr[covered_mask].mean()) if covered > 0 else 0.0
        coverage_curve.append(
            {
                "threshold": float(threshold),
                "coverage": round(float(covered / max(1, total_groups)), 6),
                "covered_groups": covered,
                "recall_at_1_on_covered": round(recall_on_covered, 6),
                "top1_accuracy_on_covered": round(top1_on_covered, 6),
            }
        )

    metrics = {
        "rows": int(len(df)),
        "groups": int(total_groups),
        "positive_episodes": int(positives),
        "none_episodes": int(none_episodes),
        "recall_at_1": round(float(hits_1 / max(1, positives)), 6),
        "recall_at_3": round(float(hits_3 / max(1, positives)), 6),
        "recall_at_5": round(float(hits_5 / max(1, positives)), 6),
        "ndcg_at_5": round(float(ndcg_5_sum / max(1, positives)), 6),
        "coverage_curve": coverage_curve,
        "max_score_mean": round(float(conf_arr.mean()) if len(conf_arr) else 0.0, 6),
        "ece": round(float(_ece_score(conf_arr, correct_arr)), 6),
        "brier": round(float(np.mean((conf_arr - correct_arr) ** 2)) if len(conf_arr) else 0.0, 6),
    }

    normalized_per_category: dict[str, Any] = {}
    for category, row in per_category.items():
        denom = int(row["positive_episodes"])
        normalized_per_category[category] = {
            "groups": int(row["groups"]),
            "positive_episodes": denom,
            "recall_at_1": round(float(row["recall_at_1"] / max(1, denom)), 6),
            "recall_at_3": round(float(row["recall_at_3"] / max(1, denom)), 6),
            "recall_at_5": round(float(row["recall_at_5"] / max(1, denom)), 6),
            "ndcg_at_5": round(float(row["ndcg_at_5"] / max(1, denom)), 6),
        }
    return metrics, normalized_per_category


def _compare_with_baselines(
    *,
    metrics_val: dict[str, Any],
    metrics_test: dict[str, Any],
    dataset_baselines: dict[str, Any],
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    baselines = (dataset_baselines or {}).get("splits") or {}
    for split_name, metrics in {"val": metrics_val, "test": metrics_test}.items():
        split_baselines = baselines.get(split_name) or {}
        out[split_name] = {}
        for baseline_name in ["popularity", "markov"]:
            base_row = split_baselines.get(baseline_name) or {}
            out[split_name][baseline_name] = {
                "recall_at_1": round(
                    float(metrics.get("recall_at_1", 0.0) - float(base_row.get("recall_at_1", 0.0))),
                    6,
                ),
                "recall_at_3": round(
                    float(metrics.get("recall_at_3", 0.0) - float(base_row.get("recall_at_3", 0.0))),
                    6,
                ),
                "recall_at_5": round(
                    float(metrics.get("recall_at_5", 0.0) - float(base_row.get("recall_at_5", 0.0))),
                    6,
                ),
                "ndcg_at_5": round(
                    float(metrics.get("ndcg_at_5", 0.0) - float(base_row.get("ndcg_at_5", 0.0))),
                    6,
                ),
            }
    return out


def _build_eval_markdown(report: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# Roadmap NextStep v4 Evaluation")
    lines.append("")
    lines.append("## Summary")
    lines.append(f"- trained_at_utc: `{report['trained_at_utc']}`")
    lines.append(f"- estimator: `{report['estimator']}`")
    lines.append(f"- selected_feature_set: `{report['selected_feature_set']}`")
    lines.append(f"- temperature: `{float(report.get('temperature', 1.0)):.4f}`")
    lines.append("")
    lines.append("## Metrics")
    lines.append("| split | recall@1 | recall@3 | recall@5 | ndcg@5 | ece | brier |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- |")
    for split_name in ["val", "test"]:
        row = report[f"metrics_{split_name}"]
        lines.append(
            f"| {split_name} | {row['recall_at_1']:.4f} | {row['recall_at_3']:.4f} | "
            f"{row['recall_at_5']:.4f} | {row['ndcg_at_5']:.4f} | {row['ece']:.4f} | {row['brier']:.4f} |"
        )
    lines.append("")
    lines.append("## Baseline Comparison (test)")
    lines.append("| baseline | dRecall@1 | dRecall@3 | dRecall@5 | dNDCG@5 |")
    lines.append("| --- | --- | --- | --- | --- |")
    for baseline_name, row in (report.get("baseline_comparison") or {}).get("test", {}).items():
        lines.append(
            f"| {baseline_name} | {row['recall_at_1']:.4f} | {row['recall_at_3']:.4f} | "
            f"{row['recall_at_5']:.4f} | {row['ndcg_at_5']:.4f} |"
        )
    lines.append("")
    lines.append("## Feature Ablation")
    lines.append("| feature_set | val_ndcg@5 | test_ndcg@5 | test_recall@1 |")
    lines.append("| --- | --- | --- | --- |")
    for name, row in (report.get("feature_ablation") or {}).items():
        metrics_val = row.get("metrics_val") or {}
        metrics_test = row.get("metrics_test") or {}
        if not metrics_val or not metrics_test:
            continue
        lines.append(
            f"| {name} | {metrics_val['ndcg_at_5']:.4f} | {metrics_test['ndcg_at_5']:.4f} | {metrics_test['recall_at_1']:.4f} |"
        )
    lines.append("")
    return "\n".join(lines)


class Command(BaseCommand):
    help = "Train Roadmap NextStep v4 ranker with negative sampling, ablation and baseline comparison."

    def add_arguments(self, parser):
        parser.add_argument("--data-dir", type=str, default="data/ml/roadmap_nextstep_v4")
        parser.add_argument("--model-dir", type=str, default="models/roadmap_next_step_v4")
        parser.add_argument("--model-version", type=str, default="roadmap_next_step_v4")
        parser.add_argument("--seed", type=int, default=42)
        parser.add_argument("--trials", type=int, default=3)
        parser.add_argument("--estimator", type=str, default="auto", help="auto|lightgbm|catboost|logistic")
        parser.add_argument(
            "--allow-fallback",
            action="store_true",
            default=False,
            help="Allow logistic fallback when no ranker library is available.",
        )
        parser.add_argument(
            "--negative-samples-per-episode",
            type=int,
            default=30,
            help="Keep all positives and up to N negatives per episode in train split.",
        )

    def handle(self, *args, **options):
        if pd is None or np is None:
            raise CommandError("pandas + numpy are required. Install requirements-ml.txt")
        if joblib is None:
            raise CommandError("joblib is required. Install requirements-ml.txt")

        data_dir = _resolve_existing_dir(str(options["data_dir"]))
        model_dir = _resolve_output_dir(str(options["model_dir"]))
        model_version = str(options["model_version"]).strip() or "roadmap_next_step_v4"
        seed = int(options["seed"])
        max_trials = int(options["trials"])
        allow_fallback = bool(options["allow_fallback"])
        estimator_name = _resolve_estimator_name(
            str(options.get("estimator") or "auto"),
            allow_fallback=allow_fallback,
        )
        max_negatives_per_episode = int(options["negative_samples_per_episode"])
        if max_trials <= 0:
            raise CommandError("--trials must be > 0")
        if max_negatives_per_episode <= 0:
            raise CommandError("--negative-samples-per-episode must be > 0")

        splits_path = data_dir / "splits.json"
        metadata_path = data_dir / "metadata.json"
        if not splits_path.exists():
            raise CommandError(f"Missing splits.json in {data_dir}")
        if not metadata_path.exists():
            raise CommandError(f"Missing metadata.json in {data_dir}")

        dataset_df, dataset_path = _load_dataset(data_dir)
        split_payload = json.loads(splits_path.read_text(encoding="utf-8"))
        dataset_meta = json.loads(metadata_path.read_text(encoding="utf-8"))

        required_cols = {"user_id", "episode_id", "group_id", "y", "category", "candidate_type"}
        missing = sorted(required_cols.difference(set(dataset_df.columns)))
        if missing:
            raise CommandError(f"Dataset missing required columns: {missing}")

        work = dataset_df.copy()
        work["user_id"] = work["user_id"].astype(int)
        work["episode_id"] = work["episode_id"].astype(int)
        work["group_id"] = work["group_id"].astype(int)
        work["y"] = work["y"].astype(int)
        work["category"] = work["category"].fillna("").astype(str).str.strip().str.lower()
        work["candidate_type"] = work["candidate_type"].fillna("").astype(str).str.strip().str.lower()
        work = work[work["candidate_type"] != ""].copy()
        if work.empty:
            raise CommandError("Dataset is empty after normalization.")

        train_users = _prepare_split_set(split_payload, "train_user_ids")
        val_users = _prepare_split_set(split_payload, "val_user_ids")
        test_users = _prepare_split_set(split_payload, "test_user_ids")
        if not train_users or not val_users or not test_users:
            raise CommandError("splits.json has empty train/val/test user lists")

        feature_columns = [str(x) for x in (dataset_meta.get("feature_columns") or []) if str(x)]
        if not feature_columns:
            ignore_cols = {"user_id", "episode_id", "group_id", "split", "t0_utc", "label", "y"}
            feature_columns = [col for col in work.columns if col not in ignore_cols]
        for col in feature_columns:
            if col not in work.columns:
                raise CommandError(f"Feature column missing in dataset: {col}")

        categorical_features = [
            col for col in (dataset_meta.get("categorical_features") or []) if col in feature_columns
        ]
        numeric_features = [
            col for col in (dataset_meta.get("numeric_features") or []) if col in feature_columns
        ]
        if not categorical_features and not numeric_features:
            guessed_cat = [col for col in feature_columns if str(work[col].dtype) == "object"]
            guessed_num = [col for col in feature_columns if col not in guessed_cat]
            categorical_features = guessed_cat
            numeric_features = guessed_num

        baseline_features = _baseline_feature_columns(feature_columns)
        if not baseline_features:
            raise CommandError("Could not resolve baseline-only feature subset.")

        train_df_full = _sort_frame(work[work["user_id"].isin(train_users)].copy())
        val_df = _sort_frame(work[work["user_id"].isin(val_users)].copy())
        test_df = _sort_frame(work[work["user_id"].isin(test_users)].copy())
        if train_df_full.empty or val_df.empty or test_df.empty:
            raise CommandError(
                "User-level split produced empty split: "
                f"train={len(train_df_full)} val={len(val_df)} test={len(test_df)}"
            )

        sampled_train_df = _negative_sample_train(
            train_df_full,
            max_negatives_per_episode=max_negatives_per_episode,
            seed=seed,
        )
        if estimator_name in {"lightgbm", "catboost"}:
            fit_train_df = _sort_frame(sampled_train_df[_positive_group_mask(sampled_train_df)].copy())
            if fit_train_df.empty:
                raise CommandError("Ranker training set has no positive episodes after sampling.")
        else:
            fit_train_df = sampled_train_df

        feature_sets = {"baseline_only": baseline_features, "full": feature_columns}
        candidate_grid = _param_grid(estimator_name)[:max_trials]
        feature_results: dict[str, Any] = {}

        for feature_set_name, feature_set_columns in feature_sets.items():
            cat_cols = [col for col in categorical_features if col in feature_set_columns]
            num_cols = [col for col in numeric_features if col in feature_set_columns]
            X_train = _prepare_features(
                fit_train_df,
                feature_columns=feature_set_columns,
                categorical_features=cat_cols,
                numeric_features=num_cols,
                estimator_name=estimator_name,
            )
            X_val = _prepare_features(
                val_df,
                feature_columns=feature_set_columns,
                categorical_features=cat_cols,
                numeric_features=num_cols,
                estimator_name=estimator_name,
            )
            X_test = _prepare_features(
                test_df,
                feature_columns=feature_set_columns,
                categorical_features=cat_cols,
                numeric_features=num_cols,
                estimator_name=estimator_name,
            )
            X_train_metrics = _prepare_features(
                sampled_train_df,
                feature_columns=feature_set_columns,
                categorical_features=cat_cols,
                numeric_features=num_cols,
                estimator_name=estimator_name,
            )

            y_train = fit_train_df["y"].astype(int).to_numpy()
            group_train = _group_sizes(fit_train_df)
            y_val = val_df["y"].astype(int).to_numpy()
            group_val = _group_sizes(val_df)

            best_score: tuple[float, float, float] | None = None
            best_bundle: dict[str, Any] | None = None
            best_params: dict[str, Any] = {}
            best_metrics_val: dict[str, Any] | None = None
            best_per_category_val: dict[str, Any] | None = None
            best_temperature = 1.0
            best_temp_before = 0.0
            best_temp_after = 0.0
            tuning_rows: list[dict[str, Any]] = []

            for trial_idx, params in enumerate(candidate_grid, start=1):
                self.stdout.write(
                    f"[train_roadmap_nextstep_model_v4] feature_set={feature_set_name} "
                    f"trial={trial_idx}/{len(candidate_grid)} estimator={estimator_name} params={params}"
                )
                if estimator_name == "logistic":
                    bundle = _build_logistic_bundle(
                        X_train=X_train,
                        y_train=y_train,
                        categorical_features=cat_cols,
                        numeric_features=num_cols,
                        params=params,
                        seed=seed,
                    )
                else:
                    bundle = _build_ranker_bundle(
                        estimator_name=estimator_name,
                        X_train=X_train,
                        y_train=y_train,
                        group_train=group_train,
                        X_val=X_val,
                        y_val=y_val,
                        group_val=group_val,
                        categorical_features=cat_cols,
                        params=params,
                        seed=seed,
                    )
                raw_val_scores = _predict_raw_scores(bundle, X_val)
                temperature, temp_before, temp_after = _fit_temperature(val_df, raw_val_scores)
                metrics_val, per_category_val = _evaluate_scores(
                    df=val_df,
                    raw_scores=raw_val_scores,
                    temperature=temperature,
                )
                metrics_val["temperature_nll_before"] = round(float(temp_before), 6)
                metrics_val["temperature_nll_after"] = round(float(temp_after), 6)
                score_tuple = (
                    float(metrics_val["ndcg_at_5"]),
                    float(metrics_val["recall_at_1"]),
                    float(metrics_val["recall_at_3"]),
                )
                tuning_rows.append(
                    {
                        "trial": trial_idx,
                        "params": params,
                        "val_metrics": {
                            "recall_at_1": metrics_val["recall_at_1"],
                            "recall_at_3": metrics_val["recall_at_3"],
                            "recall_at_5": metrics_val["recall_at_5"],
                            "ndcg_at_5": metrics_val["ndcg_at_5"],
                        },
                    }
                )
                if best_score is None or score_tuple > best_score:
                    best_score = score_tuple
                    best_bundle = bundle
                    best_params = dict(params)
                    best_metrics_val = metrics_val
                    best_per_category_val = per_category_val
                    best_temperature = float(temperature)
                    best_temp_before = float(temp_before)
                    best_temp_after = float(temp_after)

            if best_bundle is None or best_metrics_val is None or best_per_category_val is None:
                raise CommandError(f"Failed to train feature set: {feature_set_name}")

            raw_train_scores = _predict_raw_scores(best_bundle, X_train_metrics)
            raw_test_scores = _predict_raw_scores(best_bundle, X_test)
            metrics_train, per_category_train = _evaluate_scores(
                df=sampled_train_df,
                raw_scores=raw_train_scores,
                temperature=best_temperature,
            )
            metrics_test, per_category_test = _evaluate_scores(
                df=test_df,
                raw_scores=raw_test_scores,
                temperature=best_temperature,
            )
            feature_results[feature_set_name] = {
                "bundle": best_bundle,
                "params": best_params,
                "temperature": best_temperature,
                "temperature_nll_before": best_temp_before,
                "temperature_nll_after": best_temp_after,
                "feature_columns": feature_set_columns,
                "categorical_features": cat_cols,
                "numeric_features": num_cols,
                "metrics_train": metrics_train,
                "metrics_val": best_metrics_val,
                "metrics_test": metrics_test,
                "per_category_train": per_category_train,
                "per_category_val": best_per_category_val,
                "per_category_test": per_category_test,
                "tuning_rows": tuning_rows,
            }

        selected_feature_set = max(
            feature_results.keys(),
            key=lambda name: (
                float(feature_results[name]["metrics_val"]["ndcg_at_5"]),
                float(feature_results[name]["metrics_val"]["recall_at_1"]),
                float(feature_results[name]["metrics_val"]["recall_at_3"]),
            ),
        )
        selected = feature_results[selected_feature_set]
        dataset_baselines = dataset_meta.get("baselines") or {}
        baseline_comparison = _compare_with_baselines(
            metrics_val=selected["metrics_val"],
            metrics_test=selected["metrics_test"],
            dataset_baselines=dataset_baselines,
        )

        trained_at = datetime.now(tz=timezone.utc).isoformat().replace("+00:00", "Z")
        artifact = {
            "task": "roadmap_nextstep_v4_ranking",
            "model": selected["bundle"]["model"],
            "preprocessor": selected["bundle"].get("preprocessor"),
            "model_type": selected["bundle"]["model_type"],
            "feature_columns": selected["feature_columns"],
            "categorical_features": selected["categorical_features"],
            "numeric_features": selected["numeric_features"],
            "temperature": float(round(selected["temperature"], 6)),
            "trained_at_utc": trained_at,
            "model_version": model_version,
            "candidate_types_by_category": dataset_meta.get("candidate_types_by_category") or {},
            "rules_chain_by_category": dataset_meta.get("rules_chain_by_category") or {},
            "candidate_popularity_in_train_by_category": dataset_meta.get("candidate_popularity_in_train_by_category") or {},
            "owned_feature_columns": dataset_meta.get("owned_feature_columns") or [],
            "owned_feature_map": dataset_meta.get("owned_feature_map") or {},
            "selected_feature_set": selected_feature_set,
        }

        model_dir.mkdir(parents=True, exist_ok=True)
        model_path = model_dir / "model.pkl"
        joblib.dump(artifact, model_path)

        model_metadata = {
            "trained_at_utc": trained_at,
            "model_version": model_version,
            "estimator": estimator_name,
            "selected_feature_set": selected_feature_set,
            "model_type": selected["bundle"]["model_type"],
            "dataset_path": dataset_path,
            "train_rows": int(len(train_df_full)),
            "val_rows": int(len(val_df)),
            "test_rows": int(len(test_df)),
            "train_rows_fit": int(len(fit_train_df)),
            "train_rows_sampled": int(len(sampled_train_df)),
            "train_groups": int(train_df_full["episode_id"].nunique()),
            "val_groups": int(val_df["episode_id"].nunique()),
            "test_groups": int(test_df["episode_id"].nunique()),
            "feature_columns": selected["feature_columns"],
            "categorical_features": selected["categorical_features"],
            "numeric_features": selected["numeric_features"],
            "temperature": float(round(selected["temperature"], 6)),
            "negative_samples_per_episode": int(max_negatives_per_episode),
            "selected_params": selected["params"],
            "candidate_types_by_category": dataset_meta.get("candidate_types_by_category") or {},
        }
        model_metadata_path = model_dir / "metadata.json"
        model_metadata_path.write_text(json.dumps(model_metadata, ensure_ascii=False, indent=2), encoding="utf-8")

        feature_ablation = {
            name: {
                "feature_count": len(row["feature_columns"]),
                "selected_params": row["params"],
                "metrics_val": row["metrics_val"],
                "metrics_test": row["metrics_test"],
            }
            for name, row in feature_results.items()
        }
        if "baseline_only" in feature_ablation and "full" in feature_ablation:
            feature_ablation["lift_full_vs_baseline"] = {
                "val_ndcg_at_5": round(
                    float(feature_results["full"]["metrics_val"]["ndcg_at_5"] - feature_results["baseline_only"]["metrics_val"]["ndcg_at_5"]),
                    6,
                ),
                "test_ndcg_at_5": round(
                    float(feature_results["full"]["metrics_test"]["ndcg_at_5"] - feature_results["baseline_only"]["metrics_test"]["ndcg_at_5"]),
                    6,
                ),
                "test_recall_at_1": round(
                    float(feature_results["full"]["metrics_test"]["recall_at_1"] - feature_results["baseline_only"]["metrics_test"]["recall_at_1"]),
                    6,
                ),
            }

        report = {
            "trained_at_utc": trained_at,
            "model_version": model_version,
            "estimator": estimator_name,
            "selected_feature_set": selected_feature_set,
            "selected_params": selected["params"],
            "temperature": float(round(selected["temperature"], 6)),
            "dataset_path": dataset_path,
            "train_rows": int(len(train_df_full)),
            "val_rows": int(len(val_df)),
            "test_rows": int(len(test_df)),
            "train_rows_fit": int(len(fit_train_df)),
            "train_rows_sampled": int(len(sampled_train_df)),
            "metrics_train": selected["metrics_train"],
            "metrics_val": selected["metrics_val"],
            "metrics_test": selected["metrics_test"],
            "per_category_train": selected["per_category_train"],
            "per_category_val": selected["per_category_val"],
            "per_category_test": selected["per_category_test"],
            "dataset_baselines": dataset_baselines,
            "baseline_comparison": baseline_comparison,
            "feature_ablation": feature_ablation,
            "negative_sampling": {
                "max_negatives_per_episode": int(max_negatives_per_episode),
                "train_rows_before": int(len(train_df_full)),
                "train_rows_after": int(len(sampled_train_df)),
                "fit_rows_after_positive_filter": int(len(fit_train_df)),
            },
            "runtime_guard": {
                "metric": "ndcg_at_5",
                "required_delta": 0.01,
                "model_value": float(selected["metrics_test"]["ndcg_at_5"]),
                "popularity_value": float(
                    (((dataset_baselines.get("splits") or {}).get("test") or {}).get("popularity") or {}).get("ndcg_at_5", 0.0)
                ),
                "passed": float(selected["metrics_test"]["ndcg_at_5"]) >= float(
                    (((dataset_baselines.get("splits") or {}).get("test") or {}).get("popularity") or {}).get("ndcg_at_5", 0.0)
                ) + 0.01,
            },
        }

        reports_dir = (_repo_root() / "reports").resolve()
        reports_dir.mkdir(parents=True, exist_ok=True)
        report_json_path = reports_dir / "roadmap_nextstep_v4_eval.json"
        report_md_path = reports_dir / "roadmap_nextstep_v4_eval.md"
        report_json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        report_md_path.write_text(_build_eval_markdown(report), encoding="utf-8")

        self.stdout.write("[train_roadmap_nextstep_model_v4] done")
        self.stdout.write(f"[train_roadmap_nextstep_model_v4] estimator={estimator_name}")
        self.stdout.write(f"[train_roadmap_nextstep_model_v4] selected_feature_set={selected_feature_set}")
        self.stdout.write(f"[train_roadmap_nextstep_model_v4] model={model_path}")
        self.stdout.write(f"[train_roadmap_nextstep_model_v4] metadata={model_metadata_path}")
        self.stdout.write(f"[train_roadmap_nextstep_model_v4] report_json={report_json_path}")
        self.stdout.write(f"[train_roadmap_nextstep_model_v4] report_md={report_md_path}")
        self.stdout.write(
            "[train_roadmap_nextstep_model_v4] "
            f"test_recall@1={selected['metrics_test']['recall_at_1']:.4f} "
            f"test_recall@3={selected['metrics_test']['recall_at_3']:.4f} "
            f"test_recall@5={selected['metrics_test']['recall_at_5']:.4f} "
            f"test_ndcg@5={selected['metrics_test']['ndcg_at_5']:.4f}"
        )
