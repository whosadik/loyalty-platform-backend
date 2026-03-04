from __future__ import annotations

import json
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
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import confusion_matrix, log_loss
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
        candidates = [
            (Path.cwd() / candidate),
            (_repo_root() / candidate),
        ]
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
    values = payload.get(key) or []
    out: set[int] = set()
    for value in values:
        out.add(int(value))
    return out


def _make_one_hot_encoder():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:  # pragma: no cover
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def _check_dependency_or_raise(estimator_name: str) -> None:
    if estimator_name == "catboost":
        try:
            import catboost  # noqa: F401
        except Exception as exc:
            raise CommandError(
                "CatBoost is not installed. Install with: pip install catboost"
            ) from exc
    if estimator_name == "lightgbm":
        try:
            import lightgbm  # noqa: F401
        except Exception as exc:
            raise CommandError(
                "LightGBM is not installed. Install with: pip install lightgbm"
            ) from exc


def _resolve_estimator_name(requested: str) -> str:
    requested = str(requested or "auto").strip().lower()
    if requested in {"hgb", "hist_gradient_boosting", "histgradientboosting"}:
        return "hgb"
    if requested in {"catboost", "lightgbm"}:
        _check_dependency_or_raise(requested)
        return requested
    if requested != "auto":
        raise CommandError("--estimator must be one of: auto, catboost, lightgbm, hgb")

    try:
        import catboost  # noqa: F401

        return "catboost"
    except Exception:
        pass
    try:
        import lightgbm  # noqa: F401

        return "lightgbm"
    except Exception:
        pass
    return "hgb"


def _param_grid(estimator_name: str) -> list[dict[str, Any]]:
    if estimator_name == "catboost":
        return [
            {"depth": 6, "learning_rate": 0.05, "iterations": 350},
            {"depth": 8, "learning_rate": 0.05, "iterations": 500},
            {"depth": 10, "learning_rate": 0.03, "iterations": 650},
        ]
    if estimator_name == "lightgbm":
        return [
            {"learning_rate": 0.05, "n_estimators": 300, "num_leaves": 31},
            {"learning_rate": 0.05, "n_estimators": 500, "num_leaves": 63},
            {"learning_rate": 0.03, "n_estimators": 700, "num_leaves": 127},
        ]
    return [
        {"learning_rate": 0.05, "max_depth": 6, "max_iter": 320},
        {"learning_rate": 0.05, "max_depth": 8, "max_iter": 450},
        {"learning_rate": 0.03, "max_depth": 10, "max_iter": 650},
    ]


def _build_estimator(
    *,
    estimator_name: str,
    n_classes: int,
    seed: int,
    params: dict[str, Any],
):
    if estimator_name == "catboost":
        from catboost import CatBoostClassifier

        base = {
            "loss_function": "MultiClass",
            "random_seed": seed,
            "verbose": False,
            "allow_writing_files": False,
            "thread_count": 4,
        }
        base.update(params)
        return CatBoostClassifier(**base)

    if estimator_name == "lightgbm":
        from lightgbm import LGBMClassifier

        base = {
            "objective": "multiclass",
            "num_class": int(n_classes),
            "random_state": seed,
        }
        base.update(params)
        return LGBMClassifier(**base)

    base = {
        "random_state": seed,
        "learning_rate": 0.05,
        "max_depth": 8,
        "max_iter": 450,
    }
    base.update(params)
    return HistGradientBoostingClassifier(**base)


def _build_model_pipeline(
    *,
    estimator_name: str,
    n_classes: int,
    seed: int,
    params: dict[str, Any],
    categorical_features: list[str],
    numeric_features: list[str],
):
    preprocess = ColumnTransformer(
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
    estimator = _build_estimator(
        estimator_name=estimator_name,
        n_classes=n_classes,
        seed=seed,
        params=params,
    )
    return Pipeline(
        steps=[
            ("preprocess", preprocess),
            ("classifier", estimator),
        ]
    )


def _topk_hits(y_true: "np.ndarray", proba: "np.ndarray", k: int) -> "np.ndarray":
    if len(y_true) == 0:
        return np.asarray([], dtype=bool)
    k = max(1, min(int(k), int(proba.shape[1])))
    topk_idx = np.argpartition(proba, -k, axis=1)[:, -k:]
    return np.any(topk_idx == y_true[:, None], axis=1)


def _apply_temperature(proba: "np.ndarray", temperature: float) -> "np.ndarray":
    t = max(0.05, float(temperature))
    clipped = np.clip(proba, 1e-12, 1.0)
    logits = np.log(clipped)
    scaled = logits / t
    shifted = scaled - scaled.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    denom = exp.sum(axis=1, keepdims=True)
    return exp / np.clip(denom, 1e-12, None)


def _fit_temperature(val_y: "np.ndarray", val_proba: "np.ndarray") -> tuple[float, float]:
    grid = [x / 100.0 for x in range(50, 301, 5)]  # 0.50..3.00
    best_t = 1.0
    best_nll = float("inf")
    for t in grid:
        p = _apply_temperature(val_proba, t)
        nll = float(log_loss(val_y, p, labels=list(range(p.shape[1]))))
        if nll < best_nll:
            best_nll = nll
            best_t = float(t)
    return best_t, float(best_nll)


def _ece_score(y_true: "np.ndarray", proba: "np.ndarray", *, n_bins: int = 10) -> float:
    if len(y_true) == 0:
        return 0.0
    preds = np.asarray(proba.argmax(axis=1)).astype(int)
    conf = np.asarray(proba.max(axis=1)).astype(float)
    correct = (preds == y_true).astype(float)
    ece = 0.0
    for i in range(n_bins):
        low = i / n_bins
        high = (i + 1) / n_bins
        if i == 0:
            mask = (conf >= low) & (conf <= high)
        else:
            mask = (conf > low) & (conf <= high)
        if not np.any(mask):
            continue
        conf_bin = float(conf[mask].mean())
        acc_bin = float(correct[mask].mean())
        ece += abs(acc_bin - conf_bin) * float(mask.mean())
    return float(ece)


def _brier_score_multiclass(y_true: "np.ndarray", proba: "np.ndarray") -> float:
    if len(y_true) == 0:
        return 0.0
    one_hot = np.zeros_like(proba)
    one_hot[np.arange(len(y_true)), y_true] = 1.0
    return float(np.mean(np.sum((proba - one_hot) ** 2, axis=1)))


def _coverage_curve(y_true: "np.ndarray", proba: "np.ndarray") -> list[dict[str, Any]]:
    if len(y_true) == 0:
        return []
    preds = np.asarray(proba.argmax(axis=1)).astype(int)
    max_prob = np.asarray(proba.max(axis=1)).astype(float)
    rows: list[dict[str, Any]] = []
    for threshold in [x / 10.0 for x in range(1, 10)]:
        mask = max_prob >= threshold
        covered = int(mask.sum())
        if covered <= 0:
            rows.append(
                {
                    "threshold": float(threshold),
                    "coverage": 0.0,
                    "top1_accuracy_on_covered": 0.0,
                    "covered_rows": 0,
                }
            )
            continue
        acc = float((preds[mask] == y_true[mask]).mean())
        rows.append(
            {
                "threshold": float(threshold),
                "coverage": round(float(covered / len(y_true)), 6),
                "top1_accuracy_on_covered": round(acc, 6),
                "covered_rows": covered,
            }
        )
    return rows


def _evaluate_predictions(
    *,
    y_true: "np.ndarray",
    proba: "np.ndarray",
    class_count: int,
    threshold: float,
    include_confusion_matrix: bool,
) -> dict[str, Any]:
    if len(y_true) == 0:
        return {
            "rows": 0,
            "top1_accuracy": 0.0,
            "top3_accuracy": 0.0,
            "top5_accuracy": 0.0,
            "recall_at_1": 0.0,
            "recall_at_3": 0.0,
            "recall_at_5": 0.0,
            "coverage": 0.0,
            "coverage_threshold": float(threshold),
            "logloss": 0.0,
            "ece": 0.0,
            "brier": 0.0,
            "coverage_curve": [],
            "confusion_matrix": [] if include_confusion_matrix else None,
        }

    top1_hits = _topk_hits(y_true, proba, 1)
    top3_hits = _topk_hits(y_true, proba, 3)
    top5_hits = _topk_hits(y_true, proba, 5)
    preds = np.asarray(proba.argmax(axis=1)).astype(int)
    max_prob = np.asarray(proba.max(axis=1)).astype(float)
    coverage = float((max_prob >= threshold).mean())
    nll = float(log_loss(y_true, proba, labels=list(range(class_count))))
    ece = _ece_score(y_true, proba)
    brier = _brier_score_multiclass(y_true, proba)

    out = {
        "rows": int(len(y_true)),
        "top1_accuracy": round(float(top1_hits.mean()), 6),
        "top3_accuracy": round(float(top3_hits.mean()), 6),
        "top5_accuracy": round(float(top5_hits.mean()), 6),
        "recall_at_1": round(float(top1_hits.mean()), 6),
        "recall_at_3": round(float(top3_hits.mean()), 6),
        "recall_at_5": round(float(top5_hits.mean()), 6),
        "coverage": round(float(coverage), 6),
        "coverage_threshold": float(threshold),
        "logloss": round(nll, 6),
        "ece": round(float(ece), 6),
        "brier": round(float(brier), 6),
        "coverage_curve": _coverage_curve(y_true, proba),
        "confusion_matrix": None,
    }
    if include_confusion_matrix:
        labels_order = list(range(class_count))
        cm = confusion_matrix(y_true, preds, labels=labels_order)
        out["confusion_matrix"] = cm.tolist()
    return out


def _top_confusions(
    *,
    confusion: list[list[int]],
    class_labels: list[str],
    limit: int = 25,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if not confusion or not class_labels:
        return out
    for i, true_label in enumerate(class_labels):
        for j, pred_label in enumerate(class_labels):
            if i == j:
                continue
            count = int(confusion[i][j])
            if count <= 0:
                continue
            out.append({"count": count, "true": true_label, "pred": pred_label})
    out.sort(key=lambda x: (-int(x["count"]), str(x["true"]), str(x["pred"])))
    return out[:limit]


def _completion_binary_metrics(
    *,
    y_true: "np.ndarray",
    y_pred: "np.ndarray",
    class_labels: list[str],
) -> dict[str, Any] | None:
    if "__none__" not in class_labels:
        return None
    none_idx = int(class_labels.index("__none__"))
    true_complete = y_true != none_idx
    pred_complete = y_pred != none_idx

    tp = int(np.logical_and(true_complete, pred_complete).sum())
    fp = int(np.logical_and(~true_complete, pred_complete).sum())
    fn = int(np.logical_and(true_complete, ~pred_complete).sum())
    tn = int(np.logical_and(~true_complete, ~pred_complete).sum())

    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    f1 = float((2 * precision * recall) / (precision + recall)) if (precision + recall) > 0 else 0.0
    return {
        "rows": int(len(y_true)),
        "positive_class": "completion",
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1": round(f1, 6),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def _build_eval_markdown(report: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# Roadmap NextStep Evaluation")
    lines.append("")
    lines.append("## Summary")
    lines.append(f"- trained_at_utc: `{report['trained_at_utc']}`")
    lines.append(f"- estimator: `{report['estimator']}`")
    lines.append(f"- model_version: `{report['model_version']}`")
    lines.append(f"- temperature: `{report.get('temperature', 1.0):.4f}`")
    lines.append(f"- classes_total: **{report['classes_total']}**")
    lines.append(f"- train_rows: **{report['train_rows']}**")
    lines.append(f"- val_rows: **{report['val_rows']}**")
    lines.append(f"- test_rows: **{report['test_rows']}**")
    lines.append("")

    lines.append("## Metrics")
    lines.append("| split | top1 | top3 | top5 | logloss | ece | brier | coverage |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for split_key in ["val", "test"]:
        m = report[f"metrics_{split_key}"]
        lines.append(
            "| "
            f"{split_key} | {m['top1_accuracy']:.4f} | {m['top3_accuracy']:.4f} | {m['top5_accuracy']:.4f} | "
            f"{m['logloss']:.4f} | {m['ece']:.4f} | {m['brier']:.4f} | {m['coverage']:.4f} |"
        )
    lines.append("")

    per_cat = report.get("per_category_test") or {}
    lines.append("## Per-Category (test)")
    lines.append("| category | rows | top1 | top3 | top5 | coverage |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    if per_cat:
        for category in sorted(per_cat.keys()):
            row = per_cat[category]
            lines.append(
                f"| {category} | {int(row['rows'])} | {row['top1_accuracy']:.4f} | "
                f"{row['top3_accuracy']:.4f} | {row['top5_accuracy']:.4f} | {row['coverage']:.4f} |"
            )
    else:
        lines.append("| - | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |")
    lines.append("")

    lines.append("## Top Confusions (test)")
    lines.append("| count | true | pred |")
    lines.append("| --- | --- | --- |")
    top_confusions = report.get("top_confusions_test") or []
    if top_confusions:
        for row in top_confusions[:30]:
            lines.append(f"| {int(row['count'])} | {row['true']} | {row['pred']} |")
    else:
        lines.append("| 0 | - | - |")
    lines.append("")

    none_metrics = report.get("none_class_binary")
    lines.append("## __none__ Performance")
    if none_metrics:
        lines.append(
            f"- precision(completion): **{none_metrics['precision']:.4f}**; "
            f"recall(completion): **{none_metrics['recall']:.4f}**; "
            f"f1: **{none_metrics['f1']:.4f}**"
        )
        lines.append(
            f"- confusion(completion): TP={none_metrics['tp']} FP={none_metrics['fp']} "
            f"FN={none_metrics['fn']} TN={none_metrics['tn']}"
        )
    else:
        lines.append("- class `__none__` is absent in this dataset.")
    lines.append("")

    if report.get("dataset_baselines"):
        lines.append("## Dataset Baselines")
        lines.append("Popularity and Markov baseline comparison is available in dataset metadata.")
        lines.append("")

    return "\n".join(lines)


class Command(BaseCommand):
    help = "Train Roadmap NextStep v3 multiclass model with tuning/calibration and reliability metrics."

    def add_arguments(self, parser):
        parser.add_argument("--data-dir", type=str, default="data/ml/roadmap_nextstep")
        parser.add_argument("--model-dir", type=str, default="models/roadmap_next_step_v2")
        parser.add_argument("--threshold", type=float, default=0.35)
        parser.add_argument("--seed", type=int, default=42)
        parser.add_argument("--model-version", type=str, default="roadmap_next_step_v3")
        parser.add_argument(
            "--estimator",
            type=str,
            default="auto",
            help="auto|catboost|lightgbm|hgb",
        )
        parser.add_argument(
            "--trials",
            type=int,
            default=9,
            help="Max parameter trials for grid search (0 disables tuning).",
        )

    def handle(self, *args, **options):
        if pd is None or np is None:
            raise CommandError("pandas + numpy are required. Install requirements-ml.txt")
        if joblib is None:
            raise CommandError("joblib is required. Install requirements-ml.txt")

        data_dir = _resolve_existing_dir(str(options["data_dir"]))
        model_dir = _resolve_output_dir(str(options["model_dir"]))
        threshold = float(options["threshold"])
        seed = int(options["seed"])
        model_version = str(options["model_version"]).strip() or "roadmap_next_step_v3"
        estimator_name = _resolve_estimator_name(str(options.get("estimator") or "auto"))
        max_trials = int(options.get("trials") or 0)

        if threshold <= 0 or threshold >= 1:
            raise CommandError("--threshold must be in (0, 1)")
        if max_trials < 0:
            raise CommandError("--trials must be >= 0")

        splits_path = data_dir / "splits.json"
        metadata_path = data_dir / "metadata.json"
        if not splits_path.exists():
            raise CommandError(f"Missing splits.json in {data_dir}")
        if not metadata_path.exists():
            raise CommandError(f"Missing metadata.json in {data_dir}")

        dataset_df, dataset_path = _load_dataset(data_dir)
        split_payload = json.loads(splits_path.read_text(encoding="utf-8"))
        dataset_meta = json.loads(metadata_path.read_text(encoding="utf-8"))

        required_cols = {"user_id", "target_class"}
        missing = sorted(required_cols.difference(set(dataset_df.columns)))
        if missing:
            raise CommandError(f"Dataset missing required columns: {missing}")
        work = dataset_df.copy()
        work["user_id"] = work["user_id"].astype(int)
        work["target_class"] = work["target_class"].fillna("").astype(str).str.strip()
        work = work[work["target_class"] != ""].copy()

        if work.empty:
            raise CommandError("No multiclass rows found in dataset")

        train_users = _prepare_split_set(split_payload, "train_user_ids")
        val_users = _prepare_split_set(split_payload, "val_user_ids")
        test_users = _prepare_split_set(split_payload, "test_user_ids")
        if not train_users or not val_users or not test_users:
            raise CommandError("splits.json has empty train/val/test user lists")

        feature_columns = list(dataset_meta.get("feature_columns") or [])
        if not feature_columns:
            ignore_cols = {
                "user_id",
                "step_id",
                "first_exposed_at",
                "step_product_type",
                "label",
                "target_class",
                "latency_to_click_hours",
                "latency_to_complete_hours",
            }
            feature_columns = [c for c in work.columns if c not in ignore_cols]

        for col in feature_columns:
            if col not in work.columns:
                raise CommandError(f"Feature column missing in dataset: {col}")

        categorical_features = [
            c for c in list(dataset_meta.get("categorical_features") or []) if c in feature_columns
        ]
        numeric_features = [
            c for c in list(dataset_meta.get("numeric_features") or []) if c in feature_columns
        ]
        if not categorical_features and not numeric_features:
            guessed_cat = [c for c in feature_columns if str(work[c].dtype) == "object"]
            guessed_num = [c for c in feature_columns if c not in guessed_cat]
            categorical_features = guessed_cat
            numeric_features = guessed_num

        train_df = work[work["user_id"].isin(train_users)].copy()
        val_df = work[work["user_id"].isin(val_users)].copy()
        test_df = work[work["user_id"].isin(test_users)].copy()

        if train_df.empty or val_df.empty or test_df.empty:
            raise CommandError(
                "User-level split produced empty split: "
                f"train={len(train_df)} val={len(val_df)} test={len(test_df)}"
            )

        train_classes = sorted(set(train_df["target_class"].tolist()))
        if len(train_classes) < 2:
            raise CommandError(
                "Training data must contain at least 2 classes for multiclass model"
            )

        train_class_set = set(train_classes)
        dropped_val = int((~val_df["target_class"].isin(train_class_set)).sum())
        dropped_test = int((~test_df["target_class"].isin(train_class_set)).sum())
        val_df = val_df[val_df["target_class"].isin(train_class_set)].copy()
        test_df = test_df[test_df["target_class"].isin(train_class_set)].copy()

        if val_df.empty or test_df.empty:
            raise CommandError(
                "Validation/test became empty after removing unseen classes. "
                "Increase window or rebuild splits."
            )

        class_to_idx = {cls: idx for idx, cls in enumerate(train_classes)}
        X_train = train_df[feature_columns].copy()
        X_val = val_df[feature_columns].copy()
        X_test = test_df[feature_columns].copy()
        for col in categorical_features:
            X_train[col] = X_train[col].fillna("__none__").astype(str)
            X_val[col] = X_val[col].fillna("__none__").astype(str)
            X_test[col] = X_test[col].fillna("__none__").astype(str)
        for col in numeric_features:
            X_train[col] = pd.to_numeric(X_train[col], errors="coerce")
            X_val[col] = pd.to_numeric(X_val[col], errors="coerce")
            X_test[col] = pd.to_numeric(X_test[col], errors="coerce")

        y_train = train_df["target_class"].map(class_to_idx).astype(int).to_numpy()
        y_val = val_df["target_class"].map(class_to_idx).astype(int).to_numpy()
        y_test = test_df["target_class"].map(class_to_idx).astype(int).to_numpy()

        candidate_grid = _param_grid(estimator_name)
        if max_trials > 0:
            candidate_grid = candidate_grid[: max(1, min(len(candidate_grid), max_trials))]
        else:
            candidate_grid = [candidate_grid[0]]

        best_model = None
        best_params: dict[str, Any] = {}
        best_tuple: tuple[float, float, float] | None = None
        best_raw_val_proba = None
        tuning_rows: list[dict[str, Any]] = []

        for idx, params in enumerate(candidate_grid, start=1):
            model = _build_model_pipeline(
                estimator_name=estimator_name,
                n_classes=len(train_classes),
                seed=seed,
                params=params,
                categorical_features=categorical_features,
                numeric_features=numeric_features,
            )
            self.stdout.write(
                f"[train_roadmap_nextstep_model] trial={idx}/{len(candidate_grid)} "
                f"estimator={estimator_name} params={params}"
            )
            model.fit(X_train, y_train)
            val_proba_raw = np.asarray(model.predict_proba(X_val))
            val_metrics_raw = _evaluate_predictions(
                y_true=y_val,
                proba=val_proba_raw,
                class_count=len(train_classes),
                threshold=threshold,
                include_confusion_matrix=False,
            )
            score_tuple = (
                float(val_metrics_raw["top1_accuracy"]),
                float(val_metrics_raw["top3_accuracy"]),
                -float(val_metrics_raw["logloss"]),
            )
            tuning_rows.append(
                {
                    "trial": idx,
                    "params": params,
                    "score": score_tuple,
                    "val_top1": val_metrics_raw["top1_accuracy"],
                    "val_top3": val_metrics_raw["top3_accuracy"],
                    "val_logloss": val_metrics_raw["logloss"],
                }
            )
            if best_tuple is None or score_tuple > best_tuple:
                best_tuple = score_tuple
                best_params = dict(params)
                best_model = model
                best_raw_val_proba = val_proba_raw

        if best_model is None or best_raw_val_proba is None:
            raise CommandError("Failed to fit any model candidate")

        temperature, calibrated_val_logloss = _fit_temperature(y_val, best_raw_val_proba)
        val_proba = _apply_temperature(best_raw_val_proba, temperature)
        test_proba_raw = np.asarray(best_model.predict_proba(X_test))
        test_proba = _apply_temperature(test_proba_raw, temperature)

        metrics_val = _evaluate_predictions(
            y_true=y_val,
            proba=val_proba,
            class_count=len(train_classes),
            threshold=threshold,
            include_confusion_matrix=True,
        )
        metrics_test = _evaluate_predictions(
            y_true=y_test,
            proba=test_proba,
            class_count=len(train_classes),
            threshold=threshold,
            include_confusion_matrix=True,
        )
        metrics_val["logloss_uncalibrated"] = round(
            float(log_loss(y_val, best_raw_val_proba, labels=list(range(len(train_classes))))), 6
        )
        metrics_val["logloss_calibrated"] = round(float(calibrated_val_logloss), 6)

        preds_test = np.asarray(test_proba.argmax(axis=1)).astype(int)
        none_binary = _completion_binary_metrics(
            y_true=y_test,
            y_pred=preds_test,
            class_labels=train_classes,
        )

        per_category_test: dict[str, Any] = {}
        if "category" in test_df.columns:
            for category in sorted(set(str(x) for x in test_df["category"].tolist())):
                mask = (test_df["category"].astype(str) == str(category)).to_numpy()
                if not np.any(mask):
                    continue
                part_metrics = _evaluate_predictions(
                    y_true=y_test[mask],
                    proba=test_proba[mask],
                    class_count=len(train_classes),
                    threshold=threshold,
                    include_confusion_matrix=False,
                )
                per_category_test[str(category)] = {
                    "rows": int(part_metrics["rows"]),
                    "top1_accuracy": float(part_metrics["top1_accuracy"]),
                    "top3_accuracy": float(part_metrics["top3_accuracy"]),
                    "top5_accuracy": float(part_metrics["top5_accuracy"]),
                    "coverage": float(part_metrics["coverage"]),
                }

        trained_at = datetime.now(tz=timezone.utc).isoformat().replace("+00:00", "Z")

        artifact = {
            "model": best_model,
            "class_labels": train_classes,
            "feature_columns": feature_columns,
            "categorical_features": categorical_features,
            "numeric_features": numeric_features,
            "trained_at_utc": trained_at,
            "model_version": model_version,
            "threshold": threshold,
            "temperature": float(temperature),
        }

        model_dir.mkdir(parents=True, exist_ok=True)
        model_path = model_dir / "model.pkl"
        joblib.dump(artifact, model_path)

        model_metadata = {
            "trained_at_utc": trained_at,
            "model_version": model_version,
            "estimator": estimator_name,
            "dataset_path": dataset_path,
            "dataset_rows_multiclass": int(len(work)),
            "feature_columns": feature_columns,
            "categorical_features": categorical_features,
            "numeric_features": numeric_features,
            "class_labels": train_classes,
            "label_map": {str(i): cls for i, cls in enumerate(train_classes)},
            "threshold": threshold,
            "temperature": float(round(temperature, 6)),
            "train_rows": int(len(train_df)),
            "val_rows": int(len(val_df)),
            "test_rows": int(len(test_df)),
            "dropped_unseen_val_rows": int(dropped_val),
            "dropped_unseen_test_rows": int(dropped_test),
            "selected_params": best_params,
            "tuning_rows": tuning_rows,
            "top_product_types_by_category": dataset_meta.get("top_product_types_by_category") or {},
        }
        model_metadata_path = model_dir / "metadata.json"
        model_metadata_path.write_text(
            json.dumps(model_metadata, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        report = {
            "trained_at_utc": trained_at,
            "model_version": model_version,
            "estimator": estimator_name,
            "selected_params": best_params,
            "temperature": float(round(temperature, 6)),
            "threshold": threshold,
            "dataset_path": dataset_path,
            "classes_total": int(len(train_classes)),
            "class_labels": train_classes,
            "train_rows": int(len(train_df)),
            "val_rows": int(len(val_df)),
            "test_rows": int(len(test_df)),
            "dropped_unseen_val_rows": int(dropped_val),
            "dropped_unseen_test_rows": int(dropped_test),
            "metrics_val": metrics_val,
            "metrics_test": metrics_test,
            "per_category_test": per_category_test,
            "top_confusions_test": _top_confusions(
                confusion=metrics_test.get("confusion_matrix") or [],
                class_labels=train_classes,
                limit=30,
            ),
            "none_class_binary": none_binary,
            "dataset_baselines": dataset_meta.get("baselines") or {},
            "dataset_class_distribution": dataset_meta.get("class_distribution") or {},
        }

        reports_dir = (_repo_root() / "reports").resolve()
        reports_dir.mkdir(parents=True, exist_ok=True)
        report_json_path = reports_dir / "roadmap_nextstep_eval.json"
        report_md_path = reports_dir / "roadmap_nextstep_eval.md"

        report_json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        report_md_path.write_text(_build_eval_markdown(report), encoding="utf-8")

        self.stdout.write("[train_roadmap_nextstep_model] done")
        self.stdout.write(f"[train_roadmap_nextstep_model] estimator={estimator_name}")
        self.stdout.write(f"[train_roadmap_nextstep_model] model={model_path}")
        self.stdout.write(f"[train_roadmap_nextstep_model] metadata={model_metadata_path}")
        self.stdout.write(f"[train_roadmap_nextstep_model] report_json={report_json_path}")
        self.stdout.write(f"[train_roadmap_nextstep_model] report_md={report_md_path}")
        self.stdout.write(
            "[train_roadmap_nextstep_model] "
            f"test_top1={metrics_test['top1_accuracy']:.4f} "
            f"test_top3={metrics_test['top3_accuracy']:.4f} "
            f"test_top5={metrics_test['top5_accuracy']:.4f} "
            f"coverage={metrics_test['coverage']:.4f} "
            f"ece={metrics_test['ece']:.4f} brier={metrics_test['brier']:.4f}"
        )
