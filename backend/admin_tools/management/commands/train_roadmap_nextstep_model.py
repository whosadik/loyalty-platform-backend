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
from sklearn.metrics import confusion_matrix
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


def _select_estimator(n_classes: int, seed: int):
    try:
        from catboost import CatBoostClassifier

        return (
            "catboost",
            CatBoostClassifier(
                loss_function="MultiClass",
                depth=8,
                learning_rate=0.05,
                iterations=500,
                random_seed=seed,
                verbose=False,
            ),
        )
    except Exception:
        pass

    try:
        from lightgbm import LGBMClassifier

        return (
            "lightgbm",
            LGBMClassifier(
                objective="multiclass",
                num_class=n_classes,
                learning_rate=0.05,
                n_estimators=500,
                max_depth=-1,
                random_state=seed,
            ),
        )
    except Exception:
        pass

    return (
        "hist_gradient_boosting",
        HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_depth=8,
            max_iter=450,
            random_state=seed,
        ),
    )


def _topk_rate(y_true: "np.ndarray", proba: "np.ndarray", k: int) -> float:
    if len(y_true) == 0:
        return 0.0
    k = max(1, min(int(k), int(proba.shape[1])))
    topk_idx = np.argpartition(proba, -k, axis=1)[:, -k:]
    hits = 0
    for i in range(len(y_true)):
        if int(y_true[i]) in set(int(x) for x in topk_idx[i].tolist()):
            hits += 1
    return float(hits / len(y_true))


def _evaluate_split(
    *,
    model,
    split_df: "pd.DataFrame",
    feature_columns: list[str],
    class_to_idx: dict[str, int],
    threshold: float,
) -> dict[str, Any]:
    if split_df.empty:
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
            "confusion_matrix": [],
        }

    y_true = split_df["target_class"].map(class_to_idx).astype(int).to_numpy()
    X = split_df[feature_columns].copy()
    proba = model.predict_proba(X)
    proba = np.asarray(proba)

    top1 = _topk_rate(y_true, proba, 1)
    top3 = _topk_rate(y_true, proba, 3)
    top5 = _topk_rate(y_true, proba, 5)
    preds = np.asarray(proba.argmax(axis=1)).astype(int)
    max_prob = np.asarray(proba.max(axis=1))
    coverage = float((max_prob >= threshold).mean()) if len(max_prob) else 0.0

    labels_order = list(range(len(class_to_idx)))
    cm = confusion_matrix(y_true, preds, labels=labels_order)

    return {
        "rows": int(len(split_df)),
        "top1_accuracy": round(float(top1), 6),
        "top3_accuracy": round(float(top3), 6),
        "top5_accuracy": round(float(top5), 6),
        "recall_at_1": round(float(top1), 6),
        "recall_at_3": round(float(top3), 6),
        "recall_at_5": round(float(top5), 6),
        "coverage": round(float(coverage), 6),
        "coverage_threshold": float(threshold),
        "confusion_matrix": cm.tolist(),
    }


def _build_eval_markdown(report: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# Roadmap NextStep Evaluation")
    lines.append("")
    lines.append("## Summary")
    lines.append(f"- trained_at_utc: `{report['trained_at_utc']}`")
    lines.append(f"- estimator: `{report['estimator']}`")
    lines.append(f"- model_version: `{report['model_version']}`")
    lines.append(f"- classes_total: **{report['classes_total']}**")
    lines.append(f"- train_rows: **{report['train_rows']}**")
    lines.append(f"- val_rows: **{report['val_rows']}**")
    lines.append(f"- test_rows: **{report['test_rows']}**")
    lines.append("")

    lines.append("## Metrics")
    lines.append("| split | top1 | top3 | top5 | recall@1 | recall@3 | recall@5 | coverage |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for split_key in ["val", "test"]:
        m = report[f"metrics_{split_key}"]
        lines.append(
            "| "
            f"{split_key} | {m['top1_accuracy']:.4f} | {m['top3_accuracy']:.4f} | {m['top5_accuracy']:.4f} | "
            f"{m['recall_at_1']:.4f} | {m['recall_at_3']:.4f} | {m['recall_at_5']:.4f} | {m['coverage']:.4f} |"
        )
    lines.append("")

    labels = report.get("class_labels", [])
    cm = report.get("metrics_test", {}).get("confusion_matrix", [])
    pairs: list[tuple[int, str, str]] = []
    if cm and labels:
        for i, true_label in enumerate(labels):
            for j, pred_label in enumerate(labels):
                count = int(cm[i][j])
                if count > 0:
                    pairs.append((count, str(true_label), str(pred_label)))
        pairs.sort(key=lambda x: (-x[0], x[1], x[2]))

    lines.append("## Confusion Top Pairs (test)")
    lines.append("| count | true | pred |")
    lines.append("| --- | --- | --- |")
    for count, true_label, pred_label in pairs[:30]:
        lines.append(f"| {count} | {true_label} | {pred_label} |")

    return "\n".join(lines)


class Command(BaseCommand):
    help = "Train Roadmap NextStep v2 baseline multiclass model on offline dataset."

    def add_arguments(self, parser):
        parser.add_argument("--data-dir", type=str, default="data/ml/roadmap_nextstep")
        parser.add_argument("--model-dir", type=str, default="models/roadmap_next_step_v2")
        parser.add_argument("--threshold", type=float, default=0.35)
        parser.add_argument("--seed", type=int, default=42)
        parser.add_argument("--model-version", type=str, default="roadmap_next_step_v2")

    def handle(self, *args, **options):
        if pd is None or np is None:
            raise CommandError("pandas + numpy are required. Install requirements-ml.txt")
        if joblib is None:
            raise CommandError("joblib is required. Install requirements-ml.txt")

        data_dir = _resolve_existing_dir(str(options["data_dir"]))
        model_dir = _resolve_output_dir(str(options["model_dir"]))
        threshold = float(options["threshold"])
        seed = int(options["seed"])
        model_version = str(options["model_version"]).strip() or "roadmap_next_step_v2"

        if threshold <= 0 or threshold >= 1:
            raise CommandError("--threshold must be in (0, 1)")

        splits_path = data_dir / "splits.json"
        metadata_path = data_dir / "metadata.json"
        if not splits_path.exists():
            raise CommandError(f"Missing splits.json in {data_dir}")
        if not metadata_path.exists():
            raise CommandError(f"Missing metadata.json in {data_dir}")

        dataset_df, dataset_path = _load_dataset(data_dir)
        split_payload = json.loads(splits_path.read_text(encoding="utf-8"))
        dataset_meta = json.loads(metadata_path.read_text(encoding="utf-8"))

        required_cols = {"user_id", "label", "target_class"}
        missing = sorted(required_cols.difference(set(dataset_df.columns)))
        if missing:
            raise CommandError(f"Dataset missing required columns: {missing}")
        work = dataset_df.copy()
        work["user_id"] = work["user_id"].astype(int)
        work["label"] = work["label"].astype(int)
        work["target_class"] = work["target_class"].fillna("").astype(str)
        work = work[(work["label"] == 1) & (work["target_class"].str.strip() != "")].copy()

        if work.empty:
            raise CommandError("No positive multiclass rows found in dataset")

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
                "User-level split produced empty split after positive filtering: "
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
        y_train = train_df["target_class"].map(class_to_idx).astype(int).to_numpy()

        X_train = train_df[feature_columns].copy()
        for col in categorical_features:
            X_train[col] = X_train[col].fillna("__none__").astype(str)
        for col in numeric_features:
            X_train[col] = pd.to_numeric(X_train[col], errors="coerce")

        estimator_name, estimator = _select_estimator(len(train_classes), seed)

        preprocess = ColumnTransformer(
            transformers=[
                (
                    "categorical",
                    Pipeline(
                        steps=[
                            ("imputer", SimpleImputer(strategy="constant", fill_value="__none__")),
                            (
                                "onehot",
                                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                            ),
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

        model = Pipeline(
            steps=[
                ("preprocess", preprocess),
                ("classifier", estimator),
            ]
        )

        self.stdout.write(
            f"[train_roadmap_nextstep_model] fitting estimator={estimator_name} rows={len(train_df)} classes={len(train_classes)}"
        )
        model.fit(X_train, y_train)

        metrics_val = _evaluate_split(
            model=model,
            split_df=val_df,
            feature_columns=feature_columns,
            class_to_idx=class_to_idx,
            threshold=threshold,
        )
        metrics_test = _evaluate_split(
            model=model,
            split_df=test_df,
            feature_columns=feature_columns,
            class_to_idx=class_to_idx,
            threshold=threshold,
        )

        trained_at = datetime.now(tz=timezone.utc).isoformat().replace("+00:00", "Z")

        artifact = {
            "model": model,
            "class_labels": train_classes,
            "feature_columns": feature_columns,
            "categorical_features": categorical_features,
            "numeric_features": numeric_features,
            "trained_at_utc": trained_at,
            "model_version": model_version,
            "threshold": threshold,
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
            "train_rows": int(len(train_df)),
            "val_rows": int(len(val_df)),
            "test_rows": int(len(test_df)),
            "dropped_unseen_val_rows": int(dropped_val),
            "dropped_unseen_test_rows": int(dropped_test),
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
        }

        reports_dir = (_repo_root() / "reports").resolve()
        reports_dir.mkdir(parents=True, exist_ok=True)
        report_json_path = reports_dir / "roadmap_nextstep_eval.json"
        report_md_path = reports_dir / "roadmap_nextstep_eval.md"

        report_json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        report_md_path.write_text(_build_eval_markdown(report), encoding="utf-8")

        self.stdout.write("[train_roadmap_nextstep_model] done")
        self.stdout.write(f"[train_roadmap_nextstep_model] model={model_path}")
        self.stdout.write(f"[train_roadmap_nextstep_model] metadata={model_metadata_path}")
        self.stdout.write(f"[train_roadmap_nextstep_model] report_json={report_json_path}")
        self.stdout.write(f"[train_roadmap_nextstep_model] report_md={report_md_path}")
        self.stdout.write(
            "[train_roadmap_nextstep_model] "
            f"test_top1={metrics_test['top1_accuracy']:.4f} "
            f"test_top3={metrics_test['top3_accuracy']:.4f} "
            f"test_top5={metrics_test['top5_accuracy']:.4f} "
            f"coverage={metrics_test['coverage']:.4f}"
        )
