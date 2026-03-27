from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
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

try:
    from .roadmap_initial_planner_common import (
        ACTION_SPACE_BY_CATEGORY,
        NONE_TOKEN,
        STOP_TOKEN,
        build_model_pipeline,
        ensure_dependencies,
        predict_action_probabilities,
        resolve_estimator_name,
        resolve_path,
    )
except ImportError:  # pragma: no cover
    from roadmap_initial_planner_common import (
        ACTION_SPACE_BY_CATEGORY,
        NONE_TOKEN,
        STOP_TOKEN,
        build_model_pipeline,
        ensure_dependencies,
        predict_action_probabilities,
        resolve_estimator_name,
        resolve_path,
    )

ALLOWED_CATEGORIES = ("haircare", "skincare", "fragrance")
DEFAULT_SPLIT_SCHEMES = ("time", "user")
IDENTIFIER_COLUMNS = {
    "episode_id",
    "decision_id",
    "plan_id",
    "user_id",
    "t0_utc",
    "split",
    "eval_split",
    "runtime_plan_tokens_json",
}
TARGET_COLUMNS = {
    "label",
    "y",
    "candidate_type",
    "matched_by",
    "label_source",
    "trust_level",
    "excluded_reason",
}
RUNTIME_SHORTCUT_COLUMNS = {
    "current_next_product_type",
    "current_next_step_id",
    "plan_product_types",
    "current_ml_decision",
    "current_rollout_mode",
}
RUNTIME_ONLY_CONTEXT_COLUMNS = {
    "label_source",
    "matched_by",
    "excluded_reason",
}
EXCLUDE_PREFIXES = ("candidate_",)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_live_dataset_bundle(data_dir: str | Path) -> tuple["pd.DataFrame", dict[str, Any], dict[str, Any]]:
    ensure_dependencies()
    root = resolve_path(str(data_dir))
    parquet_path = root / "dataset.parquet"
    csv_path = root / "dataset.csv"
    if parquet_path.exists():
        df = pd.read_parquet(parquet_path)
    elif csv_path.exists():
        df = pd.read_csv(csv_path)
    else:  # pragma: no cover
        raise FileNotFoundError(f"dataset.parquet/dataset.csv not found in {root}")
    metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
    splits = json.loads((root / "splits.json").read_text(encoding="utf-8"))
    if "t0_utc" in df.columns:
        df["t0_utc"] = pd.to_datetime(df["t0_utc"], utc=True, format="mixed")
    return df, metadata, splits


def selected_categories(raw: str, *, allowed: tuple[str, ...] = ALLOWED_CATEGORIES) -> list[str]:
    if not str(raw or "").strip():
        return list(allowed)
    out: list[str] = []
    for item in str(raw).split(","):
        token = str(item or "").strip().lower()
        if token and token in allowed and token not in out:
            out.append(token)
    return out


def selected_split_schemes(raw: str) -> list[str]:
    if not str(raw or "").strip():
        return list(DEFAULT_SPLIT_SCHEMES)
    out: list[str] = []
    for item in str(raw).split(","):
        token = str(item or "").strip().lower()
        if token in DEFAULT_SPLIT_SCHEMES and token not in out:
            out.append(token)
    return out


def _stable_user_split(user_id: int, *, seed: int) -> str:
    digest = hashlib.md5(f"{seed}:{int(user_id)}".encode("utf-8")).hexdigest()
    bucket = int(digest[:8], 16) % 100
    if bucket < 70:
        return "train"
    if bucket < 85:
        return "val"
    return "test"


def apply_split_scheme(decisions_df: "pd.DataFrame", *, split_scheme: str, seed: int) -> "pd.DataFrame":
    df = decisions_df.copy()
    scheme = str(split_scheme or "time").strip().lower()
    if scheme == "user":
        df["eval_split"] = df["user_id"].map(lambda value: _stable_user_split(int(value), seed=seed))
        return df
    df["eval_split"] = df["split"].astype(str).str.lower()
    return df


def parse_runtime_plan_tokens(value: Any) -> list[str]:
    text = str(value or "").strip()
    if not text or text == NONE_TOKEN:
        return []
    out: list[str] = []
    seen: set[str] = set()
    for raw in text.split("|"):
        token = str(raw or "").strip().lower()
        if not token or token in seen:
            continue
        seen.add(token)
        out.append(token)
    return out


def build_live_decision_dataframe(
    *,
    dataset_df: "pd.DataFrame",
    category: str,
    split_scheme: str,
    seed: int,
    continuation_only: bool = False,
) -> "pd.DataFrame":
    ensure_dependencies()
    category = str(category or "").strip().lower()
    df = dataset_df[
        (dataset_df["category"].astype(str).str.lower() == category)
        & (pd.to_numeric(dataset_df["y"], errors="coerce").fillna(0).astype(int) == 1)
    ].copy()
    if continuation_only and "decision_type" in df.columns:
        df = df[df["decision_type"].astype(str).str.lower() != "initial_refresh"].copy()
    if df.empty:
        return df
    df["label"] = df["label"].astype(str).str.strip().str.lower()
    df["runtime_plan_tokens_json"] = df["plan_product_types"].map(parse_runtime_plan_tokens).map(
        lambda items: json.dumps(items, ensure_ascii=False)
    )
    df = apply_split_scheme(df, split_scheme=split_scheme, seed=seed)
    sort_keys = [name for name in ("t0_utc", "decision_id") if name in df.columns]
    if sort_keys:
        df = df.sort_values(sort_keys).reset_index(drop=True)
    return df


def feature_spec_for_live_dataframe(decisions_df: "pd.DataFrame") -> tuple[list[str], list[str], list[str]]:
    numeric_features: list[str] = []
    categorical_features: list[str] = []
    for column in decisions_df.columns:
        if column in IDENTIFIER_COLUMNS or column in TARGET_COLUMNS or column in RUNTIME_SHORTCUT_COLUMNS:
            continue
        if column in RUNTIME_ONLY_CONTEXT_COLUMNS:
            continue
        if column.startswith(EXCLUDE_PREFIXES):
            continue
        if column == "decision_type":
            categorical_features.append(column)
            continue
        series = decisions_df[column]
        if pd.api.types.is_bool_dtype(series.dtype):
            numeric_features.append(column)
        elif pd.api.types.is_numeric_dtype(series.dtype):
            numeric_features.append(column)
        else:
            categorical_features.append(column)
    feature_columns = categorical_features + numeric_features
    return feature_columns, categorical_features, numeric_features


def split_frames(decisions_df: "pd.DataFrame") -> dict[str, "pd.DataFrame"]:
    return {
        "train": decisions_df[decisions_df["eval_split"].astype(str) == "train"].copy(),
        "val": decisions_df[decisions_df["eval_split"].astype(str) == "val"].copy(),
        "test": decisions_df[decisions_df["eval_split"].astype(str) == "test"].copy(),
    }


class ConstantActionModel:
    def __init__(self, action: str):
        self.action = str(action or STOP_TOKEN).strip().lower() or STOP_TOKEN
        self.classes_ = np.asarray([self.action], dtype=object) if np is not None else [self.action]

    def predict(self, X):  # pragma: no cover
        if np is None:
            raise RuntimeError("numpy is required")
        return np.asarray([self.action] * len(X), dtype=object)

    def predict_proba(self, X):
        if np is None:  # pragma: no cover
            raise RuntimeError("numpy is required")
        return np.ones((len(X), 1), dtype=float)


def train_live_category_bundle(
    *,
    decisions_df: "pd.DataFrame",
    category: str,
    estimator_name: str,
    seed: int,
) -> dict[str, Any]:
    ensure_dependencies()
    feature_columns, categorical_features, numeric_features = feature_spec_for_live_dataframe(decisions_df)
    split_map = split_frames(decisions_df)
    train_df = split_map["train"].sort_values(["t0_utc", "decision_id"]).reset_index(drop=True)
    if train_df.empty:
        raise RuntimeError(f"No train rows for category={category}")
    labels = train_df["label"].astype(str).str.lower()
    unique_labels = [str(item) for item in sorted(labels.unique())]
    action_space = list(ACTION_SPACE_BY_CATEGORY.get(category) or [STOP_TOKEN])
    observed_space = [token for token in action_space if token in set(unique_labels)]
    if not observed_space:
        observed_space = unique_labels
    if len(observed_space) <= 1:
        model = ConstantActionModel(observed_space[0] if observed_space else STOP_TOKEN)
        resolved_estimator = "constant"
        model_type = "constant"
    else:
        resolved_estimator = resolve_estimator_name(estimator_name)
        model = build_model_pipeline(
            estimator_name=resolved_estimator,
            n_classes=len(observed_space),
            seed=seed,
            categorical_features=categorical_features,
            numeric_features=numeric_features,
        )
        model.fit(train_df[feature_columns], train_df["label"])
        model_type = f"{resolved_estimator}_multiclass"
    return {
        "category": category,
        "model": model,
        "model_type": model_type,
        "estimator": resolved_estimator,
        "feature_columns": feature_columns,
        "categorical_features": categorical_features,
        "numeric_features": numeric_features,
        "action_space": action_space,
        "train_label_distribution": dict(Counter(labels)),
        "train_rows": int(len(split_map["train"])),
        "val_rows": int(len(split_map["val"])),
        "test_rows": int(len(split_map["test"])),
        "seed": int(seed),
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def decision_rows_snapshot(decisions_df: "pd.DataFrame") -> dict[str, Any]:
    positives_non_stop = decisions_df[decisions_df["label"] != STOP_TOKEN].copy()
    positives_by_category = {
        str(key): int(value)
        for key, value in sorted(positives_non_stop.groupby("category").size().to_dict().items())
    }
    positives_by_action: dict[str, dict[str, int]] = {}
    for category, category_df in positives_non_stop.groupby("category"):
        counts = category_df["label"].astype(str).value_counts().to_dict()
        positives_by_action[str(category)] = {
            str(label): int(count)
            for label, count in sorted(counts.items())
        }
    return {
        "rows_total": int(len(decisions_df)),
        "decision_points_total": int(decisions_df["decision_id"].nunique()) if "decision_id" in decisions_df.columns else int(len(decisions_df)),
        "users_total": int(decisions_df["user_id"].nunique()) if "user_id" in decisions_df.columns else 0,
        "positives_excluding_stop": int(len(positives_non_stop)),
        "stop_rate": round(float((decisions_df["label"] == STOP_TOKEN).mean()), 6) if len(decisions_df) else 0.0,
        "positives_by_category": positives_by_category,
        "positives_by_action": positives_by_action,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }


def write_dataset_manifest(
    *,
    output_root: Path,
    dataset_dir: Path,
    dataset_kind: str,
    source_metadata: dict[str, Any],
    decisions_by_scheme: dict[str, "pd.DataFrame"],
) -> dict[str, Any]:
    manifest = {
        "dataset_kind": str(dataset_kind),
        "dataset_dir": str(dataset_dir),
        "source_version": str(source_metadata.get("version") or ""),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "schemes": {
            scheme: decision_rows_snapshot(frame)
            for scheme, frame in sorted(decisions_by_scheme.items())
        },
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "dataset_snapshot_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    lines = [
        f"# {dataset_kind.capitalize()} Live Dataset Snapshot",
        "",
        f"- dataset_dir: `{dataset_dir}`",
        f"- generated_at_utc: `{manifest['generated_at_utc']}`",
        "",
    ]
    for scheme, payload in sorted(manifest["schemes"].items()):
        lines.extend(
            [
                f"## {scheme}",
                "",
                f"- rows_total: **{payload['rows_total']}**",
                f"- decision_points_total: **{payload['decision_points_total']}**",
                f"- users_total: **{payload['users_total']}**",
                f"- positives_excluding_stop: **{payload['positives_excluding_stop']}**",
                f"- stop_rate: **{payload['stop_rate']:.6f}**",
                "",
                "### Positives By Category",
            ]
        )
        for category, count in sorted(payload["positives_by_category"].items()):
            lines.append(f"- {category}: **{count}**")
        lines.extend(["", "### Positives By Action"])
        for category, action_counts in sorted(payload["positives_by_action"].items()):
            summary = ", ".join(f"{label}={count}" for label, count in sorted(action_counts.items()))
            lines.append(f"- {category}: {summary}")
        lines.append("")
    (output_root / "dataset_snapshot_manifest.md").write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return manifest


def model_root_for_scheme(base_root: Path, scheme: str) -> Path:
    return base_root / str(scheme or "time").strip().lower()


def load_model_artifact(category: str, *, model_root: Path) -> dict[str, Any]:
    if joblib is None:  # pragma: no cover
        raise RuntimeError("joblib is required")
    category_dir = Path(model_root) / str(category or "").strip().lower()
    artifact = joblib.load(category_dir / "model.pkl")
    metadata = json.loads((category_dir / "metadata.json").read_text(encoding="utf-8"))
    bundle = dict(artifact)
    bundle["metadata"] = metadata
    bundle["category_dir"] = str(category_dir)
    return bundle


def predict_bundle_probabilities(bundle: dict[str, Any], decision_df: "pd.DataFrame") -> "pd.DataFrame":
    feature_columns = list(bundle.get("feature_columns") or [])
    raw = decision_df.reindex(columns=feature_columns)
    return predict_action_probabilities(bundle, raw)


def per_label_stats(y_true: list[str], y_pred: list[str], labels: list[str]) -> dict[str, Any]:
    stats: dict[str, Any] = {}
    for label in labels:
        tp = sum(int(truth == label and pred == label) for truth, pred in zip(y_true, y_pred))
        fp = sum(int(truth != label and pred == label) for truth, pred in zip(y_true, y_pred))
        fn = sum(int(truth == label and pred != label) for truth, pred in zip(y_true, y_pred))
        support = sum(int(truth == label) for truth in y_true)
        precision = float(tp / max(1, tp + fp))
        recall = float(tp / max(1, tp + fn))
        stats[str(label)] = {
            "support": int(support),
            "precision": round(precision, 6),
            "recall": round(recall, 6),
        }
    return stats


def confusion_matrix(y_true: list[str], y_pred: list[str], labels: list[str]) -> dict[str, dict[str, int]]:
    matrix: dict[str, dict[str, int]] = {
        str(truth): {str(pred): 0 for pred in labels}
        for truth in labels
    }
    for truth, pred in zip(y_true, y_pred):
        matrix.setdefault(str(truth), {str(item): 0 for item in labels})
        matrix[str(truth)].setdefault(str(pred), 0)
        matrix[str(truth)][str(pred)] += 1
    return matrix


def accuracy(y_true: list[str], y_pred: list[str]) -> float:
    if not y_true:
        return 0.0
    hits = sum(int(str(a) == str(b)) for a, b in zip(y_true, y_pred))
    return float(hits / max(1, len(y_true)))


def recall_at_k(prob_df: "pd.DataFrame", y_true: list[str], k: int) -> float:
    if prob_df.empty:
        return 0.0
    columns = list(prob_df.columns)
    values = prob_df.to_numpy(dtype=float)
    width = max(1, min(int(k), values.shape[1]))
    order = np.argsort(-values, axis=1)[:, :width]
    hits = 0
    for idx, truth in enumerate(y_true):
        labels = {columns[pos] for pos in order[idx]}
        hits += int(str(truth) in labels)
    return float(hits / max(1, len(y_true)))


def split_user_overlap(decisions_df: "pd.DataFrame") -> dict[str, int]:
    users = {
        split_name: set(decisions_df.loc[decisions_df["eval_split"] == split_name, "user_id"].astype(int).tolist())
        for split_name in ("train", "val", "test")
    }
    return {
        "train_val": int(len(users["train"] & users["val"])),
        "train_test": int(len(users["train"] & users["test"])),
        "val_test": int(len(users["val"] & users["test"])),
    }


def episode_targets_from_transitions(transitions_df: "pd.DataFrame", *, category: str) -> tuple[dict[int, list[str]], dict[int, list[str]]]:
    df = transitions_df[
        (transitions_df["category"].astype(str).str.lower() == str(category or "").strip().lower())
        & (pd.to_numeric(transitions_df["y"], errors="coerce").fillna(0).astype(int) == 1)
    ].copy()
    if df.empty:
        return {}, {}
    if "t0_utc" in df.columns:
        df["t0_utc"] = pd.to_datetime(df["t0_utc"], utc=True, format="mixed")
    df = df.sort_values(["episode_id", "t0_utc", "decision_id"]).reset_index(drop=True)
    by_episode: dict[int, list[tuple[int, str]]] = defaultdict(list)
    for row in df.itertuples(index=False):
        by_episode[int(getattr(row, "episode_id"))].append((int(getattr(row, "decision_id")), str(getattr(row, "label"))))
    episode_full_targets: dict[int, list[str]] = {}
    decision_suffix_targets: dict[int, list[str]] = {}
    for episode_id, items in by_episode.items():
        non_stop_sequence = [label for _decision_id, label in items if label != STOP_TOKEN]
        episode_full_targets[episode_id] = list(non_stop_sequence)
        for index, (decision_id, label) in enumerate(items):
            suffix = [value for _decision_id, value in items[index:] if value != STOP_TOKEN]
            decision_suffix_targets[int(decision_id)] = list(suffix)
    return episode_full_targets, decision_suffix_targets
