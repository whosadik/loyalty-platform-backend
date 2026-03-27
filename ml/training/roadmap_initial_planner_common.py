from __future__ import annotations

import json
import math
import warnings
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

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

STOP_TOKEN = "__stop__"
NONE_TOKEN = "__none__"
ACTION_SPACE_BY_CATEGORY: dict[str, list[str]] = {
    "haircare": ["shampoo", "conditioner", "hair_mask", "hair_oil", "scalp_serum", "leave_in", STOP_TOKEN],
    "skincare": ["cleanser", "serum", "moisturizer", "spf", "toner", "mask", "eye_cream", "essence", STOP_TOKEN],
    "makeup": ["foundation", "mascara", "blush", "lipstick", "eyeshadow", "primer", "setting_spray", STOP_TOKEN],
    "fragrance": ["warm_day", "warm_evening", "cold_day", "cold_evening", STOP_TOKEN],
}

BASE_CATEGORICAL_FEATURES = [
    "seed_product_type",
    "seed_action_token",
    "seed_slot",
    "seed_scent_family",
    "seed_intensity",
    "seed_hair_type",
    "seed_scalp_type",
    "seed_hair_thickness",
    "seed_finish",
    "seed_coverage",
    "seed_undertone",
    "seed_tone_family",
    "seed_area",
    "seed_spf_signal",
    "seed_effect",
    "seed_waterproof",
    "profile_skin_type",
    "profile_budget",
    "profile_hair_type",
    "profile_scalp_type",
    "profile_hair_thickness",
    "profile_makeup_finish_pref_primary",
    "profile_makeup_coverage_pref_primary",
    "profile_makeup_undertone",
    "profile_makeup_tone_family",
    "profile_fragrance_intensity_pref",
    "anchor_product_type",
    "anchor_hair_type",
    "anchor_scalp_type",
    "anchor_hair_thickness",
    "anchor_finish",
    "anchor_coverage",
    "anchor_undertone",
    "anchor_tone_family",
    "anchor_scent_family",
    "anchor_intensity",
    "prev_step_1",
    "prev_step_2",
]

BASE_NUMERIC_FEATURES = [
    "seed_price",
    "seed_notes_count",
    "seed_concerns_count",
    "seed_actives_count",
    "seed_supported_skin_types_count",
    "seed_inci_token_count",
    "prior_category_purchase_total",
    "prior_category_distinct_token_count",
    "prior_current_owned_token_count",
    "prior_days_since_category_purchase",
    "avg_price_in_category_before_anchor",
    "prior_total_purchases_all",
    "prior_distinct_categories_count",
    "profile_goals_count",
    "profile_avoid_flags_count",
    "profile_hair_concerns_count",
    "profile_scalp_objective_count",
    "profile_has_scalp_objective",
    "profile_makeup_finish_pref_count",
    "profile_makeup_coverage_pref_count",
    "profile_makeup_concerns_count",
    "profile_fragrance_liked_families_count",
    "profile_fragrance_liked_notes_count",
    "anchor_concerns_count",
    "anchor_actives_count",
    "anchor_scalp_concern_count",
    "anchor_scalp_active_count",
    "anchor_has_scalp_focus",
    "anchor_supported_skin_types_count",
    "anchor_notes_count",
    "anchor_inci_token_count",
    "position",
    "prefix_length",
    "seed_brand_matches_favorite_overall",
    "seed_brand_matches_favorite_in_category",
    "remaining_action_count",
]


class ConstantActionModel:
    def __init__(self, action: str):
        self.action = str(action or STOP_TOKEN).strip().lower() or STOP_TOKEN
        self.classes_ = np.asarray([self.action], dtype=object) if np is not None else [self.action]

    def fit(self, X, y):  # pragma: no cover
        del X, y
        return self

    def predict_proba(self, X):
        rows = len(X)
        if np is None:  # pragma: no cover
            raise RuntimeError("numpy is required")
        return np.ones((rows, 1), dtype=float)

    def predict(self, X):
        if np is None:  # pragma: no cover
            raise RuntimeError("numpy is required")
        return np.asarray([self.action] * len(X), dtype=object)


def ensure_dependencies() -> None:
    if pd is None or np is None:
        raise RuntimeError("pandas and numpy are required")
    if joblib is None:
        raise RuntimeError("joblib is required")


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def resolve_path(raw_path: str) -> Path:
    candidate = Path(str(raw_path)).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    cwd_path = (Path.cwd() / candidate).resolve()
    if cwd_path.exists() or cwd_path.parent.exists():
        return cwd_path
    return (repo_root() / candidate).resolve()


def load_teacher_dataset(data_dir: str | Path) -> tuple["pd.DataFrame", "pd.DataFrame", dict[str, Any], dict[str, Any]]:
    ensure_dependencies()
    root = resolve_path(str(data_dir))
    stepwise = pd.read_parquet(root / "stepwise_dataset.parquet")
    sequence = pd.read_parquet(root / "sequence_dataset.parquet")
    metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
    splits = json.loads((root / "splits.json").read_text(encoding="utf-8"))
    return stepwise, sequence, metadata, splits


def action_space_for_category(category: str, metadata: dict[str, Any] | None = None) -> list[str]:
    category = str(category or "").strip().lower()
    if isinstance(metadata, dict):
        raw = (((metadata.get("task_definition") or {}).get("sequence_actions") or {}).get(category)) or []
        values = [str(item or "").strip().lower() for item in raw if str(item or "").strip()]
        if values:
            return values
    return list(ACTION_SPACE_BY_CATEGORY.get(category) or [STOP_TOKEN])


def action_tokens_for_category(category: str, metadata: dict[str, Any] | None = None) -> list[str]:
    return [token for token in action_space_for_category(category, metadata) if token != STOP_TOKEN]


def parse_json_list(value: Any) -> list[str]:
    if isinstance(value, list):
        raw = value
    else:
        text = str(value or "").strip()
        if not text:
            return []
        try:
            raw = json.loads(text)
        except Exception:
            return []
    out: list[str] = []
    seen: set[str] = set()
    for item in raw:
        token = str(item or "").strip().lower()
        if not token or token in seen:
            continue
        seen.add(token)
        out.append(token)
    return out


def build_rollout_state_features(*, category: str, initial_state: dict[str, Any], prefix: list[str], metadata: dict[str, Any] | None = None) -> dict[str, Any]:
    tokens = action_tokens_for_category(category, metadata)
    prefix_tokens = [str(item or "").strip().lower() for item in prefix if str(item or "").strip()]
    prefix_tokens = [item for idx, item in enumerate(prefix_tokens) if item != STOP_TOKEN and item not in prefix_tokens[:idx]]
    owned_tokens = set(parse_json_list(initial_state.get("prior_current_owned_tokens_json")))
    seed_brand = str(initial_state.get("seed_brand") or "").strip().lower()
    favorite_overall = str(initial_state.get("favorite_brand_overall_before_anchor") or "").strip().lower()
    favorite_category = str(initial_state.get("favorite_brand_in_category_before_anchor") or "").strip().lower()

    row = dict(initial_state)
    row["category"] = str(category or "").strip().lower()
    row["position"] = int(len(prefix_tokens) + 1)
    row["prefix_length"] = int(len(prefix_tokens))
    row["prev_step_1"] = prefix_tokens[-1] if prefix_tokens else NONE_TOKEN
    row["prev_step_2"] = prefix_tokens[-2] if len(prefix_tokens) >= 2 else NONE_TOKEN
    row["seed_brand_matches_favorite_overall"] = int(bool(seed_brand and favorite_overall and seed_brand == favorite_overall))
    row["seed_brand_matches_favorite_in_category"] = int(bool(seed_brand and favorite_category and seed_brand == favorite_category))

    seen_count = 0
    for token in tokens:
        seen_flag = int(token in prefix_tokens)
        owned_flag = int(token in owned_tokens)
        row[f"seen_{token}"] = seen_flag
        row[f"owned_{token}"] = owned_flag
        seen_count += seen_flag
    row["remaining_action_count"] = max(0, int(len(tokens) - seen_count))
    row["prefix_steps_json"] = json.dumps(prefix_tokens, ensure_ascii=False)
    return row


def build_decision_state_dataframe(
    *,
    stepwise_df: "pd.DataFrame",
    category: str,
    metadata: dict[str, Any] | None = None,
) -> "pd.DataFrame":
    ensure_dependencies()
    category = str(category or "").strip().lower()
    df = stepwise_df[
        (stepwise_df["category"].astype(str).str.lower() == category)
        & (pd.to_numeric(stepwise_df["y"], errors="coerce").fillna(0).astype(int) == 1)
    ].copy()
    if df.empty:
        return df

    df["label"] = df["teacher_target_at_position"].astype(str).str.strip().str.lower()
    tokens = action_tokens_for_category(category, metadata)
    prefix_lists = df["prefix_steps_json"].map(parse_json_list)
    owned_lists = df["prior_current_owned_tokens_json"].map(parse_json_list)

    def _series(name: str, default: Any = ""):
        if name in df.columns:
            return df[name]
        return pd.Series([default] * len(df), index=df.index)

    seed_brand = _series("seed_brand").astype(str).str.strip().str.lower()
    favorite_overall = _series("favorite_brand_overall_before_anchor").astype(str).str.strip().str.lower()
    favorite_category = _series("favorite_brand_in_category_before_anchor").astype(str).str.strip().str.lower()
    df["seed_brand_matches_favorite_overall"] = ((seed_brand != "") & (seed_brand == favorite_overall)).astype(int)
    df["seed_brand_matches_favorite_in_category"] = ((seed_brand != "") & (seed_brand == favorite_category)).astype(int)

    for token in tokens:
        df[f"seen_{token}"] = prefix_lists.map(lambda items, token=token: int(token in items))
        df[f"owned_{token}"] = owned_lists.map(lambda items, token=token: int(token in items))
    if tokens:
        df["remaining_action_count"] = len(tokens) - df[[f"seen_{token}" for token in tokens]].sum(axis=1)
    else:
        df["remaining_action_count"] = 0

    df["position"] = pd.to_numeric(df.get("position"), errors="coerce").fillna(0).astype(int)
    df["prefix_length"] = pd.to_numeric(df.get("prefix_length"), errors="coerce").fillna(0).astype(int)
    return df.sort_values(["planning_id", "position"]).reset_index(drop=True)


def feature_spec_for_category(category: str, decisions_df: "pd.DataFrame", metadata: dict[str, Any] | None = None) -> tuple[list[str], list[str], list[str]]:
    tokens = action_tokens_for_category(category, metadata)
    categorical_features = [name for name in BASE_CATEGORICAL_FEATURES if name in decisions_df.columns]
    numeric_features = [name for name in BASE_NUMERIC_FEATURES if name in decisions_df.columns]
    numeric_features.extend([f"seen_{token}" for token in tokens if f"seen_{token}" in decisions_df.columns])
    numeric_features.extend([f"owned_{token}" for token in tokens if f"owned_{token}" in decisions_df.columns])
    feature_columns = categorical_features + numeric_features
    return feature_columns, categorical_features, numeric_features


def _make_one_hot_encoder():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:  # pragma: no cover
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def resolve_estimator_name(requested: str) -> str:
    requested = str(requested or "auto").strip().lower()
    if requested in {"hgb", "histgradientboosting", "hist_gradient_boosting"}:
        return "hgb"
    if requested == "catboost":
        try:
            import catboost  # noqa: F401
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("CatBoost is not installed") from exc
        return "catboost"
    if requested == "lightgbm":
        try:
            import lightgbm  # noqa: F401
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("LightGBM is not installed") from exc
        return "lightgbm"
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


def build_estimator(*, estimator_name: str, n_classes: int, seed: int):
    estimator_name = resolve_estimator_name(estimator_name)
    if estimator_name == "catboost":
        from catboost import CatBoostClassifier

        return CatBoostClassifier(
            loss_function="MultiClass",
            random_seed=seed,
            depth=6,
            learning_rate=0.05,
            iterations=250,
            verbose=False,
            allow_writing_files=False,
            thread_count=4,
        )
    if estimator_name == "lightgbm":
        from lightgbm import LGBMClassifier

        return LGBMClassifier(
            objective="multiclass",
            num_class=max(2, int(n_classes)),
            random_state=seed,
            n_estimators=250,
            learning_rate=0.05,
            num_leaves=31,
        )
    return HistGradientBoostingClassifier(
        random_state=seed,
        learning_rate=0.05,
        max_depth=8,
        max_iter=220,
    )


def build_model_pipeline(*, estimator_name: str, n_classes: int, seed: int, categorical_features: list[str], numeric_features: list[str]):
    preprocess = ColumnTransformer(
        transformers=[
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="constant", fill_value=NONE_TOKEN)),
                        ("onehot", _make_one_hot_encoder()),
                    ]
                ),
                list(categorical_features),
            ),
            (
                "numeric",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="constant", fill_value=0)),
                    ]
                ),
                list(numeric_features),
            ),
        ],
        remainder="drop",
    )
    estimator = build_estimator(estimator_name=estimator_name, n_classes=n_classes, seed=seed)
    return Pipeline(
        steps=[
            ("preprocess", preprocess),
            ("classifier", estimator),
        ]
    )


def split_frames(decisions_df: "pd.DataFrame") -> dict[str, "pd.DataFrame"]:
    return {
        "train": decisions_df[decisions_df["split"].astype(str) == "train"].copy(),
        "val": decisions_df[decisions_df["split"].astype(str) == "val"].copy(),
        "test": decisions_df[decisions_df["split"].astype(str) == "test"].copy(),
    }


def train_category_bundle(
    *,
    decisions_df: "pd.DataFrame",
    category: str,
    metadata: dict[str, Any],
    estimator_name: str,
    seed: int,
) -> dict[str, Any]:
    ensure_dependencies()
    feature_columns, categorical_features, numeric_features = feature_spec_for_category(category, decisions_df, metadata)
    split_map = split_frames(decisions_df)
    train_df = split_map["train"].sort_values(["planning_id", "position"]).reset_index(drop=True)
    labels = train_df["label"].astype(str).str.lower()
    if train_df.empty:
        raise RuntimeError(f"No train rows for category={category}")
    unique_labels = [str(item) for item in sorted(labels.unique())]
    action_space = action_space_for_category(category, metadata)
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


def predict_action_probabilities(bundle: dict[str, Any], feature_df: "pd.DataFrame") -> "pd.DataFrame":
    ensure_dependencies()
    model = bundle["model"]
    feature_columns = list(bundle["feature_columns"])
    raw = feature_df.reindex(columns=feature_columns)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="X does not have valid feature names, but LGBMClassifier was fitted with feature names",
        )
        proba = model.predict_proba(raw)
    if proba.ndim == 1:
        proba = proba.reshape(-1, 1)
    classes = [str(item).strip().lower() for item in getattr(model, "classes_", [])]
    action_space = [str(item).strip().lower() for item in bundle.get("action_space") or []]
    out = pd.DataFrame(0.0, index=raw.index, columns=action_space, dtype=float)
    for idx, label in enumerate(classes):
        if label not in out.columns:
            continue
        out[label] = np.asarray(proba[:, idx], dtype=float)
    row_sums = out.sum(axis=1).replace(0.0, np.nan)
    out = out.div(row_sums, axis=0).fillna(0.0)
    return out


def rank_actions_from_probabilities(proba_row: dict[str, Any] | "pd.Series", action_space: list[str]) -> list[dict[str, float | str]]:
    if hasattr(proba_row, "to_dict"):
        payload = proba_row.to_dict()
    else:
        payload = dict(proba_row)
    order = {token: idx for idx, token in enumerate(action_space)}
    ranked = [
        {"action": token, "prob": float(payload.get(token, 0.0) or 0.0)}
        for token in action_space
    ]
    ranked.sort(key=lambda item: (-float(item["prob"]), order.get(str(item["action"]), math.inf)))
    return ranked


def longest_common_prefix_rate(predicted: list[str], target: list[str]) -> float:
    target_len = max(1, len(target))
    match = 0
    for pred, truth in zip(predicted, target):
        if str(pred) != str(truth):
            break
        match += 1
    return float(match / target_len)


def sequence_exact_match(predicted: list[str], target: list[str]) -> int:
    return int(list(predicted) == list(target))


def majority_label_baseline(train_df: "pd.DataFrame") -> str:
    if train_df.empty:
        return STOP_TOKEN
    counts = Counter(train_df["label"].astype(str))
    return str(max(counts.items(), key=lambda item: (item[1], item[0]))[0])


def previous_step_prior_map(train_df: "pd.DataFrame") -> dict[str, str]:
    mapping: dict[str, str] = {}
    if train_df.empty:
        return mapping
    grouped: dict[str, Counter] = defaultdict(Counter)
    for row in train_df.itertuples(index=False):
        prev_step = str(getattr(row, "prev_step_1", NONE_TOKEN) or NONE_TOKEN)
        grouped[prev_step][str(getattr(row, "label", STOP_TOKEN) or STOP_TOKEN)] += 1
    for prev_step, counts in grouped.items():
        mapping[prev_step] = str(max(counts.items(), key=lambda item: (item[1], item[0]))[0])
    return mapping
