from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from .roadmap_initial_planner_common import (
        ACTION_SPACE_BY_CATEGORY,
        ensure_dependencies,
        build_decision_state_dataframe,
        load_teacher_dataset,
        repo_root,
        resolve_estimator_name,
        resolve_path,
        train_category_bundle,
    )
except ImportError:  # pragma: no cover
    from roadmap_initial_planner_common import (
        ACTION_SPACE_BY_CATEGORY,
        ensure_dependencies,
        build_decision_state_dataframe,
        load_teacher_dataset,
        repo_root,
        resolve_estimator_name,
        resolve_path,
        train_category_bundle,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="tmp/roadmap_teacher_v1")
    parser.add_argument("--model-root", type=str, default="models/roadmap_initial_planner")
    parser.add_argument("--model-version", type=str, default="roadmap_initial_planner_v1")
    parser.add_argument("--estimator", type=str, default="auto", help="auto|catboost|lightgbm|hgb")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--categories", type=str, default="")
    return parser.parse_args()


def selected_categories(raw: str) -> list[str]:
    if not str(raw or "").strip():
        return list(ACTION_SPACE_BY_CATEGORY.keys())
    out: list[str] = []
    for item in str(raw).split(","):
        token = str(item or "").strip().lower()
        if token and token not in out:
            out.append(token)
    return out


def main() -> None:
    ensure_dependencies()
    args = parse_args()
    data_dir = resolve_path(str(args.data_dir))
    model_root = resolve_path(str(args.model_root))
    model_root.mkdir(parents=True, exist_ok=True)

    stepwise_df, _sequence_df, dataset_metadata, _splits = load_teacher_dataset(data_dir)
    estimator_name = resolve_estimator_name(str(args.estimator))
    categories = selected_categories(args.categories)
    manifest: dict[str, object] = {
        "model_version": str(args.model_version),
        "dataset_dir": str(data_dir),
        "estimator": estimator_name,
        "seed": int(args.seed),
        "categories": {},
    }

    import joblib

    for category in categories:
        decisions_df = build_decision_state_dataframe(stepwise_df=stepwise_df, category=category, metadata=dataset_metadata)
        bundle = train_category_bundle(
            decisions_df=decisions_df,
            category=category,
            metadata=dataset_metadata,
            estimator_name=estimator_name,
            seed=int(args.seed),
        )
        category_dir = model_root / category
        category_dir.mkdir(parents=True, exist_ok=True)

        artifact = {
            "task": "roadmap_initial_planner_multiclass",
            "model": bundle["model"],
            "model_type": bundle["model_type"],
            "category": category,
            "model_version": str(args.model_version),
            "action_space": list(bundle["action_space"]),
            "feature_columns": list(bundle["feature_columns"]),
            "categorical_features": list(bundle["categorical_features"]),
            "numeric_features": list(bundle["numeric_features"]),
            "estimator": str(bundle["estimator"]),
            "trained_at_utc": str(bundle["trained_at_utc"]),
            "seed": int(bundle["seed"]),
        }
        joblib.dump(artifact, category_dir / "model.pkl")

        metadata = {
            "task": "roadmap_initial_planner_multiclass",
            "model_version": str(args.model_version),
            "category": category,
            "action_space": list(bundle["action_space"]),
            "observed_train_labels": [str(item) for item in getattr(bundle["model"], "classes_", [])],
            "feature_list": list(bundle["feature_columns"]),
            "categorical_features": list(bundle["categorical_features"]),
            "numeric_features": list(bundle["numeric_features"]),
            "train_rows": int(bundle["train_rows"]),
            "val_rows": int(bundle["val_rows"]),
            "test_rows": int(bundle["test_rows"]),
            "dataset_dir": str(data_dir),
            "training_timestamp_utc": str(bundle["trained_at_utc"]),
            "estimator": str(bundle["estimator"]),
            "model_type": str(bundle["model_type"]),
            "seed": int(bundle["seed"]),
            "max_steps": max(1, len(bundle["action_space"]) - 1),
            "planner_mode": "teacher_policy_imitation_initial",
            "dataset_version": str(dataset_metadata.get("version") or ""),
            "repo_root": str(repo_root()),
        }
        (category_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

        manifest["categories"][category] = {
            "model_dir": str(category_dir),
            "train_rows": int(bundle["train_rows"]),
            "val_rows": int(bundle["val_rows"]),
            "test_rows": int(bundle["test_rows"]),
            "estimator": str(bundle["estimator"]),
            "model_type": str(bundle["model_type"]),
        }
        print(
            "[train_roadmap_initial_planner] "
            f"category={category} estimator={bundle['estimator']} "
            f"rows(train/val/test)={bundle['train_rows']}/{bundle['val_rows']}/{bundle['test_rows']}"
        )

    (model_root / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[train_roadmap_initial_planner] model_root={model_root}")


if __name__ == "__main__":
    main()
