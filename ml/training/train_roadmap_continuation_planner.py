from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

try:
    from .roadmap_continuation_planner_common import (
        DEFAULT_CATEGORIES,
        build_continuation_decision_dataframe,
        continuation_categories,
        ensure_dependencies,
        load_live_dataset_bundle,
        repo_root,
        resolve_estimator_name,
        resolve_path,
        selected_split_schemes,
        train_live_category_bundle,
        write_dataset_manifest,
    )
except ImportError:  # pragma: no cover
    from roadmap_continuation_planner_common import (
        DEFAULT_CATEGORIES,
        build_continuation_decision_dataframe,
        continuation_categories,
        ensure_dependencies,
        load_live_dataset_bundle,
        repo_root,
        resolve_estimator_name,
        resolve_path,
        selected_split_schemes,
        train_live_category_bundle,
        write_dataset_manifest,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="tmp/roadmap_continuation_dataset_v1")
    parser.add_argument("--model-root", type=str, default="models/roadmap_continuation_planner")
    parser.add_argument("--model-version", type=str, default="roadmap_continuation_planner_v1")
    parser.add_argument("--estimator", type=str, default="auto", help="auto|catboost|lightgbm|hgb")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--categories", type=str, default="haircare,skincare,fragrance")
    parser.add_argument("--split-schemes", type=str, default="time,user")
    return parser.parse_args()


def _scheme_root(model_root: Path, scheme: str) -> Path:
    return model_root / str(scheme or "time").strip().lower()


def main() -> None:
    ensure_dependencies()
    args = parse_args()
    data_dir = resolve_path(str(args.data_dir))
    model_root = resolve_path(str(args.model_root))
    model_root.mkdir(parents=True, exist_ok=True)
    dataset_df, dataset_metadata, _splits = load_live_dataset_bundle(data_dir)
    estimator_name = resolve_estimator_name(str(args.estimator))
    categories = continuation_categories(args.categories)
    if not categories:
        categories = list(DEFAULT_CATEGORIES)
    split_schemes = selected_split_schemes(args.split_schemes)

    decisions_by_scheme: dict[str, Any] = {}
    for scheme in split_schemes:
        per_category_frames = [
            build_continuation_decision_dataframe(
                dataset_df=dataset_df,
                category=category,
                split_scheme=scheme,
                seed=int(args.seed),
            )
            for category in categories
        ]
        non_empty = [frame for frame in per_category_frames if not frame.empty]
        decisions_by_scheme[scheme] = non_empty[0].iloc[0:0].copy() if not non_empty else pd.concat(non_empty, ignore_index=True)
    write_dataset_manifest(
        output_root=model_root,
        dataset_dir=data_dir,
        dataset_kind="continuation",
        source_metadata=dataset_metadata,
        decisions_by_scheme=decisions_by_scheme,
    )

    import joblib

    manifest: dict[str, object] = {
        "task": "roadmap_continuation_planner_multiclass",
        "model_version": str(args.model_version),
        "dataset_dir": str(data_dir),
        "estimator": estimator_name,
        "seed": int(args.seed),
        "categories": categories,
        "split_schemes": split_schemes,
        "schemes": {},
    }

    for scheme in split_schemes:
        scheme_root = _scheme_root(model_root, scheme)
        scheme_root.mkdir(parents=True, exist_ok=True)
        scheme_manifest: dict[str, Any] = {}
        for category in categories:
            decisions_df = build_continuation_decision_dataframe(
                dataset_df=dataset_df,
                category=category,
                split_scheme=scheme,
                seed=int(args.seed),
            )
            if decisions_df.empty:
                continue
            bundle = train_live_category_bundle(
                decisions_df=decisions_df,
                category=category,
                estimator_name=estimator_name,
                seed=int(args.seed),
            )
            category_dir = scheme_root / category
            category_dir.mkdir(parents=True, exist_ok=True)
            artifact = {
                "task": "roadmap_continuation_planner_multiclass",
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
                "split_scheme": scheme,
            }
            joblib.dump(artifact, category_dir / "model.pkl")
            metadata = {
                "task": "roadmap_continuation_planner_multiclass",
                "planner_mode": "live_supervised_continuation",
                "model_version": str(args.model_version),
                "category": category,
                "action_space": list(bundle["action_space"]),
                "observed_train_labels": [str(item) for item in getattr(bundle["model"], "classes_", [])],
                "label_map": [str(item) for item in getattr(bundle["model"], "classes_", [])],
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
                "split_scheme": scheme,
                "continuation_only": True,
                "repo_root": str(repo_root()),
            }
            (category_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            scheme_manifest[category] = {
                "model_dir": str(category_dir),
                "train_rows": int(bundle["train_rows"]),
                "val_rows": int(bundle["val_rows"]),
                "test_rows": int(bundle["test_rows"]),
                "feature_count": int(len(bundle["feature_columns"])),
                "estimator": str(bundle["estimator"]),
                "model_type": str(bundle["model_type"]),
            }
            print(
                "[train_roadmap_continuation_planner] "
                f"scheme={scheme} category={category} estimator={bundle['estimator']} "
                f"rows(train/val/test)={bundle['train_rows']}/{bundle['val_rows']}/{bundle['test_rows']}"
            )
        manifest["schemes"][scheme] = scheme_manifest

    (model_root / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[train_roadmap_continuation_planner] model_root={model_root}")


if __name__ == "__main__":
    main()

