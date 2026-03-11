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

from admin_tools.management.commands.train_roadmap_nextstep_model_v4 import (
    _build_logistic_bundle,
    _build_ranker_bundle,
    _evaluate_scores,
    _fit_temperature,
    _group_sizes,
    _load_dataset,
    _negative_sample_train,
    _positive_group_mask,
    _predict_raw_scores,
    _prepare_features,
    _prepare_split_set,
    _prepare_split_set as _prepare_user_split_set,
    _param_grid,
    _repo_root,
    _resolve_estimator_name,
    _resolve_existing_dir,
    _resolve_output_dir,
)


def _planner_baseline_feature_columns(feature_columns: list[str]) -> list[str]:
    preferred = [
        "category",
        "candidate_type",
        "current_next_product_type",
        "last1_product_type",
        "last1_category",
        "candidate_position_in_generated_plan",
        "candidate_is_current_next_step",
        "candidate_popularity_in_train",
        "candidate_is_stop",
    ]
    return [col for col in preferred if col in set(feature_columns)]


def _split_user_sets(split_payload: dict[str, Any]) -> tuple[set[int], set[int], set[int]]:
    if {"train", "val", "test"}.issubset(set(split_payload.keys())):
        return (
            _prepare_user_split_set(split_payload, "train"),
            _prepare_user_split_set(split_payload, "val"),
            _prepare_user_split_set(split_payload, "test"),
        )
    return (
        _prepare_split_set(split_payload, "train_user_ids"),
        _prepare_split_set(split_payload, "val_user_ids"),
        _prepare_split_set(split_payload, "test_user_ids"),
    )


def _popularity_baseline_metrics(df: "pd.DataFrame") -> tuple[dict[str, Any], dict[str, Any]]:
    if df.empty:
        return _evaluate_scores(df=df, raw_scores=np.asarray([], dtype=float), temperature=1.0)
    raw_scores = pd.to_numeric(df.get("candidate_popularity_in_train"), errors="coerce").fillna(0.0).to_numpy()
    return _evaluate_scores(df=df, raw_scores=raw_scores, temperature=1.0)


def _candidate_popularity_priors(df: "pd.DataFrame") -> dict[str, dict[str, float]]:
    if df.empty or "candidate_popularity_in_train" not in df.columns:
        return {}
    priors: dict[str, dict[str, float]] = {}
    grouped = (
        df.groupby(["category", "candidate_type"], dropna=False)["candidate_popularity_in_train"]
        .median()
        .reset_index()
    )
    for row in grouped.itertuples(index=False):
        category = str(getattr(row, "category", "") or "").strip().lower()
        candidate_type = str(getattr(row, "candidate_type", "") or "").strip().lower()
        if not category or not candidate_type:
            continue
        priors.setdefault(category, {})[candidate_type] = round(
            float(getattr(row, "candidate_popularity_in_train", 0.0) or 0.0),
            6,
        )
    return priors


def _build_eval_markdown(report: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# Roadmap Planner v1 Evaluation")
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
    lines.append("## Popularity Baseline")
    lines.append("| split | recall@1 | recall@3 | recall@5 | ndcg@5 |")
    lines.append("| --- | --- | --- | --- | --- |")
    for split_name in ["val", "test"]:
        row = (report.get("dataset_baselines") or {}).get(split_name) or {}
        lines.append(
            f"| {split_name} | {float(row.get('recall_at_1', 0.0)):.4f} | {float(row.get('recall_at_3', 0.0)):.4f} | "
            f"{float(row.get('recall_at_5', 0.0)):.4f} | {float(row.get('ndcg_at_5', 0.0)):.4f} |"
        )
    lines.append("")
    lines.append("## Baseline Delta (test)")
    delta = (report.get("baseline_comparison") or {}).get("test") or {}
    lines.append(
        f"- dRecall@1: `{float(delta.get('recall_at_1', 0.0)):.4f}`  "
        f"dRecall@3: `{float(delta.get('recall_at_3', 0.0)):.4f}`  "
        f"dRecall@5: `{float(delta.get('recall_at_5', 0.0)):.4f}`  "
        f"dNDCG@5: `{float(delta.get('ndcg_at_5', 0.0)):.4f}`"
    )
    return "\n".join(lines) + "\n"


class Command(BaseCommand):
    help = "Train Roadmap Planner v1 baseline ranker from planner candidate-ranking dataset."

    def add_arguments(self, parser):
        parser.add_argument("--data-dir", type=str, default="data/ml/roadmap_planner_v1_ga30d")
        parser.add_argument("--model-dir", type=str, default="models/roadmap_planner_v1")
        parser.add_argument("--model-version", type=str, default="roadmap_planner_v1")
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
            default=20,
            help="Keep all positives and up to N negatives per episode in train split.",
        )

    def handle(self, *args, **options):
        if pd is None or np is None:
            raise CommandError("pandas + numpy are required. Install requirements-ml.txt")
        if joblib is None:
            raise CommandError("joblib is required. Install requirements-ml.txt")

        data_dir = _resolve_existing_dir(str(options["data_dir"]))
        model_dir = _resolve_output_dir(str(options["model_dir"]))
        model_version = str(options["model_version"]).strip() or "roadmap_planner_v1"
        seed = int(options["seed"])
        max_trials = int(options["trials"])
        allow_fallback = bool(options["allow_fallback"])
        estimator_name = _resolve_estimator_name(str(options.get("estimator") or "auto"), allow_fallback=allow_fallback)
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

        required_cols = {"user_id", "episode_id", "y", "category", "candidate_type"}
        missing = sorted(required_cols.difference(set(dataset_df.columns)))
        if missing:
            raise CommandError(f"Dataset missing required columns: {missing}")

        work = dataset_df.copy()
        work["user_id"] = work["user_id"].astype(int)
        work["episode_id"] = work["episode_id"].astype(int)
        work["y"] = work["y"].astype(int)
        work["category"] = work["category"].fillna("").astype(str).str.strip().str.lower()
        work["candidate_type"] = work["candidate_type"].fillna("").astype(str).str.strip().str.lower()
        work = work[work["candidate_type"] != ""].copy()
        if work.empty:
            raise CommandError("Dataset is empty after normalization.")

        train_users, val_users, test_users = _split_user_sets(split_payload)
        if not train_users or not val_users or not test_users:
            raise CommandError("splits.json has empty train/val/test user lists")

        feature_columns = [str(x) for x in (dataset_meta.get("feature_columns") or []) if str(x)]
        if not feature_columns:
            ignore_cols = {"user_id", "episode_id", "split", "t0_utc", "label", "y"}
            feature_columns = [col for col in work.columns if col not in ignore_cols]
        for col in feature_columns:
            if col not in work.columns:
                raise CommandError(f"Feature column missing in dataset: {col}")

        categorical_features = [col for col in (dataset_meta.get("categorical_features") or []) if col in feature_columns]
        numeric_features = [col for col in (dataset_meta.get("numeric_features") or []) if col in feature_columns]
        if not categorical_features and not numeric_features:
            guessed_cat = [col for col in feature_columns if str(work[col].dtype) == "object"]
            guessed_num = [col for col in feature_columns if col not in guessed_cat]
            categorical_features = guessed_cat
            numeric_features = guessed_num

        baseline_features = _planner_baseline_feature_columns(feature_columns)
        if not baseline_features:
            raise CommandError("Could not resolve baseline-only feature subset.")

        train_df_full = work[work["user_id"].isin(train_users)].sort_values(["episode_id", "candidate_type"]).reset_index(drop=True)
        val_df = work[work["user_id"].isin(val_users)].sort_values(["episode_id", "candidate_type"]).reset_index(drop=True)
        test_df = work[work["user_id"].isin(test_users)].sort_values(["episode_id", "candidate_type"]).reset_index(drop=True)
        if train_df_full.empty or val_df.empty or test_df.empty:
            raise CommandError(
                f"User-level split produced empty split: train={len(train_df_full)} val={len(val_df)} test={len(test_df)}"
            )

        sampled_train_df = _negative_sample_train(
            train_df_full,
            max_negatives_per_episode=max_negatives_per_episode,
            seed=seed,
        )
        candidate_popularity_priors = _candidate_popularity_priors(train_df_full)
        if estimator_name in {"lightgbm", "catboost"}:
            fit_train_df = sampled_train_df[_positive_group_mask(sampled_train_df)].sort_values(["episode_id", "candidate_type"]).reset_index(drop=True)
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

            best_score = None
            best_bundle = None
            best_params: dict[str, Any] = {}
            best_metrics_val = None
            best_per_category_val = None
            best_temperature = 1.0

            for trial_idx, params in enumerate(candidate_grid, start=1):
                self.stdout.write(
                    f"[train_roadmap_planner_model_v1] feature_set={feature_set_name} "
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
                temperature, _, _ = _fit_temperature(val_df, raw_val_scores)
                metrics_val, per_category_val = _evaluate_scores(
                    df=val_df,
                    raw_scores=raw_val_scores,
                    temperature=temperature,
                )
                score_tuple = (
                    float(metrics_val["ndcg_at_5"]),
                    float(metrics_val["recall_at_1"]),
                    float(metrics_val["recall_at_3"]),
                )
                if best_score is None or score_tuple > best_score:
                    best_score = score_tuple
                    best_bundle = bundle
                    best_params = dict(params)
                    best_metrics_val = metrics_val
                    best_per_category_val = per_category_val
                    best_temperature = float(temperature)

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
                "feature_columns": feature_set_columns,
                "categorical_features": cat_cols,
                "numeric_features": num_cols,
                "metrics_train": metrics_train,
                "metrics_val": best_metrics_val,
                "metrics_test": metrics_test,
                "per_category_train": per_category_train,
                "per_category_val": best_per_category_val,
                "per_category_test": per_category_test,
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

        popularity_val, _ = _popularity_baseline_metrics(val_df)
        popularity_test, _ = _popularity_baseline_metrics(test_df)
        dataset_baselines = {"val": popularity_val, "test": popularity_test}
        baseline_comparison = {
            split_name: {
                "recall_at_1": round(float(selected[f"metrics_{split_name}"]["recall_at_1"] - dataset_baselines[split_name]["recall_at_1"]), 6),
                "recall_at_3": round(float(selected[f"metrics_{split_name}"]["recall_at_3"] - dataset_baselines[split_name]["recall_at_3"]), 6),
                "recall_at_5": round(float(selected[f"metrics_{split_name}"]["recall_at_5"] - dataset_baselines[split_name]["recall_at_5"]), 6),
                "ndcg_at_5": round(float(selected[f"metrics_{split_name}"]["ndcg_at_5"] - dataset_baselines[split_name]["ndcg_at_5"]), 6),
            }
            for split_name in ["val", "test"]
        }

        trained_at = datetime.now(tz=timezone.utc).isoformat().replace("+00:00", "Z")
        planner_guard = {
            "metric": "ndcg_at_5",
            "required_delta_vs_popularity": 0.0,
            "model_value": float(selected["metrics_test"]["ndcg_at_5"]),
            "popularity_value": float(dataset_baselines["test"]["ndcg_at_5"]),
        }
        planner_guard["passed"] = planner_guard["model_value"] >= planner_guard["popularity_value"]

        artifact = {
            "task": "roadmap_planner_v1_ranking",
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
            "candidate_popularity_priors": candidate_popularity_priors,
            "selected_feature_set": selected_feature_set,
        }

        model_dir.mkdir(parents=True, exist_ok=True)
        model_path = model_dir / "model.pkl"
        joblib.dump(artifact, model_path)
        model_eval_report_path = model_dir / "eval_report.json"
        model_eval_report_md_path = model_dir / "eval_report.md"

        model_metadata = {
            "trained_at_utc": trained_at,
            "model_version": model_version,
            "task": "roadmap_planner_v1_ranking",
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
            "candidate_popularity_priors": candidate_popularity_priors,
            "metrics_train": selected["metrics_train"],
            "metrics_val": selected["metrics_val"],
            "metrics_test": selected["metrics_test"],
            "dataset_baselines": dataset_baselines,
            "baseline_comparison": baseline_comparison,
            "planner_guard": planner_guard,
            "eval_report_path": str(model_eval_report_path),
        }
        model_metadata_path = model_dir / "metadata.json"
        model_metadata_path.write_text(json.dumps(model_metadata, ensure_ascii=False, indent=2), encoding="utf-8")

        report = {
            "trained_at_utc": trained_at,
            "model_version": model_version,
            "task": "roadmap_planner_v1_ranking",
            "estimator": estimator_name,
            "selected_feature_set": selected_feature_set,
            "selected_params": selected["params"],
            "temperature": float(round(selected["temperature"], 6)),
            "dataset_path": dataset_path,
            "candidate_popularity_priors": candidate_popularity_priors,
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
            "feature_ablation": {
                name: {
                    "feature_count": len(row["feature_columns"]),
                    "selected_params": row["params"],
                    "metrics_val": row["metrics_val"],
                    "metrics_test": row["metrics_test"],
                }
                for name, row in feature_results.items()
            },
            "negative_sampling": {
                "max_negatives_per_episode": int(max_negatives_per_episode),
                "train_rows_before": int(len(train_df_full)),
                "train_rows_after": int(len(sampled_train_df)),
                "fit_rows_after_positive_filter": int(len(fit_train_df)),
            },
            "planner_guard": planner_guard,
        }
        model_eval_report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        model_eval_report_md_path.write_text(_build_eval_markdown(report), encoding="utf-8")

        reports_dir = (_repo_root() / "reports").resolve()
        reports_dir.mkdir(parents=True, exist_ok=True)
        report_json_path = reports_dir / "roadmap_planner_v1_eval.json"
        report_md_path = reports_dir / "roadmap_planner_v1_eval.md"
        report_json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        report_md_path.write_text(_build_eval_markdown(report), encoding="utf-8")

        self.stdout.write("[train_roadmap_planner_model_v1] done")
        self.stdout.write(f"[train_roadmap_planner_model_v1] estimator={estimator_name}")
        self.stdout.write(f"[train_roadmap_planner_model_v1] selected_feature_set={selected_feature_set}")
        self.stdout.write(f"[train_roadmap_planner_model_v1] model={model_path}")
        self.stdout.write(f"[train_roadmap_planner_model_v1] metadata={model_metadata_path}")
        self.stdout.write(f"[train_roadmap_planner_model_v1] model_eval_report={model_eval_report_path}")
        self.stdout.write(f"[train_roadmap_planner_model_v1] report_json={report_json_path}")
        self.stdout.write(f"[train_roadmap_planner_model_v1] report_md={report_md_path}")
        self.stdout.write(
            "[train_roadmap_planner_model_v1] "
            f"test_recall@1={selected['metrics_test']['recall_at_1']:.4f} "
            f"test_recall@3={selected['metrics_test']['recall_at_3']:.4f} "
            f"test_recall@5={selected['metrics_test']['recall_at_5']:.4f} "
            f"test_ndcg@5={selected['metrics_test']['ndcg_at_5']:.4f}"
        )
