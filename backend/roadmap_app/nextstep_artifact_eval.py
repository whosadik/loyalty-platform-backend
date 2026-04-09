from __future__ import annotations

from datetime import datetime, timezone as dt_timezone
from pathlib import Path
import json
from typing import Any

from django.core.management.base import CommandError

from admin_tools.management.commands.train_roadmap_nextstep_model_v4 import (
    _build_eval_markdown,
    _compare_with_baselines,
    _evaluate_scores,
    _load_dataset,
    _normalize_binary_target,
    _predict_raw_scores,
    _prepare_features,
    _prepare_split_set,
    _resolve_existing_dir,
    _sort_frame,
)
from roadmap_app.ml_artifact_proof import PROOF_FILE_EVAL, PROOF_FILE_METADATA, artifact_file_path, load_json_file
from roadmap_app.ml_next_step import _load_model_for_path


def _safe_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _safe_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _artifact_metadata(model_path: str | Path | None) -> dict[str, Any]:
    return load_json_file(artifact_file_path(model_path, PROOF_FILE_METADATA)) or {}


def _resolve_data_dir_for_artifact(
    model_path: str | Path | None,
    *,
    data_dir: str | Path | None = None,
) -> Path:
    if str(data_dir or "").strip():
        return _resolve_existing_dir(str(data_dir))

    metadata = _artifact_metadata(model_path)
    dataset_path_raw = str(metadata.get("dataset_path") or "").strip()
    if not dataset_path_raw:
        raise CommandError(
            f"Artifact metadata at {artifact_file_path(model_path, PROOF_FILE_METADATA)} "
            "does not declare dataset_path."
        )

    dataset_path = Path(dataset_path_raw).expanduser()
    if not dataset_path.exists() or not dataset_path.is_file():
        raise CommandError(f"Artifact dataset_path does not exist: {dataset_path}")
    return dataset_path.parent.resolve()


def _infer_estimator_name(*, metadata: dict[str, Any], artifact: dict[str, Any]) -> str:
    estimator = str(metadata.get("estimator") or "").strip().lower()
    if estimator:
        return estimator
    model_type = str(artifact.get("model_type") or "").strip().lower()
    if "lightgbm" in model_type:
        return "lightgbm"
    if "catboost" in model_type:
        return "catboost"
    if "logistic" in model_type:
        return "logistic"
    return model_type or "unknown"


def _runtime_guard_from_report(report: dict[str, Any]) -> dict[str, Any]:
    baselines = _safe_dict(_safe_dict(_safe_dict(report.get("dataset_baselines")).get("splits")).get("test"))
    popularity = _safe_dict(baselines.get("popularity"))
    model_value_raw = _safe_dict(report.get("metrics_test")).get("ndcg_at_5")
    baseline_value_raw = popularity.get("ndcg_at_5")
    try:
        model_value = float(model_value_raw)
        baseline_value = float(baseline_value_raw)
    except Exception:
        return {
            "applicable": True,
            "metric": "ndcg_at_5",
            "required_delta": 0.01,
            "passed": False,
            "reason": "missing_popularity_baseline",
        }

    return {
        "applicable": True,
        "metric": "ndcg_at_5",
        "required_delta": 0.01,
        "model_value": model_value,
        "popularity_value": baseline_value,
        "passed": float(model_value) >= float(baseline_value) + 0.01,
    }


def build_nextstep_v4_artifact_eval_report(
    *,
    model_path: str | Path,
    data_dir: str | Path | None = None,
) -> dict[str, Any]:
    resolved_model_path = Path(str(model_path or "").strip()).expanduser().resolve()
    if not resolved_model_path.exists() or not resolved_model_path.is_file():
        raise CommandError(f"Model path not found: {resolved_model_path}")

    artifact = _load_model_for_path(str(resolved_model_path))
    if not isinstance(artifact, dict):
        raise CommandError(f"Unable to load model artifact: {resolved_model_path}")
    if str(artifact.get("task") or "").strip() != "roadmap_nextstep_v4_ranking":
        raise CommandError(f"Artifact is not roadmap_nextstep_v4_ranking: {resolved_model_path}")

    metadata = _artifact_metadata(resolved_model_path)
    data_dir_resolved = _resolve_data_dir_for_artifact(resolved_model_path, data_dir=data_dir)
    dataset_df, dataset_path = _load_dataset(data_dir_resolved)
    splits_path = data_dir_resolved / "splits.json"
    dataset_metadata_path = data_dir_resolved / "metadata.json"
    if not splits_path.exists():
        raise CommandError(f"Missing splits.json in {data_dir_resolved}")
    if not dataset_metadata_path.exists():
        raise CommandError(f"Missing metadata.json in {data_dir_resolved}")

    split_payload = json.loads(splits_path.read_text(encoding="utf-8"))
    dataset_meta = json.loads(dataset_metadata_path.read_text(encoding="utf-8"))

    feature_columns = [str(x) for x in (_safe_list(artifact.get("feature_columns")) or _safe_list(dataset_meta.get("feature_columns"))) if str(x)]
    categorical_features = [
        str(x)
        for x in (_safe_list(artifact.get("categorical_features")) or _safe_list(dataset_meta.get("categorical_features")))
        if str(x) in feature_columns
    ]
    numeric_features = [
        str(x)
        for x in (_safe_list(artifact.get("numeric_features")) or _safe_list(dataset_meta.get("numeric_features")))
        if str(x) in feature_columns
    ]
    if not feature_columns:
        raise CommandError("Artifact evaluation could not resolve feature columns.")

    target_column = str(
        metadata.get("target_column") or dataset_meta.get("target_column") or "y"
    ).strip() or "y"
    required_cols = {"user_id", "episode_id", "group_id", "category", "candidate_type", target_column, *feature_columns}
    missing = sorted(col for col in required_cols if col not in dataset_df.columns)
    if missing:
        raise CommandError(f"Dataset missing columns required by artifact: {missing}")

    work = dataset_df.copy()
    work["user_id"] = work["user_id"].astype(int)
    work["episode_id"] = work["episode_id"].astype(int)
    work["group_id"] = work["group_id"].astype(int)
    work["y"] = _normalize_binary_target(work[target_column], target_column=target_column)
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

    train_df = _sort_frame(work[work["user_id"].isin(train_users)].copy())
    val_df = _sort_frame(work[work["user_id"].isin(val_users)].copy())
    test_df = _sort_frame(work[work["user_id"].isin(test_users)].copy())
    if train_df.empty or val_df.empty or test_df.empty:
        raise CommandError(
            "Artifact evaluation split is empty: "
            f"train={len(train_df)} val={len(val_df)} test={len(test_df)}"
        )

    estimator_name = _infer_estimator_name(metadata=metadata, artifact=artifact)
    model_bundle = {
        "model": artifact.get("model"),
        "preprocessor": artifact.get("preprocessor"),
    }
    temperature = float(artifact.get("temperature") or metadata.get("temperature") or 1.0)

    X_val = _prepare_features(
        val_df,
        feature_columns=feature_columns,
        categorical_features=categorical_features,
        numeric_features=numeric_features,
        estimator_name=estimator_name,
    )
    X_test = _prepare_features(
        test_df,
        feature_columns=feature_columns,
        categorical_features=categorical_features,
        numeric_features=numeric_features,
        estimator_name=estimator_name,
    )
    raw_scores_val = _predict_raw_scores(model_bundle, X_val)
    raw_scores_test = _predict_raw_scores(model_bundle, X_test)
    metrics_val, per_category_val = _evaluate_scores(
        df=val_df,
        raw_scores=raw_scores_val,
        temperature=temperature,
    )
    metrics_test, per_category_test = _evaluate_scores(
        df=test_df,
        raw_scores=raw_scores_test,
        temperature=temperature,
    )

    dataset_baselines = _safe_dict(dataset_meta.get("baselines"))
    baseline_comparison = _compare_with_baselines(
        metrics_val=metrics_val,
        metrics_test=metrics_test,
        dataset_baselines=dataset_baselines,
    )

    trained_at_utc = str(
        artifact.get("trained_at_utc") or metadata.get("trained_at_utc") or datetime.now(tz=dt_timezone.utc).isoformat()
    )
    report: dict[str, Any] = {
        "generated_at_utc": datetime.now(tz=dt_timezone.utc).isoformat(),
        "report_mode": "exact_artifact_rebuild",
        "trained_at_utc": trained_at_utc,
        "model_path": str(resolved_model_path),
        "model_version": str(artifact.get("model_version") or metadata.get("model_version") or "").strip(),
        "estimator": estimator_name,
        "target_column": target_column,
        "label_protocol_version": str(
            metadata.get("label_protocol_version") or dataset_meta.get("label_protocol_version") or "legacy_v4"
        ).strip() or "legacy_v4",
        "outcome_windows_days": _safe_list(
            metadata.get("outcome_windows_days") or dataset_meta.get("outcome_windows_days")
        ),
        "selected_feature_set": str(
            artifact.get("selected_feature_set") or metadata.get("selected_feature_set") or "full"
        ).strip() or "full",
        "temperature": float(round(temperature, 6)),
        "dataset_path": str(dataset_path),
        "dataset_metadata_path": str(dataset_metadata_path.resolve()),
        "splits_path": str(splits_path.resolve()),
        "train_rows": int(len(train_df)),
        "val_rows": int(len(val_df)),
        "test_rows": int(len(test_df)),
        "train_groups": int(train_df["episode_id"].nunique()),
        "val_groups": int(val_df["episode_id"].nunique()),
        "test_groups": int(test_df["episode_id"].nunique()),
        "metrics_val": metrics_val,
        "metrics_test": metrics_test,
        "per_category_val": per_category_val,
        "per_category_test": per_category_test,
        "dataset_baselines": dataset_baselines,
        "baseline_comparison": baseline_comparison,
        "runtime_guard": {},
        "notes": [
            "Exact artifact eval rebuilt by scoring the configured model.pkl on the preserved dataset/splits.",
            "No retraining performed in this report path.",
        ],
    }
    report["runtime_guard"] = _runtime_guard_from_report(report)
    return report


def write_nextstep_v4_artifact_eval_report(
    *,
    model_path: str | Path,
    data_dir: str | Path | None = None,
    output_json: str | Path | None = None,
    output_md: str | Path | None = None,
) -> tuple[dict[str, Any], Path, Path]:
    report = build_nextstep_v4_artifact_eval_report(model_path=model_path, data_dir=data_dir)
    default_json_path = artifact_file_path(model_path, PROOF_FILE_EVAL).resolve()
    json_path = Path(str(output_json or default_json_path)).expanduser().resolve()
    md_path = Path(str(output_md or json_path.with_suffix(".md"))).expanduser().resolve()

    json_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(_build_eval_markdown(report), encoding="utf-8")
    return report, json_path, md_path
