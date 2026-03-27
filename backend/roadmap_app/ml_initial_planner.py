from __future__ import annotations

import json
import sys
import warnings
from functools import lru_cache
from pathlib import Path
from typing import Any

try:
    import joblib
except Exception:  # pragma: no cover
    joblib = None

try:
    import pandas as pd
except Exception:  # pragma: no cover
    pd = None

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ml.training.roadmap_initial_planner_common import (  # noqa: E402
    ACTION_SPACE_BY_CATEGORY,
    NONE_TOKEN,
    STOP_TOKEN,
    build_rollout_state_features,
    rank_actions_from_probabilities,
)


def planner_model_root() -> Path:
    return REPO_ROOT / "models" / "roadmap_initial_planner"


def _artifact_dir(category: str, model_root: str | Path | None = None) -> Path:
    root = Path(model_root).expanduser().resolve() if model_root else planner_model_root()
    return root / str(category or "").strip().lower()


@lru_cache(maxsize=16)
def _load_artifact_cached(category: str, artifact_path: str, mtime_ns: int) -> dict[str, Any] | None:
    del category, mtime_ns
    if joblib is None:
        return None
    path = Path(artifact_path)
    if not path.exists():
        return None
    try:
        payload = joblib.load(path)
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


@lru_cache(maxsize=16)
def _load_metadata_cached(metadata_path: str, mtime_ns: int) -> dict[str, Any] | None:
    del mtime_ns
    path = Path(metadata_path)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def load_initial_planner(category: str, model_root: str | Path | None = None) -> dict[str, Any]:
    category = str(category or "").strip().lower()
    artifact_dir = _artifact_dir(category, model_root)
    model_path = artifact_dir / "model.pkl"
    metadata_path = artifact_dir / "metadata.json"
    if not model_path.exists():
        raise FileNotFoundError(f"model.pkl not found for category={category} in {artifact_dir}")
    artifact = _load_artifact_cached(category, str(model_path.resolve()), int(model_path.stat().st_mtime_ns))
    metadata = (
        _load_metadata_cached(str(metadata_path.resolve()), int(metadata_path.stat().st_mtime_ns))
        if metadata_path.exists()
        else {}
    )
    if not artifact:
        raise RuntimeError(f"Unable to load planner artifact from {model_path}")
    bundle = dict(artifact)
    bundle["metadata"] = metadata if isinstance(metadata, dict) else {}
    bundle["category"] = category
    bundle["artifact_dir"] = str(artifact_dir)
    return bundle


def build_feature_row(category: str, initial_state: dict[str, Any], prefix: list[str], model_root: str | Path | None = None) -> dict[str, Any]:
    bundle = load_initial_planner(category, model_root=model_root)
    metadata = bundle.get("metadata") if isinstance(bundle.get("metadata"), dict) else {}
    feature_row = build_rollout_state_features(category=category, initial_state=dict(initial_state), prefix=list(prefix), metadata=metadata)
    numeric_features = set(bundle.get("numeric_features") or [])
    out: dict[str, Any] = {}
    for key in bundle.get("feature_columns") or []:
        if key in feature_row:
            out[key] = feature_row.get(key)
        elif key in numeric_features or key.startswith(("seen_", "owned_")):
            out[key] = 0.0
        else:
            out[key] = NONE_TOKEN
    return out


def predict_next_action(category: str, feature_row: dict[str, Any], model_root: str | Path | None = None) -> list[dict[str, float | str]]:
    if pd is None:
        raise RuntimeError("pandas is required")
    bundle = load_initial_planner(category, model_root=model_root)
    feature_columns = list(bundle.get("feature_columns") or [])
    action_space = list((bundle.get("action_space") or ACTION_SPACE_BY_CATEGORY.get(category) or []))
    frame = pd.DataFrame([{col: feature_row.get(col) for col in feature_columns}], columns=feature_columns)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="X does not have valid feature names, but LGBMClassifier was fitted with feature names",
        )
        proba = bundle["model"].predict_proba(frame)
    classes = [str(item or "").strip().lower() for item in getattr(bundle["model"], "classes_", [])]
    scores = {token: 0.0 for token in action_space}
    if getattr(proba, "ndim", 0) == 1:
        proba = [proba]
    row = proba[0]
    for idx, label in enumerate(classes):
        if label in scores:
            scores[label] = float(row[idx] or 0.0)
    total = sum(scores.values())
    if total > 0.0:
        scores = {token: float(value / total) for token, value in scores.items()}
    return rank_actions_from_probabilities(scores, action_space)


def rollout_initial_plan(category: str, initial_state: dict[str, Any], max_steps: int = 10, model_root: str | Path | None = None) -> list[str]:
    bundle = load_initial_planner(category, model_root=model_root)
    action_space = [str(item or "").strip().lower() for item in (bundle.get("action_space") or ACTION_SPACE_BY_CATEGORY.get(category) or [])]
    non_stop_actions = [token for token in action_space if token != STOP_TOKEN]
    cap = max(1, min(int(max_steps), max(1, len(non_stop_actions))))
    chain: list[str] = []
    for _ in range(cap):
        feature_row = build_feature_row(category, initial_state, chain, model_root=model_root)
        ranked = predict_next_action(category, feature_row, model_root=model_root)
        chosen = STOP_TOKEN
        for item in ranked:
            action = str(item.get("action") or "").strip().lower()
            if not action:
                continue
            if action == STOP_TOKEN:
                chosen = STOP_TOKEN
                break
            if action not in non_stop_actions:
                continue
            if action in chain:
                continue
            chosen = action
            break
        if chosen == STOP_TOKEN:
            break
        chain.append(chosen)
    return chain
