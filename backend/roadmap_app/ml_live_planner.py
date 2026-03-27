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
    rank_actions_from_probabilities,
)


def default_model_root(task: str, *, split_scheme: str = "time") -> Path:
    task = str(task or "initial").strip().lower()
    split_scheme = str(split_scheme or "time").strip().lower()
    root_name = "roadmap_live_initial_planner_v1" if task == "initial" else "roadmap_live_transition_planner_v1"
    return REPO_ROOT / "models" / root_name / split_scheme


def _artifact_dir(category: str, model_root: str | Path | None) -> Path:
    root = Path(model_root).expanduser().resolve() if model_root else default_model_root("initial")
    return root / str(category or "").strip().lower()


@lru_cache(maxsize=32)
def _load_artifact_cached(artifact_path: str, mtime_ns: int) -> dict[str, Any] | None:
    del mtime_ns
    if joblib is None:
        return None
    path = Path(artifact_path)
    if not path.exists():
        return None
    payload = joblib.load(path)
    return payload if isinstance(payload, dict) else None


@lru_cache(maxsize=32)
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


def load_live_planner(category: str, *, model_root: str | Path | None = None) -> dict[str, Any]:
    category = str(category or "").strip().lower()
    artifact_dir = _artifact_dir(category, model_root)
    model_path = artifact_dir / "model.pkl"
    metadata_path = artifact_dir / "metadata.json"
    if not model_path.exists():
        raise FileNotFoundError(f"model.pkl not found for category={category} in {artifact_dir}")
    artifact = _load_artifact_cached(str(model_path.resolve()), int(model_path.stat().st_mtime_ns))
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


def _numeric_feature_names(bundle: dict[str, Any]) -> set[str]:
    return set(str(item) for item in (bundle.get("numeric_features") or []))


def build_live_feature_row(category: str, decision_state: dict[str, Any], *, model_root: str | Path | None = None) -> dict[str, Any]:
    bundle = load_live_planner(category, model_root=model_root)
    numeric_features = _numeric_feature_names(bundle)
    out: dict[str, Any] = {}
    for key in bundle.get("feature_columns") or []:
        if key in decision_state:
            out[key] = decision_state.get(key)
        elif key in numeric_features:
            out[key] = 0.0
        else:
            out[key] = NONE_TOKEN
    return out


def predict_live_next_action(category: str, decision_state: dict[str, Any], *, model_root: str | Path | None = None) -> list[dict[str, float | str]]:
    if pd is None:
        raise RuntimeError("pandas is required")
    bundle = load_live_planner(category, model_root=model_root)
    feature_row = build_live_feature_row(category, dict(decision_state), model_root=model_root)
    feature_columns = list(bundle.get("feature_columns") or [])
    action_space = list(bundle.get("action_space") or ACTION_SPACE_BY_CATEGORY.get(category) or [STOP_TOKEN])
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


def _parse_runtime_plan(decision_state: dict[str, Any]) -> list[str]:
    raw = decision_state.get("__runtime_plan_tokens")
    if isinstance(raw, list):
        return [str(item or "").strip().lower() for item in raw if str(item or "").strip()]
    json_raw = decision_state.get("runtime_plan_tokens_json")
    if isinstance(json_raw, list):
        return [str(item or "").strip().lower() for item in json_raw if str(item or "").strip()]
    if json_raw:
        try:
            parsed = json.loads(str(json_raw))
            if isinstance(parsed, list):
                return [str(item or "").strip().lower() for item in parsed if str(item or "").strip()]
        except Exception:
            pass
    text = str(decision_state.get("plan_product_types") or "").strip()
    if not text:
        return []
    return [str(item or "").strip().lower() for item in text.split("|") if str(item or "").strip()]


def _shift_history(state: dict[str, Any], *, action: str, category: str) -> None:
    current_types = [str(state.get(f"last{i}_product_type") or NONE_TOKEN) for i in range(1, 6)]
    current_categories = [str(state.get(f"last{i}_category") or NONE_TOKEN) for i in range(1, 6)]
    updated_types = [str(action)] + current_types[:4]
    updated_categories = [str(category)] + current_categories[:4]
    for idx, value in enumerate(updated_types, start=1):
        state[f"last{idx}_product_type"] = value
    for idx, value in enumerate(updated_categories, start=1):
        state[f"last{idx}_category"] = value


def _advance_state_after_action(category: str, state: dict[str, Any], action: str) -> None:
    action = str(action or "").strip().lower()
    if not action or action == STOP_TOKEN:
        return
    chosen = set(str(item or "").strip().lower() for item in state.get("__chosen_actions", []) if str(item or "").strip())
    chosen.add(action)
    state["__chosen_actions"] = sorted(chosen)

    runtime_plan = list(_parse_runtime_plan(state))
    if action in runtime_plan:
        runtime_plan.remove(action)
    state["__runtime_plan_tokens"] = list(runtime_plan)

    _shift_history(state, action=action, category=category)
    state["days_since_last_purchase_in_category"] = 0
    for field in (
        "prior_category_purchase_total",
        "tx_count_90d_category",
        "completed_steps_count",
        "steps_completed_in_episode_count",
    ):
        if field in state:
            state[field] = int(state.get(field) or 0) + 1
    for field in ("remaining_actionable_steps_count", "remaining_depth_in_plan"):
        if field in state:
            state[field] = max(0, int(state.get(field) or 0) - 1)
    if "prior_category_distinct_token_count" in state:
        state["prior_category_distinct_token_count"] = max(
            int(state.get("prior_category_distinct_token_count") or 0),
            len(chosen),
        )
    if category == "fragrance" and "fragrance_slot_coverage_count" in state:
        state["fragrance_slot_coverage_count"] = max(
            int(state.get("fragrance_slot_coverage_count") or 0),
            len(chosen),
        )
    if "next_step_index_current" in state:
        next_index = int(state.get("next_step_index_current") or 1)
        state["next_step_index_current"] = max(1, next_index + 1)
    if runtime_plan:
        state["current_next_product_type"] = runtime_plan[0]
    else:
        state["current_next_product_type"] = STOP_TOKEN


def rollout_live_plan(category: str, decision_state: dict[str, Any], *, model_root: str | Path | None = None, max_steps: int = 10) -> list[str]:
    bundle = load_live_planner(category, model_root=model_root)
    action_space = [str(item or "").strip().lower() for item in (bundle.get("action_space") or ACTION_SPACE_BY_CATEGORY.get(category) or [])]
    non_stop_actions = [token for token in action_space if token != STOP_TOKEN]
    state = dict(decision_state)
    state.setdefault("__runtime_plan_tokens", _parse_runtime_plan(state))
    state.setdefault("__chosen_actions", [])
    cap = max(1, min(int(max_steps), max(1, len(non_stop_actions))))
    chain: list[str] = []
    for _ in range(cap):
        ranked = predict_live_next_action(category, state, model_root=model_root)
        chosen = STOP_TOKEN
        for item in ranked:
            action = str(item.get("action") or "").strip().lower()
            if not action:
                continue
            if action == STOP_TOKEN:
                chosen = STOP_TOKEN
                break
            if action not in non_stop_actions or action in chain:
                continue
            chosen = action
            break
        if chosen == STOP_TOKEN:
            break
        chain.append(chosen)
        _advance_state_after_action(category, state, chosen)
    return chain
