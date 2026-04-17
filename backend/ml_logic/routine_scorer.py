"""Inference-time scorer for the skincare routine ranker.

Loads the trained ranker artifact (LightGBM or logistic fallback) and scores
candidate products for a given user profile + routine step. The module is
optional: if the model file is missing or the backend isn't installed,
`score_candidates` returns None so the caller can fall back to rule-based
selection.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None

try:
    import joblib
except Exception:  # pragma: no cover
    joblib = None


DEFAULT_MODEL_RELATIVE_PATH = "models/routine_ranker_v1/model.pkl"

CATEGORICAL_FEATURES = ["skin_type", "budget", "product_type", "strength", "step"]
NUMERIC_FEATURES = [
    "price",
    "has_price",
    "in_stock",
    "skin_type_match",
    "goal_concern_match_count",
    "goals_total",
    "avoid_flag_hit",
    "actives_count",
    "concerns_count",
]


def _repo_root() -> Path:
    # backend/ml_logic/routine_scorer.py -> repo root is three levels up.
    return Path(__file__).resolve().parents[2]


def _resolve_model_path() -> Path:
    try:
        from django.conf import settings
    except Exception:
        settings = None

    raw: str = ""
    if settings is not None:
        raw = str(getattr(settings, "ROUTINE_RANKER_MODEL_PATH", "") or "").strip()
    if raw:
        return Path(raw).expanduser()
    return (_repo_root() / DEFAULT_MODEL_RELATIVE_PATH).resolve()


@lru_cache(maxsize=2)
def _load_artifact_cached(path_str: str, mtime_ns: int) -> dict[str, Any] | None:
    if joblib is None:
        return None
    try:
        return joblib.load(path_str)
    except Exception:
        return None


def _load_artifact() -> dict[str, Any] | None:
    path = _resolve_model_path()
    if not path.exists() or not path.is_file():
        return None
    try:
        mtime_ns = int(path.stat().st_mtime_ns)
    except Exception:
        return None
    return _load_artifact_cached(str(path.resolve()), mtime_ns)


def is_model_available() -> bool:
    return _load_artifact() is not None


def _encode_categorical(value: Any, mapping: dict[str, int]) -> int:
    key = "" if value is None else str(value)
    return int(mapping.get(key, -1))


def _numeric(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _build_feature_row(
    *,
    profile: dict[str, Any],
    step: str,
    product: dict[str, Any],
    maps: dict[str, dict[str, int]],
) -> list[float]:
    goals = list(profile.get("goals") or [])
    avoid_flags = list(profile.get("avoid_flags") or [])
    concerns = list(product.get("concerns") or [])
    actives = list(product.get("actives") or [])
    supported = product.get("supported_skin_types") or []
    skin_type = str(profile.get("skin_type") or "")

    # Categorical features (must match training order).
    categorical_values = {
        "skin_type": skin_type or "normal",
        "budget": str(profile.get("budget") or "medium"),
        "product_type": str(product.get("product_type") or step),
        "strength": str(product.get("strength") or "low"),
        "step": str(step),
    }
    row: list[float] = [
        float(_encode_categorical(categorical_values[col], maps.get(col, {})))
        for col in CATEGORICAL_FEATURES
    ]

    # Numeric features.
    price_raw = product.get("price")
    price_val = _numeric(price_raw, 0.0)
    has_price = 1.0 if price_raw is not None else 0.0
    in_stock = 1.0 if product.get("in_stock", True) else 0.0
    skin_type_match = 1.0 if not supported or (skin_type and skin_type in supported) else 0.0
    goal_match = float(len(set(goals) & set(concerns))) if goals and concerns else 0.0
    avoid_hit = 1.0 if avoid_flags and (set(avoid_flags) & set(product.get("flags") or [])) else 0.0

    numeric_values = {
        "price": price_val,
        "has_price": has_price,
        "in_stock": in_stock,
        "skin_type_match": skin_type_match,
        "goal_concern_match_count": goal_match,
        "goals_total": float(len(goals)),
        "avoid_flag_hit": avoid_hit,
        "actives_count": float(len(actives)),
        "concerns_count": float(len(concerns)),
    }
    row.extend(float(numeric_values[col]) for col in NUMERIC_FEATURES)
    return row


def score_candidates(
    *,
    profile: dict[str, Any],
    step: str,
    candidates: Iterable[dict[str, Any]],
) -> list[float] | None:
    """Return a list of ranking scores aligned with `candidates`, or None if
    the model is unavailable.
    """
    if np is None:
        return None

    artifact = _load_artifact()
    if artifact is None:
        return None

    model = artifact.get("model")
    if model is None:
        return None

    candidate_list = list(candidates)
    if not candidate_list:
        return []

    maps = artifact.get("categorical_maps") or {}
    try:
        rows = [
            _build_feature_row(profile=profile, step=step, product=candidate, maps=maps)
            for candidate in candidate_list
        ]
        X = np.array(rows, dtype=np.float32)
    except Exception:
        return None

    model_type = str(artifact.get("model_type") or "")
    try:
        if hasattr(model, "predict_proba") and "logistic" in model_type:
            scores = model.predict_proba(X)[:, 1]
        else:
            scores = model.predict(X)
    except Exception:
        return None

    return [float(v) for v in np.asarray(scores).ravel().tolist()]
