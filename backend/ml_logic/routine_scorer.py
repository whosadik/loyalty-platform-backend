"""Inference-time scorer for the skincare routine ranker.

Loads the trained ranker artifact (LightGBM or logistic fallback) and scores
candidate products for a given user profile + routine step. The module is
optional: if the model file is missing or the backend isn't installed,
`score_candidates` returns None so the caller can fall back to rule-based
selection.

Feature computation is driven by the list stored inside the model artifact,
so old models without behavioral features still work without changes to the
caller.
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

DEFAULT_CATEGORICAL_FEATURES = ["skin_type", "budget", "product_type", "strength", "step"]
DEFAULT_NUMERIC_FEATURES = [
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


def _compute_categorical(
    *,
    name: str,
    profile: dict[str, Any],
    step: str,
    product: dict[str, Any],
) -> str:
    if name == "skin_type":
        return str(profile.get("skin_type") or "normal")
    if name == "budget":
        return str(profile.get("budget") or "medium")
    if name == "product_type":
        return str(product.get("product_type") or step)
    if name == "strength":
        return str(product.get("strength") or "low")
    if name == "step":
        return str(step)
    return ""


def _compute_numeric(
    *,
    name: str,
    profile: dict[str, Any],
    step: str,
    product: dict[str, Any],
    context: dict[str, Any],
) -> float:
    goals = list(profile.get("goals") or [])
    avoid_flags = list(profile.get("avoid_flags") or [])
    concerns = list(product.get("concerns") or [])
    actives = list(product.get("actives") or [])
    supported = product.get("supported_skin_types") or []
    skin_type = str(profile.get("skin_type") or "")
    product_id = int(product.get("id") or 0)
    product_stats = (context.get("product_signals") or {}).get(product_id) or {}

    if name == "price":
        return _numeric(product.get("price"), 0.0)
    if name == "has_price":
        return 1.0 if product.get("price") is not None else 0.0
    if name == "in_stock":
        return 1.0 if product.get("in_stock", True) else 0.0
    if name == "skin_type_match":
        if not supported:
            return 1.0
        return 1.0 if skin_type and skin_type in supported else 0.0
    if name == "goal_concern_match_count":
        if not goals or not concerns:
            return 0.0
        return float(len(set(goals) & set(concerns)))
    if name == "goals_total":
        return float(len(goals))
    if name == "avoid_flag_hit":
        if not avoid_flags:
            return 0.0
        return 1.0 if set(avoid_flags) & set(product.get("flags") or []) else 0.0
    if name == "actives_count":
        return float(len(actives))
    if name == "concerns_count":
        return float(len(concerns))
    # Behavioral features (come from context).
    if name == "user_tx_count_90d":
        return _numeric(context.get("user_tx_count_90d"), 0.0)
    if name == "user_owned_skincare_count":
        return _numeric(context.get("user_owned_skincare_count"), 0.0)
    if name == "product_popularity":
        return _numeric(product_stats.get("popularity"), 0.0)
    if name == "product_in_wishlist":
        return 1.0 if product_stats.get("in_wishlist") else 0.0
    if name == "product_roadmap_clicks_30d":
        return _numeric(product_stats.get("roadmap_clicks_30d"), 0.0)
    if name == "product_roadmap_skips_30d":
        return _numeric(product_stats.get("roadmap_skips_30d"), 0.0)
    return 0.0


def _build_feature_row(
    *,
    profile: dict[str, Any],
    step: str,
    product: dict[str, Any],
    categorical_features: list[str],
    numeric_features: list[str],
    maps: dict[str, dict[str, int]],
    context: dict[str, Any],
) -> list[float]:
    row: list[float] = []
    for name in categorical_features:
        raw = _compute_categorical(name=name, profile=profile, step=step, product=product)
        row.append(float(_encode_categorical(raw, maps.get(name, {}))))
    for name in numeric_features:
        row.append(
            _compute_numeric(
                name=name,
                profile=profile,
                step=step,
                product=product,
                context=context,
            )
        )
    return row


def score_candidates(
    *,
    profile: dict[str, Any],
    step: str,
    candidates: Iterable[dict[str, Any]],
    context: dict[str, Any] | None = None,
) -> list[float] | None:
    """Return a list of ranking scores aligned with `candidates`, or None if
    the model is unavailable.

    Args:
        profile: dict with keys skin_type, goals, avoid_flags, budget.
        step: routine step name (cleanser/serum/moisturizer/spf).
        candidates: iterable of product dicts.
        context: optional dict with runtime signals:
            - user_tx_count_90d: int
            - user_owned_skincare_count: int
            - product_signals: dict[product_id, {popularity, in_wishlist,
                roadmap_clicks_30d, roadmap_skips_30d}]
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

    categorical_features = list(artifact.get("categorical_features") or DEFAULT_CATEGORICAL_FEATURES)
    numeric_features = list(artifact.get("numeric_features") or DEFAULT_NUMERIC_FEATURES)
    maps = artifact.get("categorical_maps") or {}
    ctx = context or {}

    try:
        rows = [
            _build_feature_row(
                profile=profile,
                step=step,
                product=candidate,
                categorical_features=categorical_features,
                numeric_features=numeric_features,
                maps=maps,
                context=ctx,
            )
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
