from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

from django.conf import settings


def _model_path() -> Path:
    if bool(getattr(settings, "ROADMAP_NEXTSTEP_V3_ENABLED", False)):
        raw = str(getattr(settings, "ROADMAP_NEXTSTEP_V3_MODEL_PATH", "") or "").strip()
        if not raw:
            raw = str(getattr(settings, "ROADMAP_NEXTSTEP_MODEL_PATH", "") or "")
        return Path(raw).expanduser()
    return Path(str(getattr(settings, "ROADMAP_NEXTSTEP_MODEL_PATH", "") or "")).expanduser()


@lru_cache(maxsize=4)
def _load_model_cached(path_str: str, mtime_ns: int) -> Any | None:
    del mtime_ns
    path = Path(path_str)
    if not path.exists() or not path.is_file():
        return None
    try:
        import joblib

        return joblib.load(path_str)
    except Exception:
        pass
    try:
        import pickle

        with path.open("rb") as f:
            return pickle.load(f)
    except Exception:
        return None


def _load_model() -> Any | None:
    path = _model_path()
    if not path.exists() or not path.is_file():
        return None
    try:
        return _load_model_cached(str(path.resolve()), int(path.stat().st_mtime_ns))
    except Exception:
        return None


def _to_float(v: Any) -> float:
    try:
        return float(v)
    except Exception:
        return 0.0


def _normalize_predictions(raw: Any) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if raw is None:
        return out

    if isinstance(raw, dict):
        for k, v in raw.items():
            pt = str(k or "").strip()
            if not pt:
                continue
            out.append({"product_type": pt, "score": _to_float(v)})
    elif isinstance(raw, (list, tuple)):
        for i, item in enumerate(raw):
            if isinstance(item, dict):
                pt = str(item.get("product_type") or "").strip()
                if not pt:
                    continue
                out.append({"product_type": pt, "score": _to_float(item.get("score", 0.0))})
                continue
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                pt = str(item[0] or "").strip()
                if not pt:
                    continue
                out.append({"product_type": pt, "score": _to_float(item[1])})
                continue
            pt = str(item or "").strip()
            if pt:
                out.append({"product_type": pt, "score": max(0.0, 1.0 - (i * 0.1))})

    out.sort(key=lambda x: float(x.get("score", 0.0)), reverse=True)
    dedup: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in out:
        pt = row["product_type"]
        if pt in seen:
            continue
        seen.add(pt)
        dedup.append({"product_type": pt, "score": float(row.get("score", 0.0))})
    return dedup


def predict_next_product_types(user, context_product_ids: list[int], category: str) -> list[dict[str, Any]]:
    """
    Return ranked product_type predictions for next roadmap step.
    If model artifact is absent or incompatible, returns [].
    """
    model = _load_model()
    if model is None:
        return []

    context_ids = [int(x) for x in (context_product_ids or []) if str(x).strip()]
    raw = None

    try:
        if hasattr(model, "predict_next_product_types"):
            raw = model.predict_next_product_types(
                user_id=int(getattr(user, "id", 0) or 0),
                context_product_ids=context_ids,
                category=category,
            )
        elif callable(model):
            raw = model(
                user_id=int(getattr(user, "id", 0) or 0),
                context_product_ids=context_ids,
                category=category,
            )
        elif hasattr(model, "predict"):
            raw = model.predict(
                {
                    "user_id": int(getattr(user, "id", 0) or 0),
                    "context_product_ids": context_ids,
                    "category": category,
                }
            )
    except Exception:
        return []

    return _normalize_predictions(raw)
