"""Inference-time scorer for offer redemption propensity.

Loads the trained LogisticRegression pipeline from
`OFFER_REDEMPTION_MODEL_PATH` and returns redemption probability per
candidate offer. Mirrors the shape of `ml_logic/routine_scorer.py`:

- Optional: if joblib or the artifact is missing, `score_offers` returns
  None and the caller must fall back to rule-based scoring.
- Cached by (path, mtime_ns) so edits to the artifact are picked up
  without a process restart.

Feature schema (kept in sync with `ml/training/train_offer_redemption_lr.py`):
    categorical: campaign_name, offer_type, target_scope
    numeric: offer_value, estimated_cost, cooldown_days, expires_in_days,
             is_exposed, is_clicked, recency_days, frequency_90d,
             monetary_90d, txn_count_before, spend_before

At selection time we don't yet know if the user will be exposed/click,
so we set `is_exposed=1` (forward-looking assumption — user will see
the chosen offer) and `is_clicked=0`. The resulting score is therefore
P(redeem | exposed, not-yet-clicked, features).
"""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None

try:
    import pandas as pd
except Exception:  # pragma: no cover
    pd = None

try:
    import joblib
except Exception:  # pragma: no cover
    joblib = None


logger = logging.getLogger(__name__)


DEFAULT_CATEGORICAL_FEATURES = ["campaign_name", "offer_type", "target_scope"]
DEFAULT_NUMERIC_FEATURES = [
    "offer_value",
    "estimated_cost",
    "cooldown_days",
    "expires_in_days",
    "is_exposed",
    "is_clicked",
    "recency_days",
    "frequency_90d",
    "monetary_90d",
    "txn_count_before",
    "spend_before",
]


def _resolve_model_path() -> Path | None:
    try:
        from django.conf import settings
    except Exception:
        return None

    raw = str(getattr(settings, "OFFER_REDEMPTION_MODEL_PATH", "") or "").strip()
    if not raw:
        return None
    return Path(raw).expanduser()


@lru_cache(maxsize=2)
def _load_model_cached(path_str: str, mtime_ns: int) -> Any | None:
    if joblib is None:
        return None
    try:
        return joblib.load(path_str)
    except Exception:
        logger.exception("offer_redemption: failed to load model from %s", path_str)
        return None


def _load_model() -> Any | None:
    path = _resolve_model_path()
    if path is None or not path.exists() or not path.is_file():
        return None
    try:
        mtime_ns = int(path.stat().st_mtime_ns)
    except Exception:
        return None
    return _load_model_cached(str(path.resolve()), mtime_ns)


def is_model_available() -> bool:
    return _load_model() is not None


def _build_row(
    *,
    offer: Any,
    campaign_name: str,
    rfm: dict[str, Any],
) -> dict[str, Any]:
    """Assemble one feature row for a candidate offer.

    `offer` is the Django Offer instance; `rfm` comes from
    `offers.services._rfm(user, now)` which already returns
    recency_days / frequency_90d / monetary_90d.

    `txn_count_before` and `spend_before` are approximated from the 90d
    RFM values — we don't refetch full history for each candidate to
    avoid N+1 queries at selection time. The training data uses full
    history, so this is a mild distribution shift but preserves
    ordinal relationships (high spenders still score higher).
    """
    offer_type = str(getattr(offer, "offer_type", "") or "unknown")
    target_scope = str(getattr(offer, "target_scope", "") or "unknown")
    offer_value = float(getattr(offer, "value", 0) or 0)
    estimated_cost = float(getattr(offer, "estimated_cost", 0) or 0)
    cooldown_days = int(getattr(offer, "cooldown_days", 0) or 0)
    expires_in_days = int(getattr(offer, "expires_in_days", 0) or 0)

    recency_days = int(rfm.get("recency_days") or 9999)
    frequency_90d = int(rfm.get("frequency_90d") or 0)
    monetary_90d = float(rfm.get("monetary_90d") or 0.0)

    return {
        "campaign_name": campaign_name or "none",
        "offer_type": offer_type,
        "target_scope": target_scope,
        "offer_value": offer_value,
        "estimated_cost": estimated_cost,
        "cooldown_days": cooldown_days,
        "expires_in_days": expires_in_days,
        "is_exposed": 1,  # forward-looking — user will see chosen offer
        "is_clicked": 0,  # unknown at selection time
        "recency_days": recency_days,
        "frequency_90d": frequency_90d,
        "monetary_90d": monetary_90d,
        "txn_count_before": frequency_90d,
        "spend_before": monetary_90d,
    }


def score_offers(
    *,
    offers: Iterable[Any],
    campaign_name: str,
    rfm: dict[str, Any],
) -> list[float] | None:
    """Return redemption probabilities aligned with `offers`, or None if
    the model is unavailable or inference fails.

    Args:
        offers: iterable of Offer instances (the candidates being scored
            inside a single campaign).
        campaign_name: name of the campaign these offers belong to.
        rfm: dict with recency_days, frequency_90d, monetary_90d.
    """
    if np is None or pd is None:
        return None

    pipeline = _load_model()
    if pipeline is None:
        return None

    offer_list = list(offers)
    if not offer_list:
        return []

    try:
        rows = [
            _build_row(offer=o, campaign_name=campaign_name, rfm=rfm)
            for o in offer_list
        ]
        X = pd.DataFrame(rows, columns=DEFAULT_CATEGORICAL_FEATURES + DEFAULT_NUMERIC_FEATURES)
    except Exception:
        logger.exception("offer_redemption: failed to build feature frame")
        return None

    try:
        probs = pipeline.predict_proba(X)[:, 1]
    except Exception:
        logger.exception("offer_redemption: predict_proba failed")
        return None

    return [float(p) for p in np.asarray(probs).ravel().tolist()]
