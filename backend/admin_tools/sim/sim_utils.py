from __future__ import annotations

import math
import random
from contextlib import contextmanager
from datetime import datetime
from typing import Any
from unittest.mock import patch


@contextmanager
def patched_now(sim_now: datetime):
    """
    Patch django.utils.timezone.now for one request scope.
    """
    with patch("django.utils.timezone.now", return_value=sim_now):
        yield


def sample_poisson(rng: random.Random, lam: float) -> int:
    if lam <= 0:
        return 0
    l = math.exp(-lam)
    k = 0
    p = 1.0
    while p > l:
        k += 1
        p *= rng.random()
    return max(0, k - 1)


def weighted_choice(rng: random.Random, weights: dict[str, float], default_key: str) -> str:
    total = 0.0
    items: list[tuple[str, float]] = []
    for key, value in weights.items():
        w = float(value)
        if w <= 0:
            continue
        items.append((key, w))
        total += w
    if total <= 0 or not items:
        return default_key
    threshold = rng.random() * total
    acc = 0.0
    for key, w in items:
        acc += w
        if acc >= threshold:
            return key
    return items[-1][0]


def make_request_id(
    *,
    day_index: int,
    user_id: int,
    session_index: int,
    event_index: int,
    suffix: str = "",
) -> str:
    base = f"sim-d{day_index}-u{user_id}-s{session_index}-e{event_index}"
    if suffix:
        base = f"{base}-{suffix}"
    return base[:64]


def clamp_prob(value: Any) -> float:
    try:
        v = float(value)
    except Exception:
        return 0.0
    if v < 0:
        return 0.0
    if v > 1:
        return 1.0
    return v

