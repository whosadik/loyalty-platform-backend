from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from roadmap_app.ml_live_planner import (  # noqa: E402
    load_live_planner,
    predict_live_next_action,
    rollout_live_plan,
)


def default_model_root(*, split_scheme: str = "time") -> Path:
    return REPO_ROOT / "models" / "roadmap_continuation_planner" / str(split_scheme or "time").strip().lower()


def load_continuation_planner(category: str, *, model_root: str | Path | None = None) -> dict[str, Any]:
    return load_live_planner(category, model_root=model_root or default_model_root())


def predict_continuation_next_action(
    category: str,
    decision_state: dict[str, Any],
    *,
    model_root: str | Path | None = None,
) -> list[dict[str, float | str]]:
    return predict_live_next_action(category, decision_state, model_root=model_root or default_model_root())


def rollout_continuation_suffix(
    category: str,
    decision_state: dict[str, Any],
    *,
    model_root: str | Path | None = None,
    max_steps: int = 10,
) -> list[str]:
    return rollout_live_plan(
        category,
        decision_state,
        model_root=model_root or default_model_root(),
        max_steps=max_steps,
    )

