"""Synthetic generator for offer-redemption training data.

Produces a parquet file with the exact schema expected by
`train_offer_redemption_lr.py`, but with realistic causal structure
between features and `label_redeemed` so the model can actually learn
(unlike uniform random labels that cap AUC at ~0.5).

Causal model in short:
    - Users get an RFM-like segment (new / active / at_risk / vip).
    - Each segment has priors for recency / frequency / monetary.
    - Each campaign has a preferred offer_type and segment affinity.
    - Redemption probability = sigmoid(logit) where logit combines:
        * segment × offer_type interaction
        * recency decay (fresh users redeem more)
        * cooldown fatigue for high-frequency users
        * offer_value effect (saturating)
        * exposure / click gating
        * small campaign intrinsic effect
    - Gaussian noise is added to keep the task non-deterministic so
      AUC lands around 0.75-0.82 rather than 1.0.

Output schema matches `train_offer_redemption_lr.py`:
    assignment_id, user_id, assigned_at, label_redeemed,
    campaign_name, offer_type, target_scope,
    offer_value, estimated_cost, cooldown_days, expires_in_days,
    is_exposed, is_clicked,
    recency_days, frequency_90d, monetary_90d,
    txn_count_before, spend_before
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd


SEGMENTS = ["new", "active", "at_risk", "vip"]
SEGMENT_WEIGHTS = [0.15, 0.50, 0.20, 0.15]

OFFER_TYPES = ["discount", "points_multiplier", "gift"]
TARGET_SCOPES = ["cart", "category", "product_type", "product_id"]


@dataclass(frozen=True)
class Campaign:
    name: str
    offer_type: str
    target_scope: str
    value_range: tuple[float, float]
    cost_range: tuple[float, float]
    cooldown_days: int
    expires_in_days: int
    # per-segment logit boost
    segment_bias: dict[str, float]
    intrinsic: float  # base logit shift


CAMPAIGNS: list[Campaign] = [
    Campaign(
        name="summer_sale",
        offer_type="discount",
        target_scope="cart",
        value_range=(5.0, 25.0),
        cost_range=(100.0, 400.0),
        cooldown_days=14,
        expires_in_days=7,
        segment_bias={"new": 0.2, "active": 0.8, "at_risk": 0.4, "vip": 0.1},
        intrinsic=-0.2,
    ),
    Campaign(
        name="vip_points_boost",
        offer_type="points_multiplier",
        target_scope="cart",
        value_range=(2.0, 5.0),
        cost_range=(200.0, 800.0),
        cooldown_days=21,
        expires_in_days=14,
        segment_bias={"new": -0.4, "active": 0.3, "at_risk": -0.2, "vip": 1.6},
        intrinsic=-0.1,
    ),
    Campaign(
        name="winback_discount",
        offer_type="discount",
        target_scope="category",
        value_range=(15.0, 35.0),
        cost_range=(150.0, 500.0),
        cooldown_days=30,
        expires_in_days=10,
        segment_bias={"new": -0.1, "active": 0.0, "at_risk": 1.4, "vip": -0.2},
        intrinsic=-0.3,
    ),
    Campaign(
        name="welcome_gift",
        offer_type="gift",
        target_scope="product_type",
        value_range=(0.0, 0.0),
        cost_range=(80.0, 200.0),
        cooldown_days=90,
        expires_in_days=21,
        segment_bias={"new": 1.3, "active": 0.1, "at_risk": 0.2, "vip": -0.1},
        intrinsic=-0.2,
    ),
    Campaign(
        name="seasonal_makeup",
        offer_type="discount",
        target_scope="product_type",
        value_range=(10.0, 20.0),
        cost_range=(120.0, 350.0),
        cooldown_days=14,
        expires_in_days=7,
        segment_bias={"new": 0.1, "active": 0.6, "at_risk": 0.3, "vip": 0.3},
        intrinsic=-0.25,
    ),
    Campaign(
        name="fragrance_promo",
        offer_type="discount",
        target_scope="category",
        value_range=(8.0, 18.0),
        cost_range=(200.0, 600.0),
        cooldown_days=30,
        expires_in_days=14,
        segment_bias={"new": -0.1, "active": 0.3, "at_risk": 0.2, "vip": 0.5},
        intrinsic=-0.35,
    ),
    Campaign(
        name="vip_exclusive_gift",
        offer_type="gift",
        target_scope="product_id",
        value_range=(0.0, 0.0),
        cost_range=(300.0, 1200.0),
        cooldown_days=45,
        expires_in_days=14,
        segment_bias={"new": -0.3, "active": 0.0, "at_risk": 0.1, "vip": 1.8},
        intrinsic=-0.1,
    ),
    Campaign(
        name="everyday_points",
        offer_type="points_multiplier",
        target_scope="category",
        value_range=(1.5, 3.0),
        cost_range=(50.0, 200.0),
        cooldown_days=7,
        expires_in_days=5,
        segment_bias={"new": 0.3, "active": 0.7, "at_risk": 0.2, "vip": 0.6},
        intrinsic=-0.2,
    ),
]


def _draw_segment_rfm(seg: str, rng: np.random.Generator) -> tuple[int, int, float]:
    """Sample (recency_days, frequency_90d, monetary_90d) given segment."""
    if seg == "new":
        recency = int(rng.integers(0, 14))
        freq = int(rng.integers(0, 2))
        monetary = float(rng.gamma(shape=2.0, scale=500.0))
    elif seg == "active":
        recency = int(rng.integers(3, 30))
        freq = int(rng.integers(2, 7))
        monetary = float(rng.gamma(shape=3.0, scale=2500.0))
    elif seg == "at_risk":
        recency = int(rng.integers(60, 180))
        freq = int(rng.integers(0, 3))
        monetary = float(rng.gamma(shape=2.0, scale=1500.0))
    else:  # vip
        recency = int(rng.integers(0, 14))
        freq = int(rng.integers(6, 20))
        monetary = float(rng.gamma(shape=4.0, scale=8000.0))
    return recency, freq, monetary


def _pick_campaign_for_segment(seg: str, rng: np.random.Generator) -> Campaign:
    """Mildly biased campaign selection so each segment sees a plausible mix
    (not uniform). This mirrors what a half-decent rule engine would do and
    keeps ML's job non-trivial."""
    weights = np.array([
        1.0 + max(0.0, c.segment_bias.get(seg, 0.0)) for c in CAMPAIGNS
    ])
    weights = weights / weights.sum()
    idx = int(rng.choice(len(CAMPAIGNS), p=weights))
    return CAMPAIGNS[idx]


def _sigmoid(x: np.ndarray | float) -> np.ndarray | float:
    return 1.0 / (1.0 + np.exp(-x))


def generate(
    n_users: int,
    avg_assignments_per_user: float,
    horizon_days: int,
    seed: int,
    base_logit_shift: float = -4.2,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    now = datetime.now(timezone.utc)
    start = now - timedelta(days=horizon_days)

    segments_per_user = rng.choice(SEGMENTS, size=n_users, p=SEGMENT_WEIGHTS)

    rows = []
    assignment_id = 1

    for user_idx in range(n_users):
        user_id = user_idx + 1
        seg = str(segments_per_user[user_idx])

        # per-user propensity bias (unobserved heterogeneity)
        user_bias = float(rng.normal(0.0, 0.4))

        k = max(1, int(rng.poisson(avg_assignments_per_user)))

        for _ in range(k):
            camp = _pick_campaign_for_segment(seg, rng)

            # Sample offer value/cost from campaign ranges
            v_lo, v_hi = camp.value_range
            offer_value = float(rng.uniform(v_lo, v_hi)) if v_hi > v_lo else float(v_lo)
            c_lo, c_hi = camp.cost_range
            estimated_cost = float(rng.uniform(c_lo, c_hi))

            # RFM per assignment (redraw to simulate assignments over time)
            recency, freq_90d, monetary_90d = _draw_segment_rfm(seg, rng)

            # Historical totals (loosely tied to frequency/monetary)
            txn_count_before = int(max(0, rng.poisson(freq_90d * 1.4)))
            spend_before = float(max(0.0, monetary_90d * rng.uniform(0.6, 1.4)))

            # Exposure funnel — exposure strongly gates redemption
            p_exposed = 0.85 if seg in {"active", "vip"} else (0.75 if seg == "new" else 0.6)
            is_exposed = int(rng.random() < p_exposed)

            # Click probability given exposure
            logit_click = (
                -1.0
                + 0.6 * (seg == "vip")
                + 0.3 * (seg == "active")
                + 0.2 * (seg == "new")
                + camp.segment_bias.get(seg, 0.0) * 0.5
                + 0.02 * offer_value
                - 0.01 * recency
                + user_bias * 0.5
            )
            p_click = float(_sigmoid(logit_click)) * is_exposed
            is_clicked = int(rng.random() < p_click)

            # Final redemption logit
            logit = (
                base_logit_shift
                + camp.intrinsic
                + camp.segment_bias.get(seg, 0.0)
                + user_bias
                # Exposure / click funnel
                + (2.5 if is_exposed else -2.5)
                + (1.8 if is_clicked else -0.5)
                # Recency decay — fresher users convert better
                - 0.012 * recency
                # Frequency fatigue for high-freq users on short cooldown
                - 0.04 * max(0, freq_90d - 4)
                + (-0.03 * (camp.cooldown_days < 10) * freq_90d)
                # Offer value (saturating via log1p) — only meaningful for discount
                + (0.15 * np.log1p(offer_value) if camp.offer_type == "discount" else 0.0)
                + (0.25 * np.log1p(offer_value) if camp.offer_type == "points_multiplier" else 0.0)
                # Monetary — richer users convert more on points/gift
                + (0.00002 * monetary_90d if camp.offer_type != "discount" else 0.0)
                # Scope noise
                + (0.1 if camp.target_scope in {"product_type", "product_id"} else 0.0)
                # Gaussian noise
                + float(rng.normal(0.0, 0.6))
            )

            p_redeem = float(_sigmoid(logit))
            # Hard gate: never redeem if not exposed at all
            if is_exposed == 0:
                p_redeem *= 0.01
            label = int(rng.random() < p_redeem)

            assigned_at = start + timedelta(
                seconds=float(rng.uniform(0, horizon_days * 86400))
            )

            rows.append(
                dict(
                    assignment_id=assignment_id,
                    user_id=user_id,
                    assigned_at=assigned_at,
                    label_redeemed=label,
                    campaign_name=camp.name,
                    offer_type=camp.offer_type,
                    target_scope=camp.target_scope,
                    offer_value=offer_value,
                    estimated_cost=estimated_cost,
                    cooldown_days=camp.cooldown_days,
                    expires_in_days=camp.expires_in_days,
                    is_exposed=is_exposed,
                    is_clicked=is_clicked,
                    recency_days=recency,
                    frequency_90d=freq_90d,
                    monetary_90d=round(monetary_90d, 2),
                    txn_count_before=txn_count_before,
                    spend_before=round(spend_before, 2),
                    _segment=seg,  # debug only, dropped below
                )
            )
            assignment_id += 1

    df = pd.DataFrame(rows)
    df = df.sort_values("assigned_at").reset_index(drop=True)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", required=True, help="Where to write offer_train.parquet")
    ap.add_argument("--n_users", type=int, default=3000)
    ap.add_argument("--avg_assignments_per_user", type=float, default=6.0)
    ap.add_argument("--horizon_days", type=int, default=180)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--base_logit_shift",
        type=float,
        default=-4.2,
        help="Shift base redemption logit to control positive rate (~-4.2 gives ~26% positives).",
    )
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    df = generate(
        n_users=args.n_users,
        avg_assignments_per_user=args.avg_assignments_per_user,
        horizon_days=args.horizon_days,
        seed=args.seed,
        base_logit_shift=args.base_logit_shift,
    )

    # Print diagnostic stats before dropping debug cols
    print(f"rows: {len(df)}")
    print(f"positive_rate: {df['label_redeemed'].mean():.4f}")
    print("positive_rate by segment:")
    print(df.groupby("_segment")["label_redeemed"].mean().round(4).to_string())
    print("positive_rate by offer_type:")
    print(df.groupby("offer_type")["label_redeemed"].mean().round(4).to_string())
    print("positive_rate by campaign:")
    print(df.groupby("campaign_name")["label_redeemed"].mean().round(4).to_string())
    print("exposure / click rates:")
    print(f"  is_exposed: {df['is_exposed'].mean():.4f}")
    print(f"  is_clicked: {df['is_clicked'].mean():.4f}")

    out_df = df.drop(columns=["_segment"])
    out_path = os.path.join(args.out_dir, "offer_train.parquet")
    out_df.to_parquet(out_path, index=False)
    print(f"wrote: {out_path}")


if __name__ == "__main__":
    main()
