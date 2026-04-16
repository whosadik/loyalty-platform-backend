"""
Generate synthetic roadmap training dataset with user profile features.

Why this exists:
  The existing v4 dataset knows only WHAT users bought (purchase sequence + owned counts).
  It does NOT know WHO the user is — no skin_type, hair_type, goals, concerns.
  This means the model cannot give truly personalized recommendations.

  This generator creates synthetic users with diverse beauty profiles and simulates
  their purchase journeys following domain-aware logic:
    - oily skin  → toner/serum prioritised early
    - dry skin   → moisturizer/essence earlier
    - curly hair → leave_in/hair_oil over scalp_serum
    - scalp issue → scalp_serum promoted
    - high budget → optional tail steps included
    - anti-aging goal → serum → eye_cream progression
    etc.

  The resulting dataset teaches the model to rank candidates differently
  based on who the person is, not just what they've bought.

Output:
  data/ml/roadmap_nextstep_v5_profile/
    dataset.parquet   — training episodes
    metadata.json     — feature schema, baselines, candidate types
    splits.json       — train/val/test user_id lists

Usage:
  python manage.py generate_roadmap_profile_dataset
  python manage.py generate_roadmap_profile_dataset --n-users 5000 --seed 42
  python manage.py generate_roadmap_profile_dataset --output-dir data/ml/my_dataset

Then train:
  python manage.py train_roadmap_nextstep_model_v4 \\
      --data-dir data/ml/roadmap_nextstep_v5_profile \\
      --model-dir models/roadmap_next_step_v5_profile \\
      --model-version roadmap_next_step_v5_profile
"""
from __future__ import annotations

import json
import random
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from django.core.management.base import BaseCommand, CommandError

try:
    import numpy as np
    import pandas as pd
except ImportError:  # pragma: no cover
    np = None
    pd = None


# ---------------------------------------------------------------------------
# Chain definitions  (canonical order = default priority)
# ---------------------------------------------------------------------------

CHAIN_BY_CATEGORY: dict[str, list[str]] = {
    "skincare": ["cleanser", "serum", "moisturizer", "spf", "toner", "mask", "eye_cream", "essence"],
    "haircare": ["shampoo", "conditioner", "hair_mask", "hair_oil", "scalp_serum", "leave_in"],
    "makeup":   ["foundation", "mascara", "blush", "lipstick", "eyeshadow", "primer", "setting_spray"],
    "fragrance": ["warm_day", "warm_evening", "cold_day", "cold_evening"],
}

# popularity prior — used as baseline and for negative candidate weights
POPULARITY_BY_CATEGORY: dict[str, dict[str, float]] = {
    "skincare": {
        "cleanser": 0.158, "serum": 0.147, "moisturizer": 0.133, "spf": 0.115,
        "toner": 0.114, "mask": 0.117, "eye_cream": 0.109, "essence": 0.108,
    },
    "haircare": {
        "shampoo": 0.202, "conditioner": 0.188, "hair_mask": 0.159,
        "hair_oil": 0.150, "scalp_serum": 0.148, "leave_in": 0.152,
    },
    "makeup": {
        "foundation": 0.189, "mascara": 0.170, "blush": 0.143,
        "lipstick": 0.129, "eyeshadow": 0.119, "primer": 0.130, "setting_spray": 0.121,
    },
    "fragrance": {
        "warm_day": 0.317, "warm_evening": 0.223, "cold_day": 0.196, "cold_evening": 0.264,
    },
}


# ---------------------------------------------------------------------------
# Profile option spaces
# ---------------------------------------------------------------------------

SKIN_TYPES      = ["oily", "dry", "combination", "sensitive", "normal"]
HAIR_TYPES      = ["straight", "wavy", "curly", "coily"]
SCALP_TYPES     = ["normal", "oily", "dry", "sensitive"]
HAIR_THICKNESS  = ["thin", "medium", "thick"]
BUDGETS         = ["low", "medium", "high"]
MAKEUP_FINISHES = ["matte", "dewy", "natural", "satin"]
FRAG_INTENSITIES= ["soft", "moderate", "strong"]

SKINCARE_GOALS_POOL = [
    "hydration", "brightening", "anti_aging", "pore_minimizing",
    "acne_control", "redness_reduction", "firming",
]
HAIR_CONCERNS_POOL = [
    "frizz", "dryness", "breakage", "volume", "scalp_oiliness",
    "dandruff", "color_protection", "heat_damage",
]
AVOID_FLAGS_POOL = ["fragrance_free", "alcohol_free", "paraben_free", "silicone_free"]


# ---------------------------------------------------------------------------
# Profile-to-sequence priority maps
# These define which product types are pushed earlier given a profile.
# Lower score = higher priority = comes earlier in the personalised chain.
# ---------------------------------------------------------------------------

def _skincare_priority(profile: dict) -> dict[str, float]:
    """Return a priority score per product_type for skincare."""
    base = {t: float(i) for i, t in enumerate(CHAIN_BY_CATEGORY["skincare"])}
    skin_type   = profile.get("skin_type", "normal")
    goals       = set(profile.get("goals", []))
    avoid_flags = set(profile.get("avoid_flags", []))

    if skin_type == "oily":
        # toner and serum come before moisturiser
        base["toner"]      -= 2.5
        base["serum"]      -= 1.5
        base["mask"]       -= 0.8
        base["moisturizer"] += 0.5
    elif skin_type == "dry":
        base["moisturizer"] -= 2.0
        base["essence"]    -= 1.0
        base["toner"]      += 1.0
    elif skin_type == "sensitive":
        # minimal routine: cleanser → moisturizer → spf
        base["serum"]      += 1.0
        base["toner"]      += 2.0
        base["mask"]       += 2.0
        base["eye_cream"]  += 1.5
    elif skin_type == "combination":
        base["toner"]      -= 1.0
        base["serum"]      -= 0.5

    if "anti_aging" in goals:
        base["serum"]      -= 1.0
        base["eye_cream"]  -= 1.5
        base["essence"]    -= 0.5

    if "brightening" in goals:
        base["serum"]      -= 1.2
        base["spf"]        -= 0.8

    if "acne_control" in goals:
        base["toner"]      -= 1.5
        base["mask"]       -= 1.0

    if "hydration" in goals:
        base["moisturizer"] -= 0.8
        base["essence"]    -= 0.6

    if profile.get("budget") == "low":
        # skip optional tail
        base["mask"]       += 3.0
        base["eye_cream"]  += 3.0
        base["essence"]    += 3.0

    return base


def _haircare_priority(profile: dict) -> dict[str, float]:
    """Return a priority score per product_type for haircare."""
    base = {t: float(i) for i, t in enumerate(CHAIN_BY_CATEGORY["haircare"])}
    hair_type   = profile.get("hair_type", "straight")
    scalp_type  = profile.get("scalp_type", "normal")
    thickness   = profile.get("hair_thickness", "medium")
    concerns    = set(profile.get("hair_concerns", []))

    if hair_type in ("curly", "coily"):
        base["leave_in"]    -= 2.5
        base["hair_oil"]    -= 1.5
        base["hair_mask"]   -= 0.5
        base["scalp_serum"] += 1.0

    elif hair_type == "wavy":
        base["leave_in"]    -= 1.5
        base["hair_oil"]    -= 0.5

    if scalp_type == "oily":
        base["scalp_serum"] -= 2.0
        base["shampoo"]     -= 0.5
    elif scalp_type == "dry":
        base["scalp_serum"] -= 1.5
        base["hair_oil"]    -= 0.8

    if scalp_type == "sensitive":
        base["scalp_serum"] -= 1.0

    if "dandruff" in concerns or "scalp_oiliness" in concerns:
        base["scalp_serum"] -= 2.5

    if "dryness" in concerns or "breakage" in concerns:
        base["hair_mask"]   -= 1.5
        base["hair_oil"]    -= 1.0

    if thickness == "thick":
        base["hair_oil"]    -= 0.5
        base["hair_mask"]   -= 0.3

    if profile.get("budget") == "low":
        base["scalp_serum"] += 3.0
        base["leave_in"]    += 2.0

    return base


def _makeup_priority(profile: dict) -> dict[str, float]:
    """Return a priority score per product_type for makeup."""
    base = {t: float(i) for i, t in enumerate(CHAIN_BY_CATEGORY["makeup"])}
    finish = profile.get("makeup_finish", "natural")

    if finish in ("matte", "satin"):
        base["primer"]      -= 1.5
        base["setting_spray"] -= 1.0

    if profile.get("budget") == "low":
        base["primer"]      += 2.0
        base["setting_spray"] += 2.0
        base["eyeshadow"]   += 1.5

    return base


def _fragrance_priority(profile: dict) -> dict[str, float]:
    """Return a priority score per product_type for fragrance."""
    base = {t: float(i) for i, t in enumerate(CHAIN_BY_CATEGORY["fragrance"])}
    intensity = profile.get("fragrance_intensity", "moderate")

    if intensity == "strong":
        base["warm_evening"] -= 1.0
        base["cold_evening"] -= 0.5
    elif intensity == "soft":
        base["warm_day"]     -= 1.0
        base["cold_day"]     -= 0.5

    return base


PRIORITY_FN = {
    "skincare":  _skincare_priority,
    "haircare":  _haircare_priority,
    "makeup":    _makeup_priority,
    "fragrance": _fragrance_priority,
}


def _personalised_chain(category: str, profile: dict, rng: random.Random) -> list[str]:
    """
    Return the canonical chain reordered by profile priority.
    Adds a small noise so not all users with same profile buy in identical order.
    """
    chain = CHAIN_BY_CATEGORY[category]
    scores = PRIORITY_FN[category](profile)
    # add small noise
    noisy = {t: scores.get(t, float(i)) + rng.gauss(0, 0.3) for i, t in enumerate(chain)}
    return sorted(chain, key=lambda t: noisy[t])


# ---------------------------------------------------------------------------
# Synthetic user generation
# ---------------------------------------------------------------------------

def _sample_profile(category: str, rng: random.Random) -> dict:
    profile: dict[str, Any] = {
        "budget": rng.choices(BUDGETS, weights=[0.25, 0.50, 0.25])[0],
    }

    if category in ("skincare", "all"):
        profile["skin_type"]   = rng.choice(SKIN_TYPES)
        n_goals = rng.choices([0, 1, 2, 3, 4], weights=[0.05, 0.25, 0.35, 0.25, 0.10])[0]
        profile["goals"]       = rng.sample(SKINCARE_GOALS_POOL, min(n_goals, len(SKINCARE_GOALS_POOL)))
        n_avoid = rng.choices([0, 1, 2], weights=[0.50, 0.35, 0.15])[0]
        profile["avoid_flags"] = rng.sample(AVOID_FLAGS_POOL, min(n_avoid, len(AVOID_FLAGS_POOL)))

    if category in ("haircare", "all"):
        profile["hair_type"]      = rng.choice(HAIR_TYPES)
        profile["scalp_type"]     = rng.choice(SCALP_TYPES)
        profile["hair_thickness"] = rng.choice(HAIR_THICKNESS)
        n_concerns = rng.choices([0, 1, 2, 3], weights=[0.15, 0.35, 0.35, 0.15])[0]
        profile["hair_concerns"]  = rng.sample(HAIR_CONCERNS_POOL, min(n_concerns, len(HAIR_CONCERNS_POOL)))
        profile["has_scalp_objective"] = int(
            profile.get("scalp_type") in ("oily", "sensitive") or
            any(c in ("dandruff", "scalp_oiliness") for c in profile.get("hair_concerns", []))
        )

    if category in ("makeup", "all"):
        profile["makeup_finish"]  = rng.choice(MAKEUP_FINISHES)

    if category in ("fragrance", "all"):
        profile["fragrance_intensity"] = rng.choice(FRAG_INTENSITIES)

    return profile


def _simulate_journey(
    category: str,
    profile: dict,
    rng: random.Random,
    min_steps: int = 2,
    max_steps: int = 7,
) -> list[str]:
    """
    Simulate a user's purchase journey for one category.
    Returns an ordered list of product_types (what they bought, in order).
    """
    personalised = _personalised_chain(category, profile, rng)

    # How many steps does this user complete?
    budget = profile.get("budget", "medium")
    max_cap = {"low": 4, "medium": 6, "high": 7}[budget]
    n_steps = rng.randint(min_steps, min(max_steps, max_cap, len(personalised)))

    return personalised[:n_steps]


# ---------------------------------------------------------------------------
# Episode builder
# ---------------------------------------------------------------------------

def _build_episodes(
    user_id: int,
    category: str,
    profile: dict,
    journey: list[str],
    t0_base: datetime,
    episode_id_start: int,
    rng: random.Random,
) -> list[dict]:
    """
    Convert a purchase journey into training episodes.

    For each step k in [1, len(journey)-1]:
      - context = journey[0..k-1]  (what was bought before)
      - label   = journey[k]       (what was bought at step k)
      - candidates = all types for category
      - For each candidate: y = 1 if candidate == label else 0
    """
    all_types = CHAIN_BY_CATEGORY[category]
    pop_map   = POPULARITY_BY_CATEGORY[category]

    episodes: list[dict] = []
    owned_counts: dict[str, int] = defaultdict(int)  # running owned count per type/category

    for step_idx in range(1, len(journey)):
        context  = journey[:step_idx]
        label    = journey[step_idx]
        ep_id    = episode_id_start + step_idx - 1

        # temporal: stagger purchases by 14-45 days each
        t0 = t0_base + timedelta(days=sum(rng.randint(14, 45) for _ in range(step_idx)))
        month = t0.month
        dow   = t0.weekday()
        days_since = rng.randint(14, 90) if context else -1
        tx_count   = len(context)
        tx_amount  = float(sum(rng.uniform(500, 3000) for _ in context))

        # owned slot counts for fragrance
        owned_slots = {slot: 0 for slot in ["warm_day", "warm_evening", "cold_day", "cold_evening"]}
        if category == "fragrance":
            for pt in context:
                if pt in owned_slots:
                    owned_slots[pt] += 1

        # last 5 product types across all categories (simulate cross-category history)
        last_types    = (context + ["__none__"] * 5)[-5:][::-1][:5]
        last_cats     = [category if t != "__none__" else "__none__" for t in last_types]

        # owned counts per (category, product_type)
        owned_by_type: dict[str, int] = defaultdict(int)
        for pt in context:
            owned_by_type[f"{category}__{pt}"] += 1

        for candidate in all_types:
            is_positive = int(candidate == label)
            position    = all_types.index(candidate)
            popularity  = pop_map.get(candidate, 0.1)

            row: dict[str, Any] = {
                # identifiers
                "episode_id":   ep_id,
                "group_id":     ep_id,
                "user_id":      user_id,
                "category":     category,
                "t0_utc":       t0.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "split":        "",        # filled later
                "label":        label,
                "candidate_type": candidate,
                "y":            is_positive,

                # temporal + behavioural features (same as v4)
                "candidate_is_fragrance_slot":       int(category == "fragrance"),
                "candidate_position_in_chain":       position,
                "candidate_popularity_in_train":     round(popularity, 8),
                "month_of_year":                     month,
                "day_of_week":                       dow,
                "days_since_last_purchase_in_category": days_since,
                "tx_count_90d_category":             tx_count,
                "tx_amount_90d_category":            round(tx_amount, 2),

                # fragrance slots
                "owned_slot_warm_day":    owned_slots["warm_day"],
                "owned_slot_warm_evening": owned_slots["warm_evening"],
                "owned_slot_cold_day":    owned_slots["cold_day"],
                "owned_slot_cold_evening": owned_slots["cold_evening"],

                # last 5 purchases
                "last1_product_type": last_types[0], "last1_category": last_cats[0],
                "last2_product_type": last_types[1], "last2_category": last_cats[1],
                "last3_product_type": last_types[2], "last3_category": last_cats[2],
                "last4_product_type": last_types[3], "last4_category": last_cats[3],
                "last5_product_type": last_types[4], "last5_category": last_cats[4],

                # owned counts per (category, product_type) — all categories
                "owned_count__fragrance__perfume_oil": 0,
                "owned_count__fragrance__body_mist":   0,
                "owned_count__fragrance__edp":         0,
                "owned_count__fragrance__edt":         0,
                "owned_count__haircare__leave_in":     0,
                "owned_count__haircare__scalp_serum":  0,
                "owned_count__haircare__hair_mask":    0,
                "owned_count__haircare__conditioner":  0,
                "owned_count__haircare__shampoo":      0,
                "owned_count__haircare__hair_oil":     0,
                "owned_count__makeup__foundation":     0,
                "owned_count__makeup__mascara":        0,
                "owned_count__makeup__primer":         0,
                "owned_count__makeup__setting_spray":  0,
                "owned_count__makeup__lipstick":       0,
                "owned_count__makeup__blush":          0,
                "owned_count__makeup__eyeshadow":      0,
                "owned_count__skincare__serum":        0,
                "owned_count__skincare__moisturizer":  0,
                "owned_count__skincare__mask":         0,
                "owned_count__skincare__essence":      0,
                "owned_count__skincare__eye_cream":    0,
                "owned_count__skincare__cleanser":     0,
                "owned_count__skincare__toner":        0,
                "owned_count__skincare__spf":          0,

                # === NEW: user profile features (names match content_features.py exactly) ===
                "profile_skin_type":                  profile.get("skin_type", "__none__"),
                "profile_hair_type":                  profile.get("hair_type", "__none__"),
                "profile_scalp_type":                 profile.get("scalp_type", "__none__"),
                "profile_hair_thickness":             profile.get("hair_thickness", "__none__"),
                "profile_makeup_finish_pref_primary": profile.get("makeup_finish", "__none__"),
                "profile_fragrance_intensity_pref":   profile.get("fragrance_intensity", "__none__"),
                "profile_budget":                     profile.get("budget", "__none__"),
                "profile_goals_count":                len(profile.get("goals", [])),
                "profile_hair_concerns_count":        len(profile.get("hair_concerns", [])),
                "profile_avoid_flags_count":          len(profile.get("avoid_flags", [])),
                "profile_has_scalp_objective":        int(profile.get("has_scalp_objective", 0)),
            }

            # fill owned counts from context
            for pt in context:
                col = f"owned_count__{category}__{pt}"
                if col in row:
                    row[col] = owned_by_type.get(f"{category}__{pt}", 0)

            episodes.append(row)

    return episodes


# ---------------------------------------------------------------------------
# Dataset column schema
# ---------------------------------------------------------------------------

CATEGORICAL_FEATURES = [
    "category",
    "candidate_type",
    "last1_product_type", "last2_product_type", "last3_product_type",
    "last4_product_type", "last5_product_type",
    "last1_category", "last2_category", "last3_category",
    "last4_category", "last5_category",
    # profile categoricals — names match content_features.py exactly
    "profile_skin_type",
    "profile_hair_type",
    "profile_scalp_type",
    "profile_hair_thickness",
    "profile_makeup_finish_pref_primary",   # matches content_features.py line 483
    "profile_fragrance_intensity_pref",     # matches content_features.py line 491
    "profile_budget",
]

NUMERIC_FEATURES = [
    "month_of_year", "day_of_week",
    "days_since_last_purchase_in_category",
    "tx_count_90d_category", "tx_amount_90d_category",
    "owned_slot_warm_day", "owned_slot_warm_evening",
    "owned_slot_cold_day", "owned_slot_cold_evening",
    "candidate_is_fragrance_slot",
    "candidate_position_in_chain",
    "candidate_popularity_in_train",
    "owned_count__fragrance__perfume_oil",
    "owned_count__fragrance__body_mist",
    "owned_count__fragrance__edp",
    "owned_count__fragrance__edt",
    "owned_count__haircare__leave_in",
    "owned_count__haircare__scalp_serum",
    "owned_count__haircare__hair_mask",
    "owned_count__haircare__conditioner",
    "owned_count__haircare__shampoo",
    "owned_count__haircare__hair_oil",
    "owned_count__makeup__foundation",
    "owned_count__makeup__mascara",
    "owned_count__makeup__primer",
    "owned_count__makeup__setting_spray",
    "owned_count__makeup__lipstick",
    "owned_count__makeup__blush",
    "owned_count__makeup__eyeshadow",
    "owned_count__skincare__serum",
    "owned_count__skincare__moisturizer",
    "owned_count__skincare__mask",
    "owned_count__skincare__essence",
    "owned_count__skincare__eye_cream",
    "owned_count__skincare__cleanser",
    "owned_count__skincare__toner",
    "owned_count__skincare__spf",
    # profile numerics — names match content_features.py exactly
    "profile_goals_count",
    "profile_hair_concerns_count",
    "profile_avoid_flags_count",
    "profile_has_scalp_objective",
]

FEATURE_COLUMNS = CATEGORICAL_FEATURES + NUMERIC_FEATURES

OWNED_FEATURE_COLUMNS = [c for c in NUMERIC_FEATURES if c.startswith("owned_count__")]
OWNED_FEATURE_MAP = {
    col: {
        "category":     col.split("__")[1],
        "product_type": col.split("__")[2],
    }
    for col in OWNED_FEATURE_COLUMNS
}


# ---------------------------------------------------------------------------
# Management command
# ---------------------------------------------------------------------------

class Command(BaseCommand):
    help = (
        "Generate a synthetic roadmap training dataset with user profile features. "
        "Produces data/ml/roadmap_nextstep_v5_profile/ ready for train_roadmap_nextstep_model_v4."
    )

    def add_arguments(self, parser):
        parser.add_argument(
            "--n-users", type=int, default=4000,
            help="Total synthetic users to generate (default 4000).",
        )
        parser.add_argument(
            "--seed", type=int, default=42,
            help="Random seed for reproducibility.",
        )
        parser.add_argument(
            "--output-dir", type=str, default="data/ml/roadmap_nextstep_v5_profile",
            help="Output directory for dataset + metadata.",
        )
        parser.add_argument(
            "--val-frac", type=float, default=0.15,
            help="Fraction of users for validation split.",
        )
        parser.add_argument(
            "--test-frac", type=float, default=0.15,
            help="Fraction of users for test split.",
        )
        parser.add_argument(
            "--min-steps", type=int, default=2,
            help="Minimum purchase steps per user journey.",
        )
        parser.add_argument(
            "--max-steps", type=int, default=7,
            help="Maximum purchase steps per user journey.",
        )
        parser.add_argument(
            "--category-weights", type=str,
            default="skincare:0.35,haircare:0.25,makeup:0.25,fragrance:0.15",
            help="Category distribution as 'cat:weight,...'",
        )

    def handle(self, *args, **options):
        if pd is None or np is None:
            raise CommandError("pandas and numpy are required. Run: pip install pandas numpy")

        n_users    = int(options["n_users"])
        seed       = int(options["seed"])
        val_frac   = float(options["val_frac"])
        test_frac  = float(options["test_frac"])
        min_steps  = int(options["min_steps"])
        max_steps  = int(options["max_steps"])
        output_dir = self._resolve_output_dir(str(options["output_dir"]))

        cat_weights = self._parse_category_weights(str(options["category_weights"]))

        rng = random.Random(seed)
        np_rng = np.random.RandomState(seed)

        self.stdout.write(
            f"[generate_roadmap_profile_dataset] "
            f"n_users={n_users} seed={seed} output={output_dir}"
        )

        # ----------------------------------------------------------------
        # 1. Generate users: assign category + profile
        # ----------------------------------------------------------------
        categories_list = list(cat_weights.keys())
        weights_list    = list(cat_weights.values())

        user_categories: list[str] = rng.choices(categories_list, weights=weights_list, k=n_users)
        user_profiles:   list[dict] = [
            _sample_profile(cat, rng) for cat in user_categories
        ]

        self.stdout.write(
            f"[generate_roadmap_profile_dataset] category distribution: "
            + ", ".join(
                f"{c}={sum(1 for x in user_categories if x == c)}"
                for c in categories_list
            )
        )

        # ----------------------------------------------------------------
        # 2. Generate purchase journeys + episodes
        # ----------------------------------------------------------------
        all_rows: list[dict] = []
        episode_counter = 1
        t_now = datetime(2026, 3, 1, tzinfo=timezone.utc)

        for user_id, (category, profile) in enumerate(
            zip(user_categories, user_profiles), start=1
        ):
            # random t0 base: 90–365 days ago
            days_ago = rng.randint(90, 365)
            t0_base  = t_now - timedelta(days=days_ago)

            journey = _simulate_journey(category, profile, rng, min_steps, max_steps)
            if len(journey) < 2:
                continue

            episodes = _build_episodes(
                user_id=user_id,
                category=category,
                profile=profile,
                journey=journey,
                t0_base=t0_base,
                episode_id_start=episode_counter,
                rng=rng,
            )
            episode_counter += len(journey) - 1
            all_rows.extend(episodes)

        self.stdout.write(
            f"[generate_roadmap_profile_dataset] "
            f"total rows={len(all_rows)}"
        )

        if not all_rows:
            raise CommandError("No rows generated. Check --min-steps / --max-steps / --n-users.")

        df = pd.DataFrame(all_rows)

        # ----------------------------------------------------------------
        # 3. Train / val / test split by user_id
        # ----------------------------------------------------------------
        all_user_ids = sorted(df["user_id"].unique().tolist())
        np_rng.shuffle(all_user_ids)

        n_total = len(all_user_ids)
        n_val   = max(1, int(n_total * val_frac))
        n_test  = max(1, int(n_total * test_frac))
        n_train = n_total - n_val - n_test

        train_users = set(all_user_ids[:n_train])
        val_users   = set(all_user_ids[n_train: n_train + n_val])
        test_users  = set(all_user_ids[n_train + n_val:])

        def _assign_split(uid: int) -> str:
            if uid in val_users:
                return "val"
            if uid in test_users:
                return "test"
            return "train"

        df["split"] = df["user_id"].map(_assign_split)

        train_rows = int((df["split"] == "train").sum())
        val_rows   = int((df["split"] == "val").sum())
        test_rows  = int((df["split"] == "test").sum())
        self.stdout.write(
            f"[generate_roadmap_profile_dataset] "
            f"train={train_rows} val={val_rows} test={test_rows} rows"
        )

        # ----------------------------------------------------------------
        # 4. Compute baselines (popularity + markov) per split
        # ----------------------------------------------------------------
        baselines = self._compute_baselines(df)

        # ----------------------------------------------------------------
        # 5. Class distribution
        # ----------------------------------------------------------------
        class_dist: dict[str, dict[str, int]] = {}
        for split_name in ["train", "val", "test"]:
            split_df = df[(df["split"] == split_name) & (df["y"] == 1)]
            dist = split_df["label"].value_counts().to_dict()
            class_dist[split_name] = {k: int(v) for k, v in dist.items()}

        # ----------------------------------------------------------------
        # 6. Save
        # ----------------------------------------------------------------
        output_dir.mkdir(parents=True, exist_ok=True)
        parquet_path = output_dir / "dataset.parquet"
        df.to_parquet(parquet_path, index=False)
        self.stdout.write(f"[generate_roadmap_profile_dataset] saved {parquet_path}")

        # splits.json
        splits = {
            "train_user_ids": sorted(int(u) for u in train_users),
            "val_user_ids":   sorted(int(u) for u in val_users),
            "test_user_ids":  sorted(int(u) for u in test_users),
        }
        splits_path = output_dir / "splits.json"
        splits_path.write_text(json.dumps(splits, indent=2), encoding="utf-8")

        # metadata.json
        generated_at = datetime.now(tz=timezone.utc).isoformat().replace("+00:00", "Z")
        positive_rows = int((df["y"] == 1).sum())
        metadata = {
            "version": "v5_profile",
            "generated_at_utc": generated_at,
            "generator_command": "generate_roadmap_profile_dataset",
            "generator_params": {
                "n_users": n_users,
                "seed": seed,
                "min_steps": min_steps,
                "max_steps": max_steps,
                "category_weights": cat_weights,
            },
            "dataset_format": "parquet",
            "dataset_file": str(parquet_path),
            "rows_total": int(len(df)),
            "episodes_total": int(df["episode_id"].nunique()),
            "groups_total":   int(df["group_id"].nunique()),
            "positive_rows":  positive_rows,
            "positives":      positive_rows,
            "none_count": 0,
            "none_rate":  0.0,
            "label_outside_candidate_set": 0,
            "leakage_assertions": {
                "features_only_use_transactions_lte_t0": True,
                "status": "by_construction",
            },
            "class_distribution": class_dist,
            "candidate_types_by_category":  CHAIN_BY_CATEGORY,
            "rules_chain_by_category":      CHAIN_BY_CATEGORY,
            "top_popularity_by_category": {
                cat: sorted(pop.keys(), key=lambda k: -pop[k])
                for cat, pop in POPULARITY_BY_CATEGORY.items()
            },
            "candidate_popularity_in_train_by_category": POPULARITY_BY_CATEGORY,
            "owned_feature_columns": OWNED_FEATURE_COLUMNS,
            "owned_feature_map":     OWNED_FEATURE_MAP,
            "feature_columns":       FEATURE_COLUMNS,
            "categorical_features":  CATEGORICAL_FEATURES,
            "numeric_features":      NUMERIC_FEATURES,
            "new_profile_features": {
                "categorical": [
                    "profile_skin_type", "profile_hair_type", "profile_scalp_type",
                    "profile_hair_thickness", "profile_makeup_finish",
                    "profile_fragrance_intensity", "profile_budget",
                ],
                "numeric": [
                    "profile_goals_count", "profile_hair_concerns_count",
                    "profile_avoid_flags_count", "profile_has_scalp_objective",
                ],
                "description": (
                    "User beauty profile features added on top of v4 sequence features. "
                    "These allow the model to personalise next-step predictions based on "
                    "who the user is, not only what they have bought."
                ),
            },
            "baselines": baselines,
        }
        meta_path = output_dir / "metadata.json"
        meta_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

        self.stdout.write(f"[generate_roadmap_profile_dataset] saved {meta_path}")
        self.stdout.write(f"[generate_roadmap_profile_dataset] saved {splits_path}")
        self.stdout.write(
            f"[generate_roadmap_profile_dataset] done — "
            f"{len(df):,} rows / {df['episode_id'].nunique():,} episodes / "
            f"{len(all_user_ids):,} users"
        )
        self.stdout.write(
            f"\nNext step — train the model:\n"
            f"  python manage.py train_roadmap_nextstep_model_v4 \\\n"
            f"      --data-dir {output_dir} \\\n"
            f"      --model-dir models/roadmap_next_step_v5_profile \\\n"
            f"      --model-version roadmap_next_step_v5_profile\n"
        )

    # ----------------------------------------------------------------
    # Helpers
    # ----------------------------------------------------------------

    def _resolve_output_dir(self, raw: str) -> Path:
        candidate = Path(raw).expanduser()
        if candidate.is_absolute():
            return candidate
        repo_root = Path(__file__).resolve().parents[4]
        return (repo_root / candidate).resolve()

    def _parse_category_weights(self, raw: str) -> dict[str, float]:
        result: dict[str, float] = {}
        for part in raw.split(","):
            part = part.strip()
            if ":" not in part:
                continue
            cat, w = part.split(":", 1)
            result[cat.strip()] = float(w.strip())
        if not result:
            raise CommandError("--category-weights must be like 'skincare:0.35,haircare:0.25,...'")
        total = sum(result.values())
        return {k: v / total for k, v in result.items()}

    def _compute_baselines(self, df: "pd.DataFrame") -> dict:
        """Compute popularity and markov baselines per split."""
        from collections import Counter

        def _recall_ndcg(df_split: "pd.DataFrame", rank_fn) -> dict:
            episodes = df_split.groupby("episode_id")
            hits1 = hits3 = hits5 = ndcg = n_pos = 0
            for _, grp in episodes:
                positives = grp[grp["y"] == 1]
                if positives.empty:
                    continue
                label = str(positives.iloc[0]["label"])
                candidates = grp["candidate_type"].tolist()
                ranked = rank_fn(candidates, grp)
                try:
                    rank = ranked.index(label) + 1
                except ValueError:
                    continue
                n_pos += 1
                if rank == 1:
                    hits1 += 1
                if rank <= 3:
                    hits3 += 1
                if rank <= 5:
                    hits5 += 1
                    import math
                    ndcg += 1.0 / math.log2(rank + 1)
            denom = max(1, n_pos)
            return {
                "rows": int(len(df_split)),
                "positive_episodes": n_pos,
                "recall_at_1": round(hits1 / denom, 6),
                "recall_at_3": round(hits3 / denom, 6),
                "recall_at_5": round(hits5 / denom, 6),
                "ndcg_at_5":   round(ndcg  / denom, 6),
            }

        train_df = df[df["split"] == "train"]

        # popularity: rank by candidate_popularity_in_train descending
        def pop_rank(candidates, grp):
            pop = grp.set_index("candidate_type")["candidate_popularity_in_train"].to_dict()
            return sorted(candidates, key=lambda c: -pop.get(c, 0.0))

        # markov: rank by how often each type appears as label in train
        label_counts = Counter(train_df[train_df["y"] == 1]["label"].tolist())

        def markov_rank(candidates, grp):
            return sorted(candidates, key=lambda c: -label_counts.get(c, 0))

        result: dict = {"splits": {}}
        for split_name in ["val", "test"]:
            split_df = df[df["split"] == split_name]
            if split_df.empty:
                continue
            result["splits"][split_name] = {
                "popularity": _recall_ndcg(split_df, pop_rank),
                "markov":     _recall_ndcg(split_df, markov_rank),
            }
        return result
