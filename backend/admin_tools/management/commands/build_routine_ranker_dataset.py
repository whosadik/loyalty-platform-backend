"""Build a training dataset for the skincare routine ranker.

Positive signal: a user owns a product that fits a routine step.
Negative signal: other in-stock candidates for the same step that the user does
not own. Each (user, step) combination becomes a ranking group.
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from django.core.management.base import BaseCommand, CommandError

try:
    import pandas as pd
except Exception:  # pragma: no cover
    pd = None


ROUTINE_STEPS = ("cleanser", "serum", "moisturizer", "spf")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _resolve_output_dir(raw_path: str) -> Path:
    candidate = Path(str(raw_path)).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    return (_repo_root() / candidate).resolve()


def _product_step_key(product: dict[str, Any]) -> str | None:
    key = product.get("product_type") or product.get("step")
    if not key:
        return None
    key = str(key).strip().lower()
    return key if key in ROUTINE_STEPS else None


def _goal_concern_match_count(goals: list[str], concerns: list[str]) -> int:
    if not goals or not concerns:
        return 0
    return len(set(goals) & set(concerns))


def _skin_type_match(product: dict[str, Any], skin_type: str) -> int:
    supported = product.get("supported_skin_types") or []
    if not supported:
        return 1  # product supports any skin type
    if skin_type and skin_type in supported:
        return 1
    return 0


def _avoid_flag_hit(product: dict[str, Any], avoid_flags: list[str]) -> int:
    if not avoid_flags:
        return 0
    p_flags = set(product.get("flags") or [])
    return 1 if p_flags & set(avoid_flags) else 0


def _build_row(
    *,
    user_id: int,
    step: str,
    product: dict[str, Any],
    profile: dict[str, Any],
    label: int,
) -> dict[str, Any]:
    goals = list(profile.get("goals") or [])
    avoid_flags = list(profile.get("avoid_flags") or [])
    concerns = list(product.get("concerns") or [])
    actives = list(product.get("actives") or [])

    price = product.get("price")
    try:
        price_val = float(price) if price is not None else 0.0
        has_price = 1 if price is not None else 0
    except (TypeError, ValueError):
        price_val = 0.0
        has_price = 0

    return {
        "user_id": int(user_id),
        "product_id": int(product.get("id") or 0),
        "step": step,
        "episode_id": f"{user_id}__{step}",
        "group_id": f"{user_id}__{step}",
        "y": int(label),
        # categorical features
        "skin_type": str(profile.get("skin_type") or "normal"),
        "budget": str(profile.get("budget") or "medium"),
        "product_type": str(product.get("product_type") or step),
        "strength": str(product.get("strength") or "low"),
        # numeric features
        "price": price_val,
        "has_price": has_price,
        "in_stock": 1 if product.get("in_stock", True) else 0,
        "skin_type_match": _skin_type_match(product, str(profile.get("skin_type") or "")),
        "goal_concern_match_count": _goal_concern_match_count(goals, concerns),
        "goals_total": len(goals),
        "avoid_flag_hit": _avoid_flag_hit(product, avoid_flags),
        "actives_count": len(actives),
        "concerns_count": len(concerns),
    }


class Command(BaseCommand):
    help = "Build training dataset for the skincare routine ranker."

    def add_arguments(self, parser):
        parser.add_argument(
            "--output-dir",
            default="data/ml/routine_ranker_v1",
            help="Directory to write dataset.parquet and metadata.json.",
        )
        parser.add_argument(
            "--max-negatives",
            type=int,
            default=15,
            help="Max negative candidates sampled per (user, step) positive group.",
        )
        parser.add_argument(
            "--seed",
            type=int,
            default=42,
            help="Random seed for negative sampling.",
        )
        parser.add_argument(
            "--min-positive-users",
            type=int,
            default=1,
            help="Minimum number of unique users required to emit the dataset.",
        )

    def handle(self, *args, **options):
        if pd is None:
            raise CommandError("pandas is required. Install with: pip install pandas pyarrow")

        from catalog.models import Product
        from transactions.models import OwnedProduct
        from users_app.models import CustomerProfile

        output_dir = _resolve_output_dir(options["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)

        max_negatives = max(0, int(options["max_negatives"]))
        seed = int(options["seed"])
        rng = random.Random(seed)

        # Load all skincare products matching routine steps.
        products_qs = Product.objects.filter(category="skincare").values(
            "id",
            "name",
            "brand",
            "price",
            "product_type",
            "step",
            "strength",
            "in_stock",
            "concerns",
            "flags",
            "supported_skin_types",
            "actives",
        )
        products = list(products_qs)

        products_by_step: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for product in products:
            step_key = _product_step_key(product)
            if step_key is None:
                continue
            if product.get("in_stock") is False:
                continue
            products_by_step[step_key].append(product)

        available_steps = [s for s in ROUTINE_STEPS if products_by_step.get(s)]
        if not available_steps:
            raise CommandError("No skincare products found for routine steps.")

        self.stdout.write(
            f"Loaded {sum(len(v) for v in products_by_step.values())} candidate products "
            f"across steps: {available_steps}"
        )

        # Preload user profiles.
        profiles_by_user: dict[int, dict[str, Any]] = {}
        for profile in CustomerProfile.objects.values(
            "user_id", "skin_type", "goals", "avoid_flags", "budget"
        ):
            profiles_by_user[int(profile["user_id"])] = {
                "skin_type": profile.get("skin_type") or "normal",
                "goals": profile.get("goals") or [],
                "avoid_flags": profile.get("avoid_flags") or [],
                "budget": profile.get("budget") or "medium",
            }

        # Walk through owned skincare products. Group owned products per (user, step).
        owned_per_user_step: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
        product_by_id = {int(p["id"]): p for p in products}
        owned_qs = OwnedProduct.objects.filter(is_active=True, product__category="skincare").values(
            "user_id", "product_id"
        )
        for row in owned_qs:
            user_id = int(row["user_id"])
            product_id = int(row["product_id"])
            product = product_by_id.get(product_id)
            if product is None:
                continue
            step_key = _product_step_key(product)
            if step_key is None:
                continue
            owned_per_user_step[(user_id, step_key)].append(product)

        rows: list[dict[str, Any]] = []
        positive_users: set[int] = set()
        positive_count = 0
        negative_count = 0

        for (user_id, step), owned_products in owned_per_user_step.items():
            profile = profiles_by_user.get(user_id)
            if profile is None:
                # Skip users without a profile (no features to train on).
                continue

            step_pool = products_by_step.get(step) or []
            if not step_pool:
                continue

            owned_ids = {int(p["id"]) for p in owned_products}
            # One positive per owned product.
            for product in owned_products:
                rows.append(
                    _build_row(
                        user_id=user_id,
                        step=step,
                        product=product,
                        profile=profile,
                        label=1,
                    )
                )
                positive_count += 1

            # Sample negatives from the same step pool, excluding owned products.
            negative_pool = [p for p in step_pool if int(p["id"]) not in owned_ids]
            if negative_pool and max_negatives > 0:
                take = min(max_negatives, len(negative_pool))
                sampled = rng.sample(negative_pool, take)
                for product in sampled:
                    rows.append(
                        _build_row(
                            user_id=user_id,
                            step=step,
                            product=product,
                            profile=profile,
                            label=0,
                        )
                    )
                    negative_count += 1

            positive_users.add(user_id)

        if len(positive_users) < int(options["min_positive_users"]):
            raise CommandError(
                f"Only {len(positive_users)} users with positives found; "
                f"need at least {options['min_positive_users']}."
            )

        if not rows:
            raise CommandError("Generated empty dataset. Check OwnedProduct and Product data.")

        df = pd.DataFrame(rows)
        df = df.sort_values(["episode_id", "y"], ascending=[True, False]).reset_index(drop=True)

        dataset_path = output_dir / "dataset.parquet"
        try:
            df.to_parquet(dataset_path, index=False)
        except Exception as exc:
            # Fallback to CSV if parquet backend is missing.
            dataset_path = output_dir / "dataset.csv"
            df.to_csv(dataset_path, index=False)
            self.stdout.write(self.style.WARNING(f"Parquet not available ({exc}); wrote CSV."))

        metadata = {
            "version": "routine_ranker_v1",
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "dataset_file": str(dataset_path),
            "rows_total": int(len(df)),
            "positive_rows": int(positive_count),
            "negative_rows": int(negative_count),
            "unique_users": int(len(positive_users)),
            "unique_groups": int(df["episode_id"].nunique()),
            "steps": available_steps,
            "categorical_features": ["skin_type", "budget", "product_type", "strength", "step"],
            "numeric_features": [
                "price",
                "has_price",
                "in_stock",
                "skin_type_match",
                "goal_concern_match_count",
                "goals_total",
                "avoid_flag_hit",
                "actives_count",
                "concerns_count",
            ],
            "target_column": "y",
            "group_column": "episode_id",
            "max_negatives_per_group": max_negatives,
            "seed": seed,
        }

        with (output_dir / "metadata.json").open("w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        self.stdout.write(
            self.style.SUCCESS(
                f"Wrote {len(df)} rows ({positive_count} positive / {negative_count} negative) "
                f"for {len(positive_users)} users to {dataset_path}"
            )
        )
