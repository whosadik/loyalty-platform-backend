from __future__ import annotations

import json
import os
import re
from collections import Counter
from datetime import datetime, timezone
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

try:
    from roadmap_app.fragrance_slots import SLOTS, slot_of_fragrance
except Exception:  # pragma: no cover
    SLOTS = ["warm_day", "warm_evening", "cold_day", "cold_evening"]

    def slot_of_fragrance(attrs: dict, raw_meta: dict | None = None) -> str:
        return "warm_day"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _model_dir() -> Path:
    env_dir = os.getenv("ROADMAP_NEXTSTEP_V2_MODEL_DIR", "").strip()
    if env_dir:
        return Path(env_dir).expanduser().resolve()
    return (_repo_root() / "models" / "roadmap_next_step_v2").resolve()


def _slug_token(raw: Any) -> str:
    text = str(raw or "").strip().lower()
    text = text.replace("-", "_").replace(" ", "_")
    text = re.sub(r"[^a-z0-9_]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "unknown"


def _median(values: list[float]) -> float:
    if not values:
        return 0.0
    seq = sorted(values)
    n = len(seq)
    mid = n // 2
    if n % 2 == 1:
        return float(seq[mid])
    return float((seq[mid - 1] + seq[mid]) / 2.0)


def _safe_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


@lru_cache(maxsize=2)
def _load_artifact_cached(model_dir_path: str, mtime_ns: int):
    del mtime_ns
    if joblib is None:
        return None
    model_path = Path(model_dir_path) / "model.pkl"
    if not model_path.exists():
        return None
    try:
        return joblib.load(model_path)
    except Exception:
        return None


def _load_artifact() -> dict[str, Any] | None:
    model_dir = _model_dir()
    model_path = model_dir / "model.pkl"
    if not model_path.exists():
        return None

    artifact = _load_artifact_cached(str(model_dir), int(model_path.stat().st_mtime_ns))
    if artifact is None:
        return None

    if isinstance(artifact, dict):
        return artifact

    # Backward fallback if model.pkl contains only estimator.
    metadata = {}
    meta_path = model_dir / "metadata.json"
    if meta_path.exists():
        try:
            metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            metadata = {}
    return {
        "model": artifact,
        "class_labels": metadata.get("class_labels") or [],
        "feature_columns": metadata.get("feature_columns") or [],
        "categorical_features": metadata.get("categorical_features") or [],
        "numeric_features": metadata.get("numeric_features") or [],
    }


def _fetch_products(product_ids: list[int]) -> list[dict[str, Any]]:
    if not product_ids:
        return []
    try:
        from catalog.models import Product
    except Exception:
        return []

    qs = Product.objects.filter(id__in=product_ids).values(
        "id",
        "category",
        "product_type",
        "brand",
        "price",
        "attrs",
        "raw_meta",
    )
    rows = list(qs)
    by_id = {int(x["id"]): x for x in rows}
    ordered: list[dict[str, Any]] = []
    for pid in product_ids:
        if int(pid) in by_id:
            ordered.append(by_id[int(pid)])
    return ordered


def _build_minimal_feature_row(
    *,
    user_id: int,
    category: str,
    context_products: list[dict[str, Any]],
    feature_columns: list[str],
    categorical_features: list[str],
    numeric_features: list[str],
) -> dict[str, Any]:
    del user_id
    row: dict[str, Any] = {}
    for col in feature_columns:
        if col in categorical_features:
            row[col] = "__none__"
        else:
            row[col] = 0.0

    now = datetime.now(tz=timezone.utc)
    category_norm = str(category or "").strip().lower()
    if "category" in row:
        row["category"] = category_norm
    if "month_of_year" in row:
        row["month_of_year"] = int(now.month)
    if "days_since_last_purchase_in_category" in row:
        row["days_since_last_purchase_in_category"] = -1
    if "last_k_purchase_product_types" in row:
        row["last_k_purchase_product_types"] = "__none__"
    if "last_k_purchase_categories" in row:
        row["last_k_purchase_categories"] = "__none__"
    if "favorite_brand_top1" in row:
        row["favorite_brand_top1"] = "__none__"
    if "favorite_brands_top3" in row:
        row["favorite_brands_top3"] = "__none__"

    if "was_exposed_from_offers" in row:
        row["was_exposed_from_offers"] = 0
    if "has_offer_assignment_id" in row:
        row["has_offer_assignment_id"] = 0

    product_types: list[str] = []
    categories: list[str] = []
    brands: list[str] = []
    prices: list[float] = []
    owned_counter: Counter[tuple[str, str]] = Counter()
    slot_counter: Counter[str] = Counter()

    for item in context_products:
        item_category = str(item.get("category") or "").strip().lower()
        item_type = str(item.get("product_type") or "").strip().lower()
        item_brand = str(item.get("brand") or "").strip().lower()
        item_price = item.get("price")

        if item_type:
            product_types.append(item_type)
        if item_category:
            categories.append(item_category)
        if item_brand:
            brands.append(item_brand)
        try:
            if item_price is not None:
                prices.append(float(item_price))
        except Exception:
            pass

        owned_counter[(_slug_token(item_category), _slug_token(item_type))] += 1

        if item_category == "fragrance":
            slot_val = slot_of_fragrance(
                _safe_dict(item.get("attrs")),
                raw_meta=_safe_dict(item.get("raw_meta")),
            )
            if slot_val in SLOTS:
                slot_counter[slot_val] += 1

    if product_types and "last_k_purchase_product_types" in row:
        row["last_k_purchase_product_types"] = "|".join(product_types[:10])
    if categories and "last_k_purchase_categories" in row:
        row["last_k_purchase_categories"] = "|".join(categories[:10])
    if prices and "price_band_median_last5" in row:
        row["price_band_median_last5"] = float(_median(prices[:5]))

    if brands:
        brand_counter = Counter(brands)
        top_brands = [b for b, _ in brand_counter.most_common(3)]
        if "favorite_brand_top1" in row:
            row["favorite_brand_top1"] = top_brands[0]
        if "favorite_brands_top3" in row:
            row["favorite_brands_top3"] = "|".join(top_brands)

    for slot in SLOTS:
        col = f"owned_slot_{slot}"
        if col in row:
            row[col] = int(slot_counter.get(slot, 0))

    for col in feature_columns:
        if not col.startswith("owned_count__"):
            continue
        parts = col.split("__", 2)
        if len(parts) != 3:
            continue
        cat_token = parts[1]
        type_token = parts[2]
        row[col] = int(owned_counter.get((cat_token, type_token), 0))

    return row


def predict_next_product_types(
    user_id: int,
    context_product_ids: list[int],
    category: str,
) -> list[dict[str, Any]]:
    """
    Offline adapter contract for Roadmap NextStep model v2.

    Returns ranked predictions:
      [{"product_type": "serum", "score": 0.42}, ...]

    For fragrance, product_type values are slot classes
    (warm_day/warm_evening/cold_day/cold_evening).
    """
    if pd is None:
        return []

    artifact = _load_artifact()
    if not artifact:
        return []

    model = artifact.get("model")
    if model is None:
        return []

    feature_columns = list(artifact.get("feature_columns") or [])
    categorical_features = list(artifact.get("categorical_features") or [])
    numeric_features = list(artifact.get("numeric_features") or [])
    class_labels = list(artifact.get("class_labels") or [])

    if not feature_columns:
        return []

    safe_ids: list[int] = []
    for value in context_product_ids or []:
        try:
            safe_ids.append(int(value))
        except Exception:
            continue

    products = _fetch_products(safe_ids)
    row = _build_minimal_feature_row(
        user_id=int(user_id or 0),
        category=str(category or "").strip().lower(),
        context_products=products,
        feature_columns=feature_columns,
        categorical_features=categorical_features,
        numeric_features=numeric_features,
    )

    frame = pd.DataFrame([row], columns=feature_columns)
    try:
        proba = model.predict_proba(frame)
    except Exception:
        return []

    try:
        scores = list(proba[0])
    except Exception:
        return []

    if not class_labels:
        classes_attr = getattr(model, "classes_", None)
        if classes_attr is not None:
            class_labels = [str(x) for x in list(classes_attr)]
    if not class_labels:
        class_labels = [str(i) for i in range(len(scores))]

    ranked_idx = sorted(range(len(scores)), key=lambda i: float(scores[i]), reverse=True)
    out: list[dict[str, Any]] = []
    for idx in ranked_idx[:20]:
        label = class_labels[idx] if idx < len(class_labels) else str(idx)
        out.append({"product_type": str(label), "score": float(scores[idx])})
    return out
