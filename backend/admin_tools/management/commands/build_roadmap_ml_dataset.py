from __future__ import annotations

import hashlib
import json
import re
from bisect import bisect_left
from collections import Counter, defaultdict
from datetime import timedelta, timezone as dt_timezone
from pathlib import Path
from typing import Any

try:
    import pandas as pd
except Exception:  # pragma: no cover
    pd = None

from django.core.management.base import BaseCommand, CommandError
from django.utils import timezone

from roadmap_app.fragrance_slots import SLOTS, slot_of_fragrance
from roadmap_app.models import RoadmapEvent, RoadmapStep
from transactions.models import TransactionItem


TARGET_CATEGORIES = {"skincare", "haircare", "makeup", "fragrance"}
LAST_K_PURCHASES = 10
MIN_POSITIVES_OVERALL = 200
MIN_POSITIVES_FRAGRANCE = 30


def _safe_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _safe_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _slug_token(raw: Any) -> str:
    text = str(raw or "").strip().lower()
    if not text:
        return "unknown"
    text = text.replace("-", "_").replace(" ", "_")
    text = re.sub(r"[^a-z0-9_]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "unknown"


def _float_or_zero(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return 0.0


def _hours_between(start_dt, end_dt) -> float | None:
    if not start_dt or not end_dt:
        return None
    sec = (end_dt - start_dt).total_seconds()
    if sec < 0:
        return None
    return round(sec / 3600.0, 4)


def _median(values: list[float]) -> float:
    if not values:
        return 0.0
    values_sorted = sorted(values)
    n = len(values_sorted)
    mid = n // 2
    if n % 2 == 1:
        return float(values_sorted[mid])
    return float((values_sorted[mid - 1] + values_sorted[mid]) / 2.0)


def _deterministic_split_user_ids(user_ids: list[int], seed: int) -> dict[str, list[int]]:
    if not user_ids:
        return {"train": [], "val": [], "test": []}

    unique_ids = sorted({int(x) for x in user_ids})
    scored: list[tuple[str, int]] = []
    for user_id in unique_ids:
        payload = f"{seed}:{user_id}".encode("utf-8")
        score = hashlib.md5(payload).hexdigest()
        scored.append((score, user_id))
    scored.sort(key=lambda x: x[0])

    ordered = [x[1] for x in scored]
    n = len(ordered)
    n_train = int(round(n * 0.70))
    n_val = int(round(n * 0.15))
    if n_train < 1:
        n_train = 1
    if n_val < 1 and n >= 3:
        n_val = 1
    if n_train + n_val >= n:
        n_val = max(1, n - n_train - 1)
    n_test = n - n_train - n_val
    if n_test <= 0:
        n_test = 1
        if n_train > n_val and n_train > 1:
            n_train -= 1
        elif n_val > 1:
            n_val -= 1

    train_ids = ordered[:n_train]
    val_ids = ordered[n_train : n_train + n_val]
    test_ids = ordered[n_train + n_val :]
    return {
        "train": sorted(train_ids),
        "val": sorted(val_ids),
        "test": sorted(test_ids),
    }


def _write_dataset_frame(df: "pd.DataFrame", out_dir: Path) -> tuple[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = out_dir / "dataset.parquet"
    try:
        df.to_parquet(parquet_path, index=False)
        return "parquet", str(parquet_path)
    except Exception:
        csv_path = out_dir / "dataset.csv"
        df.to_csv(csv_path, index=False)
        return "csv", str(csv_path)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _resolve_out_dir(raw_path: str) -> Path:
    candidate = Path(str(raw_path)).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    repo_path = (_repo_root() / candidate).resolve()
    return repo_path


class Command(BaseCommand):
    help = "Build Roadmap ML v2 offline dataset from RoadmapEvent + Transactions without leakage."

    def add_arguments(self, parser):
        parser.add_argument("--days", type=int, default=180)
        parser.add_argument("--out-dir", type=str, default="data/ml/roadmap_nextstep")
        parser.add_argument("--include-ga", action="store_true", default=False)
        parser.add_argument(
            "--k",
            type=int,
            default=50,
            help="Top-N product_type features per category for owned_counts_by_product_type.",
        )
        parser.add_argument(
            "--label-window-days",
            type=int,
            default=14,
            help="Label window after first exposure to mark STEP_COMPLETED positive.",
        )
        parser.add_argument("--seed", type=int, default=42)

    def handle(self, *args, **options):
        if pd is None:
            raise CommandError("pandas is required. Install dependencies from requirements-ml.txt")

        days = int(options["days"])
        out_dir = _resolve_out_dir(str(options["out_dir"]))
        include_ga = bool(options["include_ga"])
        top_types_k = int(options["k"])
        label_window_days = int(options["label_window_days"])
        seed = int(options["seed"])

        if days <= 0:
            raise CommandError("--days must be > 0")
        if top_types_k <= 0:
            raise CommandError("--k must be > 0")
        if label_window_days <= 0:
            raise CommandError("--label-window-days must be > 0")

        now_utc = timezone.now().astimezone(dt_timezone.utc)
        since = now_utc - timedelta(days=days)
        max_t0 = now_utc - timedelta(days=label_window_days)

        self.stdout.write(
            "[build_roadmap_ml_dataset] "
            f"window={since.isoformat()}..{now_utc.isoformat()} label_window_days={label_window_days}"
        )

        base_qs = RoadmapEvent.objects.filter(
            event_type=RoadmapEvent.Type.STEP_EXPOSED,
            step_id__isnull=False,
            created_at__gte=since,
            created_at__lte=max_t0,
        )
        if not include_ga:
            base_qs = base_qs.exclude(user__username__startswith="ga_")

        raw_exposed_count = 0
        exposed_from_offers = 0
        exposed_from_roadmap_api = 0
        first_exposed_by_pair: dict[tuple[int, int], dict[str, Any]] = {}

        for row in base_qs.values("user_id", "step_id", "created_at", "context").iterator(chunk_size=5000):
            raw_exposed_count += 1
            user_id = int(row["user_id"])
            step_id = int(row["step_id"])
            created_at = row["created_at"].astimezone(dt_timezone.utc)
            ctx = _safe_dict(row.get("context"))
            sources = {str(x).strip().lower() for x in _safe_list(ctx.get("sources")) if str(x).strip()}
            has_offer_assignment_id = 1 if ctx.get("offer_assignment_id") not in (None, "") else 0
            is_from_offers = 1 if ("offers" in sources) else 0
            if not sources and has_offer_assignment_id:
                is_from_offers = 1

            if is_from_offers:
                exposed_from_offers += 1
            else:
                exposed_from_roadmap_api += 1

            key = (user_id, step_id)
            prev = first_exposed_by_pair.get(key)
            if prev is None or created_at < prev["t0"]:
                first_exposed_by_pair[key] = {
                    "user_id": user_id,
                    "step_id": step_id,
                    "t0": created_at,
                    "was_exposed_from_offers": int(is_from_offers),
                    "has_offer_assignment_id": int(has_offer_assignment_id),
                }

        if not first_exposed_by_pair:
            raise CommandError("No STEP_EXPOSED instances found for the selected window.")

        step_ids = sorted({int(x[1]) for x in first_exposed_by_pair.keys()})
        step_meta = {
            int(row["id"]): {
                "category": str(row.get("plan__category") or "").strip().lower(),
                "product_type": str(row.get("product_type") or "").strip().lower(),
                "step_index": int(row.get("step_index") or 0),
            }
            for row in RoadmapStep.objects.filter(id__in=step_ids).values(
                "id", "plan__category", "product_type", "step_index"
            )
        }

        instances: list[dict[str, Any]] = []
        for rec in first_exposed_by_pair.values():
            meta = step_meta.get(int(rec["step_id"]))
            if not meta:
                continue
            category = str(meta.get("category") or "")
            if category not in TARGET_CATEGORIES:
                continue
            product_type = str(meta.get("product_type") or "")
            if not product_type:
                continue
            instances.append(
                {
                    "user_id": int(rec["user_id"]),
                    "step_id": int(rec["step_id"]),
                    "t0": rec["t0"],
                    "category": category,
                    "step_product_type": product_type,
                    "step_index": int(meta.get("step_index") or 0),
                    "was_exposed_from_offers": int(rec["was_exposed_from_offers"]),
                    "has_offer_assignment_id": int(rec["has_offer_assignment_id"]),
                }
            )

        if not instances:
            raise CommandError("No valid funnel instances after category/product_type filtering.")
        users = sorted({int(x["user_id"]) for x in instances})
        pair_keys = {(int(x["user_id"]), int(x["step_id"])) for x in instances}

        event_qs = RoadmapEvent.objects.filter(
            step_id__in=step_ids,
            user_id__in=users,
            event_type__in=[RoadmapEvent.Type.STEP_CLICKED, RoadmapEvent.Type.STEP_COMPLETED],
            created_at__gte=since,
        )
        if not include_ga:
            event_qs = event_qs.exclude(user__username__startswith="ga_")

        click_times: dict[tuple[int, int], list[Any]] = defaultdict(list)
        complete_times: dict[tuple[int, int], list[Any]] = defaultdict(list)
        for row in event_qs.values("user_id", "step_id", "event_type", "created_at").iterator(chunk_size=5000):
            key = (int(row["user_id"]), int(row["step_id"]))
            if key not in pair_keys:
                continue
            dt = row["created_at"].astimezone(dt_timezone.utc)
            event_type = str(row["event_type"])
            if event_type == RoadmapEvent.Type.STEP_CLICKED:
                click_times[key].append(dt)
            elif event_type == RoadmapEvent.Type.STEP_COMPLETED:
                complete_times[key].append(dt)

        for values in click_times.values():
            values.sort()
        for values in complete_times.values():
            values.sort()

        latency_click_hours: list[float] = []
        latency_complete_hours: list[float] = []
        positives_total = 0
        positives_by_category: Counter[str] = Counter()
        positives_by_class: Counter[str] = Counter()

        for row in instances:
            key = (int(row["user_id"]), int(row["step_id"]))
            t0 = row["t0"]
            click_seq = click_times.get(key) or []
            complete_seq = complete_times.get(key) or []

            first_click = None
            if click_seq:
                i = bisect_left(click_seq, t0)
                if i < len(click_seq):
                    first_click = click_seq[i]

            first_complete = None
            if complete_seq:
                i = bisect_left(complete_seq, t0)
                if i < len(complete_seq):
                    first_complete = complete_seq[i]

            label = 0
            if first_complete is not None and first_complete <= (t0 + timedelta(days=label_window_days)):
                label = 1

            target_class = row["step_product_type"] if label == 1 else ""
            row["label"] = int(label)
            row["target_class"] = target_class

            click_h = _hours_between(t0, first_click)
            complete_h = _hours_between(t0, first_complete)
            row["latency_to_click_hours"] = click_h
            row["latency_to_complete_hours"] = complete_h
            if click_h is not None:
                latency_click_hours.append(float(click_h))
            if complete_h is not None:
                latency_complete_hours.append(float(complete_h))

            if label == 1:
                positives_total += 1
                positives_by_category[str(row["category"])] += 1
                positives_by_class[str(target_class)] += 1

        fragrance_positives = int(positives_by_category.get("fragrance", 0))
        if positives_total < MIN_POSITIVES_OVERALL:
            raise CommandError(
                f"Not enough positives for training: {positives_total} < {MIN_POSITIVES_OVERALL}"
            )
        if fragrance_positives < MIN_POSITIVES_FRAGRANCE:
            raise CommandError(
                f"Not enough fragrance positives: {fragrance_positives} < {MIN_POSITIVES_FRAGRANCE}"
            )

        tx_qs = TransactionItem.objects.filter(
            transaction__user_id__in=users,
            transaction__created_at__lte=max_t0,
        ).values(
            "transaction__user_id",
            "transaction__id",
            "transaction__created_at",
            "transaction__total_amount",
            "product_id",
            "product__category",
            "product__product_type",
            "product__brand",
            "quantity",
            "unit_price",
            "product__attrs",
            "product__raw_meta",
        )

        user_items: dict[int, list[dict[str, Any]]] = defaultdict(list)
        type_counter_by_category: dict[str, Counter[str]] = defaultdict(Counter)
        for row in tx_qs.iterator(chunk_size=5000):
            user_id = int(row["transaction__user_id"])
            category = str(row.get("product__category") or "").strip().lower()
            product_type = str(row.get("product__product_type") or "").strip().lower()
            brand = str(row.get("product__brand") or "").strip().lower()
            qty = int(row.get("quantity") or 0)
            unit_price = _float_or_zero(row.get("unit_price"))
            tx_total = _float_or_zero(row.get("transaction__total_amount"))
            ts = row["transaction__created_at"].astimezone(dt_timezone.utc)

            slot_value = ""
            if category == "fragrance":
                slot_value = slot_of_fragrance(
                    _safe_dict(row.get("product__attrs")),
                    raw_meta=_safe_dict(row.get("product__raw_meta")),
                )

            rec = {
                "ts": ts,
                "tx_id": int(row["transaction__id"]),
                "tx_total": tx_total,
                "category": category,
                "product_type": product_type,
                "brand": brand,
                "quantity": max(1, qty),
                "unit_price": unit_price,
                "slot": slot_value,
            }
            user_items[user_id].append(rec)
            if category in TARGET_CATEGORIES and product_type:
                type_counter_by_category[category][product_type] += max(1, qty)

        for values in user_items.values():
            values.sort(key=lambda x: (x["ts"], x["tx_id"]))

        top_types_by_category: dict[str, list[str]] = {}
        for category in sorted(TARGET_CATEGORIES):
            cnt = type_counter_by_category.get(category) or Counter()
            top_types_by_category[category] = [ptype for ptype, _ in cnt.most_common(top_types_k)]

        owned_feature_columns: list[str] = []
        owned_feature_map: dict[tuple[str, str], str] = {}
        for category in sorted(TARGET_CATEGORIES):
            for ptype in top_types_by_category.get(category, []):
                col = f"owned_count__{_slug_token(category)}__{_slug_token(ptype)}"
                owned_feature_columns.append(col)
                owned_feature_map[(category, ptype)] = col

        timeline_by_user: dict[int, list[Any]] = {
            user_id: [x["ts"] for x in items] for user_id, items in user_items.items()
        }

        rows: list[dict[str, Any]] = []
        for inst in instances:
            user_id = int(inst["user_id"])
            t0 = inst["t0"]
            category = str(inst["category"])

            items = user_items.get(user_id, [])
            timeline = timeline_by_user.get(user_id, [])
            idx = bisect_left(timeline, t0) if timeline else 0
            prior_items = items[:idx]

            recent_types: list[str] = []
            recent_categories: list[str] = []
            recent_prices: list[float] = []
            brand_counter: Counter[str] = Counter()
            category_owned_counter: Counter[str] = Counter()
            slot_counter: Counter[str] = Counter()

            last_ts_in_category = None
            tx_ids_90d: set[int] = set()
            tx_amount_90d: dict[int, float] = {}
            since_90d = t0 - timedelta(days=90)

            for item in reversed(prior_items):
                item_category = str(item["category"])
                item_type = str(item["product_type"])
                item_brand = str(item["brand"])
                item_qty = int(item["quantity"])
                item_ts = item["ts"]

                if item_type and len(recent_types) < LAST_K_PURCHASES:
                    recent_types.append(item_type)
                if item_category and len(recent_categories) < LAST_K_PURCHASES:
                    recent_categories.append(item_category)
                if len(recent_prices) < 5:
                    recent_prices.append(float(item["unit_price"]))
                if item_brand:
                    brand_counter[item_brand] += item_qty

                if item_category == category:
                    category_owned_counter[item_type] += item_qty
                    if last_ts_in_category is None:
                        last_ts_in_category = item_ts
                    if item_ts >= since_90d:
                        tx_id = int(item["tx_id"])
                        tx_ids_90d.add(tx_id)
                        tx_amount_90d[tx_id] = float(item["tx_total"])

                if item_category == "fragrance":
                    slot_val = str(item.get("slot") or "")
                    if slot_val in SLOTS:
                        slot_counter[slot_val] += item_qty

            days_since_last_purchase = None
            if last_ts_in_category is not None:
                days_since_last_purchase = (t0.date() - last_ts_in_category.date()).days

            tx_count_90d = len(tx_ids_90d)
            tx_amount_90d_val = round(float(sum(tx_amount_90d.values())), 4)
            price_band = round(_median(recent_prices), 4)

            top_brands = [b for b, _ in brand_counter.most_common(3)]
            top_brand_1 = top_brands[0] if top_brands else "__none__"
            top_brand_3 = "|".join(top_brands) if top_brands else "__none__"

            row = {
                "user_id": user_id,
                "step_id": int(inst["step_id"]),
                "first_exposed_at": t0.isoformat().replace("+00:00", "Z"),
                "category": category,
                "step_index": int(inst["step_index"]),
                "step_product_type": str(inst["step_product_type"]),
                "label": int(inst["label"]),
                "target_class": str(inst["target_class"] or ""),
                "month_of_year": int(t0.month),
                "was_exposed_from_offers": int(inst["was_exposed_from_offers"]),
                "has_offer_assignment_id": int(inst["has_offer_assignment_id"]),
                "days_since_last_purchase_in_category": (
                    int(days_since_last_purchase)
                    if days_since_last_purchase is not None
                    else -1
                ),
                "tx_count_90d_category": int(tx_count_90d),
                "tx_amount_90d_category": float(tx_amount_90d_val),
                "last_k_purchase_product_types": "|".join(recent_types) if recent_types else "__none__",
                "last_k_purchase_categories": "|".join(recent_categories) if recent_categories else "__none__",
                "price_band_median_last5": float(price_band),
                "favorite_brand_top1": top_brand_1,
                "favorite_brands_top3": top_brand_3,
                "owned_slot_warm_day": int(slot_counter.get("warm_day", 0)),
                "owned_slot_warm_evening": int(slot_counter.get("warm_evening", 0)),
                "owned_slot_cold_day": int(slot_counter.get("cold_day", 0)),
                "owned_slot_cold_evening": int(slot_counter.get("cold_evening", 0)),
                "latency_to_click_hours": inst["latency_to_click_hours"],
                "latency_to_complete_hours": inst["latency_to_complete_hours"],
            }

            for col in owned_feature_columns:
                row[col] = 0
            for product_type, count in category_owned_counter.items():
                mapped_col = owned_feature_map.get((category, product_type))
                if mapped_col:
                    row[mapped_col] = int(count)

            rows.append(row)
        if not rows:
            raise CommandError("No dataset rows produced.")

        df = pd.DataFrame(rows)
        df = df.sort_values(["user_id", "step_id"]).reset_index(drop=True)

        split_map = _deterministic_split_user_ids(df["user_id"].astype(int).tolist(), seed=seed)
        splits_payload = {
            "seed": seed,
            "strategy": "deterministic_hash_user_level",
            "ratios": {"train": 0.70, "val": 0.15, "test": 0.15},
            "train_user_ids": split_map["train"],
            "val_user_ids": split_map["val"],
            "test_user_ids": split_map["test"],
        }

        dataset_format, dataset_file = _write_dataset_frame(df, out_dir)

        categorical_features = [
            "category",
            "last_k_purchase_product_types",
            "last_k_purchase_categories",
            "favorite_brand_top1",
            "favorite_brands_top3",
        ]
        numeric_features = [
            "month_of_year",
            "was_exposed_from_offers",
            "has_offer_assignment_id",
            "days_since_last_purchase_in_category",
            "tx_count_90d_category",
            "tx_amount_90d_category",
            "price_band_median_last5",
            "owned_slot_warm_day",
            "owned_slot_warm_evening",
            "owned_slot_cold_day",
            "owned_slot_cold_evening",
            *owned_feature_columns,
        ]
        feature_columns = [*categorical_features, *numeric_features]

        avg_click_h = round(sum(latency_click_hours) / len(latency_click_hours), 4) if latency_click_hours else None
        avg_complete_h = (
            round(sum(latency_complete_hours) / len(latency_complete_hours), 4)
            if latency_complete_hours
            else None
        )

        metadata = {
            "generated_at_utc": now_utc.isoformat().replace("+00:00", "Z"),
            "window_days": days,
            "window_since_utc": since.isoformat().replace("+00:00", "Z"),
            "window_until_utc": now_utc.isoformat().replace("+00:00", "Z"),
            "label_window_days": label_window_days,
            "include_ga": include_ga,
            "top_owned_product_types_k": top_types_k,
            "dataset_format": dataset_format,
            "dataset_file": dataset_file,
            "rows_total": int(len(df)),
            "rows_positive": int(positives_total),
            "rows_negative": int(len(df) - positives_total),
            "exposed_total": int(raw_exposed_count),
            "instances_total": int(len(instances)),
            "positives_by_category": {k: int(v) for k, v in sorted(positives_by_category.items())},
            "positives_by_class": {k: int(v) for k, v in sorted(positives_by_class.items())},
            "exposures_from_offers": int(exposed_from_offers),
            "exposures_from_roadmap_api": int(exposed_from_roadmap_api),
            "avg_latency_to_click_hours": avg_click_h,
            "avg_latency_to_complete_hours": avg_complete_h,
            "feature_columns": feature_columns,
            "categorical_features": categorical_features,
            "numeric_features": numeric_features,
            "top_product_types_by_category": top_types_by_category,
            "target_categories": sorted(TARGET_CATEGORIES),
            "fragrance_slots": list(SLOTS),
        }

        out_dir.mkdir(parents=True, exist_ok=True)
        metadata_path = out_dir / "metadata.json"
        splits_path = out_dir / "splits.json"
        metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        splits_path.write_text(json.dumps(splits_payload, ensure_ascii=False, indent=2), encoding="utf-8")

        self.stdout.write("[build_roadmap_ml_dataset] done")
        self.stdout.write(f"[build_roadmap_ml_dataset] dataset={dataset_file}")
        self.stdout.write(f"[build_roadmap_ml_dataset] metadata={metadata_path}")
        self.stdout.write(f"[build_roadmap_ml_dataset] splits={splits_path}")
        self.stdout.write(
            "[build_roadmap_ml_dataset] "
            f"exposed_total={raw_exposed_count} positives_total={positives_total} "
            f"positives_by_category={dict(sorted((k, int(v)) for k, v in positives_by_category.items()))}"
        )
        self.stdout.write(
            "[build_roadmap_ml_dataset] "
            f"exposures_from_offers={exposed_from_offers} exposures_from_roadmap_api={exposed_from_roadmap_api}"
        )
        self.stdout.write(
            "[build_roadmap_ml_dataset] "
            f"avg_latency_click_h={avg_click_h} avg_latency_complete_h={avg_complete_h}"
        )
