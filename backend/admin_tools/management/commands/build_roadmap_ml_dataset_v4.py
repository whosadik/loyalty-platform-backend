from __future__ import annotations

import hashlib
import json
import math
import re
from bisect import bisect_right
from collections import Counter, defaultdict
from datetime import timedelta, timezone as dt_timezone
from pathlib import Path
from typing import Any, Callable

try:
    import pandas as pd
except Exception:  # pragma: no cover
    pd = None

from django.core.management.base import BaseCommand, CommandError
from django.utils import timezone

from roadmap_app.fragrance_slots import SLOTS, slot_of_fragrance
from roadmap_app.models import RoadmapEvent
from transactions.models import TransactionItem


TARGET_CATEGORIES = {"skincare", "haircare", "makeup", "fragrance"}
RULE_CHAIN_BY_CATEGORY: dict[str, list[str]] = {
    "skincare": [
        "cleanser",
        "serum",
        "moisturizer",
        "spf",
        "toner",
        "mask",
        "eye_cream",
        "essence",
    ],
    "haircare": [
        "shampoo",
        "conditioner",
        "hair_mask",
        "hair_oil",
        "scalp_serum",
        "leave_in",
    ],
    "makeup": [
        "foundation",
        "mascara",
        "blush",
        "lipstick",
        "eyeshadow",
        "primer",
        "setting_spray",
    ],
    "fragrance": list(SLOTS),
}


def _safe_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


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


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _resolve_out_dir(raw_path: str) -> Path:
    candidate = Path(str(raw_path)).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    return (_repo_root() / candidate).resolve()


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


def _unique(values: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        token = str(value or "").strip().lower()
        if not token:
            continue
        if token in seen:
            continue
        seen.add(token)
        out.append(token)
    return out


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


def _class_distribution_for_splits(
    episodes: list[dict[str, Any]],
    *,
    split_map: dict[str, list[int]],
) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {}
    for split_name, user_ids in split_map.items():
        uid_set = set(int(x) for x in user_ids)
        rows = [ep for ep in episodes if int(ep["user_id"]) in uid_set]
        counter = Counter(str(ep["label"]) for ep in rows)
        out[split_name] = {
            key: int(value)
            for key, value in sorted(counter.items(), key=lambda kv: (-kv[1], kv[0]))
        }
    return out


def _evaluate_candidate_rankings(
    *,
    episodes: list[dict[str, Any]],
    candidate_types_by_category: dict[str, list[str]],
    ranking_fn: Callable[[dict[str, Any], list[str]], list[str]],
) -> dict[str, Any]:
    rows_total = len(episodes)
    positives = 0
    hits_1 = 0
    hits_3 = 0
    hits_5 = 0
    ndcg_5_sum = 0.0
    outside_candidate_set = 0

    for episode in episodes:
        category = str(episode["category"])
        label = str(episode["label"])
        candidates = list(candidate_types_by_category.get(category) or [])
        if not candidates:
            continue
        ranked = ranking_fn(episode, candidates)
        if label == "__none__":
            continue
        positives += 1
        if label not in ranked:
            outside_candidate_set += 1
            continue
        rank = int(ranked.index(label) + 1)
        if rank == 1:
            hits_1 += 1
        if rank <= 3:
            hits_3 += 1
        if rank <= 5:
            hits_5 += 1
            ndcg_5_sum += float(1.0 / math.log2(rank + 1.0))

    if positives <= 0:
        return {
            "rows": int(rows_total),
            "positive_episodes": 0,
            "label_outside_candidate_set": int(outside_candidate_set),
            "recall_at_1": 0.0,
            "recall_at_3": 0.0,
            "recall_at_5": 0.0,
            "ndcg_at_5": 0.0,
        }

    return {
        "rows": int(rows_total),
        "positive_episodes": int(positives),
        "label_outside_candidate_set": int(outside_candidate_set),
        "recall_at_1": round(float(hits_1 / positives), 6),
        "recall_at_3": round(float(hits_3 / positives), 6),
        "recall_at_5": round(float(hits_5 / positives), 6),
        "ndcg_at_5": round(float(ndcg_5_sum / positives), 6),
    }


def _build_baselines(
    *,
    episodes: list[dict[str, Any]],
    split_map: dict[str, list[int]],
    candidate_types_by_category: dict[str, list[str]],
) -> dict[str, Any]:
    train_users = set(int(x) for x in split_map.get("train") or [])
    val_users = set(int(x) for x in split_map.get("val") or [])
    test_users = set(int(x) for x in split_map.get("test") or [])

    train_episodes = [ep for ep in episodes if int(ep["user_id"]) in train_users]
    val_episodes = [ep for ep in episodes if int(ep["user_id"]) in val_users]
    test_episodes = [ep for ep in episodes if int(ep["user_id"]) in test_users]

    popularity_by_category: dict[str, Counter[str]] = defaultdict(Counter)
    transitions: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    for ep in train_episodes:
        category = str(ep["category"])
        label = str(ep["label"])
        if label == "__none__":
            continue
        popularity_by_category[category][label] += 1
        state = str(ep.get("last1_product_type") or "__none__")
        transitions[(category, state)][label] += 1

    def _sort_by_counter(candidates: list[str], counter: Counter[str] | None) -> list[str]:
        counter = counter or Counter()
        return sorted(candidates, key=lambda c: (-int(counter.get(c, 0)), c))

    def _rank_popularity(ep: dict[str, Any], candidates: list[str]) -> list[str]:
        category = str(ep["category"])
        return _sort_by_counter(candidates, popularity_by_category.get(category))

    def _rank_markov(ep: dict[str, Any], candidates: list[str]) -> list[str]:
        category = str(ep["category"])
        state = str(ep.get("last1_product_type") or "__none__")
        counter = transitions.get((category, state))
        if not counter:
            counter = popularity_by_category.get(category)
        return _sort_by_counter(candidates, counter)

    return {
        "splits": {
            "val": {
                "popularity": _evaluate_candidate_rankings(
                    episodes=val_episodes,
                    candidate_types_by_category=candidate_types_by_category,
                    ranking_fn=_rank_popularity,
                ),
                "markov": _evaluate_candidate_rankings(
                    episodes=val_episodes,
                    candidate_types_by_category=candidate_types_by_category,
                    ranking_fn=_rank_markov,
                ),
            },
            "test": {
                "popularity": _evaluate_candidate_rankings(
                    episodes=test_episodes,
                    candidate_types_by_category=candidate_types_by_category,
                    ranking_fn=_rank_popularity,
                ),
                "markov": _evaluate_candidate_rankings(
                    episodes=test_episodes,
                    candidate_types_by_category=candidate_types_by_category,
                    ranking_fn=_rank_markov,
                ),
            },
        }
    }


class Command(BaseCommand):
    help = (
        "Build Roadmap NextStep v4 ranking dataset: candidate rows per episode "
        "with leakage-safe user features and candidate features."
    )

    def add_arguments(self, parser):
        parser.add_argument("--days", type=int, default=180)
        parser.add_argument("--out-dir", type=str, default="data/ml/roadmap_nextstep_v4")
        parser.add_argument("--include-ga", action="store_true", default=False)
        parser.add_argument("--label-window-days", type=int, default=14)
        parser.add_argument(
            "--popularity-top-n",
            type=int,
            default=25,
            help="Top-N popular product types per category to extend candidate set.",
        )
        parser.add_argument(
            "--owned-top-k",
            type=int,
            default=20,
            help="Top-N product_type ownership counters per category used as sparse user features.",
        )
        parser.add_argument("--seed", type=int, default=42)

    def handle(self, *args, **options):
        if pd is None:
            raise CommandError("pandas is required. Install dependencies from requirements-ml.txt")

        days = int(options["days"])
        out_dir = _resolve_out_dir(str(options["out_dir"]))
        include_ga = bool(options["include_ga"])
        label_window_days = int(options["label_window_days"])
        popularity_top_n = int(options["popularity_top_n"])
        owned_top_k = int(options["owned_top_k"])
        seed = int(options["seed"])

        if days <= 0:
            raise CommandError("--days must be > 0")
        if label_window_days <= 0:
            raise CommandError("--label-window-days must be > 0")
        if popularity_top_n <= 0:
            raise CommandError("--popularity-top-n must be > 0")
        if owned_top_k <= 0:
            raise CommandError("--owned-top-k must be > 0")

        now_utc = timezone.now().astimezone(dt_timezone.utc)
        since = now_utc - timedelta(days=days)
        max_t0 = now_utc - timedelta(days=label_window_days)

        self.stdout.write(
            "[build_roadmap_ml_dataset_v4] "
            f"window={since.isoformat()}..{now_utc.isoformat()} label_window_days={label_window_days}"
        )

        exposed_qs = RoadmapEvent.objects.filter(
            event_type=RoadmapEvent.Type.STEP_EXPOSED,
            step_id__isnull=False,
            created_at__gte=since,
            created_at__lte=max_t0,
        )
        if not include_ga:
            exposed_qs = exposed_qs.exclude(user__username__startswith="ga_")

        first_episode_by_key: dict[tuple[int, str, str], dict[str, Any]] = {}
        raw_exposed_total = 0

        for row in exposed_qs.values(
            "user_id",
            "created_at",
            "context",
            "step__plan__category",
        ).iterator(chunk_size=5000):
            raw_exposed_total += 1
            user_id = int(row["user_id"])
            created_at = row["created_at"].astimezone(dt_timezone.utc)
            context = _safe_dict(row.get("context"))
            category = str(row.get("step__plan__category") or context.get("category") or "").strip().lower()
            if category not in TARGET_CATEGORIES:
                continue
            day_key = created_at.date().isoformat()
            key = (user_id, category, day_key)
            prev = first_episode_by_key.get(key)
            if prev is None or created_at < prev["t0"]:
                first_episode_by_key[key] = {
                    "user_id": user_id,
                    "category": category,
                    "t0": created_at,
                    "day_key": day_key,
                }

        if not first_episode_by_key:
            raise CommandError("No STEP_EXPOSED episodes for selected window.")

        episodes_seed = sorted(
            first_episode_by_key.values(),
            key=lambda x: (int(x["user_id"]), str(x["category"]), str(x["day_key"]), x["t0"]),
        )
        users = sorted({int(ep["user_id"]) for ep in episodes_seed})

        tx_qs = TransactionItem.objects.filter(
            transaction__user_id__in=users,
            transaction__created_at__lte=now_utc,
        ).values(
            "id",
            "transaction__user_id",
            "transaction__id",
            "transaction__created_at",
            "transaction__total_amount",
            "quantity",
            "unit_price",
            "product__category",
            "product__product_type",
            "product__attrs",
            "product__raw_meta",
        )

        user_items: dict[int, list[dict[str, Any]]] = defaultdict(list)
        popularity_counter_by_category: dict[str, Counter[str]] = defaultdict(Counter)
        owned_counter_by_category: dict[str, Counter[str]] = defaultdict(Counter)

        for row in tx_qs.iterator(chunk_size=5000):
            user_id = int(row["transaction__user_id"])
            ts = row["transaction__created_at"].astimezone(dt_timezone.utc)
            category = str(row.get("product__category") or "").strip().lower()
            product_type = str(row.get("product__product_type") or "").strip().lower()
            quantity = max(1, int(row.get("quantity") or 0))
            slot_value = ""
            if category == "fragrance":
                slot_value = slot_of_fragrance(
                    _safe_dict(row.get("product__attrs")),
                    raw_meta=_safe_dict(row.get("product__raw_meta")),
                )
            user_items[user_id].append(
                {
                    "item_id": int(row["id"]),
                    "ts": ts,
                    "tx_id": int(row["transaction__id"]),
                    "tx_total": _float_or_zero(row.get("transaction__total_amount")),
                    "category": category,
                    "product_type": product_type,
                    "quantity": quantity,
                    "unit_price": _float_or_zero(row.get("unit_price")),
                    "slot": slot_value,
                }
            )
            if ts <= max_t0 and category in TARGET_CATEGORIES and product_type:
                if category != "fragrance":
                    popularity_counter_by_category[category][product_type] += quantity
                owned_counter_by_category[category][product_type] += quantity

        for values in user_items.values():
            values.sort(key=lambda x: (x["ts"], int(x["tx_id"]), int(x["item_id"])))

        candidate_types_by_category: dict[str, list[str]] = {}
        top_popularity_by_category: dict[str, list[str]] = {}
        for category in ["skincare", "haircare", "makeup"]:
            top_pop = [
                product_type
                for product_type, _ in (popularity_counter_by_category.get(category) or Counter()).most_common(
                    popularity_top_n
                )
            ]
            top_popularity_by_category[category] = top_pop
            candidate_types_by_category[category] = _unique(
                list(RULE_CHAIN_BY_CATEGORY.get(category) or []) + top_pop
            )
        candidate_types_by_category["fragrance"] = list(SLOTS)
        top_popularity_by_category["fragrance"] = list(SLOTS)

        top_owned_types_by_category: dict[str, list[str]] = {}
        owned_feature_columns: list[str] = []
        owned_feature_map: dict[tuple[str, str], str] = {}
        for category in sorted(TARGET_CATEGORIES):
            top_types = [
                product_type
                for product_type, _ in (owned_counter_by_category.get(category) or Counter()).most_common(owned_top_k)
            ]
            top_owned_types_by_category[category] = top_types
            for product_type in top_types:
                col = f"owned_count__{_slug_token(category)}__{_slug_token(product_type)}"
                owned_feature_columns.append(col)
                owned_feature_map[(category, product_type)] = col

        timeline_by_user: dict[int, list[Any]] = {
            user_id: [row["ts"] for row in rows] for user_id, rows in user_items.items()
        }

        split_map = _deterministic_split_user_ids(users, seed=seed)
        split_by_user: dict[int, str] = {}
        for split_name, split_users in split_map.items():
            for user_id in split_users:
                split_by_user[int(user_id)] = split_name

        episode_records: list[dict[str, Any]] = []
        episode_aux: dict[int, dict[str, Any]] = {}
        leakage_checks_total = 0

        for episode_id, seed_row in enumerate(episodes_seed, start=1):
            user_id = int(seed_row["user_id"])
            category = str(seed_row["category"])
            t0 = seed_row["t0"]
            split_name = str(split_by_user.get(user_id) or "train")

            items = user_items.get(user_id) or []
            timeline = timeline_by_user.get(user_id) or []
            pivot = bisect_right(timeline, t0) if timeline else 0
            prior_items = items[:pivot]
            future_items = items[pivot:]
            leakage_checks_total += 1
            if any(row["ts"] > t0 for row in prior_items):
                raise CommandError(f"Leakage detected: prior_items has ts > t0 for user_id={user_id}")
            if any(row["ts"] <= t0 for row in future_items):
                raise CommandError(f"Leakage detected: future_items has ts <= t0 for user_id={user_id}")

            label = "__none__"
            window_end = t0 + timedelta(days=label_window_days)
            for row in future_items:
                ts = row["ts"]
                if ts > window_end:
                    break
                if str(row["category"]) != category:
                    continue
                if category == "fragrance":
                    slot_value = str(row.get("slot") or "")
                    if slot_value in SLOTS:
                        label = slot_value
                        break
                else:
                    ptype = str(row.get("product_type") or "")
                    if ptype:
                        label = ptype
                        break

            last_product_types: list[str] = []
            last_categories: list[str] = []
            last_slot_values: list[str] = []
            slot_counter: Counter[str] = Counter()
            owned_counts_all: Counter[tuple[str, str]] = Counter()
            candidate_owned_counter: Counter[str] = Counter()
            candidate_seen_90d_counter: Counter[str] = Counter()
            candidate_last_seen_at: dict[str, Any] = {}

            last_ts_in_category = None
            tx_ids_90d: set[int] = set()
            tx_amount_90d: dict[int, float] = {}
            since_90d = t0 - timedelta(days=90)

            for row in reversed(prior_items):
                item_category = str(row["category"] or "")
                item_type = str(row["product_type"] or "")
                qty = int(row["quantity"])
                ts = row["ts"]

                if item_type and len(last_product_types) < 5:
                    last_product_types.append(item_type)
                if item_category and len(last_categories) < 5:
                    last_categories.append(item_category)

                if item_category and item_type:
                    owned_counts_all[(item_category, item_type)] += qty

                if item_category == category:
                    candidate_key = ""
                    if category == "fragrance":
                        candidate_key = str(row.get("slot") or "")
                    else:
                        candidate_key = item_type
                    if candidate_key:
                        candidate_owned_counter[candidate_key] += qty
                        candidate_last_seen_at.setdefault(candidate_key, ts)
                    if last_ts_in_category is None:
                        last_ts_in_category = ts
                    if ts >= since_90d:
                        tx_id = int(row["tx_id"])
                        tx_ids_90d.add(tx_id)
                        tx_amount_90d[tx_id] = float(row["tx_total"])
                        if candidate_key:
                            candidate_seen_90d_counter[candidate_key] += qty

                if item_category == "fragrance":
                    slot_value = str(row.get("slot") or "")
                    if slot_value in SLOTS:
                        slot_counter[slot_value] += qty
                        if len(last_slot_values) < 5:
                            last_slot_values.append(slot_value)

            days_since_last_purchase = -1
            if last_ts_in_category is not None:
                days_since_last_purchase = int((t0.date() - last_ts_in_category.date()).days)

            feature_base: dict[str, Any] = {
                "month_of_year": int(t0.month),
                "day_of_week": int(t0.weekday()),
                "days_since_last_purchase_in_category": int(days_since_last_purchase),
                "tx_count_90d_category": int(len(tx_ids_90d)),
                "tx_amount_90d_category": round(float(sum(tx_amount_90d.values())), 4),
                "owned_slot_warm_day": int(slot_counter.get("warm_day", 0)),
                "owned_slot_warm_evening": int(slot_counter.get("warm_evening", 0)),
                "owned_slot_cold_day": int(slot_counter.get("cold_day", 0)),
                "owned_slot_cold_evening": int(slot_counter.get("cold_evening", 0)),
            }
            for idx in range(5):
                feature_base[f"last{idx + 1}_product_type"] = (
                    str(last_product_types[idx]) if idx < len(last_product_types) else "__none__"
                )
                feature_base[f"last{idx + 1}_category"] = (
                    str(last_categories[idx]) if idx < len(last_categories) else "__none__"
                )

            for col in owned_feature_columns:
                feature_base[col] = 0
            for key, count in owned_counts_all.items():
                mapped = owned_feature_map.get(key)
                if mapped:
                    feature_base[mapped] = int(count)

            episode_records.append(
                {
                    "episode_id": int(episode_id),
                    "group_id": int(episode_id),
                    "user_id": user_id,
                    "category": category,
                    "t0_utc": t0.isoformat().replace("+00:00", "Z"),
                    "label": label,
                    "split": split_name,
                    "candidate_types": list(candidate_types_by_category.get(category) or []),
                    **feature_base,
                }
            )
            recent_candidate_tokens = last_slot_values if category == "fragrance" else last_product_types
            candidate_days_since_last_seen_map: dict[str, int] = {}
            for candidate_key, candidate_ts in candidate_last_seen_at.items():
                candidate_days_since_last_seen_map[str(candidate_key)] = int(
                    (t0.date() - candidate_ts.date()).days
                )
            episode_aux[int(episode_id)] = {
                "recent_candidate_tokens": list(recent_candidate_tokens[:5]),
                "candidate_owned_counter": dict(candidate_owned_counter),
                "candidate_seen_90d_counter": dict(candidate_seen_90d_counter),
                "candidate_days_since_last_seen_map": candidate_days_since_last_seen_map,
            }

        if not episode_records:
            raise CommandError("No episodes produced after filtering.")

        train_users = set(split_map["train"])
        candidate_pop_count_train: dict[str, Counter[str]] = defaultdict(Counter)
        for ep in episode_records:
            if int(ep["user_id"]) not in train_users:
                continue
            if str(ep["label"]) == "__none__":
                continue
            candidate_pop_count_train[str(ep["category"])][str(ep["label"])] += 1

        candidate_popularity_train: dict[str, dict[str, float]] = {}
        for category, candidates in candidate_types_by_category.items():
            counter = candidate_pop_count_train.get(category) or Counter()
            total = float(sum(counter.values()) or 1.0)
            candidate_popularity_train[category] = {
                candidate: round(float(counter.get(candidate, 0)) / total, 8) for candidate in candidates
            }

        rows: list[dict[str, Any]] = []
        label_outside_candidates = 0
        for ep in episode_records:
            category = str(ep["category"])
            candidates = list(ep.get("candidate_types") or [])
            label = str(ep["label"])
            aux = episode_aux.get(int(ep["episode_id"])) or {}
            recent_candidate_tokens = [
                str(x).strip().lower()
                for x in (aux.get("recent_candidate_tokens") or [])
                if str(x).strip()
            ]
            candidate_owned_counter = {
                str(k).strip().lower(): int(v)
                for k, v in (aux.get("candidate_owned_counter") or {}).items()
                if str(k).strip()
            }
            candidate_seen_90d_counter = {
                str(k).strip().lower(): int(v)
                for k, v in (aux.get("candidate_seen_90d_counter") or {}).items()
                if str(k).strip()
            }
            candidate_days_since_last_seen_map = {
                str(k).strip().lower(): int(v)
                for k, v in (aux.get("candidate_days_since_last_seen_map") or {}).items()
                if str(k).strip()
            }
            if label != "__none__" and label not in set(candidates):
                label_outside_candidates += 1
            pos_map = {token: idx for idx, token in enumerate(RULE_CHAIN_BY_CATEGORY.get(category) or [])}
            for candidate in candidates:
                seen_count_last5 = int(sum(1 for token in recent_candidate_tokens if token == candidate))
                row = {
                    "episode_id": int(ep["episode_id"]),
                    "group_id": int(ep["group_id"]),
                    "user_id": int(ep["user_id"]),
                    "category": category,
                    "t0_utc": str(ep["t0_utc"]),
                    "split": str(ep["split"]),
                    "label": label,
                    "candidate_type": str(candidate),
                    "y": int(candidate == label),
                    "candidate_is_fragrance_slot": int(candidate in SLOTS),
                    "candidate_position_in_chain": int(pos_map.get(candidate, -1)),
                    "candidate_popularity_in_train": float(
                        (candidate_popularity_train.get(category) or {}).get(candidate, 0.0)
                    ),
                    "candidate_matches_last1": int(bool(recent_candidate_tokens and recent_candidate_tokens[0] == candidate)),
                    "candidate_matches_last3_any": int(candidate in set(recent_candidate_tokens[:3])),
                    "candidate_seen_count_last5": int(seen_count_last5),
                    "candidate_owned_count_in_category": int(candidate_owned_counter.get(candidate, 0)),
                    "candidate_seen_90d_count_in_category": int(candidate_seen_90d_counter.get(candidate, 0)),
                    "candidate_days_since_last_seen_in_category": int(
                        candidate_days_since_last_seen_map.get(candidate, -1)
                    ),
                }
                for key, value in ep.items():
                    if key in {
                        "episode_id",
                        "group_id",
                        "user_id",
                        "category",
                        "t0_utc",
                        "label",
                        "split",
                        "candidate_types",
                    }:
                        continue
                    row[key] = value
                rows.append(row)

        if not rows:
            raise CommandError("No candidate rows produced.")

        df = pd.DataFrame(rows)
        df = df.sort_values(["episode_id", "candidate_type"]).reset_index(drop=True)
        dataset_format, dataset_file = _write_dataset_frame(df, out_dir)

        class_distribution = _class_distribution_for_splits(episode_records, split_map=split_map)
        positives = int(sum(1 for ep in episode_records if str(ep["label"]) != "__none__"))
        none_total = int(sum(1 for ep in episode_records if str(ep["label"]) == "__none__"))

        baselines = _build_baselines(
            episodes=episode_records,
            split_map=split_map,
            candidate_types_by_category=candidate_types_by_category,
        )

        splits_payload = {
            "seed": seed,
            "strategy": "deterministic_hash_user_level",
            "ratios": {"train": 0.70, "val": 0.15, "test": 0.15},
            "train_user_ids": split_map["train"],
            "val_user_ids": split_map["val"],
            "test_user_ids": split_map["test"],
        }
        splits_path = out_dir / "splits.json"
        splits_path.write_text(json.dumps(splits_payload, ensure_ascii=False, indent=2), encoding="utf-8")

        categorical_features = [
            "category",
            "candidate_type",
            "last1_product_type",
            "last2_product_type",
            "last3_product_type",
            "last4_product_type",
            "last5_product_type",
            "last1_category",
            "last2_category",
            "last3_category",
            "last4_category",
            "last5_category",
        ]
        numeric_features = [
            "month_of_year",
            "day_of_week",
            "days_since_last_purchase_in_category",
            "tx_count_90d_category",
            "tx_amount_90d_category",
            "owned_slot_warm_day",
            "owned_slot_warm_evening",
            "owned_slot_cold_day",
            "owned_slot_cold_evening",
            "candidate_is_fragrance_slot",
            "candidate_position_in_chain",
            "candidate_popularity_in_train",
            "candidate_matches_last1",
            "candidate_matches_last3_any",
            "candidate_seen_count_last5",
            "candidate_owned_count_in_category",
            "candidate_seen_90d_count_in_category",
            "candidate_days_since_last_seen_in_category",
            *owned_feature_columns,
        ]
        feature_columns = [*categorical_features, *numeric_features]

        metadata = {
            "version": "v4_candidate_ranking",
            "generated_at_utc": now_utc.isoformat().replace("+00:00", "Z"),
            "window_days": int(days),
            "window_since_utc": since.isoformat().replace("+00:00", "Z"),
            "window_until_utc": now_utc.isoformat().replace("+00:00", "Z"),
            "label_window_days": int(label_window_days),
            "include_ga": bool(include_ga),
            "dataset_format": dataset_format,
            "dataset_file": dataset_file,
            "rows_total": int(len(df)),
            "episodes_total": int(len(episode_records)),
            "groups_total": int(df["group_id"].nunique()),
            "positive_rows": int(df["y"].sum()),
            "positives": positives,
            "none_count": none_total,
            "none_rate": round(float(none_total / max(1, len(episode_records))), 6),
            "label_outside_candidate_set": int(label_outside_candidates),
            "raw_exposed_events": int(raw_exposed_total),
            "class_distribution": class_distribution,
            "candidate_types_by_category": candidate_types_by_category,
            "rules_chain_by_category": {k: list(v) for k, v in RULE_CHAIN_BY_CATEGORY.items()},
            "top_popularity_by_category": top_popularity_by_category,
            "candidate_popularity_in_train_by_category": candidate_popularity_train,
            "top_owned_product_types_by_category": top_owned_types_by_category,
            "owned_feature_columns": owned_feature_columns,
            "owned_feature_map": {
                col: {"category": cat, "product_type": ptype}
                for (cat, ptype), col in owned_feature_map.items()
            },
            "feature_columns": feature_columns,
            "categorical_features": categorical_features,
            "numeric_features": numeric_features,
            "baselines": baselines,
            "leakage_assertions": {
                "features_only_use_transactions_lte_t0": True,
                "checks_total": int(leakage_checks_total),
                "status": "passed",
            },
        }
        metadata_path = out_dir / "metadata.json"
        metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

        self.stdout.write("[build_roadmap_ml_dataset_v4] done")
        self.stdout.write(f"[build_roadmap_ml_dataset_v4] dataset={dataset_file}")
        self.stdout.write(f"[build_roadmap_ml_dataset_v4] metadata={metadata_path}")
        self.stdout.write(f"[build_roadmap_ml_dataset_v4] splits={splits_path}")
        self.stdout.write(
            "[build_roadmap_ml_dataset_v4] "
            f"episodes={len(episode_records)} positives={positives} none_rate={metadata['none_rate']}"
        )
