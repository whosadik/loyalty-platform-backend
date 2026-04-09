from __future__ import annotations

import json
import math
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

from admin_tools.management.commands.build_roadmap_ml_dataset_v4 import (
    RULE_CHAIN_BY_CATEGORY,
    TARGET_CATEGORIES,
    _class_distribution_for_splits,
    _deterministic_split_user_ids,
    _episode_sample_weight,
    _float_or_zero,
    _is_current_owned_purchase,
    _parse_categories_csv,
    _repo_root,
    _safe_dict,
    _slug_token,
    _write_dataset_frame,
)
from catalog.models import Product
from roadmap_app.content_features import (
    ALL_CATEGORICAL_FEATURES,
    ALL_NUMERIC_FEATURES,
    CHAIN_TRANSITION_NUMERIC_FEATURES,
    NEXTSTEP_PLAN_STATE_CATEGORICAL_FEATURES,
    NEXTSTEP_PLAN_STATE_NUMERIC_FEATURES,
    build_base_content_features,
    build_candidate_catalog_summaries,
    build_candidate_content_features,
    build_chain_transition_features,
    build_nextstep_plan_state_features,
    effective_nextstep_rules_chain,
    product_signature,
    profile_signature,
)
from roadmap_app.fragrance_slots import SLOTS, slot_of_fragrance
from roadmap_app.historical_anchor_replay import build_historical_continuation_anchor_records
from roadmap_app.nextstep_historical_anchor_dataset import (
    TRAIN_EXCLUSION_REASONS,
    bucket_flags_for_row,
    classify_train_exclusion_reason,
    completion_events_by_step,
    generated_candidates_by_product_type,
    resolve_first_completed_generated_candidate,
)
from transactions.models import TransactionItem
from users_app.models import CustomerProfile


DEFAULT_OUT_DIR = Path("data") / "ml" / "roadmap_nextstep_v5_historical_anchor_v1"
LABEL_PROTOCOL_VERSION = "v6_historical_anchor_first_completed_generated_v1"


def _resolve_out_dir(raw_path: str) -> Path:
    candidate = Path(str(raw_path)).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    return (_repo_root() / candidate).resolve()


def _safe_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _to_int(value: Any) -> int | None:
    try:
        if value is None:
            return None
        return int(value)
    except Exception:
        return None


def _evaluate_episode_rankings(
    *,
    episodes: list[dict[str, Any]],
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
        label = str(episode.get("label") or "__none__")
        candidates = [
            str(x).strip().lower()
            for x in _safe_list(episode.get("candidate_types"))
            if str(x).strip()
        ]
        if label == "__none__":
            continue
        positives += 1
        ranked = ranking_fn(episode, candidates)
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


def _build_episode_baselines(
    *,
    episodes: list[dict[str, Any]],
    split_map: dict[str, list[int]],
) -> dict[str, Any]:
    train_users = {int(x) for x in split_map.get("train") or []}
    val_users = {int(x) for x in split_map.get("val") or []}
    test_users = {int(x) for x in split_map.get("test") or []}

    train_episodes = [ep for ep in episodes if int(ep["user_id"]) in train_users]
    val_episodes = [ep for ep in episodes if int(ep["user_id"]) in val_users]
    test_episodes = [ep for ep in episodes if int(ep["user_id"]) in test_users]

    popularity_by_category: dict[str, Counter[str]] = defaultdict(Counter)
    transitions: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    for ep in train_episodes:
        category = str(ep.get("category") or "")
        label = str(ep.get("label") or "__none__")
        if label == "__none__":
            continue
        popularity_by_category[category][label] += 1
        state = str(ep.get("last1_product_type") or "__none__")
        transitions[(category, state)][label] += 1

    def _sort_by_counter(candidates: list[str], counter: Counter[str] | None) -> list[str]:
        counter = counter or Counter()
        return sorted(candidates, key=lambda c: (-int(counter.get(c, 0)), c))

    def _rank_popularity(ep: dict[str, Any], candidates: list[str]) -> list[str]:
        return _sort_by_counter(candidates, popularity_by_category.get(str(ep.get("category") or "")))

    def _rank_markov(ep: dict[str, Any], candidates: list[str]) -> list[str]:
        category = str(ep.get("category") or "")
        state = str(ep.get("last1_product_type") or "__none__")
        counter = transitions.get((category, state)) or popularity_by_category.get(category)
        return _sort_by_counter(candidates, counter)

    return {
        "splits": {
            "val": {
                "popularity": _evaluate_episode_rankings(episodes=val_episodes, ranking_fn=_rank_popularity),
                "markov": _evaluate_episode_rankings(episodes=val_episodes, ranking_fn=_rank_markov),
            },
            "test": {
                "popularity": _evaluate_episode_rankings(episodes=test_episodes, ranking_fn=_rank_popularity),
                "markov": _evaluate_episode_rankings(episodes=test_episodes, ranking_fn=_rank_markov),
            },
        }
    }


class Command(BaseCommand):
    help = (
        "Build Roadmap NextStep v5 historical-anchor ranking dataset: immutable PLAN_REFRESHED anchors, "
        "generated candidates inside the same refresh window, and first completed generated candidate truth."
    )

    def add_arguments(self, parser):
        parser.add_argument("--days", type=int, default=365)
        parser.add_argument("--out-dir", type=str, default=str(DEFAULT_OUT_DIR))
        parser.add_argument(
            "--categories",
            type=str,
            default="",
            help="Comma-separated subset of categories to include, e.g. haircare or skincare,haircare.",
        )
        parser.add_argument("--include-ga", action="store_true", default=False)
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

        days = int(options.get("days") or 365)
        out_dir = _resolve_out_dir(str(options.get("out_dir") or DEFAULT_OUT_DIR))
        include_ga = bool(options.get("include_ga"))
        owned_top_k = int(options.get("owned_top_k") or 20)
        seed = int(options.get("seed") or 42)
        selected_categories = _parse_categories_csv(options.get("categories"), default=sorted(TARGET_CATEGORIES))
        selected_category_set = set(selected_categories)
        if days <= 0:
            raise CommandError("--days must be > 0")
        if owned_top_k <= 0:
            raise CommandError("--owned-top-k must be > 0")

        now_utc = timezone.now().astimezone(dt_timezone.utc)
        since = now_utc - timedelta(days=days)

        self.stdout.write(
            "[build_roadmap_ml_dataset_v5_historical_anchor] "
            f"window={since.isoformat()}..{now_utc.isoformat()} categories={selected_categories}"
        )

        anchors = [
            row
            for row in build_historical_continuation_anchor_records(
                since=since,
                until=now_utc,
                category="all",
                include_ga=include_ga,
            )
            if str(row.get("category") or "").strip().lower() in selected_category_set
        ]
        if not anchors:
            raise CommandError("No historical PLAN_REFRESHED anchors found for selected window/categories.")

        all_generated_step_ids = {
            int(step_id)
            for anchor in anchors
            for step_id in _safe_list(anchor.get("generated_step_ids"))
            if _to_int(step_id) is not None
        }
        completions_by_step = completion_events_by_step(
            since=since,
            until=now_utc,
            step_ids=all_generated_step_ids,
        )

        exclusion_counts: Counter[str] = Counter()
        exclusion_examples: dict[str, list[str]] = defaultdict(list)
        resolved_anchor_records: list[dict[str, Any]] = []
        truth_matched_by_distribution: Counter[str] = Counter()

        for anchor in anchors:
            truth = resolve_first_completed_generated_candidate(anchor, completions_by_step=completions_by_step)
            exclusion_reason = classify_train_exclusion_reason(anchor, truth)
            if exclusion_reason:
                exclusion_counts[exclusion_reason] += 1
                if len(exclusion_examples[exclusion_reason]) < 5:
                    exclusion_examples[exclusion_reason].append(str(anchor.get("anchor_key") or ""))
                continue
            generated_candidates = generated_candidates_by_product_type(anchor)
            if not generated_candidates:
                exclusion_counts["other:no_generated_candidates_after_dedupe"] += 1
                if len(exclusion_examples["other:no_generated_candidates_after_dedupe"]) < 5:
                    exclusion_examples["other:no_generated_candidates_after_dedupe"].append(
                        str(anchor.get("anchor_key") or "")
                    )
                continue
            truth_matched_by = str(truth.get("truth_matched_by") or "").strip().lower() or "__none__"
            truth_matched_by_distribution[truth_matched_by] += 1
            resolved_anchor_records.append(
                {
                    "anchor": anchor,
                    "truth": truth,
                    "generated_candidates": generated_candidates,
                }
            )

        if not resolved_anchor_records:
            raise CommandError("No resolved historical anchors remained after exclusion rules.")

        users = sorted({int(row["anchor"]["user_id"]) for row in resolved_anchor_records})
        profile_map = {
            int(profile.user_id): profile_signature(profile)
            for profile in CustomerProfile.objects.filter(user_id__in=users)
        }

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
            "product__concerns",
            "product__actives",
            "product__flags",
            "product__supported_skin_types",
            "product__attrs",
            "product__ingredients_inci",
            "product__raw_meta",
        )

        user_items: dict[int, list[dict[str, Any]]] = defaultdict(list)
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
                    "concerns": row.get("product__concerns") if isinstance(row.get("product__concerns"), list) else [],
                    "actives": row.get("product__actives") if isinstance(row.get("product__actives"), list) else [],
                    "flags": row.get("product__flags") if isinstance(row.get("product__flags"), list) else [],
                    "supported_skin_types": (
                        row.get("product__supported_skin_types")
                        if isinstance(row.get("product__supported_skin_types"), list)
                        else []
                    ),
                    "attrs": _safe_dict(row.get("product__attrs")),
                    "ingredients_inci": str(row.get("product__ingredients_inci") or ""),
                    "raw_meta": _safe_dict(row.get("product__raw_meta")),
                    "quantity": quantity,
                    "unit_price": _float_or_zero(row.get("unit_price")),
                    "slot": slot_value,
                }
            )
            if category in selected_category_set and product_type and _is_current_owned_purchase(
                category=category,
                product_type=product_type,
                ts=ts,
                ref_ts=now_utc,
            ):
                owned_counter_by_category[category][product_type] += quantity

        for values in user_items.values():
            values.sort(key=lambda x: (x["ts"], int(x["tx_id"]), int(x["item_id"])))

        catalog_rows = list(
            Product.objects.filter(category__in=selected_categories).values(
                "category",
                "product_type",
                "concerns",
                "actives",
                "flags",
                "supported_skin_types",
                "attrs",
                "ingredients_inci",
                "raw_meta",
            )
        )
        candidate_catalog_summaries = build_candidate_catalog_summaries(catalog_rows)

        top_owned_types_by_category: dict[str, list[str]] = {}
        owned_feature_columns: list[str] = []
        owned_feature_map: dict[tuple[str, str], str] = {}
        for category in selected_categories:
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
        category_candidate_union: dict[str, set[str]] = defaultdict(set)

        sorted_payloads = sorted(
            resolved_anchor_records,
            key=lambda row: (
                row["anchor"].get("anchor_created_at"),
                int(row["anchor"].get("anchor_event_id") or 0),
            ),
        )
        for episode_id, payload in enumerate(sorted_payloads, start=1):
            anchor = payload["anchor"]
            truth = payload["truth"]
            generated_candidates = payload["generated_candidates"]
            user_id = int(anchor["user_id"])
            category = str(anchor["category"])
            t0 = anchor["anchor_created_at"].astimezone(dt_timezone.utc)
            split_name = str(split_by_user.get(user_id) or "train")
            truth_product_type = str(truth.get("truth_selected_product_type") or "").strip().lower()
            truth_step_id = int(_to_int(truth.get("truth_selected_candidate_step_id")) or 0)
            truth_matched_by = str(truth.get("truth_matched_by") or "").strip().lower() or "__none__"

            items = user_items.get(user_id) or []
            timeline = timeline_by_user.get(user_id) or []
            pivot = bisect_right(timeline, t0) if timeline else 0
            prior_items = items[:pivot]
            future_items = items[pivot:]
            leakage_checks_total += 1
            if any(row["ts"] > t0 for row in prior_items):
                raise CommandError(f"Leakage detected: prior_items has ts > anchor_created_at for user_id={user_id}")
            if any(row["ts"] <= t0 for row in future_items):
                raise CommandError(f"Leakage detected: future_items has ts <= anchor_created_at for user_id={user_id}")

            last_product_types: list[str] = []
            last_categories: list[str] = []
            recent_candidate_tokens: list[str] = []
            slot_counter: Counter[str] = Counter()
            owned_counts_all: Counter[tuple[str, str]] = Counter()
            candidate_owned_counter: Counter[str] = Counter()
            candidate_seen_90d_counter: Counter[str] = Counter()
            candidate_last_seen_at: dict[str, Any] = {}
            anchor_item: dict[str, Any] | None = None
            last_ts_in_category = None
            tx_ids_90d: set[int] = set()
            tx_amount_90d: dict[int, float] = {}
            since_90d = t0 - timedelta(days=90)
            candidate_types = [str(row.get("product_type") or "") for row in generated_candidates]
            category_candidate_union[category].update(candidate_types)

            for row in reversed(prior_items):
                item_category = str(row["category"] or "")
                item_type = str(row["product_type"] or "")
                qty = int(row["quantity"])
                ts = row["ts"]

                if item_type and len(last_product_types) < 5:
                    last_product_types.append(item_type)
                if item_category and len(last_categories) < 5:
                    last_categories.append(item_category)

                if item_category and item_type and _is_current_owned_purchase(
                    category=item_category,
                    product_type=item_type,
                    ts=ts,
                    ref_ts=t0,
                ):
                    owned_counts_all[(item_category, item_type)] += qty

                if item_category == category:
                    candidate_key = str(item_type or "").strip().lower()
                    if category == "fragrance":
                        slot_value = str(row.get("slot") or "")
                        if slot_value in SLOTS:
                            candidate_key = slot_value
                    if candidate_key:
                        if len(recent_candidate_tokens) < 5:
                            recent_candidate_tokens.append(candidate_key)
                        if _is_current_owned_purchase(
                            category=item_category,
                            product_type=item_type,
                            ts=ts,
                            ref_ts=t0,
                        ):
                            candidate_owned_counter[candidate_key] += qty
                        candidate_last_seen_at.setdefault(candidate_key, ts)
                    if last_ts_in_category is None:
                        last_ts_in_category = ts
                        anchor_item = row
                    if ts >= since_90d:
                        tx_id = int(row["tx_id"])
                        tx_ids_90d.add(tx_id)
                        tx_amount_90d[tx_id] = float(row["tx_total"])
                        if candidate_key:
                            candidate_seen_90d_counter[candidate_key] += qty

                if item_category == "fragrance":
                    slot_value = str(row.get("slot") or "")
                    if slot_value in SLOTS and _is_current_owned_purchase(
                        category=item_category,
                        product_type=item_type,
                        ts=ts,
                        ref_ts=t0,
                    ):
                        slot_counter[slot_value] += qty

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
            feature_base.update(
                build_base_content_features(
                    profile_map.get(user_id),
                    product_signature(anchor_item),
                )
            )

            planned_target_product_type = str(
                anchor.get("anchor_next_product_type")
                or anchor.get("planned_target_product_type")
                or "__none__"
            ).strip().lower() or "__none__"
            planned_target_step_index = int(
                _to_int(anchor.get("anchor_next_step_index"))
                or _to_int(anchor.get("planned_target_step_index"))
                or 0
            )

            episode_records.append(
                {
                    "episode_id": int(episode_id),
                    "group_id": int(episode_id),
                    "user_id": user_id,
                    "plan_id": int(anchor["plan_id"]),
                    "category": category,
                    "anchor_key": str(anchor.get("anchor_key") or ""),
                    "anchor_event_id": int(anchor.get("anchor_event_id") or 0),
                    "anchor_created_at": t0.isoformat().replace("+00:00", "Z"),
                    "anchor_next_step_id": int(_to_int(anchor.get("anchor_next_step_id")) or 0),
                    "anchor_next_step_index": int(_to_int(anchor.get("anchor_next_step_index")) or 0),
                    "anchor_next_product_type": str(anchor.get("anchor_next_product_type") or "__none__"),
                    "anchor_has_actionable_step": int(bool(anchor.get("anchor_has_actionable_step"))),
                    "t0_utc": t0.isoformat().replace("+00:00", "Z"),
                    "label": truth_product_type,
                    "label_source": "step_completed_event",
                    "label_matched_by": truth_matched_by,
                    "label_event_step_id": int(truth_step_id),
                    "truth_selected_candidate_step_id": int(truth_step_id),
                    "truth_selected_product_type": truth_product_type,
                    "truth_matched_by": truth_matched_by,
                    "truth_is_resolved": 1,
                    "truth_ambiguous_multiple_completed_types": int(
                        bool(truth.get("ambiguous_multiple_completed_types"))
                    ),
                    "truth_product_types_in_window": list(_safe_list(truth.get("truth_product_types_in_window"))),
                    "excluded_from_train_reason": "",
                    "split": split_name,
                    "candidate_types": list(candidate_types),
                    "planned_target_product_type": planned_target_product_type,
                    "planned_target_step_index": planned_target_step_index,
                    **feature_base,
                }
            )
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
                "anchor_item": dict(anchor_item) if isinstance(anchor_item, dict) else None,
                "generated_candidates": list(generated_candidates),
            }

        if not episode_records:
            raise CommandError("No resolved episode records produced after historical-anchor filtering.")

        candidate_types_by_category = {
            category: sorted(tokens)
            for category, tokens in sorted(category_candidate_union.items())
        }

        train_users = set(split_map["train"])
        candidate_pop_count_train: dict[str, Counter[str]] = defaultdict(Counter)
        for ep in episode_records:
            if int(ep["user_id"]) not in train_users:
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
            label = str(ep["label"])
            candidates = list(ep.get("candidate_types") or [])
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
            generated_candidates = {
                str(row.get("product_type") or "").strip().lower(): row
                for row in _safe_list(aux.get("generated_candidates"))
                if str(row.get("product_type") or "").strip()
            }
            anchor_sig = product_signature(aux.get("anchor_item"))
            profile_sig = profile_map.get(int(ep["user_id"])) or {}
            anchor_chain_token = (
                recent_candidate_tokens[0]
                if recent_candidate_tokens
                else str(anchor_sig.get("product_type") or "").strip().lower()
            )
            last1_chain_token = recent_candidate_tokens[0] if recent_candidate_tokens else ""
            last2_chain_token = recent_candidate_tokens[1] if len(recent_candidate_tokens) > 1 else ""
            planned_target_product_type = str(ep.get("planned_target_product_type") or "__none__")
            planned_target_step_index = int(ep.get("planned_target_step_index") or 0)

            if label and label != "__none__" and label not in set(candidates):
                label_outside_candidates += 1

            effective_rules_chain = effective_nextstep_rules_chain(
                category=category,
                rules_chain=RULE_CHAIN_BY_CATEGORY.get(category) or [],
                planned_target_product_type=planned_target_product_type,
                profile_sig=profile_sig,
                anchor_product_type=anchor_chain_token,
            )
            pos_map = {token: idx for idx, token in enumerate(effective_rules_chain)}

            for candidate in candidates:
                candidate_payload = _safe_dict(generated_candidates.get(str(candidate)))
                seen_count_last5 = int(sum(1 for token in recent_candidate_tokens if token == candidate))
                sample_weight = _episode_sample_weight(
                    category=category,
                    label=label,
                    label_source=str(ep.get("label_source") or "step_completed_event"),
                    label_matched_by=str(ep.get("label_matched_by") or "__none__"),
                    last1_product_type=last1_chain_token,
                )
                row = {
                    "episode_id": int(ep["episode_id"]),
                    "group_id": int(ep["group_id"]),
                    "user_id": int(ep["user_id"]),
                    "plan_id": int(ep["plan_id"]),
                    "anchor_key": str(ep["anchor_key"]),
                    "category": category,
                    "anchor_event_id": int(ep["anchor_event_id"]),
                    "anchor_created_at": str(ep["anchor_created_at"]),
                    "anchor_next_step_id": int(ep["anchor_next_step_id"]),
                    "anchor_next_step_index": int(ep["anchor_next_step_index"]),
                    "anchor_next_product_type": str(ep["anchor_next_product_type"]),
                    "anchor_has_actionable_step": int(ep["anchor_has_actionable_step"]),
                    "t0_utc": str(ep["t0_utc"]),
                    "split": str(ep["split"]),
                    "label": label,
                    "label_y": int(str(candidate) == label),
                    "y": int(str(candidate) == label),
                    "candidate_type": str(candidate),
                    "candidate_step_id": int(_to_int(candidate_payload.get("step_id")) or 0),
                    "candidate_step_index": int(_to_int(candidate_payload.get("step_index")) or 0),
                    "candidate_product_type": str(candidate),
                    "candidate_is_generated": int(bool(candidate_payload.get("is_generated", True))),
                    "truth_selected_candidate_step_id": int(ep["truth_selected_candidate_step_id"]),
                    "truth_selected_product_type": str(ep["truth_selected_product_type"]),
                    "truth_matched_by": str(ep["truth_matched_by"]),
                    "truth_is_resolved": int(ep["truth_is_resolved"]),
                    "excluded_from_train_reason": "",
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
                    "sample_weight": float(sample_weight),
                }
                row.update(
                    bucket_flags_for_row(
                        category=category,
                        truth_product_type=str(ep["truth_selected_product_type"]),
                        candidate_product_type=str(candidate),
                    )
                )
                row.update(
                    build_candidate_content_features(
                        candidate_catalog_summaries.get((category, str(candidate))),
                        profile_sig,
                        anchor_sig,
                        candidate_type=str(candidate),
                    )
                )
                row.update(
                    build_chain_transition_features(
                        rules_chain=effective_rules_chain,
                        candidate_type=str(candidate),
                        anchor_product_type=anchor_chain_token,
                        last1_product_type=last1_chain_token,
                        last2_product_type=last2_chain_token,
                    )
                )
                row.update(
                    build_nextstep_plan_state_features(
                        rules_chain=effective_rules_chain,
                        candidate_type=str(candidate),
                        planned_target_product_type=planned_target_product_type,
                        planned_target_step_index=planned_target_step_index,
                    )
                )
                for key, value in ep.items():
                    if key in {
                        "episode_id",
                        "group_id",
                        "user_id",
                        "plan_id",
                        "category",
                        "t0_utc",
                        "label",
                        "split",
                        "candidate_types",
                        "truth_product_types_in_window",
                    }:
                        continue
                    row[key] = value
                rows.append(row)

        if not rows:
            raise CommandError("No candidate rows produced for resolved historical anchors.")

        df = pd.DataFrame(rows)
        df = df.sort_values(["episode_id", "candidate_product_type", "candidate_step_index"]).reset_index(drop=True)
        dataset_format, dataset_file = _write_dataset_frame(df, out_dir)

        class_distribution = _class_distribution_for_splits(episode_records, split_map=split_map)
        baselines = _build_episode_baselines(episodes=episode_records, split_map=split_map)
        excluded_total = int(sum(exclusion_counts.values()))
        explicit_exclusion_counts = {
            key: int(exclusion_counts.get(key, 0))
            for key in sorted(TRAIN_EXCLUSION_REASONS)
        }
        other_exclusion_counts = {
            str(k): int(v)
            for k, v in sorted(exclusion_counts.items())
            if str(k) not in TRAIN_EXCLUSION_REASONS
        }

        splits_payload = {
            "seed": seed,
            "strategy": "deterministic_hash_user_level",
            "ratios": {"train": 0.70, "val": 0.15, "test": 0.15},
            "train_user_ids": split_map["train"],
            "val_user_ids": split_map["val"],
            "test_user_ids": split_map["test"],
        }
        (out_dir / "splits.json").write_text(
            json.dumps(splits_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

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
            *NEXTSTEP_PLAN_STATE_CATEGORICAL_FEATURES,
            *ALL_CATEGORICAL_FEATURES,
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
            *CHAIN_TRANSITION_NUMERIC_FEATURES,
            *NEXTSTEP_PLAN_STATE_NUMERIC_FEATURES,
            *owned_feature_columns,
            *ALL_NUMERIC_FEATURES,
        ]
        feature_columns = [*categorical_features, *numeric_features]

        truth_resolution_summary = {
            "anchors_scanned_total": int(len(anchors)),
            "anchors_resolved_for_train": int(len(resolved_anchor_records)),
            "anchors_excluded_from_train": excluded_total,
            "excluded_by_reason": explicit_exclusion_counts,
            "excluded_other_reasons": other_exclusion_counts,
            "example_anchor_keys_by_reason": {
                str(key): list(values) for key, values in sorted(exclusion_examples.items())
            },
        }

        metadata = {
            "version": "v5_historical_anchor_ranking",
            "dataset_builder_command": "build_roadmap_ml_dataset_v5_historical_anchor",
            "label_protocol_version": LABEL_PROTOCOL_VERSION,
            "truth_protocol": {
                "anchor_source": "immutable_plan_refreshed",
                "candidate_source": "step_generated_in_same_refresh_window",
                "positive_label": "first_completed_generated_candidate_in_same_refresh_window",
                "unresolved_anchors_excluded_from_supervised_train": True,
            },
            "generated_at_utc": now_utc.isoformat().replace("+00:00", "Z"),
            "window_days": int(days),
            "window_since_utc": since.isoformat().replace("+00:00", "Z"),
            "window_until_utc": now_utc.isoformat().replace("+00:00", "Z"),
            "selected_categories": selected_categories,
            "include_ga": bool(include_ga),
            "dataset_format": dataset_format,
            "dataset_file": dataset_file,
            "rows_total": int(len(df)),
            "episodes_total": int(len(episode_records)),
            "groups_total": int(df["group_id"].nunique()),
            "positive_rows": int(df["y"].sum()),
            "positives": int(len(episode_records)),
            "none_count": 0,
            "none_rate": 0.0,
            "label_outside_candidate_set": int(label_outside_candidates),
            "label_source_distribution": {"step_completed_event": int(len(episode_records))},
            "label_matched_by_distribution": {
                str(k): int(v)
                for k, v in sorted(truth_matched_by_distribution.items(), key=lambda kv: (-kv[1], kv[0]))
            },
            "raw_anchor_events": int(len(anchors)),
            "truth_resolution_summary": truth_resolution_summary,
            "class_distribution": class_distribution,
            "candidate_types_by_category": candidate_types_by_category,
            "rules_chain_by_category": {k: list(v) for k, v in RULE_CHAIN_BY_CATEGORY.items()},
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
            "dataset_contract": {
                "anchor_identity": [
                    "anchor_key",
                    "plan_id",
                    "user_id",
                    "category",
                    "anchor_event_id",
                    "anchor_created_at",
                ],
                "anchor_state": [
                    "anchor_next_step_id",
                    "anchor_next_step_index",
                    "anchor_next_product_type",
                    "anchor_has_actionable_step",
                ],
                "candidate_identity": [
                    "candidate_step_id",
                    "candidate_step_index",
                    "candidate_product_type",
                    "candidate_is_generated",
                ],
                "truth_label": [
                    "label_y",
                    "truth_selected_candidate_step_id",
                    "truth_selected_product_type",
                    "truth_matched_by",
                    "truth_is_resolved",
                ],
                "targeted_buckets": [
                    "bucket_skincare_mask",
                    "bucket_skincare_toner",
                    "bucket_skincare_eye_cream",
                    "bucket_haircare_shampoo",
                    "bucket_haircare_shampoo_to_conditioner",
                ],
                "protected_buckets": [
                    "protected_haircare_hair_mask",
                    "protected_haircare_hair_oil",
                    "protected_skincare_essence",
                    "analysis_fragrance_cold_evening",
                ],
                "diagnostics": ["excluded_from_train_reason"],
            },
            "sample_weight_policy": {
                "type": "historical_anchor_base",
                "step_completed_event_weight": 1.15,
                "recommended_product_id_multiplier": 1.10,
                "semantic_content_match_multiplier": 1.05,
            },
            "sample_weight_summary": {
                "min": round(float(df["sample_weight"].min()), 6),
                "max": round(float(df["sample_weight"].max()), 6),
                "mean": round(float(df["sample_weight"].mean()), 6),
            },
            "baselines": baselines,
            "leakage_assertions": {
                "features_only_use_transactions_lte_anchor_created_at": True,
                "checks_total": int(leakage_checks_total),
                "status": "passed",
            },
            "notes": [
                "Training rows include only resolved historical anchors from immutable PLAN_REFRESHED windows.",
                "Candidate rows are limited to STEP_GENERATED candidates inside the same refresh window.",
                "Unresolved anchors are excluded from supervised train but counted in truth_resolution_summary.",
            ],
        }
        (out_dir / "metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        self.stdout.write("[build_roadmap_ml_dataset_v5_historical_anchor] done")
        self.stdout.write(f"[build_roadmap_ml_dataset_v5_historical_anchor] out={out_dir}")
        self.stdout.write(
            "[build_roadmap_ml_dataset_v5_historical_anchor] "
            f"anchors_scanned={len(anchors)} resolved={len(resolved_anchor_records)} excluded={excluded_total}"
        )
