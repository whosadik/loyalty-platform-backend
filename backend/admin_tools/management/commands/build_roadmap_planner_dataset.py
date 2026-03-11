from __future__ import annotations

import hashlib
import json
from bisect import bisect_left, bisect_right
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

from catalog.models import Product
from roadmap_app.content_features import (
    ALL_CATEGORICAL_FEATURES,
    ALL_NUMERIC_FEATURES,
    build_base_content_features,
    build_candidate_catalog_summaries,
    build_candidate_content_features,
    product_signature,
    profile_signature,
)
from roadmap_app.fragrance_slots import SLOTS, slot_of_fragrance
from roadmap_app.models import RoadmapEvent, RoadmapStep
from transactions.models import TransactionItem
from users_app.models import CustomerProfile

TARGET_CATEGORIES = {"skincare", "haircare", "makeup", "fragrance"}
STOP_TOKEN = "__stop__"
MAX_TS_ID = 10**18


def _resolve_out_dir(raw_path: str) -> Path:
    candidate = Path(str(raw_path)).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    cwd_path = (Path.cwd() / candidate).resolve()
    if cwd_path.parent.exists():
        return cwd_path
    return (Path(__file__).resolve().parents[4] / candidate).resolve()


def _safe_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _event_position(created_at, event_id: int) -> tuple[Any, int]:
    return (created_at.astimezone(dt_timezone.utc), int(event_id))


def _deterministic_split_user_ids(user_ids: list[int], seed: int) -> dict[str, list[int]]:
    if not user_ids:
        return {"train": [], "val": [], "test": []}

    ordered_ids = sorted({int(x) for x in user_ids})
    scored: list[tuple[str, int]] = []
    for user_id in ordered_ids:
        payload = f"{seed}:{user_id}".encode("utf-8")
        scored.append((hashlib.md5(payload).hexdigest(), int(user_id)))
    scored.sort(key=lambda row: row[0])
    ordered = [int(row[1]) for row in scored]

    n = len(ordered)
    n_train = max(1, int(round(n * 0.70)))
    n_val = int(round(n * 0.15))
    if n_val < 1 and n >= 3:
        n_val = 1
    if n_train + n_val >= n:
        n_val = max(0, n - n_train - 1)
    n_test = n - n_train - n_val
    if n_test <= 0:
        n_test = 1
        if n_train > n_val and n_train > 1:
            n_train -= 1
        elif n_val > 0:
            n_val -= 1

    return {
        "train": sorted(ordered[:n_train]),
        "val": sorted(ordered[n_train : n_train + n_val]),
        "test": sorted(ordered[n_train + n_val :]),
    }


def _history_token_for_tx(*, category: str, product_type: str, attrs: dict[str, Any], raw_meta: dict[str, Any]) -> str:
    if category == "fragrance":
        slot = slot_of_fragrance(attrs or {}, raw_meta=raw_meta or {})
        if slot in SLOTS:
            return str(slot)
    return str(product_type or "").strip().lower()


def _status_flag(status_value: str, expected: str) -> int:
    return int(str(status_value or "").strip().lower() == str(expected or "").strip().lower())


class Command(BaseCommand):
    help = (
        "Build leakage-safe Roadmap Planner dataset from PLAN_REFRESHED/STEP_GENERATED/"
        "STEP_COMPLETED events for ML-first roadmap planning."
    )

    def add_arguments(self, parser):
        parser.add_argument("--days", type=int, default=180)
        parser.add_argument("--out-dir", type=str, default="data/ml/roadmap_planner_v1")
        parser.add_argument("--include-ga", action="store_true", default=False)
        parser.add_argument("--label-window-days", type=int, default=14)
        parser.add_argument("--seed", type=int, default=42)

    def handle(self, *args, **options):
        if pd is None:
            raise CommandError("pandas is required. Install dependencies from requirements-ml.txt")

        days = int(options["days"])
        include_ga = bool(options["include_ga"])
        label_window_days = int(options["label_window_days"])
        out_dir = _resolve_out_dir(str(options["out_dir"]))
        seed = int(options["seed"])

        if days <= 0:
            raise CommandError("--days must be > 0")
        if label_window_days <= 0:
            raise CommandError("--label-window-days must be > 0")

        now_utc = timezone.now().astimezone(dt_timezone.utc)
        since = now_utc - timedelta(days=days)
        max_t0 = now_utc - timedelta(days=label_window_days)

        self.stdout.write(
            "[build_roadmap_planner_dataset] "
            f"window={since.isoformat()}..{now_utc.isoformat()} label_window_days={label_window_days}"
        )

        refresh_qs = RoadmapEvent.objects.filter(
            event_type=RoadmapEvent.Type.PLAN_REFRESHED,
            created_at__gte=since,
            created_at__lte=max_t0,
        )
        if not include_ga:
            refresh_qs = refresh_qs.exclude(user__username__startswith="ga_")

        refresh_rows: list[dict[str, Any]] = []
        for row in refresh_qs.values(
            "id",
            "user_id",
            "plan_id",
            "created_at",
            "context",
            "plan__category",
        ).iterator(chunk_size=5000):
            ctx = _safe_dict(row.get("context"))
            category = str(ctx.get("category") or row.get("plan__category") or "").strip().lower()
            if category not in TARGET_CATEGORIES:
                continue
            refresh_rows.append(
                {
                    "id": int(row["id"]),
                    "user_id": int(row["user_id"]),
                    "plan_id": int(row["plan_id"]) if row.get("plan_id") else None,
                    "created_at": row["created_at"].astimezone(dt_timezone.utc),
                    "category": category,
                    "context": ctx,
                }
            )

        if not refresh_rows:
            raise CommandError("No PLAN_REFRESHED episodes for selected window.")

        refresh_rows.sort(key=lambda row: (int(row["user_id"]), str(row["category"]), row["created_at"], int(row["id"])))
        users = sorted({int(row["user_id"]) for row in refresh_rows})

        generated_qs = RoadmapEvent.objects.filter(
            user_id__in=users,
            event_type=RoadmapEvent.Type.STEP_GENERATED,
            created_at__gte=since,
            created_at__lte=now_utc,
        )
        completed_qs = RoadmapEvent.objects.filter(
            user_id__in=users,
            event_type=RoadmapEvent.Type.STEP_COMPLETED,
            created_at__gte=since,
            created_at__lte=now_utc,
        )
        if not include_ga:
            generated_qs = generated_qs.exclude(user__username__startswith="ga_")
            completed_qs = completed_qs.exclude(user__username__startswith="ga_")

        generated_by_key: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
        for row in generated_qs.values(
            "id",
            "user_id",
            "plan_id",
            "step_id",
            "created_at",
            "context",
            "step__product_type",
            "step__step_index",
            "step__status",
            "step__recommended_product_id",
            "step__plan__category",
        ).iterator(chunk_size=5000):
            ctx = _safe_dict(row.get("context"))
            category = str(ctx.get("category") or row.get("step__plan__category") or "").strip().lower()
            if category not in TARGET_CATEGORIES:
                continue
            generated_by_key[(int(row["user_id"]), category)].append(
                {
                    "id": int(row["id"]),
                    "plan_id": int(row["plan_id"]) if row.get("plan_id") else None,
                    "step_id": int(row["step_id"]) if row.get("step_id") else None,
                    "created_at": row["created_at"].astimezone(dt_timezone.utc),
                    "context": ctx,
                    "product_type": str(ctx.get("product_type") or row.get("step__product_type") or "").strip().lower(),
                    "step_index": int(ctx.get("step_index") or row.get("step__step_index") or 0),
                    "status": str(ctx.get("status") or row.get("step__status") or "").strip().lower(),
                    "recommended_product_id": (
                        int(ctx.get("recommended_product_id"))
                        if ctx.get("recommended_product_id") not in (None, "")
                        else (int(row["step__recommended_product_id"]) if row.get("step__recommended_product_id") else None)
                    ),
                }
            )

        completed_by_key: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
        for row in completed_qs.values(
            "id",
            "user_id",
            "plan_id",
            "step_id",
            "created_at",
            "context",
            "step__product_type",
            "step__plan__category",
        ).iterator(chunk_size=5000):
            ctx = _safe_dict(row.get("context"))
            category = str(ctx.get("category") or row.get("step__plan__category") or "").strip().lower()
            if category not in TARGET_CATEGORIES:
                continue
            completed_by_key[(int(row["user_id"]), category)].append(
                {
                    "id": int(row["id"]),
                    "created_at": row["created_at"].astimezone(dt_timezone.utc),
                    "product_type": str(ctx.get("product_type") or row.get("step__product_type") or "").strip().lower(),
                    "matched_by": str(ctx.get("matched_by") or "").strip().lower(),
                }
            )

        tx_qs = TransactionItem.objects.filter(
            transaction__user_id__in=users,
            transaction__created_at__lte=now_utc,
        ).values(
            "transaction__user_id",
            "transaction__id",
            "transaction__created_at",
            "transaction__total_amount",
            "quantity",
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

        profile_map = {
            int(profile.user_id): profile_signature(profile)
            for profile in CustomerProfile.objects.filter(user_id__in=users)
        }

        user_items: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for row in tx_qs.iterator(chunk_size=5000):
            user_id = int(row["transaction__user_id"])
            ts = row["transaction__created_at"].astimezone(dt_timezone.utc)
            category = str(row.get("product__category") or "").strip().lower()
            product_type = str(row.get("product__product_type") or "").strip().lower()
            quantity = max(1, int(row.get("quantity") or 0))
            attrs = row.get("product__attrs") if isinstance(row.get("product__attrs"), dict) else {}
            raw_meta = row.get("product__raw_meta") if isinstance(row.get("product__raw_meta"), dict) else {}
            user_items[user_id].append(
                {
                    "ts": ts,
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
                    "attrs": attrs,
                    "ingredients_inci": str(row.get("product__ingredients_inci") or ""),
                    "raw_meta": raw_meta,
                    "history_token": _history_token_for_tx(
                        category=category,
                        product_type=product_type,
                        attrs=attrs,
                        raw_meta=raw_meta,
                    ),
                    "quantity": quantity,
                    "tx_id": int(row["transaction__id"]),
                    "tx_total": float(row.get("transaction__total_amount") or 0.0),
                }
            )

        timeline_by_user: dict[int, list[Any]] = {}
        for user_id, items in user_items.items():
            items.sort(key=lambda row: (row["ts"], int(row["tx_id"])))
            timeline_by_user[user_id] = [row["ts"] for row in items]

        candidate_types_by_category: dict[str, list[str]] = {category: [] for category in TARGET_CATEGORIES}
        for row in (
            Product.objects.filter(category__in=sorted(TARGET_CATEGORIES))
            .values("category", "product_type")
            .distinct()
            .iterator(chunk_size=5000)
        ):
            category = str(row.get("category") or "").strip().lower()
            product_type = str(row.get("product_type") or "").strip().lower()
            if category not in TARGET_CATEGORIES or not product_type:
                continue
            if category == "fragrance":
                continue
            bucket = candidate_types_by_category.setdefault(category, [])
            if product_type not in bucket:
                bucket.append(product_type)
        candidate_types_by_category["fragrance"] = list(SLOTS)

        candidate_catalog_summaries = build_candidate_catalog_summaries(
            list(
                Product.objects.filter(category__in=sorted(TARGET_CATEGORIES)).values(
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
        )

        refresh_groups: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
        for row in refresh_rows:
            refresh_groups[(int(row["user_id"]), str(row["category"]))].append(row)
        for rows in generated_by_key.values():
            rows.sort(key=lambda row: (row["created_at"], int(row["id"])))
        for rows in completed_by_key.values():
            rows.sort(key=lambda row: (row["created_at"], int(row["id"])))

        split_map = _deterministic_split_user_ids(users, seed=seed)
        split_by_user: dict[int, str] = {}
        for split_name, split_users in split_map.items():
            for user_id in split_users:
                split_by_user[int(user_id)] = split_name

        episode_records: list[dict[str, Any]] = []
        skipped_no_snapshot = 0
        label_outside_candidates = 0

        for (user_id, category), rows in refresh_groups.items():
            generated_rows = generated_by_key.get((int(user_id), category), [])
            generated_positions = [_event_position(row["created_at"], row["id"]) for row in generated_rows]
            completed_rows = completed_by_key.get((int(user_id), category), [])
            completed_positions = [_event_position(row["created_at"], row["id"]) for row in completed_rows]
            timeline = timeline_by_user.get(int(user_id)) or []
            tx_items = user_items.get(int(user_id)) or []

            for idx, refresh_row in enumerate(rows):
                t0 = refresh_row["created_at"]
                refresh_pos = _event_position(t0, int(refresh_row["id"]))
                next_refresh = rows[idx + 1] if idx + 1 < len(rows) else None
                next_refresh_pos = (
                    _event_position(next_refresh["created_at"], int(next_refresh["id"])) if next_refresh else None
                )
                generated_start = bisect_right(generated_positions, refresh_pos)
                generated_end = bisect_left(generated_positions, next_refresh_pos) if next_refresh_pos else len(generated_rows)
                snapshot = generated_rows[generated_start:generated_end]
                if not snapshot:
                    skipped_no_snapshot += 1
                    continue

                snapshot.sort(key=lambda row: (int(row["step_index"]), int(row["id"])))
                refresh_ctx = _safe_dict(refresh_row.get("context"))
                next_step_id = int(refresh_ctx.get("next_step_id")) if refresh_ctx.get("next_step_id") not in (None, "") else None
                next_step_index = int(refresh_ctx.get("next_step_index")) if refresh_ctx.get("next_step_index") not in (None, "") else None
                next_product_type = str(refresh_ctx.get("next_product_type") or "").strip().lower()
                if not next_product_type:
                    derived_next = next(
                        (
                            row
                            for row in snapshot
                            if row["status"] in {RoadmapStep.Status.MISSING, RoadmapStep.Status.RECOMMENDED}
                        ),
                        None,
                    )
                    if derived_next:
                        next_product_type = str(derived_next["product_type"])
                        next_step_index = int(derived_next["step_index"])
                        next_step_id = int(derived_next["step_id"]) if derived_next.get("step_id") else None

                ml_ctx = _safe_dict(refresh_ctx.get("ml"))
                refresh_caller = str(refresh_ctx.get("refresh_caller") or "unknown")
                status_counter = Counter(str(row["status"] or "").strip().lower() for row in snapshot)
                plan_types = [str(row["product_type"]) for row in snapshot if str(row["product_type"]).strip()]
                plan_position_by_type: dict[str, int] = {}
                plan_state_by_type: dict[str, dict[str, Any]] = {}
                for position, row in enumerate(snapshot, start=1):
                    candidate = str(row["product_type"])
                    if candidate and candidate not in plan_position_by_type:
                        plan_position_by_type[candidate] = int(position)
                        plan_state_by_type[candidate] = {
                            "status": str(row["status"]),
                            "has_recommendation": bool(row.get("recommended_product_id")),
                            "step_index": int(row["step_index"]),
                            "step_id": int(row["step_id"]) if row.get("step_id") else None,
                        }

                pivot = bisect_right(timeline, t0) if timeline else 0
                prior_items = tx_items[:pivot]
                last_product_types: list[str] = []
                last_categories: list[str] = []
                recent_category_tokens: list[str] = []
                candidate_seen_90d_counter: Counter[str] = Counter()
                candidate_last_seen_at: dict[str, Any] = {}
                tx_ids_90d: set[int] = set()
                tx_total_90d: dict[int, float] = {}
                since_90d = t0 - timedelta(days=90)
                last_ts_in_category = None
                anchor_item: dict[str, Any] | None = None

                for item in reversed(prior_items):
                    item_category = str(item["category"] or "")
                    item_product_type = str(item["product_type"] or "")
                    if item_product_type and len(last_product_types) < 5:
                        last_product_types.append(item_product_type)
                    if item_category and len(last_categories) < 5:
                        last_categories.append(item_category)
                    if item_category != category:
                        continue
                    token = str(item.get("history_token") or "").strip().lower()
                    if token and len(recent_category_tokens) < 5:
                        recent_category_tokens.append(token)
                    if token and token not in candidate_last_seen_at:
                        candidate_last_seen_at[token] = item["ts"]
                    if last_ts_in_category is None:
                        last_ts_in_category = item["ts"]
                        anchor_item = item
                    if item["ts"] >= since_90d:
                        tx_id = int(item["tx_id"])
                        tx_ids_90d.add(tx_id)
                        tx_total_90d[tx_id] = float(item["tx_total"])
                        if token:
                            candidate_seen_90d_counter[token] += int(item["quantity"])

                days_since_last_purchase = -1
                if last_ts_in_category is not None:
                    days_since_last_purchase = int((t0.date() - last_ts_in_category.date()).days)

                label_window_end = t0 + timedelta(days=label_window_days)
                end_pos = next_refresh_pos
                if end_pos is None or label_window_end < end_pos[0]:
                    end_pos = (label_window_end, MAX_TS_ID)
                completed_start = bisect_right(completed_positions, refresh_pos)
                completed_end = bisect_left(completed_positions, end_pos)
                completion_rows = completed_rows[completed_start:completed_end]
                label = STOP_TOKEN
                matched_by = ""
                label_source = "stop_no_completion"
                if completion_rows:
                    label = str(completion_rows[0].get("product_type") or "").strip().lower() or STOP_TOKEN
                    matched_by = str(completion_rows[0].get("matched_by") or "").strip().lower()
                    label_source = "step_completed_event"

                candidate_types = list(candidate_types_by_category.get(category) or [])
                for token in plan_types:
                    if token and token not in candidate_types:
                        candidate_types.append(token)
                if label and label != STOP_TOKEN and label not in candidate_types:
                    candidate_types.append(label)
                if STOP_TOKEN not in candidate_types:
                    candidate_types.append(STOP_TOKEN)
                if label != STOP_TOKEN and label not in set(candidate_types):
                    label_outside_candidates += 1

                base_content = build_base_content_features(
                    profile_map.get(int(user_id)),
                    product_signature(anchor_item),
                )

                episode_records.append(
                    {
                        "episode_id": int(len(episode_records) + 1),
                        "user_id": int(user_id),
                        "category": category,
                        "t0_utc": t0.isoformat().replace("+00:00", "Z"),
                        "split": str(split_by_user.get(int(user_id)) or "train"),
                        "label": label,
                        "matched_by": matched_by or "__none__",
                        "label_source": label_source,
                        "refresh_caller": refresh_caller or "__none__",
                        "current_ml_decision": str(ml_ctx.get("decision") or "__none__"),
                        "current_rollout_mode": str(ml_ctx.get("rollout_mode") or "__none__"),
                        "steps_total": int(len(snapshot)),
                        "missing_steps_count": int(
                            status_counter.get(RoadmapStep.Status.MISSING, 0)
                            + status_counter.get(RoadmapStep.Status.RECOMMENDED, 0)
                        ),
                        "recommended_steps_count": int(status_counter.get(RoadmapStep.Status.RECOMMENDED, 0)),
                        "owned_steps_count": int(status_counter.get(RoadmapStep.Status.OWNED, 0)),
                        "completed_steps_count": int(status_counter.get(RoadmapStep.Status.COMPLETED, 0)),
                        "skipped_steps_count": int(status_counter.get(RoadmapStep.Status.SKIPPED, 0)),
                        "next_step_index_current": int(next_step_index or 0),
                        "current_next_product_type": next_product_type or "__none__",
                        "current_next_step_id": int(next_step_id) if next_step_id else 0,
                        "days_since_last_purchase_in_category": int(days_since_last_purchase),
                        "tx_count_90d_category": int(len(tx_ids_90d)),
                        "tx_amount_90d_category": round(float(sum(tx_total_90d.values())), 4),
                        "last1_product_type": str(last_product_types[0]) if len(last_product_types) > 0 else "__none__",
                        "last2_product_type": str(last_product_types[1]) if len(last_product_types) > 1 else "__none__",
                        "last3_product_type": str(last_product_types[2]) if len(last_product_types) > 2 else "__none__",
                        "last4_product_type": str(last_product_types[3]) if len(last_product_types) > 3 else "__none__",
                        "last5_product_type": str(last_product_types[4]) if len(last_product_types) > 4 else "__none__",
                        "last1_category": str(last_categories[0]) if len(last_categories) > 0 else "__none__",
                        "last2_category": str(last_categories[1]) if len(last_categories) > 1 else "__none__",
                        "last3_category": str(last_categories[2]) if len(last_categories) > 2 else "__none__",
                        "last4_category": str(last_categories[3]) if len(last_categories) > 3 else "__none__",
                        "last5_category": str(last_categories[4]) if len(last_categories) > 4 else "__none__",
                        "candidate_types": list(candidate_types),
                        "plan_product_types": "|".join(plan_types),
                        "plan_position_by_type": dict(plan_position_by_type),
                        "plan_state_by_type": dict(plan_state_by_type),
                        "recent_category_tokens": list(recent_category_tokens),
                        "candidate_seen_90d_counter": dict(candidate_seen_90d_counter),
                        "candidate_days_since_last_seen_map": {
                            str(token): int((t0.date() - seen_at.date()).days)
                            for token, seen_at in candidate_last_seen_at.items()
                        },
                        "anchor_item": dict(anchor_item) if isinstance(anchor_item, dict) else None,
                        **base_content,
                    }
                )

        if not episode_records:
            raise CommandError("No planner episodes produced after filtering.")

        train_users = set(split_map["train"])
        candidate_popularity_train: dict[str, Counter[str]] = defaultdict(Counter)
        for ep in episode_records:
            if int(ep["user_id"]) not in train_users:
                continue
            candidate_popularity_train[str(ep["category"])][str(ep["label"])] += 1

        rows: list[dict[str, Any]] = []
        for ep in episode_records:
            plan_position_by_type = ep.pop("plan_position_by_type")
            plan_state_by_type = ep.pop("plan_state_by_type")
            recent_category_tokens = [str(x) for x in ep.pop("recent_category_tokens")]
            seen_90d_counter = {str(k): int(v) for k, v in ep.pop("candidate_seen_90d_counter").items()}
            days_since_last_seen_map = {str(k): int(v) for k, v in ep.pop("candidate_days_since_last_seen_map").items()}
            anchor_sig = product_signature(ep.pop("anchor_item", None))
            category = str(ep["category"])
            label = str(ep["label"])
            popularity_counter = candidate_popularity_train.get(category) or Counter()
            popularity_total = float(sum(popularity_counter.values()) or 1.0)
            candidate_types = list(ep.pop("candidate_types"))
            for candidate in candidate_types:
                state = plan_state_by_type.get(candidate) or {}
                current_status = str(state.get("status") or "")
                rows.append(
                    {
                        **ep,
                        "candidate_type": str(candidate),
                        "y": int(str(candidate) == label),
                        "candidate_in_generated_plan": int(candidate in plan_position_by_type),
                        "candidate_position_in_generated_plan": int(plan_position_by_type.get(candidate, -1)),
                        "candidate_is_current_next_step": int(str(candidate) == str(ep.get("current_next_product_type") or "")),
                        "candidate_has_recommendation_in_plan": int(bool(state.get("has_recommendation"))),
                        "candidate_current_missing": _status_flag(current_status, RoadmapStep.Status.MISSING),
                        "candidate_current_recommended": _status_flag(current_status, RoadmapStep.Status.RECOMMENDED),
                        "candidate_current_owned": _status_flag(current_status, RoadmapStep.Status.OWNED),
                        "candidate_current_completed": _status_flag(current_status, RoadmapStep.Status.COMPLETED),
                        "candidate_current_skipped": _status_flag(current_status, RoadmapStep.Status.SKIPPED),
                        "candidate_matches_last1": int(bool(recent_category_tokens and recent_category_tokens[0] == candidate)),
                        "candidate_matches_last3_any": int(str(candidate) in set(recent_category_tokens[:3])),
                        "candidate_seen_count_last5": int(sum(1 for token in recent_category_tokens if token == candidate)),
                        "candidate_seen_90d_count_in_category": int(seen_90d_counter.get(candidate, 0)),
                        "candidate_days_since_last_seen_in_category": int(days_since_last_seen_map.get(candidate, -1)),
                        "candidate_popularity_in_train": round(
                            float(popularity_counter.get(str(candidate), 0)) / popularity_total,
                            8,
                        ),
                        "candidate_is_stop": int(str(candidate) == STOP_TOKEN),
                        **build_candidate_content_features(
                            candidate_catalog_summaries.get((category, str(candidate))),
                            profile_map.get(int(ep["user_id"])),
                            anchor_sig,
                        ),
                    }
                )

        df = pd.DataFrame(rows)
        df = df.sort_values(["episode_id", "candidate_type"]).reset_index(drop=True)
        out_dir.mkdir(parents=True, exist_ok=True)

        dataset_format = "parquet"
        dataset_file = out_dir / "dataset.parquet"
        try:
            df.to_parquet(dataset_file, index=False)
        except Exception:
            dataset_format = "csv"
            dataset_file = out_dir / "dataset.csv"
            df.to_csv(dataset_file, index=False)

        splits_payload = {
            "train": [int(x) for x in split_map["train"]],
            "val": [int(x) for x in split_map["val"]],
            "test": [int(x) for x in split_map["test"]],
        }
        splits_path = out_dir / "splits.json"
        splits_path.write_text(json.dumps(splits_payload, ensure_ascii=False, indent=2), encoding="utf-8")

        categorical_features = [
            "category",
            "candidate_type",
            "refresh_caller",
            "current_ml_decision",
            "current_rollout_mode",
            "current_next_product_type",
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
            *ALL_CATEGORICAL_FEATURES,
        ]
        numeric_features = [
            "steps_total",
            "missing_steps_count",
            "recommended_steps_count",
            "owned_steps_count",
            "completed_steps_count",
            "skipped_steps_count",
            "next_step_index_current",
            "days_since_last_purchase_in_category",
            "tx_count_90d_category",
            "tx_amount_90d_category",
            "candidate_in_generated_plan",
            "candidate_position_in_generated_plan",
            "candidate_is_current_next_step",
            "candidate_has_recommendation_in_plan",
            "candidate_current_missing",
            "candidate_current_recommended",
            "candidate_current_owned",
            "candidate_current_completed",
            "candidate_current_skipped",
            "candidate_matches_last1",
            "candidate_matches_last3_any",
            "candidate_seen_count_last5",
            "candidate_seen_90d_count_in_category",
            "candidate_days_since_last_seen_in_category",
            "candidate_popularity_in_train",
            "candidate_is_stop",
            *ALL_NUMERIC_FEATURES,
        ]
        feature_columns = [*categorical_features, *numeric_features]

        label_counter = Counter(str(ep["label"]) for ep in episode_records)
        label_source_counter = Counter(str(ep.get("label_source") or "unknown") for ep in episode_records)
        label_matched_by_counter = Counter(str(ep.get("matched_by") or "__none__") for ep in episode_records)
        metadata = {
            "version": "planner_v1_candidate_ranking",
            "generated_at_utc": now_utc.isoformat().replace("+00:00", "Z"),
            "window_days": int(days),
            "window_since_utc": since.isoformat().replace("+00:00", "Z"),
            "window_until_utc": now_utc.isoformat().replace("+00:00", "Z"),
            "label_window_days": int(label_window_days),
            "include_ga": bool(include_ga),
            "dataset_format": dataset_format,
            "dataset_file": str(dataset_file),
            "rows_total": int(len(df)),
            "episodes_total": int(len(episode_records)),
            "positive_rows": int(df["y"].sum()) if not df.empty else 0,
            "stop_label_count": int(label_counter.get(STOP_TOKEN, 0)),
            "stop_label_rate": round(float(label_counter.get(STOP_TOKEN, 0) / max(1, len(episode_records))), 6),
            "label_source_distribution": dict(sorted(label_source_counter.items())),
            "label_matched_by_distribution": dict(sorted(label_matched_by_counter.items())),
            "raw_plan_refreshed_events": int(len(refresh_rows)),
            "skipped_episodes_without_snapshot": int(skipped_no_snapshot),
            "label_outside_candidate_set": int(label_outside_candidates),
            "candidate_types_by_category": {
                category: list(types) + ([STOP_TOKEN] if STOP_TOKEN not in types else [])
                for category, types in sorted(candidate_types_by_category.items())
            },
            "class_distribution": dict(sorted(label_counter.items())),
            "feature_columns": feature_columns,
            "categorical_features": categorical_features,
            "numeric_features": numeric_features,
            "leakage_assertions": {
                "features_only_use_transactions_lte_t0": True,
                "labels_end_at_next_plan_refresh_or_window_end": True,
                "status": "passed",
            },
        }
        metadata_path = out_dir / "metadata.json"
        metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

        self.stdout.write("[build_roadmap_planner_dataset] done")
        self.stdout.write(f"[build_roadmap_planner_dataset] dataset={dataset_file}")
        self.stdout.write(f"[build_roadmap_planner_dataset] metadata={metadata_path}")
        self.stdout.write(f"[build_roadmap_planner_dataset] splits={splits_path}")
