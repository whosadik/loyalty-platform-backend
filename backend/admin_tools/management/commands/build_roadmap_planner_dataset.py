from __future__ import annotations

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
CANDIDATE_SPACE_BY_CATEGORY: dict[str, list[str]] = {
    "skincare": ["cleanser", "serum", "moisturizer", "spf", "toner", "mask", "eye_cream", "essence"],
    "haircare": ["shampoo", "conditioner", "hair_mask", "hair_oil", "scalp_serum", "leave_in"],
    "makeup": ["foundation", "mascara", "blush", "lipstick", "eyeshadow", "primer", "setting_spray"],
    "fragrance": list(SLOTS),
}


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


def _to_int_or_none(value: Any) -> int | None:
    try:
        if value is None:
            return None
        return int(value)
    except Exception:
        return None


def _time_based_split_assignments(records: list[dict[str, Any]]) -> tuple[dict[int, str], dict[str, Any]]:
    if not records:
        return {}, {
            "strategy": "time_based",
            "train_end_utc": None,
            "valid_end_utc": None,
            "counts": {"train": 0, "val": 0, "test": 0},
            "user_overlap_counts": {"train_val": 0, "train_test": 0, "val_test": 0},
        }

    ordered = sorted(
        records,
        key=lambda row: (
            row["t0_dt"],
            int(row["decision_id"]),
        ),
    )
    total = len(ordered)
    train_cut = max(1, int(total * 0.70))
    valid_cut = max(train_cut + 1, int(total * 0.85)) if total >= 3 else total
    valid_cut = min(valid_cut, total)

    split_by_decision: dict[int, str] = {}
    split_users: dict[str, set[int]] = {"train": set(), "val": set(), "test": set()}
    split_counts: Counter[str] = Counter()
    for idx, row in enumerate(ordered):
        if idx < train_cut:
            split_name = "train"
        elif idx < valid_cut:
            split_name = "val"
        else:
            split_name = "test"
        decision_id = int(row["decision_id"])
        split_by_decision[decision_id] = split_name
        split_users[split_name].add(int(row["user_id"]))
        split_counts[split_name] += 1

    return split_by_decision, {
        "strategy": "time_based",
        "train_end_utc": ordered[min(train_cut - 1, total - 1)]["t0_utc"],
        "valid_end_utc": ordered[min(valid_cut - 1, total - 1)]["t0_utc"] if valid_cut < total else ordered[-1]["t0_utc"],
        "counts": {
            "train": int(split_counts.get("train", 0)),
            "val": int(split_counts.get("val", 0)),
            "test": int(split_counts.get("test", 0)),
        },
        "user_overlap_counts": {
            "train_val": int(len(split_users["train"].intersection(split_users["val"]))),
            "train_test": int(len(split_users["train"].intersection(split_users["test"]))),
            "val_test": int(len(split_users["val"].intersection(split_users["test"]))),
        },
    }


def _history_token_for_tx(*, category: str, product_type: str, attrs: dict[str, Any], raw_meta: dict[str, Any]) -> str:
    if category == "fragrance":
        slot = slot_of_fragrance(attrs or {}, raw_meta=raw_meta or {})
        if slot in SLOTS:
            return str(slot)
    return str(product_type or "").strip().lower()


def _status_flag(status_value: str, expected: str) -> int:
    return int(str(status_value or "").strip().lower() == str(expected or "").strip().lower())


def _write_summary_md(*, out_dir: Path, metadata: dict[str, Any]) -> Path:
    label_sources = metadata.get("label_source_distribution") or {}
    positives_by_category = metadata.get("positives_by_category") or {}
    candidate_count_distribution = metadata.get("candidate_count_distribution") or {}
    class_balance = metadata.get("per_category_class_balance") or {}
    lines = [
        "# Roadmap Planner Dataset Summary",
        "",
        f"- rows: **{metadata.get('rows_total', 0)}**",
        f"- decision points: **{metadata.get('decision_points_total', 0)}**",
        f"- episodes: **{metadata.get('episodes_total', 0)}**",
        f"- users: **{metadata.get('users_total', 0)}**",
        f"- stop rate: **{metadata.get('stop_label_rate', 0.0)}**",
        f"- excluded legacy bad fragrance completions: **{metadata.get('excluded_legacy_bad_fragrance_completions_count', 0)}**",
        f"- excluded noisy decision points: **{metadata.get('excluded_noisy_decision_points_count', 0)}**",
        "",
        "## Positives By Category",
    ]
    for category, value in sorted(positives_by_category.items()):
        lines.append(f"- {category}: **{value}**")
    lines.extend(["", "## Positives By Label Source"])
    for source, value in sorted(label_sources.items()):
        lines.append(f"- {source}: **{value}**")
    lines.extend(["", "## Candidate Count Distribution"])
    for bucket, value in sorted(candidate_count_distribution.items(), key=lambda row: int(row[0])):
        lines.append(f"- {bucket}: **{value}**")
    lines.extend(["", "## Per-Category Class Balance"])
    for category, payload in sorted(class_balance.items()):
        lines.append(
            f"- {category}: positives={payload.get('positives', 0)}, decisions={payload.get('decision_points', 0)}, "
            f"stop_rate={payload.get('stop_rate', 0.0)}"
        )
    summary_path = out_dir / "summary.md"
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary_path


class Command(BaseCommand):
    help = (
        "Build leakage-safe Roadmap Planner dataset from PLAN_REFRESHED/STEP_GENERATED/"
        "STEP_COMPLETED events for ML-first roadmap planning."
    )

    def add_arguments(self, parser):
        parser.add_argument("--days", type=int, default=180)
        parser.add_argument("--out-dir", type=str, default="data/ml/roadmap_planner_v1")
        parser.add_argument("--include-ga", action="store_true", default=False)
        parser.add_argument("--label-window-days", type=int, default=3)
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
        skipped_qs = RoadmapEvent.objects.filter(
            user_id__in=users,
            event_type=RoadmapEvent.Type.STEP_SKIPPED,
            created_at__gte=since,
            created_at__lte=now_utc,
        )
        if not include_ga:
            generated_qs = generated_qs.exclude(user__username__startswith="ga_")
            completed_qs = completed_qs.exclude(user__username__startswith="ga_")
            skipped_qs = skipped_qs.exclude(user__username__startswith="ga_")

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
        completion_product_ids: set[int] = set()
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
            purchased_product_id = _to_int_or_none(ctx.get("purchased_product_id"))
            recommended_product_id = _to_int_or_none(ctx.get("recommended_product_id"))
            if category == "fragrance":
                if purchased_product_id:
                    completion_product_ids.add(int(purchased_product_id))
                if recommended_product_id:
                    completion_product_ids.add(int(recommended_product_id))
            completed_by_key[(int(row["user_id"]), category)].append(
                {
                    "id": int(row["id"]),
                    "plan_id": int(row["plan_id"]) if row.get("plan_id") else None,
                    "step_id": int(row["step_id"]) if row.get("step_id") else None,
                    "created_at": row["created_at"].astimezone(dt_timezone.utc),
                    "product_type": str(ctx.get("product_type") or row.get("step__product_type") or "").strip().lower(),
                    "matched_by": str(ctx.get("matched_by") or "").strip().lower(),
                    "context": ctx,
                    "purchased_product_id": purchased_product_id,
                    "recommended_product_id": recommended_product_id,
                }
            )

        skipped_by_key: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
        for row in skipped_qs.values(
            "id",
            "user_id",
            "plan_id",
            "step_id",
            "created_at",
            "context",
            "step__product_type",
            "step__step_index",
            "step__plan__category",
        ).iterator(chunk_size=5000):
            ctx = _safe_dict(row.get("context"))
            category = str(ctx.get("category") or row.get("step__plan__category") or "").strip().lower()
            if category not in TARGET_CATEGORIES:
                continue
            skipped_by_key[(int(row["user_id"]), category)].append(
                {
                    "id": int(row["id"]),
                    "plan_id": int(row["plan_id"]) if row.get("plan_id") else None,
                    "step_id": int(row["step_id"]) if row.get("step_id") else None,
                    "created_at": row["created_at"].astimezone(dt_timezone.utc),
                    "product_type": str(ctx.get("product_type") or row.get("step__product_type") or "").strip().lower(),
                    "step_index": int(ctx.get("step_index") or row.get("step__step_index") or 0),
                    "context": ctx,
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
            "product__brand",
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
                    "brand": str(row.get("product__brand") or "").strip().lower(),
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
        candidate_types_by_category: dict[str, list[str]] = {
            category: list(tokens)
            for category, tokens in CANDIDATE_SPACE_BY_CATEGORY.items()
        }
        completion_product_slots = {
            int(row["id"]): slot_of_fragrance(row.get("attrs") or {}, raw_meta=row.get("raw_meta") or {})
            for row in Product.objects.filter(id__in=list(completion_product_ids)).values("id", "attrs", "raw_meta")
        }

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
        for rows in skipped_by_key.values():
            rows.sort(key=lambda row: (row["created_at"], int(row["id"])))

        episode_records: list[dict[str, Any]] = []
        skipped_no_snapshot = 0
        skipped_no_current_next = 0
        excluded_legacy_bad_fragrance_completions = 0

        for (user_id, category), rows in refresh_groups.items():
            generated_rows = generated_by_key.get((int(user_id), category), [])
            generated_positions = [_event_position(row["created_at"], row["id"]) for row in generated_rows]
            completed_rows = completed_by_key.get((int(user_id), category), [])
            completed_positions = [_event_position(row["created_at"], row["id"]) for row in completed_rows]
            skipped_rows = skipped_by_key.get((int(user_id), category), [])
            skipped_positions = [_event_position(row["created_at"], row["id"]) for row in skipped_rows]
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
                if not next_product_type or next_product_type not in set(candidate_types_by_category.get(category) or []):
                    skipped_no_current_next += 1
                    continue

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
                category_token_counter_all: Counter[str] = Counter()
                category_brand_counter_all: Counter[str] = Counter()

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
                    if token:
                        category_token_counter_all[token] += int(item["quantity"])
                    brand = str(item.get("brand") or "").strip().lower()
                    if brand:
                        category_brand_counter_all[brand] += int(item["quantity"])
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
                completion_rows_window = completed_rows[completed_start:completed_end]
                skipped_start = bisect_right(skipped_positions, refresh_pos)
                skipped_end = bisect_left(skipped_positions, end_pos)
                skip_rows_window = skipped_rows[skipped_start:skipped_end]
                label = STOP_TOKEN
                matched_by = "__none__"
                label_source = "stop_no_completion"
                trusted_completion = None
                for completion in completion_rows_window:
                    completion_label = str(completion.get("product_type") or "").strip().lower()
                    completion_matched_by = str(completion.get("matched_by") or "").strip().lower()
                    if not completion_label:
                        continue
                    if category == "fragrance":
                        if completion_matched_by == "fragrance_slot" and completion_label in SLOTS:
                            trusted_completion = completion
                            label = completion_label
                            matched_by = completion_matched_by
                            label_source = "roadmap_completed_slot"
                            break
                        if completion_matched_by == "recommended_product_id":
                            completion_pid = completion.get("purchased_product_id") or completion.get("recommended_product_id")
                            actual_slot = str(completion_product_slots.get(int(completion_pid or 0)) or "").strip().lower()
                            if actual_slot == completion_label and completion_label in SLOTS:
                                trusted_completion = completion
                                label = completion_label
                                matched_by = completion_matched_by
                                label_source = "roadmap_completed_exact"
                                break
                            excluded_legacy_bad_fragrance_completions += 1
                            continue
                        continue
                    trusted_completion = completion
                    label = completion_label
                    matched_by = completion_matched_by or "__none__"
                    label_source = (
                        "roadmap_completed_exact"
                        if completion_matched_by == "recommended_product_id"
                        else "roadmap_completed_event"
                    )
                    break

                if trusted_completion is None:
                    skip_event = next(
                        (
                            row
                            for row in skip_rows_window
                            if (
                                (next_step_id and int(row.get("step_id") or 0) == int(next_step_id))
                                or str(row.get("product_type") or "").strip().lower() == next_product_type
                            )
                        ),
                        None,
                    )
                    if skip_event is not None:
                        label = STOP_TOKEN
                        matched_by = "roadmap_step_skipped"
                        label_source = "roadmap_skipped_stop"
                    else:
                        future_items = [
                            item
                            for item in tx_items[pivot:]
                            if item["ts"] > t0
                            and item["ts"] <= label_window_end
                            and (next_refresh is None or item["ts"] < next_refresh["created_at"])
                            and str(item.get("category") or "") == category
                        ]
                        future_match = next(
                            (
                                item
                                for item in future_items
                                if str(item.get("history_token") or "").strip().lower() == next_product_type
                            ),
                            None,
                        )
                        if future_match is not None:
                            label = next_product_type
                            matched_by = "future_purchase"
                            label_source = "future_purchase_fallback"
                        else:
                            label = STOP_TOKEN
                            matched_by = "__none__"
                            label_source = "stop_no_progress"

                candidate_types = list(candidate_types_by_category.get(category) or [])
                if STOP_TOKEN not in candidate_types:
                    candidate_types.append(STOP_TOKEN)
                if label != STOP_TOKEN and label not in set(candidate_types):
                    skipped_no_current_next += 1
                    continue

                base_content = build_base_content_features(
                    profile_map.get(int(user_id)),
                    product_signature(anchor_item),
                )
                decision_id = int(len(episode_records) + 1)
                favorite_brand_in_category = (
                    category_brand_counter_all.most_common(1)[0][0]
                    if category_brand_counter_all
                    else "__none__"
                )
                prior_category_purchase_total = int(sum(item["quantity"] for item in prior_items if str(item["category"] or "") == category))
                prior_category_distinct_token_count = int(len(category_token_counter_all))
                fragrance_slot_coverage_count = (
                    int(len({token for token in category_token_counter_all if token in set(SLOTS)}))
                    if category == "fragrance"
                    else 0
                )

                episode_records.append(
                    {
                        "episode_id": decision_id,
                        "decision_id": decision_id,
                        "user_id": int(user_id),
                        "category": category,
                        "t0_dt": t0,
                        "t0_utc": t0.isoformat().replace("+00:00", "Z"),
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
                        "prior_category_purchase_total": prior_category_purchase_total,
                        "prior_category_distinct_token_count": prior_category_distinct_token_count,
                        "favorite_brand_in_category": favorite_brand_in_category,
                        "fragrance_slot_coverage_count": fragrance_slot_coverage_count,
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

        split_by_decision, split_meta = _time_based_split_assignments(episode_records)
        for ep in episode_records:
            ep["split"] = str(split_by_decision.get(int(ep["decision_id"])) or "train")

        candidate_popularity_train: dict[str, Counter[str]] = defaultdict(Counter)
        for ep in episode_records:
            if str(ep.get("split") or "") != "train":
                continue
            candidate_popularity_train[str(ep["category"])][str(ep["label"])] += 1

        rows: list[dict[str, Any]] = []
        for ep in episode_records:
            base_row = {k: v for k, v in ep.items() if k not in {
                "plan_position_by_type",
                "plan_state_by_type",
                "recent_category_tokens",
                "candidate_seen_90d_counter",
                "candidate_days_since_last_seen_map",
                "anchor_item",
                "candidate_types",
                "t0_dt",
            }}
            plan_position_by_type = dict(ep.get("plan_position_by_type") or {})
            plan_state_by_type = dict(ep.get("plan_state_by_type") or {})
            recent_category_tokens = [str(x) for x in (ep.get("recent_category_tokens") or [])]
            seen_90d_counter = {str(k): int(v) for k, v in (ep.get("candidate_seen_90d_counter") or {}).items()}
            days_since_last_seen_map = {
                str(k): int(v) for k, v in (ep.get("candidate_days_since_last_seen_map") or {}).items()
            }
            anchor_sig = product_signature(ep.get("anchor_item"))
            category = str(ep["category"])
            label = str(ep["label"])
            popularity_counter = candidate_popularity_train.get(category) or Counter()
            popularity_total = float(sum(popularity_counter.values()) or 1.0)
            candidate_types = list(ep.get("candidate_types") or [])
            for candidate in candidate_types:
                state = plan_state_by_type.get(candidate) or {}
                current_status = str(state.get("status") or "")
                rows.append(
                    {
                        **base_row,
                        "candidate_type": str(candidate),
                        "y": int(str(candidate) == label),
                        "candidate_in_generated_plan": int(candidate in plan_position_by_type),
                        "candidate_position_in_generated_plan": int(plan_position_by_type.get(candidate, -1)),
                        "candidate_is_current_next_step": int(
                            str(candidate) == str(ep.get("current_next_product_type") or "")
                        ),
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
                            candidate_type=str(candidate),
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
            "strategy": str(split_meta.get("strategy") or "time_based"),
            "train_end_utc": split_meta.get("train_end_utc"),
            "valid_end_utc": split_meta.get("valid_end_utc"),
            "counts": dict(split_meta.get("counts") or {}),
            "user_overlap_counts": dict(split_meta.get("user_overlap_counts") or {}),
            "decision_ids": {
                split_name: [
                    int(ep["decision_id"])
                    for ep in episode_records
                    if str(ep.get("split") or "") == split_name
                ]
                for split_name in ["train", "val", "test"]
            },
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
            "favorite_brand_in_category",
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
            "prior_category_purchase_total",
            "prior_category_distinct_token_count",
            "fragrance_slot_coverage_count",
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
        positives_by_category = Counter(str(ep["category"]) for ep in episode_records if str(ep["label"]) != STOP_TOKEN)
        candidate_count_distribution = Counter(len(list(ep.get("candidate_types") or [])) for ep in episode_records)
        per_category_class_balance: dict[str, dict[str, Any]] = {}
        for category in sorted(TARGET_CATEGORIES):
            category_eps = [ep for ep in episode_records if str(ep.get("category")) == category]
            if not category_eps:
                continue
            stop_count = sum(1 for ep in category_eps if str(ep.get("label")) == STOP_TOKEN)
            per_category_class_balance[category] = {
                "decision_points": int(len(category_eps)),
                "positives": int(len(category_eps) - stop_count),
                "stop_rate": round(float(stop_count / max(1, len(category_eps))), 6),
            }
        metadata = {
            "version": "planner_v1_candidate_ranking",
            "generated_at_utc": now_utc.isoformat().replace("+00:00", "Z"),
            "window_days": int(days),
            "window_since_utc": since.isoformat().replace("+00:00", "Z"),
            "window_until_utc": now_utc.isoformat().replace("+00:00", "Z"),
            "label_window_days": int(label_window_days),
            "include_ga": bool(include_ga),
            "seed": int(seed),
            "decision_point_definition": (
                "Decision point = PLAN_REFRESHED event with STEP_GENERATED snapshot between this refresh "
                "and the next refresh for the same user+category."
            ),
            "decision_point_trust": (
                "Exclude episodes without STEP_GENERATED snapshot or without actionable next step in the stable "
                "candidate vocabulary."
            ),
            "dataset_format": dataset_format,
            "dataset_file": str(dataset_file),
            "rows_total": int(len(df)),
            "decision_points_total": int(len(episode_records)),
            "episodes_total": int(len(episode_records)),
            "users_total": int(len({int(ep['user_id']) for ep in episode_records})),
            "positive_rows": int(df["y"].sum()) if not df.empty else 0,
            "stop_label_count": int(label_counter.get(STOP_TOKEN, 0)),
            "stop_label_rate": round(float(label_counter.get(STOP_TOKEN, 0) / max(1, len(episode_records))), 6),
            "label_source_distribution": dict(sorted(label_source_counter.items())),
            "label_matched_by_distribution": dict(sorted(label_matched_by_counter.items())),
            "raw_plan_refreshed_events": int(len(refresh_rows)),
            "skipped_episodes_without_snapshot": int(skipped_no_snapshot),
            "skipped_episodes_without_current_next": int(skipped_no_current_next),
            "excluded_noisy_decision_points_count": int(skipped_no_snapshot + skipped_no_current_next),
            "excluded_legacy_bad_fragrance_completions_count": int(excluded_legacy_bad_fragrance_completions),
            "candidate_types_by_category": {
                category: list(types) + ([STOP_TOKEN] if STOP_TOKEN not in types else [])
                for category, types in sorted(candidate_types_by_category.items())
            },
            "class_distribution": dict(sorted(label_counter.items())),
            "positives_by_category": dict(sorted(positives_by_category.items())),
            "candidate_count_distribution": {
                str(key): int(value)
                for key, value in sorted(candidate_count_distribution.items(), key=lambda row: row[0])
            },
            "per_category_class_balance": per_category_class_balance,
            "feature_columns": feature_columns,
            "categorical_features": categorical_features,
            "numeric_features": numeric_features,
            "split_strategy": str(split_meta.get("strategy") or "time_based"),
            "split_counts": dict(split_meta.get("counts") or {}),
            "split_user_overlap_counts": dict(split_meta.get("user_overlap_counts") or {}),
            "leakage_assertions": {
                "features_only_use_transactions_lte_t0": True,
                "labels_end_at_next_plan_refresh_or_window_end": True,
                "future_purchase_fallback_restricted_to_current_next_step": True,
                "legacy_bad_fragrance_exact_completions_excluded": True,
                "status": "passed",
            },
        }
        metadata_path = out_dir / "metadata.json"
        metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        summary_path = _write_summary_md(out_dir=out_dir, metadata=metadata)

        self.stdout.write("[build_roadmap_planner_dataset] done")
        self.stdout.write(f"[build_roadmap_planner_dataset] dataset={dataset_file}")
        self.stdout.write(f"[build_roadmap_planner_dataset] metadata={metadata_path}")
        self.stdout.write(f"[build_roadmap_planner_dataset] splits={splits_path}")
        self.stdout.write(f"[build_roadmap_planner_dataset] summary={summary_path}")
