from __future__ import annotations

from bisect import bisect_left, bisect_right
from collections import Counter, defaultdict
from copy import deepcopy
from datetime import timedelta, timezone as dt_timezone
from typing import Any

from django.utils import timezone

from catalog.models import Product
from roadmap_app.content_features import build_candidate_catalog_summaries, profile_signature
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
INITIAL_DECISION_TYPES = {"initial_refresh", "post_refresh_rebuild"}
CONTINUATION_DECISION_TYPES = {"post_completed", "post_skipped", "other_trusted_transition"}


def _safe_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _to_int_or_none(value: Any) -> int | None:
    try:
        if value is None:
            return None
        return int(value)
    except Exception:
        return None


def _event_position(created_at, event_id: int) -> tuple[Any, int]:
    return (created_at.astimezone(dt_timezone.utc), int(event_id))


def _history_token_for_tx(*, category: str, product_type: str, attrs: dict[str, Any], raw_meta: dict[str, Any]) -> str:
    if category == "fragrance":
        slot = slot_of_fragrance(attrs or {}, raw_meta=raw_meta or {})
        if slot in SLOTS:
            return str(slot)
    return str(product_type or "").strip().lower()


def _is_actionable_status(status: str) -> bool:
    normalized = str(status or "").strip().lower()
    return normalized in {RoadmapStep.Status.MISSING, RoadmapStep.Status.RECOMMENDED}


def _candidate_types_for_category(category: str) -> list[str]:
    candidates = list(CANDIDATE_SPACE_BY_CATEGORY.get(str(category or "").strip().lower()) or [])
    if STOP_TOKEN not in candidates:
        candidates.append(STOP_TOKEN)
    return candidates


def _clone_state(snapshot: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [dict(row) for row in deepcopy(snapshot)]


def _next_actionable_step(snapshot: list[dict[str, Any]]) -> dict[str, Any] | None:
    ordered = sorted(snapshot, key=lambda row: (int(row.get("step_index") or 0), int(row.get("step_id") or 0)))
    return next((row for row in ordered if _is_actionable_status(str(row.get("status") or ""))), None)


def _status_counter(snapshot: list[dict[str, Any]]) -> Counter[str]:
    return Counter(str(row.get("status") or "").strip().lower() for row in snapshot)


def _match_step_identity(event_row: dict[str, Any], step_row: dict[str, Any] | None) -> bool:
    if not step_row:
        return False
    step_id = _to_int_or_none(step_row.get("step_id"))
    if step_id and _to_int_or_none(event_row.get("step_id")) == step_id:
        return True
    return str(event_row.get("product_type") or "").strip().lower() == str(step_row.get("product_type") or "").strip().lower()


def _apply_outcome(snapshot: list[dict[str, Any]], *, step_row: dict[str, Any] | None, new_status: str) -> list[dict[str, Any]]:
    updated = _clone_state(snapshot)
    if not step_row:
        return updated
    step_id = _to_int_or_none(step_row.get("step_id"))
    expected_type = str(step_row.get("product_type") or "").strip().lower()
    for row in updated:
        row_step_id = _to_int_or_none(row.get("step_id"))
        row_type = str(row.get("product_type") or "").strip().lower()
        if (step_id and row_step_id == step_id) or (expected_type and row_type == expected_type):
            row["status"] = str(new_status)
            break
    return updated


def _decision_type_allowed(mode: str, decision_type: str) -> bool:
    if mode == "combined":
        return True
    if mode == "initial":
        return decision_type in INITIAL_DECISION_TYPES
    if mode == "continuation":
        return decision_type in CONTINUATION_DECISION_TYPES
    raise ValueError(f"Unsupported mode: {mode}")


def _trusted_completion_info(
    completion: dict[str, Any],
    *,
    category: str,
    completion_product_slots: dict[int, str],
) -> dict[str, Any]:
    completion_label = str(completion.get("product_type") or "").strip().lower()
    completion_matched_by = str(completion.get("matched_by") or "").strip().lower()
    if not completion_label:
        return {"trusted": False, "excluded_reason": "empty_completion_label"}
    if category == "fragrance":
        if completion_matched_by == "fragrance_slot" and completion_label in SLOTS:
            return {
                "trusted": True,
                "label": completion_label,
                "matched_by": completion_matched_by,
                "label_source": "roadmap_completed_slot",
                "trust_level": "high",
            }
        if completion_matched_by == "recommended_product_id":
            completion_pid = completion.get("purchased_product_id") or completion.get("recommended_product_id")
            actual_slot = str(completion_product_slots.get(int(completion_pid or 0)) or "").strip().lower()
            if actual_slot == completion_label and completion_label in SLOTS:
                return {
                    "trusted": True,
                    "label": completion_label,
                    "matched_by": completion_matched_by,
                    "label_source": "roadmap_completed_exact",
                    "trust_level": "high",
                }
            return {
                "trusted": False,
                "excluded_reason": "legacy_bad_fragrance_exact_completion",
            }
        return {"trusted": False, "excluded_reason": "unsupported_fragrance_completion_match"}
    return {
        "trusted": True,
        "label": completion_label,
        "matched_by": completion_matched_by or "__none__",
        "label_source": (
            "roadmap_completed_exact"
            if completion_matched_by == "recommended_product_id"
            else "roadmap_completed_event"
        ),
        "trust_level": "high",
    }


def _future_purchase_label(
    *,
    tx_items: list[dict[str, Any]],
    timeline: list[Any],
    category: str,
    t0,
    label_window_end,
    next_refresh_at,
    current_next_token: str,
) -> dict[str, Any] | None:
    if not current_next_token or current_next_token == STOP_TOKEN:
        return None
    pivot = bisect_right(timeline, t0) if timeline else 0
    future_match = next(
        (
            item
            for item in tx_items[pivot:]
            if item["ts"] > t0
            and item["ts"] <= label_window_end
            and (next_refresh_at is None or item["ts"] < next_refresh_at)
            and str(item.get("category") or "") == category
            and str(item.get("history_token") or "").strip().lower() == current_next_token
        ),
        None,
    )
    if future_match is None:
        return None
    return {
        "label": current_next_token,
        "matched_by": "future_purchase",
        "label_source": "future_purchase_fallback",
        "trust_level": "medium",
    }


def _resolve_decision_label(
    *,
    category: str,
    snapshot: list[dict[str, Any]],
    t0,
    current_pos: tuple[Any, int],
    next_refresh_at,
    merged_outcomes: list[dict[str, Any]],
    tx_items: list[dict[str, Any]],
    timeline: list[Any],
    label_window_days: int,
    completion_product_slots: dict[int, str],
    bad_completion_event_ids: set[int],
) -> dict[str, Any]:
    current_state = _clone_state(snapshot)
    current_step = _next_actionable_step(current_state)
    if current_step is None:
        return {
            "label": STOP_TOKEN,
            "matched_by": "__none__",
            "label_source": "terminal_after_outcome_stop",
            "trust_level": "high",
        }

    label_window_end = t0 + timedelta(days=label_window_days)
    if next_refresh_at is not None and next_refresh_at < label_window_end:
        label_window_end = next_refresh_at
    end_pos = (label_window_end, MAX_TS_ID)

    for outcome in merged_outcomes:
        if not (current_pos < outcome["position"] < end_pos):
            continue
        current_step = _next_actionable_step(current_state)
        if current_step is None:
            break
        if outcome["kind"] == "completed":
            trust = _trusted_completion_info(
                outcome["row"],
                category=category,
                completion_product_slots=completion_product_slots,
            )
            if not trust.get("trusted"):
                if trust.get("excluded_reason") == "legacy_bad_fragrance_exact_completion":
                    bad_completion_event_ids.add(int(outcome["row"]["id"]))
                continue
            if _match_step_identity(outcome["row"], current_step):
                return {
                    "label": str(current_step.get("product_type") or ""),
                    "matched_by": str(trust.get("matched_by") or "__none__"),
                    "label_source": str(trust.get("label_source") or "roadmap_completed_event"),
                    "trust_level": str(trust.get("trust_level") or "high"),
                }
            continue

        if outcome["kind"] == "skipped" and _match_step_identity(outcome["row"], current_step):
            current_state = _apply_outcome(
                current_state,
                step_row=current_step,
                new_status=RoadmapStep.Status.SKIPPED,
            )

    current_step = _next_actionable_step(current_state)
    if current_step is None:
        return {
            "label": STOP_TOKEN,
            "matched_by": "roadmap_step_skipped",
            "label_source": "roadmap_skipped_stop",
            "trust_level": "high",
        }

    fallback = _future_purchase_label(
        tx_items=tx_items,
        timeline=timeline,
        category=category,
        t0=t0,
        label_window_end=label_window_end,
        next_refresh_at=next_refresh_at,
        current_next_token=str(current_step.get("product_type") or ""),
    )
    if fallback:
        return fallback

    return {
        "label": STOP_TOKEN,
        "matched_by": "__none__",
        "label_source": "stop_no_progress",
        "trust_level": "medium",
    }


def _find_next_applied_outcome(
    *,
    category: str,
    snapshot: list[dict[str, Any]],
    current_pos: tuple[Any, int],
    merged_outcomes: list[dict[str, Any]],
    completion_product_slots: dict[int, str],
    bad_completion_event_ids: set[int],
) -> dict[str, Any] | None:
    current_step = _next_actionable_step(snapshot)
    if current_step is None:
        return None
    for outcome in merged_outcomes:
        if outcome["position"] <= current_pos:
            continue
        if outcome["kind"] == "completed":
            trust = _trusted_completion_info(
                outcome["row"],
                category=category,
                completion_product_slots=completion_product_slots,
            )
            if not trust.get("trusted"):
                if trust.get("excluded_reason") == "legacy_bad_fragrance_exact_completion":
                    bad_completion_event_ids.add(int(outcome["row"]["id"]))
                continue
            if _match_step_identity(outcome["row"], current_step):
                return {
                    "kind": "completed",
                    "row": outcome["row"],
                    "position": outcome["position"],
                    "matched_by": str(trust.get("matched_by") or "__none__"),
                }
            continue
        if outcome["kind"] == "skipped" and _match_step_identity(outcome["row"], current_step):
            return {
                "kind": "skipped",
                "row": outcome["row"],
                "position": outcome["position"],
                "matched_by": "roadmap_step_skipped",
            }
    return None


def _build_decision_features(
    *,
    user_id: int,
    category: str,
    t0,
    snapshot: list[dict[str, Any]],
    refresh_ctx: dict[str, Any],
    profile_map: dict[int, dict[str, Any]],
    tx_items: list[dict[str, Any]],
    timeline: list[Any],
    steps_completed_in_episode: int,
    steps_skipped_in_episode: int,
) -> dict[str, Any]:
    status_counter = _status_counter(snapshot)
    plan_types = [str(row.get("product_type") or "") for row in snapshot if str(row.get("product_type") or "").strip()]
    plan_position_by_type: dict[str, int] = {}
    plan_state_by_type: dict[str, dict[str, Any]] = {}
    ordered_snapshot = sorted(snapshot, key=lambda item: (int(item.get("step_index") or 0), int(item.get("step_id") or 0)))
    for position, row in enumerate(ordered_snapshot, start=1):
        candidate = str(row.get("product_type") or "").strip().lower()
        if candidate and candidate not in plan_position_by_type:
            plan_position_by_type[candidate] = int(position)
            plan_state_by_type[candidate] = {
                "status": str(row.get("status") or ""),
                "has_recommendation": bool(row.get("recommended_product_id")),
                "step_index": int(row.get("step_index") or 0),
                "step_id": int(row.get("step_id") or 0),
            }

    current_step = _next_actionable_step(snapshot)
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

    favorite_brand_in_category = (
        category_brand_counter_all.most_common(1)[0][0]
        if category_brand_counter_all
        else "__none__"
    )
    prior_category_purchase_total = int(
        sum(item["quantity"] for item in prior_items if str(item["category"] or "") == category)
    )
    prior_category_distinct_token_count = int(len(category_token_counter_all))
    fragrance_slot_coverage_count = (
        int(len({token for token in category_token_counter_all if token in set(SLOTS)}))
        if category == "fragrance"
        else 0
    )
    remaining_actionable_steps_count = int(
        status_counter.get(RoadmapStep.Status.MISSING, 0) + status_counter.get(RoadmapStep.Status.RECOMMENDED, 0)
    )
    remaining_depth_in_plan = int(
        len([row for row in snapshot if _is_actionable_status(str(row.get("status") or ""))])
    )

    return {
        "user_id": int(user_id),
        "category": category,
        "t0_dt": t0,
        "t0_utc": t0.isoformat().replace("+00:00", "Z"),
        "steps_total": int(len(snapshot)),
        "missing_steps_count": int(
            status_counter.get(RoadmapStep.Status.MISSING, 0)
            + status_counter.get(RoadmapStep.Status.RECOMMENDED, 0)
        ),
        "recommended_steps_count": int(status_counter.get(RoadmapStep.Status.RECOMMENDED, 0)),
        "owned_steps_count": int(status_counter.get(RoadmapStep.Status.OWNED, 0)),
        "completed_steps_count": int(status_counter.get(RoadmapStep.Status.COMPLETED, 0)),
        "skipped_steps_count": int(status_counter.get(RoadmapStep.Status.SKIPPED, 0)),
        "remaining_actionable_steps_count": remaining_actionable_steps_count,
        "remaining_depth_in_plan": remaining_depth_in_plan,
        "steps_completed_in_episode_count": int(steps_completed_in_episode),
        "steps_skipped_in_episode_count": int(steps_skipped_in_episode),
        "current_next_product_type": str(current_step.get("product_type") or "__none__") if current_step else "__none__",
        "current_next_step_id": int(current_step.get("step_id") or 0) if current_step else 0,
        "next_step_index_current": int(current_step.get("step_index") or 0) if current_step else 0,
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
        "candidate_types": _candidate_types_for_category(category),
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
        "profile_signature": dict(profile_map.get(int(user_id)) or {}),
        "refresh_caller": str(refresh_ctx.get("refresh_caller") or "__none__"),
        "refresh_source": str(refresh_ctx.get("source") or "__none__"),
        "current_ml_decision": str(_safe_dict(refresh_ctx.get("ml")).get("decision") or "__none__"),
        "current_rollout_mode": str(_safe_dict(refresh_ctx.get("ml")).get("rollout_mode") or "__none__"),
    }


def load_transition_source_data(*, days: int, include_ga: bool, label_window_days: int) -> dict[str, Any]:
    now_utc = timezone.now().astimezone(dt_timezone.utc)
    since = now_utc - timedelta(days=days)
    max_t0 = now_utc - timedelta(days=label_window_days)

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

    refresh_rows.sort(key=lambda row: (int(row["user_id"]), str(row["category"]), row["created_at"], int(row["id"])))
    users = sorted({int(row["user_id"]) for row in refresh_rows})
    if not users:
        return {
            "now_utc": now_utc,
            "since": since,
            "max_t0": max_t0,
            "refresh_rows": [],
            "refresh_groups": {},
            "generated_by_key": {},
            "completed_by_key": {},
            "skipped_by_key": {},
            "user_items": {},
            "timeline_by_user": {},
            "profile_map": {},
            "candidate_catalog_summaries": {},
            "completion_product_slots": {},
            "candidate_types_by_category": {
                category: list(tokens)
                for category, tokens in CANDIDATE_SPACE_BY_CATEGORY.items()
            },
        }

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
                "supported_skin_types": row.get("product__supported_skin_types")
                if isinstance(row.get("product__supported_skin_types"), list)
                else [],
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
        items.sort(key=lambda item: (item["ts"], int(item["tx_id"])))
        timeline_by_user[user_id] = [item["ts"] for item in items]

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

    return {
        "now_utc": now_utc,
        "since": since,
        "max_t0": max_t0,
        "refresh_rows": refresh_rows,
        "refresh_groups": refresh_groups,
        "generated_by_key": generated_by_key,
        "completed_by_key": completed_by_key,
        "skipped_by_key": skipped_by_key,
        "user_items": user_items,
        "timeline_by_user": timeline_by_user,
        "profile_map": profile_map,
        "candidate_catalog_summaries": candidate_catalog_summaries,
        "completion_product_slots": completion_product_slots,
        "candidate_types_by_category": {
            category: list(tokens)
            for category, tokens in CANDIDATE_SPACE_BY_CATEGORY.items()
        },
    }


def build_transition_decision_records(
    source_data: dict[str, Any],
    *,
    label_window_days: int,
    mode: str = "combined",
) -> dict[str, Any]:
    refresh_groups = source_data.get("refresh_groups") or {}
    generated_by_key = source_data.get("generated_by_key") or {}
    completed_by_key = source_data.get("completed_by_key") or {}
    skipped_by_key = source_data.get("skipped_by_key") or {}
    timeline_by_user = source_data.get("timeline_by_user") or {}
    user_items = source_data.get("user_items") or {}
    profile_map = source_data.get("profile_map") or {}
    completion_product_slots = source_data.get("completion_product_slots") or {}

    decision_records: list[dict[str, Any]] = []
    surface_records: list[dict[str, Any]] = []
    excluded_counts: Counter[str] = Counter()
    bad_completion_event_ids: set[int] = set()
    episode_counter = 0

    for (user_id, category), refresh_rows in refresh_groups.items():
        generated_rows = generated_by_key.get((int(user_id), category), [])
        generated_positions = [_event_position(row["created_at"], row["id"]) for row in generated_rows]
        completed_rows = completed_by_key.get((int(user_id), category), [])
        skipped_rows = skipped_by_key.get((int(user_id), category), [])
        timeline = timeline_by_user.get(int(user_id)) or []
        tx_items = user_items.get(int(user_id)) or []

        for refresh_index, refresh_row in enumerate(refresh_rows):
            refresh_pos = _event_position(refresh_row["created_at"], int(refresh_row["id"]))
            next_refresh = refresh_rows[refresh_index + 1] if refresh_index + 1 < len(refresh_rows) else None
            next_refresh_pos = (
                _event_position(next_refresh["created_at"], int(next_refresh["id"]))
                if next_refresh
                else None
            )
            generated_start = bisect_right(generated_positions, refresh_pos)
            generated_end = bisect_left(generated_positions, next_refresh_pos) if next_refresh_pos else len(generated_rows)
            snapshot = _clone_state(generated_rows[generated_start:generated_end])
            base_decision_type = (
                "post_refresh_rebuild"
                if str(_safe_dict(refresh_row.get("context")).get("refresh_caller") or "").strip()
                == "update_roadmap_from_purchase"
                else "initial_refresh"
            )
            if not snapshot:
                excluded_counts["no_snapshot_after_refresh"] += 1
                surface_records.append(
                    {
                        "decision_type": base_decision_type,
                        "user_id": int(user_id),
                        "category": category,
                        "plan_id": refresh_row.get("plan_id"),
                        "t0_dt": refresh_row["created_at"],
                        "trusted": False,
                        "excluded_reason": "no_snapshot_after_refresh",
                        "label": None,
                    }
                )
                continue

            episode_counter += 1
            episode_id = int(episode_counter)
            snapshot.sort(key=lambda row: (int(row.get("step_index") or 0), int(row.get("step_id") or 0)))
            episode_completed_rows = [
                row
                for row in completed_rows
                if refresh_pos < _event_position(row["created_at"], row["id"])
                and (next_refresh_pos is None or _event_position(row["created_at"], row["id"]) < next_refresh_pos)
            ]
            episode_skipped_rows = [
                row
                for row in skipped_rows
                if refresh_pos < _event_position(row["created_at"], row["id"])
                and (next_refresh_pos is None or _event_position(row["created_at"], row["id"]) < next_refresh_pos)
            ]
            merged_outcomes = sorted(
                [
                    {"kind": "completed", "row": row, "position": _event_position(row["created_at"], row["id"])}
                    for row in episode_completed_rows
                ]
                + [
                    {"kind": "skipped", "row": row, "position": _event_position(row["created_at"], row["id"])}
                    for row in episode_skipped_rows
                ],
                key=lambda row: row["position"],
            )

            current_state = _clone_state(snapshot)
            current_pos = refresh_pos
            decision_type = base_decision_type
            steps_completed_in_episode = 0
            steps_skipped_in_episode = 0

            while True:
                next_step = _next_actionable_step(current_state)
                if next_step is None and decision_type in INITIAL_DECISION_TYPES:
                    excluded_counts["no_current_next_initial"] += 1
                    surface_records.append(
                        {
                            "decision_type": decision_type,
                            "user_id": int(user_id),
                            "category": category,
                            "plan_id": refresh_row.get("plan_id"),
                            "t0_dt": current_pos[0],
                            "trusted": False,
                            "excluded_reason": "no_current_next_initial",
                            "label": None,
                        }
                    )
                    break

                features = _build_decision_features(
                    user_id=int(user_id),
                    category=category,
                    t0=current_pos[0],
                    snapshot=current_state,
                    refresh_ctx=_safe_dict(refresh_row.get("context")),
                    profile_map=profile_map,
                    tx_items=tx_items,
                    timeline=timeline,
                    steps_completed_in_episode=steps_completed_in_episode,
                    steps_skipped_in_episode=steps_skipped_in_episode,
                )
                label_info = _resolve_decision_label(
                    category=category,
                    snapshot=current_state,
                    t0=current_pos[0],
                    current_pos=current_pos,
                    next_refresh_at=next_refresh["created_at"] if next_refresh else None,
                    merged_outcomes=merged_outcomes,
                    tx_items=tx_items,
                    timeline=timeline,
                    label_window_days=label_window_days,
                    completion_product_slots=completion_product_slots,
                    bad_completion_event_ids=bad_completion_event_ids,
                )
                label = str(label_info.get("label") or STOP_TOKEN)
                if label != STOP_TOKEN and label not in set(_candidate_types_for_category(category)):
                    excluded_counts["label_out_of_vocab"] += 1
                    surface_records.append(
                        {
                            "decision_type": decision_type,
                            "user_id": int(user_id),
                            "category": category,
                            "plan_id": refresh_row.get("plan_id"),
                            "t0_dt": current_pos[0],
                            "trusted": False,
                            "excluded_reason": "label_out_of_vocab",
                            "label": None,
                        }
                    )
                else:
                    record = {
                        "episode_id": episode_id,
                        "decision_id": int(len(decision_records) + 1),
                        "plan_id": int(refresh_row.get("plan_id") or 0),
                        "decision_type": decision_type,
                        "label": label,
                        "matched_by": str(label_info.get("matched_by") or "__none__"),
                        "label_source": str(label_info.get("label_source") or "unknown"),
                        "trust_level": str(label_info.get("trust_level") or "medium"),
                        "excluded_reason": "__none__",
                        **features,
                    }
                    surface_records.append(
                        {
                            "decision_type": decision_type,
                            "user_id": int(user_id),
                            "category": category,
                            "plan_id": refresh_row.get("plan_id"),
                            "t0_dt": current_pos[0],
                            "trusted": True,
                            "excluded_reason": "__none__",
                            "label": label,
                            "current_next_product_type": str(record.get("current_next_product_type") or "__none__"),
                            "trust_level": str(record.get("trust_level") or "medium"),
                            "label_source": str(record.get("label_source") or "unknown"),
                            "advanced_to_next_step": int(
                                decision_type in CONTINUATION_DECISION_TYPES
                                and str(record.get("current_next_product_type") or "__none__") not in {"__none__", STOP_TOKEN}
                            ),
                        }
                    )
                    if _decision_type_allowed(mode, decision_type):
                        decision_records.append(record)

                next_outcome = _find_next_applied_outcome(
                    category=category,
                    snapshot=current_state,
                    current_pos=current_pos,
                    merged_outcomes=merged_outcomes,
                    completion_product_slots=completion_product_slots,
                    bad_completion_event_ids=bad_completion_event_ids,
                )
                if next_outcome is None:
                    break
                current_state = _apply_outcome(
                    current_state,
                    step_row=_next_actionable_step(current_state),
                    new_status=(
                        RoadmapStep.Status.COMPLETED
                        if next_outcome["kind"] == "completed"
                        else RoadmapStep.Status.SKIPPED
                    ),
                )
                current_pos = next_outcome["position"]
                if next_outcome["kind"] == "completed":
                    steps_completed_in_episode += 1
                    decision_type = "post_completed"
                else:
                    steps_skipped_in_episode += 1
                    decision_type = "post_skipped"

    return {
        "decision_records": decision_records,
        "surface_records": surface_records,
        "excluded_counts": dict(sorted(excluded_counts.items())),
        "excluded_legacy_bad_fragrance_completions_count": int(len(bad_completion_event_ids)),
    }


def summarize_decision_surfaces(
    source_data: dict[str, Any],
    decision_bundle: dict[str, Any],
) -> dict[str, Any]:
    surface_records = list(decision_bundle.get("surface_records") or [])
    decision_records = list(decision_bundle.get("decision_records") or [])
    refresh_rows = list(source_data.get("refresh_rows") or [])

    by_type: dict[str, dict[str, Any]] = {}
    for decision_type in [
        "initial_refresh",
        "post_refresh_rebuild",
        "post_completed",
        "post_skipped",
        "other_trusted_transition",
    ]:
        raw_rows = [row for row in surface_records if str(row.get("decision_type") or "") == decision_type]
        trusted_rows = [row for row in raw_rows if bool(row.get("trusted"))]
        positive_rows = [row for row in trusted_rows if str(row.get("label") or STOP_TOKEN) != STOP_TOKEN]
        by_type[decision_type] = {
            "raw_count": int(len(raw_rows)),
            "trusted_count": int(len(trusted_rows)),
            "raw_users": int(len({int(row["user_id"]) for row in raw_rows})),
            "trusted_users": int(len({int(row["user_id"]) for row in trusted_rows})),
            "categories": sorted({str(row.get("category") or "") for row in trusted_rows if str(row.get("category") or "")}),
            "non_stop_positive_share": round(float(len(positive_rows) / max(1, len(trusted_rows))), 6),
            "fragrance_share": round(
                float(len([row for row in trusted_rows if str(row.get("category") or "") == "fragrance"]) / max(1, len(trusted_rows))),
                6,
            ),
            "positives_by_category": dict(
                sorted(Counter(str(row.get("category") or "") for row in positive_rows).items(), key=lambda item: item[0])
            ),
            "excluded_reasons": dict(
                sorted(
                    Counter(str(row.get("excluded_reason") or "unknown") for row in raw_rows if not bool(row.get("trusted"))).items(),
                    key=lambda item: item[0],
                )
            ),
            "step_advance_count": int(
                len([row for row in trusted_rows if int(row.get("advanced_to_next_step") or 0) == 1])
            ),
        }

    def _slice_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
        positives = [row for row in rows if str(row.get("label") or STOP_TOKEN) != STOP_TOKEN]
        fragrance_positives = [row for row in positives if str(row.get("category") or "") == "fragrance"]
        return {
            "trusted_decisions_total": int(len(rows)),
            "positives_excluding_stop": int(len(positives)),
            "stop_rate": round(float(len([row for row in rows if str(row.get("label") or STOP_TOKEN) == STOP_TOKEN]) / max(1, len(rows))), 6),
            "users": int(len({int(row["user_id"]) for row in rows})),
            "plans": int(len({int(row.get("plan_id") or 0) for row in rows if int(row.get("plan_id") or 0) > 0})),
            "positives_by_category": dict(
                sorted(Counter(str(row.get("category") or "") for row in positives).items(), key=lambda item: item[0])
            ),
            "positives_by_decision_type": dict(
                sorted(Counter(str(row.get("decision_type") or "") for row in positives).items(), key=lambda item: item[0])
            ),
            "positives_by_label_source": dict(
                sorted(Counter(str(row.get("label_source") or "") for row in positives).items(), key=lambda item: item[0])
            ),
            "fragrance_trusted_positives_count": int(len(fragrance_positives)),
        }

    initial_rows = [row for row in decision_records if str(row.get("decision_type") or "") in INITIAL_DECISION_TYPES]
    continuation_rows = [row for row in decision_records if str(row.get("decision_type") or "") in CONTINUATION_DECISION_TYPES]
    combined_rows = list(decision_records)

    return {
        "surface_types": by_type,
        "initial_only": _slice_summary(initial_rows),
        "continuation_only": _slice_summary(continuation_rows),
        "combined": _slice_summary(combined_rows),
        "raw_plan_refreshed_events": int(len(refresh_rows)),
        "excluded_noisy_decision_points_count": int(sum(int(v) for v in (decision_bundle.get("excluded_counts") or {}).values())),
        "excluded_counts": dict(sorted((decision_bundle.get("excluded_counts") or {}).items())),
        "excluded_legacy_bad_fragrance_completions_count": int(
            decision_bundle.get("excluded_legacy_bad_fragrance_completions_count") or 0
        ),
    }


def readiness_assessment(decision_records: list[dict[str, Any]]) -> dict[str, dict[str, str]]:
    initial_rows = [row for row in decision_records if str(row.get("decision_type") or "") in INITIAL_DECISION_TYPES]
    continuation_rows = [row for row in decision_records if str(row.get("decision_type") or "") in CONTINUATION_DECISION_TYPES]
    combined_rows = list(decision_records)

    def _positives(rows: list[dict[str, Any]], *, category: str | None = None) -> list[dict[str, Any]]:
        filtered = [row for row in rows if str(row.get("label") or STOP_TOKEN) != STOP_TOKEN]
        if category:
            filtered = [row for row in filtered if str(row.get("category") or "") == category]
        return filtered

    def _decision(status: str, why: str) -> dict[str, str]:
        return {"status": status, "why": why}

    initial_pos = _positives(initial_rows)
    initial_haircare_pos = _positives(initial_rows, category="haircare")
    continuation_pos = _positives(continuation_rows)
    combined_pos = _positives(combined_rows)
    initial_positive_categories = {str(row.get("category") or "") for row in initial_pos}
    combined_positive_categories = {str(row.get("category") or "") for row in combined_pos}
    initial_fragrance_pos = len([row for row in initial_pos if str(row.get("category") or "") == "fragrance"])
    combined_fragrance_pos = len([row for row in combined_pos if str(row.get("category") or "") == "fragrance"])

    if len(initial_haircare_pos) >= 50 and len({int(row["user_id"]) for row in initial_haircare_pos}) >= 20:
        haircare_initial = _decision("yes", "Enough trusted haircare initial positives and user coverage for a narrow baseline.")
    elif len(initial_haircare_pos) >= 15 and len({int(row["user_id"]) for row in initial_haircare_pos}) >= 10:
        haircare_initial = _decision("borderline", "Haircare initial positives exist, but sample size is still small.")
    else:
        haircare_initial = _decision("no", "Too few trusted haircare initial positives for a stable baseline.")

    if len(initial_pos) >= 150 and len(initial_positive_categories) >= 3 and initial_fragrance_pos >= 25:
        multi_initial = _decision("yes", "Initial positives cover multiple categories including fragrance.")
    elif len(initial_pos) >= 40 and len(initial_positive_categories) >= 2:
        multi_initial = _decision("borderline", "Some multi-category initial signal exists, but coverage is shallow.")
    else:
        multi_initial = _decision("no", "Initial positives are too sparse or too concentrated in one category.")

    if len(continuation_pos) >= 100 and len(combined_positive_categories) >= 3 and combined_fragrance_pos >= 25:
        full_planner = _decision("yes", "Continuation positives are deep enough to train lifecycle transitions.")
    elif len(continuation_pos) >= 25 and len(combined_positive_categories) >= 2:
        full_planner = _decision("borderline", "Continuation signal exists, but not enough for a robust full planner.")
    else:
        full_planner = _decision("no", "Continuation positives are too sparse and category coverage is too weak.")

    return {
        "haircare_only_initial_planner": haircare_initial,
        "multi_category_initial_planner": multi_initial,
        "full_planner_with_continuation_transitions": full_planner,
    }
