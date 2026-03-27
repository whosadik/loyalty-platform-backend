from __future__ import annotations

import json
import sys
from bisect import bisect_left
from collections import Counter, defaultdict
from datetime import timedelta, timezone as dt_timezone
from functools import lru_cache
from pathlib import Path
from typing import Any

from django.core.management.base import BaseCommand, CommandError
from django.utils import timezone

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from admin_tools.roadmap_teacher import (  # noqa: E402
    _estimate_current_owned_tokens_before_anchor,
    _position_key,
    _seed_action_token,
    _stable_user_split,
    build_teacher_examples,
    load_teacher_source_data,
)
from roadmap_app.content_features import build_base_content_features, product_signature, profile_signature  # noqa: E402
from ml.training.roadmap_initial_planner_common import resolve_path  # noqa: E402
from roadmap_app.ml_initial_planner import rollout_initial_plan  # noqa: E402
from roadmap_app.services import (  # noqa: E402
    CATEGORY_RULES,
    FRAGRANCE_DEFAULT_CHAIN,
    FRAGRANCE_SLOTS,
    _distinct_catalog_types,
)
from transactions.models import TransactionItem  # noqa: E402
from users_app.models import CustomerProfile  # noqa: E402


def _selected_categories(raw: str) -> list[str]:
    if not str(raw or "").strip():
        return list(CATEGORY_RULES.keys())
    out: list[str] = []
    for item in str(raw).split(","):
        token = str(item or "").strip().lower()
        if token and token not in out:
            out.append(token)
    return out


def _unique(values: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for raw in values:
        token = str(raw or "").strip().lower()
        if not token or token in seen:
            continue
        seen.add(token)
        out.append(token)
    return out


def _sanitize_ml_state(row: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in dict(row or {}).items():
        token = str(key or "").strip().lower()
        if token in {"planning_id", "user_id", "split"}:
            continue
        if token.startswith(("target_", "teacher_")):
            continue
        if token in {"label", "y", "candidate_type", "teacher_target_at_position"}:
            continue
        out[str(key)] = value
    return out


@lru_cache(maxsize=16)
def _catalog_type_pool(category: str) -> tuple[str, ...]:
    return tuple(_distinct_catalog_types(str(category or "").strip().lower(), exclude=None, limit=100))


def _catalog_types(category: str, *, exclude: set[str], limit: int) -> list[str]:
    out: list[str] = []
    for token in _catalog_type_pool(category):
        if token in exclude:
            continue
        out.append(token)
        if len(out) >= int(limit):
            break
    return out


def _build_runtime_initial_chain(
    *,
    category: str,
    seed_product: dict[str, Any],
    prior_category_items: list[dict[str, Any]],
    anchor_ts,
) -> tuple[list[str], dict[str, Any]]:
    category = str(category or "").strip().lower()
    rules = CATEGORY_RULES[category]
    min_steps = int(rules["min_steps"])
    max_steps = int(rules["max_steps"])
    seed_token = _seed_action_token(category, seed_product)
    purchased_types = [seed_token] if seed_token and seed_token != "__none__" else []
    owned_types_ordered = _estimate_current_owned_tokens_before_anchor(
        category=category,
        prior_items=prior_category_items,
        anchor_ts=anchor_ts,
    )

    debug: dict[str, Any] = {
        "seed_token": seed_token,
        "purchased_types": list(purchased_types),
        "owned_types_ordered": list(owned_types_ordered),
        "recent_types_before_anchor": [],
    }
    if category == "fragrance":
        purchased_slots = [x for x in purchased_types if x in FRAGRANCE_SLOTS]
        owned_slots = [x for x in owned_types_ordered if x in FRAGRANCE_SLOTS]
        chain = _unique(purchased_slots + list(FRAGRANCE_DEFAULT_CHAIN) + owned_slots)
        target_len = max(min_steps, min(max_steps, len(chain)))
        chain = chain[:target_len]
        debug["candidate_chain_before_clip"] = list(_unique(purchased_slots + list(FRAGRANCE_DEFAULT_CHAIN) + owned_slots))
        debug["target_len"] = int(target_len)
        return chain, debug

    rule_chain = _unique(list(rules["base"]) + list(rules["optional"]))
    chain = _unique(purchased_types + rule_chain + owned_types_ordered)
    recent_types: list[str] = []
    for row in reversed(prior_category_items):
        token = str(((row.get("product") or {}).get("product_type")) or "").strip().lower()
        if token:
            recent_types.append(token)
        if len(recent_types) >= 40:
            break
    recent_types = _unique(recent_types)
    chain = _unique(chain + recent_types)
    chain = _unique(chain + _catalog_types(category, exclude=set(chain), limit=30))
    owned_signal = min(max_steps - min_steps, len(set(owned_types_ordered)))
    target_len = min_steps + max(0, owned_signal)
    target_len = max(min_steps, min(max_steps, target_len))
    if len(chain) < target_len:
        chain = _unique(chain + _catalog_types(category, exclude=set(chain), limit=40))
    debug["recent_types_before_anchor"] = list(recent_types)
    debug["candidate_chain_before_clip"] = list(chain)
    debug["target_len"] = int(target_len)
    return chain[: min(max_steps, target_len)], debug


def _first_step(chain: list[str]) -> str:
    return str(chain[0]) if chain else "__stop__"


def _position_matches(left: list[str], right: list[str]) -> dict[str, float]:
    out: dict[str, float] = {}
    width = max(len(left), len(right))
    for idx in range(width):
        lval = left[idx] if idx < len(left) else "__stop__"
        rval = right[idx] if idx < len(right) else "__stop__"
        out[str(idx + 1)] = float(int(str(lval) == str(rval)))
    return out


def _prefix_match_rate(left: list[str], right: list[str]) -> float:
    target_len = max(1, len(right))
    matched = 0
    for a, b in zip(left, right):
        if str(a) != str(b):
            break
        matched += 1
    return float(matched / target_len)


def _diff_reason(left: list[str], right: list[str], *, category: str) -> str:
    if list(left) == list(right):
        return "exact"
    if _first_step(left) != _first_step(right):
        return "first_step_diff"
    if category == "fragrance" and set(left) == set(right):
        return "fragrance_slot_order_diff"
    shorter = min(len(left), len(right))
    if list(left[:shorter]) == list(right[:shorter]):
        if category == "skincare":
            left_tail = left[shorter:]
            right_tail = right[shorter:]
            if left_tail == ["mask"] or right_tail == ["mask"]:
                return "skincare_mask_vs_stop_tail"
        return "tail_length_diff"
    return "mid_plan_diff"


def _safe_chain(chain: list[str], *, category: str) -> list[str]:
    if category == "fragrance":
        return [token for token in chain if token in FRAGRANCE_SLOTS]
    return [str(token) for token in chain]


def _load_limited_source_data(*, days: int, include_ga: bool, categories: list[str], limit_per_category: int) -> dict[str, Any]:
    now_utc = timezone.now().astimezone(dt_timezone.utc)
    since = now_utc - timedelta(days=days)
    anchor_qs = (
        TransactionItem.objects.filter(
            transaction__created_at__gte=since,
            transaction__created_at__lte=now_utc,
            product__category__in=sorted(categories),
        )
        .order_by("transaction__user_id", "product__category", "transaction__created_at", "transaction__id", "id")
        .values(
            "id",
            "transaction__user_id",
            "transaction__id",
            "transaction__created_at",
            "product_id",
            "product__category",
            "product__product_type",
            "product__brand",
            "product__price",
            "product__concerns",
            "product__actives",
            "product__flags",
            "product__supported_skin_types",
            "product__attrs",
            "product__ingredients_inci",
            "product__raw_meta",
            "transaction__user__username",
        )
    )

    anchor_rows: list[dict[str, Any]] = []
    seen_anchor_keys: set[tuple[int, str]] = set()
    counts_by_category: Counter[str] = Counter()
    for row in anchor_qs.iterator(chunk_size=5000):
        username = str(row.get("transaction__user__username") or "")
        if not include_ga and username.startswith("ga_"):
            continue
        user_id = int(row["transaction__user_id"])
        category = str(row["product__category"] or "").strip().lower()
        if category not in set(categories):
            continue
        key = (user_id, category)
        if key in seen_anchor_keys:
            continue
        if int(limit_per_category) > 0 and int(counts_by_category[category]) >= int(limit_per_category):
            continue
        seen_anchor_keys.add(key)
        counts_by_category[category] += 1
        anchor_rows.append(
            {
                "planning_id": int(len(anchor_rows) + 1),
                "user_id": user_id,
                "category": category,
                "anchor_ts": row["transaction__created_at"].astimezone(dt_timezone.utc),
                "anchor_tx_id": int(row["transaction__id"]),
                "anchor_item_id": int(row["id"]),
                "seed_product": {
                    "id": int(row["product_id"]),
                    "category": category,
                    "product_type": str(row["product__product_type"] or "").strip().lower(),
                    "brand": str(row.get("product__brand") or "").strip().lower(),
                    "price": float(row.get("product__price") or 0.0),
                    "concerns": row.get("product__concerns") if isinstance(row.get("product__concerns"), list) else [],
                    "actives": row.get("product__actives") if isinstance(row.get("product__actives"), list) else [],
                    "flags": row.get("product__flags") if isinstance(row.get("product__flags"), list) else [],
                    "supported_skin_types": row.get("product__supported_skin_types")
                    if isinstance(row.get("product__supported_skin_types"), list)
                    else [],
                    "attrs": row.get("product__attrs") if isinstance(row.get("product__attrs"), dict) else {},
                    "ingredients_inci": str(row.get("product__ingredients_inci") or ""),
                    "raw_meta": row.get("product__raw_meta") if isinstance(row.get("product__raw_meta"), dict) else {},
                },
            }
        )

    users = sorted({int(row["user_id"]) for row in anchor_rows})
    profile_map = {
        int(profile.user_id): profile_signature(profile)
        for profile in CustomerProfile.objects.filter(user_id__in=users)
    }

    tx_qs = TransactionItem.objects.filter(transaction__user_id__in=users).values(
        "id",
        "transaction__user_id",
        "transaction__id",
        "transaction__created_at",
        "transaction__total_amount",
        "product_id",
        "quantity",
        "product__category",
        "product__product_type",
        "product__brand",
        "product__price",
        "product__concerns",
        "product__actives",
        "product__flags",
        "product__supported_skin_types",
        "product__attrs",
        "product__ingredients_inci",
        "product__raw_meta",
    )
    user_items: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in tx_qs.iterator(chunk_size=5000):
        user_id = int(row["transaction__user_id"])
        category = str(row.get("product__category") or "").strip().lower()
        product = {
            "id": int(row["product_id"]),
            "category": category,
            "product_type": str(row.get("product__product_type") or "").strip().lower(),
            "brand": str(row.get("product__brand") or "").strip().lower(),
            "price": float(row.get("product__price") or 0.0),
            "concerns": row.get("product__concerns") if isinstance(row.get("product__concerns"), list) else [],
            "actives": row.get("product__actives") if isinstance(row.get("product__actives"), list) else [],
            "flags": row.get("product__flags") if isinstance(row.get("product__flags"), list) else [],
            "supported_skin_types": row.get("product__supported_skin_types")
            if isinstance(row.get("product__supported_skin_types"), list)
            else [],
            "attrs": row.get("product__attrs") if isinstance(row.get("product__attrs"), dict) else {},
            "ingredients_inci": str(row.get("product__ingredients_inci") or ""),
            "raw_meta": row.get("product__raw_meta") if isinstance(row.get("product__raw_meta"), dict) else {},
        }
        ts = row["transaction__created_at"].astimezone(dt_timezone.utc)
        user_items[user_id].append(
            {
                "position": _position_key(ts, int(row["transaction__id"]), int(row["id"])),
                "ts": ts,
                "tx_id": int(row["transaction__id"]),
                "item_id": int(row["id"]),
                "quantity": max(1, int(row.get("quantity") or 0)),
                "tx_total": float(row.get("transaction__total_amount") or 0.0),
                "category": category,
                "brand": str(product.get("brand") or ""),
                "product": product,
                "history_token": _seed_action_token(category, product),
            }
        )
    for rows in user_items.values():
        rows.sort(key=lambda row: row["position"])

    return {
        "now_utc": now_utc,
        "since": since,
        "anchor_rows": anchor_rows,
        "profile_map": profile_map,
        "user_items": user_items,
        "excluded_counts": {},
    }


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Roadmap Initial Planner Runtime Shadow Diff",
        "",
        f"- days: `{report['days']}`",
        f"- include_ga: `{report['include_ga']}`",
        f"- limit_per_category: `{report['limit_per_category']}`",
        f"- processed_anchors_total: `{report['processed_anchors_total']}`",
        f"- model_root: `{report['model_root']}`",
        "",
        "## Notes",
    ]
    for note in report.get("notes") or []:
        lines.append(f"- {note}")
    for category, payload in sorted((report.get("categories") or {}).items()):
        lines.extend(
            [
                "",
                f"## {category}",
                f"- anchors: `{payload['anchors']}`",
                f"- exact match ML vs teacher/runtime/rules: `{payload['exact_match_rate']['ml_vs_teacher']:.4f}` / `{payload['exact_match_rate']['ml_vs_runtime']:.4f}` / `{payload['exact_match_rate']['teacher_vs_runtime']:.4f}`",
                f"- first-step match ML vs teacher/runtime/rules: `{payload['first_step_match_rate']['ml_vs_teacher']:.4f}` / `{payload['first_step_match_rate']['ml_vs_runtime']:.4f}` / `{payload['first_step_match_rate']['teacher_vs_runtime']:.4f}`",
                f"- avg |length diff| ML vs teacher/runtime: `{payload['avg_abs_length_diff']['ml_vs_teacher']:.4f}` / `{payload['avg_abs_length_diff']['ml_vs_runtime']:.4f}`",
                f"- dominant runtime disagreement teacher: `{payload['runtime_disagreement_breakdown']['teacher_vs_runtime_top_reason']}`",
                f"- dominant runtime disagreement ml: `{payload['runtime_disagreement_breakdown']['ml_vs_runtime_top_reason']}`",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def _flatten_sequence_rows(examples: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    out: dict[int, dict[str, Any]] = {}
    for example in examples:
        profile_sig = dict(example.get("profile_signature") or {})
        seed_product = dict(example.get("seed_product") or {})
        seed_sig = product_signature(seed_product)
        base_content = build_base_content_features(profile_sig, seed_sig)
        sequence_row = {
            key: value
            for key, value in example.items()
            if key not in {"profile_signature", "seed_product", "seed_signature"}
        }
        sequence_row.update(base_content)
        out[int(example["planning_id"])] = sequence_row
    return out


class Command(BaseCommand):
    help = "Compare ML initial planner vs teacher policy vs current runtime-rule initial roadmap policy on real anchors."

    def add_arguments(self, parser):
        parser.add_argument("--days", type=int, default=3650)
        parser.add_argument("--include-ga", action="store_true")
        parser.add_argument("--seed", type=int, default=42)
        parser.add_argument("--categories", type=str, default="")
        parser.add_argument("--limit-per-category", type=int, default=0)
        parser.add_argument("--sample-cases", type=int, default=20)
        parser.add_argument("--model-root", type=str, default="models/roadmap_initial_planner")
        parser.add_argument("--report-md", type=str, default="reports/roadmap_initial_planner_runtime_shadow_diff.md")
        parser.add_argument("--report-json", type=str, default="reports/roadmap_initial_planner_runtime_shadow_diff.json")

    def handle(self, *args, **options):
        categories = _selected_categories(str(options["categories"]))
        days = max(1, int(options["days"]))
        include_ga = bool(options["include_ga"])
        seed = int(options["seed"])
        limit_per_category = max(0, int(options["limit_per_category"]))
        sample_cases = max(1, int(options["sample_cases"]))
        model_root = resolve_path(str(options["model_root"]))
        report_md_path = resolve_path(str(options["report_md"]))
        report_json_path = resolve_path(str(options["report_json"]))
        report_md_path.parent.mkdir(parents=True, exist_ok=True)
        report_json_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            if limit_per_category > 0:
                source_data = _load_limited_source_data(
                    days=days,
                    include_ga=include_ga,
                    categories=categories,
                    limit_per_category=limit_per_category,
                )
            else:
                source_data = load_teacher_source_data(days=days, include_ga=include_ga)
        except Exception as exc:  # pragma: no cover
            raise CommandError(str(exc)) from exc
        examples_payload = build_teacher_examples(source_data, seed=seed)
        examples = [dict(row) for row in (examples_payload.get("examples") or [])]
        examples_map = {int(row["planning_id"]): row for row in examples}
        sequence_row_map = _flatten_sequence_rows(examples)
        profile_map = dict(source_data.get("profile_map") or {})
        user_items = dict(source_data.get("user_items") or {})

        anchors_by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for anchor in source_data.get("anchor_rows") or []:
            category = str(anchor.get("category") or "").strip().lower()
            if category not in categories:
                continue
            planning_id = int(anchor["planning_id"])
            if planning_id not in examples_map:
                continue
            anchors_by_category[category].append(dict(anchor))

        report: dict[str, Any] = {
            "days": days,
            "include_ga": include_ga,
            "limit_per_category": limit_per_category,
            "model_root": str(model_root),
            "processed_anchors_total": 0,
            "categories": {},
            "notes": [
                "runtime comparison uses a historical-safe simulator of initial _build_chain ordering, not live refresh_roadmap(), to avoid future leakage",
                "ML rollout input is sanitized to exclude target_/teacher_ columns before calling the shadow wrapper",
                "when limit_per_category > 0, the report covers a deterministic first-anchor slice per category rather than the full corpus",
            ],
        }

        for category in categories:
            category_anchors = anchors_by_category.get(category) or []
            if not category_anchors:
                continue
            cases: list[dict[str, Any]] = []
            samples: dict[str, list[dict[str, Any]]] = {
                "ml_not_runtime": [],
                "teacher_not_runtime": [],
                "ml_eq_teacher_not_runtime": [],
                "skincare_tail_diff": [],
                "fragrance_slot_order_diff": [],
            }

            for anchor in category_anchors:
                planning_id = int(anchor["planning_id"])
                example_row = dict(examples_map[planning_id])
                sequence_row = dict(sequence_row_map[planning_id])
                user_id = int(anchor["user_id"])
                rows = user_items.get(user_id) or []
                positions = [row["position"] for row in rows]
                anchor_position = _position_key(anchor["anchor_ts"], int(anchor["anchor_tx_id"]), int(anchor["anchor_item_id"]))
                pivot = bisect_left(positions, anchor_position)
                prior_items = rows[:pivot]
                prior_category_items = [row for row in prior_items if str(row.get("category") or "") == category]

                teacher_chain = _safe_chain(json.loads(str(example_row.get("target_sequence_json") or "[]")), category=category)
                ml_chain = _safe_chain(
                    rollout_initial_plan(category, _sanitize_ml_state(sequence_row), model_root=model_root),
                    category=category,
                )
                runtime_chain, runtime_debug = _build_runtime_initial_chain(
                    category=category,
                    seed_product=dict(anchor.get("seed_product") or {}),
                    prior_category_items=prior_category_items,
                    anchor_ts=anchor["anchor_ts"],
                )
                runtime_chain = _safe_chain(runtime_chain, category=category)

                teacher_runtime_reason = _diff_reason(teacher_chain, runtime_chain, category=category)
                ml_runtime_reason = _diff_reason(ml_chain, runtime_chain, category=category)
                case = {
                    "planning_id": planning_id,
                    "user_id": user_id,
                    "split": _stable_user_split(user_id, seed=seed),
                    "anchor_ts": anchor["anchor_ts"].isoformat().replace("+00:00", "Z"),
                    "seed_product_id": int((anchor.get("seed_product") or {}).get("id") or 0),
                    "seed_action_token": str(example_row.get("seed_action_token") or "__none__"),
                    "teacher_chain": teacher_chain,
                    "ml_chain": ml_chain,
                    "runtime_chain": runtime_chain,
                    "teacher_vs_runtime_reason": teacher_runtime_reason,
                    "ml_vs_runtime_reason": ml_runtime_reason,
                    "runtime_debug": runtime_debug,
                    "profile_signature": dict(profile_map.get(user_id) or {}),
                }
                cases.append(case)

                if teacher_chain != runtime_chain and len(samples["teacher_not_runtime"]) < sample_cases:
                    samples["teacher_not_runtime"].append(case)
                if ml_chain != runtime_chain and len(samples["ml_not_runtime"]) < sample_cases:
                    samples["ml_not_runtime"].append(case)
                if ml_chain == teacher_chain and ml_chain != runtime_chain and len(samples["ml_eq_teacher_not_runtime"]) < sample_cases:
                    samples["ml_eq_teacher_not_runtime"].append(case)
                if category == "skincare" and (
                    teacher_runtime_reason == "skincare_mask_vs_stop_tail" or ml_runtime_reason == "skincare_mask_vs_stop_tail"
                ) and len(samples["skincare_tail_diff"]) < sample_cases:
                    samples["skincare_tail_diff"].append(case)
                if category == "fragrance" and (
                    teacher_runtime_reason == "fragrance_slot_order_diff" or ml_runtime_reason == "fragrance_slot_order_diff"
                ) and len(samples["fragrance_slot_order_diff"]) < sample_cases:
                    samples["fragrance_slot_order_diff"].append(case)

            report["processed_anchors_total"] += int(len(cases))
            teacher_runtime_reasons = Counter(case["teacher_vs_runtime_reason"] for case in cases if case["teacher_chain"] != case["runtime_chain"])
            ml_runtime_reasons = Counter(case["ml_vs_runtime_reason"] for case in cases if case["ml_chain"] != case["runtime_chain"])

            def _rate(left_key: str, right_key: str) -> float:
                return float(
                    sum(int(case[left_key] == case[right_key]) for case in cases) / max(1, len(cases))
                )

            def _first_rate(left_key: str, right_key: str) -> float:
                return float(
                    sum(int(_first_step(case[left_key]) == _first_step(case[right_key])) for case in cases) / max(1, len(cases))
                )

            def _avg_len(left_key: str, right_key: str) -> float:
                return float(
                    sum(abs(len(case[left_key]) - len(case[right_key])) for case in cases) / max(1, len(cases))
                )

            per_position_ml_teacher: dict[str, list[float]] = defaultdict(list)
            per_position_ml_runtime: dict[str, list[float]] = defaultdict(list)
            per_position_teacher_runtime: dict[str, list[float]] = defaultdict(list)
            for case in cases:
                for key, value in _position_matches(case["ml_chain"], case["teacher_chain"]).items():
                    per_position_ml_teacher[key].append(float(value))
                for key, value in _position_matches(case["ml_chain"], case["runtime_chain"]).items():
                    per_position_ml_runtime[key].append(float(value))
                for key, value in _position_matches(case["teacher_chain"], case["runtime_chain"]).items():
                    per_position_teacher_runtime[key].append(float(value))

            report["categories"][category] = {
                "anchors": int(len(cases)),
                "exact_match_rate": {
                    "ml_vs_teacher": _rate("ml_chain", "teacher_chain"),
                    "ml_vs_runtime": _rate("ml_chain", "runtime_chain"),
                    "teacher_vs_runtime": _rate("teacher_chain", "runtime_chain"),
                },
                "first_step_match_rate": {
                    "ml_vs_teacher": _first_rate("ml_chain", "teacher_chain"),
                    "ml_vs_runtime": _first_rate("ml_chain", "runtime_chain"),
                    "teacher_vs_runtime": _first_rate("teacher_chain", "runtime_chain"),
                },
                "avg_abs_length_diff": {
                    "ml_vs_teacher": _avg_len("ml_chain", "teacher_chain"),
                    "ml_vs_runtime": _avg_len("ml_chain", "runtime_chain"),
                    "teacher_vs_runtime": _avg_len("teacher_chain", "runtime_chain"),
                },
                "prefix_match_rate": {
                    "ml_vs_teacher": float(sum(_prefix_match_rate(case["ml_chain"], case["teacher_chain"]) for case in cases) / max(1, len(cases))),
                    "ml_vs_runtime": float(sum(_prefix_match_rate(case["ml_chain"], case["runtime_chain"]) for case in cases) / max(1, len(cases))),
                    "teacher_vs_runtime": float(sum(_prefix_match_rate(case["teacher_chain"], case["runtime_chain"]) for case in cases) / max(1, len(cases))),
                },
                "per_position_match_rate": {
                    "ml_vs_teacher": {key: float(sum(vals) / max(1, len(vals))) for key, vals in sorted(per_position_ml_teacher.items(), key=lambda item: int(item[0]))},
                    "ml_vs_runtime": {key: float(sum(vals) / max(1, len(vals))) for key, vals in sorted(per_position_ml_runtime.items(), key=lambda item: int(item[0]))},
                    "teacher_vs_runtime": {key: float(sum(vals) / max(1, len(vals))) for key, vals in sorted(per_position_teacher_runtime.items(), key=lambda item: int(item[0]))},
                },
                "runtime_disagreement_breakdown": {
                    "teacher_vs_runtime": dict(teacher_runtime_reasons),
                    "ml_vs_runtime": dict(ml_runtime_reasons),
                    "teacher_vs_runtime_top_reason": teacher_runtime_reasons.most_common(1)[0][0] if teacher_runtime_reasons else "exact",
                    "ml_vs_runtime_top_reason": ml_runtime_reasons.most_common(1)[0][0] if ml_runtime_reasons else "exact",
                },
                "sample_cases": samples,
            }

        report_json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        report_md_path.write_text(_markdown(report), encoding="utf-8")
        self.stdout.write(f"[report_roadmap_initial_planner_runtime_shadow_diff] json={report_json_path}")
        self.stdout.write(f"[report_roadmap_initial_planner_runtime_shadow_diff] md={report_md_path}")
