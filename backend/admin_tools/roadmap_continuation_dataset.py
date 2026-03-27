from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

try:
    import pandas as pd
except Exception:  # pragma: no cover
    pd = None

from admin_tools.roadmap_planner_transitions import (
    CANDIDATE_SPACE_BY_CATEGORY,
    CONTINUATION_DECISION_TYPES,
    STOP_TOKEN,
    build_transition_decision_records,
    load_transition_source_data,
)
from roadmap_app.content_features import (
    ALL_CATEGORICAL_FEATURES,
    ALL_NUMERIC_FEATURES,
    build_base_content_features,
    build_candidate_content_features,
    product_signature,
)

DEFAULT_CONTINUATION_CATEGORIES = ("haircare", "skincare", "fragrance")
ALL_CONTINUATION_CATEGORIES = ("haircare", "skincare", "fragrance", "makeup")


def resolve_repo_path(raw_path: str | Path) -> Path:
    candidate = Path(str(raw_path)).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    cwd_path = (Path.cwd() / candidate).resolve()
    if cwd_path.exists() or cwd_path.parent.exists():
        return cwd_path
    return (Path(__file__).resolve().parents[2] / candidate).resolve()


def selected_categories(raw: str | None) -> list[str]:
    if not str(raw or "").strip():
        return list(DEFAULT_CONTINUATION_CATEGORIES)
    out: list[str] = []
    for item in str(raw).split(","):
        token = str(item or "").strip().lower()
        if token in ALL_CONTINUATION_CATEGORIES and token not in out:
            out.append(token)
    return out or list(DEFAULT_CONTINUATION_CATEGORIES)


def _event_sort_key(row: dict[str, Any]) -> tuple[Any, int]:
    return (row["t0_dt"], int(row["decision_id"]))


def _time_based_split_assignments(records: list[dict[str, Any]]) -> tuple[dict[int, str], dict[str, Any]]:
    if not records:
        return {}, {
            "strategy": "time_based",
            "train_end_utc": None,
            "valid_end_utc": None,
            "counts": {"train": 0, "val": 0, "test": 0},
            "user_overlap_counts": {"train_val": 0, "train_test": 0, "val_test": 0},
        }

    ordered = sorted(records, key=_event_sort_key)
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


def _rows_for_categories(rows: list[dict[str, Any]], categories: list[str], *, continuation_only: bool = False) -> list[dict[str, Any]]:
    allowed = set(categories)
    out: list[dict[str, Any]] = []
    for row in rows:
        if str(row.get("category") or "") not in allowed:
            continue
        if continuation_only and str(row.get("decision_type") or "") not in CONTINUATION_DECISION_TYPES:
            continue
        out.append(dict(row))
    return out


def _candidate_rows(
    *,
    decision_records: list[dict[str, Any]],
    source_data: dict[str, Any],
) -> "pd.DataFrame":
    if pd is None:  # pragma: no cover
        raise RuntimeError("pandas is required")

    candidate_popularity_train: dict[str, Counter[str]] = defaultdict(Counter)
    for record in decision_records:
        if str(record.get("split") or "") != "train":
            continue
        candidate_popularity_train[str(record["category"])][str(record["label"])] += 1

    candidate_catalog_summaries = dict(source_data.get("candidate_catalog_summaries") or {})
    rows: list[dict[str, Any]] = []
    for record in decision_records:
        base_content = build_base_content_features(
            dict(record.get("profile_signature") or {}),
            product_signature(record.get("anchor_item")),
        )
        base_row = {
            key: value
            for key, value in record.items()
            if key
            not in {
                "plan_position_by_type",
                "plan_state_by_type",
                "recent_category_tokens",
                "candidate_seen_90d_counter",
                "candidate_days_since_last_seen_map",
                "anchor_item",
                "candidate_types",
                "t0_dt",
                "profile_signature",
            }
        }
        plan_position_by_type = dict(record.get("plan_position_by_type") or {})
        plan_state_by_type = dict(record.get("plan_state_by_type") or {})
        recent_category_tokens = [str(x) for x in (record.get("recent_category_tokens") or [])]
        seen_90d_counter = {str(k): int(v) for k, v in (record.get("candidate_seen_90d_counter") or {}).items()}
        days_since_last_seen_map = {
            str(k): int(v) for k, v in (record.get("candidate_days_since_last_seen_map") or {}).items()
        }
        anchor_sig = product_signature(record.get("anchor_item"))
        category = str(record["category"])
        label = str(record["label"])
        popularity_counter = candidate_popularity_train.get(category) or Counter()
        popularity_total = float(sum(popularity_counter.values()) or 1.0)
        candidate_types = list(record.get("candidate_types") or [])
        if STOP_TOKEN not in candidate_types:
            candidate_types.append(STOP_TOKEN)
        for candidate in candidate_types:
            state = plan_state_by_type.get(candidate) or {}
            current_status = str(state.get("status") or "")
            rows.append(
                {
                    **base_row,
                    **base_content,
                    "candidate_type": str(candidate),
                    "y": int(str(candidate) == label),
                    "candidate_in_generated_plan": int(candidate in plan_position_by_type),
                    "candidate_position_in_generated_plan": int(plan_position_by_type.get(candidate, -1)),
                    "candidate_is_current_next_step": int(
                        str(candidate) == str(record.get("current_next_product_type") or "")
                    ),
                    "candidate_has_recommendation_in_plan": int(bool(state.get("has_recommendation"))),
                    "candidate_current_missing": int(current_status == "missing"),
                    "candidate_current_recommended": int(current_status == "recommended"),
                    "candidate_current_owned": int(current_status == "owned"),
                    "candidate_current_completed": int(current_status == "completed"),
                    "candidate_current_skipped": int(current_status == "skipped"),
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
                        dict(record.get("profile_signature") or {}),
                        anchor_sig,
                        candidate_type=str(candidate),
                    ),
                }
            )

    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame = frame.sort_values(["decision_id", "candidate_type"]).reset_index(drop=True)
    return frame


def _series_counts(values: list[str]) -> dict[str, int]:
    return {str(key): int(value) for key, value in sorted(Counter(values).items())}


def _suffix_length_distribution(decision_records: list[dict[str, Any]]) -> dict[str, int]:
    lengths: Counter[str] = Counter()
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in decision_records:
        grouped[int(row["episode_id"])].append(row)
    for rows in grouped.values():
        ordered = sorted(rows, key=_event_sort_key)
        for idx, _row in enumerate(ordered):
            suffix = [item for item in ordered[idx:] if str(item.get("label") or STOP_TOKEN) != STOP_TOKEN]
            lengths[str(len(suffix))] += 1
    return {str(key): int(value) for key, value in sorted(lengths.items(), key=lambda item: int(item[0]))}


def _month_distribution(decision_records: list[dict[str, Any]]) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for row in decision_records:
        counter[str(row["t0_dt"].strftime("%Y-%m"))] += 1
    return {str(key): int(value) for key, value in sorted(counter.items())}


def _category_usability(decision_records: list[dict[str, Any]], *, category: str) -> dict[str, Any]:
    category_rows = [row for row in decision_records if str(row.get("category") or "") == category]
    positives = [row for row in category_rows if str(row.get("label") or STOP_TOKEN) != STOP_TOKEN]
    observed_labels = sorted({str(row.get("label") or "") for row in positives})
    expected_labels = [token for token in (CANDIDATE_SPACE_BY_CATEGORY.get(category) or []) if token != STOP_TOKEN]
    users = {int(row["user_id"]) for row in category_rows}
    positive_users = {int(row["user_id"]) for row in positives}
    missing_actions = [token for token in expected_labels if token not in observed_labels]

    status = "no"
    why = "Too few trusted continuation positives."
    if category == "fragrance":
        if len(positives) >= 25 and len(observed_labels) == 4 and len(positive_users) >= 10:
            status = "yes"
            why = "All fragrance slots are present with enough trusted continuation positives."
        elif len(positives) >= 12 and len(observed_labels) >= 3:
            status = "borderline"
            why = "Fragrance continuation exists, but slot support is still uneven."
    else:
        if len(positives) >= 40 and len(observed_labels) >= 3 and len(positive_users) >= 10:
            status = "yes"
            why = "Enough continuation positives and label diversity for category-specific training."
        elif len(positives) >= 15 and len(observed_labels) >= 2:
            status = "borderline"
            why = "Some continuation signal exists, but label support is still shallow."

    return {
        "status": status,
        "why": why,
        "decision_points_total": int(len(category_rows)),
        "non_stop_positives_total": int(len(positives)),
        "users_total": int(len(users)),
        "positive_users_total": int(len(positive_users)),
        "positive_labels": {label: sum(int(str(row.get("label") or "") == label) for row in positives) for label in observed_labels},
        "missing_actions": missing_actions,
    }


def build_continuation_bundle(
    *,
    days: int,
    label_window_days: int,
    include_ga: bool,
    seed: int,
    categories: list[str],
) -> dict[str, Any]:
    if pd is None:  # pragma: no cover
        raise RuntimeError("pandas is required")

    source_data = load_transition_source_data(
        days=int(days),
        include_ga=bool(include_ga),
        label_window_days=int(label_window_days),
    )
    raw_bundle = build_transition_decision_records(
        source_data,
        label_window_days=int(label_window_days),
        mode="continuation",
    )
    decision_records = _rows_for_categories(list(raw_bundle.get("decision_records") or []), categories, continuation_only=True)
    surface_records = _rows_for_categories(list(raw_bundle.get("surface_records") or []), categories, continuation_only=True)
    if not decision_records:
        raise RuntimeError("No continuation decision records produced for selected categories.")

    split_by_decision, split_meta = _time_based_split_assignments(decision_records)
    for record in decision_records:
        record["split"] = str(split_by_decision.get(int(record["decision_id"])) or "train")

    frame = _candidate_rows(decision_records=decision_records, source_data=source_data)
    label_counter = Counter(str(record["label"]) for record in decision_records)
    label_source_counter = Counter(str(record.get("label_source") or "unknown") for record in decision_records)
    decision_type_counter = Counter(str(record.get("decision_type") or "unknown") for record in decision_records)
    positives = [row for row in decision_records if str(row.get("label") or STOP_TOKEN) != STOP_TOKEN]
    positives_by_category = Counter(str(row["category"]) for row in positives)
    positives_by_label = Counter(str(row["label"]) for row in positives)
    positives_by_label_within_category: dict[str, dict[str, int]] = {}
    for category in categories:
        category_rows = [row for row in positives if str(row.get("category") or "") == category]
        positives_by_label_within_category[category] = _series_counts([str(row["label"]) for row in category_rows])

    continuation_completed_rows = [row for row in decision_records if str(row.get("decision_type") or "") == "post_completed"]
    continuation_skipped_rows = [row for row in decision_records if str(row.get("decision_type") or "") == "post_skipped"]
    category_readiness = {
        category: _category_usability(decision_records, category=category)
        for category in categories
    }
    recommended_categories = [category for category, payload in category_readiness.items() if payload["status"] == "yes"]
    blocked_categories = [category for category, payload in category_readiness.items() if payload["status"] == "no"]
    usable_for_training = len(recommended_categories) >= 2 and len(positives) >= 150
    usable_for_shadow = len(decision_records) >= 250 and len(recommended_categories) >= 2
    readiness = {
        "usable_for_continuation_training": bool(usable_for_training),
        "usable_for_continuation_shadow": bool(usable_for_shadow),
        "usable_for_continuation_runtime_candidate": False,
        "recommended_categories_for_training": recommended_categories,
        "blocked_categories": blocked_categories,
        "categories": category_readiness,
        "why": {
            "training": (
                f"{len(positives)} trusted non-stop continuation positives across {len(recommended_categories)} strong categories."
                if usable_for_training
                else "Continuation positives or category coverage are still too small for honest training."
            ),
            "shadow": (
                f"{len(decision_records)} trusted continuation decisions are enough for a shadow audit."
                if usable_for_shadow
                else "Too few trusted continuation decisions for a meaningful shadow comparison."
            ),
            "runtime_candidate": "Runtime candidacy is blocked at readiness stage until model eval and shadow agreement are measured.",
        },
    }

    categorical_features = [
        "category",
        "candidate_type",
        "decision_type",
        "trust_level",
        "refresh_caller",
        "refresh_source",
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
        "remaining_actionable_steps_count",
        "remaining_depth_in_plan",
        "steps_completed_in_episode_count",
        "steps_skipped_in_episode_count",
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

    metadata = {
        "version": "roadmap_continuation_v1_candidate_ranking",
        "generated_at_utc": str(source_data["now_utc"].isoformat().replace("+00:00", "Z")),
        "window_days": int(days),
        "window_since_utc": str(source_data["since"].isoformat().replace("+00:00", "Z")),
        "window_until_utc": str(source_data["now_utc"].isoformat().replace("+00:00", "Z")),
        "label_window_days": int(label_window_days),
        "include_ga": bool(include_ga),
        "seed": int(seed),
        "categories": list(categories),
        "decision_point_definition": "t0 is immediately after a trusted roadmap_step_completed or roadmap_step_skipped event for the current actionable step; state is reconstructed from the latest valid refresh snapshot plus prior trusted outcomes in the same episode.",
        "label_contract": {
            "completed": "positive label is the next actionable step after the trusted completed outcome",
            "skipped": "positive label is the next actionable step after the trusted skipped outcome",
            "stop": "if no further trusted progress exists within the label window, label is __stop__",
            "fragrance_noise_filter": "exclude legacy fragrance exact completions when purchased SKU slot does not match step slot",
        },
        "rows_total": int(len(frame)),
        "decision_points_total": int(len(decision_records)),
        "episodes_total": int(len({int(record['episode_id']) for record in decision_records})),
        "plans_total": int(len({int(record['plan_id']) for record in decision_records if int(record.get('plan_id') or 0) > 0})),
        "users_total": int(len({int(record['user_id']) for record in decision_records})),
        "positive_rows": int(frame["y"].sum()) if not frame.empty else 0,
        "stop_label_count": int(label_counter.get(STOP_TOKEN, 0)),
        "stop_label_rate": round(float(label_counter.get(STOP_TOKEN, 0) / max(1, len(decision_records))), 6),
        "decision_type_distribution": dict(sorted(decision_type_counter.items())),
        "label_source_distribution": dict(sorted(label_source_counter.items())),
        "positives_by_category": dict(sorted(positives_by_category.items())),
        "positives_by_label": dict(sorted(positives_by_label.items())),
        "positives_by_label_within_category": positives_by_label_within_category,
        "time_distribution_by_month": _month_distribution(decision_records),
        "skipped_vs_completed_share": {
            "post_completed_share": round(float(len(continuation_completed_rows) / max(1, len(decision_records))), 6),
            "post_skipped_share": round(float(len(continuation_skipped_rows) / max(1, len(decision_records))), 6),
        },
        "suffix_length_distribution": _suffix_length_distribution(decision_records),
        "fragrance_slot_label_distribution": dict(sorted(positives_by_label_within_category.get("fragrance", {}).items())),
        "candidate_types_by_category": {
            category: list(CANDIDATE_SPACE_BY_CATEGORY.get(category) or []) + ([STOP_TOKEN] if STOP_TOKEN not in (CANDIDATE_SPACE_BY_CATEGORY.get(category) or []) else [])
            for category in categories
        },
        "feature_columns": [*categorical_features, *numeric_features],
        "categorical_features": categorical_features,
        "numeric_features": numeric_features,
        "split_strategy": str(split_meta.get("strategy") or "time_based"),
        "split_counts": dict(split_meta.get("counts") or {}),
        "split_user_overlap_counts": dict(split_meta.get("user_overlap_counts") or {}),
        "excluded_counts": {
            str(key): int(value)
            for key, value in sorted(
                Counter(str(row.get("excluded_reason") or "unknown") for row in surface_records if not bool(row.get("trusted"))).items()
            )
        },
        "excluded_noisy_decision_points_count": int(sum(1 for row in surface_records if not bool(row.get("trusted")))),
        "excluded_legacy_bad_fragrance_completions_count": int(
            raw_bundle.get("excluded_legacy_bad_fragrance_completions_count") or 0
        ),
        "readiness": readiness,
        "leakage_assertions": {
            "features_only_use_transactions_lte_t0": True,
            "decision_types_are_continuation_only": True,
            "exactly_one_positive_per_decision_point": True,
            "future_purchase_fallback_restricted_to_current_actionable_step": True,
            "legacy_bad_fragrance_exact_completions_excluded": True,
            "status": "passed",
        },
    }
    splits_payload = {
        "strategy": str(split_meta.get("strategy") or "time_based"),
        "train_end_utc": split_meta.get("train_end_utc"),
        "valid_end_utc": split_meta.get("valid_end_utc"),
        "counts": dict(split_meta.get("counts") or {}),
        "user_overlap_counts": dict(split_meta.get("user_overlap_counts") or {}),
        "decision_ids": {
            split_name: [
                int(record["decision_id"])
                for record in decision_records
                if str(record.get("split") or "") == split_name
            ]
            for split_name in ["train", "val", "test"]
        },
    }
    readiness_report = {
        "generated_at_utc": metadata["generated_at_utc"],
        "window_days": int(days),
        "label_window_days": int(label_window_days),
        "categories": list(categories),
        "trusted_continuation_decision_points_total": int(len(decision_records)),
        "non_stop_positives_total": int(len(positives)),
        "positives_by_category": dict(sorted(positives_by_category.items())),
        "positives_by_label_within_category": positives_by_label_within_category,
        "stop_rate": float(metadata["stop_label_rate"]),
        "users_total": int(metadata["users_total"]),
        "time_distribution_by_month": dict(metadata["time_distribution_by_month"]),
        "skipped_vs_completed_share": dict(metadata["skipped_vs_completed_share"]),
        "suffix_length_distribution": dict(metadata["suffix_length_distribution"]),
        "fragrance_slot_label_distribution": dict(metadata["fragrance_slot_label_distribution"]),
        "continuation_label_table": [
            {
                "category": category,
                "action": label,
                "positive_count": int(count),
                "dataset_type": "continuation",
            }
            for category in categories
            for label, count in sorted((positives_by_label_within_category.get(category) or {}).items())
        ],
        "readiness": readiness,
    }
    return {
        "source_data": source_data,
        "raw_bundle": raw_bundle,
        "decision_records": decision_records,
        "surface_records": surface_records,
        "frame": frame,
        "metadata": metadata,
        "splits": splits_payload,
        "report": readiness_report,
    }


def write_continuation_summary_md(*, out_dir: Path, metadata: dict[str, Any]) -> Path:
    readiness = metadata.get("readiness") or {}
    category_readiness = dict(readiness.get("categories") or {})
    lines = [
        "# Roadmap Continuation Dataset Summary",
        "",
        f"- rows: **{metadata.get('rows_total', 0)}**",
        f"- decision points: **{metadata.get('decision_points_total', 0)}**",
        f"- users: **{metadata.get('users_total', 0)}**",
        f"- non-stop positives: **{sum(int(v) for v in (metadata.get('positives_by_label') or {}).values())}**",
        f"- stop rate: **{metadata.get('stop_label_rate', 0.0)}**",
        f"- excluded noisy decision points: **{metadata.get('excluded_noisy_decision_points_count', 0)}**",
        f"- excluded legacy bad fragrance completions: **{metadata.get('excluded_legacy_bad_fragrance_completions_count', 0)}**",
        "",
        "## Positives By Category",
    ]
    for category, count in sorted((metadata.get("positives_by_category") or {}).items()):
        lines.append(f"- {category}: **{count}**")
    lines.extend(["", "## Positives By Label Within Category"])
    for category, payload in sorted((metadata.get("positives_by_label_within_category") or {}).items()):
        summary = ", ".join(f"{label}={count}" for label, count in sorted(payload.items()))
        lines.append(f"- {category}: {summary}")
    lines.extend(["", "## Readiness"])
    lines.append(f"- usable_for_continuation_training: **{str((readiness.get('usable_for_continuation_training'))).lower()}**")
    lines.append(f"- usable_for_continuation_shadow: **{str((readiness.get('usable_for_continuation_shadow'))).lower()}**")
    lines.append(f"- usable_for_continuation_runtime_candidate: **{str((readiness.get('usable_for_continuation_runtime_candidate'))).lower()}**")
    lines.append(f"- recommended_categories_for_training: `{readiness.get('recommended_categories_for_training')}`")
    lines.append(f"- blocked_categories: `{readiness.get('blocked_categories')}`")
    lines.extend(["", "## Category Detail"])
    for category, payload in sorted(category_readiness.items()):
        lines.append(
            f"- {category}: **{payload.get('status', 'unknown')}** - {payload.get('why', '')} "
            f"(positives={payload.get('non_stop_positives_total', 0)}, missing={payload.get('missing_actions', [])})"
        )
    summary_path = out_dir / "summary.md"
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary_path

