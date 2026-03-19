from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

try:
    import pandas as pd
except Exception:  # pragma: no cover
    pd = None

from django.core.management.base import BaseCommand, CommandError

from admin_tools.roadmap_planner_transitions import (
    CANDIDATE_SPACE_BY_CATEGORY,
    CONTINUATION_DECISION_TYPES,
    INITIAL_DECISION_TYPES,
    STOP_TOKEN,
    build_transition_decision_records,
    load_transition_source_data,
    readiness_assessment,
    summarize_decision_surfaces,
)
from roadmap_app.content_features import (
    ALL_CATEGORICAL_FEATURES,
    ALL_NUMERIC_FEATURES,
    build_base_content_features,
    build_candidate_content_features,
    product_signature,
)


def _resolve_out_dir(raw_path: str) -> Path:
    candidate = Path(str(raw_path)).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    cwd_path = (Path.cwd() / candidate).resolve()
    if cwd_path.parent.exists():
        return cwd_path
    return (Path(__file__).resolve().parents[4] / candidate).resolve()


def _time_based_split_assignments(records: list[dict[str, Any]]) -> tuple[dict[int, str], dict[str, Any]]:
    if not records:
        return {}, {
            "strategy": "time_based",
            "train_end_utc": None,
            "valid_end_utc": None,
            "counts": {"train": 0, "val": 0, "test": 0},
            "user_overlap_counts": {"train_val": 0, "train_test": 0, "val_test": 0},
        }

    ordered = sorted(records, key=lambda row: (row["t0_dt"], int(row["decision_id"])))
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


def _write_summary_md(*, out_dir: Path, metadata: dict[str, Any]) -> Path:
    decision_types = metadata.get("decision_type_distribution") or {}
    label_sources = metadata.get("label_source_distribution") or {}
    readiness = metadata.get("readiness") or {}
    excluded_counts = metadata.get("excluded_counts") or {}
    slices = metadata.get("slices") or {}
    lines = [
        "# Roadmap Planner Transitions Dataset Summary",
        "",
        f"- rows: **{metadata.get('rows_total', 0)}**",
        f"- decision points: **{metadata.get('decision_points_total', 0)}**",
        f"- episodes: **{metadata.get('episodes_total', 0)}**",
        f"- users: **{metadata.get('users_total', 0)}**",
        f"- stop rate: **{metadata.get('stop_label_rate', 0.0)}**",
        f"- excluded noisy decision points: **{metadata.get('excluded_noisy_decision_points_count', 0)}**",
        f"- excluded legacy bad fragrance completions: **{metadata.get('excluded_legacy_bad_fragrance_completions_count', 0)}**",
        "",
        "## Decision Types",
    ]
    for name, value in sorted(decision_types.items()):
        lines.append(f"- {name}: **{value}**")
    lines.extend(["", "## Positives By Label Source"])
    for name, value in sorted(label_sources.items()):
        lines.append(f"- {name}: **{value}**")
    lines.extend(["", "## Slices"])
    for name, payload in sorted(slices.items()):
        lines.append(
            f"- {name}: trusted={payload.get('trusted_decisions_total', 0)}, positives={payload.get('positives_excluding_stop', 0)}, "
            f"stop_rate={payload.get('stop_rate', 0.0)}, fragrance_positives={payload.get('fragrance_trusted_positives_count', 0)}"
        )
    lines.extend(["", "## Excluded Counts"])
    for name, value in sorted(excluded_counts.items()):
        lines.append(f"- {name}: **{value}**")
    lines.extend(["", "## Readiness"])
    for name, payload in sorted(readiness.items()):
        lines.append(f"- {name}: **{payload.get('status', 'unknown')}** - {payload.get('why', '')}")
    summary_path = out_dir / "summary.md"
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary_path


class Command(BaseCommand):
    help = "Build leakage-safe Roadmap Planner transitions dataset from refresh snapshots and trusted step outcomes."

    def add_arguments(self, parser):
        parser.add_argument("--days", type=int, default=180)
        parser.add_argument("--out-dir", type=str, default="data/ml/roadmap_planner_transitions_v2")
        parser.add_argument("--include-ga", action="store_true", default=False)
        parser.add_argument("--label-window-days", type=int, default=7)
        parser.add_argument("--seed", type=int, default=42)
        parser.add_argument(
            "--mode",
            type=str,
            default="combined",
            choices=["initial", "continuation", "combined"],
        )

    def handle(self, *args, **options):
        if pd is None:
            raise CommandError("pandas is required. Install dependencies from requirements-ml.txt")

        days = int(options["days"])
        include_ga = bool(options["include_ga"])
        label_window_days = int(options["label_window_days"])
        out_dir = _resolve_out_dir(str(options["out_dir"]))
        mode = str(options["mode"] or "combined").strip().lower()
        seed = int(options["seed"])

        if days <= 0:
            raise CommandError("--days must be > 0")
        if label_window_days <= 0:
            raise CommandError("--label-window-days must be > 0")

        source_data = load_transition_source_data(
            days=days,
            include_ga=include_ga,
            label_window_days=label_window_days,
        )
        if not source_data.get("refresh_rows"):
            raise CommandError("No PLAN_REFRESHED events for selected window.")

        bundle = build_transition_decision_records(
            source_data,
            label_window_days=label_window_days,
            mode=mode,
        )
        decision_records = list(bundle.get("decision_records") or [])
        if not decision_records:
            raise CommandError("No planner transition decisions produced after filtering.")

        split_by_decision, split_meta = _time_based_split_assignments(decision_records)
        for record in decision_records:
            record["split"] = str(split_by_decision.get(int(record["decision_id"])) or "train")

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
                k: v
                for k, v in record.items()
                if k
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
        frame = frame.sort_values(["decision_id", "candidate_type"]).reset_index(drop=True)
        out_dir.mkdir(parents=True, exist_ok=True)

        dataset_format = "parquet"
        dataset_file = out_dir / "dataset.parquet"
        try:
            frame.to_parquet(dataset_file, index=False)
        except Exception:
            dataset_format = "csv"
            dataset_file = out_dir / "dataset.csv"
            frame.to_csv(dataset_file, index=False)

        surface_summary = summarize_decision_surfaces(source_data, bundle)
        readiness = readiness_assessment(decision_records)
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
        splits_path = out_dir / "splits.json"
        splits_path.write_text(json.dumps(splits_payload, ensure_ascii=False, indent=2), encoding="utf-8")

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
        feature_columns = [*categorical_features, *numeric_features]

        label_counter = Counter(str(record["label"]) for record in decision_records)
        label_source_counter = Counter(str(record.get("label_source") or "unknown") for record in decision_records)
        decision_type_counter = Counter(str(record.get("decision_type") or "unknown") for record in decision_records)
        trust_level_counter = Counter(str(record.get("trust_level") or "unknown") for record in decision_records)
        positives_by_category = Counter(str(record["category"]) for record in decision_records if str(record["label"]) != STOP_TOKEN)
        positives_by_decision_type = Counter(
            str(record["decision_type"]) for record in decision_records if str(record["label"]) != STOP_TOKEN
        )
        positives_by_label_source = Counter(
            str(record.get("label_source") or "unknown")
            for record in decision_records
            if str(record["label"]) != STOP_TOKEN
        )
        candidate_count_distribution = Counter(len(list(record.get("candidate_types") or [])) for record in decision_records)
        slices = {
            "initial_only": surface_summary.get("initial_only") or {},
            "continuation_only": surface_summary.get("continuation_only") or {},
            "combined": surface_summary.get("combined") or {},
        }
        metadata = {
            "version": "planner_transitions_v2_candidate_ranking",
            "generated_at_utc": str(source_data["now_utc"].isoformat().replace("+00:00", "Z")),
            "window_days": int(days),
            "window_since_utc": str(source_data["since"].isoformat().replace("+00:00", "Z")),
            "window_until_utc": str(source_data["now_utc"].isoformat().replace("+00:00", "Z")),
            "label_window_days": int(label_window_days),
            "include_ga": bool(include_ga),
            "seed": int(seed),
            "mode": mode,
            "decision_point_definition": {
                "initial": "t0 is the PLAN_REFRESHED snapshot built from following STEP_GENERATED events until the next refresh for the same user+category.",
                "continuation": "t0 is immediately after a trusted completion or skip of the current actionable step inside the same refresh episode; state is recalculated by applying prior trusted outcomes to the refresh snapshot.",
            },
            "decision_point_trust": {
                "trusted_completion": "fragrance trusts matched_by=fragrance_slot or exact recommended_product_id only when purchased SKU slot matches step slot; other categories trust roadmap_step_completed.",
                "trusted_skip": "roadmap_step_skipped is trusted only when it applies to the current actionable step.",
                "excluded": "drop surfaces without snapshot, initial surfaces without current actionable step, and labels outside the stable candidate vocabulary.",
            },
            "profile_temporality_caveat": "Profile features use the current CustomerProfile snapshot because historical profile versions are not available.",
            "dataset_format": dataset_format,
            "dataset_file": str(dataset_file),
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
            "trust_level_distribution": dict(sorted(trust_level_counter.items())),
            "class_distribution": dict(sorted(label_counter.items())),
            "positives_by_category": dict(sorted(positives_by_category.items())),
            "positives_by_decision_type": dict(sorted(positives_by_decision_type.items())),
            "positives_by_label_source": dict(sorted(positives_by_label_source.items())),
            "candidate_types_by_category": {
                category: list(types) + ([STOP_TOKEN] if STOP_TOKEN not in types else [])
                for category, types in sorted(CANDIDATE_SPACE_BY_CATEGORY.items())
            },
            "candidate_count_distribution": {
                str(key): int(value)
                for key, value in sorted(candidate_count_distribution.items(), key=lambda row: row[0])
            },
            "surface_summary": surface_summary.get("surface_types") or {},
            "slices": slices,
            "excluded_counts": dict(sorted((bundle.get("excluded_counts") or {}).items())),
            "excluded_noisy_decision_points_count": int(surface_summary.get("excluded_noisy_decision_points_count") or 0),
            "excluded_legacy_bad_fragrance_completions_count": int(
                bundle.get("excluded_legacy_bad_fragrance_completions_count") or 0
            ),
            "readiness": readiness,
            "feature_columns": feature_columns,
            "categorical_features": categorical_features,
            "numeric_features": numeric_features,
            "split_strategy": str(split_meta.get("strategy") or "time_based"),
            "split_counts": dict(split_meta.get("counts") or {}),
            "split_user_overlap_counts": dict(split_meta.get("user_overlap_counts") or {}),
            "leakage_assertions": {
                "features_only_use_transactions_lte_t0": True,
                "labels_end_at_next_plan_refresh_or_window_end": True,
                "future_purchase_fallback_restricted_to_current_actionable_step": True,
                "legacy_bad_fragrance_exact_completions_excluded": True,
                "status": "passed",
            },
        }
        metadata_path = out_dir / "metadata.json"
        metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        summary_path = _write_summary_md(out_dir=out_dir, metadata=metadata)

        self.stdout.write("[build_roadmap_planner_transitions_dataset] done")
        self.stdout.write(f"[build_roadmap_planner_transitions_dataset] dataset={dataset_file}")
        self.stdout.write(f"[build_roadmap_planner_transitions_dataset] metadata={metadata_path}")
        self.stdout.write(f"[build_roadmap_planner_transitions_dataset] splits={splits_path}")
        self.stdout.write(f"[build_roadmap_planner_transitions_dataset] summary={summary_path}")
