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

from admin_tools.roadmap_teacher import (
    CANDIDATE_SPACE_BY_CATEGORY,
    STOP_TOKEN,
    TARGET_CATEGORIES,
    _candidate_types,
    _max_target_steps,
    build_teacher_examples,
    load_teacher_source_data,
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


def _write_summary_md(*, out_dir: Path, metadata: dict[str, Any]) -> Path:
    readiness = metadata.get("readiness") or {}
    lines = [
        "# Roadmap Teacher Dataset Summary",
        "",
        f"- planning examples: **{metadata.get('planning_examples_total', 0)}**",
        f"- stepwise rows: **{metadata.get('stepwise_rows_total', 0)}**",
        f"- users: **{metadata.get('users_total', 0)}**",
        f"- meaningful_non_trivial_length_share: **{metadata.get('meaningful_non_trivial_length_share', 0.0)}**",
        "",
        "## Examples By Category",
    ]
    for name, value in sorted((metadata.get("planning_examples_by_category") or {}).items()):
        lines.append(f"- {name}: **{value}**")
    lines.extend(["", "## Anchor Types"])
    for name, value in sorted((metadata.get("first_anchor_type_distribution") or {}).items()):
        lines.append(f"- {name}: **{value}**")
    lines.extend(["", "## Target Lengths"])
    for name, value in sorted((metadata.get("target_length_distribution") or {}).items(), key=lambda row: int(row[0])):
        lines.append(f"- {name}: **{value}**")
    lines.extend(["", "## Readiness"])
    for name, payload in sorted(readiness.items()):
        lines.append(f"- {name}: **{payload.get('status', 'unknown')}** - {payload.get('why', '')}")
    summary_path = out_dir / "summary.md"
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary_path


def _readiness(metadata: dict[str, Any]) -> dict[str, dict[str, str]]:
    by_category = metadata.get("planning_examples_by_category") or {}
    users = int(metadata.get("users_total") or 0)
    fragrance = int(by_category.get("fragrance", 0))
    haircare = int(by_category.get("haircare", 0))
    multi_categories = len([name for name, count in by_category.items() if int(count) > 0])

    def _decision(status: str, why: str) -> dict[str, str]:
        return {"status": status, "why": why}

    if haircare >= 100 and users >= 50:
        haircare_only = _decision("yes", "Haircare teacher anchors are large enough for an initial baseline.")
    elif haircare >= 40:
        haircare_only = _decision("borderline", "Haircare teacher anchors exist, but sample size is still moderate.")
    else:
        haircare_only = _decision("no", "Too few haircare teacher examples.")

    total = int(metadata.get("planning_examples_total") or 0)
    if total >= 400 and multi_categories >= 3:
        multi_category = _decision("yes", "Teacher dataset covers multiple categories at usable scale.")
    elif total >= 150 and multi_categories >= 2:
        multi_category = _decision("borderline", "Some cross-category coverage exists, but overall scale is limited.")
    else:
        multi_category = _decision("no", "Teacher dataset is too small or too narrow across categories.")

    if fragrance >= 50:
        fragrance_ready = _decision("yes", "Fragrance slot-level teacher examples are sufficient.")
    elif fragrance >= 15:
        fragrance_ready = _decision("borderline", "Fragrance slot-level teacher examples exist, but coverage is thin.")
    else:
        fragrance_ready = _decision("no", "Too few fragrance teacher examples for honest inclusion.")

    return {
        "haircare_only_initial_planner": haircare_only,
        "multi_category_initial_planner": multi_category,
        "fragrance_included_initial_planner": fragrance_ready,
    }


class Command(BaseCommand):
    help = "Build teacher-planner dataset for initial roadmap generation from anchor purchases and business rules."

    def add_arguments(self, parser):
        parser.add_argument("--days", type=int, default=3650)
        parser.add_argument("--out-dir", type=str, default="data/ml/roadmap_teacher_v1")
        parser.add_argument("--include-ga", action="store_true", default=False)
        parser.add_argument("--seed", type=int, default=42)

    def handle(self, *args, **options):
        if pd is None:
            raise CommandError("pandas is required. Install dependencies from requirements-ml.txt")

        days = int(options["days"])
        include_ga = bool(options["include_ga"])
        seed = int(options["seed"])
        out_dir = _resolve_out_dir(str(options["out_dir"]))

        if days <= 0:
            raise CommandError("--days must be > 0")

        source_data = load_teacher_source_data(days=days, include_ga=include_ga)
        bundle = build_teacher_examples(source_data, seed=seed)
        examples = list(bundle.get("examples") or [])
        if not examples:
            raise CommandError("No teacher planning examples produced.")

        candidate_catalog_summaries = dict(source_data.get("candidate_catalog_summaries") or {})
        sequence_rows: list[dict[str, Any]] = []
        stepwise_rows: list[dict[str, Any]] = []

        for example in examples:
            category = str(example["category"])
            sequence = json.loads(str(example.get("target_sequence_json") or "[]"))
            profile_sig = dict(example.get("profile_signature") or {})
            seed_product = dict(example.get("seed_product") or {})
            seed_sig = product_signature(seed_product)
            base_content = build_base_content_features(profile_sig, seed_sig)
            max_steps = _max_target_steps(category)

            sequence_row = {
                k: v
                for k, v in example.items()
                if k not in {"profile_signature", "seed_product", "seed_signature"}
            }
            sequence_row.update(base_content)
            for pos in range(1, max_steps + 1):
                sequence_row[f"target_step_{pos}"] = str(sequence[pos - 1]) if pos <= len(sequence) else STOP_TOKEN
            sequence_rows.append(sequence_row)

            for position in range(1, len(sequence) + 2):
                target_token = str(sequence[position - 1]) if position <= len(sequence) else STOP_TOKEN
                prefix = list(sequence[: position - 1])
                prev_step_1 = str(prefix[-1]) if len(prefix) >= 1 else "__none__"
                prev_step_2 = str(prefix[-2]) if len(prefix) >= 2 else "__none__"
                for candidate in _candidate_types(category):
                    stepwise_rows.append(
                        {
                            **{k: v for k, v in sequence_row.items() if not str(k).startswith("target_step_")},
                            "position": int(position),
                            "prefix_length": int(len(prefix)),
                            "prefix_steps_json": json.dumps(prefix, ensure_ascii=False),
                            "prev_step_1": prev_step_1,
                            "prev_step_2": prev_step_2,
                            "candidate_type": str(candidate),
                            "candidate_seen_in_prefix": int(str(candidate) in set(prefix)),
                            "candidate_is_seed_action_token": int(str(candidate) == str(example.get("seed_action_token") or "")),
                            "candidate_is_stop": int(str(candidate) == STOP_TOKEN),
                            "teacher_target_at_position": target_token,
                            "y": int(str(candidate) == target_token),
                            **build_candidate_content_features(
                                candidate_catalog_summaries.get((category, str(candidate))),
                                profile_sig,
                                seed_sig,
                                candidate_type=str(candidate),
                            ),
                        }
                    )

        sequence_df = pd.DataFrame(sequence_rows).sort_values(["planning_id"]).reset_index(drop=True)
        stepwise_df = pd.DataFrame(stepwise_rows).sort_values(["planning_id", "position", "candidate_type"]).reset_index(drop=True)

        out_dir.mkdir(parents=True, exist_ok=True)
        sequence_path = out_dir / "sequence_dataset.parquet"
        stepwise_path = out_dir / "stepwise_dataset.parquet"
        sequence_df.to_parquet(sequence_path, index=False)
        stepwise_df.to_parquet(stepwise_path, index=False)

        split_counts = dict(bundle.get("split_counts") or {})
        split_payload = {
            "strategy": "user_group_hash",
            "seed": int(seed),
            "counts": split_counts,
            "user_overlap_counts": dict(bundle.get("split_user_overlap_counts") or {}),
            "planning_ids": {
                split_name: [
                    int(row["planning_id"])
                    for row in examples
                    if str(row.get("split") or "") == split_name
                ]
                for split_name in ["train", "val", "test"]
            },
        }
        splits_path = out_dir / "splits.json"
        splits_path.write_text(json.dumps(split_payload, ensure_ascii=False, indent=2), encoding="utf-8")

        planning_examples_by_category = Counter(str(row["category"]) for row in examples)
        first_anchor_type_distribution = Counter(str(row.get("seed_action_token") or "__none__") for row in examples)
        target_length_distribution = Counter(int(row.get("target_length") or 0) for row in examples)
        target_step_frequency = Counter()
        fragrance_slot_distribution = Counter()
        for row in examples:
            sequence = json.loads(str(row.get("target_sequence_json") or "[]"))
            target_step_frequency.update(str(token) for token in sequence if str(token))
            if str(row.get("category") or "") == "fragrance":
                fragrance_slot_distribution.update(str(token) for token in sequence if str(token))

        metadata = {
            "version": "roadmap_teacher_v1",
            "generated_at_utc": str(source_data["now_utc"].isoformat().replace("+00:00", "Z")),
            "window_days": int(days),
            "window_since_utc": str(source_data["since"].isoformat().replace("+00:00", "Z")),
            "window_until_utc": str(source_data["now_utc"].isoformat().replace("+00:00", "Z")),
            "include_ga": bool(include_ga),
            "seed": int(seed),
            "planning_anchor_definition": "One planning example per user+category using the earliest TransactionItem in that category within the selected window.",
            "task_definition": {
                "input": "seed purchase plus profile and prior state at planning time",
                "output": "ordered abstract roadmap sequence for the category",
                "sequence_actions": {category: list(tokens) + [STOP_TOKEN] for category, tokens in sorted(CANDIDATE_SPACE_BY_CATEGORY.items())},
                "formats": ["sequence_dataset", "stepwise_ranking_dataset"],
            },
            "teacher_policy_definition": "teacher_policy_v1 builds desired initial roadmaps from category rules, fragrance slot ontology, profile constraints, seed product attributes, and prior state before the anchor purchase.",
            "sequence_dataset_file": str(sequence_path),
            "stepwise_dataset_file": str(stepwise_path),
            "planning_examples_total": int(len(sequence_df)),
            "sequence_rows_total": int(len(sequence_df)),
            "stepwise_rows_total": int(len(stepwise_df)),
            "users_total": int(len({int(row['user_id']) for row in examples})),
            "planning_examples_by_category": dict(sorted(planning_examples_by_category.items())),
            "first_anchor_type_distribution": dict(sorted(first_anchor_type_distribution.items())),
            "target_length_distribution": {
                str(key): int(value)
                for key, value in sorted(target_length_distribution.items(), key=lambda item: item[0])
            },
            "target_step_frequency": dict(sorted(target_step_frequency.items())),
            "fragrance_slot_distribution": dict(sorted(fragrance_slot_distribution.items())),
            "meaningful_non_trivial_length_share": round(
                float(len([row for row in examples if int(row.get("target_length") or 0) >= 3]) / max(1, len(examples))),
                6,
            ),
            "split_strategy": "user_group_hash",
            "split_counts": split_counts,
            "split_user_overlap_counts": dict(bundle.get("split_user_overlap_counts") or {}),
            "edge_exclusions": {
                **dict(source_data.get("excluded_counts") or {}),
                **dict(bundle.get("edge_counts") or {}),
            },
            "feature_columns_sequence": list(sequence_df.columns),
            "feature_columns_stepwise": list(stepwise_df.columns),
            "categorical_features_stepwise": [
                "category",
                "split",
                "seed_product_type",
                "seed_action_token",
                "seed_slot",
                "seed_brand",
                "seed_scent_family",
                "seed_intensity",
                "seed_hair_type",
                "seed_scalp_type",
                "seed_hair_thickness",
                "seed_finish",
                "seed_coverage",
                "seed_undertone",
                "seed_tone_family",
                "seed_area",
                "seed_spf_signal",
                "seed_effect",
                "seed_waterproof",
                "favorite_brand_overall_before_anchor",
                "favorite_brand_in_category_before_anchor",
                "profile_skin_type",
                "profile_budget",
                "profile_hair_type",
                "profile_scalp_type",
                "profile_hair_thickness",
                "profile_makeup_finish_pref_primary",
                "profile_makeup_coverage_pref_primary",
                "profile_makeup_undertone",
                "profile_makeup_tone_family",
                "profile_fragrance_intensity_pref",
                "anchor_product_type",
                "anchor_hair_type",
                "anchor_scalp_type",
                "anchor_hair_thickness",
                "anchor_finish",
                "anchor_coverage",
                "anchor_undertone",
                "anchor_tone_family",
                "anchor_scent_family",
                "anchor_intensity",
                "prev_step_1",
                "prev_step_2",
                "candidate_type",
                *ALL_CATEGORICAL_FEATURES,
            ],
            "numeric_features_stepwise": [
                "seed_price",
                "seed_notes_count",
                "seed_concerns_count",
                "seed_actives_count",
                "seed_supported_skin_types_count",
                "seed_inci_token_count",
                "target_length",
                "teacher_seed_in_target",
                "teacher_seed_target_position",
                "prior_category_purchase_total",
                "prior_category_distinct_token_count",
                "prior_current_owned_token_count",
                "prior_days_since_category_purchase",
                "avg_price_in_category_before_anchor",
                "prior_total_purchases_all",
                "prior_distinct_categories_count",
                "position",
                "prefix_length",
                "candidate_seen_in_prefix",
                "candidate_is_seed_action_token",
                "candidate_is_stop",
                "y",
                *ALL_NUMERIC_FEATURES,
            ],
        }
        metadata["readiness"] = _readiness(metadata)
        metadata_path = out_dir / "metadata.json"
        metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        summary_path = _write_summary_md(out_dir=out_dir, metadata=metadata)

        self.stdout.write("[build_roadmap_teacher_dataset] done")
        self.stdout.write(f"[build_roadmap_teacher_dataset] sequence={sequence_path}")
        self.stdout.write(f"[build_roadmap_teacher_dataset] stepwise={stepwise_path}")
        self.stdout.write(f"[build_roadmap_teacher_dataset] metadata={metadata_path}")
        self.stdout.write(f"[build_roadmap_teacher_dataset] splits={splits_path}")
        self.stdout.write(f"[build_roadmap_teacher_dataset] summary={summary_path}")
