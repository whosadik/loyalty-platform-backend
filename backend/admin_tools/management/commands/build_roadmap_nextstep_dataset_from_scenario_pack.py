from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

try:
    import pandas as pd
except Exception:  # pragma: no cover
    pd = None

from django.conf import settings
from django.core.management.base import BaseCommand, CommandError

from admin_tools.management.commands.replay_roadmap_scenario_pack import _load_pack, _resolve_pack_path, _to_int
from roadmap_app.content_features import (
    ALL_CATEGORICAL_FEATURES,
    ALL_NUMERIC_FEATURES,
    CHAIN_TRANSITION_NUMERIC_FEATURES,
    NEXTSTEP_PLAN_STATE_CATEGORICAL_FEATURES,
    NEXTSTEP_PLAN_STATE_NUMERIC_FEATURES,
)
from roadmap_app.ml_next_step import _build_v4_feature_frame_from_sources, _load_model_for_path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _resolve_output_dir(raw_path: str) -> Path:
    candidate = Path(str(raw_path)).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    cwd_path = (Path.cwd() / candidate).resolve()
    if cwd_path.parent.exists():
        return cwd_path
    return (_repo_root() / candidate).resolve()


def _resolve_template_model_path(raw_path: str) -> str:
    path = str(raw_path or "").strip()
    if path:
        return path
    fallback = str(getattr(settings, "ROADMAP_NEXTSTEP_V4_MODEL_PATH", "") or "").strip()
    if fallback:
        return fallback
    raise CommandError("--template-model-path is required")


def _parse_token_csv(raw: str | None) -> list[str]:
    return [
        str(part).strip().lower()
        for part in str(raw or "").split(",")
        if str(part).strip()
    ]


def _scenario_sample_weight(outcome_tag: str, expected_next: str) -> float:
    base_map = {
        "completed_exact": 1.35,
        "completed_semantic": 1.25,
        "clicked_no_purchase": 1.0,
        "exposed_no_click": 0.95,
        "skipped": 0.9,
    }
    weight = float(base_map.get(str(outcome_tag or "").strip().lower(), 1.0))
    if str(expected_next or "").strip().lower() in {"leave_in", "scalp_serum"}:
        weight *= 1.10
    return round(weight, 6)


def _split_label_group(
    users: list[int],
    *,
    rng: random.Random,
    val_ratio: float,
    test_ratio: float,
) -> tuple[list[int], list[int], list[int]]:
    pool = list(users)
    rng.shuffle(pool)
    n = len(pool)
    if n <= 0:
        return [], [], []
    if n == 1:
        return pool, [], []
    if n == 2:
        return [pool[0]], [pool[1]], []

    n_test = max(1, int(round(float(n) * test_ratio)))
    n_val = max(1, int(round(float(n) * val_ratio)))
    while n_test + n_val >= n:
        if n_test >= n_val and n_test > 1:
            n_test -= 1
        elif n_val > 1:
            n_val -= 1
        else:
            break
    test_users = pool[:n_test]
    val_users = pool[n_test : n_test + n_val]
    train_users = pool[n_test + n_val :]
    if not train_users:
        train_users = [val_users.pop()] if val_users else [test_users.pop()]
    return train_users, val_users, test_users


def _build_split_payload(
    instance_rows: list[dict[str, Any]],
    *,
    seed: int,
    val_ratio: float,
    test_ratio: float,
) -> dict[str, Any]:
    by_label: dict[str, list[int]] = defaultdict(list)
    for row in instance_rows:
        by_label[str(row["expected_next_product_type"])].append(int(row["user_id"]))

    rng = random.Random(int(seed))
    train_users: set[int] = set()
    val_users: set[int] = set()
    test_users: set[int] = set()
    for _, label_users in sorted(by_label.items()):
        label_train, label_val, label_test = _split_label_group(
            sorted(set(label_users)),
            rng=rng,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
        )
        train_users.update(label_train)
        val_users.update(label_val)
        test_users.update(label_test)

    val_users.difference_update(train_users)
    test_users.difference_update(train_users)
    test_users.difference_update(val_users)

    if not train_users or not val_users or not test_users:
        all_users = sorted({int(row["user_id"]) for row in instance_rows})
        rng.shuffle(all_users)
        n = len(all_users)
        if n >= 3:
            n_test = max(1, int(round(float(n) * test_ratio)))
            n_val = max(1, int(round(float(n) * val_ratio)))
            while n_test + n_val >= n:
                if n_test >= n_val and n_test > 1:
                    n_test -= 1
                elif n_val > 1:
                    n_val -= 1
                else:
                    break
            test_users = set(all_users[:n_test])
            val_users = set(all_users[n_test : n_test + n_val])
            train_users = set(all_users[n_test + n_val :])
            if not train_users:
                train_users = {val_users.pop()} if val_users else {test_users.pop()}

    return {
        "seed": int(seed),
        "strategy": "stratified_user_level_by_expected_next",
        "ratios": {"train": round(1.0 - val_ratio - test_ratio, 4), "val": val_ratio, "test": test_ratio},
        "train_user_ids": sorted(int(x) for x in train_users),
        "val_user_ids": sorted(int(x) for x in val_users),
        "test_user_ids": sorted(int(x) for x in test_users),
    }


class Command(BaseCommand):
    help = "Build next-step training dataset from a synthetic roadmap scenario pack."

    def add_arguments(self, parser):
        parser.add_argument("--path", type=str, required=True)
        parser.add_argument("--out-dir", type=str, required=True)
        parser.add_argument(
            "--template-model-path",
            type=str,
            default=str(getattr(settings, "ROADMAP_NEXTSTEP_V4_MODEL_PATH", "") or ""),
        )
        parser.add_argument("--seed", type=int, default=42)
        parser.add_argument("--val-ratio", type=float, default=0.2)
        parser.add_argument("--test-ratio", type=float, default=0.2)
        parser.add_argument(
            "--expected-next-product-types",
            type=str,
            default="",
            help="Optional CSV allowlist of expected_next_product_type values to keep.",
        )

    def handle(self, *args, **options):
        if pd is None:
            raise CommandError("pandas is required to build a scenario-pack dataset")

        pack_path = _resolve_pack_path(str(options.get("path") or ""))
        if not pack_path.exists() or not pack_path.is_dir():
            raise CommandError(f"Scenario pack path not found: {pack_path}")

        out_dir = _resolve_output_dir(str(options.get("out_dir") or ""))
        out_dir.mkdir(parents=True, exist_ok=True)

        seed = int(options.get("seed") or 42)
        val_ratio = float(options.get("val_ratio") or 0.2)
        test_ratio = float(options.get("test_ratio") or 0.2)
        expected_next_allowlist = set(_parse_token_csv(options.get("expected_next_product_types")))
        if val_ratio < 0 or test_ratio < 0 or (val_ratio + test_ratio) >= 1.0:
            raise CommandError("--val-ratio and --test-ratio must be >= 0 and sum to less than 1.0")

        template_model_path = _resolve_template_model_path(str(options.get("template_model_path") or ""))
        template_artifact = _load_model_for_path(template_model_path)
        if not isinstance(template_artifact, dict):
            raise CommandError(f"Template model path does not point to a valid artifact: {template_model_path}")
        if str(template_artifact.get("task") or "").strip() != "roadmap_nextstep_v4_ranking":
            raise CommandError("Template model must be a roadmap_nextstep_v4_ranking artifact")

        pack = _load_pack(pack_path)
        summary = dict(pack["summary"])
        products = list(pack["products"])
        products_by_id = dict(pack["products_by_id"])
        profiles_by_user_id = dict(pack["profiles_by_user_id"])
        items_by_user_id = dict(pack["items_by_user_id"])
        plans_by_id = dict(pack["plans_by_id"])
        scenario_instances = list(summary.get("scenario_instances") or [])
        if not scenario_instances:
            raise CommandError("summary.json does not contain scenario_instances")
        if expected_next_allowlist:
            scenario_instances = [
                row
                for row in scenario_instances
                if str(row.get("expected_next_product_type") or "").strip().lower() in expected_next_allowlist
            ]
            if not scenario_instances:
                raise CommandError(
                    "No scenario_instances matched --expected-next-product-types="
                    f"{sorted(expected_next_allowlist)}"
                )

        rows: list[dict[str, Any]] = []
        instance_rows: list[dict[str, Any]] = []
        feature_columns_ref: list[str] = []
        categorical_ref: list[str] = []
        numeric_ref: list[str] = []
        label_outside_candidate_set = 0
        category_counter: Counter[str] = Counter()
        outcome_counter: Counter[str] = Counter()

        for instance in scenario_instances:
            plan_id = _to_int(instance.get("plan_id"))
            user_id = _to_int(instance.get("user_id"))
            plan = plans_by_id.get(plan_id)
            if not plan:
                continue

            category = str(plan.get("category") or "").strip().lower()
            plan_meta = plan.get("meta") or {}
            context_meta = plan_meta.get("context") if isinstance(plan_meta, dict) else {}
            ml_meta = plan_meta.get("ml") if isinstance(plan_meta, dict) else {}
            t0_raw = str(instance.get("t0_utc") or plan.get("updated_at") or "")
            if not t0_raw:
                continue
            try:
                t0 = pd.Timestamp(t0_raw).to_pydatetime()
            except Exception:
                continue
            if t0.tzinfo is None:
                continue

            context_product_ids = [
                _to_int(raw)
                for raw in (context_meta.get("post_ctx_product_ids") or [])
                if _to_int(raw) > 0
            ]
            context_products = [
                dict(products_by_id[product_id])
                for product_id in context_product_ids
                if product_id in products_by_id
            ]
            planned_target_product_type = str(
                instance.get("planned_target_product_type")
                or (ml_meta.get("planned_target_product_type") if isinstance(ml_meta, dict) else "")
                or ""
            ).strip().lower()
            planned_target_step_index = _to_int(
                instance.get("planned_target_step_index")
                or (ml_meta.get("planned_target_step_index") if isinstance(ml_meta, dict) else 0),
                default=0,
            )
            expected_next = str(instance.get("expected_next_product_type") or "").strip().lower()
            outcome_tag = str(instance.get("outcome_tag") or "").strip().lower()
            if not expected_next:
                continue

            history_items = [
                dict(item)
                for item in (items_by_user_id.get(user_id) or [])
                if item.get("ts") is not None and item["ts"] <= t0
            ]
            profile = profiles_by_user_id.get(user_id) or {}
            frame, feature_columns, categorical_features, numeric_features = _build_v4_feature_frame_from_sources(
                artifact=template_artifact,
                category=category,
                now_utc=t0,
                items=history_items,
                profile=profile,
                context_products=context_products,
                catalog_products=products,
                planned_target_product_type=planned_target_product_type,
                planned_target_step_index=planned_target_step_index,
                candidate_types=None,
            )
            if frame is None or frame.empty:
                continue

            if not feature_columns_ref:
                feature_columns_ref = list(feature_columns)
                categorical_ref = list(categorical_features)
                numeric_ref = list(numeric_features)

            candidate_set = {str(x).strip().lower() for x in frame["candidate_type"].tolist() if str(x).strip()}
            if expected_next not in candidate_set:
                label_outside_candidate_set += 1

            sample_weight = _scenario_sample_weight(outcome_tag, expected_next)
            scenario_key = str(instance.get("scenario_key") or "").strip()
            replica = _to_int(instance.get("replica"), default=0)

            frame = frame.copy()
            frame["episode_id"] = int(plan_id)
            frame["group_id"] = int(plan_id)
            frame["user_id"] = int(user_id)
            frame["category"] = category
            frame["t0_utc"] = pd.Timestamp(t0).isoformat().replace("+00:00", "Z")
            frame["split"] = "__pending__"
            frame["label"] = expected_next
            frame["y"] = (frame["candidate_type"].astype(str).str.strip().str.lower() == expected_next).astype(int)
            frame["sample_weight"] = float(sample_weight)
            frame["scenario_key"] = scenario_key
            frame["replica"] = int(replica)
            frame["outcome_tag"] = outcome_tag
            frame["planned_target_product_type"] = planned_target_product_type or "__none__"
            frame["planned_target_step_index"] = int(planned_target_step_index)
            frame["source_pack_path"] = str(pack_path)
            rows.extend(frame.to_dict(orient="records"))

            instance_rows.append(
                {
                    "user_id": int(user_id),
                    "episode_id": int(plan_id),
                    "expected_next_product_type": expected_next,
                }
            )
            category_counter[category] += 1
            outcome_counter[outcome_tag] += 1

        if not rows:
            raise CommandError("No rows produced from scenario pack")

        df = pd.DataFrame(rows)
        splits_payload = _build_split_payload(
            instance_rows,
            seed=seed,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
        )
        train_users = set(int(x) for x in (splits_payload.get("train_user_ids") or []))
        val_users = set(int(x) for x in (splits_payload.get("val_user_ids") or []))
        test_users = set(int(x) for x in (splits_payload.get("test_user_ids") or []))
        if not train_users or not val_users or not test_users:
            raise CommandError("Scenario split produced an empty train/val/test split")

        df.loc[df["user_id"].isin(train_users), "split"] = "train"
        df.loc[df["user_id"].isin(val_users), "split"] = "val"
        df.loc[df["user_id"].isin(test_users), "split"] = "test"
        unresolved = df[df["split"] == "__pending__"]
        if not unresolved.empty:
            raise CommandError("Some rows did not receive a split assignment")

        candidate_popularity_train: dict[str, dict[str, float]] = {}
        candidate_types_by_category = {
            str(k): [str(x) for x in (v or [])]
            for k, v in (template_artifact.get("candidate_types_by_category") or {}).items()
            if str(k).strip()
        }
        train_pos = df[(df["split"] == "train") & (df["y"] == 1)]
        for category, candidates in candidate_types_by_category.items():
            counter = Counter(
                str(x).strip().lower()
                for x in train_pos.loc[train_pos["category"] == category, "candidate_type"].tolist()
                if str(x).strip()
            )
            total = float(sum(counter.values()) or 1.0)
            candidate_popularity_train[category] = {
                str(candidate).strip().lower(): round(float(counter.get(str(candidate).strip().lower(), 0)) / total, 8)
                for candidate in candidates
            }

        dataset_path = out_dir / "dataset.csv"
        df.to_csv(dataset_path, index=False, encoding="utf-8")

        splits_path = out_dir / "splits.json"
        splits_path.write_text(json.dumps(splits_payload, ensure_ascii=False, indent=2), encoding="utf-8")

        metadata = {
            "version": "scenario_pack_v1",
            "label_protocol_version": "scenario_pack_expected_next_v1",
            "generated_at_utc": pd.Timestamp.utcnow().isoformat().replace("+00:00", "Z"),
            "dataset_format": "csv",
            "dataset_file": str(dataset_path),
            "source_pack_path": str(pack_path),
            "source_scenario_set": str(summary.get("scenario_set") or ""),
            "expected_next_product_type_filter": sorted(expected_next_allowlist),
            "template_model_path": template_model_path,
            "rows_total": int(len(df)),
            "episodes_total": int(df["episode_id"].nunique()),
            "groups_total": int(df["group_id"].nunique()),
            "positive_rows": int(df["y"].sum()),
            "label_outside_candidate_set": int(label_outside_candidate_set),
            "class_distribution": {
                str(k): int(v)
                for k, v in sorted(
                    Counter(str(x).strip().lower() for x in df.loc[df["y"] == 1, "candidate_type"].tolist()).items(),
                    key=lambda kv: (-kv[1], kv[0]),
                )
            },
            "selected_categories": sorted(category_counter.keys()),
            "scenario_count_by_category": {str(k): int(v) for k, v in sorted(category_counter.items())},
            "outcome_tag_distribution": {str(k): int(v) for k, v in sorted(outcome_counter.items())},
            "candidate_types_by_category": candidate_types_by_category,
            "rules_chain_by_category": {
                str(k): [str(x) for x in (v or [])]
                for k, v in (template_artifact.get("rules_chain_by_category") or {}).items()
                if str(k).strip()
            },
            "candidate_popularity_in_train_by_category": candidate_popularity_train,
            "owned_feature_columns": list(template_artifact.get("owned_feature_columns") or []),
            "owned_feature_map": dict(template_artifact.get("owned_feature_map") or {}),
            "feature_columns": list(feature_columns_ref),
            "categorical_features": [col for col in categorical_ref if col in set(feature_columns_ref)],
            "numeric_features": [col for col in numeric_ref if col in set(feature_columns_ref)],
            "sample_weight_policy": {
                "completed_exact": 1.35,
                "completed_semantic": 1.25,
                "clicked_no_purchase": 1.0,
                "exposed_no_click": 0.95,
                "skipped": 0.9,
                "weak_target_multiplier": 1.1,
                "weak_targets": ["leave_in", "scalp_serum"],
            },
            "feature_protocol": {
                "base_categorical": list(ALL_CATEGORICAL_FEATURES),
                "base_numeric": list(ALL_NUMERIC_FEATURES),
                "chain_transition_numeric": list(CHAIN_TRANSITION_NUMERIC_FEATURES),
                "plan_state_categorical": list(NEXTSTEP_PLAN_STATE_CATEGORICAL_FEATURES),
                "plan_state_numeric": list(NEXTSTEP_PLAN_STATE_NUMERIC_FEATURES),
            },
        }
        metadata_path = out_dir / "metadata.json"
        metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

        self.stdout.write("[build_roadmap_nextstep_dataset_from_scenario_pack] done")
        self.stdout.write(f"[build_roadmap_nextstep_dataset_from_scenario_pack] dataset={dataset_path}")
        self.stdout.write(f"[build_roadmap_nextstep_dataset_from_scenario_pack] metadata={metadata_path}")
        self.stdout.write(f"[build_roadmap_nextstep_dataset_from_scenario_pack] splits={splits_path}")
        self.stdout.write(
            "[build_roadmap_nextstep_dataset_from_scenario_pack] "
            f"episodes={metadata['episodes_total']} rows={metadata['rows_total']} positives={metadata['positive_rows']}"
        )
