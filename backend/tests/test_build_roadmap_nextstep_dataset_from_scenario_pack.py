from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory

import joblib
import pandas as pd
from django.core.management import call_command
from django.test import TestCase


class BuildRoadmapNextstepDatasetFromScenarioPackTests(TestCase):
    def test_build_dataset_from_scenario_pack_writes_trainer_compatible_files(self):
        payload = {
            "scenario_set": "haircare_llm_dataset_test",
            "category": "haircare",
            "shared_catalog": [
                {
                    "product_key": "clarity_shampoo",
                    "name": "Clarity Shampoo",
                    "brand": "Scenario Lab",
                    "price": "5200",
                    "currency": "KZT",
                    "category": "haircare",
                    "product_type": "shampoo",
                    "concerns": ["oiliness", "flakes", "itchiness"],
                    "actives": ["salicylic_acid", "zinc_pca"],
                    "flags": [],
                    "attrs": {"hair_type": "wavy", "scalp_type": "oily", "hair_thickness": "medium"},
                    "raw_meta": {"line": "Scalp", "finish": "fresh"},
                    "ingredients_inci": "Aqua, Salicylic Acid, Zinc PCA",
                },
                {
                    "product_key": "soft_conditioner",
                    "name": "Soft Conditioner",
                    "brand": "Scenario Lab",
                    "price": "6100",
                    "currency": "KZT",
                    "category": "haircare",
                    "product_type": "conditioner",
                    "concerns": ["smoothness", "detangling"],
                    "actives": ["panthenol", "squalane"],
                    "flags": [],
                    "attrs": {"hair_type": "wavy", "scalp_type": "normal", "hair_thickness": "medium"},
                    "raw_meta": {"line": "Length", "finish": "soft"},
                    "ingredients_inci": "Aqua, Panthenol, Squalane",
                },
                {
                    "product_key": "protein_mask",
                    "name": "Protein Mask",
                    "brand": "Scenario Lab",
                    "price": "6900",
                    "currency": "KZT",
                    "category": "haircare",
                    "product_type": "hair_mask",
                    "concerns": ["repair", "damage"],
                    "actives": ["amino_acids", "panthenol"],
                    "flags": [],
                    "attrs": {"hair_type": "wavy", "scalp_type": "normal", "hair_thickness": "fine"},
                    "raw_meta": {"line": "Repair", "finish": "airy"},
                    "ingredients_inci": "Aqua, Amino Acids, Panthenol",
                },
                {
                    "product_key": "light_leavein",
                    "name": "Light Leave In",
                    "brand": "Scenario Lab",
                    "price": "6600",
                    "currency": "KZT",
                    "category": "haircare",
                    "product_type": "leave_in",
                    "concerns": ["lightweight_care", "frizz_control", "detangling"],
                    "actives": ["aloe", "panthenol"],
                    "flags": [],
                    "attrs": {"hair_type": "wavy", "scalp_type": "normal", "hair_thickness": "fine"},
                    "raw_meta": {"line": "Air", "finish": "airy"},
                    "ingredients_inci": "Aqua, Aloe, Panthenol",
                },
                {
                    "product_key": "scalp_serum",
                    "name": "Scalp Serum",
                    "brand": "Scenario Lab",
                    "price": "7000",
                    "currency": "KZT",
                    "category": "haircare",
                    "product_type": "scalp_serum",
                    "concerns": ["oiliness", "flakes", "itchiness", "build_up"],
                    "actives": ["salicylic_acid", "niacinamide", "zinc_pca"],
                    "flags": [],
                    "attrs": {"hair_type": "wavy", "scalp_type": "oily", "hair_thickness": "medium"},
                    "raw_meta": {"line": "Scalp", "finish": "fresh"},
                    "ingredients_inci": "Aqua, Salicylic Acid, Niacinamide, Zinc PCA",
                },
                {
                    "product_key": "scalp_serum_alt",
                    "name": "Scalp Serum Alt",
                    "brand": "Scenario Lab",
                    "price": "7200",
                    "currency": "KZT",
                    "category": "haircare",
                    "product_type": "scalp_serum",
                    "concerns": ["oiliness", "flakes", "itchiness", "build_up"],
                    "actives": ["salicylic_acid", "tea_tree", "niacinamide"],
                    "flags": [],
                    "attrs": {"hair_type": "wavy", "scalp_type": "oily", "hair_thickness": "medium"},
                    "raw_meta": {"line": "Scalp Alt", "finish": "fresh"},
                    "ingredients_inci": "Aqua, Salicylic Acid, Tea Tree, Niacinamide",
                },
            ],
            "scenarios": [
                {
                    "slug": "scalp_case",
                    "segment": "scalp_serum_beats_conditioner",
                    "expected_next_product_type": "scalp_serum",
                    "outcome_tag": "completed_semantic",
                    "profile": {
                        "skin_type": "oily",
                        "goals": ["oiliness", "scalp_balance", "scalp_health"],
                        "avoid_flags": [],
                        "budget": "medium",
                        "hair_profile": {
                            "hair_type": "wavy",
                            "scalp_type": "oily",
                            "hair_thickness": "medium",
                            "concerns": ["oiliness", "flakes", "itchiness"],
                        },
                    },
                    "transactions": [
                        {"offset_days": -24, "channel": "online", "items": [{"product_key": "clarity_shampoo", "quantity": 1}]},
                        {"offset_days": -5, "channel": "offline", "items": [{"product_key": "clarity_shampoo", "quantity": 1}]},
                    ],
                    "steps": [
                        {"step_index": 1, "product_type": "shampoo", "status": "completed", "recommended_product_key": "clarity_shampoo", "cadence": "weekly", "score": 0.7, "confidence": 0.75, "why": ["wash_done"]},
                        {"step_index": 2, "product_type": "conditioner", "status": "missing", "recommended_product_key": "soft_conditioner", "cadence": "weekly", "score": 0.4, "confidence": 0.35, "why": ["secondary_step"]},
                        {"step_index": 3, "product_type": "scalp_serum", "status": "recommended", "recommended_product_key": "scalp_serum", "cadence": "daily", "score": 0.92, "confidence": 0.9, "why": ["scalp_signal"]},
                    ],
                    "events": [
                        {"event_type": "roadmap_plan_refreshed", "offset_hours": 0, "step_index": None, "context": {}},
                        {"event_type": "roadmap_step_exposed", "offset_hours": 2, "step_index": 3, "context": {"category": "haircare", "product_type": "scalp_serum", "recommended_product_key": "scalp_serum"}},
                        {"event_type": "roadmap_step_completed", "offset_hours": 18, "step_index": 3, "context": {"category": "haircare", "product_type": "scalp_serum", "matched_by": "semantic_content_match", "match_meta": {"recommended_product_key": "scalp_serum", "purchased_product_key": "scalp_serum_alt", "semantic_score": 0.94}}},
                    ],
                },
                {
                    "slug": "conditioner_case",
                    "segment": "conditioner_beats_serum",
                    "expected_next_product_type": "conditioner",
                    "outcome_tag": "completed_exact",
                    "profile": {
                        "skin_type": "dry",
                        "goals": ["smoothness", "detangling"],
                        "avoid_flags": [],
                        "budget": "low",
                        "hair_profile": {
                            "hair_type": "wavy",
                            "scalp_type": "normal",
                            "hair_thickness": "medium",
                            "concerns": ["smoothness", "detangling"],
                        },
                    },
                    "transactions": [
                        {"offset_days": -20, "channel": "online", "items": [{"product_key": "clarity_shampoo", "quantity": 1}]},
                    ],
                    "steps": [
                        {"step_index": 1, "product_type": "shampoo", "status": "completed", "recommended_product_key": "clarity_shampoo", "cadence": "weekly", "score": 0.7, "confidence": 0.75, "why": ["wash_done"]},
                        {"step_index": 2, "product_type": "conditioner", "status": "recommended", "recommended_product_key": "soft_conditioner", "cadence": "weekly", "score": 0.88, "confidence": 0.84, "why": ["length_signal"]},
                        {"step_index": 3, "product_type": "scalp_serum", "status": "missing", "recommended_product_key": "scalp_serum", "cadence": "daily", "score": 0.31, "confidence": 0.28, "why": ["no_scalp_signal"]},
                    ],
                    "events": [
                        {"event_type": "roadmap_plan_refreshed", "offset_hours": 0, "step_index": None, "context": {}},
                        {"event_type": "roadmap_step_exposed", "offset_hours": 2, "step_index": 2, "context": {"category": "haircare", "product_type": "conditioner", "recommended_product_key": "soft_conditioner"}},
                        {"event_type": "roadmap_step_completed", "offset_hours": 16, "step_index": 2, "context": {"category": "haircare", "product_type": "conditioner", "matched_by": "recommended_product_id", "match_meta": {"recommended_product_key": "soft_conditioner", "purchased_product_key": "soft_conditioner"}}},
                    ],
                },
                {
                    "slug": "mask_case",
                    "segment": "mask_after_conditioner",
                    "expected_next_product_type": "hair_mask",
                    "outcome_tag": "clicked_no_purchase",
                    "profile": {
                        "skin_type": "normal",
                        "goals": ["repair", "volume"],
                        "avoid_flags": [],
                        "budget": "medium",
                        "hair_profile": {
                            "hair_type": "wavy",
                            "scalp_type": "normal",
                            "hair_thickness": "fine",
                            "concerns": ["damage", "flatness"],
                        },
                    },
                    "transactions": [
                        {"offset_days": -28, "channel": "online", "items": [{"product_key": "clarity_shampoo", "quantity": 1}]},
                        {"offset_days": -8, "channel": "offline", "items": [{"product_key": "soft_conditioner", "quantity": 1}]},
                    ],
                    "steps": [
                        {"step_index": 1, "product_type": "shampoo", "status": "completed", "recommended_product_key": "clarity_shampoo", "cadence": "weekly", "score": 0.7, "confidence": 0.75, "why": ["wash_done"]},
                        {"step_index": 2, "product_type": "conditioner", "status": "completed", "recommended_product_key": "soft_conditioner", "cadence": "weekly", "score": 0.76, "confidence": 0.8, "why": ["slip_done"]},
                        {"step_index": 3, "product_type": "hair_mask", "status": "recommended", "recommended_product_key": "protein_mask", "cadence": "weekly", "score": 0.9, "confidence": 0.88, "why": ["repair_gap"]},
                    ],
                    "events": [
                        {"event_type": "roadmap_plan_refreshed", "offset_hours": 0, "step_index": None, "context": {}},
                        {"event_type": "roadmap_step_exposed", "offset_hours": 2, "step_index": 3, "context": {"category": "haircare", "product_type": "hair_mask", "recommended_product_key": "protein_mask"}},
                        {"event_type": "roadmap_step_clicked", "offset_hours": 6, "step_index": 3, "context": {"category": "haircare", "product_type": "hair_mask", "recommended_product_key": "protein_mask"}},
                    ],
                },
            ],
        }

        artifact = {
            "task": "roadmap_nextstep_v4_ranking",
            "model": {"stub": True},
            "feature_columns": [
                "category",
                "candidate_type",
                "last1_product_type",
                "last1_category",
                "planned_target_product_type",
                "profile_skin_type",
                "profile_budget",
                "profile_hair_type",
                "profile_scalp_type",
                "profile_hair_thickness",
                "anchor_product_type",
                "anchor_hair_type",
                "anchor_scalp_type",
                "anchor_hair_thickness",
                "candidate_dominant_hair_type",
                "candidate_dominant_scalp_type",
                "candidate_dominant_hair_thickness",
                "month_of_year",
                "day_of_week",
                "days_since_last_purchase_in_category",
                "tx_count_90d_category",
                "tx_amount_90d_category",
                "profile_goals_count",
                "profile_avoid_flags_count",
                "profile_hair_concerns_count",
                "profile_scalp_objective_count",
                "profile_has_scalp_objective",
                "anchor_concerns_count",
                "anchor_actives_count",
                "anchor_scalp_concern_count",
                "anchor_scalp_active_count",
                "anchor_has_scalp_focus",
                "anchor_supported_skin_types_count",
                "anchor_inci_token_count",
                "candidate_position_in_chain",
                "candidate_popularity_in_train",
                "candidate_matches_last1",
                "candidate_matches_last3_any",
                "candidate_seen_count_last5",
                "candidate_owned_count_in_category",
                "candidate_seen_90d_count_in_category",
                "candidate_days_since_last_seen_in_category",
                "candidate_profile_scalp_objective_match_rate",
                "candidate_is_scalp_specialty",
                "planned_target_step_index",
                "candidate_matches_planned_target",
                "candidate_distance_from_planned_target",
                "candidate_abs_distance_from_planned_target",
                "candidate_is_after_planned_target",
                "candidate_is_before_planned_target",
            ],
            "categorical_features": [
                "category",
                "candidate_type",
                "last1_product_type",
                "last1_category",
                "planned_target_product_type",
                "profile_skin_type",
                "profile_budget",
                "profile_hair_type",
                "profile_scalp_type",
                "profile_hair_thickness",
                "anchor_product_type",
                "anchor_hair_type",
                "anchor_scalp_type",
                "anchor_hair_thickness",
                "candidate_dominant_hair_type",
                "candidate_dominant_scalp_type",
                "candidate_dominant_hair_thickness",
            ],
            "numeric_features": [
                "month_of_year",
                "day_of_week",
                "days_since_last_purchase_in_category",
                "tx_count_90d_category",
                "tx_amount_90d_category",
                "profile_goals_count",
                "profile_avoid_flags_count",
                "profile_hair_concerns_count",
                "profile_scalp_objective_count",
                "profile_has_scalp_objective",
                "anchor_concerns_count",
                "anchor_actives_count",
                "anchor_scalp_concern_count",
                "anchor_scalp_active_count",
                "anchor_has_scalp_focus",
                "anchor_supported_skin_types_count",
                "anchor_inci_token_count",
                "candidate_position_in_chain",
                "candidate_popularity_in_train",
                "candidate_matches_last1",
                "candidate_matches_last3_any",
                "candidate_seen_count_last5",
                "candidate_owned_count_in_category",
                "candidate_seen_90d_count_in_category",
                "candidate_days_since_last_seen_in_category",
                "candidate_profile_scalp_objective_match_rate",
                "candidate_is_scalp_specialty",
                "planned_target_step_index",
                "candidate_matches_planned_target",
                "candidate_distance_from_planned_target",
                "candidate_abs_distance_from_planned_target",
                "candidate_is_after_planned_target",
                "candidate_is_before_planned_target",
            ],
            "candidate_types_by_category": {
                "haircare": ["shampoo", "conditioner", "hair_mask", "scalp_serum", "leave_in"]
            },
            "rules_chain_by_category": {
                "haircare": ["shampoo", "conditioner", "hair_mask", "hair_oil", "scalp_serum", "leave_in"]
            },
            "candidate_popularity_in_train_by_category": {
                "haircare": {
                    "shampoo": 0.2,
                    "conditioner": 0.2,
                    "hair_mask": 0.2,
                    "scalp_serum": 0.2,
                    "leave_in": 0.2,
                }
            },
            "owned_feature_columns": [],
            "owned_feature_map": {},
        }

        with TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            input_path = tmp_path / "llm.json"
            pack_dir = tmp_path / "scenario_pack"
            dataset_dir = tmp_path / "dataset"
            model_path = tmp_path / "template.pkl"
            input_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            joblib.dump(artifact, model_path)

            call_command(
                "build_roadmap_scenario_pack_from_llm_json",
                input=str(input_path),
                out_dir=str(pack_dir),
                replicas=1,
                days_ago_start=60,
                id_base=985000,
            )

            call_command(
                "build_roadmap_nextstep_dataset_from_scenario_pack",
                path=str(pack_dir),
                out_dir=str(dataset_dir),
                template_model_path=str(model_path),
                seed=7,
                val_ratio=0.2,
                test_ratio=0.2,
            )

            dataset_path = dataset_dir / "dataset.csv"
            metadata_path = dataset_dir / "metadata.json"
            splits_path = dataset_dir / "splits.json"
            self.assertTrue(dataset_path.exists())
            self.assertTrue(metadata_path.exists())
            self.assertTrue(splits_path.exists())

            df = pd.read_csv(dataset_path)
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            splits = json.loads(splits_path.read_text(encoding="utf-8"))

            self.assertEqual(str(metadata.get("label_protocol_version")), "scenario_pack_expected_next_v1")
            self.assertEqual(int(metadata.get("episodes_total", 0)), 3)
            self.assertEqual(sorted(df["split"].unique().tolist()), ["test", "train", "val"])
            self.assertTrue(bool(splits.get("train_user_ids")))
            self.assertTrue(bool(splits.get("val_user_ids")))
            self.assertTrue(bool(splits.get("test_user_ids")))

            scalp_rows = df[df["scenario_key"] == "scalp_case"].copy()
            self.assertFalse(scalp_rows.empty)
            scalp_positive = scalp_rows[(scalp_rows["candidate_type"] == "scalp_serum") & (scalp_rows["y"] == 1)]
            scalp_negative = scalp_rows[(scalp_rows["candidate_type"] == "conditioner") & (scalp_rows["y"] == 0)]
            self.assertEqual(len(scalp_positive), 1)
            self.assertEqual(len(scalp_negative), 1)
            self.assertEqual(str(scalp_positive.iloc[0]["planned_target_product_type"]), "scalp_serum")
            self.assertGreater(float(scalp_positive.iloc[0]["sample_weight"]), 1.0)

    def test_scenario_style_training_keeps_canonical_report_untouched(self):
        rows = [
            {
                "user_id": 101,
                "episode_id": 1001,
                "group_id": 1001,
                "category": "haircare",
                "candidate_type": "leave_in",
                "last1_product_type": "hair_mask",
                "last1_category": "haircare",
                "month_of_year": 3,
                "day_of_week": 4,
                "candidate_is_fragrance_slot": 0,
                "candidate_position_in_chain": 5,
                "candidate_popularity_in_train": 0.2,
                "sample_weight": 1.25,
                "y": 1,
            },
            {
                "user_id": 101,
                "episode_id": 1001,
                "group_id": 1001,
                "category": "haircare",
                "candidate_type": "hair_oil",
                "last1_product_type": "hair_mask",
                "last1_category": "haircare",
                "month_of_year": 3,
                "day_of_week": 4,
                "candidate_is_fragrance_slot": 0,
                "candidate_position_in_chain": 4,
                "candidate_popularity_in_train": 0.2,
                "sample_weight": 1.0,
                "y": 0,
            },
            {
                "user_id": 102,
                "episode_id": 1002,
                "group_id": 1002,
                "category": "haircare",
                "candidate_type": "scalp_serum",
                "last1_product_type": "shampoo",
                "last1_category": "haircare",
                "month_of_year": 3,
                "day_of_week": 5,
                "candidate_is_fragrance_slot": 0,
                "candidate_position_in_chain": 5,
                "candidate_popularity_in_train": 0.2,
                "sample_weight": 1.25,
                "y": 1,
            },
            {
                "user_id": 102,
                "episode_id": 1002,
                "group_id": 1002,
                "category": "haircare",
                "candidate_type": "conditioner",
                "last1_product_type": "shampoo",
                "last1_category": "haircare",
                "month_of_year": 3,
                "day_of_week": 5,
                "candidate_is_fragrance_slot": 0,
                "candidate_position_in_chain": 2,
                "candidate_popularity_in_train": 0.2,
                "sample_weight": 1.0,
                "y": 0,
            },
            {
                "user_id": 103,
                "episode_id": 1003,
                "group_id": 1003,
                "category": "haircare",
                "candidate_type": "hair_mask",
                "last1_product_type": "conditioner",
                "last1_category": "haircare",
                "month_of_year": 3,
                "day_of_week": 6,
                "candidate_is_fragrance_slot": 0,
                "candidate_position_in_chain": 3,
                "candidate_popularity_in_train": 0.2,
                "sample_weight": 1.1,
                "y": 1,
            },
            {
                "user_id": 103,
                "episode_id": 1003,
                "group_id": 1003,
                "category": "haircare",
                "candidate_type": "leave_in",
                "last1_product_type": "conditioner",
                "last1_category": "haircare",
                "month_of_year": 3,
                "day_of_week": 6,
                "candidate_is_fragrance_slot": 0,
                "candidate_position_in_chain": 5,
                "candidate_popularity_in_train": 0.2,
                "sample_weight": 1.0,
                "y": 0,
            },
        ]
        metadata = {
            "label_protocol_version": "scenario_pack_expected_next_v1",
            "source_scenario_set": "haircare_llm_dataset_test",
            "feature_columns": [
                "category",
                "candidate_type",
                "last1_product_type",
                "last1_category",
                "month_of_year",
                "day_of_week",
                "candidate_is_fragrance_slot",
                "candidate_position_in_chain",
                "candidate_popularity_in_train",
            ],
            "categorical_features": [
                "category",
                "candidate_type",
                "last1_product_type",
                "last1_category",
            ],
            "numeric_features": [
                "month_of_year",
                "day_of_week",
                "candidate_is_fragrance_slot",
                "candidate_position_in_chain",
                "candidate_popularity_in_train",
            ],
            "candidate_types_by_category": {
                "haircare": ["shampoo", "conditioner", "hair_mask", "hair_oil", "scalp_serum", "leave_in"]
            },
            "rules_chain_by_category": {
                "haircare": ["shampoo", "conditioner", "hair_mask", "hair_oil", "scalp_serum", "leave_in"]
            },
            "candidate_popularity_in_train_by_category": {
                "haircare": {
                    "conditioner": 0.2,
                    "hair_mask": 0.2,
                    "hair_oil": 0.2,
                    "leave_in": 0.2,
                    "scalp_serum": 0.2,
                }
            },
            "owned_feature_columns": [],
            "owned_feature_map": {},
            "baselines": {},
        }
        splits = {
            "train_user_ids": [101],
            "val_user_ids": [102],
            "test_user_ids": [103],
        }

        canonical_report_path = Path(__file__).resolve().parents[2] / "reports" / "roadmap_nextstep_v4_eval.json"
        canonical_before = canonical_report_path.read_text(encoding="utf-8") if canonical_report_path.exists() else None

        with TemporaryDirectory() as data_tmp, TemporaryDirectory() as model_tmp:
            data_dir = Path(data_tmp)
            model_dir = Path(model_tmp)
            pd.DataFrame(rows).to_csv(data_dir / "dataset.csv", index=False)
            (data_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
            (data_dir / "splits.json").write_text(json.dumps(splits, ensure_ascii=False, indent=2), encoding="utf-8")

            call_command(
                "train_roadmap_nextstep_model_v4",
                data_dir=str(data_dir),
                model_dir=str(model_dir),
                estimator="logistic",
                allow_fallback=True,
                trials=1,
                negative_samples_per_episode=2,
            )

            trained_metadata = json.loads((model_dir / "metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(
                str(trained_metadata.get("model_version")),
                "roadmap_next_step_v4__haircare_llm_dataset_test",
            )
            self.assertEqual(
                str(trained_metadata.get("report_stem")),
                "roadmap_nextstep_v4_eval__haircare_llm_dataset_test",
            )

            isolated_report_path = (
                Path(__file__).resolve().parents[2]
                / "reports"
                / "roadmap_nextstep_v4_eval__haircare_llm_dataset_test.json"
            )
            self.assertTrue(isolated_report_path.exists())
            isolated_report = json.loads(isolated_report_path.read_text(encoding="utf-8"))
            self.assertEqual(
                str(isolated_report.get("model_version")),
                "roadmap_next_step_v4__haircare_llm_dataset_test",
            )
            self.assertEqual(
                str(isolated_report.get("report_stem")),
                "roadmap_nextstep_v4_eval__haircare_llm_dataset_test",
            )

        canonical_after = canonical_report_path.read_text(encoding="utf-8") if canonical_report_path.exists() else None
        self.assertEqual(canonical_after, canonical_before)

    def test_filtered_scenario_style_training_uses_unique_report_stem(self):
        rows = [
            {
                "user_id": 201,
                "episode_id": 2001,
                "group_id": 2001,
                "category": "haircare",
                "candidate_type": "scalp_serum",
                "last1_product_type": "shampoo",
                "last1_category": "haircare",
                "month_of_year": 3,
                "day_of_week": 4,
                "candidate_is_fragrance_slot": 0,
                "candidate_position_in_chain": 5,
                "candidate_popularity_in_train": 0.2,
                "sample_weight": 1.25,
                "y": 1,
            },
            {
                "user_id": 201,
                "episode_id": 2001,
                "group_id": 2001,
                "category": "haircare",
                "candidate_type": "conditioner",
                "last1_product_type": "shampoo",
                "last1_category": "haircare",
                "month_of_year": 3,
                "day_of_week": 4,
                "candidate_is_fragrance_slot": 0,
                "candidate_position_in_chain": 2,
                "candidate_popularity_in_train": 0.2,
                "sample_weight": 1.0,
                "y": 0,
            },
            {
                "user_id": 202,
                "episode_id": 2002,
                "group_id": 2002,
                "category": "haircare",
                "candidate_type": "scalp_serum",
                "last1_product_type": "shampoo",
                "last1_category": "haircare",
                "month_of_year": 3,
                "day_of_week": 5,
                "candidate_is_fragrance_slot": 0,
                "candidate_position_in_chain": 5,
                "candidate_popularity_in_train": 0.2,
                "sample_weight": 1.15,
                "y": 1,
            },
            {
                "user_id": 202,
                "episode_id": 2002,
                "group_id": 2002,
                "category": "haircare",
                "candidate_type": "hair_mask",
                "last1_product_type": "shampoo",
                "last1_category": "haircare",
                "month_of_year": 3,
                "day_of_week": 5,
                "candidate_is_fragrance_slot": 0,
                "candidate_position_in_chain": 3,
                "candidate_popularity_in_train": 0.2,
                "sample_weight": 1.0,
                "y": 0,
            },
            {
                "user_id": 203,
                "episode_id": 2003,
                "group_id": 2003,
                "category": "haircare",
                "candidate_type": "scalp_serum",
                "last1_product_type": "shampoo",
                "last1_category": "haircare",
                "month_of_year": 3,
                "day_of_week": 6,
                "candidate_is_fragrance_slot": 0,
                "candidate_position_in_chain": 5,
                "candidate_popularity_in_train": 0.2,
                "sample_weight": 1.1,
                "y": 1,
            },
            {
                "user_id": 203,
                "episode_id": 2003,
                "group_id": 2003,
                "category": "haircare",
                "candidate_type": "leave_in",
                "last1_product_type": "shampoo",
                "last1_category": "haircare",
                "month_of_year": 3,
                "day_of_week": 6,
                "candidate_is_fragrance_slot": 0,
                "candidate_position_in_chain": 6,
                "candidate_popularity_in_train": 0.2,
                "sample_weight": 1.0,
                "y": 0,
            },
        ]
        metadata = {
            "label_protocol_version": "scenario_pack_expected_next_v1",
            "source_scenario_set": "haircare_llm_dataset_test",
            "expected_next_product_type_filter": ["scalp_serum"],
            "feature_columns": [
                "category",
                "candidate_type",
                "last1_product_type",
                "last1_category",
                "month_of_year",
                "day_of_week",
                "candidate_is_fragrance_slot",
                "candidate_position_in_chain",
                "candidate_popularity_in_train",
            ],
            "categorical_features": [
                "category",
                "candidate_type",
                "last1_product_type",
                "last1_category",
            ],
            "numeric_features": [
                "month_of_year",
                "day_of_week",
                "candidate_is_fragrance_slot",
                "candidate_position_in_chain",
                "candidate_popularity_in_train",
            ],
            "candidate_types_by_category": {
                "haircare": ["shampoo", "conditioner", "hair_mask", "hair_oil", "scalp_serum", "leave_in"]
            },
            "rules_chain_by_category": {
                "haircare": ["shampoo", "conditioner", "hair_mask", "hair_oil", "scalp_serum", "leave_in"]
            },
            "candidate_popularity_in_train_by_category": {
                "haircare": {
                    "conditioner": 0.2,
                    "hair_mask": 0.2,
                    "hair_oil": 0.2,
                    "leave_in": 0.2,
                    "scalp_serum": 0.2,
                }
            },
            "owned_feature_columns": [],
            "owned_feature_map": {},
            "baselines": {},
        }
        splits = {
            "train_user_ids": [201],
            "val_user_ids": [202],
            "test_user_ids": [203],
        }

        with TemporaryDirectory() as data_tmp, TemporaryDirectory() as model_tmp:
            data_dir = Path(data_tmp)
            model_dir = Path(model_tmp)
            pd.DataFrame(rows).to_csv(data_dir / "dataset.csv", index=False)
            (data_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
            (data_dir / "splits.json").write_text(json.dumps(splits, ensure_ascii=False, indent=2), encoding="utf-8")

            call_command(
                "train_roadmap_nextstep_model_v4",
                data_dir=str(data_dir),
                model_dir=str(model_dir),
                estimator="logistic",
                allow_fallback=True,
                trials=1,
                negative_samples_per_episode=2,
            )

            trained_metadata = json.loads((model_dir / "metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(
                str(trained_metadata.get("model_version")),
                "roadmap_next_step_v4__haircare_llm_dataset_test__scalp_serum",
            )
            self.assertEqual(
                str(trained_metadata.get("report_stem")),
                "roadmap_nextstep_v4_eval__haircare_llm_dataset_test__scalp_serum",
            )

            isolated_report_path = (
                Path(__file__).resolve().parents[2]
                / "reports"
                / "roadmap_nextstep_v4_eval__haircare_llm_dataset_test__scalp_serum.json"
            )
            self.assertTrue(isolated_report_path.exists())

    def test_build_dataset_can_filter_expected_next_product_types(self):
        payload = {
            "scenario_set": "haircare_llm_filter_test",
            "category": "haircare",
            "shared_catalog": [
                {
                    "product_key": "clarity_shampoo",
                    "name": "Clarity Shampoo",
                    "brand": "Scenario Lab",
                    "price": "5200",
                    "currency": "KZT",
                    "category": "haircare",
                    "product_type": "shampoo",
                    "concerns": ["oiliness", "flakes", "itchiness"],
                    "actives": ["salicylic_acid", "zinc_pca"],
                    "flags": [],
                    "attrs": {"hair_type": "wavy", "scalp_type": "oily", "hair_thickness": "medium"},
                    "raw_meta": {"line": "Scalp", "finish": "fresh"},
                    "ingredients_inci": "Aqua, Salicylic Acid, Zinc PCA",
                },
                {
                    "product_key": "soft_conditioner",
                    "name": "Soft Conditioner",
                    "brand": "Scenario Lab",
                    "price": "6100",
                    "currency": "KZT",
                    "category": "haircare",
                    "product_type": "conditioner",
                    "concerns": ["smoothness", "detangling"],
                    "actives": ["panthenol", "squalane"],
                    "flags": [],
                    "attrs": {"hair_type": "wavy", "scalp_type": "normal", "hair_thickness": "medium"},
                    "raw_meta": {"line": "Length", "finish": "soft"},
                    "ingredients_inci": "Aqua, Panthenol, Squalane",
                },
                {
                    "product_key": "scalp_serum",
                    "name": "Scalp Serum",
                    "brand": "Scenario Lab",
                    "price": "7000",
                    "currency": "KZT",
                    "category": "haircare",
                    "product_type": "scalp_serum",
                    "concerns": ["oiliness", "flakes", "itchiness", "build_up"],
                    "actives": ["salicylic_acid", "niacinamide", "zinc_pca"],
                    "flags": [],
                    "attrs": {"hair_type": "wavy", "scalp_type": "oily", "hair_thickness": "medium"},
                    "raw_meta": {"line": "Scalp", "finish": "fresh"},
                    "ingredients_inci": "Aqua, Salicylic Acid, Niacinamide, Zinc PCA",
                },
            ],
            "scenarios": [
                {
                    "slug": "scalp_case",
                    "segment": "scalp_serum_beats_conditioner",
                    "expected_next_product_type": "scalp_serum",
                    "outcome_tag": "completed_exact",
                    "profile": {
                        "skin_type": "oily",
                        "goals": ["oiliness", "scalp_balance", "scalp_health"],
                        "avoid_flags": [],
                        "budget": "medium",
                        "hair_profile": {
                            "hair_type": "wavy",
                            "scalp_type": "oily",
                            "hair_thickness": "medium",
                            "concerns": ["oiliness", "flakes", "itchiness"],
                        },
                    },
                    "transactions": [
                        {"offset_days": -24, "channel": "online", "items": [{"product_key": "clarity_shampoo", "quantity": 1}]}
                    ],
                    "steps": [
                        {"step_index": 1, "product_type": "shampoo", "status": "completed", "recommended_product_key": "clarity_shampoo", "cadence": "weekly", "score": 0.7, "confidence": 0.75, "why": ["wash_done"]},
                        {"step_index": 2, "product_type": "scalp_serum", "status": "recommended", "recommended_product_key": "scalp_serum", "cadence": "daily", "score": 0.92, "confidence": 0.9, "why": ["scalp_signal"]},
                    ],
                    "events": [
                        {"event_type": "roadmap_plan_refreshed", "offset_hours": 0, "step_index": None, "context": {}},
                        {"event_type": "roadmap_step_completed", "offset_hours": 18, "step_index": 2, "context": {"category": "haircare", "product_type": "scalp_serum", "matched_by": "recommended_product_id", "match_meta": {"recommended_product_key": "scalp_serum", "purchased_product_key": "scalp_serum"}}},
                    ],
                },
                {
                    "slug": "conditioner_case",
                    "segment": "conditioner_beats_scalp",
                    "expected_next_product_type": "conditioner",
                    "outcome_tag": "completed_exact",
                    "profile": {
                        "skin_type": "dry",
                        "goals": ["smoothness", "detangling"],
                        "avoid_flags": [],
                        "budget": "low",
                        "hair_profile": {
                            "hair_type": "wavy",
                            "scalp_type": "normal",
                            "hair_thickness": "medium",
                            "concerns": ["smoothness", "detangling"],
                        },
                    },
                    "transactions": [
                        {"offset_days": -20, "channel": "online", "items": [{"product_key": "clarity_shampoo", "quantity": 1}]}
                    ],
                    "steps": [
                        {"step_index": 1, "product_type": "shampoo", "status": "completed", "recommended_product_key": "clarity_shampoo", "cadence": "weekly", "score": 0.7, "confidence": 0.75, "why": ["wash_done"]},
                        {"step_index": 2, "product_type": "conditioner", "status": "recommended", "recommended_product_key": "soft_conditioner", "cadence": "weekly", "score": 0.88, "confidence": 0.84, "why": ["length_signal"]},
                    ],
                    "events": [
                        {"event_type": "roadmap_plan_refreshed", "offset_hours": 0, "step_index": None, "context": {}},
                        {"event_type": "roadmap_step_completed", "offset_hours": 16, "step_index": 2, "context": {"category": "haircare", "product_type": "conditioner", "matched_by": "recommended_product_id", "match_meta": {"recommended_product_key": "soft_conditioner", "purchased_product_key": "soft_conditioner"}}},
                    ],
                },
            ],
        }

        artifact = {
            "task": "roadmap_nextstep_v4_ranking",
            "model": {"stub": True},
            "feature_columns": [
                "category",
                "candidate_type",
                "last1_product_type",
                "last1_category",
                "planned_target_product_type",
                "month_of_year",
                "day_of_week",
                "candidate_position_in_chain",
                "candidate_popularity_in_train",
            ],
            "categorical_features": [
                "category",
                "candidate_type",
                "last1_product_type",
                "last1_category",
                "planned_target_product_type",
            ],
            "numeric_features": [
                "month_of_year",
                "day_of_week",
                "candidate_position_in_chain",
                "candidate_popularity_in_train",
            ],
            "candidate_types_by_category": {
                "haircare": ["shampoo", "conditioner", "hair_mask", "scalp_serum", "leave_in"]
            },
            "rules_chain_by_category": {
                "haircare": ["shampoo", "conditioner", "hair_mask", "hair_oil", "scalp_serum", "leave_in"]
            },
            "candidate_popularity_in_train_by_category": {
                "haircare": {
                    "shampoo": 0.2,
                    "conditioner": 0.2,
                    "hair_mask": 0.2,
                    "scalp_serum": 0.2,
                    "leave_in": 0.2,
                }
            },
            "owned_feature_columns": [],
            "owned_feature_map": {},
        }

        with TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            input_path = tmp_path / "llm.json"
            pack_dir = tmp_path / "scenario_pack"
            dataset_dir = tmp_path / "dataset"
            model_path = tmp_path / "template.pkl"
            input_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            joblib.dump(artifact, model_path)

            call_command(
                "build_roadmap_scenario_pack_from_llm_json",
                input=str(input_path),
                out_dir=str(pack_dir),
                replicas=3,
                days_ago_start=60,
                id_base=986000,
            )

            call_command(
                "build_roadmap_nextstep_dataset_from_scenario_pack",
                path=str(pack_dir),
                out_dir=str(dataset_dir),
                template_model_path=str(model_path),
                seed=7,
                val_ratio=0.25,
                test_ratio=0.25,
                expected_next_product_types="scalp_serum",
            )

            df = pd.read_csv(dataset_dir / "dataset.csv")
            metadata = json.loads((dataset_dir / "metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(sorted(df["label"].unique().tolist()), ["scalp_serum"])
            self.assertEqual(sorted(df.loc[df["y"] == 1, "candidate_type"].unique().tolist()), ["scalp_serum"])
            self.assertEqual(list(metadata.get("expected_next_product_type_filter") or []), ["scalp_serum"])
