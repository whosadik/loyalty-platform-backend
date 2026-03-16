from __future__ import annotations

import json
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from django.core.management import call_command
from django.test import TestCase, override_settings


class PlannedTargetModel:
    def predict(self, X):
        candidate_types = X["candidate_type"].astype(str).tolist()
        planned_targets = X["planned_target_product_type"].astype(str).tolist()
        return [1.0 if candidate == planned else 0.0 for candidate, planned in zip(candidate_types, planned_targets)]


class ConditionerBiasModel:
    def predict(self, X):
        candidate_types = X["candidate_type"].astype(str).tolist()
        planned_targets = X["planned_target_product_type"].astype(str).tolist()
        out: list[float] = []
        for candidate, planned in zip(candidate_types, planned_targets):
            if candidate == "conditioner":
                out.append(2.0)
            elif candidate == planned:
                out.append(0.8)
            else:
                out.append(0.0)
        return out


class ConditionerSlightBiasModel:
    def predict(self, X):
        candidate_types = X["candidate_type"].astype(str).tolist()
        planned_targets = X["planned_target_product_type"].astype(str).tolist()
        out: list[float] = []
        for candidate, planned in zip(candidate_types, planned_targets):
            if candidate == "conditioner":
                out.append(0.55)
            elif candidate == planned:
                out.append(0.45)
            else:
                out.append(0.0)
        return out


class ReplayRoadmapScenarioPackTests(TestCase):
    @override_settings(
        ROADMAP_NEXTSTEP_V4_HAIRCARE_SCALP_RERANK_ENABLED=False,
        ROADMAP_NEXTSTEP_V4_HAIRCARE_LEAVEIN_RERANK_ENABLED=False,
        ROADMAP_NEXTSTEP_V4_HAIRCARE_CORECHAIN_TEACHER_RERANK_ENABLED=False,
    )
    def test_replay_reports_expected_and_outcome_hit_rates(self):
        with TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            pack_dir = tmp_path / "scenario_pack"
            active_model_path = tmp_path / "active.pkl"
            candidate_model_path = tmp_path / "candidate.pkl"
            active_model_path.write_text("placeholder", encoding="utf-8")
            candidate_model_path.write_text("placeholder", encoding="utf-8")

            call_command(
                "generate_roadmap_scenario_pack",
                out_dir=str(pack_dir),
                scenario_set="haircare_v1",
                replicas=1,
                days_ago_start=75,
                id_base=951000,
            )

            active_artifact = {
                "task": "roadmap_nextstep_v4_ranking",
                "model": PlannedTargetModel(),
                "preprocessor": None,
                "model_type": "custom",
                "feature_columns": [
                    "category",
                    "candidate_type",
                    "planned_target_product_type",
                    "anchor_product_type",
                ],
                "categorical_features": [
                    "category",
                    "candidate_type",
                    "planned_target_product_type",
                    "anchor_product_type",
                ],
                "numeric_features": [],
                "candidate_types_by_category": {
                    "haircare": [
                        "shampoo",
                        "conditioner",
                        "hair_mask",
                        "scalp_serum",
                        "leave_in",
                    ]
                },
                "rules_chain_by_category": {
                    "haircare": [
                        "shampoo",
                        "conditioner",
                        "hair_mask",
                        "hair_oil",
                        "scalp_serum",
                        "leave_in",
                    ]
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
                "temperature": 1.0,
            }
            candidate_artifact = dict(active_artifact)
            candidate_artifact["model"] = ConditionerBiasModel()

            def _artifact_for_path(path):
                raw = str(path)
                if raw == str(active_model_path):
                    return active_artifact
                if raw == str(candidate_model_path):
                    return candidate_artifact
                return None

            def _summary_for_path(path):
                name = Path(str(path)).stem
                return {
                    "model_path": str(path),
                    "exists": True,
                    "metadata_exists": False,
                    "eval_report_exists": False,
                    "model_version": name,
                    "selected_feature_set": "test",
                    "metrics_test": {},
                    "runtime_guard": {},
                }

            out = StringIO()
            with patch(
                "admin_tools.management.commands.replay_roadmap_scenario_pack._load_model_for_path",
                side_effect=_artifact_for_path,
            ), patch(
                "admin_tools.management.commands.replay_roadmap_scenario_pack.nextstep_model_artifact_summary",
                side_effect=_summary_for_path,
            ):
                call_command(
                    "replay_roadmap_scenario_pack",
                    path=str(pack_dir),
                    active_model_path=str(active_model_path),
                    candidate_model_path=str(candidate_model_path),
                    format="json",
                    stdout=out,
                )

            payload = json.loads(out.getvalue())
            compare = payload["compare"]
            self.assertEqual(int(compare["episodes_scored"]), 6)
            self.assertEqual(int(compare["outcome_eligible"]), 5)
            self.assertAlmostEqual(float(compare["active_expected_hit_rate"]), 1.0)
            self.assertAlmostEqual(float(compare["candidate_expected_hit_rate"]), 2.0 / 6.0, places=6)
            self.assertAlmostEqual(float(compare["active_expected_hit_rate_at3"]), 1.0)
            self.assertAlmostEqual(float(compare["candidate_expected_hit_rate_at3"]), 1.0)
            self.assertAlmostEqual(float(compare["active_outcome_hit_rate"]), 1.0)
            self.assertAlmostEqual(float(compare["candidate_outcome_hit_rate"]), 2.0 / 5.0, places=6)
            self.assertAlmostEqual(float(compare["active_outcome_hit_rate_at3"]), 1.0)
            self.assertAlmostEqual(float(compare["candidate_outcome_hit_rate_at3"]), 1.0)

            by_scenario = {row["scenario_key"]: row for row in payload["by_scenario_key"]}
            self.assertAlmostEqual(
                float(by_scenario["semantic_conditioner_match"]["candidate_outcome_hit_rate"]),
                1.0,
            )
            self.assertAlmostEqual(
                float(by_scenario["leave_in_after_mask"]["candidate_expected_hit_rate_at3"]),
                1.0,
            )
            self.assertAlmostEqual(
                float(by_scenario["leave_in_after_mask"]["candidate_expected_hit_rate"]),
                0.0,
            )

            swap_rows = payload["swap_rows"]
            self.assertTrue(
                any(
                    row["active_top1"] == "hair_mask"
                    and row["candidate_top1"] == "conditioner"
                    and int(row["plans"]) == 2
                    for row in swap_rows
                )
            )

    @override_settings(
        ROADMAP_NEXTSTEP_V4_HAIRCARE_SCALP_RERANK_ENABLED=False,
        ROADMAP_NEXTSTEP_V4_HAIRCARE_LEAVEIN_RERANK_ENABLED=False,
        ROADMAP_NEXTSTEP_V4_HAIRCARE_CORECHAIN_TEACHER_RERANK_ENABLED=False,
    )
    def test_replay_can_blend_candidate_with_teacher_model(self):
        with TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            pack_dir = tmp_path / "scenario_pack"
            active_model_path = tmp_path / "active.pkl"
            candidate_model_path = tmp_path / "candidate.pkl"
            teacher_model_path = tmp_path / "teacher.pkl"
            active_model_path.write_text("placeholder", encoding="utf-8")
            candidate_model_path.write_text("placeholder", encoding="utf-8")
            teacher_model_path.write_text("placeholder", encoding="utf-8")

            call_command(
                "generate_roadmap_scenario_pack",
                out_dir=str(pack_dir),
                scenario_set="haircare_v1",
                replicas=1,
                days_ago_start=75,
                id_base=952000,
            )

            active_artifact = {
                "task": "roadmap_nextstep_v4_ranking",
                "model": PlannedTargetModel(),
                "preprocessor": None,
                "model_type": "custom",
                "feature_columns": [
                    "category",
                    "candidate_type",
                    "planned_target_product_type",
                    "anchor_product_type",
                ],
                "categorical_features": [
                    "category",
                    "candidate_type",
                    "planned_target_product_type",
                    "anchor_product_type",
                ],
                "numeric_features": [],
                "candidate_types_by_category": {
                    "haircare": [
                        "shampoo",
                        "conditioner",
                        "hair_mask",
                        "scalp_serum",
                        "leave_in",
                    ]
                },
                "rules_chain_by_category": {
                    "haircare": [
                        "shampoo",
                        "conditioner",
                        "hair_mask",
                        "hair_oil",
                        "scalp_serum",
                        "leave_in",
                    ]
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
                "temperature": 1.0,
            }
            candidate_artifact = dict(active_artifact)
            candidate_artifact["model"] = ConditionerBiasModel()
            teacher_artifact = dict(active_artifact)
            teacher_artifact["model"] = PlannedTargetModel()

            def _artifact_for_path(path):
                raw = str(path)
                if raw == str(active_model_path):
                    return active_artifact
                if raw == str(candidate_model_path):
                    return candidate_artifact
                if raw == str(teacher_model_path):
                    return teacher_artifact
                return None

            def _summary_for_path(path):
                name = Path(str(path)).stem
                return {
                    "model_path": str(path),
                    "exists": True,
                    "metadata_exists": False,
                    "eval_report_exists": False,
                    "model_version": name,
                    "selected_feature_set": "test",
                    "metrics_test": {},
                    "runtime_guard": {},
                }

            out = StringIO()
            with patch(
                "admin_tools.management.commands.replay_roadmap_scenario_pack._load_model_for_path",
                side_effect=_artifact_for_path,
            ), patch(
                "admin_tools.management.commands.replay_roadmap_scenario_pack.nextstep_model_artifact_summary",
                side_effect=_summary_for_path,
            ):
                call_command(
                    "replay_roadmap_scenario_pack",
                    path=str(pack_dir),
                    active_model_path=str(active_model_path),
                    candidate_model_path=str(candidate_model_path),
                    candidate_teacher_model_path=str(teacher_model_path),
                    candidate_teacher_weight=2.0,
                    format="json",
                    stdout=out,
                )

            payload = json.loads(out.getvalue())
            self.assertEqual(payload["params"]["candidate_teacher_model_path"], str(teacher_model_path))
            self.assertAlmostEqual(float(payload["params"]["candidate_teacher_weight"]), 2.0, places=6)
            self.assertEqual(payload["candidate_teacher_model"]["model_version"], "teacher")
            compare = payload["compare"]
            self.assertAlmostEqual(float(compare["candidate_expected_hit_rate"]), 1.0, places=6)
            self.assertAlmostEqual(float(compare["candidate_outcome_hit_rate"]), 1.0, places=6)
            self.assertEqual(len(payload["swap_rows"]), 0)

    @override_settings(
        ROADMAP_NEXTSTEP_V4_HAIRCARE_RUNTIME_BIAS_ENABLED=False,
        ROADMAP_NEXTSTEP_V4_HAIRCARE_SCALP_RERANK_ENABLED=True,
        ROADMAP_NEXTSTEP_V4_HAIRCARE_LEAVEIN_RERANK_ENABLED=False,
        ROADMAP_NEXTSTEP_V4_HAIRCARE_CORECHAIN_TEACHER_RERANK_ENABLED=False,
    )
    def test_replay_can_disable_runtime_policies_for_model_only_compare(self):
        llm_payload = {
            "scenario_set": "haircare_llm_test_scalp_only",
            "category": "haircare",
            "shared_catalog": [
                {
                    "product_key": "clarity_shampoo",
                    "name": "Clarity Shampoo",
                    "brand": "TestBrand",
                    "price": "5000",
                    "currency": "KZT",
                    "category": "haircare",
                    "product_type": "shampoo",
                    "concerns": ["oiliness", "flakes", "itchiness"],
                    "actives": ["salicylic_acid", "zinc_pca"],
                    "flags": [],
                    "attrs": {
                        "hair_type": "wavy",
                        "scalp_type": "oily",
                        "hair_thickness": "medium",
                    },
                    "raw_meta": {"line": "Scalp", "finish": "fresh"},
                    "ingredients_inci": "Aqua, Salicylic Acid, Zinc PCA",
                },
                {
                    "product_key": "soft_conditioner",
                    "name": "Soft Conditioner",
                    "brand": "TestBrand",
                    "price": "5200",
                    "currency": "KZT",
                    "category": "haircare",
                    "product_type": "conditioner",
                    "concerns": ["smoothness", "detangling"],
                    "actives": ["panthenol", "squalane"],
                    "flags": [],
                    "attrs": {
                        "hair_type": "wavy",
                        "scalp_type": "normal",
                        "hair_thickness": "medium",
                    },
                    "raw_meta": {"line": "Length", "finish": "soft"},
                    "ingredients_inci": "Aqua, Panthenol, Squalane",
                },
                {
                    "product_key": "clear_scalp_serum",
                    "name": "Clear Scalp Serum",
                    "brand": "TestBrand",
                    "price": "6500",
                    "currency": "KZT",
                    "category": "haircare",
                    "product_type": "scalp_serum",
                    "concerns": ["oiliness", "flakes", "itchiness", "build_up"],
                    "actives": ["salicylic_acid", "niacinamide", "zinc_pca"],
                    "flags": [],
                    "attrs": {
                        "hair_type": "wavy",
                        "scalp_type": "oily",
                        "hair_thickness": "medium",
                    },
                    "raw_meta": {"line": "Scalp", "finish": "fresh"},
                    "ingredients_inci": "Aqua, Salicylic Acid, Niacinamide, Zinc PCA",
                },
            ],
            "scenarios": [
                {
                    "slug": "oily_scalp_serum_counterexample",
                    "segment": "scalp_serum_should_beat_conditioner",
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
                        {
                            "offset_days": -21,
                            "channel": "online",
                            "items": [{"product_key": "clarity_shampoo", "quantity": 1}],
                        },
                        {
                            "offset_days": -4,
                            "channel": "offline",
                            "items": [{"product_key": "clarity_shampoo", "quantity": 1}],
                        },
                    ],
                    "steps": [
                        {
                            "step_index": 1,
                            "product_type": "shampoo",
                            "status": "completed",
                            "recommended_product_key": "clarity_shampoo",
                            "cadence": "weekly",
                            "score": 0.7,
                            "confidence": 0.75,
                            "why": ["wash_done"],
                        },
                        {
                            "step_index": 2,
                            "product_type": "conditioner",
                            "status": "missing",
                            "recommended_product_key": "soft_conditioner",
                            "cadence": "weekly",
                            "score": 0.4,
                            "confidence": 0.35,
                            "why": ["secondary_step"],
                        },
                        {
                            "step_index": 3,
                            "product_type": "scalp_serum",
                            "status": "recommended",
                            "recommended_product_key": "clear_scalp_serum",
                            "cadence": "daily",
                            "score": 0.92,
                            "confidence": 0.9,
                            "why": ["scalp_signal"],
                        },
                    ],
                    "events": [
                        {
                            "event_type": "roadmap_plan_refreshed",
                            "offset_hours": 0,
                            "step_index": None,
                            "context": {"category": "haircare"},
                        },
                        {
                            "event_type": "roadmap_step_exposed",
                            "offset_hours": 2,
                            "step_index": 3,
                            "context": {
                                "category": "haircare",
                                "product_type": "scalp_serum",
                                "recommended_product_key": "clear_scalp_serum",
                            },
                        },
                        {
                            "event_type": "roadmap_step_completed",
                            "offset_hours": 20,
                            "step_index": 3,
                            "context": {
                                "category": "haircare",
                                "product_type": "scalp_serum",
                                "matched_by": "recommended_product_id",
                                "match_meta": {
                                    "recommended_product_key": "clear_scalp_serum",
                                    "purchased_product_key": "clear_scalp_serum",
                                },
                            },
                        },
                    ],
                }
            ],
        }

        with TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            input_path = tmp_path / "llm.json"
            pack_dir = tmp_path / "scenario_pack"
            active_model_path = tmp_path / "active.pkl"
            candidate_model_path = tmp_path / "candidate.pkl"
            input_path.write_text(json.dumps(llm_payload, ensure_ascii=False, indent=2), encoding="utf-8")
            active_model_path.write_text("placeholder", encoding="utf-8")
            candidate_model_path.write_text("placeholder", encoding="utf-8")

            call_command(
                "build_roadmap_scenario_pack_from_llm_json",
                input=str(input_path),
                out_dir=str(pack_dir),
                id_base=975000,
            )

            artifact = {
                "task": "roadmap_nextstep_v4_ranking",
                "model": ConditionerSlightBiasModel(),
                "preprocessor": None,
                "model_type": "custom",
                "feature_columns": [
                    "category",
                    "candidate_type",
                    "planned_target_product_type",
                    "anchor_product_type",
                ],
                "categorical_features": [
                    "category",
                    "candidate_type",
                    "planned_target_product_type",
                    "anchor_product_type",
                ],
                "numeric_features": [],
                "candidate_types_by_category": {
                    "haircare": ["shampoo", "conditioner", "hair_mask", "scalp_serum", "leave_in"]
                },
                "rules_chain_by_category": {
                    "haircare": [
                        "shampoo",
                        "conditioner",
                        "hair_mask",
                        "hair_oil",
                        "scalp_serum",
                        "leave_in",
                    ]
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
                "temperature": 1.0,
            }

            def _artifact_for_path(path):
                raw = str(path)
                if raw in {str(active_model_path), str(candidate_model_path)}:
                    return artifact
                return None

            def _summary_for_path(path):
                name = Path(str(path)).stem
                return {
                    "model_path": str(path),
                    "exists": True,
                    "metadata_exists": False,
                    "eval_report_exists": False,
                    "model_version": name,
                    "selected_feature_set": "test",
                    "metrics_test": {},
                    "runtime_guard": {},
                }

            out_with_policy = StringIO()
            out_model_only = StringIO()
            with patch(
                "admin_tools.management.commands.replay_roadmap_scenario_pack._load_model_for_path",
                side_effect=_artifact_for_path,
            ), patch(
                "admin_tools.management.commands.replay_roadmap_scenario_pack.nextstep_model_artifact_summary",
                side_effect=_summary_for_path,
            ):
                call_command(
                    "replay_roadmap_scenario_pack",
                    path=str(pack_dir),
                    active_model_path=str(active_model_path),
                    candidate_model_path=str(candidate_model_path),
                    format="json",
                    stdout=out_with_policy,
                )
                call_command(
                    "replay_roadmap_scenario_pack",
                    path=str(pack_dir),
                    active_model_path=str(active_model_path),
                    candidate_model_path=str(candidate_model_path),
                    disable_runtime_policies=True,
                    format="json",
                    stdout=out_model_only,
                )

            payload_with_policy = json.loads(out_with_policy.getvalue())
            payload_model_only = json.loads(out_model_only.getvalue())

            self.assertFalse(bool(payload_with_policy["params"]["disable_runtime_policies"]))
            self.assertTrue(bool(payload_model_only["params"]["disable_runtime_policies"]))
            self.assertEqual(
                payload_with_policy["episode_rows"][0]["planned_target_product_type"],
                "scalp_serum",
            )
            self.assertEqual(payload_with_policy["episode_rows"][0]["active_top1"], "scalp_serum")
            self.assertEqual(payload_model_only["episode_rows"][0]["active_top1"], "conditioner")
            self.assertAlmostEqual(
                float(payload_with_policy["compare"]["active_expected_hit_rate"]),
                1.0,
                places=6,
            )
            self.assertAlmostEqual(
                float(payload_model_only["compare"]["active_expected_hit_rate"]),
                0.0,
                places=6,
            )
