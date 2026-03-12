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


class ReplayRoadmapScenarioPackTests(TestCase):
    @override_settings(
        ROADMAP_NEXTSTEP_V4_HAIRCARE_SCALP_RERANK_ENABLED=False,
        ROADMAP_NEXTSTEP_V4_HAIRCARE_LEAVEIN_RERANK_ENABLED=False,
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
