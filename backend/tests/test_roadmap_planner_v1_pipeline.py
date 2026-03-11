from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from django.core.management import call_command
from django.test import TestCase

try:
    import joblib
except Exception:  # pragma: no cover
    joblib = None

try:
    import pandas as pd
except Exception:  # pragma: no cover
    pd = None


def _episode_rows(
    *,
    episode_id: int,
    user_id: int,
    category: str,
    current_next_product_type: str,
    label: str,
    last1_product_type: str,
    last1_category: str,
    popularity_map: dict[str, float],
) -> list[dict[str, object]]:
    candidates = [current_next_product_type, "fallback_other", "__stop__"]
    if label not in candidates:
        candidates.insert(1, label)

    rows: list[dict[str, object]] = []
    for idx, candidate in enumerate(dict.fromkeys(candidates).keys(), start=1):
        rows.append(
            {
                "episode_id": episode_id,
                "user_id": user_id,
                "category": category,
                "t0_utc": "2026-03-01T00:00:00Z",
                "split": "__unused__",
                "label": label,
                "candidate_type": candidate,
                "y": int(candidate == label),
                "current_next_product_type": current_next_product_type,
                "last1_product_type": last1_product_type,
                "last1_category": last1_category,
                "candidate_position_in_generated_plan": idx if candidate != "__stop__" else -1,
                "candidate_is_current_next_step": int(candidate == current_next_product_type),
                "candidate_popularity_in_train": float(popularity_map.get(candidate, 0.01)),
                "candidate_is_stop": int(candidate == "__stop__"),
            }
        )
    return rows


class RoadmapPlannerV1PipelineTests(TestCase):
    def setUp(self):
        if pd is None or joblib is None:
            self.skipTest("pandas + joblib are required")

    def test_trainer_builds_model_and_eval_artifacts_from_planner_dataset(self):
        feature_columns = [
            "category",
            "candidate_type",
            "current_next_product_type",
            "last1_product_type",
            "last1_category",
            "candidate_position_in_generated_plan",
            "candidate_is_current_next_step",
            "candidate_popularity_in_train",
            "candidate_is_stop",
        ]
        categorical_features = [
            "category",
            "candidate_type",
            "current_next_product_type",
            "last1_product_type",
            "last1_category",
        ]
        numeric_features = [
            "candidate_position_in_generated_plan",
            "candidate_is_current_next_step",
            "candidate_popularity_in_train",
            "candidate_is_stop",
        ]

        rows: list[dict[str, object]] = []
        rows.extend(
            _episode_rows(
                episode_id=1,
                user_id=101,
                category="makeup",
                current_next_product_type="foundation",
                label="foundation",
                last1_product_type="primer",
                last1_category="makeup",
                popularity_map={"foundation": 0.60, "fallback_other": 0.10, "__stop__": 0.05},
            )
        )
        rows.extend(
            _episode_rows(
                episode_id=2,
                user_id=102,
                category="skincare",
                current_next_product_type="serum",
                label="serum",
                last1_product_type="cleanser",
                last1_category="skincare",
                popularity_map={"serum": 0.55, "fallback_other": 0.12, "__stop__": 0.04},
            )
        )
        rows.extend(
            _episode_rows(
                episode_id=3,
                user_id=103,
                category="haircare",
                current_next_product_type="shampoo",
                label="__stop__",
                last1_product_type="shampoo",
                last1_category="haircare",
                popularity_map={"shampoo": 0.20, "fallback_other": 0.08, "__stop__": 0.65},
            )
        )
        rows.extend(
            _episode_rows(
                episode_id=4,
                user_id=104,
                category="makeup",
                current_next_product_type="foundation",
                label="foundation",
                last1_product_type="primer",
                last1_category="makeup",
                popularity_map={"foundation": 0.60, "fallback_other": 0.10, "__stop__": 0.05},
            )
        )
        rows.extend(
            _episode_rows(
                episode_id=5,
                user_id=105,
                category="skincare",
                current_next_product_type="serum",
                label="__stop__",
                last1_product_type="serum",
                last1_category="skincare",
                popularity_map={"serum": 0.20, "fallback_other": 0.09, "__stop__": 0.60},
            )
        )
        rows.extend(
            _episode_rows(
                episode_id=6,
                user_id=106,
                category="haircare",
                current_next_product_type="shampoo",
                label="shampoo",
                last1_product_type="conditioner",
                last1_category="haircare",
                popularity_map={"shampoo": 0.52, "fallback_other": 0.10, "__stop__": 0.05},
            )
        )
        frame = pd.DataFrame(rows)

        with TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            data_dir = tmp_path / "planner_data"
            model_dir = tmp_path / "planner_model"
            report_dir = tmp_path / "reports"
            data_dir.mkdir(parents=True, exist_ok=True)
            model_dir.mkdir(parents=True, exist_ok=True)
            report_dir.mkdir(parents=True, exist_ok=True)

            frame.to_parquet(data_dir / "dataset.parquet", index=False)
            (data_dir / "metadata.json").write_text(
                json.dumps(
                    {
                        "feature_columns": feature_columns,
                        "categorical_features": categorical_features,
                        "numeric_features": numeric_features,
                        "candidate_types_by_category": {
                            "makeup": ["foundation", "fallback_other", "__stop__"],
                            "skincare": ["serum", "fallback_other", "__stop__"],
                            "haircare": ["shampoo", "fallback_other", "__stop__"],
                        },
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            (data_dir / "splits.json").write_text(
                json.dumps(
                    {
                        "train": [101, 102, 103],
                        "val": [104],
                        "test": [105, 106],
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )

            with patch("admin_tools.management.commands.train_roadmap_planner_model_v1._repo_root", return_value=tmp_path):
                call_command(
                    "train_roadmap_planner_model_v1",
                    data_dir=str(data_dir),
                    model_dir=str(model_dir),
                    model_version="planner_test_v1",
                    estimator="logistic",
                    allow_fallback=True,
                    trials=2,
                    negative_samples_per_episode=10,
                )

            model_path = model_dir / "model.pkl"
            metadata_path = model_dir / "metadata.json"
            eval_path = model_dir / "eval_report.json"
            eval_md_path = model_dir / "eval_report.md"
            self.assertTrue(model_path.exists())
            self.assertTrue(metadata_path.exists())
            self.assertTrue(eval_path.exists())
            self.assertTrue(eval_md_path.exists())
            self.assertTrue((tmp_path / "reports" / "roadmap_planner_v1_eval.json").exists())

            artifact = joblib.load(model_path)
            self.assertEqual(str(artifact["task"]), "roadmap_planner_v1_ranking")
            self.assertIn(str(artifact["selected_feature_set"]), {"baseline_only", "full"})
            self.assertGreater(len(artifact["feature_columns"]), 0)
            self.assertIn("candidate_popularity_priors", artifact)
            self.assertIn("makeup", artifact["candidate_popularity_priors"])

            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            self.assertEqual(str(metadata["model_version"]), "planner_test_v1")
            self.assertEqual(str(metadata["task"]), "roadmap_planner_v1_ranking")
            self.assertIn("planner_guard", metadata)
            self.assertIn("metrics_test", metadata)
            self.assertIn("dataset_baselines", metadata)
            self.assertIn("candidate_popularity_priors", metadata)

            report = json.loads(eval_path.read_text(encoding="utf-8"))
            self.assertIn("feature_ablation", report)
            self.assertIn("baseline_only", report["feature_ablation"])
            self.assertIn("full", report["feature_ablation"])
            self.assertIn("baseline_comparison", report)
            self.assertIn("test", report["baseline_comparison"])
