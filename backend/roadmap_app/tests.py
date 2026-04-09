from __future__ import annotations

import json
from datetime import timedelta
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import joblib
import pandas as pd
from django.contrib.auth import get_user_model
from django.core.management import call_command
from django.test import SimpleTestCase, TestCase, override_settings
from django.utils import timezone

from roadmap_app.ml_artifact_qualification import (
    build_roadmap_ml_artifact_qualification_payload,
    nextstep_pass_fail_manifest,
)
from roadmap_app.nextstep_artifact_eval import build_nextstep_v4_artifact_eval_report
from roadmap_app.nextstep_decision_quality import build_nextstep_v4_decision_quality_payload
from roadmap_app.nextstep_targeted_retrain import (
    apply_targeted_retrain_weights,
    build_targeted_retrain_comparison_payload,
)
from roadmap_app.ml_next_step import (
    v4_category_staged_rollout_status,
    v4_min_lift_guard_status,
)
from roadmap_app.ml_planner import planner_runtime_guard_status
from roadmap_app.models import RoadmapPlan, RoadmapStep
from roadmap_app.models import RoadmapEvent
from roadmap_app.shadow_evidence import normalized_model_path


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


class ArtifactEvalDummyModel:
    def predict(self, X):
        series = pd.to_numeric(X["candidate_popularity_in_train"], errors="coerce").fillna(0.0)
        return series.astype(float).to_numpy()


def _base_nextstep_metadata(model_version: str) -> dict:
    return {
        "model_version": model_version,
        "task": "roadmap_nextstep_v4_ranking",
        "selected_feature_set": "full",
    }


def _base_eval_report(model_version: str) -> dict:
    return {
        "model_version": model_version,
        "metrics_test": {"ndcg_at_5": 0.65},
        "dataset_baselines": {
            "splits": {
                "test": {
                    "popularity": {
                        "ndcg_at_5": 0.50,
                    }
                }
            }
        },
    }


def _base_uplift_report(days: int) -> dict:
    return {
        "model_path": "",
        "model_version": "",
        "params": {
            "days": days,
            "category": "all",
            "cohort_mode": "fresh",
            "control": "non_model",
        },
        "breakdowns": {
            "by_category": {
                "skincare": {
                    "model_used": {"plans_total": 150},
                    "control": {"plans_total": 150},
                },
                "haircare": {
                    "model_used": {"plans_total": 150},
                    "control": {"plans_total": 150},
                },
                "makeup": {
                    "model_used": {"plans_total": 150},
                    "control": {"plans_total": 150},
                },
            }
        },
        "uplift": {
            "by_category": {
                "skincare": {
                    "step_funnel": {
                        "step_completion_rate": {"abs_lift": 0.02},
                        "step_ctr": {"abs_lift": 0.0},
                    },
                    "offer_funnel": {
                        "offer_redeem_rate": {"abs_lift": 0.01},
                        "offer_ctr": {"abs_lift": 0.0},
                    },
                },
                "haircare": {
                    "step_funnel": {
                        "step_completion_rate": {"abs_lift": 0.02},
                        "step_ctr": {"abs_lift": 0.0},
                    },
                    "offer_funnel": {
                        "offer_redeem_rate": {"abs_lift": 0.01},
                        "offer_ctr": {"abs_lift": 0.0},
                    },
                },
                "makeup": {
                    "step_funnel": {
                        "step_completion_rate": {"abs_lift": 0.02},
                        "step_ctr": {"abs_lift": 0.0},
                    },
                    "offer_funnel": {
                        "offer_redeem_rate": {"abs_lift": 0.01},
                        "offer_ctr": {"abs_lift": 0.0},
                    },
                },
            }
        },
    }


def _create_nextstep_artifact(
    root: Path,
    name: str,
    *,
    with_eval: bool,
    with_uplift: bool,
    model_version: str,
) -> Path:
    artifact_dir = root / name
    artifact_dir.mkdir(parents=True, exist_ok=True)
    model_path = artifact_dir / "model.pkl"
    model_path.write_bytes(b"placeholder-model")
    _write_json(artifact_dir / "metadata.json", _base_nextstep_metadata(model_version))
    if with_eval:
        _write_json(artifact_dir / "eval_report.json", _base_eval_report(model_version))
    if with_uplift:
        uplift_7d = _base_uplift_report(7)
        uplift_7d["model_path"] = str(model_path)
        uplift_7d["model_version"] = model_version
        uplift_30d = _base_uplift_report(30)
        uplift_30d["model_path"] = str(model_path)
        uplift_30d["model_version"] = model_version
        _write_json(artifact_dir / "uplift_report_7d.json", uplift_7d)
        _write_json(artifact_dir / "uplift_report_30d.json", uplift_30d)
    return model_path


def _create_planner_artifact(
    root: Path,
    name: str,
    *,
    with_shadow: bool,
    model_version: str,
) -> Path:
    artifact_dir = root / name
    artifact_dir.mkdir(parents=True, exist_ok=True)
    model_path = artifact_dir / "model.pkl"
    model_path.write_bytes(b"placeholder-model")
    _write_json(
        artifact_dir / "metadata.json",
        {
            "model_version": model_version,
            "task": "roadmap_planner_v1_ranking",
            "selected_feature_set": "full",
            "planner_guard": {
                "metric": "ndcg_at_5",
                "required_delta_vs_popularity": 0.0,
                "model_value": 0.8,
                "popularity_value": 0.6,
                "passed": True,
            },
        },
    )
    _write_json(
        artifact_dir / "eval_report.json",
        {
            "model_version": model_version,
            "metrics_test": {"ndcg_at_5": 0.8},
            "dataset_baselines": {
                "splits": {
                    "test": {
                        "popularity": {"ndcg_at_5": 0.6}
                    }
                }
            },
        },
    )
    if with_shadow:
        _write_json(
            artifact_dir / "shadow_report.json",
            {
                "model_version": model_version,
                "overall": {"eligible_plans": 10},
            },
        )
    return model_path


class RoadmapMLArtifactQualificationTests(SimpleTestCase):
    def test_nextstep_guard_uses_artifact_local_eval_not_stale_top_level_fallback(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            model_path = _create_nextstep_artifact(
                root,
                "active_artifact",
                with_eval=False,
                with_uplift=True,
                model_version="artifact_a",
            )
            stale_eval_path = root / "reports" / "roadmap_nextstep_v4_eval.json"
            _write_json(stale_eval_path, _base_eval_report("stale_top_level"))
            with override_settings(
                ROADMAP_NEXTSTEP_V4_MODEL_PATH=str(model_path),
                ROADMAP_NEXTSTEP_V4_EVAL_PATH=str(stale_eval_path),
            ):
                guard = v4_min_lift_guard_status(str(model_path))
            self.assertFalse(guard["passed"])
            self.assertEqual(guard["reason"], "missing_eval_report")
            self.assertEqual(
                Path(str(guard["eval_path"])).resolve(),
                (model_path.parent / "eval_report.json").resolve(),
            )

    def test_missing_local_eval_produces_hold_not_false_pass(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            model_path = _create_nextstep_artifact(
                root,
                "active_artifact",
                with_eval=False,
                with_uplift=True,
                model_version="artifact_a",
            )
            with override_settings(
                ROADMAP_NEXTSTEP_V4_ENABLED=True,
                ROADMAP_NEXTSTEP_V4_MODEL_PATH=str(model_path),
                ROADMAP_NEXTSTEP_V4_ENABLED_CATEGORIES=["skincare", "haircare", "makeup"],
                ROADMAP_NEXTSTEP_V4_DISABLED_CATEGORIES=["fragrance"],
            ):
                manifest = nextstep_pass_fail_manifest(str(model_path))
            self.assertEqual(manifest["by_category"]["skincare"]["status"], "HOLD")
            self.assertEqual(manifest["by_category"]["skincare"]["reason"], "missing_eval_report")
            self.assertEqual(manifest["by_category"]["fragrance"]["status"], "DISABLE")

    def test_pass_fail_manifest_reflects_exact_configured_model_path(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            active_model_path = _create_nextstep_artifact(
                root,
                "active_artifact",
                with_eval=False,
                with_uplift=True,
                model_version="active_model",
            )
            shadow_model_path = _create_nextstep_artifact(
                root,
                "shadow_artifact",
                with_eval=True,
                with_uplift=False,
                model_version="shadow_model",
            )
            planner_model_path = _create_planner_artifact(
                root,
                "planner_artifact",
                with_shadow=False,
                model_version="planner_model",
            )
            with override_settings(
                ROADMAP_PLANNER_V1_MODEL_PATH=str(planner_model_path),
                ROADMAP_PLANNER_V1_MODE="serve",
                ROADMAP_PLANNER_V1_ENABLED_CATEGORIES=["makeup"],
                ROADMAP_NEXTSTEP_V4_ENABLED=True,
                ROADMAP_NEXTSTEP_V4_MODEL_PATH=str(active_model_path),
                ROADMAP_NEXTSTEP_V4_SHADOW_MODEL_PATH=str(shadow_model_path),
                ROADMAP_NEXTSTEP_V4_ENABLED_CATEGORIES=["skincare", "haircare", "makeup"],
                ROADMAP_NEXTSTEP_V4_DISABLED_CATEGORIES=["fragrance"],
                ROADMAP_NEXTSTEP_V4_HAIRCARE_CORECHAIN_TEACHER_RERANK_ENABLED=False,
                ROADMAP_NEXTSTEP_V4_HAIRCARE_SCALP_TEACHER_RERANK_ENABLED=False,
            ), patch(
                "roadmap_app.ml_artifact_qualification.active_fragrance_runtime_integrity_counts",
                return_value={"active_fragrance_slot_mismatch_count": 0},
            ), patch(
                "roadmap_app.ml_artifact_qualification.legacy_bad_fragrance_completion_details",
                return_value={"legacy_bucket": "clean"},
            ):
                payload = build_roadmap_ml_artifact_qualification_payload()
            configured = {row["key"]: row for row in payload["configured_artifacts"]}
            self.assertEqual(configured["nextstep_v4_active"]["model_path"], str(active_model_path))
            self.assertEqual(configured["nextstep_v4_shadow"]["model_path"], str(shadow_model_path))
            self.assertEqual(
                payload["per_category_manifest"]["nextstep_v4"]["model_path"],
                str(active_model_path),
            )

    def test_planner_requires_local_shadow_report_for_runtime_guard(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            planner_model_path = _create_planner_artifact(
                root,
                "planner_artifact",
                with_shadow=False,
                model_version="planner_model",
            )
            with override_settings(
                ROADMAP_PLANNER_V1_MODEL_PATH=str(planner_model_path),
                ROADMAP_PLANNER_V1_ENABLED_CATEGORIES=["makeup"],
            ):
                guard = planner_runtime_guard_status(
                    "makeup",
                    model_path=str(planner_model_path),
                    require_shadow_report=True,
                )
            self.assertFalse(guard["passed"])
            self.assertEqual(guard["reason"], "missing_shadow_report")

    def test_partial_candidate_rollout_uses_its_own_uplift_bundle(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            active_model_path = _create_nextstep_artifact(
                root,
                "active_artifact",
                with_eval=True,
                with_uplift=True,
                model_version="active_model",
            )
            partial_model_path = _create_nextstep_artifact(
                root,
                "partial_artifact",
                with_eval=True,
                with_uplift=False,
                model_version="partial_model",
            )
            with override_settings(
                ROADMAP_NEXTSTEP_V4_ENABLED=True,
                ROADMAP_NEXTSTEP_V4_MODEL_PATH=str(active_model_path),
                ROADMAP_NEXTSTEP_V4_ENABLED_CATEGORIES=["skincare", "haircare", "makeup"],
                ROADMAP_NEXTSTEP_V4_DISABLED_CATEGORIES=["fragrance"],
            ):
                status = v4_category_staged_rollout_status("haircare", model_path=str(partial_model_path))
            self.assertEqual(status["final_status"], "HOLD")
            self.assertEqual(status["guard_7d"]["reason"], "insufficient_sample")
            self.assertEqual(
                Path(str(status["source_report_path_7d"])).resolve(),
                (partial_model_path.parent / "uplift_report_7d.json").resolve(),
            )

    def test_artifact_eval_rebuild_scores_exact_model_path(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifact_dir = root / "artifact"
            artifact_dir.mkdir(parents=True, exist_ok=True)
            model_path = artifact_dir / "model.pkl"
            joblib.dump(
                {
                    "task": "roadmap_nextstep_v4_ranking",
                    "model": ArtifactEvalDummyModel(),
                    "preprocessor": None,
                    "model_type": "lightgbm_ranker",
                    "feature_columns": ["candidate_popularity_in_train"],
                    "categorical_features": [],
                    "numeric_features": ["candidate_popularity_in_train"],
                    "temperature": 1.0,
                    "trained_at_utc": "2026-04-08T00:00:00Z",
                    "model_version": "exact_artifact",
                    "selected_feature_set": "baseline_only",
                },
                model_path,
            )

            data_dir = root / "data"
            data_dir.mkdir(parents=True, exist_ok=True)
            df = pd.DataFrame(
                [
                    {"user_id": 1, "episode_id": 101, "group_id": 101, "category": "haircare", "candidate_type": "shampoo", "candidate_popularity_in_train": 1.0, "y": 1},
                    {"user_id": 1, "episode_id": 101, "group_id": 101, "category": "haircare", "candidate_type": "conditioner", "candidate_popularity_in_train": 0.0, "y": 0},
                    {"user_id": 2, "episode_id": 201, "group_id": 201, "category": "haircare", "candidate_type": "shampoo", "candidate_popularity_in_train": 1.0, "y": 1},
                    {"user_id": 2, "episode_id": 201, "group_id": 201, "category": "haircare", "candidate_type": "conditioner", "candidate_popularity_in_train": 0.0, "y": 0},
                    {"user_id": 3, "episode_id": 301, "group_id": 301, "category": "haircare", "candidate_type": "shampoo", "candidate_popularity_in_train": 1.0, "y": 1},
                    {"user_id": 3, "episode_id": 301, "group_id": 301, "category": "haircare", "candidate_type": "conditioner", "candidate_popularity_in_train": 0.0, "y": 0},
                ]
            )
            df.to_parquet(data_dir / "dataset.parquet", index=False)
            _write_json(
                data_dir / "splits.json",
                {
                    "train_user_ids": [1],
                    "val_user_ids": [2],
                    "test_user_ids": [3],
                },
            )
            _write_json(
                data_dir / "metadata.json",
                {
                    "feature_columns": ["candidate_popularity_in_train"],
                    "categorical_features": [],
                    "numeric_features": ["candidate_popularity_in_train"],
                    "baselines": {
                        "splits": {
                            "val": {"popularity": {"ndcg_at_5": 1.0, "recall_at_1": 1.0, "recall_at_3": 1.0, "recall_at_5": 1.0}},
                            "test": {"popularity": {"ndcg_at_5": 1.0, "recall_at_1": 1.0, "recall_at_3": 1.0, "recall_at_5": 1.0}},
                        }
                    },
                },
            )
            _write_json(
                artifact_dir / "metadata.json",
                {
                    "model_version": "exact_artifact",
                    "dataset_path": str((data_dir / "dataset.parquet").resolve()),
                    "estimator": "lightgbm",
                    "selected_feature_set": "baseline_only",
                },
            )

            report = build_nextstep_v4_artifact_eval_report(model_path=str(model_path))
            self.assertEqual(report["model_path"], str(model_path.resolve()))
            self.assertEqual(report["model_version"], "exact_artifact")
            self.assertEqual(report["metrics_test"]["recall_at_1"], 1.0)
            self.assertIn("runtime_guard", report)


class RoadmapShadowEvidenceReplayTests(TestCase):
    def _create_plan(self, *, username: str, category: str, meta: dict) -> RoadmapPlan:
        user = get_user_model().objects.create_user(username=username, password="testpass123")
        plan = RoadmapPlan.objects.create(
            user=user,
            category=category,
            is_active=True,
            meta=meta,
        )
        RoadmapStep.objects.create(
            plan=plan,
            step_index=1,
            product_type="shampoo",
            status=RoadmapStep.Status.RECOMMENDED,
        )
        RoadmapStep.objects.create(
            plan=plan,
            step_index=2,
            product_type="conditioner",
            status=RoadmapStep.Status.MISSING,
        )
        return plan

    @override_settings(
        ROADMAP_RUNTIME_FREEZE_ML=True,
        ROADMAP_NEXTSTEP_V4_CONFIDENCE_THRESHOLD=0.35,
    )
    def test_backfill_persists_exact_shadow_evidence_for_active_model_path_without_enabling_serve(self):
        with TemporaryDirectory() as tmp:
            model_path = str((Path(tmp) / "roadmap_next_step_v4" / "model.pkl").resolve())
            plan = self._create_plan(
                username="shadow_case_1",
                category="haircare",
                meta={
                    "ml": {
                        "decision": "disabled",
                        "disabled_reason": "roadmap_ml_frozen",
                        "mode": "legacy",
                        "model_path": model_path,
                    },
                    "context": {
                        "post_ctx_product_ids": [101, 202],
                    },
                },
            )
            out = StringIO()
            with override_settings(ROADMAP_NEXTSTEP_V4_SHADOW_MODEL_PATH=model_path), patch(
                "roadmap_app.management.commands.backfill_roadmap_shadow_meta.nextstep_model_artifact_summary",
                return_value={
                    "exists": True,
                    "model_version": "shadow_exact_v1",
                    "selected_feature_set": "baseline_only",
                },
            ), patch(
                "roadmap_app.management.commands.backfill_roadmap_shadow_meta.predict_next_product_types_for_model_path",
                return_value=[
                    {
                        "product_type": "shampoo",
                        "score": 0.91,
                        "runtime_policies": ["haircare_guard"],
                    },
                    {
                        "product_type": "conditioner",
                        "score": 0.11,
                    },
                ],
            ):
                call_command(
                    "backfill_roadmap_shadow_meta",
                    "--days",
                    "30",
                    "--category",
                    "all",
                    "--model-path",
                    model_path,
                    "--write",
                    stdout=out,
                )
            plan.refresh_from_db()
            ml = dict(plan.meta.get("ml") or {})
            self.assertEqual(ml.get("decision"), "disabled")
            evidence = dict((ml.get("shadow_evidence") or {}).get(normalized_model_path(model_path)) or {})
            control_evidence = dict((ml.get("baseline_control_evidence") or {}).get(normalized_model_path(model_path)) or {})
            self.assertTrue(evidence)
            self.assertTrue(control_evidence)
            self.assertTrue(evidence.get("was_model_considered"))
            self.assertTrue(evidence.get("was_model_selected"))
            self.assertEqual(evidence.get("comparable_decision"), "model_used")
            self.assertEqual(evidence.get("model_version"), "shadow_exact_v1")
            self.assertTrue(control_evidence.get("was_control_available"))
            self.assertTrue(control_evidence.get("was_control_selected"))
            self.assertEqual(control_evidence.get("comparable_decision"), "control_used")
            self.assertEqual(control_evidence.get("selected_product_type"), "shampoo")
            self.assertIn("plans updated: `1`", out.getvalue())

    @override_settings(
        ROADMAP_RUNTIME_FREEZE_ML=True,
        ROADMAP_NEXTSTEP_V4_CONFIDENCE_THRESHOLD=0.35,
    )
    def test_shadow_backfill_is_idempotent(self):
        with TemporaryDirectory() as tmp:
            model_path = str((Path(tmp) / "roadmap_next_step_v4" / "model.pkl").resolve())
            plan = self._create_plan(
                username="shadow_case_2",
                category="haircare",
                meta={
                    "ml": {
                        "decision": "disabled",
                        "disabled_reason": "roadmap_ml_frozen",
                        "mode": "legacy",
                    }
                },
            )
            patch_summary = patch(
                "roadmap_app.management.commands.backfill_roadmap_shadow_meta.nextstep_model_artifact_summary",
                return_value={
                    "exists": True,
                    "model_version": "shadow_exact_v1",
                    "selected_feature_set": "baseline_only",
                },
            )
            patch_predict = patch(
                "roadmap_app.management.commands.backfill_roadmap_shadow_meta.predict_next_product_types_for_model_path",
                return_value=[
                    {
                        "product_type": "shampoo",
                        "score": 0.91,
                    }
                ],
            )
            with override_settings(ROADMAP_NEXTSTEP_V4_SHADOW_MODEL_PATH=model_path), patch_summary, patch_predict:
                first_out = StringIO()
                call_command(
                    "backfill_roadmap_shadow_meta",
                    "--days",
                    "30",
                    "--model-path",
                    model_path,
                    "--write",
                    stdout=first_out,
                )
                plan.refresh_from_db()
                meta_after_first = json.loads(json.dumps(plan.meta))

                second_out = StringIO()
                call_command(
                    "backfill_roadmap_shadow_meta",
                    "--days",
                    "30",
                    "--model-path",
                    model_path,
                    "--write",
                    stdout=second_out,
                )

            plan.refresh_from_db()
            self.assertEqual(plan.meta, meta_after_first)
            self.assertIn("plans updated: `0`", second_out.getvalue())
            self.assertIn("- already_up_to_date: `1`", second_out.getvalue())

    def test_shadow_replay_uplift_counts_exact_model_path(self):
        with TemporaryDirectory() as tmp:
            model_path = str((Path(tmp) / "roadmap_next_step_v4" / "model.pkl").resolve())
            normalized_path = normalized_model_path(model_path)
            user_model = get_user_model()
            user_a = user_model.objects.create_user(username="uplift_shadow_a", password="testpass123")
            user_b = user_model.objects.create_user(username="uplift_shadow_b", password="testpass123")
            RoadmapPlan.objects.create(
                user=user_a,
                category="haircare",
                is_active=True,
                meta={
                    "ml": {
                        "decision": "disabled",
                        "shadow_evidence": {
                            normalized_path: {
                                "model_path": normalized_path,
                                "model_version": "shadow_exact_v1",
                                "was_model_considered": True,
                                "was_model_selected": True,
                                "comparable_decision": "model_used",
                                "comparable_reason": "selected_top1",
                            }
                        },
                        "baseline_control_evidence": {
                            normalized_path: {
                                "model_path": normalized_path,
                                "baseline_source": "current_rule_plan",
                                "was_control_available": True,
                                "was_control_selected": True,
                                "comparable_decision": "control_used",
                                "comparable_reason": "selected_current_plan_next_step",
                                "selected_product_type": "shampoo",
                            }
                        },
                    }
                },
            )
            RoadmapPlan.objects.create(
                user=user_b,
                category="haircare",
                is_active=True,
                meta={
                    "ml": {
                        "decision": "disabled",
                        "shadow_evidence": {
                            normalized_path: {
                                "model_path": normalized_path,
                                "model_version": "shadow_exact_v1",
                                "was_model_considered": True,
                                "was_model_selected": False,
                                "comparable_decision": "fallback",
                                "comparable_reason": "low_confidence",
                            }
                        },
                        "baseline_control_evidence": {
                            normalized_path: {
                                "model_path": normalized_path,
                                "baseline_source": "current_rule_plan",
                                "was_control_available": True,
                                "was_control_selected": True,
                                "comparable_decision": "control_used",
                                "comparable_reason": "selected_current_plan_next_step",
                                "selected_product_type": "conditioner",
                            }
                        },
                    }
                },
            )
            out_stem = Path(tmp) / "shadow_replay_uplift"
            call_command(
                "report_roadmap_ml_uplift",
                "--days",
                "30",
                "--category",
                "all",
                "--format",
                "json",
                "--evidence-source",
                "shadow_replay",
                "--model-path",
                model_path,
                "--min-plans",
                "1",
                "--out",
                str(out_stem),
            )
            payload = json.loads(out_stem.with_suffix(".json").read_text(encoding="utf-8"))
            self.assertEqual(payload["params"]["evidence_source"], "shadow_replay")
            self.assertEqual(payload["model_path"], normalized_path)
            self.assertEqual(payload["overall"]["model_used_plans_total"], 1)
            self.assertEqual(payload["overall"]["control_plans_total"], 1)
            self.assertEqual(
                payload["runtime_observability"]["comparability"]["comparable_anchor_plans_total"],
                1,
            )
            self.assertEqual(
                payload["runtime_observability"]["comparability"]["excluded_reasons"]["low_confidence"],
                1,
            )
            self.assertEqual(
                payload["runtime_observability"]["decision_counts"]["model_used"],
                1,
            )
            self.assertEqual(
                payload["runtime_observability"]["decision_counts"]["fallback"],
                1,
            )


class RoadmapHistoricalAnchorReplayTests(TestCase):
    def _create_historical_anchor_plan(self, *, username: str) -> tuple[RoadmapPlan, RoadmapStep, RoadmapStep, str]:
        user = get_user_model().objects.create_user(username=username, password="testpass123")
        model_path = str((Path.cwd() / "tmp" / f"{username}_artifact" / "model.pkl").resolve())
        plan = RoadmapPlan.objects.create(
            user=user,
            category="haircare",
            is_active=True,
            meta={
                "ml": {
                    "decision": "disabled",
                    "disabled_reason": "roadmap_ml_frozen",
                    "mode": "legacy",
                }
            },
        )
        step1 = RoadmapStep.objects.create(
            plan=plan,
            step_index=1,
            product_type="shampoo",
            status=RoadmapStep.Status.COMPLETED,
        )
        step2 = RoadmapStep.objects.create(
            plan=plan,
            step_index=2,
            product_type="conditioner",
            status=RoadmapStep.Status.COMPLETED,
        )

        refresh_time = timezone.now() - timedelta(days=1)
        refresh = RoadmapEvent.objects.create(
            user=user,
            plan=plan,
            event_type=RoadmapEvent.Type.PLAN_REFRESHED,
            created_at=refresh_time,
            context={
                "plan_id": plan.id,
                "category": "haircare",
                "next_step_id": step1.id,
                "next_step_index": 1,
                "next_product_type": "shampoo",
                "ml": {"decision": "disabled"},
            },
        )
        RoadmapEvent.objects.create(
            user=user,
            plan=plan,
            step=step1,
            event_type=RoadmapEvent.Type.STEP_GENERATED,
            created_at=refresh_time + timedelta(seconds=5),
            context={
                "plan_id": plan.id,
                "step_id": step1.id,
                "step_index": 1,
                "category": "haircare",
                "product_type": "shampoo",
                "status": "recommended",
                "recommended_product_id": 1001,
                "has_recommendation": True,
                "ml": {"decision": "disabled"},
            },
        )
        RoadmapEvent.objects.create(
            user=user,
            plan=plan,
            step=step2,
            event_type=RoadmapEvent.Type.STEP_GENERATED,
            created_at=refresh_time + timedelta(seconds=6),
            context={
                "plan_id": plan.id,
                "step_id": step2.id,
                "step_index": 2,
                "category": "haircare",
                "product_type": "conditioner",
                "status": "missing",
                "recommended_product_id": 1002,
                "has_recommendation": True,
                "ml": {"decision": "disabled"},
            },
        )
        RoadmapEvent.objects.create(
            user=user,
            plan=plan,
            step=step1,
            event_type=RoadmapEvent.Type.STEP_EXPOSED,
            created_at=refresh_time + timedelta(minutes=5),
            context={
                "category": "haircare",
                "step_index": 1,
                "product_type": "shampoo",
                "recommended_product_id": 1001,
                "sources": ["roadmap_api"],
            },
        )
        RoadmapEvent.objects.create(
            user=user,
            plan=plan,
            step=step1,
            event_type=RoadmapEvent.Type.STEP_COMPLETED,
            created_at=refresh_time + timedelta(minutes=10),
            context={
                "category": "haircare",
                "step_index": 1,
                "product_type": "shampoo",
                "recommended_product_id": 1001,
                "matched_by": "product_type",
            },
        )
        return plan, step1, step2, model_path

    @override_settings(
        ROADMAP_RUNTIME_FREEZE_ML=True,
        ROADMAP_NEXTSTEP_V4_CONFIDENCE_THRESHOLD=0.35,
    )
    def test_historical_anchor_backfill_recovers_anchor_lost_by_current_snapshot_drift(self):
        plan, _, _, model_path = self._create_historical_anchor_plan(username="historical_replay_a")
        out = StringIO()
        with patch(
            "roadmap_app.management.commands.backfill_roadmap_shadow_meta.nextstep_model_artifact_summary",
            return_value={"exists": True, "model_version": "historical_v1", "selected_feature_set": "baseline_only"},
        ), patch(
            "roadmap_app.management.commands.backfill_roadmap_shadow_meta._load_model_for_path",
            return_value={"task": "roadmap_nextstep_v4_ranking"},
        ), patch(
            "roadmap_app.management.commands.backfill_roadmap_shadow_meta._predict_with_v4_artifact_from_sources",
            return_value=[{"product_type": "shampoo", "score": 0.91}],
        ):
            call_command(
                "backfill_roadmap_shadow_meta",
                "--days",
                "30",
                "--replay-mode",
                "historical_anchors",
                "--model-path",
                model_path,
                "--write",
                stdout=out,
            )
        plan.refresh_from_db()
        ml = dict(plan.meta.get("ml") or {})
        historical_shadow = dict((ml.get("historical_shadow_evidence") or {}).get(normalized_model_path(model_path)) or {})
        historical_control = dict((ml.get("historical_control_evidence") or {}).get(normalized_model_path(model_path)) or {})
        self.assertEqual(ml.get("decision"), "disabled")
        self.assertIn("plan_refresh:", next(iter(historical_shadow.keys())))
        anchor_key = next(iter(historical_shadow.keys()))
        self.assertEqual(historical_shadow[anchor_key]["model_path"], normalized_model_path(model_path))
        self.assertTrue(historical_shadow[anchor_key]["was_model_selected"])
        self.assertEqual(historical_shadow[anchor_key]["top1_product_type"], "shampoo")
        self.assertTrue(historical_control[anchor_key]["was_control_selected"])
        self.assertEqual(historical_control[anchor_key]["selected_product_type"], "shampoo")
        self.assertIn("- plans updated: `1`", out.getvalue())

    @override_settings(
        ROADMAP_RUNTIME_FREEZE_ML=True,
        ROADMAP_NEXTSTEP_V4_CONFIDENCE_THRESHOLD=0.35,
    )
    def test_historical_anchor_backfill_is_idempotent(self):
        plan, _, _, model_path = self._create_historical_anchor_plan(username="historical_replay_b")
        patch_summary = patch(
            "roadmap_app.management.commands.backfill_roadmap_shadow_meta.nextstep_model_artifact_summary",
            return_value={"exists": True, "model_version": "historical_v1", "selected_feature_set": "baseline_only"},
        )
        patch_load = patch(
            "roadmap_app.management.commands.backfill_roadmap_shadow_meta._load_model_for_path",
            return_value={"task": "roadmap_nextstep_v4_ranking"},
        )
        patch_predict = patch(
            "roadmap_app.management.commands.backfill_roadmap_shadow_meta._predict_with_v4_artifact_from_sources",
            return_value=[{"product_type": "shampoo", "score": 0.91}],
        )
        with patch_summary, patch_load, patch_predict:
            first_out = StringIO()
            call_command(
                "backfill_roadmap_shadow_meta",
                "--days",
                "30",
                "--replay-mode",
                "historical_anchors",
                "--model-path",
                model_path,
                "--write",
                stdout=first_out,
            )
            plan.refresh_from_db()
            meta_after_first = json.loads(json.dumps(plan.meta))
            second_out = StringIO()
            call_command(
                "backfill_roadmap_shadow_meta",
                "--days",
                "30",
                "--replay-mode",
                "historical_anchors",
                "--model-path",
                model_path,
                "--write",
                stdout=second_out,
            )
        plan.refresh_from_db()
        self.assertEqual(plan.meta, meta_after_first)
        self.assertIn("- plans updated: `0`", second_out.getvalue())

    def test_historical_replay_uplift_recovers_anchor_counts_from_historical_evidence(self):
        plan_a, _, _, model_path = self._create_historical_anchor_plan(username="historical_replay_c")
        plan_b, _, _, _ = self._create_historical_anchor_plan(username="historical_replay_d")
        normalized_path = normalized_model_path(model_path)

        def _anchor_key(plan: RoadmapPlan) -> str:
            refresh = RoadmapEvent.objects.filter(plan=plan, event_type=RoadmapEvent.Type.PLAN_REFRESHED).first()
            return f"plan_refresh:{refresh.id}"

        key_a = _anchor_key(plan_a)
        key_b = _anchor_key(plan_b)
        plan_a.meta = {
            "ml": {
                "decision": "disabled",
                "historical_shadow_evidence": {
                    normalized_path: {
                        key_a: {
                            "anchor_key": key_a,
                            "model_path": normalized_path,
                            "model_version": "historical_v1",
                            "was_model_considered": True,
                            "was_model_selected": True,
                            "comparable_decision": "model_used",
                            "comparable_reason": "selected_top1",
                        }
                    }
                },
                "historical_control_evidence": {
                    normalized_path: {
                        key_a: {
                            "anchor_key": key_a,
                            "model_path": normalized_path,
                            "was_control_available": True,
                            "was_control_selected": True,
                            "comparable_decision": "control_used",
                            "comparable_reason": "selected_historical_next_step",
                        }
                    }
                },
            }
        }
        plan_a.save(update_fields=["meta"])
        plan_b.meta = {
            "ml": {
                "decision": "disabled",
                "historical_shadow_evidence": {
                    normalized_path: {
                        key_b: {
                            "anchor_key": key_b,
                            "model_path": normalized_path,
                            "model_version": "historical_v1",
                            "was_model_considered": True,
                            "was_model_selected": False,
                            "comparable_decision": "fallback",
                            "comparable_reason": "low_confidence",
                        }
                    }
                },
                "historical_control_evidence": {
                    normalized_path: {
                        key_b: {
                            "anchor_key": key_b,
                            "model_path": normalized_path,
                            "was_control_available": True,
                            "was_control_selected": True,
                            "comparable_decision": "control_used",
                            "comparable_reason": "selected_historical_next_step",
                        }
                    }
                },
            }
        }
        plan_b.save(update_fields=["meta"])

        out_stem = Path.cwd() / "tmp" / "historical_replay_uplift"
        call_command(
            "report_roadmap_ml_uplift",
            "--days",
            "30",
            "--category",
            "all",
            "--format",
            "json",
            "--evidence-source",
            "historical_replay",
            "--model-path",
            model_path,
            "--min-plans",
            "1",
            "--out",
            str(out_stem),
        )
        payload = json.loads(out_stem.with_suffix(".json").read_text(encoding="utf-8"))
        self.assertEqual(payload["params"]["evidence_source"], "historical_replay")
        self.assertEqual(payload["model_path"], normalized_path)
        self.assertEqual(payload["overall"]["model_used_plans_total"], 1)
        self.assertEqual(payload["overall"]["control_plans_total"], 1)
        self.assertEqual(
            payload["runtime_observability"]["comparability"]["excluded_reasons"]["low_confidence"],
            1,
        )
        self.assertGreaterEqual(
            payload["runtime_observability"]["historical_reconstruction"]["anchors_recovered_historically"],
            1,
        )


class RoadmapNextstepDecisionQualityTests(TestCase):
    def _create_decision_quality_plan(
        self,
        *,
        username: str,
        baseline_type: str,
        model_type: str,
        completed_type: str | None,
        category: str = "haircare",
    ) -> tuple[RoadmapPlan, str, str]:
        user = get_user_model().objects.create_user(username=username, password="testpass123")
        model_path = str((Path.cwd() / "tmp" / f"{username}_dq_artifact" / "model.pkl").resolve())
        normalized_path = normalized_model_path(model_path)
        plan = RoadmapPlan.objects.create(
            user=user,
            category=category,
            is_active=True,
            meta={"ml": {"decision": "disabled", "disabled_reason": "roadmap_ml_frozen"}},
        )
        baseline_step = RoadmapStep.objects.create(
            plan=plan,
            step_index=1,
            product_type=baseline_type,
            status=RoadmapStep.Status.RECOMMENDED,
        )
        alt_step = RoadmapStep.objects.create(
            plan=plan,
            step_index=2,
            product_type=model_type,
            status=RoadmapStep.Status.MISSING,
        )

        refresh_time = timezone.now() - timedelta(days=1)
        refresh = RoadmapEvent.objects.create(
            user=user,
            plan=plan,
            event_type=RoadmapEvent.Type.PLAN_REFRESHED,
            created_at=refresh_time,
            context={
                "plan_id": plan.id,
                "category": category,
                "next_step_id": baseline_step.id,
                "next_step_index": 1,
                "next_product_type": baseline_type,
                "ml": {"decision": "disabled"},
            },
        )
        RoadmapEvent.objects.create(
            user=user,
            plan=plan,
            step=baseline_step,
            event_type=RoadmapEvent.Type.STEP_GENERATED,
            created_at=refresh_time + timedelta(seconds=5),
            context={
                "plan_id": plan.id,
                "step_id": baseline_step.id,
                "step_index": 1,
                "category": category,
                "product_type": baseline_type,
                "status": "recommended",
                "recommended_product_id": 1001,
                "has_recommendation": True,
            },
        )
        RoadmapEvent.objects.create(
            user=user,
            plan=plan,
            step=alt_step,
            event_type=RoadmapEvent.Type.STEP_GENERATED,
            created_at=refresh_time + timedelta(seconds=6),
            context={
                "plan_id": plan.id,
                "step_id": alt_step.id,
                "step_index": 2,
                "category": category,
                "product_type": model_type,
                "status": "missing",
                "recommended_product_id": 1002,
                "has_recommendation": True,
            },
        )
        RoadmapEvent.objects.create(
            user=user,
            plan=plan,
            step=baseline_step,
            event_type=RoadmapEvent.Type.STEP_EXPOSED,
            created_at=refresh_time + timedelta(minutes=1),
            context={
                "category": category,
                "step_index": 1,
                "product_type": baseline_type,
                "recommended_product_id": 1001,
                "sources": ["roadmap_api"],
            },
        )
        if completed_type:
            completed_step = baseline_step if completed_type == baseline_type else alt_step
            recommended_product_id = 1001 if completed_type == baseline_type else 1002
            RoadmapEvent.objects.create(
                user=user,
                plan=plan,
                step=completed_step,
                event_type=RoadmapEvent.Type.STEP_COMPLETED,
                created_at=refresh_time + timedelta(minutes=3),
                context={
                    "category": category,
                    "step_index": completed_step.step_index,
                    "product_type": completed_type,
                    "recommended_product_id": recommended_product_id,
                    "matched_by": "recommended_product_id",
                },
            )

        anchor_key = f"plan_refresh:{refresh.id}"
        plan.meta = {
            "ml": {
                "decision": "disabled",
                "historical_shadow_evidence": {
                    normalized_path: {
                        anchor_key: {
                            "anchor_key": anchor_key,
                            "model_path": normalized_path,
                            "model_version": "dq_v1",
                            "was_model_considered": True,
                            "was_model_selected": True,
                            "comparable_decision": "model_used",
                            "comparable_reason": "selected_top1",
                            "top1_product_type": model_type,
                        }
                    }
                },
                "historical_control_evidence": {
                    normalized_path: {
                        anchor_key: {
                            "anchor_key": anchor_key,
                            "model_path": normalized_path,
                            "was_control_available": True,
                            "was_control_selected": True,
                            "comparable_decision": "control_used",
                            "comparable_reason": "selected_historical_next_step",
                            "selected_product_type": baseline_type,
                        }
                    }
                },
            }
        }
        plan.save(update_fields=["meta"])
        return plan, model_path, normalized_path

    def test_decision_quality_uses_first_completed_generated_candidate_not_baseline_selected_step(self):
        _, model_path, normalized_path = self._create_decision_quality_plan(
            username="dq_case_model_win",
            baseline_type="shampoo",
            model_type="hair_mask",
            completed_type="hair_mask",
            category="haircare",
        )

        payload = build_nextstep_v4_decision_quality_payload(
            model_path=model_path,
            days=30,
            category="all",
            min_slice_size=1,
        )

        self.assertEqual(payload["model_path"], normalized_path)
        haircare = payload["per_category"]["haircare"]
        self.assertEqual(haircare["model_wins_total"], 1)
        self.assertEqual(haircare["baseline_wins_total"], 0)
        self.assertEqual(haircare["truth_matched_by"]["recommended_product_id"], 1)
        promising_pairs = haircare["promising_disagreement_pairs"]
        self.assertEqual(promising_pairs[0]["baseline_product_type"], "shampoo")
        self.assertEqual(promising_pairs[0]["model_product_type"], "hair_mask")

    def test_decision_quality_marks_no_completion_window_as_unresolved(self):
        _, model_path, _ = self._create_decision_quality_plan(
            username="dq_case_unresolved",
            baseline_type="cleanser",
            model_type="serum",
            completed_type=None,
            category="skincare",
        )

        payload = build_nextstep_v4_decision_quality_payload(
            model_path=model_path,
            days=30,
            category="all",
            min_slice_size=1,
        )

        skincare = payload["per_category"]["skincare"]
        self.assertEqual(skincare["resolved_truth_anchors_total"], 0)
        self.assertEqual(skincare["unresolved_truth_anchors_total"], 1)
        self.assertEqual(
            skincare["unresolved_reasons"]["no_completed_truth_in_window_exposed_no_completion"],
            1,
        )

    def test_decision_quality_is_read_only_for_runtime_meta(self):
        plan, model_path, _ = self._create_decision_quality_plan(
            username="dq_case_read_only",
            baseline_type="blush",
            model_type="foundation",
            completed_type="blush",
            category="makeup",
        )
        meta_before = json.loads(json.dumps(plan.meta))

        _ = build_nextstep_v4_decision_quality_payload(
            model_path=model_path,
            days=30,
            category="all",
            min_slice_size=1,
        )

        plan.refresh_from_db()
        self.assertEqual(plan.meta, meta_before)
        self.assertEqual((plan.meta.get("ml") or {}).get("decision"), "disabled")


class RoadmapNextstepTargetedRetrainTests(SimpleTestCase):
    def test_targeted_retrain_weights_apply_target_and_protect_rules_without_touching_fragrance(self):
        df = pd.DataFrame(
            [
                {"category": "skincare", "label": "mask", "candidate_type": "mask", "y": 1},
                {"category": "skincare", "label": "mask", "candidate_type": "essence", "y": 0},
                {"category": "haircare", "label": "hair_mask", "candidate_type": "hair_mask", "y": 1},
                {"category": "fragrance", "label": "cold_evening", "candidate_type": "cold_evening", "y": 1},
            ]
        )
        weighted, summary = apply_targeted_retrain_weights(df)
        self.assertEqual(float(weighted.loc[0, "sample_weight"]), 1.8)
        self.assertEqual(float(weighted.loc[1, "sample_weight"]), 1.45)
        self.assertEqual(float(weighted.loc[2, "sample_weight"]), 1.15)
        self.assertEqual(float(weighted.loc[3, "sample_weight"]), 1.0)
        self.assertEqual(summary["rows_reweighted_total"], 3)
        self.assertEqual(summary["bucket_distribution"]["default"], 1)

    def test_targeted_retrain_comparison_payload_checks_candidate_proof_and_slice_deltas(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            base_dir = root / "base"
            candidate_dir = root / "candidate"
            for artifact_dir, version in [(base_dir, "base_v1"), (candidate_dir, "candidate_v1")]:
                artifact_dir.mkdir(parents=True, exist_ok=True)
                model_path = artifact_dir / "model.pkl"
                model_path.write_bytes(b"placeholder")
                _write_json(artifact_dir / "metadata.json", {"model_version": version})
                _write_json(artifact_dir / "eval_report.json", {"model_version": version, "metrics_test": {"recall_at_1": 0.2, "ndcg_at_5": 0.5}})
                _write_json(artifact_dir / "shadow_report.json", {"model_version": version, "model_path": str(model_path)})
                _write_json(artifact_dir / "uplift_report_7d.json", {"model_version": version, "model_path": str(model_path)})
                _write_json(artifact_dir / "uplift_report_30d.json", {"model_version": version, "model_path": str(model_path)})

            base_payload = {
                "per_category": {
                    "haircare": {
                        "rollout_reason": "low_uplift",
                        "diagnosis": {"code": "C"},
                        "model_win_rate_vs_truth": 0.10,
                        "baseline_win_rate_vs_truth": 0.20,
                        "both_wrong_rate": 0.30,
                        "resolved_truth_anchors_total": 100,
                    },
                    "skincare": {
                        "rollout_reason": "low_uplift",
                        "diagnosis": {"code": "B"},
                        "model_win_rate_vs_truth": 0.08,
                        "baseline_win_rate_vs_truth": 0.22,
                        "both_wrong_rate": 0.15,
                        "resolved_truth_anchors_total": 100,
                    },
                    "makeup": {
                        "rollout_reason": "sample_too_small_but_nonzero_control",
                        "diagnosis": {"code": "A"},
                        "model_win_rate_vs_truth": 0.0,
                        "baseline_win_rate_vs_truth": 0.0,
                        "both_wrong_rate": 0.0,
                        "resolved_truth_anchors_total": 20,
                    },
                    "fragrance": {
                        "rollout_reason": "category_disabled",
                        "diagnosis": {"code": "C"},
                        "model_win_rate_vs_truth": 0.2,
                        "baseline_win_rate_vs_truth": 0.3,
                        "both_wrong_rate": 0.2,
                        "resolved_truth_anchors_total": 20,
                    },
                },
                "slice_analysis": {
                    "truth_slice_lookup": {
                        "haircare:shampoo": {"model_win_rate_vs_truth": 0.0, "baseline_win_rate_vs_truth": 1.0, "net_wins_model_minus_baseline": -11},
                        "haircare:hair_mask": {"model_win_rate_vs_truth": 0.3, "baseline_win_rate_vs_truth": 0.0, "net_wins_model_minus_baseline": 10},
                    },
                    "disagreement_pair_lookup": {
                        "haircare:shampoo:conditioner": {"model_win_rate_vs_truth": 0.0, "baseline_win_rate_vs_truth": 1.0, "net_wins_model_minus_baseline": -11},
                    },
                },
            }
            candidate_payload = {
                "per_category": {
                    "haircare": {
                        "rollout_reason": "low_uplift",
                        "diagnosis": {"code": "C"},
                        "model_win_rate_vs_truth": 0.16,
                        "baseline_win_rate_vs_truth": 0.12,
                        "both_wrong_rate": 0.24,
                        "resolved_truth_anchors_total": 100,
                    },
                    "skincare": {
                        "rollout_reason": "low_uplift",
                        "diagnosis": {"code": "C"},
                        "model_win_rate_vs_truth": 0.11,
                        "baseline_win_rate_vs_truth": 0.18,
                        "both_wrong_rate": 0.14,
                        "resolved_truth_anchors_total": 100,
                    },
                    "makeup": {
                        "rollout_reason": "sample_too_small_but_nonzero_control",
                        "diagnosis": {"code": "A"},
                        "model_win_rate_vs_truth": 0.0,
                        "baseline_win_rate_vs_truth": 0.0,
                        "both_wrong_rate": 0.0,
                        "resolved_truth_anchors_total": 20,
                    },
                    "fragrance": {
                        "rollout_reason": "category_disabled",
                        "diagnosis": {"code": "C"},
                        "model_win_rate_vs_truth": 0.2,
                        "baseline_win_rate_vs_truth": 0.28,
                        "both_wrong_rate": 0.2,
                        "resolved_truth_anchors_total": 20,
                    },
                },
                "slice_analysis": {
                    "truth_slice_lookup": {
                        "haircare:shampoo": {"model_win_rate_vs_truth": 0.2, "baseline_win_rate_vs_truth": 0.8, "net_wins_model_minus_baseline": -6},
                        "haircare:hair_mask": {"model_win_rate_vs_truth": 0.28, "baseline_win_rate_vs_truth": 0.0, "net_wins_model_minus_baseline": 9},
                    },
                    "disagreement_pair_lookup": {
                        "haircare:shampoo:conditioner": {"model_win_rate_vs_truth": 0.2, "baseline_win_rate_vs_truth": 0.8, "net_wins_model_minus_baseline": -6},
                    },
                },
            }

            with patch(
                "roadmap_app.nextstep_targeted_retrain.build_nextstep_v4_decision_quality_payload",
                side_effect=[base_payload, candidate_payload],
            ):
                payload = build_targeted_retrain_comparison_payload(
                    base_model_path=str(base_dir / "model.pkl"),
                    candidate_model_path=str(candidate_dir / "model.pkl"),
                    days=30,
                )

            self.assertTrue(payload["candidate_proof_bundle"]["required_complete"])
            haircare = next(row for row in payload["category_comparison"] if row["category"] == "haircare")
            self.assertEqual(haircare["delta_model_win_rate"], 0.06)
            targeted_pair = payload["targeted_disagreement_pairs"][0]
            self.assertEqual(targeted_pair["base"]["net_wins_model_minus_baseline"], -11)
            self.assertEqual(targeted_pair["candidate"]["net_wins_model_minus_baseline"], -6)
