from __future__ import annotations

import json
from datetime import timedelta
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import joblib
import pandas as pd
from django.conf import settings
from django.contrib.auth import get_user_model
from django.core.management import call_command
from django.db.utils import OperationalError
from django.test import SimpleTestCase, TestCase, override_settings
from django.utils import timezone

from roadmap_app.ml_artifact_qualification import (
    build_roadmap_ml_artifact_qualification_payload,
    freeze_candidate_promotion_manifest,
    nextstep_pass_fail_manifest,
)
from roadmap_app.nextstep_candidate_promotion import (
    build_v5_candidate_promotion_under_freeze_payload,
    render_v5_candidate_promotion_under_freeze_markdown,
)
from roadmap_app.nextstep_artifact_eval import build_nextstep_v4_artifact_eval_report
from roadmap_app.nextstep_decision_quality import build_nextstep_v4_decision_quality_payload
from roadmap_app.nextstep_historical_anchor_context import build_historical_anchor_read_context
from roadmap_app.nextstep_historical_anchor_dataset import (
    bucket_flags_for_row,
    classify_train_exclusion_reason,
    resolve_first_completed_generated_candidate,
)
from roadmap_app.nextstep_haircare_shampoo_gate import (
    _analyze_single_model_shampoo_gate,
    build_nextstep_haircare_shampoo_gate_payload,
)
from roadmap_app.nextstep_haircare_shampoo_truth_design import (
    build_nextstep_haircare_shampoo_truth_design_payload,
    evaluate_haircare_shampoo_truth_designs,
)
from roadmap_app.nextstep_skincare_freeze_qualification import (
    build_v5_skincare_freeze_qualification_payload,
    render_v5_skincare_freeze_qualification_markdown,
)
from roadmap_app.nextstep_targeted_retrain import (
    _slice_lookup,
    apply_targeted_retrain_weights,
    build_historical_anchor_candidate_comparison_payload,
    build_targeted_retrain_comparison_payload,
    materialize_historical_anchor_candidate_comparison_payload,
    render_historical_anchor_candidate_comparison_markdown,
)
from roadmap_app.ml_next_step import (
    v4_category_staged_rollout_status,
    v4_min_lift_guard_status,
)
from roadmap_app.ml_planner import planner_runtime_guard_status
from roadmap_app.models import RoadmapPlan, RoadmapStep
from roadmap_app.models import RoadmapEvent
from roadmap_app.models import RoadmapMLInvocation
from roadmap_app.shadow_evidence import (
    HISTORICAL_CONTROL_EVIDENCE_KEY,
    HISTORICAL_SHADOW_EVIDENCE_KEY,
    normalized_model_path,
)


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

    def test_artifact_qualification_payload_survives_fragrance_db_timeout(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            active_model_path = _create_nextstep_artifact(
                root,
                "active_artifact",
                with_eval=True,
                with_uplift=True,
                model_version="active_model",
            )
            with override_settings(
                ROADMAP_NEXTSTEP_V4_ENABLED=True,
                ROADMAP_NEXTSTEP_V4_MODEL_PATH=str(active_model_path),
                ROADMAP_NEXTSTEP_V4_ENABLED_CATEGORIES=["skincare", "haircare", "makeup"],
                ROADMAP_NEXTSTEP_V4_DISABLED_CATEGORIES=["fragrance"],
                ROADMAP_NEXTSTEP_V4_HAIRCARE_CORECHAIN_TEACHER_RERANK_ENABLED=False,
                ROADMAP_NEXTSTEP_V4_HAIRCARE_SCALP_TEACHER_RERANK_ENABLED=False,
            ), patch(
                "roadmap_app.ml_artifact_qualification.active_fragrance_runtime_integrity_counts",
                side_effect=OperationalError("connection timeout expired"),
            ), patch(
                "roadmap_app.ml_artifact_qualification.build_v5_candidate_promotion_under_freeze_payload",
                return_value={
                    "promotion_state": {
                        "active_runtime_continuation_artifact": {"model_path": str(active_model_path)},
                        "promoted_freeze_only_continuation_candidate": {"model_path": "models/v5.pkl"},
                        "runtime_serve": {"serve_enabled": False},
                    },
                    "executive_verdict": {
                        "canonical_freeze_candidate": True,
                        "runtime_still_frozen": True,
                        "active_runtime_artifact_unchanged": True,
                        "recommendation_code": "A",
                        "recommendation_label": "continue qualification with v5 as the new best candidate",
                    },
                    "provenance": {
                        "report_materialization": "materialized_from_saved_artifacts",
                        "source_of_truth": "cached_artifact",
                        "generated_from": "comparison_json",
                    },
                    "report_paths": {},
                    "read_only_guards": {
                        "catalog_writes_performed": False,
                        "runtime_config_changed": False,
                        "runtime_enablement_allowed": False,
                    },
                },
            ):
                payload = build_roadmap_ml_artifact_qualification_payload()

            fragrance = payload["fragrance_slot_qualification"]
            self.assertEqual(fragrance["source_of_truth"], "db_unavailable")
            self.assertEqual(fragrance["runtime"]["reason"], "db_unavailable")

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

    def test_decision_quality_reuses_provided_historical_context_without_extra_db_reads(self):
        _, model_path, _ = self._create_decision_quality_plan(
            username="dq_case_shared_context",
            baseline_type="shampoo",
            model_type="hair_mask",
            completed_type="hair_mask",
            category="haircare",
        )
        now_utc = timezone.now()
        historical_context = build_historical_anchor_read_context(
            since=now_utc - timedelta(days=30),
            until=now_utc,
            category="all",
            include_ga=False,
        )

        with patch(
            "roadmap_app.nextstep_historical_anchor_context.build_historical_continuation_anchor_records",
            side_effect=AssertionError("historical anchors should be reused from provided context"),
        ), patch(
            "roadmap_app.nextstep_historical_anchor_context.RoadmapPlan.objects.filter",
            side_effect=AssertionError("plan meta should be reused from provided context"),
        ), patch(
            "roadmap_app.nextstep_historical_anchor_context.completion_events_by_step",
            side_effect=AssertionError("step completions should be reused from provided context"),
        ):
            payload = build_nextstep_v4_decision_quality_payload(
                model_path=model_path,
                days=30,
                category="all",
                min_slice_size=1,
                historical_context=historical_context,
            )

        haircare = payload["per_category"]["haircare"]
        self.assertEqual(haircare["model_wins_total"], 1)
        self.assertEqual(haircare["baseline_wins_total"], 0)


class RoadmapNextstepTargetedRetrainTests(SimpleTestCase):
    def setUp(self):
        super().setUp()
        patcher = patch(
            "roadmap_app.nextstep_targeted_retrain.build_historical_anchor_read_context",
            return_value={
                "since": timezone.now() - timedelta(days=30),
                "until": timezone.now(),
                "category": "all",
                "include_ga": False,
                "anchors": [],
                "meta_by_plan": {},
                "completions_by_step": {},
                "read_only": True,
            },
        )
        patcher.start()
        self.addCleanup(patcher.stop)

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


class RoadmapNextstepHistoricalAnchorDatasetTests(SimpleTestCase):
    def test_resolve_first_completed_generated_candidate_prefers_first_completion_in_window(self):
        anchor = {
            "anchor_created_at": timezone.now(),
            "anchor_event_id": 100,
            "next_refresh_at": timezone.now() + timedelta(hours=1),
            "generated_candidates": [
                {"step_id": 11, "step_index": 1, "product_type": "mask"},
                {"step_id": 12, "step_index": 2, "product_type": "essence"},
            ],
        }
        completions_by_step = {
            11: [
                {
                    "id": 201,
                    "step_id": 11,
                    "created_at": anchor["anchor_created_at"] + timedelta(minutes=20),
                    "context": {"product_type": "mask", "matched_by": "recommended_product_id"},
                }
            ],
            12: [
                {
                    "id": 202,
                    "step_id": 12,
                    "created_at": anchor["anchor_created_at"] + timedelta(minutes=30),
                    "context": {"product_type": "essence", "matched_by": "recommended_product_id"},
                }
            ],
        }
        truth = resolve_first_completed_generated_candidate(anchor, completions_by_step=completions_by_step)
        self.assertTrue(truth["resolved"])
        self.assertEqual(truth["truth_selected_candidate_step_id"], 11)
        self.assertEqual(truth["truth_selected_product_type"], "mask")
        self.assertEqual(truth["truth_matched_by"], "recommended_product_id")

    def test_unresolved_anchor_reasons_are_explicit(self):
        anchor = {
            "anchor_has_actionable_step": 1,
            "anchor_next_step_id": 0,
            "next_refresh_at": timezone.now() + timedelta(hours=1),
            "reconstruction_reason": "",
        }
        truth = {"resolved": False, "reason": "no_completed_generated_candidate"}
        self.assertEqual(classify_train_exclusion_reason(anchor, truth), "missing_next_step_id")

        anchor_missing_actionable = {
            "anchor_has_actionable_step": 0,
            "anchor_next_step_id": None,
            "next_refresh_at": timezone.now() + timedelta(hours=1),
            "reconstruction_reason": "",
        }
        self.assertEqual(
            classify_train_exclusion_reason(anchor_missing_actionable, truth),
            "no_actionable_step",
        )

        anchor_incomplete = {
            "anchor_has_actionable_step": 1,
            "anchor_next_step_id": 10,
            "next_refresh_at": None,
            "reconstruction_reason": "",
        }
        self.assertEqual(
            classify_train_exclusion_reason(anchor_incomplete, truth),
            "incomplete_refresh_window",
        )

    def test_bucket_flags_cover_targeted_and_protected_rows(self):
        targeted = bucket_flags_for_row(
            category="haircare",
            truth_product_type="shampoo",
            candidate_product_type="conditioner",
        )
        self.assertEqual(targeted["bucket_haircare_shampoo_to_conditioner"], 1)
        self.assertEqual(targeted["bucket_haircare_shampoo"], 0)

        protected = bucket_flags_for_row(
            category="skincare",
            truth_product_type="essence",
            candidate_product_type="essence",
        )
        self.assertEqual(protected["protected_skincare_essence"], 1)
        self.assertEqual(protected["analysis_fragrance_cold_evening"], 0)


class RoadmapNextstepHistoricalAnchorTrainMetadataTests(SimpleTestCase):
    def test_train_command_propagates_dataset_source_and_truth_protocol(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_dir = root / "data"
            model_dir = root / "model"
            data_dir.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(
                [
                    {"user_id": 1, "episode_id": 1, "group_id": 1, "category": "haircare", "candidate_type": "shampoo", "candidate_popularity_in_train": 1.0, "y": 1},
                    {"user_id": 1, "episode_id": 1, "group_id": 1, "category": "haircare", "candidate_type": "conditioner", "candidate_popularity_in_train": 0.0, "y": 0},
                    {"user_id": 2, "episode_id": 2, "group_id": 2, "category": "haircare", "candidate_type": "shampoo", "candidate_popularity_in_train": 1.0, "y": 1},
                    {"user_id": 2, "episode_id": 2, "group_id": 2, "category": "haircare", "candidate_type": "conditioner", "candidate_popularity_in_train": 0.0, "y": 0},
                    {"user_id": 3, "episode_id": 3, "group_id": 3, "category": "haircare", "candidate_type": "shampoo", "candidate_popularity_in_train": 1.0, "y": 1},
                    {"user_id": 3, "episode_id": 3, "group_id": 3, "category": "haircare", "candidate_type": "conditioner", "candidate_popularity_in_train": 0.0, "y": 0},
                ]
            ).to_parquet(data_dir / "dataset.parquet", index=False)
            _write_json(
                data_dir / "splits.json",
                {"train_user_ids": [1], "val_user_ids": [2], "test_user_ids": [3]},
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
                    "dataset_builder_command": "build_roadmap_ml_dataset_v5_historical_anchor",
                    "source_dataset_dir": str(data_dir),
                    "source_dataset_file": str((data_dir / "dataset.parquet").resolve()),
                    "truth_protocol": {"positive_label": "first_completed_generated_candidate_in_same_refresh_window"},
                    "truth_resolution_summary": {"anchors_resolved_for_train": 3},
                },
            )

            call_command(
                "train_roadmap_nextstep_model_v4",
                "--data-dir",
                str(data_dir),
                "--model-dir",
                str(model_dir),
                "--model-version",
                "tiny_historical_anchor_model",
                "--estimator",
                "logistic",
                "--allow-fallback",
                "--trials",
                "1",
                "--negative-samples-per-episode",
                "1",
            )

            metadata = json.loads((model_dir / "metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["dataset_builder_command"], "build_roadmap_ml_dataset_v5_historical_anchor")
            self.assertEqual(metadata["source_dataset_dir"], str(data_dir))
            self.assertEqual(
                metadata["truth_protocol"]["positive_label"],
                "first_completed_generated_candidate_in_same_refresh_window",
            )
            self.assertEqual(metadata["truth_resolution_summary"]["anchors_resolved_for_train"], 3)


class RoadmapNextstepHistoricalAnchorComparisonTests(SimpleTestCase):
    def setUp(self):
        super().setUp()
        patcher = patch(
            "roadmap_app.nextstep_targeted_retrain.build_historical_anchor_read_context",
            return_value={
                "since": timezone.now() - timedelta(days=30),
                "until": timezone.now(),
                "category": "all",
                "include_ga": False,
                "anchors": [],
                "meta_by_plan": {},
                "completions_by_step": {},
                "read_only": True,
            },
        )
        patcher.start()
        self.addCleanup(patcher.stop)

    def _two_stage_truth_payload(
        self,
        *,
        stage1_model_win_rate: float,
        stage2_model_win_rate: float,
        resolved: int = 40,
        unresolved: int = 12,
        current_gate_status: str = "fail_closed_missing_truth",
    ) -> dict:
        stage1_model_wins = int(round(stage1_model_win_rate * resolved))
        stage2_model_wins = int(round(stage2_model_win_rate * resolved))
        return {
            "executive_verdict": {
                "recommended_truth_design": "D_two_stage_truth",
                "recommendation_status": "adopt_redesigned_truth_and_rerun_v5_comparison",
            },
            "candidate": {
                "truth_designs": {
                    "current_gate": {
                        "resolved_anchors_total": resolved,
                        "unresolved_anchors_total": unresolved,
                        "standalone_shampoo_truth_rows_total": 0,
                        "resolved_shampoo_conditioner_comparable_rows_total": 0,
                        "verdict": {"status": current_gate_status},
                    },
                    "designs": {
                        "D_two_stage_truth": {
                            "resolved_anchors_total": resolved,
                            "unresolved_anchors_total": unresolved,
                            "unresolved_anchors_by_reason": {
                                "incomplete_refresh_window": max(unresolved - 1, 0),
                                "no_completed_generated_candidate": 1 if unresolved > 0 else 0,
                            },
                            "model_vs_baseline_comparable": resolved > 0,
                            "gate_informativeness_status": "measurable_and_most_defensible",
                            "shampoo_conditioner_observability": {
                                "status": "observable_family_level_zero_positive_pair_rows",
                                "positive_truth_rows_total": 0,
                            },
                            "stage_1_family": {
                                "resolved_anchors_total": resolved,
                                "unresolved_anchors_total": unresolved,
                                "outcome_matrix": {
                                    "model_wins": stage1_model_wins,
                                    "baseline_wins": 0,
                                    "both_correct": 0,
                                    "both_wrong": 0,
                                    "model_win_rate": stage1_model_win_rate,
                                    "baseline_win_rate": 0.0,
                                },
                            },
                            "stage_2_concrete_step": {
                                "resolved_anchors_total": resolved,
                                "unresolved_anchors_total": unresolved,
                                "outcome_matrix": {
                                    "model_wins": stage2_model_wins,
                                    "baseline_wins": 0,
                                    "both_correct": 0,
                                    "both_wrong": 0,
                                    "model_win_rate": stage2_model_win_rate,
                                    "baseline_win_rate": 0.0,
                                },
                            },
                        }
                    },
                },
            },
            "catalog_safety": {"catalog_writes_performed": False},
        }

    def test_historical_anchor_acceptance_gates_fail_on_protected_regression(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            active_dir = root / "active"
            retrain_dir = root / "retrain"
            candidate_dir = root / "candidate"
            for artifact_dir, version in [
                (active_dir, "active_v1"),
                (retrain_dir, "retrain_v1"),
                (candidate_dir, "v5_v1"),
            ]:
                artifact_dir.mkdir(parents=True, exist_ok=True)
                model_path = artifact_dir / "model.pkl"
                model_path.write_bytes(b"placeholder")
                _write_json(artifact_dir / "metadata.json", {"model_version": version})
                _write_json(artifact_dir / "eval_report.json", {"model_version": version, "metrics_test": {"recall_at_1": 0.23, "ndcg_at_5": 0.52}})
                _write_json(artifact_dir / "shadow_report.json", {"model_version": version, "model_path": str(model_path)})
                _write_json(artifact_dir / "uplift_report_7d.json", {"model_version": version, "model_path": str(model_path)})
                _write_json(artifact_dir / "uplift_report_30d.json", {"model_version": version, "model_path": str(model_path)})

            active_payload = {
                "per_category": {
                    "haircare": {"rollout_reason": "low_uplift", "diagnosis": {"code": "C"}, "model_win_rate_vs_truth": 0.16, "baseline_win_rate_vs_truth": 0.02, "both_wrong_rate": 0.42, "resolved_truth_anchors_total": 120},
                    "skincare": {"rollout_reason": "low_uplift", "diagnosis": {"code": "B"}, "model_win_rate_vs_truth": 0.08, "baseline_win_rate_vs_truth": 0.25, "both_wrong_rate": 0.15, "resolved_truth_anchors_total": 300},
                    "makeup": {"rollout_reason": "sample_too_small_but_nonzero_control", "diagnosis": {"code": "A"}, "model_win_rate_vs_truth": 0.0, "baseline_win_rate_vs_truth": 0.0, "both_wrong_rate": 0.0, "resolved_truth_anchors_total": 20},
                    "fragrance": {"rollout_reason": "category_disabled", "diagnosis": {"code": "C"}, "model_win_rate_vs_truth": 0.20, "baseline_win_rate_vs_truth": 0.30, "both_wrong_rate": 0.20, "resolved_truth_anchors_total": 20},
                },
                "overall_enabled_categories": {"net_win_rate_model_minus_baseline": -0.11, "both_wrong_rate": 0.26},
                "slice_analysis": {
                    "truth_slice_lookup": {
                        "haircare:shampoo": {"model_win_rate_vs_truth": 0.0, "baseline_win_rate_vs_truth": 1.0, "net_wins_model_minus_baseline": -3},
                        "haircare:hair_mask": {"model_win_rate_vs_truth": 0.3, "baseline_win_rate_vs_truth": 0.0, "net_wins_model_minus_baseline": 10},
                        "haircare:hair_oil": {"model_win_rate_vs_truth": 0.4, "baseline_win_rate_vs_truth": 0.0, "net_wins_model_minus_baseline": 10},
                        "skincare:essence": {"model_win_rate_vs_truth": 0.3, "baseline_win_rate_vs_truth": 0.0, "net_wins_model_minus_baseline": 10},
                    },
                    "disagreement_pair_lookup": {
                        "haircare:shampoo:conditioner": {"model_win_rate_vs_truth": 0.0, "baseline_win_rate_vs_truth": 1.0, "net_wins_model_minus_baseline": -3},
                    },
                },
            }
            retrain_payload = active_payload
            candidate_payload = {
                "per_category": {
                    "haircare": {"rollout_reason": "low_uplift", "diagnosis": {"code": "C"}, "model_win_rate_vs_truth": 0.17, "baseline_win_rate_vs_truth": 0.01, "both_wrong_rate": 0.40, "resolved_truth_anchors_total": 120},
                    "skincare": {"rollout_reason": "low_uplift", "diagnosis": {"code": "B"}, "model_win_rate_vs_truth": 0.10, "baseline_win_rate_vs_truth": 0.20, "both_wrong_rate": 0.14, "resolved_truth_anchors_total": 300},
                    "makeup": {"rollout_reason": "sample_too_small_but_nonzero_control", "diagnosis": {"code": "A"}, "model_win_rate_vs_truth": 0.0, "baseline_win_rate_vs_truth": 0.0, "both_wrong_rate": 0.0, "resolved_truth_anchors_total": 20},
                    "fragrance": {"rollout_reason": "category_disabled", "diagnosis": {"code": "C"}, "model_win_rate_vs_truth": 0.20, "baseline_win_rate_vs_truth": 0.25, "both_wrong_rate": 0.20, "resolved_truth_anchors_total": 20},
                },
                "overall_enabled_categories": {"net_win_rate_model_minus_baseline": -0.10, "both_wrong_rate": 0.25},
                "slice_analysis": {
                    "truth_slice_lookup": {
                        "haircare:shampoo": {"model_win_rate_vs_truth": 0.2, "baseline_win_rate_vs_truth": 0.7, "net_wins_model_minus_baseline": -1},
                        "haircare:hair_mask": {"model_win_rate_vs_truth": 0.1, "baseline_win_rate_vs_truth": 0.0, "net_wins_model_minus_baseline": 4},
                        "haircare:hair_oil": {"model_win_rate_vs_truth": 0.4, "baseline_win_rate_vs_truth": 0.0, "net_wins_model_minus_baseline": 10},
                        "skincare:essence": {"model_win_rate_vs_truth": 0.2, "baseline_win_rate_vs_truth": 0.0, "net_wins_model_minus_baseline": 8},
                    },
                    "disagreement_pair_lookup": {
                        "haircare:shampoo:conditioner": {"model_win_rate_vs_truth": 0.2, "baseline_win_rate_vs_truth": 0.7, "net_wins_model_minus_baseline": -1},
                    },
                },
            }

            with patch(
                "roadmap_app.nextstep_targeted_retrain.build_nextstep_v4_decision_quality_payload",
                side_effect=[active_payload, retrain_payload, candidate_payload],
            ), patch(
                "roadmap_app.nextstep_targeted_retrain.build_nextstep_haircare_shampoo_truth_design_payload",
                side_effect=[
                    self._two_stage_truth_payload(stage1_model_win_rate=1.0, stage2_model_win_rate=0.5),
                    self._two_stage_truth_payload(stage1_model_win_rate=1.0, stage2_model_win_rate=0.5),
                    self._two_stage_truth_payload(stage1_model_win_rate=1.0, stage2_model_win_rate=1.0),
                ],
            ):
                payload = build_historical_anchor_candidate_comparison_payload(
                    active_model_path=str(active_dir / "model.pkl"),
                    retrain_v1_model_path=str(retrain_dir / "model.pkl"),
                    candidate_model_path=str(candidate_dir / "model.pkl"),
                    days=30,
                )

            self.assertFalse(payload["acceptance_gates"]["overall_passed"])
            gate_names = {gate["name"]: gate for gate in payload["acceptance_gates"]["gates"]}
            self.assertFalse(gate_names["protected_slices_non_regression"]["passed"])

    def test_slice_lookup_falls_back_to_per_category_lists(self):
        payload = {
            "per_category": {
                "haircare": {
                    "worst_slices": [
                        {
                            "category": "haircare",
                            "truth_product_type": "shampoo",
                            "net_wins_model_minus_baseline": -11,
                        }
                    ],
                    "worst_disagreement_pairs": [
                        {
                            "category": "haircare",
                            "baseline_product_type": "shampoo",
                            "model_product_type": "conditioner",
                            "net_wins_model_minus_baseline": -11,
                        }
                    ],
                }
            },
            "slice_analysis": {
                "truth_slice_lookup": {},
                "disagreement_pair_lookup": {},
            },
        }

        self.assertEqual(
            _slice_lookup(payload, kind="truth_slice_lookup", key="haircare:shampoo").get(
                "net_wins_model_minus_baseline"
            ),
            -11,
        )
        self.assertEqual(
            _slice_lookup(
                payload,
                kind="disagreement_pair_lookup",
                key="haircare:shampoo:conditioner",
            ).get("net_wins_model_minus_baseline"),
            -11,
        )

    def test_overall_gate_treats_zero_both_wrong_as_real_value(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            active_dir = root / "active"
            retrain_dir = root / "retrain"
            candidate_dir = root / "candidate"
            for artifact_dir, version in [
                (active_dir, "active_v1"),
                (retrain_dir, "retrain_v1"),
                (candidate_dir, "v5_v1"),
            ]:
                artifact_dir.mkdir(parents=True, exist_ok=True)
                model_path = artifact_dir / "model.pkl"
                model_path.write_bytes(b"placeholder")
                _write_json(artifact_dir / "metadata.json", {"model_version": version})
                _write_json(
                    artifact_dir / "eval_report.json",
                    {"model_version": version, "metrics_test": {"recall_at_1": 0.23, "ndcg_at_5": 0.52}},
                )
                _write_json(artifact_dir / "shadow_report.json", {"model_version": version, "model_path": str(model_path)})
                _write_json(artifact_dir / "uplift_report_7d.json", {"model_version": version, "model_path": str(model_path)})
                _write_json(artifact_dir / "uplift_report_30d.json", {"model_version": version, "model_path": str(model_path)})

            active_payload = {
                "per_category": {
                    "haircare": {"rollout_reason": "low_uplift", "diagnosis": {"code": "C"}, "model_win_rate_vs_truth": 0.16, "baseline_win_rate_vs_truth": 0.02, "both_wrong_rate": 0.42, "resolved_truth_anchors_total": 120},
                    "skincare": {"rollout_reason": "low_uplift", "diagnosis": {"code": "B"}, "model_win_rate_vs_truth": 0.08, "baseline_win_rate_vs_truth": 0.25, "both_wrong_rate": 0.15, "resolved_truth_anchors_total": 300},
                    "makeup": {"rollout_reason": "sample_too_small_but_nonzero_control", "diagnosis": {"code": "A"}, "model_win_rate_vs_truth": 0.0, "baseline_win_rate_vs_truth": 0.0, "both_wrong_rate": 0.0, "resolved_truth_anchors_total": 20},
                    "fragrance": {"rollout_reason": "category_disabled", "diagnosis": {"code": "C"}, "model_win_rate_vs_truth": 0.20, "baseline_win_rate_vs_truth": 0.30, "both_wrong_rate": 0.20, "resolved_truth_anchors_total": 20},
                },
                "overall_enabled_categories": {"net_win_rate_model_minus_baseline": -0.11, "both_wrong_rate": 0.26},
                "slice_analysis": {
                    "truth_slice_lookup": {
                        "haircare:shampoo": {"model_win_rate_vs_truth": 0.0, "baseline_win_rate_vs_truth": 1.0, "net_wins_model_minus_baseline": -3},
                        "haircare:hair_mask": {"model_win_rate_vs_truth": 0.3, "baseline_win_rate_vs_truth": 0.0, "net_wins_model_minus_baseline": 10},
                        "haircare:hair_oil": {"model_win_rate_vs_truth": 0.4, "baseline_win_rate_vs_truth": 0.0, "net_wins_model_minus_baseline": 10},
                        "skincare:essence": {"model_win_rate_vs_truth": 0.3, "baseline_win_rate_vs_truth": 0.0, "net_wins_model_minus_baseline": 10},
                    },
                    "disagreement_pair_lookup": {
                        "haircare:shampoo:conditioner": {"model_win_rate_vs_truth": 0.0, "baseline_win_rate_vs_truth": 1.0, "net_wins_model_minus_baseline": -3},
                    },
                },
            }
            retrain_payload = active_payload
            candidate_payload = {
                "per_category": {
                    "haircare": {"rollout_reason": "low_uplift", "diagnosis": {"code": "C"}, "model_win_rate_vs_truth": 0.30, "baseline_win_rate_vs_truth": 0.0, "both_wrong_rate": 0.10, "resolved_truth_anchors_total": 120},
                    "skincare": {"rollout_reason": "low_uplift", "diagnosis": {"code": "C"}, "model_win_rate_vs_truth": 0.20, "baseline_win_rate_vs_truth": 0.10, "both_wrong_rate": 0.05, "resolved_truth_anchors_total": 300},
                    "makeup": {"rollout_reason": "sample_too_small_but_nonzero_control", "diagnosis": {"code": "A"}, "model_win_rate_vs_truth": 0.0, "baseline_win_rate_vs_truth": 0.0, "both_wrong_rate": 0.0, "resolved_truth_anchors_total": 20},
                    "fragrance": {"rollout_reason": "category_disabled", "diagnosis": {"code": "C"}, "model_win_rate_vs_truth": 0.25, "baseline_win_rate_vs_truth": 0.10, "both_wrong_rate": 0.0, "resolved_truth_anchors_total": 20},
                },
                "overall_enabled_categories": {"net_win_rate_model_minus_baseline": 0.20, "both_wrong_rate": 0.0},
                "slice_analysis": {
                    "truth_slice_lookup": {
                        "haircare:shampoo": {"model_win_rate_vs_truth": 0.2, "baseline_win_rate_vs_truth": 0.7, "net_wins_model_minus_baseline": -1},
                        "haircare:hair_mask": {"model_win_rate_vs_truth": 0.3, "baseline_win_rate_vs_truth": 0.0, "net_wins_model_minus_baseline": 10},
                        "haircare:hair_oil": {"model_win_rate_vs_truth": 0.4, "baseline_win_rate_vs_truth": 0.0, "net_wins_model_minus_baseline": 10},
                        "skincare:essence": {"model_win_rate_vs_truth": 0.5, "baseline_win_rate_vs_truth": 0.0, "net_wins_model_minus_baseline": 20},
                    },
                    "disagreement_pair_lookup": {
                        "haircare:shampoo:conditioner": {"model_win_rate_vs_truth": 0.2, "baseline_win_rate_vs_truth": 0.7, "net_wins_model_minus_baseline": -1},
                    },
                },
            }

            with patch(
                "roadmap_app.nextstep_targeted_retrain.build_nextstep_v4_decision_quality_payload",
                side_effect=[active_payload, retrain_payload, candidate_payload],
            ), patch(
                "roadmap_app.nextstep_targeted_retrain.build_nextstep_haircare_shampoo_truth_design_payload",
                side_effect=[
                    self._two_stage_truth_payload(stage1_model_win_rate=1.0, stage2_model_win_rate=0.5),
                    self._two_stage_truth_payload(stage1_model_win_rate=1.0, stage2_model_win_rate=0.5),
                    self._two_stage_truth_payload(stage1_model_win_rate=1.0, stage2_model_win_rate=1.0),
                ],
            ):
                payload = build_historical_anchor_candidate_comparison_payload(
                    active_model_path=str(active_dir / "model.pkl"),
                    retrain_v1_model_path=str(retrain_dir / "model.pkl"),
                    candidate_model_path=str(candidate_dir / "model.pkl"),
                    days=30,
                )

            gate_names = {gate["name"]: gate for gate in payload["acceptance_gates"]["gates"]}
            self.assertTrue(gate_names["overall_decision_quality_not_worse_than_active"]["passed"])
            self.assertTrue(gate_names["haircare_shampoo_two_stage_truth_improves"]["passed"])

    def test_two_stage_truth_is_used_instead_of_old_exact_shampoo_rule(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            active_dir = root / "active"
            retrain_dir = root / "retrain"
            candidate_dir = root / "candidate"
            for artifact_dir, version in [
                (active_dir, "active_v1"),
                (retrain_dir, "retrain_v1"),
                (candidate_dir, "v5_v1"),
            ]:
                artifact_dir.mkdir(parents=True, exist_ok=True)
                model_path = artifact_dir / "model.pkl"
                model_path.write_bytes(b"placeholder")
                _write_json(artifact_dir / "metadata.json", {"model_version": version})
                _write_json(artifact_dir / "eval_report.json", {"model_version": version, "metrics_test": {"recall_at_1": 0.23, "ndcg_at_5": 0.52}})
                _write_json(artifact_dir / "shadow_report.json", {"model_version": version, "model_path": str(model_path)})
                _write_json(artifact_dir / "uplift_report_7d.json", {"model_version": version, "model_path": str(model_path)})
                _write_json(artifact_dir / "uplift_report_30d.json", {"model_version": version, "model_path": str(model_path)})

            active_payload = {
                "per_category": {
                    "haircare": {"rollout_reason": "low_uplift", "diagnosis": {"code": "C"}, "model_win_rate_vs_truth": 0.16, "baseline_win_rate_vs_truth": 0.02, "both_wrong_rate": 0.42, "resolved_truth_anchors_total": 120},
                    "skincare": {"rollout_reason": "low_uplift", "diagnosis": {"code": "C"}, "model_win_rate_vs_truth": 0.20, "baseline_win_rate_vs_truth": 0.10, "both_wrong_rate": 0.05, "resolved_truth_anchors_total": 300},
                    "makeup": {"rollout_reason": "sample_too_small_but_nonzero_control", "diagnosis": {"code": "A"}, "model_win_rate_vs_truth": 0.0, "baseline_win_rate_vs_truth": 0.0, "both_wrong_rate": 0.0, "resolved_truth_anchors_total": 20},
                    "fragrance": {"rollout_reason": "category_disabled", "diagnosis": {"code": "C"}, "model_win_rate_vs_truth": 0.20, "baseline_win_rate_vs_truth": 0.30, "both_wrong_rate": 0.20, "resolved_truth_anchors_total": 20},
                },
                "overall_enabled_categories": {"net_win_rate_model_minus_baseline": -0.11, "both_wrong_rate": 0.26},
                "slice_analysis": {
                    "truth_slice_lookup": {
                        "haircare:shampoo": {"model_win_rate_vs_truth": 0.0, "baseline_win_rate_vs_truth": 1.0, "net_wins_model_minus_baseline": -3},
                        "haircare:hair_mask": {"model_win_rate_vs_truth": 0.3, "baseline_win_rate_vs_truth": 0.0, "net_wins_model_minus_baseline": 10},
                        "haircare:hair_oil": {"model_win_rate_vs_truth": 0.4, "baseline_win_rate_vs_truth": 0.0, "net_wins_model_minus_baseline": 10},
                        "skincare:essence": {"model_win_rate_vs_truth": 0.3, "baseline_win_rate_vs_truth": 0.0, "net_wins_model_minus_baseline": 10},
                    },
                    "disagreement_pair_lookup": {
                        "haircare:shampoo:conditioner": {"model_win_rate_vs_truth": 0.0, "baseline_win_rate_vs_truth": 1.0, "net_wins_model_minus_baseline": -3},
                    },
                },
            }
            retrain_payload = active_payload
            candidate_payload = {
                "per_category": {
                    "haircare": {"rollout_reason": "low_uplift", "diagnosis": {"code": "C"}, "model_win_rate_vs_truth": 0.30, "baseline_win_rate_vs_truth": 0.0, "both_wrong_rate": 0.10, "resolved_truth_anchors_total": 120},
                    "skincare": {"rollout_reason": "low_uplift", "diagnosis": {"code": "C"}, "model_win_rate_vs_truth": 0.20, "baseline_win_rate_vs_truth": 0.10, "both_wrong_rate": 0.05, "resolved_truth_anchors_total": 300},
                    "makeup": {"rollout_reason": "sample_too_small_but_nonzero_control", "diagnosis": {"code": "A"}, "model_win_rate_vs_truth": 0.0, "baseline_win_rate_vs_truth": 0.0, "both_wrong_rate": 0.0, "resolved_truth_anchors_total": 20},
                    "fragrance": {"rollout_reason": "category_disabled", "diagnosis": {"code": "C"}, "model_win_rate_vs_truth": 0.25, "baseline_win_rate_vs_truth": 0.10, "both_wrong_rate": 0.0, "resolved_truth_anchors_total": 20},
                },
                "overall_enabled_categories": {"net_win_rate_model_minus_baseline": 0.20, "both_wrong_rate": 0.0},
                "slice_analysis": {
                    "truth_slice_lookup": {
                        "haircare:shampoo": {"model_win_rate_vs_truth": 0.0, "baseline_win_rate_vs_truth": 1.0, "net_wins_model_minus_baseline": -4},
                        "haircare:hair_mask": {"model_win_rate_vs_truth": 0.3, "baseline_win_rate_vs_truth": 0.0, "net_wins_model_minus_baseline": 10},
                        "haircare:hair_oil": {"model_win_rate_vs_truth": 0.4, "baseline_win_rate_vs_truth": 0.0, "net_wins_model_minus_baseline": 10},
                        "skincare:essence": {"model_win_rate_vs_truth": 0.5, "baseline_win_rate_vs_truth": 0.0, "net_wins_model_minus_baseline": 20},
                    },
                    "disagreement_pair_lookup": {
                        "haircare:shampoo:conditioner": {"model_win_rate_vs_truth": 0.0, "baseline_win_rate_vs_truth": 1.0, "net_wins_model_minus_baseline": -4},
                    },
                },
            }

            with patch(
                "roadmap_app.nextstep_targeted_retrain.build_nextstep_v4_decision_quality_payload",
                side_effect=[active_payload, retrain_payload, candidate_payload],
            ), patch(
                "roadmap_app.nextstep_targeted_retrain.build_nextstep_haircare_shampoo_truth_design_payload",
                side_effect=[
                    self._two_stage_truth_payload(stage1_model_win_rate=1.0, stage2_model_win_rate=0.5),
                    self._two_stage_truth_payload(stage1_model_win_rate=1.0, stage2_model_win_rate=0.5),
                    self._two_stage_truth_payload(stage1_model_win_rate=1.0, stage2_model_win_rate=1.0),
                ],
            ):
                payload = build_historical_anchor_candidate_comparison_payload(
                    active_model_path=str(active_dir / "model.pkl"),
                    retrain_v1_model_path=str(retrain_dir / "model.pkl"),
                    candidate_model_path=str(candidate_dir / "model.pkl"),
                    days=30,
                )

            gate_names = {gate["name"]: gate for gate in payload["acceptance_gates"]["gates"]}
            self.assertTrue(gate_names["haircare_shampoo_two_stage_truth_improves"]["passed"])
            self.assertEqual(
                gate_names["haircare_shampoo_two_stage_truth_improves"]["details"]["old_gate_semantics"],
                "exact_shampoo_truth_or_shampoo_to_conditioner_pair",
            )

    def test_two_stage_truth_gate_keeps_unresolved_rows_fail_closed(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            active_dir = root / "active"
            retrain_dir = root / "retrain"
            candidate_dir = root / "candidate"
            for artifact_dir, version in [
                (active_dir, "active_v1"),
                (retrain_dir, "retrain_v1"),
                (candidate_dir, "v5_v1"),
            ]:
                artifact_dir.mkdir(parents=True, exist_ok=True)
                model_path = artifact_dir / "model.pkl"
                model_path.write_bytes(b"placeholder")
                _write_json(artifact_dir / "metadata.json", {"model_version": version})
                _write_json(artifact_dir / "eval_report.json", {"model_version": version, "metrics_test": {"recall_at_1": 0.23, "ndcg_at_5": 0.52}})
                _write_json(artifact_dir / "shadow_report.json", {"model_version": version, "model_path": str(model_path)})
                _write_json(artifact_dir / "uplift_report_7d.json", {"model_version": version, "model_path": str(model_path)})
                _write_json(artifact_dir / "uplift_report_30d.json", {"model_version": version, "model_path": str(model_path)})

            dq_payload = {
                "per_category": {
                    "haircare": {"rollout_reason": "low_uplift", "diagnosis": {"code": "C"}, "model_win_rate_vs_truth": 0.16, "baseline_win_rate_vs_truth": 0.02, "both_wrong_rate": 0.42, "resolved_truth_anchors_total": 120},
                    "skincare": {"rollout_reason": "low_uplift", "diagnosis": {"code": "C"}, "model_win_rate_vs_truth": 0.20, "baseline_win_rate_vs_truth": 0.10, "both_wrong_rate": 0.05, "resolved_truth_anchors_total": 300},
                    "makeup": {"rollout_reason": "sample_too_small_but_nonzero_control", "diagnosis": {"code": "A"}, "model_win_rate_vs_truth": 0.0, "baseline_win_rate_vs_truth": 0.0, "both_wrong_rate": 0.0, "resolved_truth_anchors_total": 20},
                    "fragrance": {"rollout_reason": "category_disabled", "diagnosis": {"code": "C"}, "model_win_rate_vs_truth": 0.25, "baseline_win_rate_vs_truth": 0.10, "both_wrong_rate": 0.0, "resolved_truth_anchors_total": 20},
                },
                "overall_enabled_categories": {"net_win_rate_model_minus_baseline": 0.20, "both_wrong_rate": 0.0},
                "slice_analysis": {
                    "truth_slice_lookup": {
                        "haircare:hair_mask": {"model_win_rate_vs_truth": 0.3, "baseline_win_rate_vs_truth": 0.0, "net_wins_model_minus_baseline": 10},
                        "haircare:hair_oil": {"model_win_rate_vs_truth": 0.4, "baseline_win_rate_vs_truth": 0.0, "net_wins_model_minus_baseline": 10},
                        "skincare:essence": {"model_win_rate_vs_truth": 0.5, "baseline_win_rate_vs_truth": 0.0, "net_wins_model_minus_baseline": 20},
                    },
                    "disagreement_pair_lookup": {},
                },
            }

            with patch(
                "roadmap_app.nextstep_targeted_retrain.build_nextstep_v4_decision_quality_payload",
                side_effect=[dq_payload, dq_payload, dq_payload],
            ), patch(
                "roadmap_app.nextstep_targeted_retrain.build_nextstep_haircare_shampoo_truth_design_payload",
                side_effect=[
                    self._two_stage_truth_payload(stage1_model_win_rate=1.0, stage2_model_win_rate=0.5, resolved=40, unresolved=12),
                    self._two_stage_truth_payload(stage1_model_win_rate=1.0, stage2_model_win_rate=0.5, resolved=40, unresolved=12),
                    self._two_stage_truth_payload(stage1_model_win_rate=1.0, stage2_model_win_rate=1.0, resolved=0, unresolved=52),
                ],
            ):
                payload = build_historical_anchor_candidate_comparison_payload(
                    active_model_path=str(active_dir / "model.pkl"),
                    retrain_v1_model_path=str(retrain_dir / "model.pkl"),
                    candidate_model_path=str(candidate_dir / "model.pkl"),
                    days=30,
                )

            gate_names = {gate["name"]: gate for gate in payload["acceptance_gates"]["gates"]}
            self.assertFalse(gate_names["haircare_shampoo_two_stage_truth_improves"]["passed"])
            self.assertEqual(
                gate_names["haircare_shampoo_two_stage_truth_improves"]["reason"],
                "haircare_two_stage_truth_not_measurable",
            )

    def test_comparison_payload_under_two_stage_truth_does_not_modify_catalog(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            active_dir = root / "active"
            retrain_dir = root / "retrain"
            candidate_dir = root / "candidate"
            for artifact_dir, version in [
                (active_dir, "active_v1"),
                (retrain_dir, "retrain_v1"),
                (candidate_dir, "v5_v1"),
            ]:
                artifact_dir.mkdir(parents=True, exist_ok=True)
                model_path = artifact_dir / "model.pkl"
                model_path.write_bytes(b"placeholder")
                _write_json(artifact_dir / "metadata.json", {"model_version": version})
                _write_json(artifact_dir / "eval_report.json", {"model_version": version, "metrics_test": {"recall_at_1": 0.23, "ndcg_at_5": 0.52}})
                _write_json(artifact_dir / "shadow_report.json", {"model_version": version, "model_path": str(model_path)})
                _write_json(artifact_dir / "uplift_report_7d.json", {"model_version": version, "model_path": str(model_path)})
                _write_json(artifact_dir / "uplift_report_30d.json", {"model_version": version, "model_path": str(model_path)})

            dq_payload = {
                "per_category": {
                    "haircare": {"rollout_reason": "low_uplift", "diagnosis": {"code": "C"}, "model_win_rate_vs_truth": 0.16, "baseline_win_rate_vs_truth": 0.02, "both_wrong_rate": 0.42, "resolved_truth_anchors_total": 120},
                    "skincare": {"rollout_reason": "low_uplift", "diagnosis": {"code": "C"}, "model_win_rate_vs_truth": 0.20, "baseline_win_rate_vs_truth": 0.10, "both_wrong_rate": 0.05, "resolved_truth_anchors_total": 300},
                    "makeup": {"rollout_reason": "sample_too_small_but_nonzero_control", "diagnosis": {"code": "A"}, "model_win_rate_vs_truth": 0.0, "baseline_win_rate_vs_truth": 0.0, "both_wrong_rate": 0.0, "resolved_truth_anchors_total": 20},
                    "fragrance": {"rollout_reason": "category_disabled", "diagnosis": {"code": "C"}, "model_win_rate_vs_truth": 0.25, "baseline_win_rate_vs_truth": 0.10, "both_wrong_rate": 0.0, "resolved_truth_anchors_total": 20},
                },
                "overall_enabled_categories": {"net_win_rate_model_minus_baseline": 0.20, "both_wrong_rate": 0.0},
                "slice_analysis": {
                    "truth_slice_lookup": {
                        "haircare:hair_mask": {"model_win_rate_vs_truth": 0.3, "baseline_win_rate_vs_truth": 0.0, "net_wins_model_minus_baseline": 10},
                        "haircare:hair_oil": {"model_win_rate_vs_truth": 0.4, "baseline_win_rate_vs_truth": 0.0, "net_wins_model_minus_baseline": 10},
                        "skincare:essence": {"model_win_rate_vs_truth": 0.5, "baseline_win_rate_vs_truth": 0.0, "net_wins_model_minus_baseline": 20},
                    },
                    "disagreement_pair_lookup": {},
                },
            }

            with patch(
                "roadmap_app.nextstep_targeted_retrain.build_nextstep_v4_decision_quality_payload",
                side_effect=[dq_payload, dq_payload, dq_payload],
            ), patch(
                "roadmap_app.nextstep_targeted_retrain.build_nextstep_haircare_shampoo_truth_design_payload",
                side_effect=[
                    self._two_stage_truth_payload(stage1_model_win_rate=1.0, stage2_model_win_rate=0.5),
                    self._two_stage_truth_payload(stage1_model_win_rate=1.0, stage2_model_win_rate=0.5),
                    self._two_stage_truth_payload(stage1_model_win_rate=1.0, stage2_model_win_rate=1.0),
                ],
            ), patch(
                "catalog.models.Product.save",
                side_effect=AssertionError("catalog write should not happen"),
            ), patch(
                "catalog.models.Product.delete",
                side_effect=AssertionError("catalog delete should not happen"),
            ):
                payload = build_historical_anchor_candidate_comparison_payload(
                    active_model_path=str(active_dir / "model.pkl"),
                    retrain_v1_model_path=str(retrain_dir / "model.pkl"),
                    candidate_model_path=str(candidate_dir / "model.pkl"),
                    days=30,
                )

            self.assertEqual(
                payload["haircare_shampoo_truth_gate_comparison"]["v5_historical_anchor"]["stage_2_concrete_model_win_rate"],
                1.0,
            )

    def test_broader_qualification_summary_recommends_a_for_v5(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            active_dir = root / "active"
            retrain_dir = root / "retrain"
            candidate_dir = root / "candidate"
            for artifact_dir, version, metrics in [
                (active_dir, "active_v1", {"recall_at_1": 0.23, "ndcg_at_5": 0.52}),
                (retrain_dir, "retrain_v1", {"recall_at_1": 0.22, "ndcg_at_5": 0.51}),
                (candidate_dir, "v5_v1", {"recall_at_1": 0.97, "ndcg_at_5": 0.98}),
            ]:
                artifact_dir.mkdir(parents=True, exist_ok=True)
                model_path = artifact_dir / "model.pkl"
                model_path.write_bytes(b"placeholder")
                _write_json(artifact_dir / "metadata.json", {"model_version": version})
                _write_json(artifact_dir / "eval_report.json", {"model_version": version, "metrics_test": metrics})
                _write_json(artifact_dir / "shadow_report.json", {"model_version": version, "model_path": str(model_path)})
                _write_json(artifact_dir / "uplift_report_7d.json", {"model_version": version, "model_path": str(model_path)})
                _write_json(artifact_dir / "uplift_report_30d.json", {"model_version": version, "model_path": str(model_path)})

            active_payload = {
                "per_category": {
                    "haircare": {"rollout_reason": "low_uplift", "diagnosis": {"code": "C"}, "model_win_rate_vs_truth": 0.17, "baseline_win_rate_vs_truth": 0.0, "both_wrong_rate": 0.45, "resolved_truth_anchors_total": 117},
                    "skincare": {"rollout_reason": "low_uplift", "diagnosis": {"code": "B"}, "model_win_rate_vs_truth": 0.08, "baseline_win_rate_vs_truth": 0.23, "both_wrong_rate": 0.17, "resolved_truth_anchors_total": 276},
                    "makeup": {"rollout_reason": "sample_too_small_but_nonzero_control", "diagnosis": {"code": "A"}, "model_win_rate_vs_truth": 0.0, "baseline_win_rate_vs_truth": 0.0, "both_wrong_rate": 0.0, "resolved_truth_anchors_total": 28},
                    "fragrance": {"rollout_reason": "category_disabled", "diagnosis": {"code": "C"}, "model_win_rate_vs_truth": 0.24, "baseline_win_rate_vs_truth": 0.30, "both_wrong_rate": 0.29, "resolved_truth_anchors_total": 63},
                },
                "overall_enabled_categories": {"net_win_rate_model_minus_baseline": -0.10, "both_wrong_rate": 0.25},
                "slice_analysis": {
                    "truth_slice_lookup": {
                        "haircare:hair_mask": {"model_win_rate_vs_truth": 0.31, "baseline_win_rate_vs_truth": 0.0, "net_wins_model_minus_baseline": 10},
                        "haircare:hair_oil": {"model_win_rate_vs_truth": 0.48, "baseline_win_rate_vs_truth": 0.0, "net_wins_model_minus_baseline": 10},
                        "skincare:essence": {"model_win_rate_vs_truth": 0.33, "baseline_win_rate_vs_truth": 0.0, "net_wins_model_minus_baseline": 10},
                    },
                    "disagreement_pair_lookup": {
                        "haircare:shampoo:conditioner": {"model_win_rate_vs_truth": 0.0, "baseline_win_rate_vs_truth": 1.0, "net_wins_model_minus_baseline": -3},
                    },
                },
            }
            retrain_payload = {
                "per_category": {
                    "haircare": {"rollout_reason": "low_uplift", "diagnosis": {"code": "C"}, "model_win_rate_vs_truth": 0.09, "baseline_win_rate_vs_truth": 0.0, "both_wrong_rate": 0.54, "resolved_truth_anchors_total": 117},
                    "skincare": {"rollout_reason": "low_uplift", "diagnosis": {"code": "B"}, "model_win_rate_vs_truth": 0.04, "baseline_win_rate_vs_truth": 0.22, "both_wrong_rate": 0.21, "resolved_truth_anchors_total": 276},
                    "makeup": {"rollout_reason": "sample_too_small_but_nonzero_control", "diagnosis": {"code": "A"}, "model_win_rate_vs_truth": 0.0, "baseline_win_rate_vs_truth": 0.0, "both_wrong_rate": 0.0, "resolved_truth_anchors_total": 28},
                    "fragrance": {"rollout_reason": "category_disabled", "diagnosis": {"code": "B"}, "model_win_rate_vs_truth": 0.0, "baseline_win_rate_vs_truth": 0.17, "both_wrong_rate": 0.52, "resolved_truth_anchors_total": 63},
                },
                "overall_enabled_categories": {"net_win_rate_model_minus_baseline": -0.12, "both_wrong_rate": 0.30},
                "slice_analysis": {
                    "truth_slice_lookup": {
                        "haircare:hair_mask": {"model_win_rate_vs_truth": 0.0, "baseline_win_rate_vs_truth": 0.0, "net_wins_model_minus_baseline": 0},
                        "haircare:hair_oil": {"model_win_rate_vs_truth": 0.48, "baseline_win_rate_vs_truth": 0.0, "net_wins_model_minus_baseline": 10},
                        "skincare:essence": {"model_win_rate_vs_truth": 0.0, "baseline_win_rate_vs_truth": 0.0, "net_wins_model_minus_baseline": 0},
                    },
                    "disagreement_pair_lookup": {
                        "haircare:shampoo:conditioner": {"model_win_rate_vs_truth": 0.0, "baseline_win_rate_vs_truth": 1.0, "net_wins_model_minus_baseline": -4},
                    },
                },
            }
            candidate_payload = {
                "per_category": {
                    "haircare": {"rollout_reason": "low_uplift", "diagnosis": {"code": "C"}, "model_win_rate_vs_truth": 0.62, "baseline_win_rate_vs_truth": 0.0, "both_wrong_rate": 0.0, "resolved_truth_anchors_total": 117},
                    "skincare": {"rollout_reason": "low_uplift", "diagnosis": {"code": "C"}, "model_win_rate_vs_truth": 0.25, "baseline_win_rate_vs_truth": 0.0, "both_wrong_rate": 0.0, "resolved_truth_anchors_total": 276},
                    "makeup": {"rollout_reason": "sample_too_small_but_nonzero_control", "diagnosis": {"code": "A"}, "model_win_rate_vs_truth": 0.0, "baseline_win_rate_vs_truth": 0.0, "both_wrong_rate": 0.0, "resolved_truth_anchors_total": 28},
                    "fragrance": {"rollout_reason": "category_disabled", "diagnosis": {"code": "C"}, "model_win_rate_vs_truth": 0.41, "baseline_win_rate_vs_truth": 0.05, "both_wrong_rate": 0.11, "resolved_truth_anchors_total": 63},
                },
                "overall_enabled_categories": {"net_win_rate_model_minus_baseline": 0.30, "both_wrong_rate": 0.0},
                "slice_analysis": {
                    "truth_slice_lookup": {
                        "haircare:hair_mask": {"model_win_rate_vs_truth": 0.31, "baseline_win_rate_vs_truth": 0.0, "net_wins_model_minus_baseline": 10},
                        "haircare:hair_oil": {"model_win_rate_vs_truth": 0.48, "baseline_win_rate_vs_truth": 0.0, "net_wins_model_minus_baseline": 10},
                        "skincare:essence": {"model_win_rate_vs_truth": 1.0, "baseline_win_rate_vs_truth": 0.0, "net_wins_model_minus_baseline": 30},
                    },
                    "disagreement_pair_lookup": {
                        "haircare:shampoo:conditioner": {"model_win_rate_vs_truth": 0.0, "baseline_win_rate_vs_truth": 1.0, "net_wins_model_minus_baseline": -4},
                    },
                },
            }

            with patch(
                "roadmap_app.nextstep_targeted_retrain.build_nextstep_v4_decision_quality_payload",
                side_effect=[active_payload, retrain_payload, candidate_payload],
            ), patch(
                "roadmap_app.nextstep_targeted_retrain.build_nextstep_haircare_shampoo_truth_design_payload",
                side_effect=[
                    self._two_stage_truth_payload(stage1_model_win_rate=1.0, stage2_model_win_rate=0.5),
                    self._two_stage_truth_payload(stage1_model_win_rate=0.75, stage2_model_win_rate=0.25),
                    self._two_stage_truth_payload(stage1_model_win_rate=1.0, stage2_model_win_rate=1.0),
                ],
            ):
                payload = build_historical_anchor_candidate_comparison_payload(
                    active_model_path=str(active_dir / "model.pkl"),
                    retrain_v1_model_path=str(retrain_dir / "model.pkl"),
                    candidate_model_path=str(candidate_dir / "model.pkl"),
                    days=30,
                )

            broader = payload["broader_qualification"]
            self.assertEqual(broader["global"]["recommendation_code"], "A")
            self.assertTrue(broader["global"]["is_v5_new_best_continuation_candidate"])
            self.assertEqual(
                broader["per_category"]["haircare"]["status"],
                "candidate_for_next_stage_under_freeze",
            )
            self.assertEqual(
                broader["per_category"]["skincare"]["status"],
                "improved_enough_for_next_stage_under_freeze",
            )
            self.assertEqual(
                broader["per_category"]["makeup"]["status"],
                "sample_limited_hold",
            )
            self.assertEqual(
                broader["per_category"]["fragrance"]["status"],
                "analysis_only_positive_signal",
            )
            markdown = render_historical_anchor_candidate_comparison_markdown(payload)
            self.assertIn("final recommendation: `A`", markdown)

    @override_settings(
        ROADMAP_RUNTIME_FREEZE_ML=True,
        ROADMAP_NEXTSTEP_V4_MODEL_PATH="models/original_active.pkl",
        ROADMAP_NEXTSTEP_V4_SHADOW_MODEL_PATH="models/original_shadow.pkl",
    )
    def test_comparison_payload_does_not_modify_runtime_config(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            active_dir = root / "active"
            retrain_dir = root / "retrain"
            candidate_dir = root / "candidate"
            for artifact_dir, version in [
                (active_dir, "active_v1"),
                (retrain_dir, "retrain_v1"),
                (candidate_dir, "v5_v1"),
            ]:
                artifact_dir.mkdir(parents=True, exist_ok=True)
                model_path = artifact_dir / "model.pkl"
                model_path.write_bytes(b"placeholder")
                _write_json(artifact_dir / "metadata.json", {"model_version": version})
                _write_json(artifact_dir / "eval_report.json", {"model_version": version, "metrics_test": {"recall_at_1": 0.23, "ndcg_at_5": 0.52}})
                _write_json(artifact_dir / "shadow_report.json", {"model_version": version, "model_path": str(model_path)})
                _write_json(artifact_dir / "uplift_report_7d.json", {"model_version": version, "model_path": str(model_path)})
                _write_json(artifact_dir / "uplift_report_30d.json", {"model_version": version, "model_path": str(model_path)})

            dq_payload = {
                "per_category": {
                    "haircare": {"rollout_reason": "low_uplift", "diagnosis": {"code": "C"}, "model_win_rate_vs_truth": 0.16, "baseline_win_rate_vs_truth": 0.02, "both_wrong_rate": 0.42, "resolved_truth_anchors_total": 120},
                    "skincare": {"rollout_reason": "low_uplift", "diagnosis": {"code": "C"}, "model_win_rate_vs_truth": 0.20, "baseline_win_rate_vs_truth": 0.10, "both_wrong_rate": 0.05, "resolved_truth_anchors_total": 300},
                    "makeup": {"rollout_reason": "sample_too_small_but_nonzero_control", "diagnosis": {"code": "A"}, "model_win_rate_vs_truth": 0.0, "baseline_win_rate_vs_truth": 0.0, "both_wrong_rate": 0.0, "resolved_truth_anchors_total": 20},
                    "fragrance": {"rollout_reason": "category_disabled", "diagnosis": {"code": "C"}, "model_win_rate_vs_truth": 0.25, "baseline_win_rate_vs_truth": 0.10, "both_wrong_rate": 0.0, "resolved_truth_anchors_total": 20},
                },
                "overall_enabled_categories": {"net_win_rate_model_minus_baseline": 0.20, "both_wrong_rate": 0.0},
                "slice_analysis": {
                    "truth_slice_lookup": {
                        "haircare:hair_mask": {"model_win_rate_vs_truth": 0.3, "baseline_win_rate_vs_truth": 0.0, "net_wins_model_minus_baseline": 10},
                        "haircare:hair_oil": {"model_win_rate_vs_truth": 0.4, "baseline_win_rate_vs_truth": 0.0, "net_wins_model_minus_baseline": 10},
                        "skincare:essence": {"model_win_rate_vs_truth": 0.5, "baseline_win_rate_vs_truth": 0.0, "net_wins_model_minus_baseline": 20},
                    },
                    "disagreement_pair_lookup": {},
                },
            }

            before = (
                settings.ROADMAP_RUNTIME_FREEZE_ML,
                settings.ROADMAP_NEXTSTEP_V4_MODEL_PATH,
                settings.ROADMAP_NEXTSTEP_V4_SHADOW_MODEL_PATH,
            )
            with patch(
                "roadmap_app.nextstep_targeted_retrain.build_nextstep_v4_decision_quality_payload",
                side_effect=[dq_payload, dq_payload, dq_payload],
            ), patch(
                "roadmap_app.nextstep_targeted_retrain.build_nextstep_haircare_shampoo_truth_design_payload",
                side_effect=[
                    self._two_stage_truth_payload(stage1_model_win_rate=1.0, stage2_model_win_rate=0.5),
                    self._two_stage_truth_payload(stage1_model_win_rate=1.0, stage2_model_win_rate=0.5),
                    self._two_stage_truth_payload(stage1_model_win_rate=1.0, stage2_model_win_rate=1.0),
                ],
            ):
                payload = build_historical_anchor_candidate_comparison_payload(
                    active_model_path=str(active_dir / "model.pkl"),
                    retrain_v1_model_path=str(retrain_dir / "model.pkl"),
                    candidate_model_path=str(candidate_dir / "model.pkl"),
                    days=30,
                )

            after = (
                settings.ROADMAP_RUNTIME_FREEZE_ML,
                settings.ROADMAP_NEXTSTEP_V4_MODEL_PATH,
                settings.ROADMAP_NEXTSTEP_V4_SHADOW_MODEL_PATH,
            )
            self.assertEqual(before, after)
            self.assertFalse(payload["runtime_guardrails"]["runtime_config_changed"])
            self.assertFalse(payload["qualification_scope"]["runtime_enablement"])
            self.assertFalse(payload["qualification_scope"]["rule_baseline_behavior_changed"])

    def test_historical_anchor_comparison_reuses_one_shared_live_context(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            active_dir = root / "active"
            retrain_dir = root / "retrain"
            candidate_dir = root / "candidate"
            for artifact_dir, version in [
                (active_dir, "active_v1"),
                (retrain_dir, "retrain_v1"),
                (candidate_dir, "v5_v1"),
            ]:
                artifact_dir.mkdir(parents=True, exist_ok=True)
                model_path = artifact_dir / "model.pkl"
                model_path.write_bytes(b"placeholder")
                _write_json(artifact_dir / "metadata.json", {"model_version": version})
                _write_json(
                    artifact_dir / "eval_report.json",
                    {"model_version": version, "metrics_test": {"recall_at_1": 0.23, "ndcg_at_5": 0.52}},
                )
                _write_json(artifact_dir / "shadow_report.json", {"model_version": version, "model_path": str(model_path)})
                _write_json(artifact_dir / "uplift_report_7d.json", {"model_version": version, "model_path": str(model_path)})
                _write_json(artifact_dir / "uplift_report_30d.json", {"model_version": version, "model_path": str(model_path)})

            shared_context = {
                "since": timezone.now() - timedelta(days=30),
                "until": timezone.now(),
                "category": "all",
                "include_ga": False,
                "anchors": [],
                "meta_by_plan": {},
                "completions_by_step": {},
                "read_only": True,
            }
            dq_payloads = [
                {
                    "per_category": {
                        "haircare": {"rollout_reason": "low_uplift", "diagnosis": {"code": "C"}, "model_win_rate_vs_truth": 0.16, "baseline_win_rate_vs_truth": 0.02, "both_wrong_rate": 0.42, "resolved_truth_anchors_total": 120},
                        "skincare": {"rollout_reason": "low_uplift", "diagnosis": {"code": "C"}, "model_win_rate_vs_truth": 0.18, "baseline_win_rate_vs_truth": 0.10, "both_wrong_rate": 0.10, "resolved_truth_anchors_total": 300},
                        "makeup": {"rollout_reason": "sample_too_small_but_nonzero_control", "diagnosis": {"code": "A"}, "model_win_rate_vs_truth": 0.0, "baseline_win_rate_vs_truth": 0.0, "both_wrong_rate": 0.0, "resolved_truth_anchors_total": 20},
                        "fragrance": {"rollout_reason": "category_disabled", "diagnosis": {"code": "C"}, "model_win_rate_vs_truth": 0.20, "baseline_win_rate_vs_truth": 0.15, "both_wrong_rate": 0.10, "resolved_truth_anchors_total": 20},
                    },
                    "overall_enabled_categories": {"net_win_rate_model_minus_baseline": 0.05, "both_wrong_rate": 0.15},
                    "slice_analysis": {
                        "truth_slice_lookup": {
                            "haircare:hair_mask": {"model_win_rate_vs_truth": 0.3, "baseline_win_rate_vs_truth": 0.0, "net_wins_model_minus_baseline": 10},
                            "haircare:hair_oil": {"model_win_rate_vs_truth": 0.4, "baseline_win_rate_vs_truth": 0.0, "net_wins_model_minus_baseline": 10},
                            "skincare:essence": {"model_win_rate_vs_truth": 0.4, "baseline_win_rate_vs_truth": 0.0, "net_wins_model_minus_baseline": 12},
                        },
                        "disagreement_pair_lookup": {
                            "haircare:shampoo:conditioner": {"model_win_rate_vs_truth": 0.1, "baseline_win_rate_vs_truth": 0.7, "net_wins_model_minus_baseline": -2},
                        },
                    },
                },
                {
                    "per_category": {
                        "haircare": {"rollout_reason": "low_uplift", "diagnosis": {"code": "C"}, "model_win_rate_vs_truth": 0.18, "baseline_win_rate_vs_truth": 0.02, "both_wrong_rate": 0.40, "resolved_truth_anchors_total": 120},
                        "skincare": {"rollout_reason": "low_uplift", "diagnosis": {"code": "C"}, "model_win_rate_vs_truth": 0.19, "baseline_win_rate_vs_truth": 0.10, "both_wrong_rate": 0.09, "resolved_truth_anchors_total": 300},
                        "makeup": {"rollout_reason": "sample_too_small_but_nonzero_control", "diagnosis": {"code": "A"}, "model_win_rate_vs_truth": 0.0, "baseline_win_rate_vs_truth": 0.0, "both_wrong_rate": 0.0, "resolved_truth_anchors_total": 20},
                        "fragrance": {"rollout_reason": "category_disabled", "diagnosis": {"code": "C"}, "model_win_rate_vs_truth": 0.22, "baseline_win_rate_vs_truth": 0.15, "both_wrong_rate": 0.09, "resolved_truth_anchors_total": 20},
                    },
                    "overall_enabled_categories": {"net_win_rate_model_minus_baseline": 0.07, "both_wrong_rate": 0.14},
                    "slice_analysis": {
                        "truth_slice_lookup": {
                            "haircare:hair_mask": {"model_win_rate_vs_truth": 0.31, "baseline_win_rate_vs_truth": 0.0, "net_wins_model_minus_baseline": 10},
                            "haircare:hair_oil": {"model_win_rate_vs_truth": 0.41, "baseline_win_rate_vs_truth": 0.0, "net_wins_model_minus_baseline": 10},
                            "skincare:essence": {"model_win_rate_vs_truth": 0.41, "baseline_win_rate_vs_truth": 0.0, "net_wins_model_minus_baseline": 12},
                        },
                        "disagreement_pair_lookup": {
                            "haircare:shampoo:conditioner": {"model_win_rate_vs_truth": 0.12, "baseline_win_rate_vs_truth": 0.7, "net_wins_model_minus_baseline": -2},
                        },
                    },
                },
                {
                    "per_category": {
                        "haircare": {"rollout_reason": "low_uplift", "diagnosis": {"code": "C"}, "model_win_rate_vs_truth": 0.30, "baseline_win_rate_vs_truth": 0.0, "both_wrong_rate": 0.10, "resolved_truth_anchors_total": 120},
                        "skincare": {"rollout_reason": "low_uplift", "diagnosis": {"code": "C"}, "model_win_rate_vs_truth": 0.25, "baseline_win_rate_vs_truth": 0.10, "both_wrong_rate": 0.05, "resolved_truth_anchors_total": 300},
                        "makeup": {"rollout_reason": "sample_too_small_but_nonzero_control", "diagnosis": {"code": "A"}, "model_win_rate_vs_truth": 0.0, "baseline_win_rate_vs_truth": 0.0, "both_wrong_rate": 0.0, "resolved_truth_anchors_total": 20},
                        "fragrance": {"rollout_reason": "category_disabled", "diagnosis": {"code": "C"}, "model_win_rate_vs_truth": 0.25, "baseline_win_rate_vs_truth": 0.10, "both_wrong_rate": 0.0, "resolved_truth_anchors_total": 20},
                    },
                    "overall_enabled_categories": {"net_win_rate_model_minus_baseline": 0.20, "both_wrong_rate": 0.0},
                    "slice_analysis": {
                        "truth_slice_lookup": {
                            "haircare:hair_mask": {"model_win_rate_vs_truth": 0.32, "baseline_win_rate_vs_truth": 0.0, "net_wins_model_minus_baseline": 10},
                            "haircare:hair_oil": {"model_win_rate_vs_truth": 0.42, "baseline_win_rate_vs_truth": 0.0, "net_wins_model_minus_baseline": 10},
                            "skincare:essence": {"model_win_rate_vs_truth": 0.50, "baseline_win_rate_vs_truth": 0.0, "net_wins_model_minus_baseline": 20},
                        },
                        "disagreement_pair_lookup": {
                            "haircare:shampoo:conditioner": {"model_win_rate_vs_truth": 0.2, "baseline_win_rate_vs_truth": 0.7, "net_wins_model_minus_baseline": -1},
                        },
                    },
                },
            ]
            decision_context_ids: list[int] = []
            shampoo_context_ids: list[int] = []

            def _decision_quality_side_effect(*args, **kwargs):
                decision_context_ids.append(id(kwargs.get("historical_context")))
                return dq_payloads[len(decision_context_ids) - 1]

            shampoo_payloads = [
                self._two_stage_truth_payload(stage1_model_win_rate=1.0, stage2_model_win_rate=0.5),
                self._two_stage_truth_payload(stage1_model_win_rate=0.75, stage2_model_win_rate=0.25),
                self._two_stage_truth_payload(stage1_model_win_rate=1.0, stage2_model_win_rate=1.0),
            ]

            def _shampoo_side_effect(*args, **kwargs):
                shampoo_context_ids.append(id(kwargs.get("historical_context")))
                return shampoo_payloads[len(shampoo_context_ids) - 1]

            with patch(
                "roadmap_app.nextstep_targeted_retrain.build_historical_anchor_read_context",
                return_value=shared_context,
            ) as build_context_mock, patch(
                "roadmap_app.nextstep_targeted_retrain.build_nextstep_v4_decision_quality_payload",
                side_effect=_decision_quality_side_effect,
            ), patch(
                "roadmap_app.nextstep_targeted_retrain.build_nextstep_haircare_shampoo_truth_design_payload",
                side_effect=_shampoo_side_effect,
            ):
                payload = build_historical_anchor_candidate_comparison_payload(
                    active_model_path=str(active_dir / "model.pkl"),
                    retrain_v1_model_path=str(retrain_dir / "model.pkl"),
                    candidate_model_path=str(candidate_dir / "model.pkl"),
                    days=30,
                )

            self.assertEqual(build_context_mock.call_count, 1)
            self.assertEqual(decision_context_ids, [id(shared_context)] * 3)
            self.assertEqual(shampoo_context_ids, [id(shared_context)] * 3)
            self.assertFalse(payload["catalog_safety"]["catalog_writes_performed"])


class RoadmapNextstepCandidatePromotionTests(SimpleTestCase):
    def _cached_comparison_payload(self, *, active_model_path: str, retrain_model_path: str, candidate_model_path: str) -> dict:
        return {
            "artifacts": {
                "active": {"model_path": active_model_path},
                "retrain_v1": {"model_path": retrain_model_path},
                "v5_historical_anchor": {"model_path": candidate_model_path},
            },
            "category_comparison": [
                {
                    "category": "haircare",
                    "active": {"rollout_reason": "low_uplift", "diagnosis": {"code": "C"}, "model_win_rate": 0.17, "baseline_win_rate": 0.0, "both_wrong_rate": 0.45, "resolved_truth": 117},
                    "retrain_v1": {"rollout_reason": "low_uplift", "diagnosis": {"code": "C"}, "model_win_rate": 0.09, "baseline_win_rate": 0.0, "both_wrong_rate": 0.54, "resolved_truth": 117},
                    "v5_historical_anchor": {"rollout_reason": "low_uplift", "diagnosis": {"code": "C"}, "model_win_rate": 0.62, "baseline_win_rate": 0.0, "both_wrong_rate": 0.0, "resolved_truth": 117},
                },
                {
                    "category": "skincare",
                    "active": {"rollout_reason": "low_uplift", "diagnosis": {"code": "B"}, "model_win_rate": 0.08, "baseline_win_rate": 0.23, "both_wrong_rate": 0.17, "resolved_truth": 276},
                    "retrain_v1": {"rollout_reason": "low_uplift", "diagnosis": {"code": "B"}, "model_win_rate": 0.04, "baseline_win_rate": 0.22, "both_wrong_rate": 0.21, "resolved_truth": 276},
                    "v5_historical_anchor": {"rollout_reason": "low_uplift", "diagnosis": {"code": "C"}, "model_win_rate": 0.25, "baseline_win_rate": 0.0, "both_wrong_rate": 0.0, "resolved_truth": 276},
                },
                {
                    "category": "makeup",
                    "active": {"rollout_reason": "sample_too_small_but_nonzero_control", "diagnosis": {"code": "A"}, "model_win_rate": 0.0, "baseline_win_rate": 0.0, "both_wrong_rate": 0.0, "resolved_truth": 28},
                    "retrain_v1": {"rollout_reason": "sample_too_small_but_nonzero_control", "diagnosis": {"code": "A"}, "model_win_rate": 0.0, "baseline_win_rate": 0.0, "both_wrong_rate": 0.0, "resolved_truth": 28},
                    "v5_historical_anchor": {"rollout_reason": "sample_too_small_but_nonzero_control", "diagnosis": {"code": "A"}, "model_win_rate": 0.0, "baseline_win_rate": 0.0, "both_wrong_rate": 0.0, "resolved_truth": 28},
                },
                {
                    "category": "fragrance",
                    "active": {"rollout_reason": "category_disabled", "diagnosis": {"code": "C"}, "model_win_rate": 0.24, "baseline_win_rate": 0.30, "both_wrong_rate": 0.29, "resolved_truth": 63},
                    "retrain_v1": {"rollout_reason": "category_disabled", "diagnosis": {"code": "B"}, "model_win_rate": 0.0, "baseline_win_rate": 0.17, "both_wrong_rate": 0.52, "resolved_truth": 63},
                    "v5_historical_anchor": {"rollout_reason": "category_disabled", "diagnosis": {"code": "C"}, "model_win_rate": 0.41, "baseline_win_rate": 0.05, "both_wrong_rate": 0.11, "resolved_truth": 63},
                },
            ],
            "acceptance_gates": {
                "overall_passed": True,
                "gates": [
                    {"name": "skincare_not_clearly_B_low_uplift", "passed": True, "reason": "passed", "details": {}},
                    {"name": "haircare_shampoo_two_stage_truth_improves", "passed": True, "reason": "passed", "details": {}},
                    {"name": "protected_slices_non_regression", "passed": True, "reason": "passed", "details": {}},
                    {"name": "overall_decision_quality_not_worse_than_active", "passed": True, "reason": "passed", "details": {}},
                    {"name": "offline_eval_not_materially_worse_than_active", "passed": True, "reason": "passed", "details": {}},
                ],
            },
            "haircare_shampoo_truth_gate_comparison": {
                "v5_historical_anchor": {
                    "stage_1_family_model_win_rate": 1.0,
                    "stage_2_concrete_model_win_rate": 1.0,
                    "two_stage_resolved": 40,
                    "two_stage_unresolved": 12,
                }
            },
            "targeted_truth_slices": [],
            "protected_truth_slices": [],
        }

    @override_settings(
        ROADMAP_RUNTIME_FREEZE_ML=True,
        ROADMAP_NEXTSTEP_V4_MODEL_PATH="models/runtime_active.pkl",
    )
    def test_materialized_comparison_payload_marks_cached_provenance(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            cached_path = root / "comparison.json"
            active_model_path = str((root / "active" / "model.pkl").resolve())
            retrain_model_path = str((root / "retrain" / "model.pkl").resolve())
            candidate_model_path = str((root / "candidate" / "model.pkl").resolve())
            _write_json(
                cached_path,
                self._cached_comparison_payload(
                    active_model_path=active_model_path,
                    retrain_model_path=retrain_model_path,
                    candidate_model_path=candidate_model_path,
                ),
            )

            payload = materialize_historical_anchor_candidate_comparison_payload(
                active_model_path=active_model_path,
                retrain_v1_model_path=retrain_model_path,
                candidate_model_path=candidate_model_path,
                source_preference="cached_artifact",
                cached_comparison_json_path=str(cached_path),
            )

            self.assertEqual(payload["report_provenance"]["source_of_truth"], "cached_artifact")
            self.assertEqual(
                payload["report_provenance"]["report_materialization"],
                "materialized_from_saved_artifacts",
            )
            self.assertEqual(payload["report_provenance"]["generated_from"], "comparison_json")
            self.assertFalse(payload["runtime_guardrails"]["runtime_config_changed"])

    def test_candidate_promotion_keeps_active_runtime_separate_from_promoted_candidate(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            cached_path = root / "comparison.json"
            active_model_path = str((root / "active" / "model.pkl").resolve())
            retrain_model_path = str((root / "retrain" / "model.pkl").resolve())
            candidate_model_path = str((root / "candidate" / "model.pkl").resolve())
            _write_json(
                cached_path,
                self._cached_comparison_payload(
                    active_model_path=active_model_path,
                    retrain_model_path=retrain_model_path,
                    candidate_model_path=candidate_model_path,
                ),
            )

            with override_settings(
                ROADMAP_RUNTIME_FREEZE_ML=True,
                ROADMAP_NEXTSTEP_V4_MODEL_PATH=active_model_path,
            ):
                before = (settings.ROADMAP_RUNTIME_FREEZE_ML, settings.ROADMAP_NEXTSTEP_V4_MODEL_PATH)
                with patch(
                    "catalog.models.Product.save",
                    side_effect=AssertionError("catalog write should not happen"),
                ), patch(
                    "catalog.models.Product.delete",
                    side_effect=AssertionError("catalog delete should not happen"),
                ):
                    payload = build_v5_candidate_promotion_under_freeze_payload(
                        active_model_path=active_model_path,
                        retrain_v1_model_path=retrain_model_path,
                        candidate_model_path=candidate_model_path,
                        source_preference="cached_artifact",
                        cached_comparison_json_path=str(cached_path),
                    )
                after = (settings.ROADMAP_RUNTIME_FREEZE_ML, settings.ROADMAP_NEXTSTEP_V4_MODEL_PATH)

            promotion_state = payload["promotion_state"]
            self.assertEqual(
                promotion_state["active_runtime_continuation_artifact"]["model_path"],
                active_model_path,
            )
            self.assertEqual(
                promotion_state["promoted_freeze_only_continuation_candidate"]["model_path"],
                candidate_model_path,
            )
            self.assertTrue(payload["executive_verdict"]["canonical_freeze_candidate"])
            self.assertEqual(payload["executive_verdict"]["recommendation_code"], "A")
            self.assertFalse(promotion_state["runtime_serve"]["serve_enabled"])
            self.assertFalse(
                promotion_state["runtime_serve"]["runtime_model_path_switched_to_candidate"]
            )
            self.assertFalse(payload["read_only_guards"]["runtime_enablement_allowed"])
            self.assertFalse(payload["read_only_guards"]["catalog_writes_performed"])
            self.assertEqual(before, after)
            self.assertTrue(
                payload["executive_verdict"]["active_runtime_artifact_unchanged"]
            )
            markdown = render_v5_candidate_promotion_under_freeze_markdown(payload)
            self.assertIn("current active runtime continuation artifact", markdown)
            self.assertIn("promoted freeze-only continuation candidate", markdown)

    def test_artifact_qualification_manifest_includes_freeze_candidate_promotion(self):
        promotion_payload = {
            "promotion_state": {
                "active_runtime_continuation_artifact": {"model_path": "models/active.pkl"},
                "promoted_freeze_only_continuation_candidate": {"model_path": "models/v5.pkl"},
                "runtime_serve": {"serve_enabled": False},
            },
            "executive_verdict": {
                "status": "promoted_under_freeze",
                "recommendation_code": "A",
                "recommendation_label": "continue qualification with v5 as the new best candidate",
                "canonical_freeze_candidate": True,
                "runtime_still_frozen": True,
                "active_runtime_artifact_unchanged": True,
            },
            "provenance": {
                "report_materialization": "materialized_from_saved_artifacts",
                "source_of_truth": "cached_artifact",
                "generated_from": "comparison_json",
            },
            "report_paths": {},
            "read_only_guards": {
                "catalog_writes_performed": False,
                "runtime_config_changed": False,
                "runtime_enablement_allowed": False,
            },
        }
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            active_model_path = _create_nextstep_artifact(
                root,
                "active_artifact",
                with_eval=True,
                with_uplift=True,
                model_version="active_model",
            )
            with override_settings(
                ROADMAP_NEXTSTEP_V4_ENABLED=True,
                ROADMAP_NEXTSTEP_V4_MODEL_PATH=str(active_model_path),
                ROADMAP_NEXTSTEP_V4_ENABLED_CATEGORIES=["skincare", "haircare", "makeup"],
                ROADMAP_NEXTSTEP_V4_DISABLED_CATEGORIES=["fragrance"],
            ), patch(
                "roadmap_app.ml_artifact_qualification.active_fragrance_runtime_integrity_counts",
                return_value={"active_fragrance_slot_mismatch_count": 0},
            ), patch(
                "roadmap_app.ml_artifact_qualification.legacy_bad_fragrance_completion_details",
                return_value={"legacy_bucket": "clean"},
            ), patch(
                "roadmap_app.ml_artifact_qualification.build_v5_candidate_promotion_under_freeze_payload",
                return_value=promotion_payload,
            ):
                payload = build_roadmap_ml_artifact_qualification_payload()

        promotion = payload["freeze_candidate_promotion"]
        self.assertEqual(promotion["status"], "available")
        self.assertTrue(promotion["executive_verdict"]["canonical_freeze_candidate"])
        self.assertEqual(
            promotion["promotion_state"]["active_runtime_continuation_artifact"]["model_path"],
            "models/active.pkl",
        )
        self.assertEqual(
            promotion["promotion_state"]["promoted_freeze_only_continuation_candidate"]["model_path"],
            "models/v5.pkl",
        )
        self.assertEqual(
            promotion["provenance"]["report_materialization"],
            "materialized_from_saved_artifacts",
        )


class RoadmapNextstepSkincareFreezeQualificationTests(SimpleTestCase):
    def _cached_comparison_payload(self, *, active_model_path: str, retrain_model_path: str, candidate_model_path: str) -> dict:
        return {
            "artifacts": {
                "active": {"model_path": active_model_path},
                "retrain_v1": {"model_path": retrain_model_path},
                "v5_historical_anchor": {"model_path": candidate_model_path},
            },
            "category_comparison": [
                {
                    "category": "haircare",
                    "active": {"rollout_reason": "low_uplift", "diagnosis": {"code": "C"}, "model_win_rate": 0.31, "baseline_win_rate": 0.0, "both_wrong_rate": 0.0, "resolved_truth": 22},
                    "retrain_v1": {"rollout_reason": "low_uplift", "diagnosis": {"code": "C"}, "model_win_rate": 0.31, "baseline_win_rate": 0.0, "both_wrong_rate": 0.0, "resolved_truth": 22},
                    "v5_historical_anchor": {"rollout_reason": "low_uplift", "diagnosis": {"code": "C"}, "model_win_rate": 0.31, "baseline_win_rate": 0.0, "both_wrong_rate": 0.0, "resolved_truth": 22},
                },
                {
                    "category": "skincare",
                    "active": {"rollout_reason": "low_uplift", "diagnosis": {"code": "B"}, "model_win_rate": 0.0, "baseline_win_rate": 0.0, "both_wrong_rate": 0.33, "resolved_truth": 61},
                    "retrain_v1": {"rollout_reason": "low_uplift", "diagnosis": {"code": "C"}, "model_win_rate": 0.07, "baseline_win_rate": 0.0, "both_wrong_rate": 0.26, "resolved_truth": 61},
                    "v5_historical_anchor": {"rollout_reason": "low_uplift", "diagnosis": {"code": "D"}, "model_win_rate": 0.33, "baseline_win_rate": 0.0, "both_wrong_rate": 0.0, "resolved_truth": 61},
                },
                {
                    "category": "makeup",
                    "active": {"rollout_reason": "sample_too_small_but_nonzero_control", "diagnosis": {"code": "D"}, "model_win_rate": None, "baseline_win_rate": None, "both_wrong_rate": None, "resolved_truth": 7},
                    "retrain_v1": {"rollout_reason": "sample_too_small_but_nonzero_control", "diagnosis": {"code": "D"}, "model_win_rate": None, "baseline_win_rate": None, "both_wrong_rate": None, "resolved_truth": 7},
                    "v5_historical_anchor": {"rollout_reason": "sample_too_small_but_nonzero_control", "diagnosis": {"code": "D"}, "model_win_rate": None, "baseline_win_rate": None, "both_wrong_rate": None, "resolved_truth": 7},
                },
                {
                    "category": "fragrance",
                    "active": {"rollout_reason": "category_disabled", "diagnosis": {"code": "C"}, "model_win_rate": 0.19, "baseline_win_rate": 0.0, "both_wrong_rate": 0.62, "resolved_truth": 16},
                    "retrain_v1": {"rollout_reason": "category_disabled", "diagnosis": {"code": "C"}, "model_win_rate": 0.0, "baseline_win_rate": 0.0, "both_wrong_rate": 0.81, "resolved_truth": 16},
                    "v5_historical_anchor": {"rollout_reason": "category_disabled", "diagnosis": {"code": "C"}, "model_win_rate": 0.81, "baseline_win_rate": 0.0, "both_wrong_rate": 0.0, "resolved_truth": 16},
                },
            ],
            "acceptance_gates": {
                "overall_passed": False,
                "gates": [
                    {"name": "skincare_not_clearly_B_low_uplift", "passed": True, "reason": "passed", "details": {}},
                    {
                        "name": "haircare_shampoo_two_stage_truth_improves",
                        "passed": False,
                        "reason": "haircare_two_stage_stage2_not_improved",
                        "details": {},
                    },
                    {"name": "protected_slices_non_regression", "passed": True, "reason": "passed", "details": {}},
                    {"name": "overall_decision_quality_not_worse_than_active", "passed": True, "reason": "passed", "details": {}},
                    {"name": "offline_eval_not_materially_worse_than_active", "passed": True, "reason": "passed", "details": {}},
                ],
            },
            "targeted_truth_slices": [
                {
                    "category": "skincare",
                    "truth_product_type": "mask",
                    "active": {"model_win_rate_vs_truth": 0.0},
                    "retrain_v1": {"model_win_rate_vs_truth": 0.24},
                    "v5_historical_anchor": {
                        "model_win_rate_vs_truth": 0.41,
                        "net_wins_model_minus_baseline": 7,
                    },
                },
                {
                    "category": "skincare",
                    "truth_product_type": "eye_cream",
                    "active": {"model_win_rate_vs_truth": 0.0},
                    "retrain_v1": {"model_win_rate_vs_truth": 0.0},
                    "v5_historical_anchor": {
                        "model_win_rate_vs_truth": 0.41,
                        "net_wins_model_minus_baseline": 7,
                    },
                },
            ],
            "protected_truth_slices": [
                {
                    "category": "skincare",
                    "truth_product_type": "essence",
                    "active": {"model_win_rate_vs_truth": 0.0},
                    "retrain_v1": {"model_win_rate_vs_truth": 0.0},
                    "v5_historical_anchor": {
                        "model_win_rate_vs_truth": 1.0,
                        "net_wins_model_minus_baseline": 6,
                    },
                }
            ],
        }

    def test_skincare_freeze_qualification_selects_skincare_as_next_lane(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            cached_path = root / "comparison.json"
            active_model_path = str((root / "active" / "model.pkl").resolve())
            retrain_model_path = str((root / "retrain" / "model.pkl").resolve())
            candidate_model_path = str((root / "candidate" / "model.pkl").resolve())
            _write_json(
                cached_path,
                self._cached_comparison_payload(
                    active_model_path=active_model_path,
                    retrain_model_path=retrain_model_path,
                    candidate_model_path=candidate_model_path,
                ),
            )

            with override_settings(
                ROADMAP_RUNTIME_FREEZE_ML=True,
                ROADMAP_NEXTSTEP_V4_MODEL_PATH=active_model_path,
            ), patch(
                "catalog.models.Product.save",
                side_effect=AssertionError("catalog write should not happen"),
            ), patch(
                "catalog.models.Product.delete",
                side_effect=AssertionError("catalog delete should not happen"),
            ):
                payload = build_v5_skincare_freeze_qualification_payload(
                    active_model_path=active_model_path,
                    retrain_v1_model_path=retrain_model_path,
                    candidate_model_path=candidate_model_path,
                    source_preference="cached_artifact",
                    cached_comparison_json_path=str(cached_path),
                )

        executive = payload["executive_verdict"]
        self.assertEqual(executive["status"], "candidate_for_next_freeze_qualification_stage")
        self.assertEqual(executive["recommendation_code"], "C")
        self.assertEqual(executive["lane_category"], "skincare")
        self.assertTrue(executive["continue_under_freeze"])
        self.assertTrue(executive["runtime_still_frozen"])
        self.assertFalse(executive["runtime_enablement_allowed"])
        self.assertTrue(executive["active_runtime_artifact_unchanged"])
        self.assertTrue(executive["haircare_blocker_still_present"])
        self.assertEqual(executive["next_stage_focus_categories"], ["skincare"])
        self.assertFalse(payload["read_only_guards"]["catalog_writes_performed"])
        self.assertFalse(payload["read_only_guards"]["runtime_config_changed"])
        self.assertEqual(payload["provenance"]["source_of_truth"], "cached_artifact")

        markdown = render_v5_skincare_freeze_qualification_markdown(payload)
        self.assertIn("Skincare is the only current next-stage focus category under recommendation `C`.", markdown)
        self.assertIn("haircare_shampoo_two_stage_truth_improves:haircare_two_stage_stage2_not_improved", markdown)
        self.assertIn("runtime enablement allowed: `False`", markdown)
        self.assertIn("skincare/mask", markdown)

    def test_skincare_freeze_qualification_command_writes_report(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            cached_path = root / "comparison.json"
            out_stem = root / "roadmap_nextstep_v5_skincare_freeze_qualification"
            active_model_path = str((root / "active" / "model.pkl").resolve())
            retrain_model_path = str((root / "retrain" / "model.pkl").resolve())
            candidate_model_path = str((root / "candidate" / "model.pkl").resolve())
            _write_json(
                cached_path,
                self._cached_comparison_payload(
                    active_model_path=active_model_path,
                    retrain_model_path=retrain_model_path,
                    candidate_model_path=candidate_model_path,
                ),
            )
            stdout = StringIO()
            with override_settings(
                ROADMAP_RUNTIME_FREEZE_ML=True,
                ROADMAP_NEXTSTEP_V4_MODEL_PATH=active_model_path,
            ):
                call_command(
                    "report_roadmap_nextstep_v5_skincare_freeze_qualification",
                    active_model_path=active_model_path,
                    retrain_v1_model_path=retrain_model_path,
                    candidate_model_path=candidate_model_path,
                    source_preference="cached_artifact",
                    cached_comparison_json=str(cached_path),
                    out=str(out_stem),
                    format="both",
                    stdout=stdout,
                )

            self.assertTrue(out_stem.with_suffix(".md").exists())
            self.assertTrue(out_stem.with_suffix(".json").exists())
            markdown = out_stem.with_suffix(".md").read_text(encoding="utf-8")
            self.assertIn("lane category: `skincare`", markdown)
            self.assertIn("continue under freeze: `True`", markdown)


class RoadmapNextstepHaircareShampooGateTests(SimpleTestCase):
    def _model_meta(self, *, model_path: str, anchor_key: str, model_type: str, baseline_type: str) -> dict:
        normalized_path = normalized_model_path(model_path)
        return {
            "ml": {
                HISTORICAL_SHADOW_EVIDENCE_KEY: {
                    normalized_path: {
                        anchor_key: {
                            "model_path": normalized_path,
                            "was_model_selected": True,
                            "top1_product_type": model_type,
                        }
                    }
                },
                HISTORICAL_CONTROL_EVIDENCE_KEY: {
                    normalized_path: {
                        anchor_key: {
                            "model_path": normalized_path,
                            "was_control_selected": True,
                            "selected_product_type": baseline_type,
                        }
                    }
                },
            }
        }

    def _anchor(
        self,
        *,
        anchor_key: str,
        step_id: int,
        truth_product_type: str,
        next_refresh_at: str = "2026-04-09T00:00:00+00:00",
    ) -> tuple[dict, dict[int, list[dict]]]:
        anchor = {
            "anchor_key": anchor_key,
            "anchor_event_id": 1001,
            "anchor_created_at": "2026-04-08T00:00:00+00:00",
            "plan_id": 1,
            "category": "haircare",
            "anchor_has_actionable_step": True,
            "anchor_next_step_id": step_id,
            "anchor_next_step_index": 1,
            "anchor_next_product_type": "shampoo",
            "next_refresh_at": next_refresh_at,
            "reconstruction_reason": "",
            "generated_step_ids": [step_id],
            "generated_candidates": [
                {
                    "event_id": 2001,
                    "step_id": step_id,
                    "step_index": 1,
                    "product_type": "shampoo",
                    "created_at": "2026-04-08T00:01:00+00:00",
                    "is_generated": True,
                }
            ],
        }
        completions = {
            step_id: [
                {
                    "id": 3001,
                    "step_id": step_id,
                    "created_at": "2026-04-08T00:10:00+00:00",
                    "context": {
                        "product_type": truth_product_type,
                        "matched_by": "recommended_product_id",
                    },
                }
            ]
        }
        return anchor, completions

    def test_shampoo_historical_anchors_are_recoverable_when_present(self):
        model_path = str(Path("models/test_shampoo_model.pkl").resolve())
        anchor, completions = self._anchor(
            anchor_key="plan_refresh:1",
            step_id=11,
            truth_product_type="hair_mask",
        )
        payload = _analyze_single_model_shampoo_gate(
            model_path=model_path,
            anchors=[anchor],
            meta_by_plan={
                1: self._model_meta(
                    model_path=model_path,
                    anchor_key="plan_refresh:1",
                    model_type="hair_mask",
                    baseline_type="shampoo",
                )
            },
            completions_by_step=completions,
        )

        self.assertEqual(payload["anchors_scanned_total"], 1)
        self.assertEqual(payload["resolved_comparable_anchors_total"], 1)
        self.assertEqual(payload["resolved_truth_distribution"]["hair_mask"], 1)
        self.assertEqual(payload["resolved_outcome_matrix"]["model_wins"], 1)

    def test_shampoo_conditioner_pair_truth_is_surfaced_when_reconstructable(self):
        model_path = str(Path("models/test_shampoo_pair_model.pkl").resolve())
        anchor, completions = self._anchor(
            anchor_key="plan_refresh:2",
            step_id=12,
            truth_product_type="shampoo",
        )
        payload = _analyze_single_model_shampoo_gate(
            model_path=model_path,
            anchors=[anchor],
            meta_by_plan={
                1: self._model_meta(
                    model_path=model_path,
                    anchor_key="plan_refresh:2",
                    model_type="conditioner",
                    baseline_type="shampoo",
                )
            },
            completions_by_step=completions,
        )

        self.assertEqual(payload["resolved_shampoo_conditioner_comparable_rows_total"], 1)
        self.assertEqual(payload["exact_harmful_shampoo_conditioner_rows_total"], 1)
        self.assertEqual(payload["verdict"]["status"], "not_closed_because_model_still_loses")

    def test_unresolved_shampoo_anchors_are_counted_by_explicit_reason(self):
        model_path = str(Path("models/test_shampoo_unresolved_model.pkl").resolve())
        anchor_incomplete, completions_incomplete = self._anchor(
            anchor_key="plan_refresh:3",
            step_id=13,
            truth_product_type="shampoo",
            next_refresh_at=None,
        )
        anchor_no_completion = dict(anchor_incomplete)
        anchor_no_completion.update(
            {
                "anchor_key": "plan_refresh:4",
                "plan_id": 2,
                "anchor_event_id": 1002,
                "next_refresh_at": "2026-04-09T00:00:00+00:00",
                "anchor_next_step_id": 14,
                "generated_step_ids": [14],
                "generated_candidates": [
                    {
                        "event_id": 2002,
                        "step_id": 14,
                        "step_index": 1,
                        "product_type": "shampoo",
                        "created_at": "2026-04-08T00:01:00+00:00",
                        "is_generated": True,
                    }
                ],
            }
        )
        payload = _analyze_single_model_shampoo_gate(
            model_path=model_path,
            anchors=[anchor_incomplete, anchor_no_completion],
            meta_by_plan={
                1: self._model_meta(
                    model_path=model_path,
                    anchor_key="plan_refresh:3",
                    model_type="hair_mask",
                    baseline_type="shampoo",
                ),
                2: self._model_meta(
                    model_path=model_path,
                    anchor_key="plan_refresh:4",
                    model_type="hair_mask",
                    baseline_type="shampoo",
                ),
            },
            completions_by_step=completions_incomplete,
        )

        self.assertEqual(payload["unresolved_anchors_total"], 2)
        self.assertEqual(payload["unresolved_anchors_by_reason"]["incomplete_refresh_window"], 1)
        self.assertEqual(payload["unresolved_anchors_by_reason"]["no_completed_generated_candidate"], 1)

    def test_shampoo_gate_fails_closed_when_no_defensible_pair_truth_exists(self):
        model_path = str(Path("models/test_shampoo_missing_truth_model.pkl").resolve())
        anchor, completions = self._anchor(
            anchor_key="plan_refresh:5",
            step_id=15,
            truth_product_type="hair_oil",
        )
        payload = _analyze_single_model_shampoo_gate(
            model_path=model_path,
            anchors=[anchor],
            meta_by_plan={
                1: self._model_meta(
                    model_path=model_path,
                    anchor_key="plan_refresh:5",
                    model_type="hair_oil",
                    baseline_type="shampoo",
                )
            },
            completions_by_step=completions,
        )

        self.assertEqual(payload["standalone_shampoo_truth_rows_total"], 0)
        self.assertEqual(payload["resolved_shampoo_conditioner_comparable_rows_total"], 0)
        self.assertEqual(payload["verdict"]["status"], "not_closed_because_missing_truth")

    def test_shampoo_gate_path_does_not_modify_catalog(self):
        model_path = str(Path("models/test_catalog_safe_model.pkl").resolve())
        anchor, completions = self._anchor(
            anchor_key="plan_refresh:6",
            step_id=16,
            truth_product_type="hair_mask",
        )

        class _ValuesWrapper:
            def values(self, *args, **kwargs):
                return [{"id": 1, "meta": self.meta}]

            def __init__(self, meta):
                self.meta = meta

        with patch(
            "roadmap_app.nextstep_haircare_shampoo_gate.build_historical_continuation_anchor_records",
            return_value=[anchor],
        ), patch(
            "roadmap_app.nextstep_haircare_shampoo_gate.completion_events_by_step",
            return_value=completions,
        ), patch(
            "roadmap_app.nextstep_haircare_shampoo_gate.RoadmapPlan.objects.filter",
            return_value=_ValuesWrapper(
                self._model_meta(
                    model_path=model_path,
                    anchor_key="plan_refresh:6",
                    model_type="hair_mask",
                    baseline_type="shampoo",
                )
            ),
        ), patch(
            "catalog.models.Product.save",
            side_effect=AssertionError("catalog write should not happen"),
        ), patch(
            "catalog.models.Product.delete",
            side_effect=AssertionError("catalog delete should not happen"),
        ):
            payload = build_nextstep_haircare_shampoo_gate_payload(
                model_path=model_path,
                reference_model_path=model_path,
                days=30,
                include_ga=False,
            )

        self.assertFalse(payload["catalog_safety"]["catalog_writes_performed"])


class RoadmapNextstepHaircareShampooTruthDesignTests(SimpleTestCase):
    def _row(
        self,
        *,
        anchor_key: str = "plan_refresh:1",
        anchor_truth_type: str = "shampoo",
        baseline_type: str = "shampoo",
        model_type: str = "hair_mask",
        truth_type: str = "hair_mask",
        truth_resolved: bool = True,
        truth_reason: str = "ok",
        structural_reason: str = "",
        pair_available: bool = True,
    ) -> dict:
        family_map = {
            "shampoo": "repeat_shampoo",
            "conditioner": "pair_conditioner",
            "hair_mask": "downstream_treatment",
            "hair_oil": "downstream_treatment",
            "leave_in": "downstream_treatment",
            "scalp_serum": "downstream_treatment",
        }
        return {
            "anchor_key": anchor_key,
            "anchor_next_product_type": anchor_truth_type,
            "baseline_selected_product_type": baseline_type,
            "model_top1_product_type": model_type,
            "truth_selected_product_type": truth_type if truth_resolved else "",
            "truth_resolved": truth_resolved,
            "truth_reason": truth_reason if not truth_resolved else "ok",
            "truth_transition_family": family_map.get(truth_type, "other") if truth_resolved else "",
            "baseline_transition_family": family_map.get(baseline_type, "other"),
            "model_transition_family": family_map.get(model_type, "other"),
            "pair_available": pair_available,
            "comparability_exclusion_reason": "" if pair_available else "pair_mapping_unavailable",
            "structural_exclusion_reason": structural_reason,
        }

    def _model_meta(self, *, model_path: str, anchor_key: str, model_type: str, baseline_type: str) -> dict:
        normalized_path = normalized_model_path(model_path)
        return {
            "ml": {
                HISTORICAL_SHADOW_EVIDENCE_KEY: {
                    normalized_path: {
                        anchor_key: {
                            "model_path": normalized_path,
                            "was_model_selected": True,
                            "top1_product_type": model_type,
                        }
                    }
                },
                HISTORICAL_CONTROL_EVIDENCE_KEY: {
                    normalized_path: {
                        anchor_key: {
                            "model_path": normalized_path,
                            "was_control_selected": True,
                            "selected_product_type": baseline_type,
                        }
                    }
                },
            }
        }

    def _anchor(
        self,
        *,
        anchor_key: str,
        step_id: int,
        truth_product_type: str,
        next_refresh_at: str = "2026-04-09T00:00:00+00:00",
    ) -> tuple[dict, dict[int, list[dict]]]:
        anchor = {
            "anchor_key": anchor_key,
            "anchor_event_id": 2001,
            "anchor_created_at": "2026-04-08T00:00:00+00:00",
            "plan_id": 1,
            "category": "haircare",
            "anchor_has_actionable_step": True,
            "anchor_next_step_id": step_id,
            "anchor_next_step_index": 1,
            "anchor_next_product_type": "shampoo",
            "next_refresh_at": next_refresh_at,
            "reconstruction_reason": "",
            "generated_step_ids": [step_id],
            "generated_candidates": [
                {
                    "event_id": 2101,
                    "step_id": step_id,
                    "step_index": 1,
                    "product_type": "shampoo",
                    "created_at": "2026-04-08T00:01:00+00:00",
                    "is_generated": True,
                }
            ],
        }
        completions = {
            step_id: [
                {
                    "id": 2201,
                    "step_id": step_id,
                    "created_at": "2026-04-08T00:10:00+00:00",
                    "context": {
                        "product_type": truth_product_type,
                        "matched_by": "recommended_product_id",
                    },
                }
            ]
        }
        return anchor, completions

    def test_current_gate_truth_design_reproduces_fail_closed_behavior(self):
        rows = [
            self._row(anchor_key="plan_refresh:1", model_type="hair_mask", truth_type="hair_mask"),
            self._row(anchor_key="plan_refresh:2", model_type="hair_oil", truth_type="hair_oil"),
        ]
        payload = evaluate_haircare_shampoo_truth_designs(rows)

        self.assertEqual(payload["current_gate"]["verdict"]["status"], "fail_closed_missing_truth")
        self.assertEqual(payload["current_gate"]["standalone_shampoo_truth_rows_total"], 0)
        self.assertEqual(payload["current_gate"]["resolved_shampoo_conditioner_comparable_rows_total"], 0)

    def test_alternative_truth_designs_produce_explicit_resolved_and_unresolved_counts(self):
        rows = [
            self._row(anchor_key="plan_refresh:1", model_type="hair_mask", truth_type="hair_mask"),
            self._row(
                anchor_key="plan_refresh:2",
                model_type="hair_mask",
                truth_type="hair_mask",
                truth_resolved=False,
                truth_reason="no_completed_generated_candidate",
            ),
        ]
        payload = evaluate_haircare_shampoo_truth_designs(rows)
        designs = payload["designs"]

        self.assertEqual(designs["A_anchor_step_correctness"]["resolved_anchors_total"], 2)
        self.assertEqual(designs["A_anchor_step_correctness"]["unresolved_anchors_total"], 0)
        self.assertEqual(designs["B_immediate_pair_closure"]["resolved_anchors_total"], 1)
        self.assertEqual(designs["B_immediate_pair_closure"]["unresolved_anchors_by_reason"]["no_completed_generated_candidate"], 1)
        self.assertEqual(designs["C_downstream_treatment_truth"]["resolved_anchors_total"], 1)
        self.assertEqual(designs["D_two_stage_truth"]["resolved_anchors_total"], 1)

    def test_recommended_truth_design_is_measurable_from_immutable_data(self):
        rows = [
            self._row(anchor_key="plan_refresh:1", model_type="hair_mask", truth_type="hair_mask"),
            self._row(anchor_key="plan_refresh:2", model_type="hair_oil", truth_type="hair_oil"),
        ]
        payload = evaluate_haircare_shampoo_truth_designs(rows)

        self.assertEqual(payload["recommendation"]["recommended_truth_design"], "D_two_stage_truth")
        self.assertTrue(payload["recommendation"]["rerun_v5_comparison_under_recommended_truth"])

    @override_settings(ROADMAP_NEXTSTEP_V4_MODEL_PATH="models/original_active.pkl")
    def test_truth_design_payload_does_not_modify_catalog_or_runtime_config(self):
        model_path = str(Path("models/test_truth_design_model.pkl").resolve())
        anchor, completions = self._anchor(
            anchor_key="plan_refresh:7",
            step_id=17,
            truth_product_type="hair_mask",
        )

        class _ValuesWrapper:
            def values(self, *args, **kwargs):
                return [{"id": 1, "meta": self.meta}]

            def __init__(self, meta):
                self.meta = meta

        before_model_path = settings.ROADMAP_NEXTSTEP_V4_MODEL_PATH
        with patch(
            "roadmap_app.nextstep_historical_anchor_context.build_historical_continuation_anchor_records",
            return_value=[anchor],
        ), patch(
            "roadmap_app.nextstep_historical_anchor_context.completion_events_by_step",
            return_value=completions,
        ), patch(
            "roadmap_app.nextstep_historical_anchor_context.RoadmapPlan.objects.filter",
            return_value=_ValuesWrapper(
                self._model_meta(
                    model_path=model_path,
                    anchor_key="plan_refresh:7",
                    model_type="hair_mask",
                    baseline_type="shampoo",
                )
            ),
        ), patch(
            "catalog.models.Product.save",
            side_effect=AssertionError("catalog write should not happen"),
        ), patch(
            "catalog.models.Product.delete",
            side_effect=AssertionError("catalog delete should not happen"),
        ):
            payload = build_nextstep_haircare_shampoo_truth_design_payload(
                model_path=model_path,
                reference_model_path=model_path,
                days=30,
                include_ga=False,
            )

        self.assertFalse(payload["catalog_safety"]["catalog_writes_performed"])
        self.assertEqual(settings.ROADMAP_NEXTSTEP_V4_MODEL_PATH, before_model_path)

    def test_truth_design_payload_reuses_provided_historical_context_without_extra_db_reads(self):
        model_path = str(Path("models/test_truth_design_shared_context_model.pkl").resolve())
        anchor, completions = self._anchor(
            anchor_key="plan_refresh:8",
            step_id=18,
            truth_product_type="hair_mask",
        )
        historical_context = {
            "since": timezone.now() - timedelta(days=30),
            "until": timezone.now(),
            "category": "all",
            "include_ga": False,
            "anchors": [anchor],
            "meta_by_plan": {
                1: self._model_meta(
                    model_path=model_path,
                    anchor_key="plan_refresh:8",
                    model_type="hair_mask",
                    baseline_type="shampoo",
                )
            },
            "completions_by_step": completions,
            "read_only": True,
        }

        with patch(
            "roadmap_app.nextstep_historical_anchor_context.build_historical_continuation_anchor_records",
            side_effect=AssertionError("historical anchors should be reused from provided context"),
        ), patch(
            "roadmap_app.nextstep_historical_anchor_context.RoadmapPlan.objects.filter",
            side_effect=AssertionError("plan meta should be reused from provided context"),
        ), patch(
            "roadmap_app.nextstep_historical_anchor_context.completion_events_by_step",
            side_effect=AssertionError("step completions should be reused from provided context"),
        ):
            payload = build_nextstep_haircare_shampoo_truth_design_payload(
                model_path=model_path,
                reference_model_path=model_path,
                days=30,
                include_ga=False,
                historical_context=historical_context,
            )

        self.assertEqual(
            payload["candidate"]["truth_designs"]["designs"]["D_two_stage_truth"]["resolved_anchors_total"],
            1,
        )
        self.assertFalse(payload["catalog_safety"]["catalog_writes_performed"])


class RoadmapMLPredictTimingTests(SimpleTestCase):
    def test_timed_predict_success_returns_ms_and_no_error(self):
        from roadmap_app.services import _timed_predict

        result, ms, error = _timed_predict(lambda: [{"product_type": "shampoo", "score": 0.7}])
        self.assertEqual(result, [{"product_type": "shampoo", "score": 0.7}])
        self.assertIsNone(error)
        self.assertIsInstance(ms, float)
        self.assertGreaterEqual(ms, 0.0)

    def test_timed_predict_exception_returns_none_with_error(self):
        from roadmap_app.services import _timed_predict

        def boom():
            raise RuntimeError("predict blew up")

        result, ms, error = _timed_predict(boom)
        self.assertIsNone(result)
        self.assertEqual(error, "predict blew up")
        self.assertIsInstance(ms, float)
        self.assertGreaterEqual(ms, 0.0)

    def test_timed_predict_truncates_long_error_to_500_chars(self):
        from roadmap_app.services import _timed_predict

        long_message = "x" * 2000

        def boom():
            raise ValueError(long_message)

        _, _, error = _timed_predict(boom)
        self.assertIsNotNone(error)
        self.assertEqual(len(error), 500)
        self.assertEqual(error, "x" * 500)

    def test_normalize_plan_meta_fills_timing_defaults_on_legacy_meta(self):
        from roadmap_app.services import _normalize_plan_meta

        legacy_meta = {
            "ml": {
                "mode": "v4_ranking",
                "model_path": "/fake/model.pkl",
                "decision": "model_used",
                "used": True,
                "shadow": {
                    "enabled": False,
                    "reason": "shadow_not_configured",
                },
            }
        }
        normalized = _normalize_plan_meta(legacy_meta)
        self.assertIsNone(normalized["ml"]["predict_ms"])
        self.assertIsNone(normalized["ml"]["predict_error"])
        self.assertIsNone(normalized["ml"]["shadow"]["predict_ms"])
        self.assertIsNone(normalized["ml"]["shadow"]["predict_error"])

    def test_normalize_plan_meta_preserves_timing_values_when_present(self):
        from roadmap_app.services import _normalize_plan_meta

        meta = {
            "ml": {
                "mode": "v4_ranking",
                "model_path": "/fake/model.pkl",
                "decision": "model_used",
                "used": True,
                "predict_ms": 12.5,
                "predict_error": None,
                "shadow": {
                    "enabled": True,
                    "reason": "ok",
                    "predict_ms": 9.25,
                    "predict_error": "timeout",
                },
            }
        }
        normalized = _normalize_plan_meta(meta)
        self.assertEqual(normalized["ml"]["predict_ms"], 12.5)
        self.assertIsNone(normalized["ml"]["predict_error"])
        self.assertEqual(normalized["ml"]["shadow"]["predict_ms"], 9.25)
        self.assertEqual(normalized["ml"]["shadow"]["predict_error"], "timeout")


class RoadmapMLInvocationRecordTests(TestCase):
    @classmethod
    def setUpTestData(cls):
        User = get_user_model()
        cls.user = User.objects.create(username="ml_invocation_user_1")
        cls.plan = RoadmapPlan.objects.create(
            user=cls.user,
            category=RoadmapPlan.Category.SKINCARE,
            is_active=True,
            meta={},
        )

    def _sample_meta(self, **overrides):
        base = {
            "ml": {
                "mode": "v4_ranking",
                "decision": "model_used",
                "fallback_reason": None,
                "disabled_reason": None,
                "model_path": "/models/roadmap_next_step_v4/model.pkl",
                "model_version": "roadmap_next_step_v4",
                "selected_feature_set": "full",
                "model_slot": "active",
                "predict_ms": 12.5,
                "predict_error": None,
                "rollout_mode": "partial",
                "rollout_selected": True,
                "rollout_bucket": 17,
                "rollout_percent": 20,
                "planned_target_product_type": "moisturizer",
                "planned_target_step_index": 3,
                "predictions": [
                    {"product_type": "moisturizer", "score": 0.82},
                    {"product_type": "serum", "score": 0.55},
                ],
                "shadow": {
                    "enabled": True,
                    "reason": "ok",
                    "model_path": "/models/shadow/model.pkl",
                    "model_version": "shadow_v1",
                    "predict_ms": 9.1,
                    "predict_error": None,
                    "predictions": [{"product_type": "serum", "score": 0.71}],
                },
            },
            "planner": {
                "mode": "serve",
                "served": True,
                "decision": "model_used",
                "model_path": "/models/roadmap_planner_v1/model.pkl",
                "predict_ms": 6.4,
                "predict_error": None,
            },
        }
        ml = base["ml"]
        for k, v in overrides.items():
            ml[k] = v
        return base

    def test_records_model_used_decision_with_shadow_and_planner(self):
        from roadmap_app.services import _record_ml_invocation

        _record_ml_invocation(
            user=self.user,
            plan=self.plan,
            category="skincare",
            refresh_caller="refresh_roadmap",
            meta=self._sample_meta(),
        )
        row = RoadmapMLInvocation.objects.get()
        self.assertEqual(row.user_id, self.user.id)
        self.assertEqual(row.plan_id, self.plan.id)
        self.assertEqual(row.category, "skincare")
        self.assertEqual(row.refresh_caller, "refresh_roadmap")
        self.assertEqual(row.ml_mode, "v4_ranking")
        self.assertEqual(row.decision, "model_used")
        self.assertEqual(row.fallback_reason, "")
        self.assertEqual(row.disabled_reason, "")
        self.assertEqual(row.model_version, "roadmap_next_step_v4")
        self.assertEqual(row.predict_ms, 12.5)
        self.assertEqual(row.predict_error, "")
        self.assertEqual(row.active_top_product_type, "moisturizer")
        self.assertAlmostEqual(row.active_top_score, 0.82)
        self.assertTrue(row.shadow_enabled)
        self.assertEqual(row.shadow_reason, "ok")
        self.assertEqual(row.shadow_model_version, "shadow_v1")
        self.assertEqual(row.shadow_predict_ms, 9.1)
        self.assertEqual(row.shadow_top_product_type, "serum")
        self.assertAlmostEqual(row.shadow_top_score, 0.71)
        self.assertEqual(row.planner_mode, "serve")
        self.assertTrue(row.planner_served)
        self.assertEqual(row.planner_decision, "model_used")
        self.assertEqual(row.planner_predict_ms, 6.4)
        self.assertEqual(row.rollout_mode, "partial")
        self.assertTrue(row.rollout_selected)
        self.assertEqual(row.rollout_bucket, 17)
        self.assertEqual(row.rollout_percent, 20)
        self.assertEqual(row.planned_target_product_type, "moisturizer")
        self.assertEqual(row.planned_target_step_index, 3)

    def test_records_fallback_with_predict_error(self):
        from roadmap_app.services import _record_ml_invocation

        meta = self._sample_meta(
            decision="fallback",
            fallback_reason="predict_error",
            predict_error="RuntimeError: boom",
            predict_ms=4.2,
            predictions=[],
        )
        _record_ml_invocation(
            user=self.user,
            plan=self.plan,
            category="skincare",
            refresh_caller="update_roadmap_from_purchase",
            meta=meta,
        )
        row = RoadmapMLInvocation.objects.get()
        self.assertEqual(row.decision, "fallback")
        self.assertEqual(row.fallback_reason, "predict_error")
        self.assertEqual(row.predict_error, "RuntimeError: boom")
        self.assertEqual(row.predict_ms, 4.2)
        self.assertEqual(row.active_top_product_type, "")
        self.assertIsNone(row.active_top_score)

    def test_records_disabled_when_frozen(self):
        from roadmap_app.services import _record_ml_invocation

        meta = self._sample_meta(
            decision="disabled",
            disabled_reason="roadmap_ml_frozen",
            predictions=[],
        )
        meta["ml"]["shadow"] = {"enabled": False, "reason": "disabled"}
        _record_ml_invocation(
            user=self.user,
            plan=self.plan,
            category="skincare",
            refresh_caller="refresh_roadmap",
            meta=meta,
        )
        row = RoadmapMLInvocation.objects.get()
        self.assertEqual(row.decision, "disabled")
        self.assertEqual(row.disabled_reason, "roadmap_ml_frozen")
        self.assertFalse(row.shadow_enabled)
        self.assertEqual(row.shadow_reason, "disabled")

    def test_feature_flag_off_skips_write(self):
        from roadmap_app.services import _record_ml_invocation

        with override_settings(ROADMAP_ML_INVOCATION_LOG_ENABLED=False):
            _record_ml_invocation(
                user=self.user,
                plan=self.plan,
                category="skincare",
                refresh_caller="refresh_roadmap",
                meta=self._sample_meta(),
            )
        self.assertEqual(RoadmapMLInvocation.objects.count(), 0)

    def test_handles_empty_or_missing_meta_without_crashing(self):
        from roadmap_app.services import _record_ml_invocation

        _record_ml_invocation(
            user=self.user,
            plan=self.plan,
            category="haircare",
            refresh_caller="",
            meta=None,
        )
        row = RoadmapMLInvocation.objects.get()
        self.assertEqual(row.category, "haircare")
        self.assertEqual(row.decision, "")
        self.assertEqual(row.ml_mode, "")
        self.assertIsNone(row.predict_ms)
        self.assertFalse(row.shadow_enabled)
        self.assertFalse(row.planner_served)

    def test_truncates_overly_long_string_fields(self):
        from roadmap_app.services import _record_ml_invocation

        meta = self._sample_meta(
            predict_error="x" * 1000,
            model_path="y" * 1000,
        )
        _record_ml_invocation(
            user=self.user,
            plan=self.plan,
            category="skincare",
            refresh_caller="refresh_roadmap",
            meta=meta,
        )
        row = RoadmapMLInvocation.objects.get()
        self.assertEqual(len(row.predict_error), 512)
        self.assertEqual(len(row.model_path), 512)

    def test_telemetry_failure_does_not_raise(self):
        from roadmap_app import services as services_module

        def boom(**kwargs):
            raise RuntimeError("db unreachable")

        with patch.object(services_module.RoadmapMLInvocation.objects, "create", side_effect=boom):
            services_module._record_ml_invocation(
                user=self.user,
                plan=self.plan,
                category="skincare",
                refresh_caller="refresh_roadmap",
                meta=self._sample_meta(),
            )
        self.assertEqual(RoadmapMLInvocation.objects.count(), 0)


class RoadmapRuntimeConfigTests(TestCase):
    """Runtime config table + helper functions in roadmap_app.runtime_config."""

    def setUp(self):
        from roadmap_app import runtime_config

        runtime_config.invalidate_cache()
        from roadmap_app.models import RoadmapRuntimeConfig

        RoadmapRuntimeConfig.objects.all().delete()

    def test_get_bool_returns_default_when_no_override(self):
        from roadmap_app import runtime_config

        self.assertTrue(runtime_config.get_bool("missing_key", default=True))
        self.assertFalse(runtime_config.get_bool("missing_key", default=False))

    def test_set_and_read_bool_value_overrides_default(self):
        from roadmap_app import runtime_config

        runtime_config.set_value("feature_x", "false")
        self.assertFalse(runtime_config.get_bool("feature_x", default=True))
        runtime_config.set_value("feature_x", "true")
        self.assertTrue(runtime_config.get_bool("feature_x", default=False))

    def test_get_bool_accepts_various_truthy_tokens(self):
        from roadmap_app import runtime_config

        for token in ["1", "true", "YES", " on ", "T", "y"]:
            runtime_config.set_value("flag", token)
            self.assertTrue(
                runtime_config.get_bool("flag", default=False),
                f"expected truthy for {token!r}",
            )

    def test_get_bool_invalid_value_returns_default(self):
        from roadmap_app import runtime_config

        runtime_config.set_value("flag", "banana")
        self.assertTrue(runtime_config.get_bool("flag", default=True))
        self.assertFalse(runtime_config.get_bool("flag", default=False))

    def test_get_int_parses_and_falls_back(self):
        from roadmap_app import runtime_config

        runtime_config.set_value("percent", "42")
        self.assertEqual(runtime_config.get_int("percent", default=0), 42)
        runtime_config.set_value("percent", "nope")
        self.assertEqual(runtime_config.get_int("percent", default=7), 7)
        self.assertEqual(runtime_config.get_int("missing", default=9), 9)

    def test_unset_removes_override(self):
        from roadmap_app import runtime_config

        runtime_config.set_value("k", "v")
        self.assertEqual(runtime_config.get_str("k", default=""), "v")
        self.assertTrue(runtime_config.unset_value("k"))
        self.assertEqual(runtime_config.get_str("k", default="default"), "default")
        self.assertFalse(runtime_config.unset_value("k"))

    def test_list_values_reflects_writes(self):
        from roadmap_app import runtime_config

        runtime_config.set_value("a", "1")
        runtime_config.set_value("b", "2")
        self.assertEqual(runtime_config.list_values(), {"a": "1", "b": "2"})

    def test_set_value_rejects_empty_key(self):
        from roadmap_app import runtime_config

        with self.assertRaises(ValueError):
            runtime_config.set_value("", "value")
        with self.assertRaises(ValueError):
            runtime_config.set_value("   ", "value")

    def test_set_value_rejects_key_over_64_chars(self):
        from roadmap_app import runtime_config

        with self.assertRaises(ValueError):
            runtime_config.set_value("x" * 65, "value")

    def test_set_value_truncates_updated_by_and_note(self):
        from roadmap_app import runtime_config
        from roadmap_app.models import RoadmapRuntimeConfig

        runtime_config.set_value(
            "k",
            "v",
            updated_by="u" * 300,
            note="n" * 400,
        )
        row = RoadmapRuntimeConfig.objects.get(key="k")
        self.assertEqual(len(row.updated_by), 128)
        self.assertEqual(len(row.note), 256)

    def test_cache_invalidated_on_set_and_unset(self):
        from roadmap_app import runtime_config

        runtime_config.set_value("k", "1")
        self.assertEqual(runtime_config.get_str("k"), "1")
        runtime_config.set_value("k", "2")
        self.assertEqual(runtime_config.get_str("k"), "2")
        runtime_config.unset_value("k")
        self.assertEqual(runtime_config.get_str("k", default="gone"), "gone")

    def test_is_runtime_ml_frozen_prefers_override_over_settings(self):
        from roadmap_app import runtime_config

        with override_settings(ROADMAP_RUNTIME_FREEZE_ML=True):
            runtime_config.invalidate_cache()
            self.assertTrue(runtime_config.is_runtime_ml_frozen())
            runtime_config.set_value(runtime_config.FREEZE_KEY, "false")
            self.assertFalse(runtime_config.is_runtime_ml_frozen())
            runtime_config.set_value(runtime_config.FREEZE_KEY, "true")
            self.assertTrue(runtime_config.is_runtime_ml_frozen())

    def test_is_runtime_ml_frozen_falls_back_to_settings_when_no_override(self):
        from roadmap_app import runtime_config

        with override_settings(ROADMAP_RUNTIME_FREEZE_ML=False):
            runtime_config.invalidate_cache()
            self.assertFalse(runtime_config.is_runtime_ml_frozen())
        with override_settings(ROADMAP_RUNTIME_FREEZE_ML=True):
            runtime_config.invalidate_cache()
            self.assertTrue(runtime_config.is_runtime_ml_frozen())

    def test_is_ml_invocation_log_enabled_prefers_override(self):
        from roadmap_app import runtime_config

        with override_settings(ROADMAP_ML_INVOCATION_LOG_ENABLED=True):
            runtime_config.invalidate_cache()
            self.assertTrue(runtime_config.is_ml_invocation_log_enabled())
            runtime_config.set_value(runtime_config.LOG_ENABLED_KEY, "false")
            self.assertFalse(runtime_config.is_ml_invocation_log_enabled())

    def test_services_freeze_reads_runtime_config_override(self):
        from roadmap_app import runtime_config, services as services_module

        with override_settings(
            ROADMAP_RUNTIME_FREEZE_ML=True,
            ROADMAP_NEXTSTEP_V4_ENABLED=True,
            ROADMAP_NEXTSTEP_V4_MODEL_PATH="models/fake/model.pkl",
        ):
            runtime_config.invalidate_cache()
            mode, _ = services_module._default_ml_mode_and_path()
            self.assertEqual(mode, "legacy")

            runtime_config.set_value(runtime_config.FREEZE_KEY, "false")
            mode_after, _ = services_module._default_ml_mode_and_path()
            self.assertEqual(mode_after, "v4_ranking")


class RoadmapRuntimeConfigCommandTests(TestCase):
    def setUp(self):
        from roadmap_app import runtime_config
        from roadmap_app.models import RoadmapRuntimeConfig

        RoadmapRuntimeConfig.objects.all().delete()
        runtime_config.invalidate_cache()

    def _call(self, *args):
        stdout = StringIO()
        call_command("roadmap_runtime_config", *args, stdout=stdout)
        return stdout.getvalue()

    def test_list_empty_prints_placeholder(self):
        out = self._call("--list")
        self.assertIn("(no runtime overrides set)", out)

    def test_set_writes_row_and_invalidates_cache(self):
        from roadmap_app import runtime_config

        out = self._call("--set", "runtime_freeze_ml=false", "--by", "ops", "--note", "rollback test")
        self.assertIn("set   runtime_freeze_ml=false", out)
        self.assertEqual(runtime_config.get_str("runtime_freeze_ml"), "false")

        from roadmap_app.models import RoadmapRuntimeConfig

        row = RoadmapRuntimeConfig.objects.get(key="runtime_freeze_ml")
        self.assertEqual(row.value, "false")
        self.assertEqual(row.updated_by, "ops")
        self.assertEqual(row.note, "rollback test")

    def test_set_multiple_keys_at_once(self):
        self._call("--set", "a=1", "b=two", "c=on")
        from roadmap_app import runtime_config

        values = runtime_config.list_values()
        self.assertEqual(values, {"a": "1", "b": "two", "c": "on"})

    def test_unset_removes_row(self):
        self._call("--set", "k=v")
        out = self._call("--unset", "k")
        self.assertIn("unset k", out)
        from roadmap_app.models import RoadmapRuntimeConfig

        self.assertFalse(RoadmapRuntimeConfig.objects.filter(key="k").exists())

    def test_unset_missing_key_is_noop(self):
        out = self._call("--unset", "nope")
        self.assertIn("noop", out)
        self.assertIn("nope", out)

    def test_set_without_equals_raises(self):
        from django.core.management.base import CommandError

        with self.assertRaises(CommandError):
            self._call("--set", "not_a_pair")

    def test_set_with_empty_key_raises(self):
        from django.core.management.base import CommandError

        with self.assertRaises(CommandError):
            self._call("--set", "=value")

    def test_list_shows_sorted_keys(self):
        self._call("--set", "zebra=1", "apple=2", "mango=3")
        out = self._call("--list")
        apple_idx = out.index("apple=")
        mango_idx = out.index("mango=")
        zebra_idx = out.index("zebra=")
        self.assertLess(apple_idx, mango_idx)
        self.assertLess(mango_idx, zebra_idx)

    def test_no_args_prints_current_values(self):
        self._call("--set", "k=v")
        out = self._call()
        self.assertIn("k=v", out)


class RoadmapMLRollbackGuardTests(TestCase):
    @classmethod
    def setUpTestData(cls):
        cls.user = get_user_model().objects.create_user(
            username="guard_user",
            email="guard@example.com",
            password="x",
        )
        cls.plan = RoadmapPlan.objects.create(
            user=cls.user,
            category=RoadmapPlan.Category.SKINCARE,
            is_active=True,
            version=1,
        )

    def setUp(self):
        from roadmap_app import runtime_config
        from roadmap_app.models import RoadmapRuntimeConfig

        RoadmapRuntimeConfig.objects.all().delete()
        runtime_config.invalidate_cache()
        RoadmapMLInvocation.objects.all().delete()

    def _make(self, *, category="skincare", decision="model_used", predict_ms=None, predict_error="", created_at=None):
        row = RoadmapMLInvocation.objects.create(
            user=self.user,
            plan=self.plan,
            category=category,
            decision=decision,
            predict_ms=predict_ms,
            predict_error=predict_error,
        )
        if created_at is not None:
            RoadmapMLInvocation.objects.filter(pk=row.pk).update(created_at=created_at)
            row.refresh_from_db()
        return row

    def _seed(self, count, *, category="skincare", decision="model_used", predict_ms=50.0, predict_error=""):
        for _ in range(count):
            self._make(
                category=category,
                decision=decision,
                predict_ms=predict_ms,
                predict_error=predict_error,
            )

    def test_empty_db_no_breach(self):
        from roadmap_app.ml_rollback_guard import evaluate_rollback_guard

        report = evaluate_rollback_guard()
        self.assertFalse(report["any_breach"])
        self.assertEqual(report["per_category"], {})

    def test_healthy_traffic_no_breach(self):
        from roadmap_app.ml_rollback_guard import evaluate_rollback_guard

        self._seed(80, predict_ms=40.0)
        report = evaluate_rollback_guard()
        payload = report["per_category"]["skincare"]
        self.assertEqual(payload["total"], 80)
        self.assertEqual(payload["predict_attempts"], 80)
        self.assertEqual(payload["errors"], 0)
        self.assertFalse(payload.get("insufficient_sample"))
        self.assertFalse(payload["breaches"])
        self.assertFalse(report["any_breach"])

    def test_high_error_rate_breaches(self):
        from roadmap_app.ml_rollback_guard import evaluate_rollback_guard

        self._seed(80, predict_ms=40.0)
        self._seed(20, predict_ms=None, predict_error="predict crashed")
        report = evaluate_rollback_guard()
        payload = report["per_category"]["skincare"]
        self.assertEqual(payload["predict_attempts"], 100)
        self.assertEqual(payload["errors"], 20)
        self.assertAlmostEqual(payload["error_rate_pct"], 20.0, places=2)
        breach_metrics = {b["metric"] for b in payload["breaches"]}
        self.assertIn("error_rate_pct", breach_metrics)
        self.assertTrue(report["any_breach"])

    def test_high_p95_latency_breaches(self):
        from roadmap_app.ml_rollback_guard import evaluate_rollback_guard

        for ms in list(range(50, 150)):
            self._make(predict_ms=float(ms + 500))
        report = evaluate_rollback_guard()
        payload = report["per_category"]["skincare"]
        self.assertIsNotNone(payload["p95_latency_ms"])
        self.assertGreater(payload["p95_latency_ms"], 500.0)
        breach_metrics = {b["metric"] for b in payload["breaches"]}
        self.assertIn("p95_latency_ms", breach_metrics)

    def test_high_fallback_rate_breaches(self):
        from roadmap_app.ml_rollback_guard import evaluate_rollback_guard

        self._seed(40, decision="model_used", predict_ms=30.0)
        self._seed(60, decision="fallback", predict_ms=30.0)
        report = evaluate_rollback_guard()
        payload = report["per_category"]["skincare"]
        self.assertEqual(payload["fallbacks"], 60)
        self.assertAlmostEqual(payload["fallback_rate_pct"], 60.0, places=2)
        breach_metrics = {b["metric"] for b in payload["breaches"]}
        self.assertIn("fallback_rate_pct", breach_metrics)

    def test_insufficient_sample_suppresses_breach(self):
        from roadmap_app.ml_rollback_guard import evaluate_rollback_guard

        self._seed(10, predict_ms=None, predict_error="boom")
        report = evaluate_rollback_guard()
        payload = report["per_category"]["skincare"]
        self.assertTrue(payload["insufficient_sample"])
        self.assertEqual(payload["breaches"], [])
        self.assertFalse(report["any_breach"])

    def test_disabled_invocations_do_not_count_as_attempts(self):
        from roadmap_app.ml_rollback_guard import evaluate_rollback_guard

        self._seed(100, decision="disabled", predict_ms=None, predict_error="")
        report = evaluate_rollback_guard()
        payload = report["per_category"]["skincare"]
        self.assertEqual(payload["total"], 100)
        self.assertEqual(payload["predict_attempts"], 0)
        self.assertTrue(payload["insufficient_sample"])
        self.assertFalse(report["any_breach"])

    def test_window_excludes_old_rows(self):
        from roadmap_app.ml_rollback_guard import evaluate_rollback_guard

        now = timezone.now()
        for _ in range(100):
            self._make(
                predict_ms=None,
                predict_error="old boom",
                created_at=now - timedelta(hours=2),
            )
        self._seed(10, predict_ms=20.0)
        report = evaluate_rollback_guard(window_minutes=15, now=now)
        payload = report["per_category"]["skincare"]
        self.assertEqual(payload["total"], 10)
        self.assertEqual(payload["errors"], 0)
        self.assertFalse(report["any_breach"])

    def test_per_category_isolation(self):
        from roadmap_app.ml_rollback_guard import evaluate_rollback_guard

        self._seed(80, category="skincare", predict_ms=30.0)
        self._seed(20, category="skincare", predict_ms=None, predict_error="boom")
        self._seed(80, category="haircare", predict_ms=30.0)
        report = evaluate_rollback_guard()
        self.assertTrue(report["per_category"]["skincare"]["breaches"])
        self.assertFalse(report["per_category"]["haircare"]["breaches"])
        self.assertTrue(report["any_breach"])

    def test_enforce_flips_freeze_on_breach(self):
        from roadmap_app import runtime_config
        from roadmap_app.ml_rollback_guard import enforce_rollback_guard

        with override_settings(ROADMAP_RUNTIME_FREEZE_ML=False):
            runtime_config.invalidate_cache()
            self._seed(80, predict_ms=40.0)
            self._seed(20, predict_ms=None, predict_error="boom")
            report = enforce_rollback_guard()
            self.assertFalse(report["frozen_before"])
            self.assertTrue(report["frozen_after"])
            self.assertEqual(report["action_taken"], "freeze_set")
            self.assertIn("skincare:error_rate_pct", report["freeze_note"])
            self.assertTrue(runtime_config.is_runtime_ml_frozen())

    def test_enforce_no_action_when_already_frozen(self):
        from roadmap_app import runtime_config
        from roadmap_app.ml_rollback_guard import enforce_rollback_guard

        with override_settings(ROADMAP_RUNTIME_FREEZE_ML=True):
            runtime_config.invalidate_cache()
            self._seed(80, predict_ms=40.0)
            self._seed(20, predict_ms=None, predict_error="boom")
            report = enforce_rollback_guard()
            self.assertTrue(report["frozen_before"])
            self.assertTrue(report["frozen_after"])
            self.assertEqual(report["action_taken"], "already_frozen")
            self.assertNotIn("freeze_note", report)

    def test_enforce_no_action_when_no_breach(self):
        from roadmap_app import runtime_config
        from roadmap_app.ml_rollback_guard import enforce_rollback_guard

        with override_settings(ROADMAP_RUNTIME_FREEZE_ML=False):
            runtime_config.invalidate_cache()
            self._seed(80, predict_ms=40.0)
            report = enforce_rollback_guard()
            self.assertFalse(report["frozen_before"])
            self.assertFalse(report["frozen_after"])
            self.assertEqual(report["action_taken"], "none")
            self.assertFalse(runtime_config.is_runtime_ml_frozen())

    def test_custom_thresholds_propagated(self):
        from roadmap_app.ml_rollback_guard import GuardThresholds, evaluate_rollback_guard

        self._seed(40, predict_ms=40.0)
        self._seed(10, predict_ms=None, predict_error="boom")
        report = evaluate_rollback_guard(
            thresholds=GuardThresholds(
                max_error_rate_pct=30.0,
                min_sample_size=20,
            )
        )
        payload = report["per_category"]["skincare"]
        self.assertFalse(payload["insufficient_sample"])
        self.assertEqual(payload["breaches"], [])
        self.assertFalse(report["any_breach"])


class RoadmapMLRollbackGuardCommandTests(TestCase):
    @classmethod
    def setUpTestData(cls):
        cls.user = get_user_model().objects.create_user(
            username="guard_cmd_user",
            email="guard_cmd@example.com",
            password="x",
        )
        cls.plan = RoadmapPlan.objects.create(
            user=cls.user,
            category=RoadmapPlan.Category.SKINCARE,
            is_active=True,
        )

    def setUp(self):
        from roadmap_app import runtime_config
        from roadmap_app.models import RoadmapRuntimeConfig

        RoadmapRuntimeConfig.objects.all().delete()
        runtime_config.invalidate_cache()
        RoadmapMLInvocation.objects.all().delete()

    def _seed_breach(self):
        for _ in range(80):
            RoadmapMLInvocation.objects.create(
                user=self.user, plan=self.plan, category="skincare",
                decision="model_used", predict_ms=40.0,
            )
        for _ in range(20):
            RoadmapMLInvocation.objects.create(
                user=self.user, plan=self.plan, category="skincare",
                decision="fallback", predict_error="boom",
            )

    def _call(self, *args):
        stdout = StringIO()
        call_command("roadmap_ml_rollback_guard", *args, stdout=stdout)
        return stdout.getvalue()

    def test_command_dry_run_does_not_mutate(self):
        from roadmap_app import runtime_config

        with override_settings(ROADMAP_RUNTIME_FREEZE_ML=False):
            runtime_config.invalidate_cache()
            self._seed_breach()
            out = self._call()
            self.assertIn("action=dry_run", out)
            self.assertIn("BREACH error_rate_pct", out)
            self.assertFalse(runtime_config.is_runtime_ml_frozen())

    def test_command_enforce_flips_freeze(self):
        from roadmap_app import runtime_config

        with override_settings(ROADMAP_RUNTIME_FREEZE_ML=False):
            runtime_config.invalidate_cache()
            self._seed_breach()
            out = self._call("--enforce", "--actor", "cron_guard")
            self.assertIn("action=freeze_set", out)
            self.assertTrue(runtime_config.is_runtime_ml_frozen())
            from roadmap_app.models import RoadmapRuntimeConfig

            row = RoadmapRuntimeConfig.objects.get(key=runtime_config.FREEZE_KEY)
            self.assertEqual(row.value, "true")
            self.assertEqual(row.updated_by, "cron_guard")

    def test_command_json_output(self):
        with override_settings(ROADMAP_RUNTIME_FREEZE_ML=True):
            from roadmap_app import runtime_config

            runtime_config.invalidate_cache()
            self._seed_breach()
            out = self._call("--json")
            parsed = json.loads(out)
            self.assertIn("per_category", parsed)
            self.assertIn("thresholds", parsed)
            self.assertTrue(parsed["any_breach"])

    def test_command_no_invocations_message(self):
        out = self._call()
        self.assertIn("(no invocations in window)", out)

    def test_command_custom_thresholds(self):
        for _ in range(40):
            RoadmapMLInvocation.objects.create(
                user=self.user, plan=self.plan, category="skincare",
                decision="model_used", predict_ms=40.0,
            )
        for _ in range(10):
            RoadmapMLInvocation.objects.create(
                user=self.user, plan=self.plan, category="skincare",
                decision="fallback", predict_error="boom",
            )
        out = self._call("--min-sample-size", "20", "--max-error-rate-pct", "30.0")
        self.assertIn("action=dry_run", out)
        self.assertNotIn("BREACH error_rate_pct", out)


class RoadmapMLDiffReportTests(TestCase):
    @classmethod
    def setUpTestData(cls):
        cls.user = get_user_model().objects.create_user(
            username="diff_user", email="diff@example.com", password="x",
        )
        cls.plan = RoadmapPlan.objects.create(
            user=cls.user, category=RoadmapPlan.Category.SKINCARE, is_active=True,
        )

    def setUp(self):
        RoadmapMLInvocation.objects.all().delete()

    def _make(self, **kwargs):
        defaults = dict(
            user=self.user,
            plan=self.plan,
            category="skincare",
            decision="model_used",
            fallback_reason="",
            ml_mode="v4_ranking",
            rollout_selected=False,
            active_top_product_type="",
            shadow_top_product_type="",
            planned_target_product_type="",
        )
        defaults.update(kwargs)
        created_at = defaults.pop("created_at", None)
        row = RoadmapMLInvocation.objects.create(**defaults)
        if created_at is not None:
            RoadmapMLInvocation.objects.filter(pk=row.pk).update(created_at=created_at)
            row.refresh_from_db()
        return row

    def test_empty_db_returns_empty_report(self):
        from roadmap_app.ml_diff_report import build_control_vs_ml_diff_report

        report = build_control_vs_ml_diff_report()
        self.assertEqual(report["total_invocations"], 0)
        self.assertEqual(report["per_category"], {})

    def test_served_vs_active_agreement_rate(self):
        from roadmap_app.ml_diff_report import build_control_vs_ml_diff_report

        for _ in range(7):
            self._make(
                planned_target_product_type="serum",
                active_top_product_type="serum",
            )
        for _ in range(3):
            self._make(
                planned_target_product_type="moisturizer",
                active_top_product_type="serum",
            )
        report = build_control_vs_ml_diff_report()
        payload = report["per_category"]["skincare"]
        agr = payload["agreement"]["served_vs_active"]
        self.assertEqual(agr["compared"], 10)
        self.assertEqual(agr["matches"], 7)
        self.assertAlmostEqual(agr["agreement_pct"], 70.0, places=2)

    def test_skips_pairs_with_missing_side(self):
        from roadmap_app.ml_diff_report import build_control_vs_ml_diff_report

        self._make(planned_target_product_type="serum", active_top_product_type="")
        self._make(planned_target_product_type="", active_top_product_type="serum")
        self._make(planned_target_product_type="serum", active_top_product_type="serum")
        report = build_control_vs_ml_diff_report()
        agr = report["per_category"]["skincare"]["agreement"]["served_vs_active"]
        self.assertEqual(agr["compared"], 1)
        self.assertEqual(agr["matches"], 1)

    def test_top_divergences_ranked_and_limited(self):
        from roadmap_app.ml_diff_report import build_control_vs_ml_diff_report

        for _ in range(5):
            self._make(
                planned_target_product_type="moisturizer",
                active_top_product_type="serum",
            )
        for _ in range(3):
            self._make(
                planned_target_product_type="cleanser",
                active_top_product_type="toner",
            )
        for _ in range(1):
            self._make(
                planned_target_product_type="spf",
                active_top_product_type="essence",
            )
        report = build_control_vs_ml_diff_report(top_divergences=2)
        top = report["per_category"]["skincare"]["top_divergences"]["served_vs_active"]
        self.assertEqual(len(top), 2)
        self.assertEqual(top[0], {"served": "moisturizer", "active": "serum", "count": 5})
        self.assertEqual(top[1], {"served": "cleanser", "active": "toner", "count": 3})

    def test_decision_and_fallback_reason_distribution(self):
        from roadmap_app.ml_diff_report import build_control_vs_ml_diff_report

        for _ in range(6):
            self._make(decision="model_used")
        for _ in range(3):
            self._make(decision="fallback", fallback_reason="predict_error")
        for _ in range(2):
            self._make(decision="fallback", fallback_reason="empty_candidates")
        for _ in range(4):
            self._make(decision="disabled")
        report = build_control_vs_ml_diff_report()
        payload = report["per_category"]["skincare"]
        self.assertEqual(payload["decision_counts"], {
            "model_used": 6, "fallback": 5, "disabled": 4,
        })
        self.assertEqual(payload["fallback_reason_counts"], {
            "predict_error": 3, "empty_candidates": 2,
        })

    def test_rollout_selected_counter(self):
        from roadmap_app.ml_diff_report import build_control_vs_ml_diff_report

        for _ in range(3):
            self._make(rollout_selected=True)
        for _ in range(7):
            self._make(rollout_selected=False)
        report = build_control_vs_ml_diff_report()
        self.assertEqual(report["per_category"]["skincare"]["rollout_selected_count"], 3)

    def test_window_excludes_old_rows(self):
        from roadmap_app.ml_diff_report import build_control_vs_ml_diff_report

        now = timezone.now()
        for _ in range(10):
            self._make(
                planned_target_product_type="old",
                active_top_product_type="old",
                created_at=now - timedelta(days=2),
            )
        for _ in range(3):
            self._make(
                planned_target_product_type="new",
                active_top_product_type="new",
            )
        report = build_control_vs_ml_diff_report(window_minutes=60, now=now)
        payload = report["per_category"]["skincare"]
        self.assertEqual(payload["total"], 3)
        self.assertEqual(payload["agreement"]["served_vs_active"]["compared"], 3)

    def test_category_filter(self):
        from roadmap_app.ml_diff_report import build_control_vs_ml_diff_report

        self._make(category="skincare", planned_target_product_type="a", active_top_product_type="a")
        self._make(category="haircare", planned_target_product_type="b", active_top_product_type="b")
        self._make(category="makeup", planned_target_product_type="c", active_top_product_type="c")
        report = build_control_vs_ml_diff_report(categories=["skincare", "makeup"])
        self.assertEqual(set(report["per_category"].keys()), {"skincare", "makeup"})
        self.assertEqual(report["category_filter"], ["skincare", "makeup"])

    def test_served_vs_shadow_and_active_vs_shadow_agreements(self):
        from roadmap_app.ml_diff_report import build_control_vs_ml_diff_report

        for _ in range(4):
            self._make(
                planned_target_product_type="x",
                active_top_product_type="x",
                shadow_top_product_type="x",
            )
        for _ in range(6):
            self._make(
                planned_target_product_type="x",
                active_top_product_type="x",
                shadow_top_product_type="y",
            )
        report = build_control_vs_ml_diff_report()
        agrs = report["per_category"]["skincare"]["agreement"]
        self.assertEqual(agrs["served_vs_active"]["agreement_pct"], 100.0)
        self.assertEqual(agrs["served_vs_shadow"]["matches"], 4)
        self.assertEqual(agrs["served_vs_shadow"]["compared"], 10)
        self.assertEqual(agrs["active_vs_shadow"]["matches"], 4)
        self.assertEqual(agrs["active_vs_shadow"]["compared"], 10)


class RoadmapMLDiffReportCommandTests(TestCase):
    @classmethod
    def setUpTestData(cls):
        cls.user = get_user_model().objects.create_user(
            username="diff_cmd_user", email="diff_cmd@example.com", password="x",
        )
        cls.plan = RoadmapPlan.objects.create(
            user=cls.user, category=RoadmapPlan.Category.SKINCARE, is_active=True,
        )

    def setUp(self):
        RoadmapMLInvocation.objects.all().delete()

    def _call(self, *args):
        stdout = StringIO()
        call_command("roadmap_ml_diff_report", *args, stdout=stdout)
        return stdout.getvalue()

    def _seed(self):
        for _ in range(7):
            RoadmapMLInvocation.objects.create(
                user=self.user, plan=self.plan, category="skincare",
                decision="model_used", ml_mode="v4_ranking",
                planned_target_product_type="serum",
                active_top_product_type="serum",
                shadow_top_product_type="serum",
            )
        for _ in range(3):
            RoadmapMLInvocation.objects.create(
                user=self.user, plan=self.plan, category="skincare",
                decision="model_used", ml_mode="v4_ranking",
                planned_target_product_type="moisturizer",
                active_top_product_type="serum",
                shadow_top_product_type="toner",
            )

    def test_empty_db_prints_placeholder(self):
        out = self._call()
        self.assertIn("(no invocations in window)", out)

    def test_text_output_contains_agreement_rates(self):
        self._seed()
        out = self._call()
        self.assertIn("[skincare]", out)
        self.assertIn("served vs active:", out)
        self.assertIn("70.00%", out)

    def test_json_output_is_parseable(self):
        self._seed()
        out = self._call("--json")
        parsed = json.loads(out)
        self.assertEqual(parsed["total_invocations"], 10)
        self.assertIn("skincare", parsed["per_category"])

    def test_category_filter_narrows_output(self):
        RoadmapMLInvocation.objects.create(
            user=self.user, plan=self.plan, category="haircare",
            decision="model_used", planned_target_product_type="shampoo",
            active_top_product_type="shampoo",
        )
        self._seed()
        out = self._call("--category", "skincare", "--json")
        parsed = json.loads(out)
        self.assertEqual(list(parsed["per_category"].keys()), ["skincare"])
        self.assertEqual(parsed["category_filter"], ["skincare"])
