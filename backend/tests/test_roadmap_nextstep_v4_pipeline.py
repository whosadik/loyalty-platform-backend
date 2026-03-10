from __future__ import annotations

import json
import sys
import types
from datetime import timedelta
from decimal import Decimal
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from django.contrib.auth import get_user_model
from django.core.management import call_command
from django.core.management.base import CommandError
from django.test import TestCase
from django.test.utils import override_settings
from django.utils import timezone

from catalog.models import Product
from roadmap_app.ml_next_step import (
    predict_next_product_types,
    v4_category_rollout_status,
    v4_category_staged_rollout_status_from_reports,
    v4_category_uplift_guard_status_from_report,
    v4_min_lift_guard_status,
)
from roadmap_app.models import RoadmapEvent, RoadmapPlan, RoadmapStep
from roadmap_app.services import refresh_roadmap
from transactions.models import Transaction, TransactionItem

try:
    import pandas as pd
except Exception:  # pragma: no cover
    pd = None


class FakeLGBMRanker:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.columns: list[str] = []

    def fit(self, X, y, group=None, eval_set=None, eval_group=None, eval_at=None, categorical_feature=None):
        self.columns = list(X.columns)
        self.group = list(group or [])
        self.eval_group = list((eval_group or [])[0] or []) if eval_group else []
        self.categorical_feature = list(categorical_feature or [])
        return self

    def predict(self, X):
        if pd is None:
            return [0.0 for _ in range(len(X))]
        frame = X.copy()
        score = pd.Series([0.0] * len(frame), index=frame.index, dtype=float)
        if "candidate_matches_last1" in frame.columns:
            score = score + (pd.to_numeric(frame["candidate_matches_last1"], errors="coerce").fillna(0.0) * 5.0)
        if "candidate_popularity_in_train" in frame.columns:
            score = score + pd.to_numeric(frame["candidate_popularity_in_train"], errors="coerce").fillna(0.0)
        return score.to_numpy()


def _read_dataset(out_dir: Path) -> "pd.DataFrame":
    parquet_path = out_dir / "dataset.parquet"
    if parquet_path.exists():
        return pd.read_parquet(parquet_path)
    return pd.read_csv(out_dir / "dataset.csv")


class RoadmapNextStepV4DatasetTests(TestCase):
    def setUp(self):
        if pd is None:
            self.skipTest("pandas is required")

        User = get_user_model()
        self.user = User.objects.create_user(username="v4_ds_u1", password="pass12345")

        self.p_serum = Product.objects.create(
            name="V4 Serum",
            brand="B",
            price=Decimal("12.00"),
            category="skincare",
            product_type="serum",
            in_stock=True,
        )
        self.p_cleanser = Product.objects.create(
            name="V4 Cleanser",
            brand="B",
            price=Decimal("11.00"),
            category="skincare",
            product_type="cleanser",
            in_stock=True,
        )

        self.plan = RoadmapPlan.objects.create(user=self.user, category="skincare", is_active=True, meta={})
        self.step = RoadmapStep.objects.create(
            plan=self.plan,
            step_index=1,
            product_type="serum",
            status=RoadmapStep.Status.MISSING,
        )

    def _create_exposed(self, at_dt):
        event = RoadmapEvent.objects.create(
            user=self.user,
            plan=self.plan,
            step=self.step,
            event_type=RoadmapEvent.Type.STEP_EXPOSED,
            context={"category": "skincare", "sources": ["roadmap_api"]},
        )
        RoadmapEvent.objects.filter(id=event.id).update(created_at=at_dt)

    def _create_tx(self, product: Product, at_dt, idem_key: str):
        tx = Transaction.objects.create(
            user=self.user,
            total_amount=Decimal("12.00"),
            channel="web",
            idempotency_key=idem_key,
        )
        Transaction.objects.filter(id=tx.id).update(created_at=at_dt)
        TransactionItem.objects.create(
            transaction=tx,
            product=product,
            quantity=1,
            unit_price=Decimal("12.00"),
        )

    def test_v4_dataset_features_do_not_use_transactions_after_t0(self):
        t0 = timezone.now() - timedelta(days=30)
        self._create_exposed(t0)
        self._create_tx(self.p_cleanser, t0 - timedelta(days=1), "v4-leak-1")
        self._create_tx(self.p_serum, t0 + timedelta(days=1), "v4-leak-2")

        with TemporaryDirectory() as tmp_dir:
            out_dir = Path(tmp_dir)
            call_command(
                "build_roadmap_ml_dataset_v4",
                days=180,
                out_dir=str(out_dir),
                label_window_days=7,
                popularity_top_n=10,
                owned_top_k=10,
            )
            frame = _read_dataset(out_dir)
            self.assertFalse(frame.empty)

            sample = frame.iloc[0]
            self.assertEqual(str(sample["last1_product_type"]), "cleanser")
            self.assertEqual(int(sample["tx_count_90d_category"]), 1)
            self.assertEqual(str(sample["label"]), "serum")

    def test_v4_dataset_groups_and_none_label_present(self):
        t0_positive = timezone.now() - timedelta(days=40)
        t0_none = timezone.now() - timedelta(days=20)

        self._create_exposed(t0_positive)
        self._create_tx(self.p_serum, t0_positive + timedelta(days=1), "v4-groups-1")

        self._create_exposed(t0_none)
        # no purchase inside label window for second episode -> "__none__"

        with TemporaryDirectory() as tmp_dir:
            out_dir = Path(tmp_dir)
            call_command(
                "build_roadmap_ml_dataset_v4",
                days=180,
                out_dir=str(out_dir),
                label_window_days=7,
                popularity_top_n=10,
                owned_top_k=10,
            )
            frame = _read_dataset(out_dir)
            self.assertFalse(frame.empty)
            self.assertIn("__none__", set(frame["label"].astype(str).tolist()))

            self.assertEqual(int(frame["group_id"].nunique()), int(frame["episode_id"].nunique()))
            positives_per_group = frame.groupby("group_id")["y"].sum()
            self.assertLessEqual(int(positives_per_group.max()), 1)
            self.assertTrue(bool((frame.groupby("group_id").size() > 1).all()))

            metadata = json.loads((out_dir / "metadata.json").read_text(encoding="utf-8"))
            self.assertIn("baselines", metadata)
            self.assertIn("class_distribution", metadata)


class RoadmapNextStepV4AdapterTests(TestCase):
    def test_adapter_returns_sorted_unique_candidates(self):
        User = get_user_model()
        user = User.objects.create_user(username="v4_adapter_u1", password="pass12345")

        class DummyModel:
            def predict_next_product_types(self, **kwargs):
                return [
                    {"candidate_type": "serum", "score": 0.20},
                    {"candidate_type": "cleanser", "score": 0.90},
                    {"candidate_type": "serum", "score": 0.80},
                ]

        with patch("roadmap_app.ml_next_step._load_model", return_value=DummyModel()):
            rows = predict_next_product_types(
                user=user,
                context_product_ids=[],
                category="skincare",
            )

        self.assertEqual([row["candidate_type"] for row in rows], ["cleanser", "serum"])
        self.assertEqual(len(rows), len({row["candidate_type"] for row in rows}))
        self.assertGreaterEqual(float(rows[0]["score"]), float(rows[1]["score"]))


class RoadmapNextStepV4EvalSourceTests(TestCase):
    def _write_eval_report(self, path: Path, *, model_value: float, baseline_value: float) -> None:
        path.write_text(
            json.dumps(
                {
                    "metrics_test": {"ndcg_at_5": model_value},
                    "dataset_baselines": {
                        "splits": {
                            "test": {
                                "popularity": {
                                    "ndcg_at_5": baseline_value,
                                }
                            }
                        }
                    },
                }
            ),
            encoding="utf-8",
        )

    def test_min_lift_guard_prefers_model_dir_eval_sidecar(self):
        with TemporaryDirectory() as model_tmp, TemporaryDirectory() as report_tmp:
            model_dir = Path(model_tmp)
            fallback_report = Path(report_tmp) / "fallback_eval.json"
            sidecar_report = model_dir / "eval_report.json"
            self._write_eval_report(sidecar_report, model_value=0.48, baseline_value=0.12)
            self._write_eval_report(fallback_report, model_value=0.12, baseline_value=0.11)

            with override_settings(
                ROADMAP_NEXTSTEP_V4_ENABLED=True,
                ROADMAP_NEXTSTEP_V4_MODEL_PATH=str(model_dir / "model.pkl"),
                ROADMAP_NEXTSTEP_V4_EVAL_PATH=str(fallback_report),
                ROADMAP_NEXTSTEP_V4_MIN_LIFT_DELTA=0.05,
            ):
                status = v4_min_lift_guard_status()

        self.assertTrue(bool(status.get("passed")))
        self.assertEqual(str(status.get("eval_path")), str(sidecar_report))

    def test_min_lift_guard_can_use_embedded_metadata_snapshot(self):
        with TemporaryDirectory() as model_tmp, TemporaryDirectory() as report_tmp:
            model_dir = Path(model_tmp)
            metadata_path = model_dir / "metadata.json"
            metadata_path.write_text(
                json.dumps(
                    {
                        "metrics_test": {"ndcg_at_5": 0.41},
                        "dataset_baselines": {
                            "splits": {
                                "test": {
                                    "popularity": {
                                        "ndcg_at_5": 0.12,
                                    }
                                }
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )
            missing_report = Path(report_tmp) / "missing_eval.json"

            with override_settings(
                ROADMAP_NEXTSTEP_V4_ENABLED=True,
                ROADMAP_NEXTSTEP_V4_MODEL_PATH=str(model_dir / "model.pkl"),
                ROADMAP_NEXTSTEP_V4_EVAL_PATH=str(missing_report),
                ROADMAP_NEXTSTEP_V4_MIN_LIFT_DELTA=0.05,
            ):
                status = v4_min_lift_guard_status()

        self.assertTrue(bool(status.get("passed")))
        self.assertIn("#embedded_eval", str(status.get("eval_path")))


class RoadmapNextStepV4CategoryGuardTests(TestCase):
    def _make_report(
        self,
        *,
        category: str,
        model_plans: int,
        control_plans: int,
        step_completion_abs_lift: float | None,
        offer_redeem_abs_lift: float | None,
        step_ctr_abs_lift: float | None,
        offer_ctr_abs_lift: float | None,
    ) -> dict[str, object]:
        return {
            "params": {
                "cohort_mode": "fresh",
                "control": "non_model",
            },
            "breakdowns": {
                "by_category": {
                    category: {
                        "model_used": {"plans_total": model_plans},
                        "control": {"plans_total": control_plans},
                    }
                }
            },
            "uplift": {
                "by_category": {
                    category: {
                        "step_funnel": {
                            "step_completion_rate": {"abs_lift": step_completion_abs_lift},
                            "step_ctr": {"abs_lift": step_ctr_abs_lift},
                        },
                        "offer_funnel": {
                            "offer_redeem_rate": {"abs_lift": offer_redeem_abs_lift},
                            "offer_ctr": {"abs_lift": offer_ctr_abs_lift},
                        },
                    }
                }
            },
        }

    def test_guard_passes_when_primary_metrics_are_good_and_ctr_drop_is_mild(self):
        report = self._make_report(
            category="skincare",
            model_plans=220,
            control_plans=210,
            step_completion_abs_lift=0.02,
            offer_redeem_abs_lift=0.007,
            step_ctr_abs_lift=-0.015,
            offer_ctr_abs_lift=-0.02,
        )
        with override_settings(
            ROADMAP_NEXTSTEP_V4_CATEGORY_MIN_PLANS=100,
            ROADMAP_NEXTSTEP_V4_MIN_STEP_COMPLETION_LIFT=0.01,
            ROADMAP_NEXTSTEP_V4_MIN_OFFER_REDEEM_LIFT=0.005,
            ROADMAP_NEXTSTEP_V4_MAX_NEGATIVE_STEP_CTR_LIFT_SOFT=-0.02,
            ROADMAP_NEXTSTEP_V4_MAX_NEGATIVE_OFFER_CTR_LIFT_SOFT=-0.03,
            ROADMAP_NEXTSTEP_V4_ALLOW_PRIMARY_WIN_DESPITE_SOFT_CTR_DROP=True,
        ):
            status = v4_category_uplift_guard_status_from_report(
                "skincare",
                report,
                report_path="inline://test",
            )
        self.assertTrue(bool(status.get("passed")))
        self.assertTrue(bool(status.get("primary_passed")))
        self.assertTrue(bool(status.get("secondary_passed")))
        self.assertEqual(str(status.get("reason")), "passed")

    def test_guard_fails_on_severe_negative_offer_ctr_even_when_primary_metrics_pass(self):
        report = self._make_report(
            category="skincare",
            model_plans=220,
            control_plans=210,
            step_completion_abs_lift=0.03,
            offer_redeem_abs_lift=0.006,
            step_ctr_abs_lift=-0.01,
            offer_ctr_abs_lift=-0.05,
        )
        with override_settings(
            ROADMAP_NEXTSTEP_V4_CATEGORY_MIN_PLANS=100,
            ROADMAP_NEXTSTEP_V4_MIN_STEP_COMPLETION_LIFT=0.01,
            ROADMAP_NEXTSTEP_V4_MIN_OFFER_REDEEM_LIFT=0.005,
            ROADMAP_NEXTSTEP_V4_MAX_NEGATIVE_STEP_CTR_LIFT_SOFT=-0.02,
            ROADMAP_NEXTSTEP_V4_MAX_NEGATIVE_OFFER_CTR_LIFT_SOFT=-0.03,
            ROADMAP_NEXTSTEP_V4_ALLOW_PRIMARY_WIN_DESPITE_SOFT_CTR_DROP=True,
        ):
            status = v4_category_uplift_guard_status_from_report(
                "skincare",
                report,
                report_path="inline://test",
            )
        self.assertFalse(bool(status.get("passed")))
        self.assertTrue(bool(status.get("primary_passed")))
        self.assertFalse(bool(status.get("secondary_passed")))
        self.assertEqual(str(status.get("reason")), "severe_negative_offer_ctr_lift")

    def test_guard_fails_with_insufficient_sample(self):
        report = self._make_report(
            category="skincare",
            model_plans=20,
            control_plans=15,
            step_completion_abs_lift=0.05,
            offer_redeem_abs_lift=0.02,
            step_ctr_abs_lift=0.01,
            offer_ctr_abs_lift=0.00,
        )
        with override_settings(ROADMAP_NEXTSTEP_V4_CATEGORY_MIN_PLANS=100):
            status = v4_category_uplift_guard_status_from_report(
                "skincare",
                report,
                report_path="inline://test",
            )
        self.assertFalse(bool(status.get("passed")))
        self.assertEqual(str(status.get("reason")), "insufficient_sample")

    def test_fragrance_rollout_is_disabled_even_if_uplift_is_positive(self):
        report = self._make_report(
            category="fragrance",
            model_plans=500,
            control_plans=500,
            step_completion_abs_lift=0.03,
            offer_redeem_abs_lift=0.01,
            step_ctr_abs_lift=0.0,
            offer_ctr_abs_lift=0.0,
        )
        with override_settings(
            ROADMAP_NEXTSTEP_V4_ENABLED=True,
            ROADMAP_NEXTSTEP_V4_ENABLED_CATEGORIES=["skincare", "haircare", "makeup"],
            ROADMAP_NEXTSTEP_V4_DISABLED_CATEGORIES=["fragrance"],
        ):
            rollout = v4_category_rollout_status("fragrance")
            guard = v4_category_uplift_guard_status_from_report(
                "fragrance",
                report,
                report_path="inline://test",
            )
        self.assertFalse(bool(rollout.get("passed")))
        self.assertEqual(str(rollout.get("reason")), "category_disabled")
        self.assertTrue(bool(guard.get("passed")))

    def test_staged_rollout_enable_when_7d_and_30d_pass(self):
        report_7d = self._make_report(
            category="haircare",
            model_plans=220,
            control_plans=230,
            step_completion_abs_lift=0.04,
            offer_redeem_abs_lift=0.01,
            step_ctr_abs_lift=0.01,
            offer_ctr_abs_lift=0.00,
        )
        report_30d = self._make_report(
            category="haircare",
            model_plans=600,
            control_plans=620,
            step_completion_abs_lift=0.03,
            offer_redeem_abs_lift=0.008,
            step_ctr_abs_lift=0.00,
            offer_ctr_abs_lift=0.00,
        )
        with override_settings(
            ROADMAP_NEXTSTEP_V4_ENABLED=True,
            ROADMAP_NEXTSTEP_V4_ENABLED_CATEGORIES=["skincare", "haircare", "makeup"],
            ROADMAP_NEXTSTEP_V4_DISABLED_CATEGORIES=["fragrance"],
        ):
            status = v4_category_staged_rollout_status_from_reports(
                "haircare",
                report_7d=report_7d,
                report_30d=report_30d,
                report_path_7d="inline://7d",
                report_path_30d="inline://30d",
            )
        self.assertEqual(str(status.get("final_status")), "ENABLE")
        self.assertEqual(str(status.get("recommendation_7d")), "ENABLE")
        self.assertEqual(str(status.get("recommendation_30d")), "ENABLE")
        self.assertEqual(str(status.get("reason")), "passed")
        self.assertEqual(list(status.get("stability_gate_failures") or []), [])

    def test_staged_rollout_hold_when_30d_pass_but_7d_fails(self):
        report_7d = self._make_report(
            category="makeup",
            model_plans=220,
            control_plans=230,
            step_completion_abs_lift=0.03,
            offer_redeem_abs_lift=0.008,
            step_ctr_abs_lift=-0.05,
            offer_ctr_abs_lift=0.0,
        )
        report_30d = self._make_report(
            category="makeup",
            model_plans=600,
            control_plans=620,
            step_completion_abs_lift=0.02,
            offer_redeem_abs_lift=0.006,
            step_ctr_abs_lift=0.00,
            offer_ctr_abs_lift=0.00,
        )
        with override_settings(
            ROADMAP_NEXTSTEP_V4_ENABLED=True,
            ROADMAP_NEXTSTEP_V4_ENABLED_CATEGORIES=["skincare", "haircare", "makeup"],
            ROADMAP_NEXTSTEP_V4_DISABLED_CATEGORIES=["fragrance"],
        ):
            status = v4_category_staged_rollout_status_from_reports(
                "makeup",
                report_7d=report_7d,
                report_30d=report_30d,
                report_path_7d="inline://7d",
                report_path_30d="inline://30d",
            )
        self.assertEqual(str(status.get("final_status")), "HOLD")
        self.assertEqual(str(status.get("recommendation_7d")), "HOLD")
        self.assertEqual(str(status.get("recommendation_30d")), "ENABLE")
        self.assertEqual(str(status.get("reason")), "7d_unstable")
        self.assertIn("7d:severe_negative_step_ctr_lift", list(status.get("stability_gate_failures") or []))

    def test_staged_rollout_disable_when_category_explicitly_blocked(self):
        report_7d = self._make_report(
            category="fragrance",
            model_plans=220,
            control_plans=230,
            step_completion_abs_lift=0.03,
            offer_redeem_abs_lift=0.008,
            step_ctr_abs_lift=0.02,
            offer_ctr_abs_lift=0.01,
        )
        report_30d = self._make_report(
            category="fragrance",
            model_plans=600,
            control_plans=620,
            step_completion_abs_lift=0.03,
            offer_redeem_abs_lift=0.008,
            step_ctr_abs_lift=0.02,
            offer_ctr_abs_lift=0.01,
        )
        with override_settings(
            ROADMAP_NEXTSTEP_V4_ENABLED=True,
            ROADMAP_NEXTSTEP_V4_ENABLED_CATEGORIES=["skincare", "haircare", "makeup"],
            ROADMAP_NEXTSTEP_V4_DISABLED_CATEGORIES=["fragrance"],
        ):
            status = v4_category_staged_rollout_status_from_reports(
                "fragrance",
                report_7d=report_7d,
                report_30d=report_30d,
                report_path_7d="inline://7d",
                report_path_30d="inline://30d",
            )
        self.assertEqual(str(status.get("final_status")), "DISABLE")
        self.assertEqual(str(status.get("reason")), "category_disabled")


class RoadmapNextStepV4PartialRolloutRuntimeTests(TestCase):
    def _create_makeup_products(self, suffix: str) -> None:
        Product.objects.create(
            name=f"Makeup Foundation {suffix}",
            brand="B",
            category="makeup",
            product_type="foundation",
            in_stock=True,
        )
        Product.objects.create(
            name=f"Makeup Mascara {suffix}",
            brand="B",
            category="makeup",
            product_type="mascara",
            in_stock=True,
        )
        Product.objects.create(
            name=f"Makeup Blush {suffix}",
            brand="B",
            category="makeup",
            product_type="blush",
            in_stock=True,
        )

    def _create_skincare_products(self, suffix: str) -> None:
        Product.objects.create(
            name=f"Skincare Cleanser {suffix}",
            brand="B",
            category="skincare",
            product_type="cleanser",
            in_stock=True,
        )
        Product.objects.create(
            name=f"Skincare Serum {suffix}",
            brand="B",
            category="skincare",
            product_type="serum",
            in_stock=True,
        )
        Product.objects.create(
            name=f"Skincare Moisturizer {suffix}",
            brand="B",
            category="skincare",
            product_type="moisturizer",
            in_stock=True,
        )
        Product.objects.create(
            name=f"Skincare SPF {suffix}",
            brand="B",
            category="skincare",
            product_type="spf",
            in_stock=True,
        )

    def _hold_status(self, category: str) -> dict[str, object]:
        return {
            "passed": False,
            "final_status": "HOLD",
            "current_decision": "HOLD",
            "reason": "7d_unstable",
            "hold_reason": "severe_negative_offer_ctr_lift",
            "category": category,
            "recommendation_7d": "HOLD",
            "recommendation_30d": "ENABLE",
            "stability_gate_failures": ["7d:severe_negative_offer_ctr_lift"],
            "guard_7d": {"passed": False, "reason": "severe_negative_offer_ctr_lift"},
            "guard_30d": {"passed": True, "reason": "passed"},
        }

    def test_partial_rollout_selection_is_deterministic_for_same_user_hash(self):
        User = get_user_model()
        user = User.objects.create_user(username="v4_partial_det_u1", password="pass12345")
        self._create_makeup_products("det")

        with override_settings(
            ROADMAP_NEXTSTEP_V4_ENABLED=True,
            ROADMAP_NEXTSTEP_V3_ENABLED=False,
            ROADMAP_NEXTSTEP_V4_ENABLED_CATEGORIES=["skincare", "haircare", "makeup"],
            ROADMAP_NEXTSTEP_V4_DISABLED_CATEGORIES=["fragrance"],
            ROADMAP_NEXTSTEP_V4_PARTIAL_ENABLED_CATEGORIES=["makeup"],
            ROADMAP_NEXTSTEP_V4_PARTIAL_MAKEUP_PRODUCT_TYPES=["foundation"],
            ROADMAP_NEXTSTEP_V4_PARTIAL_MAKEUP_STEP_INDEXES=["1"],
            ROADMAP_NEXTSTEP_V4_PARTIAL_PERCENT=100,
            ROADMAP_NEXTSTEP_V4_PARTIAL_SALT="partial_det_test",
            ROADMAP_NEXTSTEP_V4_CONFIDENCE_THRESHOLD=0.1,
        ), patch(
            "roadmap_app.services.v4_min_lift_guard_status",
            return_value={"passed": True, "reason": "ok"},
        ), patch(
            "roadmap_app.services.v4_category_staged_rollout_status",
            side_effect=lambda cat: self._hold_status(cat),
        ), patch(
            "roadmap_app.services.predict_next_product_types",
            return_value=[{"candidate_type": "foundation", "score": 0.92}],
        ):
            plan_first = refresh_roadmap(user, category="makeup", post_ctx=None)
            plan_second = refresh_roadmap(user, category="makeup", post_ctx=None)

        ml_first = (plan_first.meta or {}).get("ml") if isinstance(plan_first.meta, dict) else {}
        ml_second = (plan_second.meta or {}).get("ml") if isinstance(plan_second.meta, dict) else {}
        self.assertEqual(str(ml_first.get("rollout_mode")), "partial")
        self.assertTrue(bool(ml_first.get("rollout_selected")))
        self.assertEqual(ml_first.get("rollout_bucket"), ml_second.get("rollout_bucket"))
        self.assertEqual(ml_first.get("rollout_selected"), ml_second.get("rollout_selected"))

    def test_non_selected_makeup_users_fallback_with_partial_reason(self):
        User = get_user_model()
        user = User.objects.create_user(username="v4_partial_no_u1", password="pass12345")
        self._create_makeup_products("no")

        with override_settings(
            ROADMAP_NEXTSTEP_V4_ENABLED=True,
            ROADMAP_NEXTSTEP_V3_ENABLED=False,
            ROADMAP_NEXTSTEP_V4_ENABLED_CATEGORIES=["skincare", "haircare", "makeup"],
            ROADMAP_NEXTSTEP_V4_DISABLED_CATEGORIES=["fragrance"],
            ROADMAP_NEXTSTEP_V4_PARTIAL_ENABLED_CATEGORIES=["makeup"],
            ROADMAP_NEXTSTEP_V4_PARTIAL_MAKEUP_PRODUCT_TYPES=["foundation"],
            ROADMAP_NEXTSTEP_V4_PARTIAL_MAKEUP_STEP_INDEXES=["1"],
            ROADMAP_NEXTSTEP_V4_PARTIAL_PERCENT=0,
            ROADMAP_NEXTSTEP_V4_PARTIAL_SALT="partial_no_test",
        ), patch(
            "roadmap_app.services.v4_min_lift_guard_status",
            return_value={"passed": True, "reason": "ok"},
        ), patch(
            "roadmap_app.services.v4_category_staged_rollout_status",
            side_effect=lambda cat: self._hold_status(cat),
        ), patch(
            "roadmap_app.services.predict_next_product_types",
        ) as predict_mock:
            plan = refresh_roadmap(user, category="makeup", post_ctx=None)

        ml_meta = (plan.meta or {}).get("ml") if isinstance(plan.meta, dict) else {}
        self.assertEqual(str(ml_meta.get("decision")), "fallback")
        self.assertEqual(str(ml_meta.get("fallback_reason")), "partial_rollout_not_selected")
        self.assertEqual(str(ml_meta.get("rollout_mode")), "partial")
        self.assertFalse(bool(ml_meta.get("rollout_selected")))
        predict_mock.assert_not_called()

    def test_selected_makeup_users_can_reach_model_used(self):
        User = get_user_model()
        user = User.objects.create_user(username="v4_partial_yes_u1", password="pass12345")
        self._create_makeup_products("yes")

        with override_settings(
            ROADMAP_NEXTSTEP_V4_ENABLED=True,
            ROADMAP_NEXTSTEP_V3_ENABLED=False,
            ROADMAP_NEXTSTEP_V4_ENABLED_CATEGORIES=["skincare", "haircare", "makeup"],
            ROADMAP_NEXTSTEP_V4_DISABLED_CATEGORIES=["fragrance"],
            ROADMAP_NEXTSTEP_V4_PARTIAL_ENABLED_CATEGORIES=["makeup"],
            ROADMAP_NEXTSTEP_V4_PARTIAL_MAKEUP_PRODUCT_TYPES=["foundation"],
            ROADMAP_NEXTSTEP_V4_PARTIAL_MAKEUP_STEP_INDEXES=["1"],
            ROADMAP_NEXTSTEP_V4_PARTIAL_PERCENT=100,
            ROADMAP_NEXTSTEP_V4_PARTIAL_SALT="partial_yes_test",
            ROADMAP_NEXTSTEP_V4_CONFIDENCE_THRESHOLD=0.1,
        ), patch(
            "roadmap_app.services.v4_min_lift_guard_status",
            return_value={"passed": True, "reason": "ok"},
        ), patch(
            "roadmap_app.services.v4_category_staged_rollout_status",
            side_effect=lambda cat: self._hold_status(cat),
        ), patch(
            "roadmap_app.services.predict_next_product_types",
            return_value=[{"candidate_type": "foundation", "score": 0.97}],
        ) as predict_mock:
            plan = refresh_roadmap(user, category="makeup", post_ctx=None)

        ml_meta = (plan.meta or {}).get("ml") if isinstance(plan.meta, dict) else {}
        self.assertEqual(str(ml_meta.get("decision")), "model_used")
        self.assertEqual(str(ml_meta.get("rollout_mode")), "partial")
        self.assertTrue(bool(ml_meta.get("rollout_selected")))
        self.assertIsNone(ml_meta.get("fallback_reason"))
        predict_mock.assert_called_once()

    def test_non_makeup_category_is_unaffected_by_partial_makeup_rollout(self):
        User = get_user_model()
        user = User.objects.create_user(username="v4_partial_sk_u1", password="pass12345")
        self._create_skincare_products("no-touch")

        with override_settings(
            ROADMAP_NEXTSTEP_V4_ENABLED=True,
            ROADMAP_NEXTSTEP_V3_ENABLED=False,
            ROADMAP_NEXTSTEP_V4_ENABLED_CATEGORIES=["skincare", "haircare", "makeup"],
            ROADMAP_NEXTSTEP_V4_DISABLED_CATEGORIES=["fragrance"],
            ROADMAP_NEXTSTEP_V4_PARTIAL_ENABLED_CATEGORIES=["makeup"],
            ROADMAP_NEXTSTEP_V4_PARTIAL_MAKEUP_PRODUCT_TYPES=["foundation"],
            ROADMAP_NEXTSTEP_V4_PARTIAL_MAKEUP_STEP_INDEXES=["1"],
            ROADMAP_NEXTSTEP_V4_PARTIAL_PERCENT=100,
            ROADMAP_NEXTSTEP_V4_PARTIAL_SALT="partial_skincare_test",
        ), patch(
            "roadmap_app.services.v4_min_lift_guard_status",
            return_value={"passed": True, "reason": "ok"},
        ), patch(
            "roadmap_app.services.v4_category_staged_rollout_status",
            side_effect=lambda cat: self._hold_status(cat),
        ), patch(
            "roadmap_app.services.predict_next_product_types",
        ) as predict_mock:
            plan = refresh_roadmap(user, category="skincare", post_ctx=None)

        ml_meta = (plan.meta or {}).get("ml") if isinstance(plan.meta, dict) else {}
        self.assertEqual(str(ml_meta.get("decision")), "fallback")
        self.assertEqual(str(ml_meta.get("fallback_reason")), "category_guard_failed")
        self.assertEqual(str(ml_meta.get("rollout_mode")), "none")
        self.assertFalse(bool(ml_meta.get("rollout_selected")))
        predict_mock.assert_not_called()


class RoadmapNextStepV4TrainingTests(TestCase):
    def _write_train_fixture(self, out_dir: Path) -> None:
        rows: list[dict[str, object]] = []
        split_payload = {
            "train_user_ids": [1, 2, 3, 4, 5, 6],
            "val_user_ids": [7, 8],
            "test_user_ids": [9, 10],
        }
        candidate_types = ["serum", "cleanser", "mask"]
        episode_id = 1
        user_ids = split_payload["train_user_ids"] + split_payload["val_user_ids"] + split_payload["test_user_ids"]
        for user_id in user_ids:
            for idx in range(4):
                label = "serum" if (user_id + idx) % 2 == 0 else "cleanser"
                for candidate in candidate_types:
                    rows.append(
                        {
                            "episode_id": episode_id,
                            "group_id": episode_id,
                            "user_id": user_id,
                            "category": "skincare",
                            "candidate_type": candidate,
                            "label": label,
                            "y": int(candidate == label),
                            "last1_product_type": "serum" if label == "serum" else "cleanser",
                            "last2_product_type": "__none__",
                            "last3_product_type": "__none__",
                            "last4_product_type": "__none__",
                            "last5_product_type": "__none__",
                            "last1_category": "skincare",
                            "last2_category": "__none__",
                            "last3_category": "__none__",
                            "last4_category": "__none__",
                            "last5_category": "__none__",
                            "month_of_year": 1 + (idx % 12),
                            "day_of_week": idx % 7,
                            "days_since_last_purchase_in_category": 5 + idx,
                            "tx_count_90d_category": 1 + idx,
                            "tx_amount_90d_category": float(20 + idx),
                            "owned_slot_warm_day": 0,
                            "owned_slot_warm_evening": 0,
                            "owned_slot_cold_day": 0,
                            "owned_slot_cold_evening": 0,
                            "candidate_is_fragrance_slot": 0,
                            "candidate_position_in_chain": candidate_types.index(candidate),
                            "candidate_popularity_in_train": 0.7 if candidate == "serum" else (0.6 if candidate == "cleanser" else 0.1),
                            "candidate_matches_last1": int(candidate == label),
                            "candidate_matches_last3_any": int(candidate == label),
                            "candidate_seen_count_last5": int(candidate == label),
                            "candidate_owned_count_in_category": int(candidate == label),
                            "candidate_seen_90d_count_in_category": int(candidate == label),
                            "candidate_days_since_last_seen_in_category": 1 if candidate == label else 30,
                            "t0_utc": "2026-01-01T00:00:00Z",
                            "split": "train" if user_id in split_payload["train_user_ids"] else ("val" if user_id in split_payload["val_user_ids"] else "test"),
                        }
                    )
                episode_id += 1

        frame = pd.DataFrame(rows)
        frame.to_csv(out_dir / "dataset.csv", index=False)
        metadata = {
            "feature_columns": [
                "category",
                "candidate_type",
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
                "month_of_year",
                "day_of_week",
                "days_since_last_purchase_in_category",
                "tx_count_90d_category",
                "tx_amount_90d_category",
                "owned_slot_warm_day",
                "owned_slot_warm_evening",
                "owned_slot_cold_day",
                "owned_slot_cold_evening",
                "candidate_is_fragrance_slot",
                "candidate_position_in_chain",
                "candidate_popularity_in_train",
                "candidate_matches_last1",
                "candidate_matches_last3_any",
                "candidate_seen_count_last5",
                "candidate_owned_count_in_category",
                "candidate_seen_90d_count_in_category",
                "candidate_days_since_last_seen_in_category",
            ],
            "categorical_features": [
                "category",
                "candidate_type",
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
            ],
            "numeric_features": [
                "month_of_year",
                "day_of_week",
                "days_since_last_purchase_in_category",
                "tx_count_90d_category",
                "tx_amount_90d_category",
                "owned_slot_warm_day",
                "owned_slot_warm_evening",
                "owned_slot_cold_day",
                "owned_slot_cold_evening",
                "candidate_is_fragrance_slot",
                "candidate_position_in_chain",
                "candidate_popularity_in_train",
                "candidate_matches_last1",
                "candidate_matches_last3_any",
                "candidate_seen_count_last5",
                "candidate_owned_count_in_category",
                "candidate_seen_90d_count_in_category",
                "candidate_days_since_last_seen_in_category",
            ],
            "candidate_types_by_category": {"skincare": candidate_types},
            "rules_chain_by_category": {"skincare": candidate_types},
            "candidate_popularity_in_train_by_category": {
                "skincare": {"serum": 0.7, "cleanser": 0.6, "mask": 0.1}
            },
            "owned_feature_columns": [],
            "owned_feature_map": {},
            "baselines": {
                "splits": {
                    "val": {
                        "popularity": {"recall_at_1": 0.50, "recall_at_3": 1.0, "recall_at_5": 1.0, "ndcg_at_5": 0.70},
                        "markov": {"recall_at_1": 0.40, "recall_at_3": 1.0, "recall_at_5": 1.0, "ndcg_at_5": 0.60},
                    },
                    "test": {
                        "popularity": {"recall_at_1": 0.50, "recall_at_3": 1.0, "recall_at_5": 1.0, "ndcg_at_5": 0.70},
                        "markov": {"recall_at_1": 0.40, "recall_at_3": 1.0, "recall_at_5": 1.0, "ndcg_at_5": 0.60},
                    },
                }
            },
        }
        (out_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        (out_dir / "splits.json").write_text(json.dumps(split_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def test_training_raises_when_ranker_library_missing(self):
        with TemporaryDirectory() as data_tmp, TemporaryDirectory() as model_tmp:
            with patch(
                "admin_tools.management.commands.train_roadmap_nextstep_model_v4._module_available",
                return_value=False,
            ):
                with self.assertRaisesMessage(CommandError, "LightGBM is not installed"):
                    call_command(
                        "train_roadmap_nextstep_model_v4",
                        data_dir=str(data_tmp),
                        model_dir=str(model_tmp),
                        estimator="lightgbm",
                    )

    def test_training_ranker_writes_baseline_comparison_and_ablation(self):
        if pd is None:
            self.skipTest("pandas is required")

        with TemporaryDirectory() as data_tmp, TemporaryDirectory() as model_tmp:
            data_dir = Path(data_tmp)
            model_dir = Path(model_tmp)
            self._write_train_fixture(data_dir)

            fake_lightgbm = types.ModuleType("lightgbm")
            fake_lightgbm.LGBMRanker = FakeLGBMRanker
            with patch(
                "admin_tools.management.commands.train_roadmap_nextstep_model_v4._module_available",
                side_effect=lambda name: name == "lightgbm",
            ), patch.dict(sys.modules, {"lightgbm": fake_lightgbm}):
                call_command(
                    "train_roadmap_nextstep_model_v4",
                    data_dir=str(data_dir),
                    model_dir=str(model_dir),
                    estimator="lightgbm",
                    trials=1,
                    negative_samples_per_episode=2,
                )

            metadata = json.loads((model_dir / "metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(str(metadata.get("estimator")), "lightgbm")
            self.assertIn(str(metadata.get("selected_feature_set")), {"baseline_only", "full"})
            model_report_path = model_dir / "eval_report.json"
            self.assertTrue(model_report_path.exists())
            model_report = json.loads(model_report_path.read_text(encoding="utf-8"))
            self.assertEqual(int(model_report.get("test_rows", 0)), int(metadata.get("test_rows", 0)))
            self.assertEqual(str(metadata.get("eval_report_path")), str(model_report_path))

            report_path = Path(__file__).resolve().parents[2] / "reports" / "roadmap_nextstep_v4_eval.json"
            report = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertIn("baseline_comparison", report)
            self.assertIn("feature_ablation", report)
            self.assertIn("full", report["feature_ablation"])
            self.assertIn("baseline_only", report["feature_ablation"])
            self.assertEqual(str(report.get("estimator")), "lightgbm")
