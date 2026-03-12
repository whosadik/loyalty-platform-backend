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
from transactions.models import OwnedProduct, Transaction, TransactionItem
from users_app.models import CustomerProfile

try:
    import pandas as pd
except Exception:  # pragma: no cover
    pd = None


class FakeLGBMRanker:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.columns: list[str] = []

    def fit(
        self,
        X,
        y,
        group=None,
        sample_weight=None,
        eval_set=None,
        eval_group=None,
        eval_sample_weight=None,
        eval_at=None,
        categorical_feature=None,
    ):
        self.columns = list(X.columns)
        self.group = list(group or [])
        self.sample_weight = list(sample_weight) if sample_weight is not None else []
        self.eval_group = list((eval_group or [])[0] or []) if eval_group else []
        self.eval_sample_weight = (
            list(eval_sample_weight[0]) if eval_sample_weight and eval_sample_weight[0] is not None else []
        )
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

    def _create_clicked(self, at_dt):
        event = RoadmapEvent.objects.create(
            user=self.user,
            step=self.step,
            event_type=RoadmapEvent.Type.STEP_CLICKED,
            context={},
        )
        RoadmapEvent.objects.filter(id=event.id).update(created_at=at_dt)

    def _create_completed(self, at_dt, *, matched_by: str, product_type: str | None = None, match_meta: dict | None = None):
        event = RoadmapEvent.objects.create(
            user=self.user,
            plan=self.plan,
            step=self.step,
            event_type=RoadmapEvent.Type.STEP_COMPLETED,
            context={
                "category": "skincare",
                "product_type": product_type or self.step.product_type,
                "matched_by": matched_by,
                "match_meta": match_meta or {},
            },
        )
        RoadmapEvent.objects.filter(id=event.id).update(created_at=at_dt)

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
            self.assertLess(float(sample["sample_weight"]), 1.0)

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

    def test_v4_dataset_includes_content_aware_features(self):
        profile = CustomerProfile.objects.get(user=self.user)
        profile.skin_type = "dry"
        profile.goals = ["hydration"]
        profile.hair_profile = {}
        profile.makeup_profile = {}
        profile.fragrance_profile = {}
        profile.save(
            update_fields=[
                "skin_type",
                "goals",
                "hair_profile",
                "makeup_profile",
                "fragrance_profile",
            ]
        )
        self.p_cleanser.concerns = ["cleanse"]
        self.p_cleanser.actives = ["ceramides"]
        self.p_cleanser.supported_skin_types = ["dry"]
        self.p_cleanser.ingredients_inci = "aqua, glycerin, ceramide np"
        self.p_cleanser.save(
            update_fields=["concerns", "actives", "supported_skin_types", "ingredients_inci"]
        )
        self.p_serum.concerns = ["hydration"]
        self.p_serum.actives = ["niacinamide"]
        self.p_serum.supported_skin_types = ["dry"]
        self.p_serum.ingredients_inci = "aqua, glycerin, niacinamide"
        self.p_serum.save(
            update_fields=["concerns", "actives", "supported_skin_types", "ingredients_inci"]
        )

        t0 = timezone.now() - timedelta(days=25)
        self._create_exposed(t0)
        self._create_tx(self.p_cleanser, t0 - timedelta(days=1), "v4-content-1")
        self._create_tx(self.p_serum, t0 + timedelta(days=1), "v4-content-2")

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
            serum_row = frame[frame["candidate_type"].astype(str) == "serum"].iloc[0]

            self.assertIn("profile_skin_type", frame.columns)
            self.assertIn("candidate_profile_goal_match_rate", frame.columns)
            self.assertEqual(str(serum_row["profile_skin_type"]), "dry")
            self.assertEqual(str(serum_row["anchor_product_type"]), "cleanser")
            self.assertGreater(float(serum_row["candidate_profile_goal_match_rate"]), 0.0)
            self.assertGreater(float(serum_row["candidate_anchor_shared_inci_rate"]), 0.0)

    def test_v4_dataset_includes_chain_transition_features(self):
        t0 = timezone.now() - timedelta(days=22)
        self._create_exposed(t0)
        self._create_tx(self.p_cleanser, t0 - timedelta(days=1), "v4-chain-1")
        self._create_tx(self.p_serum, t0 + timedelta(days=1), "v4-chain-2")

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
            serum_row = frame[frame["candidate_type"].astype(str) == "serum"].iloc[0]
            cleanser_row = frame[frame["candidate_type"].astype(str) == "cleanser"].iloc[0]

            self.assertIn("candidate_distance_from_anchor", frame.columns)
            self.assertIn("candidate_is_immediate_followup_to_anchor", frame.columns)
            self.assertEqual(int(serum_row["anchor_position_in_chain"]), 0)
            self.assertEqual(int(serum_row["last1_position_in_chain"]), 0)
            self.assertEqual(int(serum_row["candidate_distance_from_anchor"]), 1)
            self.assertEqual(int(serum_row["candidate_distance_from_last1"]), 1)
            self.assertEqual(int(serum_row["candidate_is_immediate_followup_to_anchor"]), 1)
            self.assertEqual(int(serum_row["candidate_is_after_anchor"]), 1)
            self.assertEqual(int(cleanser_row["candidate_distance_from_anchor"]), 0)
            self.assertEqual(int(cleanser_row["candidate_is_same_as_anchor"]), 1)

            metadata = json.loads((out_dir / "metadata.json").read_text(encoding="utf-8"))
            self.assertIn(
                "candidate_is_immediate_followup_to_anchor",
                list(metadata.get("numeric_features") or []),
            )

    def test_v4_dataset_includes_planned_target_features(self):
        t0 = timezone.now() - timedelta(days=21)
        self._create_exposed(t0)
        self._create_tx(self.p_cleanser, t0 - timedelta(days=1), "v4-planned-1")
        self._create_tx(self.p_serum, t0 + timedelta(days=1), "v4-planned-2")

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
            serum_row = frame[frame["candidate_type"].astype(str) == "serum"].iloc[0]
            cleanser_row = frame[frame["candidate_type"].astype(str) == "cleanser"].iloc[0]

            self.assertIn("planned_target_product_type", frame.columns)
            self.assertIn("candidate_matches_planned_target", frame.columns)
            self.assertEqual(str(serum_row["planned_target_product_type"]), "serum")
            self.assertEqual(int(serum_row["planned_target_step_index"]), 1)
            self.assertEqual(int(serum_row["candidate_matches_planned_target"]), 1)
            self.assertEqual(int(serum_row["candidate_distance_from_planned_target"]), 0)
            self.assertEqual(int(cleanser_row["candidate_distance_from_planned_target"]), -1)

            metadata = json.loads((out_dir / "metadata.json").read_text(encoding="utf-8"))
            self.assertIn(
                "planned_target_product_type",
                list(metadata.get("categorical_features") or []),
            )
            self.assertIn(
                "candidate_matches_planned_target",
                list(metadata.get("numeric_features") or []),
            )

    def test_v4_dataset_prefers_step_completed_semantic_label_over_transaction_fallback(self):
        t0 = timezone.now() - timedelta(days=18)
        self._create_exposed(t0)
        self._create_tx(self.p_cleanser, t0 - timedelta(days=1), "v4-semantic-prior")
        self._create_completed(
            t0 + timedelta(days=1),
            matched_by="semantic_content_match",
            product_type="serum",
            match_meta={
                "recommended_product_id": self.p_serum.id,
                "purchased_product_id": self.p_serum.id,
                "semantic_score": 1.5,
            },
        )

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
            self.assertEqual(str(sample["label"]), "serum")
            self.assertEqual(str(sample["label_source"]), "step_completed_event")
            self.assertEqual(str(sample["label_matched_by"]), "semantic_content_match")
            self.assertGreater(float(sample["sample_weight"]), 1.0)

            metadata = json.loads((out_dir / "metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(
                int((metadata.get("label_source_distribution") or {}).get("step_completed_event", 0)),
                1,
            )
            self.assertEqual(
                int((metadata.get("label_matched_by_distribution") or {}).get("semantic_content_match", 0)),
                1,
            )
            self.assertIn("sample_weight_policy", metadata)

    def test_v4_dataset_includes_multi_window_funnel_targets(self):
        t0 = timezone.now() - timedelta(days=20)
        self._create_exposed(t0)
        self._create_tx(self.p_cleanser, t0 - timedelta(days=1), "v4-funnel-prior")
        self._create_clicked(t0 + timedelta(days=1))
        self._create_completed(
            t0 + timedelta(days=3),
            matched_by="recommended_product_id",
            product_type="serum",
        )
        self._create_tx(self.p_serum, t0 + timedelta(days=5), "v4-funnel-buy-serum")
        self._create_tx(self.p_cleanser, t0 + timedelta(days=10), "v4-funnel-buy-cleanser")

        with TemporaryDirectory() as tmp_dir:
            out_dir = Path(tmp_dir)
            call_command(
                "build_roadmap_ml_dataset_v4",
                days=180,
                out_dir=str(out_dir),
                label_window_days=7,
                outcome_windows="7,14,30",
                popularity_top_n=10,
                owned_top_k=10,
            )
            frame = _read_dataset(out_dir)
            serum_row = frame[frame["candidate_type"].astype(str) == "serum"].iloc[0]
            cleanser_row = frame[frame["candidate_type"].astype(str) == "cleanser"].iloc[0]

            self.assertIn("clicked_in_7d", frame.columns)
            self.assertIn("candidate_primary_target_in_14d", frame.columns)
            self.assertIn("candidate_bought_any_in_14d", frame.columns)
            self.assertEqual(int(serum_row["window_7d_fully_observed"]), 1)
            self.assertEqual(int(serum_row["window_30d_fully_observed"]), 0)
            self.assertEqual(int(serum_row["clicked_in_7d"]), 1)
            self.assertEqual(str(serum_row["completed_product_type_7d"]), "serum")
            self.assertEqual(str(serum_row["primary_target_source_7d"]), "step_completed_event")
            self.assertEqual(int(serum_row["candidate_clicked_planned_target_in_7d"]), 1)
            self.assertEqual(int(serum_row["candidate_completed_in_7d"]), 1)
            self.assertEqual(int(serum_row["candidate_bought_first_in_7d"]), 1)
            self.assertEqual(int(serum_row["candidate_primary_target_in_7d"]), 1)
            self.assertEqual(int(cleanser_row["candidate_bought_any_in_14d"]), 1)
            self.assertEqual(int(cleanser_row["candidate_primary_target_in_14d"]), 0)

            metadata = json.loads((out_dir / "metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(str(metadata.get("label_protocol_version")), "v5_exposure_funnel")
            self.assertEqual(list(metadata.get("outcome_windows_days") or []), [7, 14, 30])
            summary_7d = (metadata.get("outcome_window_summary") or {}).get("7") or {}
            summary_30d = (metadata.get("outcome_window_summary") or {}).get("30") or {}
            self.assertEqual(int(summary_7d.get("clicked_episodes", 0)), 1)
            self.assertEqual(int(summary_7d.get("completed_episodes", 0)), 1)
            self.assertEqual(
                int((summary_7d.get("primary_target_source_distribution") or {}).get("step_completed_event", 0)),
                1,
            )
            self.assertEqual(int(summary_30d.get("fully_observed_episodes", 0)), 0)


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

    def test_v4_artifact_runtime_uses_content_features(self):
        if pd is None:
            self.skipTest("pandas is required")

        User = get_user_model()
        user = User.objects.create_user(username="v4_artifact_u1", password="pass12345")
        profile = CustomerProfile.objects.get(user=user)
        profile.skin_type = "dry"
        profile.goals = ["hydration"]
        profile.hair_profile = {}
        profile.makeup_profile = {}
        profile.fragrance_profile = {}
        profile.save(
            update_fields=[
                "skin_type",
                "goals",
                "hair_profile",
                "makeup_profile",
                "fragrance_profile",
            ]
        )
        cleanser = Product.objects.create(
            name="Artifact Cleanser",
            brand="B",
            price=Decimal("11.00"),
            category="skincare",
            product_type="cleanser",
            concerns=["cleanse"],
            actives=["ceramides"],
            supported_skin_types=["dry"],
            ingredients_inci="aqua, glycerin, ceramide np",
            in_stock=True,
        )
        Product.objects.create(
            name="Artifact Serum",
            brand="B",
            price=Decimal("15.00"),
            category="skincare",
            product_type="serum",
            concerns=["hydration"],
            actives=["niacinamide"],
            supported_skin_types=["dry"],
            ingredients_inci="aqua, glycerin, niacinamide",
            in_stock=True,
        )
        tx = Transaction.objects.create(
            user=user,
            total_amount=Decimal("11.00"),
            channel="web",
            idempotency_key="artifact-runtime-1",
        )
        TransactionItem.objects.create(
            transaction=tx,
            product=cleanser,
            quantity=1,
            unit_price=Decimal("11.00"),
        )

        class DummyArtifactModel:
            def __init__(self):
                self.columns: list[str] = []
                self.seen = None

            def predict(self, X):
                self.columns = list(X.columns)
                self.seen = X.copy()
                score = (
                    pd.to_numeric(
                        X["candidate_is_immediate_followup_to_anchor"], errors="coerce"
                    ).fillna(0.0)
                    * 10.0
                )
                score = score + pd.to_numeric(
                    X["candidate_profile_goal_match_rate"], errors="coerce"
                ).fillna(0.0)
                score = score + pd.to_numeric(
                    X["candidate_anchor_shared_inci_rate"], errors="coerce"
                ).fillna(0.0)
                return score.to_numpy()

        dummy = DummyArtifactModel()
        artifact = {
            "task": "roadmap_nextstep_v4_ranking",
            "model": dummy,
            "preprocessor": None,
            "model_type": "lightgbm_ranker",
            "feature_columns": [
                "category",
                "candidate_type",
                "profile_skin_type",
                "anchor_product_type",
                "candidate_distance_from_anchor",
                "candidate_is_immediate_followup_to_anchor",
                "candidate_profile_goal_match_rate",
                "candidate_anchor_shared_inci_rate",
            ],
            "categorical_features": [
                "category",
                "candidate_type",
                "profile_skin_type",
                "anchor_product_type",
            ],
            "numeric_features": [
                "candidate_distance_from_anchor",
                "candidate_is_immediate_followup_to_anchor",
                "candidate_profile_goal_match_rate",
                "candidate_anchor_shared_inci_rate",
            ],
            "candidate_types_by_category": {"skincare": ["cleanser", "serum"]},
            "rules_chain_by_category": {"skincare": ["cleanser", "serum"]},
            "candidate_popularity_in_train_by_category": {
                "skincare": {"cleanser": 0.5, "serum": 0.5}
            },
            "owned_feature_columns": [],
            "owned_feature_map": {},
            "temperature": 1.0,
        }

        with patch("roadmap_app.ml_next_step._load_model", return_value=artifact):
            rows = predict_next_product_types(
                user=user,
                context_product_ids=[],
                category="skincare",
                candidate_types=["cleanser", "serum"],
            )

        self.assertEqual([row["candidate_type"] for row in rows], ["serum", "cleanser"])
        self.assertIn("candidate_profile_goal_match_rate", dummy.columns)
        self.assertIn("candidate_anchor_shared_inci_rate", dummy.columns)
        self.assertIn("candidate_distance_from_anchor", dummy.columns)
        self.assertIn("candidate_is_immediate_followup_to_anchor", dummy.columns)
        self.assertIsNotNone(dummy.seen)
        serum_row = dummy.seen[dummy.seen["candidate_type"].astype(str) == "serum"].iloc[0]
        self.assertEqual(str(serum_row["profile_skin_type"]), "dry")
        self.assertEqual(str(serum_row["anchor_product_type"]), "cleanser")
        self.assertEqual(int(serum_row["candidate_distance_from_anchor"]), 1)
        self.assertEqual(int(serum_row["candidate_is_immediate_followup_to_anchor"]), 1)
        self.assertGreater(float(serum_row["candidate_profile_goal_match_rate"]), 0.0)

    def test_v4_artifact_runtime_uses_planned_target_features(self):
        if pd is None:
            self.skipTest("pandas is required")

        User = get_user_model()
        user = User.objects.create_user(username="v4_planned_target_u1", password="pass12345")
        cleanser = Product.objects.create(
            name="Planned Cleanser",
            brand="B",
            price=Decimal("11.00"),
            category="skincare",
            product_type="cleanser",
            in_stock=True,
        )
        Product.objects.create(
            name="Planned Serum",
            brand="B",
            price=Decimal("15.00"),
            category="skincare",
            product_type="serum",
            in_stock=True,
        )
        tx = Transaction.objects.create(
            user=user,
            total_amount=Decimal("11.00"),
            channel="web",
            idempotency_key="artifact-planned-target-1",
        )
        TransactionItem.objects.create(
            transaction=tx,
            product=cleanser,
            quantity=1,
            unit_price=Decimal("11.00"),
        )

        class DummyArtifactModel:
            def __init__(self):
                self.seen = None

            def predict(self, X):
                self.seen = X.copy()
                score = pd.to_numeric(X["candidate_matches_planned_target"], errors="coerce").fillna(0.0)
                return score.to_numpy()

        dummy = DummyArtifactModel()
        artifact = {
            "task": "roadmap_nextstep_v4_ranking",
            "model": dummy,
            "preprocessor": None,
            "model_type": "lightgbm_ranker",
            "feature_columns": [
                "category",
                "candidate_type",
                "planned_target_product_type",
                "planned_target_step_index",
                "candidate_matches_planned_target",
            ],
            "categorical_features": [
                "category",
                "candidate_type",
                "planned_target_product_type",
            ],
            "numeric_features": [
                "planned_target_step_index",
                "candidate_matches_planned_target",
            ],
            "candidate_types_by_category": {"skincare": ["cleanser", "serum"]},
            "rules_chain_by_category": {"skincare": ["cleanser", "serum"]},
            "candidate_popularity_in_train_by_category": {
                "skincare": {"cleanser": 0.5, "serum": 0.5}
            },
            "owned_feature_columns": [],
            "owned_feature_map": {},
            "temperature": 1.0,
        }

        with patch("roadmap_app.ml_next_step._load_model", return_value=artifact):
            rows = predict_next_product_types(
                user=user,
                context_product_ids=[],
                category="skincare",
                planned_target_product_type="serum",
                planned_target_step_index=1,
                candidate_types=["cleanser", "serum"],
            )

        self.assertEqual([row["candidate_type"] for row in rows], ["serum", "cleanser"])
        self.assertIsNotNone(dummy.seen)
        self.assertEqual(set(dummy.seen["planned_target_product_type"].astype(str).tolist()), {"serum"})
        serum_row = dummy.seen[dummy.seen["candidate_type"].astype(str) == "serum"].iloc[0]
        self.assertEqual(int(serum_row["planned_target_step_index"]), 1)
        self.assertEqual(int(serum_row["candidate_matches_planned_target"]), 1)

    def test_v4_artifact_runtime_uses_context_product_ids_as_anchor(self):
        if pd is None:
            self.skipTest("pandas is required")

        User = get_user_model()
        user = User.objects.create_user(username="v4_context_anchor_u1", password="pass12345")
        shampoo = Product.objects.create(
            name="Context Shampoo",
            brand="B",
            price=Decimal("11.00"),
            category="haircare",
            product_type="shampoo",
            in_stock=True,
        )
        conditioner = Product.objects.create(
            name="Context Conditioner",
            brand="B",
            price=Decimal("13.00"),
            category="haircare",
            product_type="conditioner",
            in_stock=True,
        )
        Product.objects.create(
            name="Context Mask",
            brand="B",
            price=Decimal("17.00"),
            category="haircare",
            product_type="hair_mask",
            in_stock=True,
        )
        tx = Transaction.objects.create(
            user=user,
            total_amount=Decimal("11.00"),
            channel="web",
            idempotency_key="artifact-context-anchor-1",
        )
        TransactionItem.objects.create(
            transaction=tx,
            product=shampoo,
            quantity=1,
            unit_price=Decimal("11.00"),
        )

        class DummyArtifactModel:
            def __init__(self):
                self.seen = None

            def predict(self, X):
                self.seen = X.copy()
                return pd.Series([0.0] * len(X), index=X.index, dtype=float).to_numpy()

        dummy = DummyArtifactModel()
        artifact = {
            "task": "roadmap_nextstep_v4_ranking",
            "model": dummy,
            "preprocessor": None,
            "model_type": "lightgbm_ranker",
            "feature_columns": [
                "category",
                "candidate_type",
                "anchor_product_type",
            ],
            "categorical_features": [
                "category",
                "candidate_type",
                "anchor_product_type",
            ],
            "numeric_features": [],
            "candidate_types_by_category": {"haircare": ["shampoo", "conditioner", "hair_mask"]},
            "rules_chain_by_category": {"haircare": ["shampoo", "conditioner", "hair_mask"]},
            "candidate_popularity_in_train_by_category": {
                "haircare": {"shampoo": 0.5, "conditioner": 0.3, "hair_mask": 0.2}
            },
            "owned_feature_columns": [],
            "owned_feature_map": {},
            "temperature": 1.0,
        }

        with patch("roadmap_app.ml_next_step._load_model", return_value=artifact):
            predict_next_product_types(
                user=user,
                context_product_ids=[conditioner.id],
                category="haircare",
                candidate_types=["shampoo", "conditioner", "hair_mask"],
            )

        self.assertIsNotNone(dummy.seen)
        self.assertEqual(set(dummy.seen["anchor_product_type"].astype(str).tolist()), {"conditioner"})

    def test_v4_artifact_runtime_applies_haircare_progression_bias(self):
        if pd is None:
            self.skipTest("pandas is required")

        User = get_user_model()
        user = User.objects.create_user(username="v4_haircare_bias_u1", password="pass12345")
        shampoo = Product.objects.create(
            name="Bias Shampoo",
            brand="B",
            price=Decimal("11.00"),
            category="haircare",
            product_type="shampoo",
            in_stock=True,
        )
        Product.objects.create(
            name="Bias Conditioner",
            brand="B",
            price=Decimal("13.00"),
            category="haircare",
            product_type="conditioner",
            in_stock=True,
        )
        tx = Transaction.objects.create(
            user=user,
            total_amount=Decimal("11.00"),
            channel="web",
            idempotency_key="artifact-haircare-bias-1",
        )
        TransactionItem.objects.create(
            transaction=tx,
            product=shampoo,
            quantity=1,
            unit_price=Decimal("11.00"),
        )

        class DummyArtifactModel:
            def predict(self, X):
                is_shampoo = X["candidate_type"].astype(str) == "shampoo"
                score = pd.Series([0.52] * len(X), index=X.index, dtype=float)
                score.loc[is_shampoo] = 0.60
                score.loc[~is_shampoo] = 0.55
                return score.to_numpy()

        artifact = {
            "task": "roadmap_nextstep_v4_ranking",
            "model": DummyArtifactModel(),
            "preprocessor": None,
            "model_type": "lightgbm_ranker",
            "feature_columns": [
                "category",
                "candidate_type",
                "anchor_product_type",
            ],
            "categorical_features": [
                "category",
                "candidate_type",
                "anchor_product_type",
            ],
            "numeric_features": [],
            "candidate_types_by_category": {"haircare": ["shampoo", "conditioner"]},
            "rules_chain_by_category": {"haircare": ["shampoo", "conditioner"]},
            "candidate_popularity_in_train_by_category": {
                "haircare": {"shampoo": 0.5, "conditioner": 0.5}
            },
            "owned_feature_columns": [],
            "owned_feature_map": {},
            "temperature": 1.0,
        }

        with override_settings(
            ROADMAP_NEXTSTEP_V4_HAIRCARE_RUNTIME_BIAS_ENABLED=True,
        ), patch("roadmap_app.ml_next_step._load_model", return_value=artifact):
            rows = predict_next_product_types(
                user=user,
                context_product_ids=[],
                category="haircare",
                candidate_types=["shampoo", "conditioner"],
            )

        self.assertEqual([row["candidate_type"] for row in rows], ["conditioner", "shampoo"])


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


class RoadmapNextStepV4HaircareScalpChainTests(TestCase):
    def setUp(self):
        if pd is None:
            self.skipTest("pandas is required")

        User = get_user_model()
        self.user = User.objects.create_user(username="v4_haircare_scalp_u1", password="pass12345")
        profile = CustomerProfile.objects.get(user=self.user)
        profile.skin_type = "normal"
        profile.goals = ["scalp_balance"]
        profile.hair_profile = {
            "hair_type": "wavy",
            "scalp_type": "oily",
            "hair_thickness": "medium",
            "concerns": ["flakes", "itchiness", "oiliness"],
        }
        profile.makeup_profile = {}
        profile.fragrance_profile = {}
        profile.save(
            update_fields=[
                "skin_type",
                "goals",
                "hair_profile",
                "makeup_profile",
                "fragrance_profile",
            ]
        )

        self.p_shampoo = Product.objects.create(
            name="Scalp Chain Shampoo",
            brand="B",
            price=Decimal("12.00"),
            category="haircare",
            product_type="shampoo",
            concerns=["oiliness", "flakes"],
            actives=["salicylic_acid", "niacinamide"],
            attrs={"hair_type": "wavy", "scalp_type": "oily", "hair_thickness": "medium"},
            ingredients_inci="water, salicylic acid, niacinamide",
            in_stock=True,
        )
        self.p_scalp_serum = Product.objects.create(
            name="Scalp Chain Serum",
            brand="B",
            price=Decimal("19.00"),
            category="haircare",
            product_type="scalp_serum",
            concerns=["oiliness", "flakes", "itchiness"],
            actives=["salicylic_acid", "niacinamide", "tea_tree"],
            attrs={"hair_type": "wavy", "scalp_type": "oily", "hair_thickness": "medium"},
            ingredients_inci="water, salicylic acid, niacinamide, tea tree",
            in_stock=True,
        )
        self.p_conditioner = Product.objects.create(
            name="Scalp Chain Conditioner",
            brand="B",
            price=Decimal("14.00"),
            category="haircare",
            product_type="conditioner",
            concerns=["damage", "dryness"],
            actives=["keratin"],
            attrs={"hair_type": "wavy", "scalp_type": "normal", "hair_thickness": "medium"},
            ingredients_inci="water, keratin, panthenol",
            in_stock=True,
        )

        self.plan = RoadmapPlan.objects.create(user=self.user, category="haircare", is_active=True, meta={})
        self.step = RoadmapStep.objects.create(
            plan=self.plan,
            step_index=2,
            product_type="scalp_serum",
            status=RoadmapStep.Status.RECOMMENDED,
        )

    def _create_exposed(self, at_dt):
        event = RoadmapEvent.objects.create(
            user=self.user,
            plan=self.plan,
            step=self.step,
            event_type=RoadmapEvent.Type.STEP_EXPOSED,
            context={"category": "haircare", "sources": ["roadmap_api"]},
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

    def test_dataset_reorders_scalp_serum_chain_after_shampoo(self):
        t0 = timezone.now() - timedelta(days=14)
        self._create_exposed(t0)
        self._create_tx(self.p_shampoo, t0 - timedelta(days=1), "v4-scalp-chain-ds-1")
        self._create_tx(self.p_scalp_serum, t0 + timedelta(days=2), "v4-scalp-chain-ds-2")

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
            scalp_row = frame[frame["candidate_type"].astype(str) == "scalp_serum"].iloc[0]
            conditioner_row = frame[frame["candidate_type"].astype(str) == "conditioner"].iloc[0]

            self.assertEqual(int(scalp_row["candidate_position_in_chain"]), 1)
            self.assertEqual(int(conditioner_row["candidate_position_in_chain"]), 2)
            self.assertEqual(int(scalp_row["candidate_is_immediate_followup_to_anchor"]), 1)
            self.assertEqual(int(conditioner_row["candidate_is_immediate_followup_to_anchor"]), 0)
            self.assertEqual(int(scalp_row["candidate_matches_planned_target"]), 1)

    def test_dataset_includes_scalp_objective_features(self):
        t0 = timezone.now() - timedelta(days=14)
        self._create_exposed(t0)
        self._create_tx(self.p_shampoo, t0 - timedelta(days=1), "v4-scalp-obj-ds-1")
        self._create_tx(self.p_scalp_serum, t0 + timedelta(days=2), "v4-scalp-obj-ds-2")

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
            scalp_row = frame[frame["candidate_type"].astype(str) == "scalp_serum"].iloc[0]
            conditioner_row = frame[frame["candidate_type"].astype(str) == "conditioner"].iloc[0]

            self.assertIn("profile_has_scalp_objective", frame.columns)
            self.assertIn("candidate_profile_scalp_objective_match_rate", frame.columns)
            self.assertIn("candidate_scalp_concern_focus_rate", frame.columns)
            self.assertIn("candidate_is_scalp_specialty", frame.columns)
            self.assertEqual(int(scalp_row["profile_has_scalp_objective"]), 1)
            self.assertEqual(int(scalp_row["anchor_has_scalp_focus"]), 1)
            self.assertEqual(int(scalp_row["candidate_is_scalp_specialty"]), 1)
            self.assertEqual(int(conditioner_row["candidate_is_scalp_specialty"]), 0)
            self.assertGreater(
                float(scalp_row["candidate_profile_scalp_objective_match_rate"]),
                float(conditioner_row["candidate_profile_scalp_objective_match_rate"]),
            )
            self.assertGreater(
                float(scalp_row["candidate_scalp_concern_focus_rate"]),
                float(conditioner_row["candidate_scalp_concern_focus_rate"]),
            )

    def test_runtime_reorders_scalp_serum_chain_after_shampoo(self):
        if pd is None:
            self.skipTest("pandas is required")

        t0 = timezone.now() - timedelta(days=10)
        self._create_tx(self.p_shampoo, t0 - timedelta(days=1), "v4-scalp-chain-rt-1")

        class DummyArtifactModel:
            def __init__(self):
                self.seen = None

            def predict(self, X):
                self.seen = X.copy()
                return pd.to_numeric(
                    X["candidate_is_immediate_followup_to_anchor"],
                    errors="coerce",
                ).fillna(0.0).to_numpy()

        dummy = DummyArtifactModel()
        artifact = {
            "task": "roadmap_nextstep_v4_ranking",
            "model": dummy,
            "preprocessor": None,
            "model_type": "lightgbm_ranker",
            "feature_columns": [
                "category",
                "candidate_type",
                "planned_target_product_type",
                "candidate_position_in_chain",
                "candidate_is_immediate_followup_to_anchor",
            ],
            "categorical_features": [
                "category",
                "candidate_type",
                "planned_target_product_type",
            ],
            "numeric_features": [
                "candidate_position_in_chain",
                "candidate_is_immediate_followup_to_anchor",
            ],
            "candidate_types_by_category": {"haircare": ["conditioner", "scalp_serum", "hair_mask"]},
            "rules_chain_by_category": {
                "haircare": ["shampoo", "conditioner", "hair_mask", "hair_oil", "scalp_serum", "leave_in"]
            },
            "candidate_popularity_in_train_by_category": {
                "haircare": {"conditioner": 0.5, "scalp_serum": 0.2, "hair_mask": 0.3}
            },
            "owned_feature_columns": [],
            "owned_feature_map": {},
            "temperature": 1.0,
        }

        with patch("roadmap_app.ml_next_step._load_model", return_value=artifact):
            rows = predict_next_product_types(
                user=self.user,
                context_product_ids=[],
                category="haircare",
                planned_target_product_type="scalp_serum",
                planned_target_step_index=2,
                candidate_types=["conditioner", "scalp_serum", "hair_mask"],
            )

        self.assertEqual(rows[0]["candidate_type"], "scalp_serum")
        self.assertIsNotNone(dummy.seen)
        scalp_row = dummy.seen[dummy.seen["candidate_type"].astype(str) == "scalp_serum"].iloc[0]
        conditioner_row = dummy.seen[dummy.seen["candidate_type"].astype(str) == "conditioner"].iloc[0]
        self.assertEqual(int(scalp_row["candidate_position_in_chain"]), 1)
        self.assertEqual(int(conditioner_row["candidate_position_in_chain"]), 2)
        self.assertEqual(int(scalp_row["candidate_is_immediate_followup_to_anchor"]), 1)
        self.assertEqual(int(conditioner_row["candidate_is_immediate_followup_to_anchor"]), 0)

    def test_runtime_scalp_rerank_promotes_scalp_serum_for_scalp_target(self):
        if pd is None:
            self.skipTest("pandas is required")

        t0 = timezone.now() - timedelta(days=10)
        self._create_tx(self.p_shampoo, t0 - timedelta(days=1), "v4-scalp-rerank-1")

        class DummyArtifactModel:
            def predict(self, X):
                base = pd.Series([0.0] * len(X), index=X.index, dtype=float)
                base = base + X["candidate_type"].astype(str).map(
                    {"conditioner": 1.0, "scalp_serum": 0.2, "hair_mask": 0.3}
                ).fillna(0.0)
                return base.to_numpy()

        artifact = {
            "task": "roadmap_nextstep_v4_ranking",
            "model": DummyArtifactModel(),
            "preprocessor": None,
            "model_type": "lightgbm_ranker",
            "feature_columns": [
                "category",
                "candidate_type",
                "planned_target_product_type",
            ],
            "categorical_features": [
                "category",
                "candidate_type",
                "planned_target_product_type",
            ],
            "numeric_features": [],
            "candidate_types_by_category": {"haircare": ["conditioner", "scalp_serum", "hair_mask"]},
            "rules_chain_by_category": {
                "haircare": ["shampoo", "conditioner", "hair_mask", "hair_oil", "scalp_serum", "leave_in"]
            },
            "candidate_popularity_in_train_by_category": {
                "haircare": {"conditioner": 0.5, "scalp_serum": 0.2, "hair_mask": 0.3}
            },
            "owned_feature_columns": [],
            "owned_feature_map": {},
            "temperature": 1.0,
        }

        with override_settings(ROADMAP_NEXTSTEP_V4_HAIRCARE_SCALP_RERANK_ENABLED=True):
            with patch("roadmap_app.ml_next_step._load_model", return_value=artifact):
                rows = predict_next_product_types(
                    user=self.user,
                    context_product_ids=[],
                    category="haircare",
                    planned_target_product_type="scalp_serum",
                    planned_target_step_index=2,
                    candidate_types=["conditioner", "scalp_serum", "hair_mask"],
                )

        self.assertEqual(rows[0]["candidate_type"], "scalp_serum")
        self.assertIn("haircare_scalp_rerank", rows[0]["runtime_policies"])
        self.assertGreater(float(rows[0]["runtime_policy_biases"]["haircare_scalp_rerank"]), 0.0)

    def test_runtime_scalp_rerank_does_not_override_conditioner_control(self):
        if pd is None:
            self.skipTest("pandas is required")

        t0 = timezone.now() - timedelta(days=10)
        self._create_tx(self.p_shampoo, t0 - timedelta(days=1), "v4-scalp-rerank-2")

        class DummyArtifactModel:
            def predict(self, X):
                base = pd.Series([0.0] * len(X), index=X.index, dtype=float)
                base = base + X["candidate_type"].astype(str).map(
                    {"conditioner": 1.0, "scalp_serum": 0.2, "hair_mask": 0.3}
                ).fillna(0.0)
                return base.to_numpy()

        artifact = {
            "task": "roadmap_nextstep_v4_ranking",
            "model": DummyArtifactModel(),
            "preprocessor": None,
            "model_type": "lightgbm_ranker",
            "feature_columns": [
                "category",
                "candidate_type",
                "planned_target_product_type",
            ],
            "categorical_features": [
                "category",
                "candidate_type",
                "planned_target_product_type",
            ],
            "numeric_features": [],
            "candidate_types_by_category": {"haircare": ["conditioner", "scalp_serum", "hair_mask"]},
            "rules_chain_by_category": {
                "haircare": ["shampoo", "conditioner", "hair_mask", "hair_oil", "scalp_serum", "leave_in"]
            },
            "candidate_popularity_in_train_by_category": {
                "haircare": {"conditioner": 0.5, "scalp_serum": 0.2, "hair_mask": 0.3}
            },
            "owned_feature_columns": [],
            "owned_feature_map": {},
            "temperature": 1.0,
        }

        with override_settings(ROADMAP_NEXTSTEP_V4_HAIRCARE_SCALP_RERANK_ENABLED=True):
            with patch("roadmap_app.ml_next_step._load_model", return_value=artifact):
                rows = predict_next_product_types(
                    user=self.user,
                    context_product_ids=[],
                    category="haircare",
                    planned_target_product_type="conditioner",
                    planned_target_step_index=2,
                    candidate_types=["conditioner", "scalp_serum", "hair_mask"],
                )

        self.assertEqual(rows[0]["candidate_type"], "conditioner")


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


    def _create_haircare_products(self, suffix: str) -> None:
        Product.objects.create(
            name=f"Haircare Shampoo {suffix}",
            brand="B",
            category="haircare",
            product_type="shampoo",
            in_stock=True,
        )
        Product.objects.create(
            name=f"Haircare Conditioner {suffix}",
            brand="B",
            category="haircare",
            product_type="conditioner",
            in_stock=True,
        )
        Product.objects.create(
            name=f"Haircare Mask {suffix}",
            brand="B",
            category="haircare",
            product_type="hair_mask",
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
            ROADMAP_PLANNER_V1_MODE="off",
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
            ROADMAP_PLANNER_V1_MODE="off",
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
        self.assertEqual(str(ml_meta.get("planned_target_product_type")), "foundation")
        self.assertEqual(int(ml_meta.get("planned_target_step_index")), 1)
        predict_mock.assert_not_called()

    def test_selected_makeup_users_can_reach_model_used(self):
        User = get_user_model()
        user = User.objects.create_user(username="v4_partial_yes_u1", password="pass12345")
        self._create_makeup_products("yes")

        with override_settings(
            ROADMAP_NEXTSTEP_V4_ENABLED=True,
            ROADMAP_NEXTSTEP_V3_ENABLED=False,
            ROADMAP_PLANNER_V1_MODE="off",
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

    def test_selected_haircare_users_can_reach_model_used_via_generic_partial_rollout(self):
        User = get_user_model()
        user = User.objects.create_user(username="v4_partial_hair_yes_u1", password="pass12345")
        self._create_haircare_products("yes")

        with override_settings(
            ROADMAP_NEXTSTEP_V4_ENABLED=True,
            ROADMAP_NEXTSTEP_V3_ENABLED=False,
            ROADMAP_PLANNER_V1_MODE="off",
            ROADMAP_NEXTSTEP_V4_ENABLED_CATEGORIES=["skincare", "haircare", "makeup"],
            ROADMAP_NEXTSTEP_V4_DISABLED_CATEGORIES=["fragrance"],
            ROADMAP_NEXTSTEP_V4_PARTIAL_ENABLED_CATEGORIES=["haircare"],
            ROADMAP_NEXTSTEP_V4_PARTIAL_PURCHASE_CONTEXT_ONLY_CATEGORIES=[],
            ROADMAP_NEXTSTEP_V4_PARTIAL_HAIRCARE_PRODUCT_TYPES=["shampoo"],
            ROADMAP_NEXTSTEP_V4_PARTIAL_HAIRCARE_STEP_INDEXES=["1"],
            ROADMAP_NEXTSTEP_V4_PARTIAL_HAIRCARE_MODEL_PATH="",
            ROADMAP_NEXTSTEP_V4_PARTIAL_PERCENT=100,
            ROADMAP_NEXTSTEP_V4_PARTIAL_SALT="partial_hair_yes_test",
            ROADMAP_NEXTSTEP_V4_CONFIDENCE_THRESHOLD=0.1,
        ), patch(
            "roadmap_app.services.v4_min_lift_guard_status",
            return_value={"passed": True, "reason": "ok"},
        ), patch(
            "roadmap_app.services.v4_category_staged_rollout_status",
            side_effect=lambda cat: self._hold_status(cat),
        ), patch(
            "roadmap_app.services.predict_next_product_types",
            return_value=[{"candidate_type": "shampoo", "score": 0.93}],
        ) as predict_mock:
            plan = refresh_roadmap(user, category="haircare", post_ctx=None)

        ml_meta = (plan.meta or {}).get("ml") if isinstance(plan.meta, dict) else {}
        self.assertEqual(str(ml_meta.get("decision")), "model_used")
        self.assertEqual(str(ml_meta.get("rollout_mode")), "partial")
        self.assertTrue(bool(ml_meta.get("rollout_selected")))
        self.assertEqual(str(ml_meta.get("partial_match_product_type")), "shampoo")
        self.assertEqual(int(ml_meta.get("partial_match_step_index")), 1)
        self.assertEqual(str(ml_meta.get("planned_target_product_type")), "shampoo")
        self.assertEqual(int(ml_meta.get("planned_target_step_index")), 1)
        self.assertIsNone(ml_meta.get("fallback_reason"))
        predict_mock.assert_called_once()

    def test_purchase_context_progression_can_select_hair_mask_partial_candidate(self):
        User = get_user_model()
        user = User.objects.create_user(username="v4_partial_hair_purchase_ctx_u1", password="pass12345")
        self._create_haircare_products("purchase-ctx")
        shampoo = Product.objects.get(category="haircare", product_type="shampoo", name__contains="purchase-ctx")
        conditioner = Product.objects.get(
            category="haircare",
            product_type="conditioner",
            name__contains="purchase-ctx",
        )

        now = timezone.now()
        OwnedProduct.objects.create(
            user=user,
            product=shampoo,
            quantity_total=1,
            is_active=True,
            last_acquired_at=now - timedelta(days=7),
        )
        OwnedProduct.objects.create(
            user=user,
            product=conditioner,
            quantity_total=1,
            is_active=True,
            last_acquired_at=now,
        )

        def _artifact_summary(path: str | None):
            raw = str(path or "")
            if raw == "C:/tmp/roadmap_nextstep_semantic_v4.pkl":
                return {
                    "model_version": "semantic_v4_canary",
                    "selected_feature_set": "full",
                }
            return {
                "model_version": "active_v4",
                "selected_feature_set": "baseline_only",
            }

        with override_settings(
            ROADMAP_NEXTSTEP_V4_ENABLED=True,
            ROADMAP_NEXTSTEP_V3_ENABLED=False,
            ROADMAP_PLANNER_V1_MODE="off",
            ROADMAP_NEXTSTEP_V4_ENABLED_CATEGORIES=["skincare", "haircare", "makeup"],
            ROADMAP_NEXTSTEP_V4_DISABLED_CATEGORIES=["fragrance"],
            ROADMAP_NEXTSTEP_V4_PARTIAL_ENABLED_CATEGORIES=["haircare"],
            ROADMAP_NEXTSTEP_V4_PARTIAL_PURCHASE_CONTEXT_ONLY_CATEGORIES=["haircare"],
            ROADMAP_NEXTSTEP_V4_PARTIAL_FORCE_ENABLED_CATEGORIES=["haircare"],
            ROADMAP_NEXTSTEP_V4_PARTIAL_HAIRCARE_PRODUCT_TYPES=["hair_mask"],
            ROADMAP_NEXTSTEP_V4_PARTIAL_HAIRCARE_STEP_INDEXES=["3"],
            ROADMAP_NEXTSTEP_V4_PARTIAL_PERCENT=0,
            ROADMAP_NEXTSTEP_V4_PARTIAL_HAIRCARE_PERCENT=100,
            ROADMAP_NEXTSTEP_V4_PARTIAL_SALT="partial_hair_purchase_ctx_test",
            ROADMAP_NEXTSTEP_V4_PARTIAL_HAIRCARE_MODEL_PATH="C:/tmp/roadmap_nextstep_semantic_v4.pkl",
            ROADMAP_NEXTSTEP_V4_SHADOW_MODEL_PATH="",
            ROADMAP_NEXTSTEP_V4_CONFIDENCE_THRESHOLD=0.1,
        ), patch(
            "roadmap_app.services.v4_min_lift_guard_status",
            return_value={"passed": True, "reason": "ok"},
        ), patch(
            "roadmap_app.services.v4_category_staged_rollout_status",
            side_effect=lambda cat: self._hold_status(cat),
        ), patch(
            "roadmap_app.services.predict_next_product_types",
        ) as active_mock, patch(
            "roadmap_app.services.predict_next_product_types_for_model_path",
            return_value=[{"candidate_type": "hair_mask", "score": 0.97}],
        ) as candidate_mock, patch(
            "roadmap_app.services.nextstep_model_artifact_summary",
            side_effect=_artifact_summary,
        ):
            plan = refresh_roadmap(
                user,
                category="haircare",
                post_ctx={
                    "categories": ["haircare"],
                    "product_ids": [int(conditioner.id)],
                },
            )

        ml_meta = (plan.meta or {}).get("ml") if isinstance(plan.meta, dict) else {}
        self.assertEqual(str(ml_meta.get("decision")), "model_used")
        self.assertEqual(str(ml_meta.get("rollout_mode")), "partial")
        self.assertTrue(bool(ml_meta.get("rollout_selected")))
        self.assertEqual(str(ml_meta.get("model_slot")), "partial_candidate")
        self.assertEqual(str(ml_meta.get("planned_target_product_type")), "hair_mask")
        self.assertEqual(int(ml_meta.get("planned_target_step_index")), 3)
        self.assertEqual(str(ml_meta.get("partial_match_product_type")), "hair_mask")
        self.assertEqual(int(ml_meta.get("partial_match_step_index")), 3)
        self.assertEqual(str(((plan.meta or {}).get("context") or {}).get("refresh_caller")), "update_roadmap_from_purchase")
        active_mock.assert_not_called()
        candidate_mock.assert_called_once()

    def test_purchase_context_only_partial_does_not_run_on_cold_refresh(self):
        User = get_user_model()
        user = User.objects.create_user(username="v4_partial_hair_purchase_only_cold_u1", password="pass12345")
        self._create_haircare_products("purchase-only-cold")

        with override_settings(
            ROADMAP_NEXTSTEP_V4_ENABLED=True,
            ROADMAP_NEXTSTEP_V3_ENABLED=False,
            ROADMAP_PLANNER_V1_MODE="off",
            ROADMAP_NEXTSTEP_V4_ENABLED_CATEGORIES=["skincare", "haircare", "makeup"],
            ROADMAP_NEXTSTEP_V4_DISABLED_CATEGORIES=["fragrance"],
            ROADMAP_NEXTSTEP_V4_PARTIAL_ENABLED_CATEGORIES=["haircare"],
            ROADMAP_NEXTSTEP_V4_PARTIAL_PURCHASE_CONTEXT_ONLY_CATEGORIES=["haircare"],
            ROADMAP_NEXTSTEP_V4_PARTIAL_FORCE_ENABLED_CATEGORIES=["haircare"],
            ROADMAP_NEXTSTEP_V4_PARTIAL_HAIRCARE_PRODUCT_TYPES=["shampoo"],
            ROADMAP_NEXTSTEP_V4_PARTIAL_HAIRCARE_STEP_INDEXES=["1"],
            ROADMAP_NEXTSTEP_V4_PARTIAL_PERCENT=0,
            ROADMAP_NEXTSTEP_V4_PARTIAL_HAIRCARE_PERCENT=100,
            ROADMAP_NEXTSTEP_V4_PARTIAL_SALT="partial_hair_purchase_only_cold_test",
            ROADMAP_NEXTSTEP_V4_PARTIAL_HAIRCARE_MODEL_PATH="C:/tmp/roadmap_nextstep_semantic_v4.pkl",
            ROADMAP_NEXTSTEP_V4_SHADOW_MODEL_PATH="",
            ROADMAP_NEXTSTEP_V4_CONFIDENCE_THRESHOLD=0.1,
        ), patch(
            "roadmap_app.services.v4_min_lift_guard_status",
            return_value={"passed": True, "reason": "ok"},
        ), patch(
            "roadmap_app.services.v4_category_staged_rollout_status",
            side_effect=lambda cat: self._hold_status(cat),
        ), patch(
            "roadmap_app.services.predict_next_product_types",
        ) as active_mock, patch(
            "roadmap_app.services.predict_next_product_types_for_model_path",
        ) as candidate_mock:
            plan = refresh_roadmap(user, category="haircare", post_ctx=None)

        ml_meta = (plan.meta or {}).get("ml") if isinstance(plan.meta, dict) else {}
        self.assertEqual(str(ml_meta.get("decision")), "fallback")
        self.assertEqual(str(ml_meta.get("fallback_reason")), "partial_rollout_not_selected")
        self.assertEqual(str(ml_meta.get("rollout_mode")), "partial")
        self.assertFalse(bool(ml_meta.get("rollout_selected")))
        self.assertEqual(str(ml_meta.get("rollout_reason")), "purchase_context_only")
        self.assertEqual(str(ml_meta.get("planned_target_product_type")), "shampoo")
        active_mock.assert_not_called()
        candidate_mock.assert_not_called()

    def test_purchase_context_partial_does_not_materialize_when_no_actionable_step(self):
        User = get_user_model()
        user = User.objects.create_user(username="v4_partial_hair_purchase_complete_u1", password="pass12345")
        owned_products = {}
        for product_type in [
            "shampoo",
            "conditioner",
            "hair_mask",
            "hair_oil",
            "scalp_serum",
            "leave_in",
        ]:
            product = Product.objects.create(
                name=f"Haircare {product_type} complete",
                brand="B",
                category="haircare",
                product_type=product_type,
                in_stock=True,
            )
            owned_products[product_type] = product
            OwnedProduct.objects.create(
                user=user,
                product=product,
                quantity_total=1,
                is_active=True,
            )

        with override_settings(
            ROADMAP_NEXTSTEP_V4_ENABLED=True,
            ROADMAP_NEXTSTEP_V3_ENABLED=False,
            ROADMAP_PLANNER_V1_MODE="off",
            ROADMAP_NEXTSTEP_V4_ENABLED_CATEGORIES=["skincare", "haircare", "makeup"],
            ROADMAP_NEXTSTEP_V4_DISABLED_CATEGORIES=["fragrance"],
            ROADMAP_NEXTSTEP_V4_PARTIAL_ENABLED_CATEGORIES=["haircare"],
            ROADMAP_NEXTSTEP_V4_PARTIAL_PURCHASE_CONTEXT_ONLY_CATEGORIES=["haircare"],
            ROADMAP_NEXTSTEP_V4_PARTIAL_FORCE_ENABLED_CATEGORIES=["haircare"],
            ROADMAP_NEXTSTEP_V4_PARTIAL_HAIRCARE_PRODUCT_TYPES=["hair_mask"],
            ROADMAP_NEXTSTEP_V4_PARTIAL_HAIRCARE_STEP_INDEXES=["3"],
            ROADMAP_NEXTSTEP_V4_PARTIAL_PERCENT=0,
            ROADMAP_NEXTSTEP_V4_PARTIAL_HAIRCARE_PERCENT=100,
            ROADMAP_NEXTSTEP_V4_PARTIAL_SALT="partial_hair_purchase_complete_test",
            ROADMAP_NEXTSTEP_V4_PARTIAL_HAIRCARE_MODEL_PATH="C:/tmp/roadmap_nextstep_semantic_v4.pkl",
            ROADMAP_NEXTSTEP_V4_SHADOW_MODEL_PATH="",
            ROADMAP_NEXTSTEP_V4_CONFIDENCE_THRESHOLD=0.1,
        ), patch(
            "roadmap_app.services.v4_min_lift_guard_status",
            return_value={"passed": True, "reason": "ok"},
        ), patch(
            "roadmap_app.services.v4_category_staged_rollout_status",
            side_effect=lambda cat: self._hold_status(cat),
        ), patch(
            "roadmap_app.services.predict_next_product_types",
        ) as active_mock, patch(
            "roadmap_app.services.predict_next_product_types_for_model_path",
        ) as candidate_mock:
            plan = refresh_roadmap(
                user,
                category="haircare",
                post_ctx={
                    "categories": ["haircare"],
                    "product_ids": [
                        int(owned_products["shampoo"].id),
                        int(owned_products["leave_in"].id),
                    ],
                },
            )

        ml_meta = (plan.meta or {}).get("ml") if isinstance(plan.meta, dict) else {}
        self.assertEqual(str(ml_meta.get("decision")), "disabled")
        self.assertEqual(str(ml_meta.get("disabled_reason")), "no_actionable_step")
        self.assertEqual(str(ml_meta.get("rollout_reason")), "no_actionable_step")
        self.assertEqual(str(ml_meta.get("model_slot")), "active")
        self.assertEqual(str(ml_meta.get("planned_target_product_type")), "")
        self.assertEqual(int(ml_meta.get("planned_target_step_index") or 0), 0)
        active_mock.assert_not_called()
        candidate_mock.assert_not_called()

    def test_partial_rollout_can_serve_candidate_model_path(self):
        User = get_user_model()
        user = User.objects.create_user(username="v4_partial_hair_canary_u1", password="pass12345")
        self._create_haircare_products("canary")

        def _artifact_summary(path: str | None):
            raw = str(path or "")
            if raw == "C:/tmp/roadmap_nextstep_semantic_v4.pkl":
                return {
                    "model_version": "semantic_v4_canary",
                    "selected_feature_set": "full",
                }
            return {
                "model_version": "active_v4",
                "selected_feature_set": "baseline_only",
            }

        with override_settings(
            ROADMAP_NEXTSTEP_V4_ENABLED=True,
            ROADMAP_NEXTSTEP_V3_ENABLED=False,
            ROADMAP_PLANNER_V1_MODE="off",
            ROADMAP_NEXTSTEP_V4_ENABLED_CATEGORIES=["skincare", "haircare", "makeup"],
            ROADMAP_NEXTSTEP_V4_DISABLED_CATEGORIES=["fragrance"],
            ROADMAP_NEXTSTEP_V4_PARTIAL_ENABLED_CATEGORIES=["haircare"],
            ROADMAP_NEXTSTEP_V4_PARTIAL_PURCHASE_CONTEXT_ONLY_CATEGORIES=[],
            ROADMAP_NEXTSTEP_V4_PARTIAL_HAIRCARE_PRODUCT_TYPES=["shampoo"],
            ROADMAP_NEXTSTEP_V4_PARTIAL_HAIRCARE_STEP_INDEXES=["1"],
            ROADMAP_NEXTSTEP_V4_PARTIAL_PERCENT=100,
            ROADMAP_NEXTSTEP_V4_PARTIAL_SALT="partial_hair_canary_test",
            ROADMAP_NEXTSTEP_V4_PARTIAL_HAIRCARE_MODEL_PATH="C:/tmp/roadmap_nextstep_semantic_v4.pkl",
            ROADMAP_NEXTSTEP_V4_SHADOW_MODEL_PATH="C:/tmp/roadmap_nextstep_semantic_v4.pkl",
            ROADMAP_NEXTSTEP_V4_CONFIDENCE_THRESHOLD=0.1,
        ), patch(
            "roadmap_app.services.v4_min_lift_guard_status",
            return_value={"passed": True, "reason": "ok"},
        ), patch(
            "roadmap_app.services.v4_category_staged_rollout_status",
            side_effect=lambda cat: self._hold_status(cat),
        ), patch(
            "roadmap_app.services.predict_next_product_types",
        ) as active_mock, patch(
            "roadmap_app.services.predict_next_product_types_for_model_path",
            return_value=[{"candidate_type": "shampoo", "score": 0.95}],
        ) as candidate_mock, patch(
            "roadmap_app.services.nextstep_model_artifact_summary",
            side_effect=_artifact_summary,
        ):
            plan = refresh_roadmap(user, category="haircare", post_ctx=None)

        ml_meta = (plan.meta or {}).get("ml") if isinstance(plan.meta, dict) else {}
        shadow_meta = ml_meta.get("shadow") or {}
        self.assertEqual(str(ml_meta.get("decision")), "model_used")
        self.assertEqual(str(ml_meta.get("rollout_mode")), "partial")
        self.assertEqual(str(ml_meta.get("model_slot")), "partial_candidate")
        self.assertEqual(str(ml_meta.get("model_path")), "C:/tmp/roadmap_nextstep_semantic_v4.pkl")
        self.assertEqual(str(ml_meta.get("model_version")), "semantic_v4_canary")
        self.assertEqual(str(ml_meta.get("selected_feature_set")), "full")
        self.assertEqual(str(ml_meta.get("planned_target_product_type")), "shampoo")
        self.assertEqual(int(ml_meta.get("planned_target_step_index")), 1)
        self.assertFalse(bool(shadow_meta.get("enabled")))
        self.assertEqual(str(shadow_meta.get("reason")), "shadow_same_as_active")
        active_mock.assert_not_called()
        candidate_mock.assert_called_once()

    def test_partial_rollout_can_override_full_enable_when_force_enabled_for_category(self):
        User = get_user_model()
        user = User.objects.create_user(username="v4_partial_force_enable_u1", password="pass12345")
        self._create_haircare_products("force-enable")

        def _artifact_summary(path: str | None):
            raw = str(path or "")
            if raw == "C:/tmp/roadmap_nextstep_semantic_v4.pkl":
                return {
                    "model_version": "semantic_v4_canary",
                    "selected_feature_set": "full",
                }
            return {
                "model_version": "active_v4",
                "selected_feature_set": "baseline_only",
            }

        with override_settings(
            ROADMAP_NEXTSTEP_V4_ENABLED=True,
            ROADMAP_NEXTSTEP_V3_ENABLED=False,
            ROADMAP_PLANNER_V1_MODE="off",
            ROADMAP_NEXTSTEP_V4_ENABLED_CATEGORIES=["skincare", "haircare", "makeup"],
            ROADMAP_NEXTSTEP_V4_DISABLED_CATEGORIES=["fragrance"],
            ROADMAP_NEXTSTEP_V4_PARTIAL_ENABLED_CATEGORIES=["haircare"],
            ROADMAP_NEXTSTEP_V4_PARTIAL_PURCHASE_CONTEXT_ONLY_CATEGORIES=[],
            ROADMAP_NEXTSTEP_V4_PARTIAL_FORCE_ENABLED_CATEGORIES=["haircare"],
            ROADMAP_NEXTSTEP_V4_PARTIAL_HAIRCARE_PRODUCT_TYPES=["shampoo"],
            ROADMAP_NEXTSTEP_V4_PARTIAL_HAIRCARE_STEP_INDEXES=["1"],
            ROADMAP_NEXTSTEP_V4_PARTIAL_PERCENT=100,
            ROADMAP_NEXTSTEP_V4_PARTIAL_SALT="partial_hair_force_enable_test",
            ROADMAP_NEXTSTEP_V4_PARTIAL_HAIRCARE_MODEL_PATH="C:/tmp/roadmap_nextstep_semantic_v4.pkl",
            ROADMAP_NEXTSTEP_V4_SHADOW_MODEL_PATH="",
            ROADMAP_NEXTSTEP_V4_CONFIDENCE_THRESHOLD=0.1,
        ), patch(
            "roadmap_app.services.v4_min_lift_guard_status",
            return_value={"passed": True, "reason": "ok"},
        ), patch(
            "roadmap_app.services.v4_category_staged_rollout_status",
            return_value={
                "passed": True,
                "final_status": "ENABLE",
                "current_decision": "ENABLE",
                "reason": "passed",
                "hold_reason": None,
                "category": "haircare",
                "recommendation_7d": "ENABLE",
                "recommendation_30d": "ENABLE",
                "stability_gate_failures": [],
                "guard_7d": {"passed": True, "reason": "passed"},
                "guard_30d": {"passed": True, "reason": "passed"},
            },
        ), patch(
            "roadmap_app.services.predict_next_product_types",
        ) as active_mock, patch(
            "roadmap_app.services.predict_next_product_types_for_model_path",
            return_value=[{"candidate_type": "shampoo", "score": 0.96}],
        ) as candidate_mock, patch(
            "roadmap_app.services.nextstep_model_artifact_summary",
            side_effect=_artifact_summary,
        ):
            plan = refresh_roadmap(user, category="haircare", post_ctx=None)

        ml_meta = (plan.meta or {}).get("ml") if isinstance(plan.meta, dict) else {}
        self.assertEqual(str(ml_meta.get("decision")), "model_used")
        self.assertEqual(str(ml_meta.get("rollout_mode")), "partial")
        self.assertTrue(bool(ml_meta.get("rollout_selected")))
        self.assertEqual(str(ml_meta.get("model_slot")), "partial_candidate")
        self.assertEqual(str(ml_meta.get("model_path")), "C:/tmp/roadmap_nextstep_semantic_v4.pkl")
        self.assertEqual(str(ml_meta.get("model_version")), "semantic_v4_canary")
        self.assertEqual(str(ml_meta.get("planned_target_product_type")), "shampoo")
        self.assertEqual(int(ml_meta.get("planned_target_step_index")), 1)
        active_mock.assert_not_called()
        candidate_mock.assert_called_once()

    def test_partial_rollout_can_override_to_active_model_for_target_product_type(self):
        User = get_user_model()
        user = User.objects.create_user(username="v4_partial_hair_override_u1", password="pass12345")
        self._create_haircare_products("override")

        def _artifact_summary(path: str | None):
            raw = str(path or "")
            if raw == "C:/tmp/roadmap_nextstep_semantic_v4.pkl":
                return {
                    "model_version": "semantic_v4_canary",
                    "selected_feature_set": "full",
                }
            return {
                "model_version": "active_v4",
                "selected_feature_set": "baseline_only",
            }

        with override_settings(
            ROADMAP_NEXTSTEP_V4_ENABLED=True,
            ROADMAP_NEXTSTEP_V3_ENABLED=False,
            ROADMAP_PLANNER_V1_MODE="off",
            ROADMAP_NEXTSTEP_V4_ENABLED_CATEGORIES=["skincare", "haircare", "makeup"],
            ROADMAP_NEXTSTEP_V4_DISABLED_CATEGORIES=["fragrance"],
            ROADMAP_NEXTSTEP_V4_PARTIAL_ENABLED_CATEGORIES=["haircare"],
            ROADMAP_NEXTSTEP_V4_PARTIAL_PURCHASE_CONTEXT_ONLY_CATEGORIES=[],
            ROADMAP_NEXTSTEP_V4_PARTIAL_HAIRCARE_PRODUCT_TYPES=["shampoo"],
            ROADMAP_NEXTSTEP_V4_PARTIAL_HAIRCARE_STEP_INDEXES=["1"],
            ROADMAP_NEXTSTEP_V4_PARTIAL_HAIRCARE_ACTIVE_MODEL_PRODUCT_TYPES=["shampoo"],
            ROADMAP_NEXTSTEP_V4_PARTIAL_PERCENT=100,
            ROADMAP_NEXTSTEP_V4_PARTIAL_SALT="partial_hair_override_test",
            ROADMAP_NEXTSTEP_V4_PARTIAL_HAIRCARE_MODEL_PATH="C:/tmp/roadmap_nextstep_semantic_v4.pkl",
            ROADMAP_NEXTSTEP_V4_SHADOW_MODEL_PATH="",
            ROADMAP_NEXTSTEP_V4_CONFIDENCE_THRESHOLD=0.1,
        ), patch(
            "roadmap_app.services.v4_min_lift_guard_status",
            return_value={"passed": True, "reason": "ok"},
        ), patch(
            "roadmap_app.services.v4_category_staged_rollout_status",
            side_effect=lambda cat: self._hold_status(cat),
        ), patch(
            "roadmap_app.services.predict_next_product_types",
            return_value=[{"candidate_type": "shampoo", "score": 0.94}],
        ) as active_mock, patch(
            "roadmap_app.services.predict_next_product_types_for_model_path",
        ) as candidate_mock, patch(
            "roadmap_app.services.nextstep_model_artifact_summary",
            side_effect=_artifact_summary,
        ):
            plan = refresh_roadmap(user, category="haircare", post_ctx=None)

        ml_meta = (plan.meta or {}).get("ml") if isinstance(plan.meta, dict) else {}
        self.assertEqual(str(ml_meta.get("decision")), "model_used")
        self.assertEqual(str(ml_meta.get("rollout_mode")), "partial")
        self.assertEqual(str(ml_meta.get("model_slot")), "partial_active_override")
        self.assertEqual(str(ml_meta.get("model_version")), "active_v4")
        self.assertEqual(str(ml_meta.get("partial_model_override_reason")), "active_model_product_type")
        self.assertEqual(str(ml_meta.get("planned_target_product_type")), "shampoo")
        self.assertEqual(int(ml_meta.get("planned_target_step_index")), 1)
        active_mock.assert_called_once()
        candidate_mock.assert_not_called()

    def test_non_makeup_category_is_unaffected_by_partial_makeup_rollout(self):
        User = get_user_model()
        user = User.objects.create_user(username="v4_partial_sk_u1", password="pass12345")
        self._create_skincare_products("no-touch")

        with override_settings(
            ROADMAP_NEXTSTEP_V4_ENABLED=True,
            ROADMAP_NEXTSTEP_V3_ENABLED=False,
            ROADMAP_PLANNER_V1_MODE="off",
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

    def test_shadow_model_predictions_are_logged_without_affecting_primary_decision(self):
        User = get_user_model()
        user = User.objects.create_user(username="v4_shadow_u1", password="pass12345")
        self._create_skincare_products("shadow")

        def _artifact_summary(path: str | None):
            raw = str(path or "")
            if raw == "C:/tmp/roadmap_nextstep_shadow.pkl":
                return {
                    "model_version": "shadow_semantic_v2",
                    "selected_feature_set": "full",
                }
            return {
                "model_version": "active_v4",
                "selected_feature_set": "full",
            }

        with override_settings(
            ROADMAP_NEXTSTEP_V4_ENABLED=True,
            ROADMAP_NEXTSTEP_V3_ENABLED=False,
            ROADMAP_PLANNER_V1_MODE="off",
            ROADMAP_NEXTSTEP_V4_ENABLED_CATEGORIES=["skincare", "haircare", "makeup"],
            ROADMAP_NEXTSTEP_V4_DISABLED_CATEGORIES=["fragrance"],
            ROADMAP_NEXTSTEP_V4_CONFIDENCE_THRESHOLD=0.1,
            ROADMAP_NEXTSTEP_V4_SHADOW_MODEL_PATH="C:/tmp/roadmap_nextstep_shadow.pkl",
        ), patch(
            "roadmap_app.services.v4_min_lift_guard_status",
            return_value={"passed": True, "reason": "ok"},
        ), patch(
            "roadmap_app.services.v4_category_staged_rollout_status",
            return_value={
                "passed": True,
                "final_status": "ENABLE",
                "current_decision": "ENABLE",
                "reason": "passed",
                "hold_reason": None,
                "category": "skincare",
                "recommendation_7d": "ENABLE",
                "recommendation_30d": "ENABLE",
                "stability_gate_failures": [],
                "guard_7d": {"passed": True, "reason": "passed"},
                "guard_30d": {"passed": True, "reason": "passed"},
            },
        ), patch(
            "roadmap_app.services.predict_next_product_types",
            return_value=[{"candidate_type": "serum", "score": 0.91}],
        ) as primary_mock, patch(
            "roadmap_app.services.predict_next_product_types_for_model_path",
            return_value=[{"candidate_type": "cleanser", "score": 0.77}],
        ) as shadow_mock, patch(
            "roadmap_app.services.nextstep_model_artifact_summary",
            side_effect=_artifact_summary,
        ):
            plan = refresh_roadmap(user, category="skincare", post_ctx=None)

        ml_meta = (plan.meta or {}).get("ml") if isinstance(plan.meta, dict) else {}
        shadow_meta = ml_meta.get("shadow") or {}
        self.assertEqual(str(ml_meta.get("decision")), "model_used")
        self.assertTrue(bool(shadow_meta.get("enabled")))
        self.assertEqual(str(shadow_meta.get("reason")), "ok")
        self.assertEqual(str(shadow_meta.get("model_version")), "shadow_semantic_v2")
        self.assertEqual(str(shadow_meta.get("selected_feature_set")), "full")
        self.assertEqual(len(shadow_meta.get("predictions") or []), 1)
        primary_mock.assert_called_once()
        shadow_mock.assert_called_once()


class RoadmapNextStepV4HaircareLeaveInRerankTests(TestCase):
    def setUp(self):
        if pd is None:
            self.skipTest("pandas is required")

        User = get_user_model()
        self.user = User.objects.create_user(username="v4_haircare_leavein_u1", password="pass12345")
        profile = CustomerProfile.objects.get(user=self.user)
        profile.skin_type = "normal"
        profile.goals = ["definition", "frizz_control"]
        profile.avoid_flags = ["heavy_oils"]
        profile.hair_profile = {
            "hair_type": "curly",
            "scalp_type": "normal",
            "hair_thickness": "thick",
            "concerns": ["frizz", "definition", "dryness"],
        }
        profile.makeup_profile = {}
        profile.fragrance_profile = {}
        profile.save(
            update_fields=[
                "skin_type",
                "goals",
                "avoid_flags",
                "hair_profile",
                "makeup_profile",
                "fragrance_profile",
            ]
        )

        self.p_shampoo = Product.objects.create(
            name="LeaveIn Shampoo",
            brand="B",
            price=Decimal("12.00"),
            category="haircare",
            product_type="shampoo",
            concerns=["dryness", "frizz"],
            actives=["glycerin"],
            attrs={"hair_type": "curly", "scalp_type": "normal", "hair_thickness": "thick"},
            ingredients_inci="water, glycerin, aloe vera",
            in_stock=True,
        )
        self.p_conditioner = Product.objects.create(
            name="LeaveIn Conditioner",
            brand="B",
            price=Decimal("13.00"),
            category="haircare",
            product_type="conditioner",
            concerns=["dryness", "frizz", "detangling"],
            actives=["glycerin", "aloe"],
            attrs={"hair_type": "curly", "scalp_type": "normal", "hair_thickness": "thick"},
            ingredients_inci="water, glycerin, aloe vera, shea butter",
            in_stock=True,
        )
        self.p_mask = Product.objects.create(
            name="LeaveIn Mask",
            brand="B",
            price=Decimal("17.00"),
            category="haircare",
            product_type="hair_mask",
            concerns=["dryness", "frizz", "definition"],
            actives=["glycerin", "shea_butter"],
            attrs={"hair_type": "curly", "scalp_type": "normal", "hair_thickness": "thick"},
            ingredients_inci="water, glycerin, shea butter, coconut oil",
            in_stock=True,
        )
        self.p_leavein = Product.objects.create(
            name="LeaveIn Curl Cream",
            brand="B",
            price=Decimal("16.00"),
            category="haircare",
            product_type="leave_in",
            concerns=["frizz", "definition", "dryness"],
            actives=["glycerin", "aloe", "linseed"],
            attrs={"hair_type": "curly", "scalp_type": "normal", "hair_thickness": "thick"},
            ingredients_inci="water, glycerin, aloe vera, linseed extract",
            in_stock=True,
        )
        self.p_hair_oil = Product.objects.create(
            name="LeaveIn Hair Oil",
            brand="B",
            price=Decimal("17.00"),
            category="haircare",
            product_type="hair_oil",
            concerns=["shine", "seal", "frizz"],
            actives=["argan_oil", "jojoba_oil"],
            attrs={"hair_type": "curly", "scalp_type": "normal", "hair_thickness": "thick"},
            ingredients_inci="cyclopentasiloxane, jojoba oil, argan oil",
            in_stock=True,
        )

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

    def test_runtime_leavein_rerank_promotes_leave_in_for_leavein_target(self):
        t0 = timezone.now() - timedelta(days=10)
        self._create_tx(self.p_shampoo, t0 - timedelta(days=20), "v4-leavein-rerank-1")
        self._create_tx(self.p_conditioner, t0 - timedelta(days=12), "v4-leavein-rerank-2")
        self._create_tx(self.p_mask, t0 - timedelta(days=4), "v4-leavein-rerank-3")

        class DummyArtifactModel:
            def predict(self, X):
                base = pd.Series([0.0] * len(X), index=X.index, dtype=float)
                base = base + X["candidate_type"].astype(str).map(
                    {"hair_oil": 1.0, "leave_in": 0.1, "conditioner": 0.3}
                ).fillna(0.0)
                return base.to_numpy()

        artifact = {
            "task": "roadmap_nextstep_v4_ranking",
            "model": DummyArtifactModel(),
            "preprocessor": None,
            "model_type": "lightgbm_ranker",
            "feature_columns": [
                "category",
                "candidate_type",
                "planned_target_product_type",
            ],
            "categorical_features": [
                "category",
                "candidate_type",
                "planned_target_product_type",
            ],
            "numeric_features": [],
            "candidate_types_by_category": {"haircare": ["hair_oil", "leave_in", "conditioner"]},
            "rules_chain_by_category": {
                "haircare": ["shampoo", "conditioner", "hair_mask", "hair_oil", "scalp_serum", "leave_in"]
            },
            "candidate_popularity_in_train_by_category": {
                "haircare": {"hair_oil": 0.5, "leave_in": 0.2, "conditioner": 0.3}
            },
            "owned_feature_columns": [],
            "owned_feature_map": {},
            "temperature": 1.0,
        }

        with override_settings(ROADMAP_NEXTSTEP_V4_HAIRCARE_LEAVEIN_RERANK_ENABLED=True):
            with patch("roadmap_app.ml_next_step._load_model", return_value=artifact):
                rows = predict_next_product_types(
                    user=self.user,
                    context_product_ids=[],
                    category="haircare",
                    planned_target_product_type="leave_in",
                    planned_target_step_index=4,
                    candidate_types=["hair_oil", "leave_in", "conditioner"],
                )

        self.assertEqual(rows[0]["candidate_type"], "leave_in")
        self.assertIn("haircare_leavein_rerank", rows[0]["runtime_policies"])
        self.assertGreater(float(rows[0]["runtime_policy_biases"]["haircare_leavein_rerank"]), 0.0)

    def test_runtime_leavein_rerank_does_not_override_hair_oil_target(self):
        t0 = timezone.now() - timedelta(days=10)
        self._create_tx(self.p_shampoo, t0 - timedelta(days=20), "v4-leavein-rerank-4")
        self._create_tx(self.p_conditioner, t0 - timedelta(days=12), "v4-leavein-rerank-5")
        self._create_tx(self.p_mask, t0 - timedelta(days=4), "v4-leavein-rerank-6")

        class DummyArtifactModel:
            def predict(self, X):
                base = pd.Series([0.0] * len(X), index=X.index, dtype=float)
                base = base + X["candidate_type"].astype(str).map(
                    {"hair_oil": 1.0, "leave_in": 0.1, "conditioner": 0.3}
                ).fillna(0.0)
                return base.to_numpy()

        artifact = {
            "task": "roadmap_nextstep_v4_ranking",
            "model": DummyArtifactModel(),
            "preprocessor": None,
            "model_type": "lightgbm_ranker",
            "feature_columns": [
                "category",
                "candidate_type",
                "planned_target_product_type",
            ],
            "categorical_features": [
                "category",
                "candidate_type",
                "planned_target_product_type",
            ],
            "numeric_features": [],
            "candidate_types_by_category": {"haircare": ["hair_oil", "leave_in", "conditioner"]},
            "rules_chain_by_category": {
                "haircare": ["shampoo", "conditioner", "hair_mask", "hair_oil", "scalp_serum", "leave_in"]
            },
            "candidate_popularity_in_train_by_category": {
                "haircare": {"hair_oil": 0.5, "leave_in": 0.2, "conditioner": 0.3}
            },
            "owned_feature_columns": [],
            "owned_feature_map": {},
            "temperature": 1.0,
        }

        with override_settings(ROADMAP_NEXTSTEP_V4_HAIRCARE_LEAVEIN_RERANK_ENABLED=True):
            with patch("roadmap_app.ml_next_step._load_model", return_value=artifact):
                rows = predict_next_product_types(
                    user=self.user,
                    context_product_ids=[],
                    category="haircare",
                    planned_target_product_type="hair_oil",
                    planned_target_step_index=4,
                    candidate_types=["hair_oil", "leave_in", "conditioner"],
                )

        self.assertEqual(rows[0]["candidate_type"], "hair_oil")


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
                            "candidate_primary_target_in_14d": int(candidate == label),
                            "window_14d_fully_observed": 1,
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
            "label_protocol_version": "v5_exposure_funnel",
            "outcome_windows_days": [7, 14, 30],
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

    def test_training_supports_alternate_target_column(self):
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
                    target_column="candidate_primary_target_in_14d",
                )

            metadata = json.loads((model_dir / "metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(str(metadata.get("target_column")), "candidate_primary_target_in_14d")
            self.assertEqual(str(metadata.get("label_protocol_version")), "v5_exposure_funnel")
            self.assertEqual(list(metadata.get("outcome_windows_days") or []), [7, 14, 30])
            baseline_comparison = metadata.get("baseline_comparison") or {}
            runtime_guard = metadata.get("runtime_guard") or {}
            self.assertFalse(bool(baseline_comparison.get("applicable", True)))
            self.assertIn("legacy target 'y'", str(baseline_comparison.get("reason") or ""))
            self.assertFalse(bool(runtime_guard.get("applicable", True)))
            self.assertEqual(str(runtime_guard.get("status")), "experimental_target")

            report = json.loads((model_dir / "eval_report.json").read_text(encoding="utf-8"))
            self.assertEqual(str(report.get("target_column")), "candidate_primary_target_in_14d")
            self.assertEqual(str(report.get("label_protocol_version")), "v5_exposure_funnel")
            target_report_path = (
                Path(__file__).resolve().parents[2]
                / "reports"
                / "roadmap_nextstep_v4_eval__candidate_primary_target_in_14d.json"
            )
            self.assertTrue(target_report_path.exists())
