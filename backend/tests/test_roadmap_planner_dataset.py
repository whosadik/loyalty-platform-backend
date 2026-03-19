import json
from datetime import timedelta
from decimal import Decimal
from pathlib import Path
from tempfile import TemporaryDirectory

from django.contrib.auth import get_user_model
from django.core.management import call_command
from django.test import TestCase
from django.utils import timezone

from catalog.models import Product
from roadmap_app.models import RoadmapEvent, RoadmapPlan, RoadmapStep
from transactions.models import Transaction, TransactionItem
from users_app.models import CustomerProfile

try:
    import pandas as pd
except Exception:  # pragma: no cover
    pd = None


def _read_dataset(out_dir: Path) -> "pd.DataFrame":
    parquet_path = out_dir / "dataset.parquet"
    if parquet_path.exists():
        return pd.read_parquet(parquet_path)
    return pd.read_csv(out_dir / "dataset.csv")


class RoadmapPlannerDatasetTests(TestCase):
    def setUp(self):
        if pd is None:
            self.skipTest("pandas is required")

        User = get_user_model()
        self.user = User.objects.create_user(username="planner_u1", password="pass12345")
        self.plan = RoadmapPlan.objects.create(
            user=self.user,
            category="makeup",
            is_active=True,
            meta={"ml": {"decision": "fallback", "rollout_mode": "none"}},
        )
        self.step1 = RoadmapStep.objects.create(
            plan=self.plan,
            step_index=1,
            product_type="foundation",
            status=RoadmapStep.Status.RECOMMENDED,
        )
        self.step2 = RoadmapStep.objects.create(
            plan=self.plan,
            step_index=2,
            product_type="mascara",
            status=RoadmapStep.Status.MISSING,
        )
        self.step3 = RoadmapStep.objects.create(
            plan=self.plan,
            step_index=3,
            product_type="blush",
            status=RoadmapStep.Status.MISSING,
        )

        self.p_foundation = Product.objects.create(
            name="Planner Foundation",
            brand="B",
            price=Decimal("10.00"),
            category="makeup",
            product_type="foundation",
            in_stock=True,
        )
        self.p_mascara = Product.objects.create(
            name="Planner Mascara",
            brand="B",
            price=Decimal("11.00"),
            category="makeup",
            product_type="mascara",
            in_stock=True,
        )
        self.p_blush = Product.objects.create(
            name="Planner Blush",
            brand="B",
            price=Decimal("12.00"),
            category="makeup",
            product_type="blush",
            in_stock=True,
        )
        self.p_primer = Product.objects.create(
            name="Planner Primer",
            brand="B",
            price=Decimal("13.00"),
            category="makeup",
            product_type="primer",
            in_stock=True,
        )

    def _event(self, *, event_type: str, created_at, step=None, context=None):
        event = RoadmapEvent.objects.create(
            user=self.user,
            plan=self.plan,
            step=step,
            event_type=event_type,
            context=context or {},
        )
        RoadmapEvent.objects.filter(id=event.id).update(created_at=created_at)
        event.refresh_from_db()
        return event

    def _tx(self, *, product: Product, created_at, idem_key: str):
        tx = Transaction.objects.create(
            user=self.user,
            total_amount=Decimal("10.00"),
            channel="web",
            idempotency_key=idem_key,
        )
        Transaction.objects.filter(id=tx.id).update(created_at=created_at)
        TransactionItem.objects.create(
            transaction=tx,
            product=product,
            quantity=1,
            unit_price=Decimal("10.00"),
        )

    def test_planner_dataset_uses_completed_step_label_and_prior_history(self):
        t0 = timezone.now() - timedelta(days=30)
        self._tx(product=self.p_primer, created_at=t0 - timedelta(days=1), idem_key="planner-prior")
        self._tx(product=self.p_blush, created_at=t0 + timedelta(days=1), idem_key="planner-future")

        self._event(
            event_type=RoadmapEvent.Type.PLAN_REFRESHED,
            created_at=t0,
            context={
                "category": "makeup",
                "next_step_id": self.step1.id,
                "next_step_index": 1,
                "next_product_type": "foundation",
                "refresh_caller": "refresh_roadmap",
                "ml": {"decision": "fallback", "rollout_mode": "none"},
            },
        )
        self._event(
            event_type=RoadmapEvent.Type.STEP_GENERATED,
            created_at=t0 + timedelta(seconds=1),
            step=self.step1,
            context={
                "category": "makeup",
                "step_id": self.step1.id,
                "step_index": 1,
                "product_type": "foundation",
                "status": "recommended",
                "recommended_product_id": self.p_foundation.id,
                "has_recommendation": True,
            },
        )
        self._event(
            event_type=RoadmapEvent.Type.STEP_GENERATED,
            created_at=t0 + timedelta(seconds=2),
            step=self.step2,
            context={
                "category": "makeup",
                "step_id": self.step2.id,
                "step_index": 2,
                "product_type": "mascara",
                "status": "missing",
                "has_recommendation": False,
            },
        )
        self._event(
            event_type=RoadmapEvent.Type.STEP_COMPLETED,
            created_at=t0 + timedelta(days=1),
            step=self.step1,
            context={
                "category": "makeup",
                "product_type": "foundation",
                "matched_by": "recommended_product_id",
            },
        )

        with TemporaryDirectory() as tmp_dir:
            out_dir = Path(tmp_dir)
            call_command(
                "build_roadmap_planner_dataset",
                days=180,
                out_dir=str(out_dir),
                label_window_days=7,
            )
            frame = _read_dataset(out_dir)
            self.assertFalse(frame.empty)

            episode = frame[frame["episode_id"] == 1].sort_values("candidate_type").reset_index(drop=True)
            self.assertEqual(set(episode["label"].astype(str)), {"foundation"})
            self.assertIn("__stop__", set(episode["candidate_type"].astype(str)))

            foundation = episode[episode["candidate_type"].astype(str) == "foundation"].iloc[0]
            self.assertEqual(int(foundation["y"]), 1)
            self.assertEqual(int(foundation["candidate_position_in_generated_plan"]), 1)
            self.assertEqual(int(foundation["candidate_is_current_next_step"]), 1)
            self.assertEqual(str(foundation["last1_product_type"]), "primer")

            metadata = json.loads((out_dir / "metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(int(metadata["episodes_total"]), 1)
            self.assertEqual(int(metadata["stop_label_count"]), 0)
            self.assertNotIn("matched_by", list(metadata.get("feature_columns") or []))
            self.assertEqual(
                int((metadata.get("label_source_distribution") or {}).get("roadmap_completed_exact", 0)),
                1,
            )

    def test_planner_dataset_ends_episode_at_next_refresh(self):
        t0 = timezone.now() - timedelta(days=40)
        t1 = t0 + timedelta(days=1)

        self._event(
            event_type=RoadmapEvent.Type.PLAN_REFRESHED,
            created_at=t0,
            context={
                "category": "makeup",
                "next_step_id": self.step1.id,
                "next_step_index": 1,
                "next_product_type": "foundation",
                "refresh_caller": "refresh_roadmap",
                "ml": {"decision": "fallback", "rollout_mode": "none"},
            },
        )
        self._event(
            event_type=RoadmapEvent.Type.STEP_GENERATED,
            created_at=t0 + timedelta(seconds=1),
            step=self.step1,
            context={
                "category": "makeup",
                "step_id": self.step1.id,
                "step_index": 1,
                "product_type": "foundation",
                "status": "recommended",
                "recommended_product_id": self.p_foundation.id,
                "has_recommendation": True,
            },
        )
        self._event(
            event_type=RoadmapEvent.Type.STEP_GENERATED,
            created_at=t0 + timedelta(seconds=2),
            step=self.step2,
            context={
                "category": "makeup",
                "step_id": self.step2.id,
                "step_index": 2,
                "product_type": "mascara",
                "status": "missing",
                "has_recommendation": False,
            },
        )

        self._event(
            event_type=RoadmapEvent.Type.PLAN_REFRESHED,
            created_at=t1,
            context={
                "category": "makeup",
                "next_step_id": self.step2.id,
                "next_step_index": 2,
                "next_product_type": "mascara",
                "refresh_caller": "update_roadmap_from_purchase",
                "ml": {"decision": "fallback", "rollout_mode": "none"},
            },
        )
        self._event(
            event_type=RoadmapEvent.Type.STEP_GENERATED,
            created_at=t1 + timedelta(seconds=1),
            step=self.step1,
            context={
                "category": "makeup",
                "step_id": self.step1.id,
                "step_index": 1,
                "product_type": "foundation",
                "status": "completed",
                "has_recommendation": False,
            },
        )
        self._event(
            event_type=RoadmapEvent.Type.STEP_GENERATED,
            created_at=t1 + timedelta(seconds=2),
            step=self.step2,
            context={
                "category": "makeup",
                "step_id": self.step2.id,
                "step_index": 2,
                "product_type": "mascara",
                "status": "recommended",
                "recommended_product_id": self.p_mascara.id,
                "has_recommendation": True,
            },
        )
        self._event(
            event_type=RoadmapEvent.Type.STEP_COMPLETED,
            created_at=t1 + timedelta(days=1),
            step=self.step2,
            context={
                "category": "makeup",
                "product_type": "mascara",
                "matched_by": "recommended_product_id",
            },
        )

        with TemporaryDirectory() as tmp_dir:
            out_dir = Path(tmp_dir)
            call_command(
                "build_roadmap_planner_dataset",
                days=180,
                out_dir=str(out_dir),
                label_window_days=7,
            )
            frame = _read_dataset(out_dir)
            self.assertFalse(frame.empty)

            by_episode = (
                frame.groupby("episode_id")["label"].first().sort_index().astype(str).to_dict()
            )
            self.assertEqual(by_episode[1], "__stop__")
            self.assertEqual(by_episode[2], "mascara")

            episode1_stop = frame[
                (frame["episode_id"] == 1) & (frame["candidate_type"].astype(str) == "__stop__")
            ].iloc[0]
            self.assertEqual(int(episode1_stop["y"]), 1)

    def test_planner_dataset_includes_content_aware_features(self):
        profile = CustomerProfile.objects.get(user=self.user)
        profile.makeup_profile = {
            "finish_pref": ["dewy"],
            "coverage_pref": ["medium"],
            "undertone": "neutral",
            "tone_family": "light",
            "concerns": ["long_wear"],
        }
        profile.save(update_fields=["makeup_profile"])

        self.p_primer.attrs = {"finish": "dewy"}
        self.p_primer.ingredients_inci = "dimethicone, silica"
        self.p_primer.save(update_fields=["attrs", "ingredients_inci"])
        self.p_foundation.attrs = {
            "finish": "dewy",
            "coverage": "medium",
            "undertone": "neutral",
            "tone_family": "light",
        }
        self.p_foundation.concerns = ["long_wear"]
        self.p_foundation.ingredients_inci = "water, dimethicone, iron_oxides"
        self.p_foundation.save(
            update_fields=["attrs", "concerns", "ingredients_inci"]
        )

        t0 = timezone.now() - timedelta(days=25)
        self._tx(product=self.p_primer, created_at=t0 - timedelta(days=1), idem_key="planner-content-prior")
        self._event(
            event_type=RoadmapEvent.Type.PLAN_REFRESHED,
            created_at=t0,
            context={
                "category": "makeup",
                "next_step_id": self.step1.id,
                "next_step_index": 1,
                "next_product_type": "foundation",
                "refresh_caller": "refresh_roadmap",
                "ml": {"decision": "fallback", "rollout_mode": "none"},
            },
        )
        self._event(
            event_type=RoadmapEvent.Type.STEP_GENERATED,
            created_at=t0 + timedelta(seconds=1),
            step=self.step1,
            context={
                "category": "makeup",
                "step_id": self.step1.id,
                "step_index": 1,
                "product_type": "foundation",
                "status": "recommended",
                "recommended_product_id": self.p_foundation.id,
                "has_recommendation": True,
            },
        )
        self._event(
            event_type=RoadmapEvent.Type.STEP_COMPLETED,
            created_at=t0 + timedelta(days=1),
            step=self.step1,
            context={
                "category": "makeup",
                "product_type": "foundation",
                "matched_by": "recommended_product_id",
            },
        )

        with TemporaryDirectory() as tmp_dir:
            out_dir = Path(tmp_dir)
            call_command(
                "build_roadmap_planner_dataset",
                days=180,
                out_dir=str(out_dir),
                label_window_days=7,
            )
            frame = _read_dataset(out_dir)
            foundation = frame[frame["candidate_type"].astype(str) == "foundation"].iloc[0]

            self.assertIn("profile_makeup_finish_pref_primary", frame.columns)
            self.assertIn("candidate_profile_makeup_finish_match_rate", frame.columns)
            self.assertEqual(str(foundation["profile_makeup_finish_pref_primary"]), "dewy")
            self.assertEqual(str(foundation["anchor_product_type"]), "primer")
            self.assertGreater(float(foundation["candidate_profile_makeup_finish_match_rate"]), 0.0)
            self.assertGreater(float(foundation["candidate_anchor_shared_inci_rate"]), 0.0)


class RoadmapPlannerDatasetBlock2Tests(TestCase):
    def setUp(self):
        if pd is None:
            self.skipTest("pandas is required")

        User = get_user_model()
        self.user = User.objects.create_user(username="planner_block2_u1", password="pass12345")
        CustomerProfile.objects.get_or_create(user=self.user)

        self.fragrance_plan = RoadmapPlan.objects.create(
            user=self.user,
            category="fragrance",
            is_active=True,
            meta={},
        )
        self.fragrance_step = RoadmapStep.objects.create(
            plan=self.fragrance_plan,
            step_index=1,
            product_type="warm_evening",
            status=RoadmapStep.Status.RECOMMENDED,
        )
        self.fragrance_followup = RoadmapStep.objects.create(
            plan=self.fragrance_plan,
            step_index=2,
            product_type="cold_evening",
            status=RoadmapStep.Status.MISSING,
        )

        self.p_warm_day = Product.objects.create(
            name="Planner Warm Day",
            brand="F",
            price=Decimal("40.00"),
            category="fragrance",
            product_type="edp",
            attrs={"scent_family": "citrus", "notes": ["bergamot"], "intensity": "soft"},
            in_stock=True,
        )
        self.p_warm_evening = Product.objects.create(
            name="Planner Warm Evening",
            brand="F",
            price=Decimal("45.00"),
            category="fragrance",
            product_type="edp",
            attrs={"scent_family": "citrus", "notes": ["neroli"], "intensity": "strong"},
            in_stock=True,
        )

    def _event(self, *, event_type: str, created_at, step=None, context=None):
        event = RoadmapEvent.objects.create(
            user=self.user,
            plan=self.fragrance_plan,
            step=step,
            event_type=event_type,
            context=context or {},
        )
        RoadmapEvent.objects.filter(id=event.id).update(created_at=created_at)
        event.refresh_from_db()
        return event

    def _tx(self, *, product: Product, created_at, idem_key: str):
        tx = Transaction.objects.create(
            user=self.user,
            total_amount=Decimal("10.00"),
            channel="web",
            idempotency_key=idem_key,
        )
        Transaction.objects.filter(id=tx.id).update(created_at=created_at)
        TransactionItem.objects.create(
            transaction=tx,
            product=product,
            quantity=1,
            unit_price=Decimal("10.00"),
        )

    def _build(self, *, out_dir: Path, days: int = 180, label_window_days: int = 7):
        call_command(
            "build_roadmap_planner_dataset",
            days=days,
            out_dir=str(out_dir),
            label_window_days=label_window_days,
        )
        return _read_dataset(out_dir), json.loads((out_dir / "metadata.json").read_text(encoding="utf-8"))

    def test_planner_dataset_excludes_bad_fragrance_exact_completion(self):
        t0 = timezone.now() - timedelta(days=20)
        self._event(
            event_type=RoadmapEvent.Type.PLAN_REFRESHED,
            created_at=t0,
            context={
                "category": "fragrance",
                "next_step_id": self.fragrance_step.id,
                "next_step_index": 1,
                "next_product_type": "warm_evening",
            },
        )
        self._event(
            event_type=RoadmapEvent.Type.STEP_GENERATED,
            created_at=t0 + timedelta(seconds=1),
            step=self.fragrance_step,
            context={
                "category": "fragrance",
                "step_id": self.fragrance_step.id,
                "step_index": 1,
                "product_type": "warm_evening",
                "status": "recommended",
                "recommended_product_id": self.p_warm_day.id,
            },
        )
        self._event(
            event_type=RoadmapEvent.Type.STEP_GENERATED,
            created_at=t0 + timedelta(seconds=2),
            step=self.fragrance_followup,
            context={
                "category": "fragrance",
                "step_id": self.fragrance_followup.id,
                "step_index": 2,
                "product_type": "cold_evening",
                "status": "missing",
            },
        )
        self._event(
            event_type=RoadmapEvent.Type.STEP_COMPLETED,
            created_at=t0 + timedelta(days=1),
            step=self.fragrance_step,
            context={
                "category": "fragrance",
                "product_type": "warm_evening",
                "matched_by": "recommended_product_id",
                "recommended_product_id": self.p_warm_day.id,
                "purchased_product_id": self.p_warm_day.id,
            },
        )

        with TemporaryDirectory() as tmp_dir:
            frame, metadata = self._build(out_dir=Path(tmp_dir))
            episode = frame[frame["decision_id"] == 1]
            stop_row = episode[episode["candidate_type"].astype(str) == "__stop__"].iloc[0]
            self.assertEqual(int(stop_row["y"]), 1)
            self.assertEqual(str(stop_row["label_source"]), "stop_no_progress")
            self.assertEqual(int(metadata["excluded_legacy_bad_fragrance_completions_count"]), 1)

    def test_planner_dataset_has_one_positive_per_decision(self):
        t0 = timezone.now() - timedelta(days=18)
        self._event(
            event_type=RoadmapEvent.Type.PLAN_REFRESHED,
            created_at=t0,
            context={
                "category": "fragrance",
                "next_step_id": self.fragrance_step.id,
                "next_step_index": 1,
                "next_product_type": "warm_evening",
            },
        )
        self._event(
            event_type=RoadmapEvent.Type.STEP_GENERATED,
            created_at=t0 + timedelta(seconds=1),
            step=self.fragrance_step,
            context={
                "category": "fragrance",
                "step_id": self.fragrance_step.id,
                "step_index": 1,
                "product_type": "warm_evening",
                "status": "recommended",
                "recommended_product_id": self.p_warm_evening.id,
            },
        )
        self._event(
            event_type=RoadmapEvent.Type.STEP_COMPLETED,
            created_at=t0 + timedelta(days=1),
            step=self.fragrance_step,
            context={
                "category": "fragrance",
                "product_type": "warm_evening",
                "matched_by": "fragrance_slot",
                "purchased_product_id": self.p_warm_evening.id,
            },
        )

        with TemporaryDirectory() as tmp_dir:
            frame, _metadata = self._build(out_dir=Path(tmp_dir))
            positives = frame.groupby("decision_id")["y"].sum().astype(int).to_dict()
            self.assertEqual(positives, {1: 1})

    def test_planner_dataset_fragrance_candidates_use_slots_not_edp(self):
        t0 = timezone.now() - timedelta(days=16)
        self._event(
            event_type=RoadmapEvent.Type.PLAN_REFRESHED,
            created_at=t0,
            context={
                "category": "fragrance",
                "next_step_id": self.fragrance_step.id,
                "next_step_index": 1,
                "next_product_type": "warm_evening",
            },
        )
        self._event(
            event_type=RoadmapEvent.Type.STEP_GENERATED,
            created_at=t0 + timedelta(seconds=1),
            step=self.fragrance_step,
            context={
                "category": "fragrance",
                "step_id": self.fragrance_step.id,
                "step_index": 1,
                "product_type": "warm_evening",
                "status": "recommended",
                "recommended_product_id": self.p_warm_evening.id,
            },
        )
        self._event(
            event_type=RoadmapEvent.Type.STEP_COMPLETED,
            created_at=t0 + timedelta(days=1),
            step=self.fragrance_step,
            context={
                "category": "fragrance",
                "product_type": "warm_evening",
                "matched_by": "fragrance_slot",
                "purchased_product_id": self.p_warm_evening.id,
            },
        )

        with TemporaryDirectory() as tmp_dir:
            frame, metadata = self._build(out_dir=Path(tmp_dir))
            candidates = set(frame["candidate_type"].astype(str))
            self.assertIn("warm_evening", candidates)
            self.assertIn("__stop__", candidates)
            self.assertNotIn("edp", candidates)
            self.assertEqual(
                metadata["candidate_types_by_category"]["fragrance"],
                ["warm_day", "warm_evening", "cold_day", "cold_evening", "__stop__"],
            )

    def test_planner_dataset_output_is_reproducible(self):
        t0 = timezone.now() - timedelta(days=14)
        self._event(
            event_type=RoadmapEvent.Type.PLAN_REFRESHED,
            created_at=t0,
            context={
                "category": "fragrance",
                "next_step_id": self.fragrance_step.id,
                "next_step_index": 1,
                "next_product_type": "warm_evening",
            },
        )
        self._event(
            event_type=RoadmapEvent.Type.STEP_GENERATED,
            created_at=t0 + timedelta(seconds=1),
            step=self.fragrance_step,
            context={
                "category": "fragrance",
                "step_id": self.fragrance_step.id,
                "step_index": 1,
                "product_type": "warm_evening",
                "status": "recommended",
                "recommended_product_id": self.p_warm_evening.id,
            },
        )
        self._event(
            event_type=RoadmapEvent.Type.STEP_COMPLETED,
            created_at=t0 + timedelta(days=1),
            step=self.fragrance_step,
            context={
                "category": "fragrance",
                "product_type": "warm_evening",
                "matched_by": "fragrance_slot",
                "purchased_product_id": self.p_warm_evening.id,
            },
        )

        with TemporaryDirectory() as tmp_dir_1, TemporaryDirectory() as tmp_dir_2:
            frame_1, metadata_1 = self._build(out_dir=Path(tmp_dir_1))
            frame_2, metadata_2 = self._build(out_dir=Path(tmp_dir_2))
            pd.testing.assert_frame_equal(
                frame_1.sort_values(["decision_id", "candidate_type"]).reset_index(drop=True),
                frame_2.sort_values(["decision_id", "candidate_type"]).reset_index(drop=True),
                check_like=False,
            )
            self.assertEqual(metadata_1["rows_total"], metadata_2["rows_total"])
            self.assertEqual(
                metadata_1["label_source_distribution"],
                metadata_2["label_source_distribution"],
            )
