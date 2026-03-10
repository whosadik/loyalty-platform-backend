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
