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


class RoadmapPlannerTransitionsDatasetTests(TestCase):
    def setUp(self):
        if pd is None:
            self.skipTest("pandas is required")

        User = get_user_model()
        self.User = User

        self.p_foundation = Product.objects.create(
            name="Transitions Foundation",
            brand="B",
            price=Decimal("10.00"),
            category="makeup",
            product_type="foundation",
            in_stock=True,
        )
        self.p_mascara = Product.objects.create(
            name="Transitions Mascara",
            brand="B",
            price=Decimal("11.00"),
            category="makeup",
            product_type="mascara",
            in_stock=True,
        )
        self.p_blush = Product.objects.create(
            name="Transitions Blush",
            brand="B",
            price=Decimal("12.00"),
            category="makeup",
            product_type="blush",
            in_stock=True,
        )
        self.p_warm_day = Product.objects.create(
            name="Transitions Warm Day",
            brand="F",
            price=Decimal("20.00"),
            category="fragrance",
            product_type="edp",
            attrs={"scent_family": "citrus", "notes": ["bergamot"], "intensity": "soft"},
            in_stock=True,
        )
        self.p_warm_evening = Product.objects.create(
            name="Transitions Warm Evening",
            brand="F",
            price=Decimal("21.00"),
            category="fragrance",
            product_type="edp",
            attrs={"scent_family": "citrus", "notes": ["neroli"], "intensity": "strong"},
            in_stock=True,
        )
        self.p_cold_evening = Product.objects.create(
            name="Transitions Cold Evening",
            brand="F",
            price=Decimal("22.00"),
            category="fragrance",
            product_type="edp",
            attrs={"scent_family": "woody", "notes": ["oud"], "intensity": "strong"},
            in_stock=True,
        )

    def _user(self, username: str):
        user = self.User.objects.create_user(username=username, password="pass12345")
        CustomerProfile.objects.get_or_create(user=user)
        return user

    def _plan(self, *, user, category: str, meta=None):
        return RoadmapPlan.objects.create(
            user=user,
            category=category,
            is_active=True,
            meta=meta or {"ml": {"decision": "fallback", "rollout_mode": "none"}},
        )

    def _event(self, *, user, plan, event_type: str, created_at, step=None, context=None):
        event = RoadmapEvent.objects.create(
            user=user,
            plan=plan,
            step=step,
            event_type=event_type,
            context=context or {},
        )
        RoadmapEvent.objects.filter(id=event.id).update(created_at=created_at)
        event.refresh_from_db()
        return event

    def _tx(self, *, user, product: Product, created_at, idem_key: str):
        tx = Transaction.objects.create(
            user=user,
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

    def _build(self, *, out_dir: Path, days: int = 180, label_window_days: int = 7, mode: str = "combined"):
        call_command(
            "build_roadmap_planner_transitions_dataset",
            days=days,
            out_dir=str(out_dir),
            label_window_days=label_window_days,
            mode=mode,
        )
        frame = _read_dataset(out_dir)
        metadata = json.loads((out_dir / "metadata.json").read_text(encoding="utf-8"))
        return frame, metadata

    def test_no_future_leakage_for_continuation_decision(self):
        user = self._user("transitions_makeup_u1")
        plan = self._plan(user=user, category="makeup")
        step1 = RoadmapStep.objects.create(plan=plan, step_index=1, product_type="foundation", status=RoadmapStep.Status.RECOMMENDED)
        step2 = RoadmapStep.objects.create(plan=plan, step_index=2, product_type="mascara", status=RoadmapStep.Status.MISSING)
        t0 = timezone.now() - timedelta(days=20)

        self._tx(user=user, product=self.p_blush, created_at=t0 - timedelta(days=1), idem_key="prior-blush")
        self._event(
            user=user,
            plan=plan,
            event_type=RoadmapEvent.Type.PLAN_REFRESHED,
            created_at=t0,
            context={"category": "makeup", "next_step_id": step1.id, "next_step_index": 1, "next_product_type": "foundation"},
        )
        self._event(
            user=user,
            plan=plan,
            step=step1,
            event_type=RoadmapEvent.Type.STEP_GENERATED,
            created_at=t0 + timedelta(seconds=1),
            context={"category": "makeup", "step_id": step1.id, "step_index": 1, "product_type": "foundation", "status": "recommended", "recommended_product_id": self.p_foundation.id},
        )
        self._event(
            user=user,
            plan=plan,
            step=step2,
            event_type=RoadmapEvent.Type.STEP_GENERATED,
            created_at=t0 + timedelta(seconds=2),
            context={"category": "makeup", "step_id": step2.id, "step_index": 2, "product_type": "mascara", "status": "missing"},
        )
        completion_at = t0 + timedelta(days=1)
        self._tx(user=user, product=self.p_foundation, created_at=completion_at, idem_key="foundation-buy")
        self._event(
            user=user,
            plan=plan,
            step=step1,
            event_type=RoadmapEvent.Type.STEP_COMPLETED,
            created_at=completion_at + timedelta(seconds=1),
            context={"category": "makeup", "product_type": "foundation", "matched_by": "recommended_product_id", "recommended_product_id": self.p_foundation.id, "purchased_product_id": self.p_foundation.id},
        )
        later_buy = t0 + timedelta(days=2)
        self._tx(user=user, product=self.p_blush, created_at=later_buy, idem_key="future-blush")
        self._tx(user=user, product=self.p_mascara, created_at=later_buy + timedelta(hours=1), idem_key="future-mascara")
        self._event(
            user=user,
            plan=plan,
            step=step2,
            event_type=RoadmapEvent.Type.STEP_COMPLETED,
            created_at=later_buy + timedelta(hours=1, seconds=1),
            context={"category": "makeup", "product_type": "mascara", "matched_by": "product_type", "purchased_product_id": self.p_mascara.id},
        )

        with TemporaryDirectory() as tmp_dir:
            frame, _metadata = self._build(out_dir=Path(tmp_dir))
            continuation = frame[frame["decision_type"].astype(str) == "post_completed"]
            self.assertFalse(continuation.empty)
            mascara_row = continuation[continuation["candidate_type"].astype(str) == "mascara"].iloc[0]
            self.assertEqual(str(mascara_row["last1_product_type"]), "foundation")
            self.assertEqual(int(mascara_row["y"]), 1)

    def test_one_positive_per_decision_point(self):
        user = self._user("transitions_makeup_u2")
        plan = self._plan(user=user, category="makeup")
        step1 = RoadmapStep.objects.create(plan=plan, step_index=1, product_type="foundation", status=RoadmapStep.Status.RECOMMENDED)
        step2 = RoadmapStep.objects.create(plan=plan, step_index=2, product_type="mascara", status=RoadmapStep.Status.MISSING)
        t0 = timezone.now() - timedelta(days=22)

        self._event(user=user, plan=plan, event_type=RoadmapEvent.Type.PLAN_REFRESHED, created_at=t0, context={"category": "makeup", "next_step_id": step1.id, "next_step_index": 1, "next_product_type": "foundation"})
        self._event(user=user, plan=plan, step=step1, event_type=RoadmapEvent.Type.STEP_GENERATED, created_at=t0 + timedelta(seconds=1), context={"category": "makeup", "step_id": step1.id, "step_index": 1, "product_type": "foundation", "status": "recommended", "recommended_product_id": self.p_foundation.id})
        self._event(user=user, plan=plan, step=step2, event_type=RoadmapEvent.Type.STEP_GENERATED, created_at=t0 + timedelta(seconds=2), context={"category": "makeup", "step_id": step2.id, "step_index": 2, "product_type": "mascara", "status": "missing"})
        self._event(user=user, plan=plan, step=step1, event_type=RoadmapEvent.Type.STEP_SKIPPED, created_at=t0 + timedelta(days=1), context={"category": "makeup", "step_id": step1.id, "step_index": 1, "product_type": "foundation"})
        self._tx(user=user, product=self.p_mascara, created_at=t0 + timedelta(days=2), idem_key="mascara-pos")
        self._event(user=user, plan=plan, step=step2, event_type=RoadmapEvent.Type.STEP_COMPLETED, created_at=t0 + timedelta(days=2, seconds=1), context={"category": "makeup", "product_type": "mascara", "matched_by": "product_type", "purchased_product_id": self.p_mascara.id})

        with TemporaryDirectory() as tmp_dir:
            frame, _metadata = self._build(out_dir=Path(tmp_dir))
            positives = frame.groupby("decision_id")["y"].sum().astype(int).to_dict()
            self.assertTrue(all(value == 1 for value in positives.values()))

    def test_skipped_decision_semantics_are_deterministic(self):
        user = self._user("transitions_makeup_u3")
        plan = self._plan(user=user, category="makeup")
        step1 = RoadmapStep.objects.create(plan=plan, step_index=1, product_type="foundation", status=RoadmapStep.Status.RECOMMENDED)
        t0 = timezone.now() - timedelta(days=24)

        self._event(user=user, plan=plan, event_type=RoadmapEvent.Type.PLAN_REFRESHED, created_at=t0, context={"category": "makeup", "next_step_id": step1.id, "next_step_index": 1, "next_product_type": "foundation"})
        self._event(user=user, plan=plan, step=step1, event_type=RoadmapEvent.Type.STEP_GENERATED, created_at=t0 + timedelta(seconds=1), context={"category": "makeup", "step_id": step1.id, "step_index": 1, "product_type": "foundation", "status": "recommended", "recommended_product_id": self.p_foundation.id})
        self._event(user=user, plan=plan, step=step1, event_type=RoadmapEvent.Type.STEP_SKIPPED, created_at=t0 + timedelta(days=1), context={"category": "makeup", "step_id": step1.id, "step_index": 1, "product_type": "foundation"})

        with TemporaryDirectory() as tmp_dir:
            frame, _metadata = self._build(out_dir=Path(tmp_dir))
            post_skipped = frame[frame["decision_type"].astype(str) == "post_skipped"]
            stop_row = post_skipped[post_skipped["candidate_type"].astype(str) == "__stop__"].iloc[0]
            self.assertEqual(int(stop_row["y"]), 1)
            self.assertEqual(str(stop_row["label_source"]), "terminal_after_outcome_stop")

    def test_fragrance_continuation_respects_slot_filtering(self):
        user = self._user("transitions_fragrance_u1")
        plan = self._plan(user=user, category="fragrance")
        step1 = RoadmapStep.objects.create(plan=plan, step_index=1, product_type="warm_evening", status=RoadmapStep.Status.RECOMMENDED)
        step2 = RoadmapStep.objects.create(plan=plan, step_index=2, product_type="cold_evening", status=RoadmapStep.Status.MISSING)
        t0 = timezone.now() - timedelta(days=26)

        self._event(user=user, plan=plan, event_type=RoadmapEvent.Type.PLAN_REFRESHED, created_at=t0, context={"category": "fragrance", "next_step_id": step1.id, "next_step_index": 1, "next_product_type": "warm_evening"})
        self._event(user=user, plan=plan, step=step1, event_type=RoadmapEvent.Type.STEP_GENERATED, created_at=t0 + timedelta(seconds=1), context={"category": "fragrance", "step_id": step1.id, "step_index": 1, "product_type": "warm_evening", "status": "recommended", "recommended_product_id": self.p_warm_day.id})
        self._event(user=user, plan=plan, step=step2, event_type=RoadmapEvent.Type.STEP_GENERATED, created_at=t0 + timedelta(seconds=2), context={"category": "fragrance", "step_id": step2.id, "step_index": 2, "product_type": "cold_evening", "status": "missing"})
        self._event(user=user, plan=plan, step=step1, event_type=RoadmapEvent.Type.STEP_COMPLETED, created_at=t0 + timedelta(days=1), context={"category": "fragrance", "product_type": "warm_evening", "matched_by": "recommended_product_id", "recommended_product_id": self.p_warm_day.id, "purchased_product_id": self.p_warm_day.id})
        self._event(user=user, plan=plan, step=step1, event_type=RoadmapEvent.Type.STEP_SKIPPED, created_at=t0 + timedelta(days=2), context={"category": "fragrance", "step_id": step1.id, "step_index": 1, "product_type": "warm_evening"})
        self._tx(user=user, product=self.p_cold_evening, created_at=t0 + timedelta(days=3), idem_key="cold-evening-buy")
        self._event(user=user, plan=plan, step=step2, event_type=RoadmapEvent.Type.STEP_COMPLETED, created_at=t0 + timedelta(days=3, seconds=1), context={"category": "fragrance", "product_type": "cold_evening", "matched_by": "fragrance_slot", "purchased_product_id": self.p_cold_evening.id})

        with TemporaryDirectory() as tmp_dir:
            frame, metadata = self._build(out_dir=Path(tmp_dir))
            self.assertEqual(int(metadata["excluded_legacy_bad_fragrance_completions_count"]), 1)
            continuation = frame[
                (frame["decision_type"].astype(str) == "post_skipped")
                & (frame["candidate_type"].astype(str) == "cold_evening")
            ].iloc[0]
            self.assertEqual(int(continuation["y"]), 1)
            self.assertEqual(str(continuation["current_next_product_type"]), "cold_evening")

    def test_initial_and_continuation_decision_types_are_separated(self):
        user = self._user("transitions_makeup_u4")
        plan = self._plan(user=user, category="makeup")
        step1 = RoadmapStep.objects.create(plan=plan, step_index=1, product_type="foundation", status=RoadmapStep.Status.RECOMMENDED)
        step2 = RoadmapStep.objects.create(plan=plan, step_index=2, product_type="mascara", status=RoadmapStep.Status.MISSING)
        t0 = timezone.now() - timedelta(days=28)

        self._event(user=user, plan=plan, event_type=RoadmapEvent.Type.PLAN_REFRESHED, created_at=t0, context={"category": "makeup", "next_step_id": step1.id, "next_step_index": 1, "next_product_type": "foundation"})
        self._event(user=user, plan=plan, step=step1, event_type=RoadmapEvent.Type.STEP_GENERATED, created_at=t0 + timedelta(seconds=1), context={"category": "makeup", "step_id": step1.id, "step_index": 1, "product_type": "foundation", "status": "recommended"})
        self._event(user=user, plan=plan, step=step2, event_type=RoadmapEvent.Type.STEP_GENERATED, created_at=t0 + timedelta(seconds=2), context={"category": "makeup", "step_id": step2.id, "step_index": 2, "product_type": "mascara", "status": "missing"})
        self._event(user=user, plan=plan, step=step1, event_type=RoadmapEvent.Type.STEP_COMPLETED, created_at=t0 + timedelta(days=1), context={"category": "makeup", "product_type": "foundation", "matched_by": "product_type", "purchased_product_id": self.p_foundation.id})

        with TemporaryDirectory() as tmp_dir:
            frame, _metadata = self._build(out_dir=Path(tmp_dir))
            decision_types = set(frame["decision_type"].astype(str))
            self.assertIn("initial_refresh", decision_types)
            self.assertIn("post_completed", decision_types)

    def test_reproducible_output(self):
        user = self._user("transitions_makeup_u5")
        plan = self._plan(user=user, category="makeup")
        step1 = RoadmapStep.objects.create(plan=plan, step_index=1, product_type="foundation", status=RoadmapStep.Status.RECOMMENDED)
        step2 = RoadmapStep.objects.create(plan=plan, step_index=2, product_type="mascara", status=RoadmapStep.Status.MISSING)
        t0 = timezone.now() - timedelta(days=30)

        self._event(user=user, plan=plan, event_type=RoadmapEvent.Type.PLAN_REFRESHED, created_at=t0, context={"category": "makeup", "next_step_id": step1.id, "next_step_index": 1, "next_product_type": "foundation"})
        self._event(user=user, plan=plan, step=step1, event_type=RoadmapEvent.Type.STEP_GENERATED, created_at=t0 + timedelta(seconds=1), context={"category": "makeup", "step_id": step1.id, "step_index": 1, "product_type": "foundation", "status": "recommended"})
        self._event(user=user, plan=plan, step=step2, event_type=RoadmapEvent.Type.STEP_GENERATED, created_at=t0 + timedelta(seconds=2), context={"category": "makeup", "step_id": step2.id, "step_index": 2, "product_type": "mascara", "status": "missing"})
        self._tx(user=user, product=self.p_foundation, created_at=t0 + timedelta(days=1), idem_key="repro-foundation")
        self._event(user=user, plan=plan, step=step1, event_type=RoadmapEvent.Type.STEP_COMPLETED, created_at=t0 + timedelta(days=1, seconds=1), context={"category": "makeup", "product_type": "foundation", "matched_by": "product_type", "purchased_product_id": self.p_foundation.id})

        with TemporaryDirectory() as tmp_dir_1, TemporaryDirectory() as tmp_dir_2:
            frame_1, metadata_1 = self._build(out_dir=Path(tmp_dir_1))
            frame_2, metadata_2 = self._build(out_dir=Path(tmp_dir_2))
            pd.testing.assert_frame_equal(
                frame_1.sort_values(["decision_id", "candidate_type"]).reset_index(drop=True),
                frame_2.sort_values(["decision_id", "candidate_type"]).reset_index(drop=True),
                check_like=False,
            )
            self.assertEqual(metadata_1["rows_total"], metadata_2["rows_total"])
            self.assertEqual(metadata_1["decision_type_distribution"], metadata_2["decision_type_distribution"])
