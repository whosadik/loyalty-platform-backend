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
from transactions.models import Transaction, TransactionItem
from users_app.models import CustomerProfile

try:
    import pandas as pd
except Exception:  # pragma: no cover
    pd = None


def _read_parquet(path: Path) -> "pd.DataFrame":
    return pd.read_parquet(path)


class RoadmapTeacherDatasetTests(TestCase):
    def setUp(self):
        if pd is None:
            self.skipTest("pandas is required")

        User = get_user_model()
        self.User = User

        self.p_shampoo = Product.objects.create(
            name="Teacher Shampoo",
            brand="H",
            price=Decimal("10.00"),
            category="haircare",
            product_type="shampoo",
            attrs={"hair_type": "straight", "scalp_type": "oily"},
            concerns=["oiliness"],
            in_stock=True,
        )
        self.p_mask = Product.objects.create(
            name="Teacher Mask",
            brand="H",
            price=Decimal("11.00"),
            category="haircare",
            product_type="hair_mask",
            attrs={"hair_type": "curly", "hair_thickness": "thick"},
            concerns=["dryness", "damage"],
            in_stock=True,
        )
        self.p_foundation = Product.objects.create(
            name="Teacher Foundation",
            brand="M",
            price=Decimal("12.00"),
            category="makeup",
            product_type="foundation",
            attrs={"finish": "natural", "coverage": "medium"},
            concerns=["long_wear"],
            in_stock=True,
        )
        self.p_warm_day = Product.objects.create(
            name="Teacher Warm Day",
            brand="F",
            price=Decimal("20.00"),
            category="fragrance",
            product_type="edp",
            attrs={"scent_family": "citrus", "notes": ["bergamot", "orange"], "intensity": "soft"},
            in_stock=True,
        )
        self.p_warm_evening = Product.objects.create(
            name="Teacher Warm Evening",
            brand="F",
            price=Decimal("21.00"),
            category="fragrance",
            product_type="edp",
            attrs={"scent_family": "amber", "notes": ["amber", "vanilla"], "intensity": "strong"},
            in_stock=True,
        )

    def _user(self, username: str, **profile_kwargs):
        user = self.User.objects.create_user(username=username, password="pass12345")
        defaults = {
            "skin_type": profile_kwargs.pop("skin_type", "normal"),
            "goals": profile_kwargs.pop("goals", []),
            "avoid_flags": profile_kwargs.pop("avoid_flags", []),
            "budget": profile_kwargs.pop("budget", "mid"),
            "hair_profile": profile_kwargs.pop("hair_profile", {}),
            "makeup_profile": profile_kwargs.pop("makeup_profile", {}),
            "fragrance_profile": profile_kwargs.pop("fragrance_profile", {}),
        }
        CustomerProfile.objects.update_or_create(user=user, defaults=defaults)
        return user

    def _tx(self, *, user, product: Product, created_at, idem_key: str):
        tx = Transaction.objects.create(
            user=user,
            total_amount=Decimal("10.00"),
            channel="web",
            idempotency_key=idem_key,
        )
        Transaction.objects.filter(id=tx.id).update(created_at=created_at)
        tx.refresh_from_db()
        item = TransactionItem.objects.create(
            transaction=tx,
            product=product,
            quantity=1,
            unit_price=Decimal("10.00"),
        )
        return item

    def _build(self, out_dir: Path):
        call_command("build_roadmap_teacher_dataset", out_dir=str(out_dir), days=3650, seed=42)
        sequence = _read_parquet(out_dir / "sequence_dataset.parquet")
        stepwise = _read_parquet(out_dir / "stepwise_dataset.parquet")
        metadata = json.loads((out_dir / "metadata.json").read_text(encoding="utf-8"))
        splits = json.loads((out_dir / "splits.json").read_text(encoding="utf-8"))
        return sequence, stepwise, metadata, splits

    def _seed_fixture(self):
        base = timezone.now() - timedelta(days=120)
        fragrance_user = self._user(
            "teacher_fragrance_u1",
            fragrance_profile={"liked_families": ["citrus"], "liked_notes": ["bergamot"], "intensity_pref": "soft"},
        )
        haircare_scalp = self._user(
            "teacher_hair_scalp",
            hair_profile={"hair_type": "straight", "scalp_type": "oily", "concerns": ["oiliness"]},
            goals=["scalp_balance"],
        )
        haircare_basic = self._user(
            "teacher_hair_basic",
            hair_profile={"hair_type": "straight", "scalp_type": "normal", "concerns": []},
        )
        makeup_user = self._user(
            "teacher_makeup_u1",
            makeup_profile={"finish_pref": ["natural"], "coverage_pref": ["medium"], "tone_family": "warm"},
        )

        self._tx(user=fragrance_user, product=self.p_warm_day, created_at=base, idem_key="teacher-fragrance-anchor")
        self._tx(user=haircare_scalp, product=self.p_shampoo, created_at=base + timedelta(days=1), idem_key="teacher-hair-scalp-anchor")
        self._tx(user=haircare_basic, product=self.p_shampoo, created_at=base + timedelta(days=2), idem_key="teacher-hair-basic-anchor")
        self._tx(user=makeup_user, product=self.p_foundation, created_at=base + timedelta(days=3), idem_key="teacher-makeup-anchor")

        for idx in range(8):
            user = self._user(f"teacher_extra_{idx}")
            self._tx(user=user, product=self.p_mask if idx % 2 else self.p_shampoo, created_at=base + timedelta(days=10 + idx), idem_key=f"teacher-extra-{idx}")

    def test_fragrance_targets_use_slots_only(self):
        self._seed_fixture()
        with TemporaryDirectory() as tmp_dir:
            sequence, _stepwise, _metadata, _splits = self._build(Path(tmp_dir))
            fragrance_rows = sequence[sequence["category"].astype(str) == "fragrance"]
            self.assertFalse(fragrance_rows.empty)
            for payload in fragrance_rows["target_sequence_json"].astype(str):
                sequence_tokens = json.loads(payload)
                self.assertTrue(set(sequence_tokens).issubset({"warm_day", "warm_evening", "cold_day", "cold_evening"}))
                self.assertNotIn("edp", sequence_tokens)

    def test_no_user_leakage_across_splits(self):
        self._seed_fixture()
        with TemporaryDirectory() as tmp_dir:
            sequence, _stepwise, metadata, splits = self._build(Path(tmp_dir))
            self.assertEqual(metadata["split_user_overlap_counts"], {"train_val": 0, "train_test": 0, "val_test": 0})
            train_users = set(sequence[sequence["split"].astype(str) == "train"]["user_id"].astype(int))
            val_users = set(sequence[sequence["split"].astype(str) == "val"]["user_id"].astype(int))
            test_users = set(sequence[sequence["split"].astype(str) == "test"]["user_id"].astype(int))
            self.assertFalse(train_users.intersection(val_users))
            self.assertFalse(train_users.intersection(test_users))
            self.assertFalse(val_users.intersection(test_users))
            self.assertEqual(splits["strategy"], "user_group_hash")

    def test_sequence_targets_deterministic(self):
        self._seed_fixture()
        with TemporaryDirectory() as tmp_dir:
            sequence, _stepwise, _metadata, _splits = self._build(Path(tmp_dir))
            hair_rows = sequence[sequence["category"].astype(str) == "haircare"].sort_values("user_id").reset_index(drop=True)
            self.assertGreaterEqual(len(hair_rows), 2)
            first_sequence = json.loads(str(hair_rows.iloc[0]["target_sequence_json"]))
            self.assertTrue(first_sequence)
            self.assertEqual(first_sequence[0], "shampoo")

    def test_stepwise_ranking_has_exactly_one_positive_per_position(self):
        self._seed_fixture()
        with TemporaryDirectory() as tmp_dir:
            _sequence, stepwise, _metadata, _splits = self._build(Path(tmp_dir))
            positives = stepwise.groupby(["planning_id", "position"])["y"].sum().astype(int)
            self.assertTrue((positives == 1).all())

    def test_teacher_dataset_reproducible(self):
        self._seed_fixture()
        with TemporaryDirectory() as tmp_dir_1, TemporaryDirectory() as tmp_dir_2:
            sequence_1, stepwise_1, metadata_1, _splits_1 = self._build(Path(tmp_dir_1))
            sequence_2, stepwise_2, metadata_2, _splits_2 = self._build(Path(tmp_dir_2))
            pd.testing.assert_frame_equal(
                sequence_1.sort_values(["planning_id"]).reset_index(drop=True),
                sequence_2.sort_values(["planning_id"]).reset_index(drop=True),
                check_like=False,
            )
            pd.testing.assert_frame_equal(
                stepwise_1.sort_values(["planning_id", "position", "candidate_type"]).reset_index(drop=True),
                stepwise_2.sort_values(["planning_id", "position", "candidate_type"]).reset_index(drop=True),
                check_like=False,
            )
            self.assertEqual(metadata_1["planning_examples_total"], metadata_2["planning_examples_total"])
            self.assertEqual(metadata_1["target_length_distribution"], metadata_2["target_length_distribution"])

    def test_seed_purchase_or_profile_changes_target_sequence(self):
        base = timezone.now() - timedelta(days=100)
        user_a = self._user(
            "teacher_change_a",
            hair_profile={"hair_type": "straight", "scalp_type": "oily", "concerns": ["oiliness"]},
            goals=["scalp_balance"],
        )
        user_b = self._user(
            "teacher_change_b",
            hair_profile={"hair_type": "straight", "scalp_type": "normal", "concerns": []},
        )
        self._tx(user=user_a, product=self.p_shampoo, created_at=base, idem_key="change-a")
        self._tx(user=user_b, product=self.p_mask, created_at=base + timedelta(minutes=1), idem_key="change-b")

        with TemporaryDirectory() as tmp_dir:
            sequence, _stepwise, _metadata, _splits = self._build(Path(tmp_dir))
            rows = sequence[sequence["category"].astype(str) == "haircare"].sort_values("user_id")
            self.assertEqual(len(rows), 2)
            seq_a = json.loads(str(rows.iloc[0]["target_sequence_json"]))
            seq_b = json.loads(str(rows.iloc[1]["target_sequence_json"]))
            self.assertNotEqual(seq_a, seq_b)
            self.assertIn("scalp_serum", seq_a)
            self.assertNotIn("scalp_serum", seq_b)

    def test_stop_appears_after_target_length(self):
        self._seed_fixture()
        with TemporaryDirectory() as tmp_dir:
            sequence, stepwise, _metadata, _splits = self._build(Path(tmp_dir))
            for row in sequence.itertuples(index=False):
                planning_id = int(row.planning_id)
                target_length = int(row.target_length)
                planning_rows = stepwise[stepwise["planning_id"].astype(int) == planning_id]
                stop_rows = planning_rows[
                    (planning_rows["candidate_type"].astype(str) == "__stop__")
                    & (planning_rows["y"].astype(int) == 1)
                ]
                self.assertEqual(len(stop_rows), 1)
                self.assertEqual(int(stop_rows.iloc[0]["position"]), target_length + 1)
