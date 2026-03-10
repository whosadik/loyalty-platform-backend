from __future__ import annotations

import json
import random
from collections import Counter
from contextlib import contextmanager
from datetime import timedelta
from decimal import Decimal
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from django.contrib.auth import get_user_model
from django.core.management import call_command
from django.test import TestCase
from django.utils import timezone

from catalog.models import Product
from checkout_app.pricing import is_eligible
from offers.models import CampaignBudget, Offer, OfferAssignment
from roadmap_app.models import RoadmapEvent, RoadmapPlan, RoadmapStep
from transactions.models import Transaction, TransactionItem

try:
    import pandas as pd
except Exception:  # pragma: no cover
    pd = None


class SimulatorThrottleDisableTests(TestCase):
    def test_disable_throttle_uses_override_settings(self):
        from admin_tools.management.commands import simulate_roadmap_sessions as sim_cmd

        User = get_user_model()
        user = User.objects.create_user(username="sim_throttle_u1", password="pass12345")

        captured: dict[str, object] = {}

        @contextmanager
        def fake_override_settings(**kwargs):
            captured.update(kwargs)
            yield

        snapshot_zero = {
            "transactions": 0,
            "transaction_items": 0,
            "owned_products": 0,
            "roadmap_step_exposed": 0,
            "roadmap_clicked": 0,
            "roadmap_skipped": 0,
            "roadmap_completed": 0,
            "roadmap_completed_fragrance": 0,
            "offer_exposed": 0,
            "offer_clicked": 0,
            "offer_redeemed": 0,
            "offer_assignments": 0,
        }
        state_map = {
            int(user.id): sim_cmd.UserState(
                segment=sim_cmd.SEGMENT_CONFIGS["new"],
                favorite_category="skincare",
                profile=None,
            )
        }

        with patch.object(sim_cmd, "override_settings", side_effect=fake_override_settings), patch.object(
            sim_cmd.Command, "_soft_fix_ownedproduct_sequence", return_value=None
        ), patch.object(sim_cmd.Command, "_validate_catalog_coverage", return_value=[]), patch.object(
            sim_cmd.Command, "_select_or_create_users", return_value=[user]
        ), patch.object(
            sim_cmd.Command, "_ensure_profiles_and_loyalty", return_value=None
        ), patch.object(
            sim_cmd.Command, "_build_user_states", return_value=state_map
        ), patch.object(
            sim_cmd.Command, "_simulate_user_days", side_effect=lambda **kwargs: int(kwargs["idem_counter"])
        ), patch.object(
            sim_cmd.Command, "_snapshot_counts", return_value=snapshot_zero
        ), patch.object(
            sim_cmd.Command, "_count_completed", return_value=0
        ), patch.object(
            sim_cmd.Command, "_prepare_error_log", return_value=None
        ), patch.object(
            sim_cmd.Command, "_close_error_log", return_value=None
        ), patch.object(
            sim_cmd.Command, "_print_error_summary", return_value=None
        ), patch.object(
            sim_cmd, "call_command", return_value=None
        ):
            call_command(
                "simulate_roadmap_sessions",
                days=1,
                users=1,
                seed=1,
                avg_sessions=0.0,
                batch_users=1,
                progress_every=1,
            )

        rest_framework = captured.get("REST_FRAMEWORK")
        self.assertIsInstance(rest_framework, dict)
        self.assertEqual(rest_framework.get("DEFAULT_THROTTLE_CLASSES"), [])
        self.assertEqual(rest_framework.get("DEFAULT_THROTTLE_RATES"), {})


class DatasetIncludeNegativesTests(TestCase):
    def test_include_negatives_adds_none_class(self):
        if pd is None:
            self.skipTest("pandas is required for dataset command")

        from admin_tools.management.commands import build_roadmap_ml_dataset as build_cmd

        User = get_user_model()
        user = User.objects.create_user(username="ds_v3_u1", password="pass12345")

        p_serum = Product.objects.create(
            name="DS Serum",
            brand="B",
            price=Decimal("10.00"),
            category="skincare",
            product_type="serum",
            in_stock=True,
        )
        p_cleanser = Product.objects.create(
            name="DS Cleanser",
            brand="B",
            price=Decimal("9.00"),
            category="skincare",
            product_type="cleanser",
            in_stock=True,
        )

        plan = RoadmapPlan.objects.create(user=user, category="skincare", is_active=True, meta={})
        step_pos = RoadmapStep.objects.create(
            plan=plan,
            step_index=1,
            product_type="serum",
            status=RoadmapStep.Status.MISSING,
        )
        step_neg = RoadmapStep.objects.create(
            plan=plan,
            step_index=2,
            product_type="cleanser",
            status=RoadmapStep.Status.MISSING,
        )

        t0 = timezone.now() - timedelta(days=40)

        ev_pos_exp = RoadmapEvent.objects.create(
            user=user,
            plan=plan,
            step=step_pos,
            event_type=RoadmapEvent.Type.STEP_EXPOSED,
            context={"sources": ["roadmap_api"], "category": "skincare"},
        )
        ev_neg_exp = RoadmapEvent.objects.create(
            user=user,
            plan=plan,
            step=step_neg,
            event_type=RoadmapEvent.Type.STEP_EXPOSED,
            context={"sources": ["roadmap_api"], "category": "skincare"},
        )
        ev_pos_done = RoadmapEvent.objects.create(
            user=user,
            plan=plan,
            step=step_pos,
            event_type=RoadmapEvent.Type.STEP_COMPLETED,
            context={"category": "skincare"},
        )
        RoadmapEvent.objects.filter(id=ev_pos_exp.id).update(created_at=t0)
        RoadmapEvent.objects.filter(id=ev_neg_exp.id).update(created_at=t0)
        RoadmapEvent.objects.filter(id=ev_pos_done.id).update(created_at=t0 + timedelta(days=1))

        tx = Transaction.objects.create(
            user=user,
            total_amount=Decimal("10.00"),
            channel="web",
            idempotency_key="ds-v3-1",
        )
        Transaction.objects.filter(id=tx.id).update(created_at=t0 - timedelta(days=1))
        TransactionItem.objects.create(
            transaction=tx,
            product=p_serum,
            quantity=1,
            unit_price=Decimal("10.00"),
        )
        TransactionItem.objects.create(
            transaction=tx,
            product=p_cleanser,
            quantity=1,
            unit_price=Decimal("9.00"),
        )

        with TemporaryDirectory() as tmp_dir:
            out_dir = Path(tmp_dir)
            with patch.object(build_cmd, "MIN_POSITIVES_OVERALL", 1), patch.object(
                build_cmd, "MIN_POSITIVES_FRAGRANCE", 0
            ):
                call_command(
                    "build_roadmap_ml_dataset",
                    days=180,
                    out_dir=str(out_dir),
                    include_negatives=True,
                    k=10,
                )

            dataset_path = out_dir / "dataset.parquet"
            if dataset_path.exists():
                frame = pd.read_parquet(dataset_path)
            else:
                frame = pd.read_csv(out_dir / "dataset.csv")

            self.assertIn("__none__", set(frame["target_class"].astype(str).tolist()))
            self.assertGreater(int((frame["target_class"].astype(str) == "__none__").sum()), 0)

            metadata = json.loads((out_dir / "metadata.json").read_text(encoding="utf-8"))
            self.assertTrue(bool(metadata.get("include_negatives")))
            self.assertIn("class_distribution", metadata)


class TrainingCalibrationTests(TestCase):
    def test_training_writes_temperature_and_ece(self):
        if pd is None:
            self.skipTest("pandas is required for training command")

        with TemporaryDirectory() as data_tmp, TemporaryDirectory() as model_tmp:
            data_dir = Path(data_tmp)
            model_dir = Path(model_tmp)

            rows = []
            classes = ["serum", "cleanser", "__none__"]
            categories = ["skincare", "haircare", "fragrance"]
            for user_id in range(1, 16):
                for idx in range(6):
                    cls = classes[(user_id + idx) % len(classes)]
                    category = categories[(user_id + idx) % len(categories)]
                    rows.append(
                        {
                            "user_id": user_id,
                            "target_class": cls,
                            "category": category,
                            "last_k_purchase_product_types": "serum|cleanser",
                            "last_k_purchase_categories": "skincare|haircare",
                            "favorite_brand_top1": f"brand_{user_id % 3}",
                            "favorite_brands_top3": "brand_0|brand_1|brand_2",
                            "month_of_year": 1 + (idx % 12),
                            "was_exposed_from_offers": idx % 2,
                            "has_offer_assignment_id": (idx + 1) % 2,
                            "days_since_last_purchase_in_category": 5 + idx,
                            "tx_count_90d_category": 1 + (idx % 4),
                            "tx_amount_90d_category": float(10 + idx),
                            "price_band_median_last5": float(20 + idx),
                            "owned_slot_warm_day": idx % 3,
                            "owned_slot_warm_evening": (idx + 1) % 3,
                            "owned_slot_cold_day": (idx + 2) % 3,
                            "owned_slot_cold_evening": (idx + 1) % 2,
                        }
                    )
            frame = pd.DataFrame(rows)
            frame.to_csv(data_dir / "dataset.csv", index=False)

            feature_columns = [
                "category",
                "last_k_purchase_product_types",
                "last_k_purchase_categories",
                "favorite_brand_top1",
                "favorite_brands_top3",
                "month_of_year",
                "was_exposed_from_offers",
                "has_offer_assignment_id",
                "days_since_last_purchase_in_category",
                "tx_count_90d_category",
                "tx_amount_90d_category",
                "price_band_median_last5",
                "owned_slot_warm_day",
                "owned_slot_warm_evening",
                "owned_slot_cold_day",
                "owned_slot_cold_evening",
            ]
            metadata = {
                "feature_columns": feature_columns,
                "categorical_features": [
                    "category",
                    "last_k_purchase_product_types",
                    "last_k_purchase_categories",
                    "favorite_brand_top1",
                    "favorite_brands_top3",
                ],
                "numeric_features": [c for c in feature_columns if c not in {
                    "category",
                    "last_k_purchase_product_types",
                    "last_k_purchase_categories",
                    "favorite_brand_top1",
                    "favorite_brands_top3",
                }],
                "baselines": {"splits": {}},
                "class_distribution": {"train": {}, "val": {}, "test": {}},
            }
            (data_dir / "metadata.json").write_text(
                json.dumps(metadata, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            splits = {
                "train_user_ids": list(range(1, 10)),
                "val_user_ids": list(range(10, 13)),
                "test_user_ids": list(range(13, 16)),
            }
            (data_dir / "splits.json").write_text(
                json.dumps(splits, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

            call_command(
                "train_roadmap_nextstep_model",
                data_dir=str(data_dir),
                model_dir=str(model_dir),
                estimator="hgb",
                trials=1,
                threshold=0.35,
            )

            model_meta = json.loads((model_dir / "metadata.json").read_text(encoding="utf-8"))
            self.assertGreater(float(model_meta.get("temperature") or 0.0), 0.0)

            report_json = Path("reports/roadmap_nextstep_eval.json")
            report_md = Path("reports/roadmap_nextstep_eval.md")
            self.assertTrue(report_json.exists())
            self.assertTrue(report_md.exists())

            payload = json.loads(report_json.read_text(encoding="utf-8"))
            self.assertIn("ece", payload.get("metrics_test", {}))
            self.assertIn("brier", payload.get("metrics_test", {}))
            self.assertIn("none_class_binary", payload)

            report_text = report_md.read_text(encoding="utf-8")
            self.assertIn("__none__ Performance", report_text)


class SimulatorCheckoutEligibilityBuilderTests(TestCase):
    def setUp(self):
        from admin_tools.management.commands import simulate_roadmap_sessions as sim_cmd

        self.sim_cmd = sim_cmd
        User = get_user_model()
        self.user = User.objects.create_user(username="sim_builder_u1", password="pass12345")
        self.command = sim_cmd.Command()
        self.command._max_errors = 1000
        self.command._error_rows_written = 0
        self.command._error_counts = Counter()
        self.command._error_log_handle = None
        self.rng = random.Random(11)
        self.product_cache = sim_cmd.ProductPoolCache(rng=self.rng)

        self.target_product = Product.objects.create(
            name="Builder Target Product",
            brand="B",
            price=Decimal("15.00"),
            category="skincare",
            product_type="serum",
            in_stock=True,
        )
        self.other_product = Product.objects.create(
            name="Builder Other Product",
            brand="B",
            price=Decimal("12.00"),
            category="skincare",
            product_type="cleanser",
            in_stock=True,
        )
        self.haircare_product = Product.objects.create(
            name="Builder Haircare Product",
            brand="B",
            price=Decimal("13.00"),
            category="haircare",
            product_type="shampoo",
            in_stock=True,
        )

        self.campaign = CampaignBudget.objects.create(
            name="builder_campaign",
            weekly_limit=Decimal("1000.00"),
            weekly_spent=Decimal("0.00"),
            priority=100,
            is_active=True,
        )
        self.offer = Offer.objects.create(
            name="Builder Offer",
            offer_type=Offer.Type.DISCOUNT,
            value=Decimal("10.00"),
            estimated_cost=Decimal("2.00"),
            is_active=True,
            target_scope="product_id",
            campaign=self.campaign,
            cooldown_days=0,
            expires_in_days=7,
        )

    def test_builder_includes_product_id_target_item(self):
        assignment = OfferAssignment.objects.create(
            user=self.user,
            offer=self.offer,
            target={
                "scope": "product_id",
                "value": int(self.target_product.id),
                "category": "skincare",
            },
        )
        payload, applied = self.command._build_checkout_payload(
            user_id=int(self.user.id),
            category="skincare",
            base_items=[{"product": int(self.other_product.id), "quantity": 1}],
            base_product_ids=[int(self.other_product.id)],
            max_items=3,
            assignment_id=int(assignment.id),
            assignment_target=assignment.target,
            idempotency_key="builder-k1",
            request_id="req-builder-1",
            warnings=[],
            rng=self.rng,
            product_cache=self.product_cache,
        )

        self.assertTrue(applied)
        self.assertEqual(int(payload.get("apply_assignment_id")), int(assignment.id))
        ids = {int(x.get("product")) for x in payload.get("items", [])}
        self.assertIn(int(self.target_product.id), ids)
        self.assertTrue(
            any(
                is_eligible(Product.objects.get(id=int(item["product"])), assignment.target)
                for item in payload.get("items", [])
            )
        )

    def test_builder_skips_apply_when_no_eligible_item(self):
        assignment = OfferAssignment.objects.create(
            user=self.user,
            offer=self.offer,
            target={
                "scope": "product_id",
                "value": 999999999,
                "category": "skincare",
            },
        )
        payload, applied = self.command._build_checkout_payload(
            user_id=int(self.user.id),
            category="skincare",
            base_items=[{"product": int(self.other_product.id), "quantity": 1}],
            base_product_ids=[int(self.other_product.id)],
            max_items=3,
            assignment_id=int(assignment.id),
            assignment_target=assignment.target,
            idempotency_key="builder-k2",
            request_id="req-builder-2",
            warnings=[],
            rng=self.rng,
            product_cache=self.product_cache,
        )

        self.assertFalse(applied)
        self.assertNotIn("apply_assignment_id", payload)

    def test_builder_includes_product_type_eligible_item(self):
        assignment = OfferAssignment.objects.create(
            user=self.user,
            offer=self.offer,
            target={
                "scope": "product_type",
                "value": "serum",
                "category": "skincare",
            },
        )
        payload, applied = self.command._build_checkout_payload(
            user_id=int(self.user.id),
            category="skincare",
            base_items=[{"product": int(self.other_product.id), "quantity": 1}],
            base_product_ids=[int(self.other_product.id)],
            max_items=3,
            assignment_id=int(assignment.id),
            assignment_target=assignment.target,
            idempotency_key="builder-k3",
            request_id="req-builder-3",
            warnings=[],
            rng=self.rng,
            product_cache=self.product_cache,
        )

        self.assertTrue(applied)
        self.assertEqual(int(payload.get("apply_assignment_id")), int(assignment.id))
        self.assertTrue(
            any(
                is_eligible(Product.objects.get(id=int(item["product"])), assignment.target)
                for item in payload.get("items", [])
            )
        )

    def test_builder_includes_category_eligible_item(self):
        assignment = OfferAssignment.objects.create(
            user=self.user,
            offer=self.offer,
            target={
                "scope": "category",
                "value": "skincare",
            },
        )
        payload, applied = self.command._build_checkout_payload(
            user_id=int(self.user.id),
            category="haircare",
            base_items=[{"product": int(self.haircare_product.id), "quantity": 1}],
            base_product_ids=[int(self.haircare_product.id)],
            max_items=3,
            assignment_id=int(assignment.id),
            assignment_target=assignment.target,
            idempotency_key="builder-k4",
            request_id="req-builder-4",
            warnings=[],
            rng=self.rng,
            product_cache=self.product_cache,
        )

        self.assertTrue(applied)
        self.assertEqual(int(payload.get("apply_assignment_id")), int(assignment.id))
        self.assertTrue(
            any(
                is_eligible(Product.objects.get(id=int(item["product"])), assignment.target)
                for item in payload.get("items", [])
            )
        )
