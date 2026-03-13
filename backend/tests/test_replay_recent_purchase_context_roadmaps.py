from __future__ import annotations

from io import StringIO
from unittest.mock import patch

from django.contrib.auth import get_user_model
from django.core.management import call_command
from django.test import TestCase

from catalog.models import Product
from roadmap_app.models import RoadmapPlan
from transactions.models import Transaction, TransactionItem


class ReplayRecentPurchaseContextRoadmapsTests(TestCase):
    def _fixture(self, username: str) -> tuple[RoadmapPlan, Transaction]:
        User = get_user_model()
        user = User.objects.create_user(username=username, password="pass12345")
        product = Product.objects.create(
            name=f"{username} conditioner",
            brand="Test",
            category="haircare",
            product_type="conditioner",
            price="19.90",
            in_stock=True,
        )
        plan = RoadmapPlan.objects.create(
            user=user,
            category="haircare",
            is_active=True,
            meta={
                "source": "roadmap_v1",
                "ml": {
                    "decision": "model_used",
                    "model_slot": "active",
                    "planned_target_product_type": "shampoo",
                    "planned_target_step_index": 1,
                    "rollout_reason": "category_enabled",
                },
                "context": {
                    "refresh_caller": "update_roadmap_from_purchase",
                    "post_ctx_categories": ["haircare"],
                    "post_ctx_product_ids": [product.id],
                },
            },
        )
        txn = Transaction.objects.create(user=user, total_amount="19.90", channel="offline")
        TransactionItem.objects.create(transaction=txn, product=product, quantity=1, unit_price="19.90")
        return plan, txn

    def test_dry_run_replays_recent_purchase_context_without_persisting(self):
        plan, _ = self._fixture("purchase_replay_dry")

        def _refresh(user, category: str, post_ctx: dict | None = None):
            current = RoadmapPlan.objects.get(id=plan.id)
            meta = dict(current.meta or {})
            ml = dict(meta.get("ml") or {})
            ml["model_slot"] = "partial_candidate"
            ml["planned_target_product_type"] = "hair_mask"
            ml["planned_target_step_index"] = 3
            ml["rollout_reason"] = "selected"
            meta["ml"] = ml
            current.meta = meta
            current.save(update_fields=["meta", "updated_at"])
            return current

        out = StringIO()
        with patch(
            "roadmap_app.management.commands.replay_recent_purchase_context_roadmaps.refresh_roadmap",
            side_effect=_refresh,
        ):
            call_command(
                "replay_recent_purchase_context_roadmaps",
                days=30,
                category="haircare",
                stdout=out,
            )

        text = out.getvalue()
        plan.refresh_from_db()
        ml = (plan.meta or {}).get("ml") or {}

        self.assertIn("mode: `dry-run`", text)
        self.assertIn("transactions selected: `1`", text)
        self.assertIn("plans replayed: `1`", text)
        self.assertIn("changed after replay: `1`", text)
        self.assertIn("partial_candidate: `1`", text)
        self.assertIn("active -> partial_candidate: `1`", text)
        self.assertEqual(str(ml.get("model_slot")), "active")
        self.assertEqual(str(ml.get("planned_target_product_type")), "shampoo")

    def test_write_persists_replayed_purchase_context_result(self):
        plan, _ = self._fixture("purchase_replay_write")

        def _refresh(user, category: str, post_ctx: dict | None = None):
            current = RoadmapPlan.objects.get(id=plan.id)
            meta = dict(current.meta or {})
            ml = dict(meta.get("ml") or {})
            ml["model_slot"] = "partial_candidate"
            ml["planned_target_product_type"] = "hair_mask"
            ml["planned_target_step_index"] = 3
            ml["rollout_reason"] = "selected"
            meta["ml"] = ml
            current.meta = meta
            current.save(update_fields=["meta", "updated_at"])
            return current

        out = StringIO()
        with patch(
            "roadmap_app.management.commands.replay_recent_purchase_context_roadmaps.refresh_roadmap",
            side_effect=_refresh,
        ):
            call_command(
                "replay_recent_purchase_context_roadmaps",
                days=30,
                category="haircare",
                write=True,
                stdout=out,
            )

        text = out.getvalue()
        plan.refresh_from_db()
        ml = (plan.meta or {}).get("ml") or {}

        self.assertIn("mode: `write`", text)
        self.assertIn("plans updated: `1`", text)
        self.assertEqual(str(ml.get("model_slot")), "partial_candidate")
        self.assertEqual(str(ml.get("planned_target_product_type")), "hair_mask")

    def test_analysis_only_uses_build_chain_without_refreshing_plan(self):
        plan, _ = self._fixture("purchase_replay_analysis")

        out = StringIO()
        with patch(
            "roadmap_app.management.commands.replay_recent_purchase_context_roadmaps._build_chain",
            return_value=(
                ["shampoo", "conditioner", "hair_mask"],
                {},
                [],
                {
                    "decision": "model_used",
                    "model_slot": "partial_candidate",
                    "planned_target_product_type": "hair_mask",
                    "planned_target_step_index": 3,
                    "rollout_reason": "selected",
                    "model_version": "semantic_v4_local",
                },
            ),
        ) as build_chain_mock, patch(
            "roadmap_app.management.commands.replay_recent_purchase_context_roadmaps.refresh_roadmap",
        ) as refresh_mock:
            call_command(
                "replay_recent_purchase_context_roadmaps",
                days=30,
                category="haircare",
                analysis_only=True,
                stdout=out,
            )

        text = out.getvalue()
        plan.refresh_from_db()
        ml = (plan.meta or {}).get("ml") or {}

        self.assertIn("mode: `analysis-only`", text)
        self.assertIn("partial_candidate: `1`", text)
        self.assertIn("hair_mask: `1`", text)
        self.assertEqual(str(ml.get("model_slot")), "active")
        build_chain_mock.assert_called_once()
        refresh_mock.assert_not_called()
