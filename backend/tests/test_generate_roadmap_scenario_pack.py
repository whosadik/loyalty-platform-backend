from __future__ import annotations

import csv
import json
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory

from django.core.management import call_command
from django.test import TestCase


class GenerateRoadmapScenarioPackTests(TestCase):
    def test_generate_haircare_pack_and_validate_import_contract(self):
        with TemporaryDirectory() as tmp_dir:
            out_dir = Path(tmp_dir) / "scenario_pack"
            call_command(
                "generate_roadmap_scenario_pack",
                out_dir=str(out_dir),
                scenario_set="haircare_v1",
                replicas=1,
                days_ago_start=75,
                id_base=950000,
            )

            required_files = [
                "products.csv",
                "users.csv",
                "customer_profiles.csv",
                "transactions.csv",
                "transaction_items.csv",
                "owned_products.csv",
                "roadmap_plans.csv",
                "roadmap_steps.csv",
                "campaign_budgets.csv",
                "offers.csv",
                "offer_assignments.csv",
                "offer_events.csv",
                "roadmap_events.csv",
                "recommendation_events.csv",
                "summary.json",
                "README.md",
            ]
            for file_name in required_files:
                self.assertTrue((out_dir / file_name).exists(), file_name)

            summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(str(summary.get("scenario_set")), "haircare_v1")
            self.assertEqual(int(summary.get("users_count", 0)), 6)
            self.assertEqual(int(summary.get("transactions_count", 0)), 16)
            self.assertEqual(int(summary.get("roadmap_plans_count", 0)), 6)
            self.assertEqual(int(summary.get("roadmap_steps_count", 0)), 20)
            self.assertEqual(int(summary.get("roadmap_events_count", 0)), 23)
            self.assertEqual(
                summary.get("expected_next_distribution"),
                {
                    "conditioner": 2,
                    "hair_mask": 2,
                    "leave_in": 1,
                    "scalp_serum": 1,
                },
            )
            self.assertEqual(
                summary.get("outcome_tag_distribution"),
                {
                    "completed_exact": 4,
                    "completed_semantic": 1,
                    "no_conversion": 1,
                },
            )

            with (out_dir / "roadmap_events.csv").open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            matched_by_values = {
                str(json.loads(row["context"]).get("matched_by") or "").strip().lower()
                for row in rows
                if row.get("event_type") == "roadmap_step_completed"
            }
            event_types = {str(row.get("event_type") or "").strip().lower() for row in rows}
            self.assertIn("semantic_content_match", matched_by_values)
            self.assertIn("roadmap_step_skipped", event_types)

            out = StringIO()
            call_command(
                "import_synth_dataset",
                path=str(out_dir),
                dry_run=True,
                stdout=out,
            )
            self.assertIn("Dry-run completed successfully", out.getvalue())

    def test_generate_haircare_hardcases_v2_pack_and_validate_summary(self):
        with TemporaryDirectory() as tmp_dir:
            out_dir = Path(tmp_dir) / "scenario_pack_hardcases"
            call_command(
                "generate_roadmap_scenario_pack",
                out_dir=str(out_dir),
                scenario_set="haircare_hardcases_v2",
                replicas=1,
                days_ago_start=75,
                id_base=952000,
            )

            summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(str(summary.get("scenario_set")), "haircare_hardcases_v2")
            self.assertEqual(int(summary.get("users_count", 0)), 5)
            self.assertEqual(int(summary.get("transactions_count", 0)), 17)
            self.assertEqual(int(summary.get("roadmap_plans_count", 0)), 5)
            self.assertEqual(int(summary.get("roadmap_steps_count", 0)), 18)
            self.assertEqual(int(summary.get("roadmap_events_count", 0)), 20)
            self.assertEqual(
                summary.get("expected_next_distribution"),
                {
                    "conditioner": 1,
                    "hair_oil": 1,
                    "leave_in": 2,
                    "scalp_serum": 1,
                },
            )
            self.assertEqual(
                summary.get("outcome_tag_distribution"),
                {
                    "completed_exact": 4,
                    "completed_semantic": 1,
                },
            )

            with (out_dir / "products.csv").open("r", encoding="utf-8", newline="") as handle:
                product_rows = list(csv.DictReader(handle))
            product_types = {str(row.get("product_type") or "").strip().lower() for row in product_rows}
            self.assertIn("hair_oil", product_types)
            self.assertIn("leave_in", product_types)
            self.assertIn("scalp_serum", product_types)

            with (out_dir / "roadmap_events.csv").open("r", encoding="utf-8", newline="") as handle:
                event_rows = list(csv.DictReader(handle))
            completed_product_types = {
                str(json.loads(row["context"]).get("product_type") or "").strip().lower()
                for row in event_rows
                if row.get("event_type") == "roadmap_step_completed"
            }
            matched_by_values = {
                str(json.loads(row["context"]).get("matched_by") or "").strip().lower()
                for row in event_rows
                if row.get("event_type") == "roadmap_step_completed"
            }
            self.assertIn("leave_in", completed_product_types)
            self.assertIn("hair_oil", completed_product_types)
            self.assertIn("scalp_serum", completed_product_types)
            self.assertIn("semantic_content_match", matched_by_values)

            out = StringIO()
            call_command(
                "import_synth_dataset",
                path=str(out_dir),
                dry_run=True,
                stdout=out,
            )
            self.assertIn("Dry-run completed successfully", out.getvalue())
