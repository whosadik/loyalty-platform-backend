from __future__ import annotations

import csv
import json
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory

from django.core.management import call_command
from django.test import TestCase


class BuildRoadmapScenarioPackFromLlmJsonTests(TestCase):
    def test_build_pack_from_llm_json_and_validate_import_contract(self):
        payload = {
            "scenario_set": "haircare_llm_test",
            "category": "haircare",
            "shared_catalog": [
                {
                    "product_key": "shampoo_a",
                    "name": "Hydra Shampoo",
                    "brand": "Scenario Lab",
                    "price": "6400",
                    "currency": "KZT",
                    "category": "haircare",
                    "product_type": "shampoo",
                    "concerns": ["dryness", "frizz"],
                    "actives": ["glycerin", "aloe"],
                    "flags": [],
                    "attrs": {
                        "hair_type": "curly",
                        "scalp_type": "normal",
                        "hair_thickness": "thick",
                    },
                    "raw_meta": {"line": "hydrate", "finish": "soft"},
                    "ingredients_inci": "Water, Glycerin, Aloe Vera Juice",
                },
                {
                    "product_key": "conditioner_a",
                    "name": "Hydra Conditioner",
                    "brand": "Scenario Lab",
                    "price": "6900",
                    "currency": "KZT",
                    "category": "haircare",
                    "product_type": "conditioner",
                    "concerns": ["dryness", "frizz", "detangling"],
                    "actives": ["glycerin", "aloe", "panthenol"],
                    "flags": [],
                    "attrs": {
                        "hair_type": "curly",
                        "scalp_type": "normal",
                        "hair_thickness": "thick",
                    },
                    "raw_meta": {"line": "hydrate", "finish": "soft"},
                    "ingredients_inci": "Water, Glycerin, Aloe Vera Juice, Panthenol",
                },
                {
                    "product_key": "conditioner_alt",
                    "name": "Hydra Conditioner Plus",
                    "brand": "Scenario Lab",
                    "price": "7200",
                    "currency": "KZT",
                    "category": "haircare",
                    "product_type": "conditioner",
                    "concerns": ["dryness", "frizz", "detangling"],
                    "actives": ["glycerin", "aloe", "panthenol"],
                    "flags": [],
                    "attrs": {
                        "hair_type": "curly",
                        "scalp_type": "normal",
                        "hair_thickness": "thick",
                    },
                    "raw_meta": {"line": "hydrate_plus", "finish": "soft"},
                    "ingredients_inci": "Water, Aloe Vera Juice, Glycerin, Panthenol",
                },
                {
                    "product_key": "mask_a",
                    "name": "Repair Mask",
                    "brand": "Scenario Lab",
                    "price": "8200",
                    "currency": "KZT",
                    "category": "haircare",
                    "product_type": "hair_mask",
                    "concerns": ["repair", "dryness"],
                    "actives": ["keratin", "amino_acids"],
                    "flags": [],
                    "attrs": {
                        "hair_type": "curly",
                        "scalp_type": "normal",
                        "hair_thickness": "thick",
                    },
                    "raw_meta": {"line": "repair", "finish": "rich"},
                    "ingredients_inci": "Water, Hydrolyzed Keratin, Amino Acids",
                },
            ],
            "scenarios": [
                {
                    "slug": "cond_semantic",
                    "segment": "semantic_conditioner",
                    "expected_next_product_type": "conditioner",
                    "outcome_tag": "completed_semantic",
                    "profile": {
                        "skin_type": "normal",
                        "goals": ["hydration", "definition"],
                        "avoid_flags": [],
                        "budget": "medium",
                        "hair_profile": {
                            "hair_type": "curly",
                            "scalp_type": "normal",
                            "hair_thickness": "thick",
                            "concerns": ["dryness", "frizz"],
                        },
                    },
                    "transactions": [
                        {
                            "offset_days": -12,
                            "channel": "offline",
                            "items": [{"product_key": "shampoo_a", "quantity": 1}],
                        },
                        {
                            "offset_days": 4,
                            "channel": "online",
                            "items": [{"product_key": "conditioner_alt", "quantity": 1}],
                        },
                    ],
                    "steps": [
                        {
                            "step_index": 1,
                            "product_type": "shampoo",
                            "status": "completed",
                            "recommended_product_key": "shampoo_a",
                            "cadence": "weekly",
                            "score": 0.97,
                            "confidence": 0.91,
                            "why": ["already_owned", "hydrate_anchor"],
                        },
                        {
                            "step_index": 2,
                            "product_type": "conditioner",
                            "status": "recommended",
                            "recommended_product_key": "conditioner_a",
                            "cadence": "weekly",
                            "score": 0.92,
                            "confidence": 0.87,
                            "why": ["follow_shampoo", "curl_profile_match"],
                        },
                        {
                            "step_index": 3,
                            "product_type": "hair_mask",
                            "status": "missing",
                            "recommended_product_key": "mask_a",
                            "cadence": "optional",
                            "score": 0.71,
                            "confidence": 0.56,
                            "why": ["future_repair"],
                        },
                    ],
                    "events": [
                        {
                            "event_type": "roadmap_plan_refreshed",
                            "offset_hours": -1,
                            "step_index": None,
                            "context": {},
                        },
                        {
                            "event_type": "roadmap_step_exposed",
                            "offset_hours": 0,
                            "step_index": 2,
                            "context": {
                                "category": "haircare",
                                "product_type": "conditioner",
                                "recommended_product_key": "conditioner_a",
                                "sources": ["roadmap_api"],
                            },
                        },
                        {
                            "event_type": "roadmap_step_clicked",
                            "offset_hours": 3,
                            "step_index": 2,
                            "context": {
                                "category": "haircare",
                                "product_type": "conditioner",
                                "recommended_product_key": "conditioner_a",
                            },
                        },
                        {
                            "event_type": "roadmap_step_completed",
                            "offset_hours": 96,
                            "step_index": 2,
                            "context": {
                                "category": "haircare",
                                "product_type": "conditioner",
                                "recommended_product_key": "conditioner_a",
                                "matched_by": "semantic_content_match",
                                "match_meta": {
                                    "recommended_product_key": "conditioner_a",
                                    "purchased_product_key": "conditioner_alt",
                                    "semantic_score": 1.22,
                                },
                            },
                        },
                    ],
                }
            ],
        }

        with TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            input_path = tmp_path / "llm.json"
            out_dir = tmp_path / "scenario_pack"
            input_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

            call_command(
                "build_roadmap_scenario_pack_from_llm_json",
                input=str(input_path),
                out_dir=str(out_dir),
                replicas=1,
                days_ago_start=60,
                id_base=975000,
            )

            summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(str(summary.get("scenario_set")), "haircare_llm_test")
            self.assertEqual(int(summary.get("users_count", 0)), 1)
            self.assertEqual(int(summary.get("products_count", 0)), 4)
            self.assertEqual(int(summary.get("transactions_count", 0)), 2)
            self.assertEqual(int(summary.get("roadmap_plans_count", 0)), 1)
            self.assertEqual(int(summary.get("roadmap_steps_count", 0)), 3)
            self.assertEqual(int(summary.get("roadmap_events_count", 0)), 4)
            self.assertEqual(summary.get("expected_next_distribution"), {"conditioner": 1})
            self.assertEqual(summary.get("outcome_tag_distribution"), {"completed_semantic": 1})

            with (out_dir / "roadmap_events.csv").open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            completed_rows = [row for row in rows if row.get("event_type") == "roadmap_step_completed"]
            self.assertEqual(len(completed_rows), 1)
            completed_context = json.loads(completed_rows[0]["context"])
            self.assertEqual(completed_context.get("matched_by"), "semantic_content_match")
            match_meta = completed_context.get("match_meta") or {}
            self.assertIn("recommended_product_id", match_meta)
            self.assertIn("purchased_product_id", match_meta)
            self.assertNotIn("recommended_product_key", match_meta)
            self.assertNotIn("purchased_product_key", match_meta)

            out = StringIO()
            call_command(
                "import_synth_dataset",
                path=str(out_dir),
                dry_run=True,
                stdout=out,
            )
            self.assertIn("Dry-run completed successfully", out.getvalue())

    def test_build_pack_uses_recommended_step_as_planned_target_even_if_missing_step_comes_first(self):
        payload = {
            "scenario_set": "haircare_llm_target_order_test",
            "category": "haircare",
            "shared_catalog": [
                {
                    "product_key": "scalp_shampoo",
                    "name": "Scalp Shampoo",
                    "brand": "Scenario Lab",
                    "price": "6100",
                    "currency": "KZT",
                    "category": "haircare",
                    "product_type": "shampoo",
                    "concerns": ["oiliness", "flakes"],
                    "actives": ["salicylic_acid", "niacinamide"],
                    "flags": [],
                    "attrs": {
                        "hair_type": "wavy",
                        "scalp_type": "oily",
                        "hair_thickness": "medium",
                    },
                    "raw_meta": {"line": "scalp", "finish": "fresh"},
                    "ingredients_inci": "Water, Salicylic Acid, Niacinamide",
                },
                {
                    "product_key": "length_conditioner",
                    "name": "Length Conditioner",
                    "brand": "Scenario Lab",
                    "price": "6900",
                    "currency": "KZT",
                    "category": "haircare",
                    "product_type": "conditioner",
                    "concerns": ["smoothness", "detangling"],
                    "actives": ["panthenol"],
                    "flags": [],
                    "attrs": {
                        "hair_type": "wavy",
                        "scalp_type": "normal",
                        "hair_thickness": "medium",
                    },
                    "raw_meta": {"line": "length", "finish": "soft"},
                    "ingredients_inci": "Water, Panthenol",
                },
                {
                    "product_key": "scalp_serum",
                    "name": "Scalp Serum",
                    "brand": "Scenario Lab",
                    "price": "7300",
                    "currency": "KZT",
                    "category": "haircare",
                    "product_type": "scalp_serum",
                    "concerns": ["oiliness", "flakes", "itchiness"],
                    "actives": ["salicylic_acid", "niacinamide", "zinc_pca"],
                    "flags": [],
                    "attrs": {
                        "hair_type": "wavy",
                        "scalp_type": "oily",
                        "hair_thickness": "medium",
                    },
                    "raw_meta": {"line": "scalp", "finish": "fresh"},
                    "ingredients_inci": "Water, Salicylic Acid, Niacinamide, Zinc PCA",
                },
            ],
            "scenarios": [
                {
                    "slug": "serum_after_missing_conditioner",
                    "segment": "scalp_over_conditioner",
                    "expected_next_product_type": "scalp_serum",
                    "outcome_tag": "completed_exact",
                    "profile": {
                        "skin_type": "oily",
                        "goals": ["oiliness", "scalp_balance"],
                        "avoid_flags": [],
                        "budget": "medium",
                        "hair_profile": {
                            "hair_type": "wavy",
                            "scalp_type": "oily",
                            "hair_thickness": "medium",
                            "concerns": ["oiliness", "flakes", "itchiness"],
                        },
                    },
                    "transactions": [
                        {
                            "offset_days": -12,
                            "channel": "online",
                            "items": [{"product_key": "scalp_shampoo", "quantity": 1}],
                        },
                        {
                            "offset_days": 4,
                            "channel": "online",
                            "items": [{"product_key": "scalp_serum", "quantity": 1}],
                        },
                    ],
                    "steps": [
                        {
                            "step_index": 1,
                            "product_type": "shampoo",
                            "status": "completed",
                            "recommended_product_key": "scalp_shampoo",
                            "cadence": "weekly",
                            "score": 0.77,
                            "confidence": 0.82,
                            "why": ["wash_step_done"],
                        },
                        {
                            "step_index": 2,
                            "product_type": "conditioner",
                            "status": "missing",
                            "recommended_product_key": "length_conditioner",
                            "cadence": "weekly",
                            "score": 0.4,
                            "confidence": 0.38,
                            "why": ["not_primary_issue"],
                        },
                        {
                            "step_index": 3,
                            "product_type": "scalp_serum",
                            "status": "recommended",
                            "recommended_product_key": "scalp_serum",
                            "cadence": "daily",
                            "score": 0.93,
                            "confidence": 0.9,
                            "why": ["oily_scalp_signal"],
                        },
                    ],
                    "events": [
                        {
                            "event_type": "roadmap_plan_refreshed",
                            "offset_hours": 0,
                            "step_index": None,
                            "context": {},
                        },
                        {
                            "event_type": "roadmap_step_exposed",
                            "offset_hours": 2,
                            "step_index": 3,
                            "context": {
                                "category": "haircare",
                                "product_type": "scalp_serum",
                                "recommended_product_key": "scalp_serum",
                            },
                        },
                        {
                            "event_type": "roadmap_step_clicked",
                            "offset_hours": 5,
                            "step_index": 3,
                            "context": {
                                "category": "haircare",
                                "product_type": "scalp_serum",
                                "recommended_product_key": "scalp_serum",
                            },
                        },
                        {
                            "event_type": "roadmap_step_completed",
                            "offset_hours": 20,
                            "step_index": 3,
                            "context": {
                                "category": "haircare",
                                "product_type": "scalp_serum",
                                "matched_by": "recommended_product_id",
                                "match_meta": {
                                    "recommended_product_key": "scalp_serum",
                                    "purchased_product_key": "scalp_serum",
                                },
                            },
                        },
                    ],
                }
            ],
        }

        with TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            input_path = tmp_path / "llm_target_order.json"
            out_dir = tmp_path / "scenario_pack"
            input_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

            call_command(
                "build_roadmap_scenario_pack_from_llm_json",
                input=str(input_path),
                out_dir=str(out_dir),
                replicas=1,
                days_ago_start=60,
                id_base=976000,
            )

            with (out_dir / "roadmap_plans.csv").open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 1)
            meta = json.loads(rows[0]["meta"])
            ml = meta.get("ml") or {}
            self.assertEqual(ml.get("planned_target_product_type"), "scalp_serum")
            self.assertEqual(int(ml.get("planned_target_step_index") or 0), 3)
