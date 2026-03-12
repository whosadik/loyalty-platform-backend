from __future__ import annotations

import csv
import json
from collections import Counter
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any

from django.core.management.base import BaseCommand, CommandError


SCENARIO_SET_HAIRCARE_V1 = "haircare_v1"
SCENARIO_SET_HAIRCARE_HARDCASES_V2 = "haircare_hardcases_v2"

CSV_HEADERS: dict[str, list[str]] = {
    "products.csv": [
        "id", "name", "brand", "price", "source_product_id", "currency", "category", "product_type",
        "concerns", "attrs", "step", "actives", "flags", "supported_skin_types", "strength", "in_stock",
        "image_url", "image_urls", "description", "application_text", "ingredients_inci", "volume_raw",
        "raw_meta", "created_at", "updated_at",
    ],
    "users.csv": ["user_id", "username", "segment", "favorite_category", "created_at"],
    "customer_profiles.csv": [
        "user_id", "skin_type", "goals", "avoid_flags", "budget", "hair_profile", "makeup_profile",
        "fragrance_profile",
    ],
    "transactions.csv": [
        "transaction_id", "user_id", "created_at", "total_amount", "channel", "idempotency_key", "pricing_meta",
    ],
    "transaction_items.csv": [
        "transaction_item_id", "transaction_id", "user_id", "product_id", "quantity", "unit_price",
    ],
    "owned_products.csv": [
        "owned_product_id", "user_id", "product_id", "quantity_total", "is_active", "last_acquired_at",
        "acquired_at", "source",
    ],
    "roadmap_plans.csv": ["plan_id", "user_id", "category", "is_active", "version", "meta", "created_at", "updated_at"],
    "roadmap_steps.csv": [
        "step_id", "plan_id", "step_index", "product_type", "status", "recommended_product_id", "suggestions",
        "score", "confidence", "why", "cadence", "created_at", "updated_at",
    ],
    "campaign_budgets.csv": [
        "campaign_id", "name", "weekly_limit", "weekly_spent", "priority", "is_active", "allowed_steps",
        "allowed_categories",
    ],
    "offers.csv": [
        "offer_id", "is_active", "campaign_id", "name", "offer_type", "value", "min_total_spend_90d",
        "allowed_steps", "estimated_cost", "cooldown_days", "expires_in_days", "allowed_categories",
        "allowed_product_types", "target_scope", "created_at",
    ],
    "offer_assignments.csv": [
        "assignment_id", "user_id", "offer_id", "assigned_at", "expires_at", "reason", "is_active",
        "is_redeemed", "redeemed_transaction_id", "superseded_at", "superseded_by", "target",
    ],
    "offer_events.csv": [
        "offer_event_id", "assignment_id", "user_id", "offer_id", "campaign_name", "event_type", "event_key",
        "event_version", "created_at", "request_id", "context",
    ],
    "roadmap_events.csv": [
        "roadmap_event_id", "created_at", "user_id", "plan_id", "step_id", "event_type", "request_id", "context",
    ],
    "recommendation_events.csv": [
        "rec_event_id", "created_at", "user_id", "action", "page", "section_key", "request_id", "product_id",
        "algo_mode", "score", "components", "context",
    ],
}

OPTIONAL_EMPTY_FILES = [
    "campaign_budgets.csv",
    "offers.csv",
    "offer_assignments.csv",
    "offer_events.csv",
    "recommendation_events.csv",
]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _resolve_out_dir(raw_path: str) -> Path:
    candidate = Path(str(raw_path)).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    cwd_path = (Path.cwd() / candidate).resolve()
    if cwd_path.parent.exists():
        return cwd_path
    return (_repo_root() / candidate).resolve()


def _iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False)


def _decimal_str(value: Decimal | str | float | int) -> str:
    if isinstance(value, Decimal):
        return f"{value:.2f}"
    if isinstance(value, (int, float)):
        return f"{Decimal(str(value)):.2f}"
    return str(value)


def _normalize_step_status(value: str) -> str:
    token = str(value or "").strip().lower()
    return token or "missing"


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def _planned_target_step(scenario: dict[str, Any]) -> dict[str, Any]:
    for step in scenario["steps"]:
        if _normalize_step_status(step.get("status")) in {"recommended", "missing"}:
            return step
    return scenario["steps"][0]


def _product_rows(created_at: datetime, *, id_base: int) -> tuple[list[dict[str, Any]], dict[str, int]]:
    catalog = [
        ("repair_shampoo", "Repair Bond Shampoo", "shampoo", "12.90", ["damage", "dryness"], ["keratin", "panthenol"], [], {"hair_type": "wavy", "scalp_type": "oily", "hair_thickness": "medium"}, {"line": "repair", "finish": "soft"}, "water, keratin, panthenol, argan oil"),
        ("repair_conditioner", "Repair Bond Conditioner", "conditioner", "13.90", ["damage", "dryness", "smoothness"], ["keratin", "panthenol", "ceramide"], [], {"hair_type": "wavy", "scalp_type": "oily", "hair_thickness": "medium"}, {"line": "repair", "finish": "soft"}, "water, keratin, panthenol, ceramide np"),
        ("repair_mask", "Repair Rescue Hair Mask", "hair_mask", "18.50", ["damage", "dryness", "split_ends"], ["keratin", "amino_acids", "shea_butter"], [], {"hair_type": "wavy", "scalp_type": "normal", "hair_thickness": "medium"}, {"line": "repair", "finish": "rich"}, "water, shea butter, keratin, arginine"),
        ("hydrate_shampoo", "Hydra Balance Shampoo", "shampoo", "11.50", ["dryness", "frizz"], ["glycerin", "aloe"], [], {"hair_type": "curly", "scalp_type": "dry", "hair_thickness": "thick"}, {"line": "hydrate", "finish": "glossy"}, "water, glycerin, aloe vera, betaine"),
        ("hydrate_conditioner", "Hydra Balance Conditioner", "conditioner", "12.20", ["dryness", "frizz", "detangling"], ["glycerin", "aloe", "shea_butter"], [], {"hair_type": "curly", "scalp_type": "dry", "hair_thickness": "thick"}, {"line": "hydrate", "finish": "glossy"}, "water, glycerin, aloe vera, shea butter"),
        ("hydrate_mask", "Hydra Deep Hair Mask", "hair_mask", "17.80", ["dryness", "frizz", "definition"], ["glycerin", "shea_butter", "coconut_oil"], [], {"hair_type": "curly", "scalp_type": "dry", "hair_thickness": "thick"}, {"line": "hydrate", "finish": "glossy"}, "water, glycerin, shea butter, coconut oil"),
        ("smooth_shampoo", "Smooth Guard Shampoo", "shampoo", "10.90", ["frizz", "smoothness"], ["panthenol", "argan_oil"], [], {"hair_type": "straight", "scalp_type": "normal", "hair_thickness": "fine"}, {"line": "smooth", "finish": "sleek"}, "water, panthenol, argan oil, wheat protein"),
        ("smooth_conditioner", "Smooth Guard Conditioner", "conditioner", "11.90", ["frizz", "smoothness"], ["panthenol", "argan_oil", "amino_acids"], [], {"hair_type": "straight", "scalp_type": "normal", "hair_thickness": "fine"}, {"line": "smooth", "finish": "sleek"}, "water, panthenol, argan oil, amino acids"),
        ("smooth_conditioner_alt", "Smooth Guard Conditioner Plus", "conditioner", "12.40", ["frizz", "smoothness", "shine"], ["panthenol", "argan_oil", "amino_acids"], [], {"hair_type": "straight", "scalp_type": "normal", "hair_thickness": "fine"}, {"line": "smooth_plus", "finish": "sleek"}, "water, panthenol, argan oil, amino acids"),
        ("volume_shampoo", "Lift Volume Shampoo", "shampoo", "10.40", ["flatness", "oiliness"], ["biotin", "rice_protein"], ["sulfates"], {"hair_type": "straight", "scalp_type": "oily", "hair_thickness": "fine"}, {"line": "volume", "finish": "airy"}, "water, sodium laureth sulfate, biotin, rice protein"),
        ("volume_conditioner", "Lift Volume Conditioner", "conditioner", "11.40", ["flatness", "lightweight_care"], ["biotin", "rice_protein"], [], {"hair_type": "straight", "scalp_type": "oily", "hair_thickness": "fine"}, {"line": "volume", "finish": "airy"}, "water, biotin, rice protein, panthenol"),
        ("volume_mask", "Lift Volume Cloud Mask", "hair_mask", "16.10", ["flatness", "damage"], ["rice_protein", "panthenol"], [], {"hair_type": "straight", "scalp_type": "oily", "hair_thickness": "fine"}, {"line": "volume", "finish": "airy"}, "water, rice protein, panthenol, quinoa"),
        ("scalp_shampoo", "Scalp Reset Shampoo", "shampoo", "12.00", ["oiliness", "build_up", "flakes"], ["salicylic_acid", "niacinamide"], [], {"hair_type": "wavy", "scalp_type": "oily", "hair_thickness": "medium"}, {"line": "scalp", "finish": "fresh"}, "water, salicylic acid, niacinamide, zinc pca"),
        ("scalp_serum", "Scalp Reset Serum", "scalp_serum", "19.40", ["oiliness", "flakes", "itchiness"], ["salicylic_acid", "niacinamide", "tea_tree"], [], {"hair_type": "wavy", "scalp_type": "oily", "hair_thickness": "medium"}, {"line": "scalp", "finish": "fresh"}, "water, salicylic acid, niacinamide, tea tree oil"),
        ("scalp_serum_alt", "Scalp Reset Serum Plus", "scalp_serum", "20.10", ["oiliness", "flakes", "itchiness"], ["salicylic_acid", "niacinamide", "tea_tree"], [], {"hair_type": "wavy", "scalp_type": "oily", "hair_thickness": "medium"}, {"line": "scalp_plus", "finish": "fresh"}, "water, niacinamide, salicylic acid, tea tree extract"),
        ("curl_leave_in", "Curl Definition Leave-In", "leave_in", "15.80", ["frizz", "definition", "dryness"], ["glycerin", "aloe", "linseed"], [], {"hair_type": "curly", "scalp_type": "normal", "hair_thickness": "thick"}, {"line": "curl", "finish": "defined"}, "water, glycerin, aloe vera, linseed extract"),
        ("curl_leave_in_alt", "Curl Definition Leave-In Plus", "leave_in", "16.30", ["frizz", "definition", "dryness"], ["glycerin", "aloe", "linseed"], [], {"hair_type": "curly", "scalp_type": "normal", "hair_thickness": "thick"}, {"line": "curl_plus", "finish": "defined"}, "water, aloe vera, glycerin, linseed extract"),
        ("repair_hair_oil", "Repair Finish Hair Oil", "hair_oil", "17.20", ["dryness", "damage", "shine"], ["argan_oil", "ceramide", "squalane"], [], {"hair_type": "wavy", "scalp_type": "normal", "hair_thickness": "medium"}, {"line": "repair", "finish": "glossy"}, "cyclopentasiloxane, argan oil, squalane, ceramide np"),
        ("curl_hair_oil", "Curl Seal Hair Oil", "hair_oil", "16.90", ["frizz", "shine", "seal"], ["jojoba_oil", "argan_oil"], [], {"hair_type": "curly", "scalp_type": "normal", "hair_thickness": "thick"}, {"line": "curl", "finish": "glossy"}, "cyclopentasiloxane, jojoba oil, argan oil"),
    ]
    rows: list[dict[str, Any]] = []
    id_by_key: dict[str, int] = {}
    for offset, (key, name, product_type, price, concerns, actives, flags, attrs, raw_meta, inci) in enumerate(catalog, start=1):
        product_id = id_base + offset
        id_by_key[key] = product_id
        rows.append({
            "id": product_id,
            "name": name,
            "brand": "Scenario Lab",
            "price": price,
            "source_product_id": f"scenario::{key}",
            "currency": "KZT",
            "category": "haircare",
            "product_type": product_type,
            "concerns": _json(concerns),
            "attrs": _json(attrs),
            "step": "",
            "actives": _json(actives),
            "flags": _json(flags),
            "supported_skin_types": _json(["normal"]),
            "strength": "low",
            "in_stock": "true",
            "image_url": f"https://example.com/scenario/{key}.jpg",
            "image_urls": _json([f"https://example.com/scenario/{key}.jpg"]),
            "description": f"Synthetic scenario product for {key}.",
            "application_text": "Use as directed in the roadmap scenario.",
            "ingredients_inci": inci,
            "volume_raw": "250 ml",
            "raw_meta": _json(raw_meta),
            "created_at": _iso(created_at),
            "updated_at": _iso(created_at),
        })
    return rows, id_by_key


def _haircare_scenarios() -> list[dict[str, Any]]:
    return [
        {
            "slug": "repair_exact_match",
            "segment": "repair_exact",
            "expected_next_product_type": "conditioner",
            "outcome_tag": "completed_exact",
            "profile": {"skin_type": "normal", "goals": ["repair", "hydration"], "avoid_flags": [], "budget": "medium", "hair_profile": {"hair_type": "wavy", "scalp_type": "oily", "hair_thickness": "medium", "concerns": ["damage", "dryness"]}},
            "transactions": [{"offset_days": -24, "channel": "offline", "items": [("repair_shampoo", 1)]}, {"offset_days": -3, "channel": "offline", "items": [("repair_shampoo", 1)]}, {"offset_days": 3, "channel": "online", "items": [("repair_conditioner", 1)]}],
            "steps": [{"step_index": 1, "product_type": "shampoo", "status": "completed", "recommended_product_key": "repair_shampoo", "cadence": "weekly", "score": 0.98, "confidence": 0.96, "why": ["already_owned", "repair_anchor"]}, {"step_index": 2, "product_type": "conditioner", "status": "recommended", "recommended_product_key": "repair_conditioner", "cadence": "weekly", "score": 0.95, "confidence": 0.91, "why": ["follow_shampoo", "repair_profile_match"]}, {"step_index": 3, "product_type": "hair_mask", "status": "missing", "recommended_product_key": "repair_mask", "cadence": "optional", "score": 0.78, "confidence": 0.63, "why": ["post_conditioner_repair"]}],
            "events": [{"event_type": "roadmap_plan_refreshed", "offset_hours": -1, "step_index": None, "context": {}}, {"event_type": "roadmap_step_exposed", "offset_hours": 0, "step_index": 2, "context": {"category": "haircare", "sources": ["roadmap_api"]}}, {"event_type": "roadmap_step_clicked", "offset_hours": 4, "step_index": 2, "context": {"scenario_action": "open_pdp"}}, {"event_type": "roadmap_step_completed", "offset_hours": 72, "step_index": 2, "context": {"category": "haircare", "product_type": "conditioner", "matched_by": "recommended_product_id", "match_meta": {"recommended_product_key": "repair_conditioner", "purchased_product_key": "repair_conditioner"}}}],
        },
        {
            "slug": "hydration_to_mask",
            "segment": "hydrate_mask",
            "expected_next_product_type": "hair_mask",
            "outcome_tag": "completed_exact",
            "profile": {"skin_type": "normal", "goals": ["hydration", "definition"], "avoid_flags": [], "budget": "medium", "hair_profile": {"hair_type": "curly", "scalp_type": "dry", "hair_thickness": "thick", "concerns": ["dryness", "frizz"]}},
            "transactions": [{"offset_days": -21, "channel": "offline", "items": [("hydrate_shampoo", 1)]}, {"offset_days": -10, "channel": "offline", "items": [("hydrate_conditioner", 1)]}, {"offset_days": 5, "channel": "online", "items": [("hydrate_mask", 1)]}],
            "steps": [{"step_index": 1, "product_type": "shampoo", "status": "completed", "recommended_product_key": "hydrate_shampoo", "cadence": "weekly", "score": 0.97, "confidence": 0.94, "why": ["already_owned", "hydrate_anchor"]}, {"step_index": 2, "product_type": "conditioner", "status": "completed", "recommended_product_key": "hydrate_conditioner", "cadence": "weekly", "score": 0.94, "confidence": 0.89, "why": ["already_owned", "hydrate_support"]}, {"step_index": 3, "product_type": "hair_mask", "status": "recommended", "recommended_product_key": "hydrate_mask", "cadence": "optional", "score": 0.92, "confidence": 0.86, "why": ["deep_hydration", "after_conditioner"]}, {"step_index": 4, "product_type": "leave_in", "status": "missing", "recommended_product_key": "curl_leave_in", "cadence": "optional", "score": 0.71, "confidence": 0.58, "why": ["future_definition_step"]}],
            "events": [{"event_type": "roadmap_plan_refreshed", "offset_hours": -1, "step_index": None, "context": {}}, {"event_type": "roadmap_step_exposed", "offset_hours": 0, "step_index": 3, "context": {"category": "haircare", "sources": ["roadmap_api"]}}, {"event_type": "roadmap_step_clicked", "offset_hours": 6, "step_index": 3, "context": {"scenario_action": "open_bundle"}}, {"event_type": "roadmap_step_completed", "offset_hours": 120, "step_index": 3, "context": {"category": "haircare", "product_type": "hair_mask", "matched_by": "recommended_product_id", "match_meta": {"recommended_product_key": "hydrate_mask", "purchased_product_key": "hydrate_mask"}}}],
        },
        {
            "slug": "semantic_conditioner_match",
            "segment": "semantic_conditioner",
            "expected_next_product_type": "conditioner",
            "outcome_tag": "completed_semantic",
            "profile": {"skin_type": "normal", "goals": ["smoothness", "repair"], "avoid_flags": [], "budget": "medium", "hair_profile": {"hair_type": "straight", "scalp_type": "normal", "hair_thickness": "fine", "concerns": ["frizz", "shine"]}},
            "transactions": [{"offset_days": -17, "channel": "offline", "items": [("smooth_shampoo", 1)]}, {"offset_days": 4, "channel": "online", "items": [("smooth_conditioner_alt", 1)]}],
            "steps": [{"step_index": 1, "product_type": "shampoo", "status": "completed", "recommended_product_key": "smooth_shampoo", "cadence": "weekly", "score": 0.95, "confidence": 0.90, "why": ["already_owned", "frizz_anchor"]}, {"step_index": 2, "product_type": "conditioner", "status": "recommended", "recommended_product_key": "smooth_conditioner", "cadence": "weekly", "score": 0.91, "confidence": 0.84, "why": ["follow_shampoo", "semantic_neighbor"]}, {"step_index": 3, "product_type": "hair_mask", "status": "missing", "recommended_product_key": "repair_mask", "cadence": "optional", "score": 0.68, "confidence": 0.55, "why": ["optional_repair"]}],
            "events": [{"event_type": "roadmap_plan_refreshed", "offset_hours": -1, "step_index": None, "context": {}}, {"event_type": "roadmap_step_exposed", "offset_hours": 0, "step_index": 2, "context": {"category": "haircare", "sources": ["roadmap_api"]}}, {"event_type": "roadmap_step_clicked", "offset_hours": 3, "step_index": 2, "context": {"scenario_action": "compare_variants"}}, {"event_type": "roadmap_step_completed", "offset_hours": 96, "step_index": 2, "context": {"category": "haircare", "product_type": "conditioner", "matched_by": "semantic_content_match", "match_meta": {"recommended_product_key": "smooth_conditioner", "purchased_product_key": "smooth_conditioner_alt", "semantic_score": 1.37}}}],
        },
        {
            "slug": "guardrail_skip_repeat_shampoo",
            "segment": "guardrail_skip",
            "expected_next_product_type": "hair_mask",
            "outcome_tag": "no_conversion",
            "profile": {"skin_type": "normal", "goals": ["volume", "repair"], "avoid_flags": ["sulfates"], "budget": "low", "hair_profile": {"hair_type": "straight", "scalp_type": "oily", "hair_thickness": "fine", "concerns": ["flatness", "oiliness"]}},
            "transactions": [{"offset_days": -15, "channel": "offline", "items": [("volume_shampoo", 1)]}, {"offset_days": -6, "channel": "offline", "items": [("volume_conditioner", 1)]}],
            "steps": [{"step_index": 1, "product_type": "shampoo", "status": "completed", "recommended_product_key": "volume_shampoo", "cadence": "weekly", "score": 0.93, "confidence": 0.88, "why": ["already_owned", "volume_anchor"]}, {"step_index": 2, "product_type": "conditioner", "status": "completed", "recommended_product_key": "volume_conditioner", "cadence": "weekly", "score": 0.90, "confidence": 0.81, "why": ["already_owned", "lightweight_support"]}, {"step_index": 3, "product_type": "hair_mask", "status": "recommended", "recommended_product_key": "volume_mask", "cadence": "optional", "score": 0.76, "confidence": 0.59, "why": ["repair_boost", "post_conditioner"]}],
            "events": [{"event_type": "roadmap_plan_refreshed", "offset_hours": -1, "step_index": None, "context": {}}, {"event_type": "roadmap_step_exposed", "offset_hours": 0, "step_index": 3, "context": {"category": "haircare", "sources": ["roadmap_api"]}}, {"event_type": "roadmap_step_skipped", "offset_hours": 48, "step_index": 3, "context": {"category": "haircare", "reason": "not_ready"}}],
        },
        {
            "slug": "scalp_serum_progression",
            "segment": "scalp_serum",
            "expected_next_product_type": "scalp_serum",
            "outcome_tag": "completed_exact",
            "profile": {"skin_type": "normal", "goals": ["scalp_balance"], "avoid_flags": [], "budget": "medium", "hair_profile": {"hair_type": "wavy", "scalp_type": "oily", "hair_thickness": "medium", "concerns": ["flakes", "oiliness"]}},
            "transactions": [{"offset_days": -16, "channel": "offline", "items": [("scalp_shampoo", 1)]}, {"offset_days": 2, "channel": "online", "items": [("scalp_serum", 1)]}],
            "steps": [{"step_index": 1, "product_type": "shampoo", "status": "completed", "recommended_product_key": "scalp_shampoo", "cadence": "weekly", "score": 0.96, "confidence": 0.91, "why": ["already_owned", "scalp_anchor"]}, {"step_index": 2, "product_type": "scalp_serum", "status": "recommended", "recommended_product_key": "scalp_serum", "cadence": "daily", "score": 0.93, "confidence": 0.87, "why": ["scalp_progression", "concern_match"]}, {"step_index": 3, "product_type": "conditioner", "status": "missing", "recommended_product_key": "repair_conditioner", "cadence": "optional", "score": 0.52, "confidence": 0.41, "why": ["optional_lengths_support"]}],
            "events": [{"event_type": "roadmap_plan_refreshed", "offset_hours": -1, "step_index": None, "context": {}}, {"event_type": "roadmap_step_exposed", "offset_hours": 0, "step_index": 2, "context": {"category": "haircare", "sources": ["roadmap_api"]}}, {"event_type": "roadmap_step_clicked", "offset_hours": 2, "step_index": 2, "context": {"scenario_action": "read_benefits"}}, {"event_type": "roadmap_step_completed", "offset_hours": 48, "step_index": 2, "context": {"category": "haircare", "product_type": "scalp_serum", "matched_by": "recommended_product_id", "match_meta": {"recommended_product_key": "scalp_serum", "purchased_product_key": "scalp_serum"}}}],
        },
        {
            "slug": "leave_in_after_mask",
            "segment": "leave_in_progression",
            "expected_next_product_type": "leave_in",
            "outcome_tag": "completed_exact",
            "profile": {"skin_type": "normal", "goals": ["definition", "hydration"], "avoid_flags": [], "budget": "high", "hair_profile": {"hair_type": "curly", "scalp_type": "normal", "hair_thickness": "thick", "concerns": ["frizz", "definition"]}},
            "transactions": [{"offset_days": -28, "channel": "offline", "items": [("hydrate_shampoo", 1)]}, {"offset_days": -20, "channel": "offline", "items": [("hydrate_conditioner", 1)]}, {"offset_days": -8, "channel": "offline", "items": [("hydrate_mask", 1)]}, {"offset_days": 4, "channel": "online", "items": [("curl_leave_in", 1)]}],
            "steps": [{"step_index": 1, "product_type": "shampoo", "status": "completed", "recommended_product_key": "hydrate_shampoo", "cadence": "weekly", "score": 0.96, "confidence": 0.93, "why": ["already_owned", "curly_anchor"]}, {"step_index": 2, "product_type": "conditioner", "status": "completed", "recommended_product_key": "hydrate_conditioner", "cadence": "weekly", "score": 0.94, "confidence": 0.90, "why": ["already_owned", "curly_support"]}, {"step_index": 3, "product_type": "hair_mask", "status": "completed", "recommended_product_key": "hydrate_mask", "cadence": "optional", "score": 0.91, "confidence": 0.86, "why": ["already_owned", "deep_hydration"]}, {"step_index": 4, "product_type": "leave_in", "status": "recommended", "recommended_product_key": "curl_leave_in", "cadence": "daily", "score": 0.89, "confidence": 0.82, "why": ["post_mask_definition", "curl_profile_match"]}],
            "events": [{"event_type": "roadmap_plan_refreshed", "offset_hours": -1, "step_index": None, "context": {}}, {"event_type": "roadmap_step_exposed", "offset_hours": 0, "step_index": 4, "context": {"category": "haircare", "sources": ["roadmap_api"]}}, {"event_type": "roadmap_step_clicked", "offset_hours": 3, "step_index": 4, "context": {"scenario_action": "view_routine"}}, {"event_type": "roadmap_step_completed", "offset_hours": 96, "step_index": 4, "context": {"category": "haircare", "product_type": "leave_in", "matched_by": "recommended_product_id", "match_meta": {"recommended_product_key": "curl_leave_in", "purchased_product_key": "curl_leave_in"}}}],
        },
    ]


def _haircare_hardcases_v2_scenarios() -> list[dict[str, Any]]:
    return [
        {
            "slug": "leave_in_after_mask_exact_v2",
            "segment": "leave_in_exact_v2",
            "expected_next_product_type": "leave_in",
            "outcome_tag": "completed_exact",
            "profile": {"skin_type": "normal", "goals": ["definition", "frizz_control"], "avoid_flags": ["heavy_oils"], "budget": "high", "hair_profile": {"hair_type": "curly", "scalp_type": "normal", "hair_thickness": "thick", "concerns": ["frizz", "definition", "dryness"]}},
            "transactions": [{"offset_days": -29, "channel": "offline", "items": [("hydrate_shampoo", 1)]}, {"offset_days": -20, "channel": "offline", "items": [("hydrate_conditioner", 1)]}, {"offset_days": -7, "channel": "offline", "items": [("hydrate_mask", 1)]}, {"offset_days": 4, "channel": "online", "items": [("curl_leave_in", 1)]}],
            "steps": [{"step_index": 1, "product_type": "shampoo", "status": "completed", "recommended_product_key": "hydrate_shampoo", "cadence": "weekly", "score": 0.97, "confidence": 0.94, "why": ["already_owned", "curl_anchor"]}, {"step_index": 2, "product_type": "conditioner", "status": "completed", "recommended_product_key": "hydrate_conditioner", "cadence": "weekly", "score": 0.95, "confidence": 0.90, "why": ["already_owned", "curl_support"]}, {"step_index": 3, "product_type": "hair_mask", "status": "completed", "recommended_product_key": "hydrate_mask", "cadence": "weekly", "score": 0.92, "confidence": 0.86, "why": ["already_owned", "deep_hydration"]}, {"step_index": 4, "product_type": "leave_in", "status": "recommended", "recommended_product_key": "curl_leave_in", "cadence": "daily", "score": 0.91, "confidence": 0.85, "why": ["post_mask_definition", "avoid_heavy_oils"]}],
            "events": [{"event_type": "roadmap_plan_refreshed", "offset_hours": -1, "step_index": None, "context": {}}, {"event_type": "roadmap_step_exposed", "offset_hours": 0, "step_index": 4, "context": {"category": "haircare", "sources": ["roadmap_api"]}}, {"event_type": "roadmap_step_clicked", "offset_hours": 2, "step_index": 4, "context": {"scenario_action": "view_leave_in_routine"}}, {"event_type": "roadmap_step_completed", "offset_hours": 96, "step_index": 4, "context": {"category": "haircare", "product_type": "leave_in", "matched_by": "recommended_product_id", "match_meta": {"recommended_product_key": "curl_leave_in", "purchased_product_key": "curl_leave_in"}}}],
        },
        {
            "slug": "leave_in_semantic_match_v2",
            "segment": "leave_in_semantic_v2",
            "expected_next_product_type": "leave_in",
            "outcome_tag": "completed_semantic",
            "profile": {"skin_type": "normal", "goals": ["definition", "frizz_control"], "avoid_flags": ["heavy_oils"], "budget": "high", "hair_profile": {"hair_type": "curly", "scalp_type": "normal", "hair_thickness": "thick", "concerns": ["frizz", "definition"]}},
            "transactions": [{"offset_days": -26, "channel": "offline", "items": [("hydrate_shampoo", 1)]}, {"offset_days": -18, "channel": "offline", "items": [("hydrate_conditioner", 1)]}, {"offset_days": -6, "channel": "offline", "items": [("hydrate_mask", 1)]}, {"offset_days": 3, "channel": "online", "items": [("curl_leave_in_alt", 1)]}],
            "steps": [{"step_index": 1, "product_type": "shampoo", "status": "completed", "recommended_product_key": "hydrate_shampoo", "cadence": "weekly", "score": 0.96, "confidence": 0.92, "why": ["already_owned", "curl_anchor"]}, {"step_index": 2, "product_type": "conditioner", "status": "completed", "recommended_product_key": "hydrate_conditioner", "cadence": "weekly", "score": 0.94, "confidence": 0.89, "why": ["already_owned", "curl_support"]}, {"step_index": 3, "product_type": "hair_mask", "status": "completed", "recommended_product_key": "hydrate_mask", "cadence": "weekly", "score": 0.90, "confidence": 0.84, "why": ["already_owned", "deep_hydration"]}, {"step_index": 4, "product_type": "leave_in", "status": "recommended", "recommended_product_key": "curl_leave_in", "cadence": "daily", "score": 0.89, "confidence": 0.82, "why": ["semantic_neighbor", "curl_profile_match"]}],
            "events": [{"event_type": "roadmap_plan_refreshed", "offset_hours": -1, "step_index": None, "context": {}}, {"event_type": "roadmap_step_exposed", "offset_hours": 0, "step_index": 4, "context": {"category": "haircare", "sources": ["roadmap_api"]}}, {"event_type": "roadmap_step_clicked", "offset_hours": 3, "step_index": 4, "context": {"scenario_action": "compare_textures"}}, {"event_type": "roadmap_step_completed", "offset_hours": 72, "step_index": 4, "context": {"category": "haircare", "product_type": "leave_in", "matched_by": "semantic_content_match", "match_meta": {"recommended_product_key": "curl_leave_in", "purchased_product_key": "curl_leave_in_alt", "semantic_score": 1.42}}}],
        },
        {
            "slug": "hair_oil_finish_control_v2",
            "segment": "hair_oil_control_v2",
            "expected_next_product_type": "hair_oil",
            "outcome_tag": "completed_exact",
            "profile": {"skin_type": "normal", "goals": ["shine", "seal_damage"], "avoid_flags": [], "budget": "high", "hair_profile": {"hair_type": "wavy", "scalp_type": "normal", "hair_thickness": "medium", "concerns": ["dryness", "damage", "shine"]}},
            "transactions": [{"offset_days": -24, "channel": "offline", "items": [("repair_shampoo", 1)]}, {"offset_days": -15, "channel": "offline", "items": [("repair_conditioner", 1)]}, {"offset_days": -6, "channel": "offline", "items": [("repair_mask", 1)]}, {"offset_days": 3, "channel": "online", "items": [("repair_hair_oil", 1)]}],
            "steps": [{"step_index": 1, "product_type": "shampoo", "status": "completed", "recommended_product_key": "repair_shampoo", "cadence": "weekly", "score": 0.97, "confidence": 0.94, "why": ["already_owned", "repair_anchor"]}, {"step_index": 2, "product_type": "conditioner", "status": "completed", "recommended_product_key": "repair_conditioner", "cadence": "weekly", "score": 0.95, "confidence": 0.90, "why": ["already_owned", "repair_support"]}, {"step_index": 3, "product_type": "hair_mask", "status": "completed", "recommended_product_key": "repair_mask", "cadence": "weekly", "score": 0.92, "confidence": 0.86, "why": ["already_owned", "repair_boost"]}, {"step_index": 4, "product_type": "hair_oil", "status": "recommended", "recommended_product_key": "repair_hair_oil", "cadence": "optional", "score": 0.90, "confidence": 0.83, "why": ["seal_in_benefits", "high_shine_goal"]}],
            "events": [{"event_type": "roadmap_plan_refreshed", "offset_hours": -1, "step_index": None, "context": {}}, {"event_type": "roadmap_step_exposed", "offset_hours": 0, "step_index": 4, "context": {"category": "haircare", "sources": ["roadmap_api"]}}, {"event_type": "roadmap_step_clicked", "offset_hours": 5, "step_index": 4, "context": {"scenario_action": "view_finish_step"}}, {"event_type": "roadmap_step_completed", "offset_hours": 72, "step_index": 4, "context": {"category": "haircare", "product_type": "hair_oil", "matched_by": "recommended_product_id", "match_meta": {"recommended_product_key": "repair_hair_oil", "purchased_product_key": "repair_hair_oil"}}}],
        },
        {
            "slug": "scalp_serum_progression_strong_v2",
            "segment": "scalp_serum_exact_v2",
            "expected_next_product_type": "scalp_serum",
            "outcome_tag": "completed_exact",
            "profile": {"skin_type": "normal", "goals": ["scalp_balance"], "avoid_flags": [], "budget": "medium", "hair_profile": {"hair_type": "wavy", "scalp_type": "oily", "hair_thickness": "medium", "concerns": ["flakes", "itchiness", "oiliness"]}},
            "transactions": [{"offset_days": -18, "channel": "offline", "items": [("scalp_shampoo", 1)]}, {"offset_days": -5, "channel": "offline", "items": [("scalp_shampoo", 1)]}, {"offset_days": 2, "channel": "online", "items": [("scalp_serum", 1)]}],
            "steps": [{"step_index": 1, "product_type": "shampoo", "status": "completed", "recommended_product_key": "scalp_shampoo", "cadence": "weekly", "score": 0.97, "confidence": 0.93, "why": ["already_owned", "scalp_anchor"]}, {"step_index": 2, "product_type": "scalp_serum", "status": "recommended", "recommended_product_key": "scalp_serum", "cadence": "daily", "score": 0.95, "confidence": 0.89, "why": ["persistent_scalp_concerns", "after_reset_shampoo"]}, {"step_index": 3, "product_type": "conditioner", "status": "missing", "recommended_product_key": "repair_conditioner", "cadence": "optional", "score": 0.48, "confidence": 0.39, "why": ["optional_lengths_support"]}],
            "events": [{"event_type": "roadmap_plan_refreshed", "offset_hours": -1, "step_index": None, "context": {}}, {"event_type": "roadmap_step_exposed", "offset_hours": 0, "step_index": 2, "context": {"category": "haircare", "sources": ["roadmap_api"]}}, {"event_type": "roadmap_step_clicked", "offset_hours": 2, "step_index": 2, "context": {"scenario_action": "read_scalp_benefits"}}, {"event_type": "roadmap_step_completed", "offset_hours": 48, "step_index": 2, "context": {"category": "haircare", "product_type": "scalp_serum", "matched_by": "recommended_product_id", "match_meta": {"recommended_product_key": "scalp_serum", "purchased_product_key": "scalp_serum"}}}],
        },
        {
            "slug": "conditioner_after_scalp_reset_control_v2",
            "segment": "conditioner_control_v2",
            "expected_next_product_type": "conditioner",
            "outcome_tag": "completed_exact",
            "profile": {"skin_type": "normal", "goals": ["repair_lengths"], "avoid_flags": [], "budget": "medium", "hair_profile": {"hair_type": "wavy", "scalp_type": "oily", "hair_thickness": "medium", "concerns": ["damage", "dryness"]}},
            "transactions": [{"offset_days": -14, "channel": "offline", "items": [("scalp_shampoo", 1)]}, {"offset_days": 3, "channel": "online", "items": [("repair_conditioner", 1)]}],
            "steps": [{"step_index": 1, "product_type": "shampoo", "status": "completed", "recommended_product_key": "scalp_shampoo", "cadence": "weekly", "score": 0.95, "confidence": 0.91, "why": ["already_owned", "scalp_reset_anchor"]}, {"step_index": 2, "product_type": "conditioner", "status": "recommended", "recommended_product_key": "repair_conditioner", "cadence": "weekly", "score": 0.88, "confidence": 0.80, "why": ["lengths_repair", "dryness_match"]}, {"step_index": 3, "product_type": "scalp_serum", "status": "missing", "recommended_product_key": "scalp_serum_alt", "cadence": "optional", "score": 0.52, "confidence": 0.43, "why": ["secondary_scalp_option"]}],
            "events": [{"event_type": "roadmap_plan_refreshed", "offset_hours": -1, "step_index": None, "context": {}}, {"event_type": "roadmap_step_exposed", "offset_hours": 0, "step_index": 2, "context": {"category": "haircare", "sources": ["roadmap_api"]}}, {"event_type": "roadmap_step_clicked", "offset_hours": 3, "step_index": 2, "context": {"scenario_action": "open_conditioner_pdp"}}, {"event_type": "roadmap_step_completed", "offset_hours": 72, "step_index": 2, "context": {"category": "haircare", "product_type": "conditioner", "matched_by": "recommended_product_id", "match_meta": {"recommended_product_key": "repair_conditioner", "purchased_product_key": "repair_conditioner"}}}],
        },
    ]


def _scenario_sets() -> dict[str, list[dict[str, Any]]]:
    return {
        SCENARIO_SET_HAIRCARE_V1: _haircare_scenarios(),
        SCENARIO_SET_HAIRCARE_HARDCASES_V2: _haircare_hardcases_v2_scenarios(),
    }


class Command(BaseCommand):
    help = "Generate import-compatible synthetic roadmap scenario pack with deterministic logical cases."

    def add_arguments(self, parser):
        parser.add_argument("--out-dir", type=str, default="data/generated/roadmap_scenario_pack_haircare_v1")
        parser.add_argument("--scenario-set", type=str, default=SCENARIO_SET_HAIRCARE_V1)
        parser.add_argument("--replicas", type=int, default=1)
        parser.add_argument("--days-ago-start", type=int, default=75)
        parser.add_argument("--id-base", type=int, default=900000)

    def handle(self, *args, **options):
        scenario_set_name = str(options.get("scenario_set") or SCENARIO_SET_HAIRCARE_V1).strip().lower()
        replicas = int(options.get("replicas") or 1)
        days_ago_start = int(options.get("days_ago_start") or 75)
        id_base = int(options.get("id_base") or 900000)
        if scenario_set_name not in _scenario_sets():
            raise CommandError(f"Unknown --scenario-set: {scenario_set_name}")
        if replicas <= 0:
            raise CommandError("--replicas must be > 0")
        if days_ago_start <= 0:
            raise CommandError("--days-ago-start must be > 0")
        if id_base <= 0:
            raise CommandError("--id-base must be > 0")

        out_dir = _resolve_out_dir(str(options.get("out_dir") or ""))
        out_dir.mkdir(parents=True, exist_ok=True)
        scenarios = list(_scenario_sets()[scenario_set_name])
        now_utc = datetime.now(timezone.utc).replace(microsecond=0)
        base_t0 = now_utc - timedelta(days=days_ago_start)
        product_rows, product_id_by_key = _product_rows(base_t0 - timedelta(days=60), id_base=id_base)
        product_price_by_id = {int(row["id"]): Decimal(str(row["price"])) for row in product_rows}

        file_rows: dict[str, list[dict[str, Any]]] = {name: [] for name in CSV_HEADERS}
        file_rows["products.csv"] = product_rows

        next_user_id = id_base + 10000
        next_tx_id = id_base + 20000
        next_tx_item_id = id_base + 30000
        next_owned_id = id_base + 40000
        next_plan_id = id_base + 50000
        next_step_id = id_base + 60000
        next_roadmap_event_id = id_base + 70000

        owned_agg: dict[tuple[int, int], dict[str, Any]] = {}
        expected_transition_counts: Counter[str] = Counter()
        outcome_tag_counts: Counter[str] = Counter()
        scenario_instances: list[dict[str, Any]] = []

        for replica_idx in range(replicas):
            for scenario_idx, scenario in enumerate(scenarios, start=1):
                t0 = base_t0 + timedelta(days=(replica_idx * 14) + ((scenario_idx - 1) * 5))
                user_id = next_user_id
                next_user_id += 1
                plan_id = next_plan_id
                next_plan_id += 1
                username = f"scenario_{scenario['slug']}_r{replica_idx + 1}"
                profile = dict(scenario.get("profile") or {})
                planned_step = _planned_target_step(scenario)
                expected_next = str(scenario.get("expected_next_product_type") or "").strip().lower()

                file_rows["users.csv"].append({
                    "user_id": user_id,
                    "username": username,
                    "segment": str(scenario.get("segment") or scenario["slug"]),
                    "favorite_category": "haircare",
                    "created_at": _iso(t0 - timedelta(days=45)),
                })
                file_rows["customer_profiles.csv"].append({
                    "user_id": user_id,
                    "skin_type": str(profile.get("skin_type") or "normal"),
                    "goals": _json(profile.get("goals") or []),
                    "avoid_flags": _json(profile.get("avoid_flags") or []),
                    "budget": str(profile.get("budget") or "medium"),
                    "hair_profile": _json(profile.get("hair_profile") or {}),
                    "makeup_profile": _json(profile.get("makeup_profile") or {}),
                    "fragrance_profile": _json(profile.get("fragrance_profile") or {}),
                })

                step_id_by_index: dict[int, int] = {}
                latest_product_times: list[tuple[datetime, int]] = []

                for tx_spec in scenario["transactions"]:
                    tx_time = t0 + timedelta(days=float(tx_spec.get("offset_days") or 0))
                    transaction_id = next_tx_id
                    next_tx_id += 1
                    total = Decimal("0")
                    for product_key, quantity in list(tx_spec.get("items") or []):
                        product_id = int(product_id_by_key[str(product_key)])
                        unit_price = Decimal(product_price_by_id[product_id])
                        total += unit_price * int(quantity)
                        file_rows["transaction_items.csv"].append({
                            "transaction_item_id": next_tx_item_id,
                            "transaction_id": transaction_id,
                            "user_id": user_id,
                            "product_id": product_id,
                            "quantity": int(quantity),
                            "unit_price": _decimal_str(unit_price),
                        })
                        next_tx_item_id += 1
                        if tx_time <= t0:
                            latest_product_times.append((tx_time, product_id))
                        aggregate = owned_agg.setdefault((user_id, product_id), {
                            "quantity_total": 0,
                            "acquired_at": tx_time,
                            "last_acquired_at": tx_time,
                        })
                        aggregate["quantity_total"] = int(aggregate["quantity_total"]) + int(quantity)
                        if tx_time < aggregate["acquired_at"]:
                            aggregate["acquired_at"] = tx_time
                        if tx_time > aggregate["last_acquired_at"]:
                            aggregate["last_acquired_at"] = tx_time

                    file_rows["transactions.csv"].append({
                        "transaction_id": transaction_id,
                        "user_id": user_id,
                        "created_at": _iso(tx_time),
                        "total_amount": _decimal_str(total),
                        "channel": str(tx_spec.get("channel") or "offline"),
                        "idempotency_key": f"scenario::{scenario['slug']}::r{replica_idx + 1}::tx{transaction_id}",
                        "pricing_meta": _json({"source": "generate_roadmap_scenario_pack", "scenario_key": scenario["slug"], "replica": replica_idx + 1}),
                    })

                latest_context_product_ids = [product_id for _ts, product_id in sorted(latest_product_times)[-3:]]
                plan_meta = {
                    "source": "synthetic_scenario_pack",
                    "scenario_set": scenario_set_name,
                    "scenario_key": scenario["slug"],
                    "scenario_description": str(scenario.get("description") or ""),
                    "context": {"post_ctx_product_ids": latest_context_product_ids},
                    "ml": {
                        "mode": "v4_ranking",
                        "decision": "model_used",
                        "model_path": "synthetic://scenario_pack",
                        "model_version": f"synthetic_{scenario_set_name}",
                        "model_slot": "active",
                        "selected_feature_set": "full",
                        "planned_target_product_type": str(planned_step.get("product_type") or ""),
                        "planned_target_step_index": int(planned_step.get("step_index") or 0),
                        "predictions": [{"candidate_type": expected_next, "score": 0.93}],
                    },
                }
                file_rows["roadmap_plans.csv"].append({
                    "plan_id": plan_id,
                    "user_id": user_id,
                    "category": "haircare",
                    "is_active": "true",
                    "version": 1,
                    "meta": _json(plan_meta),
                    "created_at": _iso(t0 - timedelta(days=2)),
                    "updated_at": _iso(t0),
                })

                for step in scenario["steps"]:
                    step_id = next_step_id
                    next_step_id += 1
                    step_index = int(step["step_index"])
                    step_id_by_index[step_index] = step_id
                    rec_key = str(step.get("recommended_product_key") or "")
                    rec_id = product_id_by_key.get(rec_key)
                    file_rows["roadmap_steps.csv"].append({
                        "step_id": step_id,
                        "plan_id": plan_id,
                        "step_index": step_index,
                        "product_type": str(step.get("product_type") or ""),
                        "status": _normalize_step_status(str(step.get("status") or "")),
                        "recommended_product_id": rec_id or "",
                        "suggestions": _json(([{"product_id": rec_id, "score": float(step.get("score") or 0.0)}] if rec_id else [])),
                        "score": f"{float(step.get('score') or 0.0):.4f}",
                        "confidence": f"{float(step.get('confidence') or 0.0):.4f}",
                        "why": _json(step.get("why") or []),
                        "cadence": str(step.get("cadence") or ""),
                        "created_at": _iso(t0 - timedelta(days=1)),
                        "updated_at": _iso(t0),
                    })

                for event_spec in scenario["events"]:
                    step_id = step_id_by_index.get(int(event_spec["step_index"])) if event_spec.get("step_index") is not None else None
                    created_at = t0 + timedelta(hours=float(event_spec.get("offset_hours") or 0))
                    context = dict(event_spec.get("context") or {})
                    context["scenario_key"] = scenario["slug"]
                    context["scenario_set"] = scenario_set_name
                    if event_spec["event_type"] == "roadmap_step_completed":
                        match_meta = dict(context.get("match_meta") or {})
                        rec_key = str(match_meta.pop("recommended_product_key", "") or "")
                        pur_key = str(match_meta.pop("purchased_product_key", "") or "")
                        if rec_key:
                            match_meta["recommended_product_id"] = int(product_id_by_key[rec_key])
                        if pur_key:
                            match_meta["purchased_product_id"] = int(product_id_by_key[pur_key])
                        context["match_meta"] = match_meta
                    file_rows["roadmap_events.csv"].append({
                        "roadmap_event_id": next_roadmap_event_id,
                        "created_at": _iso(created_at),
                        "user_id": user_id,
                        "plan_id": plan_id,
                        "step_id": step_id or "",
                        "event_type": str(event_spec["event_type"]),
                        "request_id": f"scenario::{scenario['slug']}::r{replica_idx + 1}::evt{next_roadmap_event_id}",
                        "context": _json(context),
                    })
                    next_roadmap_event_id += 1

                expected_transition_counts[expected_next] += 1
                outcome_tag_counts[str(scenario.get("outcome_tag") or "unknown")] += 1
                scenario_instances.append({
                    "scenario_key": scenario["slug"],
                    "replica": replica_idx + 1,
                    "username": username,
                    "user_id": user_id,
                    "plan_id": plan_id,
                    "t0_utc": _iso(t0),
                    "expected_next_product_type": expected_next,
                    "planned_target_product_type": str(planned_step.get("product_type") or ""),
                    "planned_target_step_index": int(planned_step.get("step_index") or 0),
                    "outcome_tag": str(scenario.get("outcome_tag") or "unknown"),
                })

        for (user_id, product_id), aggregate in sorted(owned_agg.items(), key=lambda item: (item[0][0], item[0][1])):
            file_rows["owned_products.csv"].append({
                "owned_product_id": next_owned_id,
                "user_id": user_id,
                "product_id": product_id,
                "quantity_total": int(aggregate["quantity_total"]),
                "is_active": "true",
                "last_acquired_at": _iso(aggregate["last_acquired_at"]),
                "acquired_at": _iso(aggregate["acquired_at"]),
                "source": "scenario_pack",
            })
            next_owned_id += 1

        for file_name in OPTIONAL_EMPTY_FILES:
            file_rows[file_name] = []

        for file_name, headers in CSV_HEADERS.items():
            _write_csv(out_dir / file_name, headers, list(file_rows.get(file_name) or []))

        summary = {
            "generated_at_utc": _iso(now_utc),
            "scenario_set": scenario_set_name,
            "replicas": replicas,
            "users_count": len(file_rows["users.csv"]),
            "products_count": len(file_rows["products.csv"]),
            "transactions_count": len(file_rows["transactions.csv"]),
            "transaction_items_count": len(file_rows["transaction_items.csv"]),
            "owned_products_count": len(file_rows["owned_products.csv"]),
            "roadmap_plans_count": len(file_rows["roadmap_plans.csv"]),
            "roadmap_steps_count": len(file_rows["roadmap_steps.csv"]),
            "roadmap_events_count": len(file_rows["roadmap_events.csv"]),
            "expected_next_distribution": {str(k): int(v) for k, v in sorted(expected_transition_counts.items())},
            "outcome_tag_distribution": {str(k): int(v) for k, v in sorted(outcome_tag_counts.items())},
            "scenario_instances": scenario_instances,
        }
        (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        (out_dir / "README.md").write_text(
            "\n".join([
                f"Synthetic roadmap scenario pack: {scenario_set_name}",
                "",
                "Validate:",
                f"python manage.py import_synth_dataset --path {out_dir} --dry-run",
                "",
                "Import into disposable DB:",
                f"python manage.py import_synth_dataset --path {out_dir} --truncate --i-understand-its-destructive",
                "",
                f"Expected next distribution: {summary['expected_next_distribution']}",
                f"Outcome tags: {summary['outcome_tag_distribution']}",
            ]),
            encoding="utf-8",
        )

        self.stdout.write(f"[generate_roadmap_scenario_pack] out_dir={out_dir}")
        self.stdout.write(
            "[generate_roadmap_scenario_pack] "
            f"users={summary['users_count']} tx={summary['transactions_count']} roadmap_events={summary['roadmap_events_count']}"
        )
        self.stdout.write(
            "[generate_roadmap_scenario_pack] "
            f"expected_next_distribution={summary['expected_next_distribution']}"
        )
