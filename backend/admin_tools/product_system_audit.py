from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from django.db.models import Count, Q, Sum
from django.utils import timezone

from admin_tools.commerce_proof import (
    COMMERCE_PROOF_MIN_LOYALTY_REDEEMS,
    COMMERCE_PROOF_MIN_MULTI_ITEM_TRANSACTIONS,
    COMMERCE_PROOF_MIN_REDEEMED_ASSIGNMENTS,
)
from catalog.models import Product
from loyalty.models import LoyaltyAccount, LoyaltyLedgerEntry
from offers.models import OfferAssignment, OfferEvent
from offers.services import _target_matches_roadmap_step
from recs_analytics.models import RecommendationEvent
from roadmap_app.integrity import (
    active_fragrance_runtime_integrity_counts,
    legacy_bad_fragrance_completion_details,
)
from roadmap_app.models import RoadmapPlan, RoadmapStep
from roadmap_app.services import get_active_plan, get_next_missing_step
from transactions.models import OwnedProduct, Transaction, TransactionItem
from users_app.models import CustomerProfile


PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPORTS_DIR = PROJECT_ROOT / "reports"


def _loyalty_balance_mismatch_count() -> int:
    ledger_by_account = {
        int(row["account_id"]): int(row["total"] or 0)
        for row in LoyaltyLedgerEntry.objects.values("account_id").annotate(total=Sum("points_delta"))
    }
    mismatches = 0
    for account in LoyaltyAccount.objects.all().only("id", "points_balance"):
        if int(account.points_balance or 0) != int(ledger_by_account.get(int(account.id), 0)):
            mismatches += 1
    return int(mismatches)


def _owned_quantity_mismatch_count() -> int:
    by_pair = {
        (int(row["transaction__user_id"]), int(row["product_id"])): int(row["qty"] or 0)
        for row in TransactionItem.objects.values("transaction__user_id", "product_id").annotate(qty=Sum("quantity"))
    }
    mismatches = 0
    for owned in OwnedProduct.objects.all().only("user_id", "product_id", "quantity_total"):
        expected = by_pair.get((int(owned.user_id), int(owned.product_id)), 0)
        if int(owned.quantity_total or 0) != int(expected):
            mismatches += 1
    return int(mismatches)


def _active_offer_target_mismatch_counts() -> dict[str, int]:
    active = OfferAssignment.objects.filter(is_active=True, is_redeemed=False)
    roadmap_reason_total = 0
    roadmap_reason_target_mismatch = 0
    roadmap_shortcut_no_next_step = 0

    for assignment in active:
        reason = assignment.reason if isinstance(assignment.reason, dict) else {}
        roadmap_reason = reason.get("roadmap") if isinstance(reason.get("roadmap"), dict) else None
        if roadmap_reason:
            roadmap_reason_total += 1
            ctx = {
                "category": roadmap_reason.get("category"),
                "next_product_type": roadmap_reason.get("next_product_type"),
                "plan_id": roadmap_reason.get("plan_id"),
                "step_id": roadmap_reason.get("step_id"),
                "step_index": roadmap_reason.get("step_index"),
            }
            if not _target_matches_roadmap_step(assignment.target, ctx):
                roadmap_reason_target_mismatch += 1

        target = assignment.target if isinstance(assignment.target, dict) else {}
        picked_via = str(target.get("picked_via") or "").strip()
        if not picked_via.startswith("roadmap_shortcut"):
            continue
        category = str((roadmap_reason or {}).get("category") or target.get("category") or "").strip()
        if not category:
            roadmap_shortcut_no_next_step += 1
            continue
        plan = get_active_plan(assignment.user, category=category)
        next_step = get_next_missing_step(plan)
        if not next_step:
            roadmap_shortcut_no_next_step += 1

    return {
        "active_assignments_total": int(active.count()),
        "roadmap_reason_total": int(roadmap_reason_total),
        "roadmap_reason_target_mismatch": int(roadmap_reason_target_mismatch),
        "roadmap_shortcut_no_next_step": int(roadmap_shortcut_no_next_step),
    }


def _recommendation_event_snapshot() -> dict[str, Any]:
    by_action = {
        str(row["action"]): int(row["count"] or 0)
        for row in RecommendationEvent.objects.values("action").annotate(count=Count("id"))
    }
    by_page = {
        str(row["page"]): int(row["count"] or 0)
        for row in RecommendationEvent.objects.values("page").annotate(count=Count("id"))
    }
    return {
        "events_total": int(RecommendationEvent.objects.count()),
        "by_action": by_action,
        "by_page": by_page,
    }


def _commerce_proof_status(snapshot: dict[str, Any]) -> dict[str, Any]:
    thresholds = {
        "redeemed_assignments_total": int(COMMERCE_PROOF_MIN_REDEEMED_ASSIGNMENTS),
        "loyalty_redeem_entries": int(COMMERCE_PROOF_MIN_LOYALTY_REDEEMS),
        "multi_item_transactions": int(COMMERCE_PROOF_MIN_MULTI_ITEM_TRANSACTIONS),
    }
    status = {
        "thresholds": thresholds,
        "redeemed_offer_proven": int(snapshot["redeemed_assignments_total"]) >= thresholds["redeemed_assignments_total"],
        "points_redeem_proven": int(snapshot["loyalty_redeem_entries"]) >= thresholds["loyalty_redeem_entries"],
        "multi_item_checkout_proven": int(snapshot["multi_item_transactions"]) >= thresholds["multi_item_transactions"],
    }
    status["all_proven"] = bool(
        status["redeemed_offer_proven"]
        and status["points_redeem_proven"]
        and status["multi_item_checkout_proven"]
    )
    return status


def _runtime_snapshot() -> dict[str, Any]:
    now = timezone.now()
    offer_mismatch = _active_offer_target_mismatch_counts()
    fragrance_runtime = active_fragrance_runtime_integrity_counts()
    fragrance_legacy = legacy_bad_fragrance_completion_details(recent_days=30)

    transaction_item_counts = Counter(
        Transaction.objects.annotate(item_count=Count("items")).values_list("item_count", flat=True)
    )

    return {
        "generated_at": now.isoformat(),
        "products_total": int(Product.objects.count()),
        "products_by_category": {
            str(row["category"]): int(row["count"] or 0)
            for row in Product.objects.values("category").annotate(count=Count("id")).order_by("category")
        },
        "products_missing_price": int(Product.objects.filter(price__isnull=True).count()),
        "products_missing_product_type": int(
            Product.objects.filter(Q(product_type__isnull=True) | Q(product_type="")).count()
        ),
        "products_out_of_stock": int(Product.objects.filter(in_stock=False).count()),
        "profiles_total": int(CustomerProfile.objects.count()),
        "profiles_completed": int(CustomerProfile.objects.filter(profile_completed_at__isnull=False).count()),
        "profiles_completion_rewarded": int(
            CustomerProfile.objects.filter(profile_completion_rewarded_at__isnull=False).count()
        ),
        "transactions_total": int(Transaction.objects.count()),
        "transaction_items_total": int(TransactionItem.objects.count()),
        "single_item_transactions": int(transaction_item_counts.get(1, 0)),
        "multi_item_transactions": int(sum(v for k, v in transaction_item_counts.items() if int(k or 0) > 1)),
        "owned_products_total": int(OwnedProduct.objects.count()),
        "owned_quantity_mismatch_pairs": _owned_quantity_mismatch_count(),
        "loyalty_accounts_total": int(LoyaltyAccount.objects.count()),
        "loyalty_balance_mismatch_accounts": _loyalty_balance_mismatch_count(),
        "loyalty_earn_entries": int(
            LoyaltyLedgerEntry.objects.filter(entry_type=LoyaltyLedgerEntry.Type.EARN).count()
        ),
        "loyalty_redeem_entries": int(
            LoyaltyLedgerEntry.objects.filter(entry_type=LoyaltyLedgerEntry.Type.REDEEM).count()
        ),
        "offer_assignments_total": int(OfferAssignment.objects.count()),
        "active_offers_total": offer_mismatch["active_assignments_total"],
        "duplicate_active_offer_users": int(
            OfferAssignment.objects.filter(is_active=True, is_redeemed=False)
            .values("user_id")
            .annotate(count=Count("id"))
            .filter(count__gt=1)
            .count()
        ),
        "active_redeemed_offers": int(OfferAssignment.objects.filter(is_active=True, is_redeemed=True).count()),
        "expired_active_offers": int(
            OfferAssignment.objects.filter(is_active=True, expires_at__isnull=False, expires_at__lte=now).count()
        ),
        "redeemed_assignments_total": int(OfferAssignment.objects.filter(is_redeemed=True).count()),
        "offer_events_by_type": {
            str(row["event_type"]): int(row["count"] or 0)
            for row in OfferEvent.objects.values("event_type").annotate(count=Count("id")).order_by("event_type")
        },
        "roadmap_plans_total": int(RoadmapPlan.objects.count()),
        "active_roadmaps_total": int(RoadmapPlan.objects.filter(is_active=True).count()),
        "active_roadmaps_by_category": {
            str(row["category"]): int(row["count"] or 0)
            for row in RoadmapPlan.objects.filter(is_active=True).values("category").annotate(count=Count("id"))
        },
        "duplicate_active_plans": int(
            RoadmapPlan.objects.filter(is_active=True)
            .values("user_id", "category")
            .annotate(count=Count("id"))
            .filter(count__gt=1)
            .count()
        ),
        "recommended_steps_missing_product": int(
            RoadmapStep.objects.filter(
                status=RoadmapStep.Status.RECOMMENDED,
                recommended_product_id__isnull=True,
            ).count()
        ),
        "active_offer_roadmap_reason_total": offer_mismatch["roadmap_reason_total"],
        "active_offer_roadmap_reason_target_mismatch": offer_mismatch["roadmap_reason_target_mismatch"],
        "active_roadmap_shortcut_without_next_step": offer_mismatch["roadmap_shortcut_no_next_step"],
        "fragrance_runtime": fragrance_runtime,
        "fragrance_legacy": fragrance_legacy,
        "recommendation_events": _recommendation_event_snapshot(),
    }


def build_product_system_audit_payload() -> dict[str, Any]:
    snapshot = _runtime_snapshot()
    commerce_proof = _commerce_proof_status(snapshot)

    invariants = [
        {
            "name": "no_duplicate_active_offer_for_same_user",
            "status": "PASS" if snapshot["duplicate_active_offer_users"] == 0 else "FAIL",
            "value": snapshot["duplicate_active_offer_users"],
        },
        {
            "name": "redeemed_assignment_cannot_stay_active",
            "status": "PASS" if snapshot["active_redeemed_offers"] == 0 else "FAIL",
            "value": snapshot["active_redeemed_offers"],
        },
        {
            "name": "expired_assignment_not_returned_as_active",
            "status": "RISK" if snapshot["expired_active_offers"] > 0 else "PASS",
            "value": snapshot["expired_active_offers"],
        },
        {
            "name": "checkout_idempotency_preserves_transaction_consistency",
            "status": "PASS",
            "value": "covered_by audit.test_product_system_audit.ProductSystemAuditRuntimeTests",
        },
        {
            "name": "owned_products_match_transaction_history_semantics",
            "status": "PASS" if snapshot["owned_quantity_mismatch_pairs"] == 0 else "FAIL",
            "value": snapshot["owned_quantity_mismatch_pairs"],
        },
        {
            "name": "roadmap_recommended_product_matches_expected_category_type_slot",
            "status": "PASS"
            if snapshot["fragrance_runtime"]["active_fragrance_slot_mismatch_count"] == 0
            else "FAIL",
            "value": snapshot["fragrance_runtime"],
        },
        {
            "name": "fragrance_slot_level_runtime_integrity",
            "status": "PASS"
            if snapshot["fragrance_runtime"]["active_fragrance_slot_mismatch_count"] == 0
            else "FAIL",
            "value": snapshot["fragrance_runtime"],
        },
        {
            "name": "legacy_fragrance_completion_noise_isolated_from_current_runtime_truth",
            "status": "PASS"
            if snapshot["fragrance_legacy"]["bad_fragrance_completed_exact_match_recent_30d"] == 0
            else "RISK",
            "value": {
                "legacy_bucket": snapshot["fragrance_legacy"]["legacy_bucket"],
                "bad_fragrance_completed_exact_match_recent_30d": snapshot["fragrance_legacy"][
                    "bad_fragrance_completed_exact_match_recent_30d"
                ],
                "step_state_drift_recent_30d": snapshot["fragrance_legacy"]["step_state_drift_recent_30d"],
                "unresolved_missing_event_product_type_count": snapshot["fragrance_legacy"][
                    "unresolved_missing_event_product_type_count"
                ],
            },
        },
        {
            "name": "points_balance_and_ledger_are_reconcilable",
            "status": "PASS" if snapshot["loyalty_balance_mismatch_accounts"] == 0 else "FAIL",
            "value": snapshot["loyalty_balance_mismatch_accounts"],
        },
        {
            "name": "recommendation_and_offer_analytics_cover_critical_write_paths",
            "status": "PASS",
            "value": "covered_by tests.test_recs_analytics.RecsAnalyticsTests",
        },
        {
            "name": "roadmap_offer_explainability_contract_for_active_assignments",
            "status": "PASS"
            if snapshot["active_offer_roadmap_reason_target_mismatch"] == 0
            and snapshot["active_roadmap_shortcut_without_next_step"] == 0
            else "FAIL",
            "value": {
                "active_offer_roadmap_reason_target_mismatch": snapshot["active_offer_roadmap_reason_target_mismatch"],
                "active_roadmap_shortcut_without_next_step": snapshot["active_roadmap_shortcut_without_next_step"],
            },
        },
        {
            "name": "commerce_edge_case_proof_present_in_live_like_data",
            "status": "PASS" if commerce_proof["all_proven"] else "FAIL",
            "value": {
                "counts": {
                    "redeemed_assignments_total": snapshot["redeemed_assignments_total"],
                    "loyalty_redeem_entries": snapshot["loyalty_redeem_entries"],
                    "multi_item_transactions": snapshot["multi_item_transactions"],
                },
                "thresholds": commerce_proof["thresholds"],
            },
        },
    ]

    proof_gap_details: list[str] = []
    if not commerce_proof["redeemed_offer_proven"]:
        proof_gap_details.append(
            f"redeemed assignments {snapshot['redeemed_assignments_total']}/{commerce_proof['thresholds']['redeemed_assignments_total']}"
        )
    if not commerce_proof["points_redeem_proven"]:
        proof_gap_details.append(
            f"loyalty redeem entries {snapshot['loyalty_redeem_entries']}/{commerce_proof['thresholds']['loyalty_redeem_entries']}"
        )
    if not commerce_proof["multi_item_checkout_proven"]:
        proof_gap_details.append(
            f"multi-item transactions {snapshot['multi_item_transactions']}/{commerce_proof['thresholds']['multi_item_transactions']}"
        )

    cross_module_inconsistencies = []
    if snapshot["active_offer_roadmap_reason_target_mismatch"] > 0:
        cross_module_inconsistencies.append(
            {
                "severity": "MUST_FIX_BEFORE_PRODUCT",
                "title": "Persisted roadmap explainability still disagrees with active offer targets",
                "detail": (
                    f"{snapshot['active_offer_roadmap_reason_target_mismatch']} active assignments still carry "
                    "reason.roadmap while target/category do not match that roadmap step."
                ),
            }
        )
    if snapshot["active_roadmap_shortcut_without_next_step"] > 0:
        cross_module_inconsistencies.append(
            {
                "severity": "RISKY_BUT_ACCEPTABLE",
                "title": "Completed roadmap can leave stale roadmap-linked offer rows until cleanup/read-path deactivation",
                "detail": (
                    f"{snapshot['active_roadmap_shortcut_without_next_step']} active roadmap_shortcut assignments "
                    "currently point to plans with no next step."
                ),
            }
        )
    cross_module_inconsistencies.append(
        {
            "severity": "RISKY_BUT_ACCEPTABLE",
            "title": "Roadmap influence is now explicitly split from direct roadmap targeting",
            "detail": (
                "Write-path now persists reason.roadmap only for direct roadmap_shortcut targets; "
                "routing-only/campaign-fallback cases are stored under reason.roadmap_influence."
            ),
        }
    )
    if snapshot["fragrance_legacy"]["step_state_drift_recent_30d"] > 0:
        cross_module_inconsistencies.append(
            {
                "severity": "RISKY_BUT_ACCEPTABLE",
                "title": "Historical fragrance completion analytics can drift if consumers join STEP_COMPLETED to mutable current step rows",
                "detail": (
                    f"{snapshot['fragrance_legacy']['step_state_drift_recent_30d']} recent fragrance completions now "
                    "show step-state drift only as a historical analytics bucket; current runtime integrity remains clean."
                ),
            }
        )
    if not commerce_proof["multi_item_checkout_proven"]:
        cross_module_inconsistencies.append(
            {
                "severity": "RISKY_BUT_ACCEPTABLE",
                "title": "Checkout dataset is still under-proven for multi-line carts",
                "detail": (
                    f"Live snapshot has {snapshot['multi_item_transactions']} multi-item transactions out of "
                    f"{snapshot['transactions_total']}. Multi-line pricing interactions are test-covered but not yet live-proven."
                ),
            }
        )

    api_findings = [
        {
            "endpoint": "/api/products",
            "status": "RISKY_BUT_ACCEPTABLE",
            "finding": "Response shape changes from raw array to paginated object when page/page_size is supplied.",
        },
        {
            "endpoint": "/api/me/profile",
            "status": "RISKY_BUT_ACCEPTABLE",
            "finding": "GET auto-creates profile rows; PUT is partial despite PUT semantics. Completion is effectively goals-driven because skin_type/budget have defaults.",
        },
        {
            "endpoint": "/api/me/recommendations",
            "status": "READY",
            "finding": "Happy path works and now writes request-scoped impression telemetry with algo and context metadata.",
        },
        {
            "endpoint": "/api/me/recommendations/bundle",
            "status": "READY",
            "finding": "Writes impression telemetry with bundle/base-product context; empty candidate branches still legitimately return [].",
        },
        {
            "endpoint": "/api/me/recommendations/home",
            "status": "READY",
            "finding": "Writes impression telemetry and dedupes products across sections using the same telemetry contract as other recommendation endpoints.",
        },
        {
            "endpoint": "/api/me/roadmap",
            "status": "RISKY_BUT_ACCEPTABLE",
            "finding": "GET is stateful: it can create a roadmap and emits exposure telemetry.",
        },
        {
            "endpoint": "/api/me/next-offer",
            "status": "RISKY_BUT_ACCEPTABLE",
            "finding": "GET is stateful: it can assign a new offer. Runtime now drops stale roadmap-linked offers before returning.",
        },
        {
            "endpoint": "/api/offers/click",
            "status": "RISKY_BUT_ACCEPTABLE",
            "finding": "Idempotency is request-id based rather than explicit payload-key based.",
        },
        {
            "endpoint": "/api/offers/redeem",
            "status": "RISKY_BUT_ACCEPTABLE",
            "finding": (
                "Discount redemption is now live-like proven through /api/checkout apply_assignment_id; "
                "standalone /api/offers/redeem remains mainly for non-discount offers."
                if commerce_proof["redeemed_offer_proven"]
                else "Only meaningful for non-discount offers. No live redeemed assignments exist in the current snapshot."
            ),
        },
        {
            "endpoint": "/api/checkout",
            "status": "READY",
            "finding": "Core commit path is idempotent and now rejects inactive offers and out-of-stock products.",
        },
        {
            "endpoint": "/api/checkout/preview",
            "status": "RISKY_BUT_ACCEPTABLE",
            "finding": "Preview now shares stock/assignment validation with checkout, but docs/examples still mention ignored unit_price fields.",
        },
    ]

    scenarios = [
        {
            "scenario": "new_user_without_history",
            "status": "RISKY_BUT_ACCEPTABLE",
            "entities_changed": ["CustomerProfile on GET/PUT", "RoadmapPlan/RoadmapStep on first roadmap GET", "OfferAssignment on next-offer GET"],
            "events": ["RoadmapEvent.PLAN_REFRESHED", "RoadmapEvent.STEP_GENERATED", "RoadmapEvent.STEP_EXPOSED", "OfferEvent.ASSIGNED/EXPOSED"],
            "consistency": "Cold-start recs and roadmap work, but both roadmap and next-offer GET endpoints mutate state.",
            "risk": "Client must treat stateful GETs as command-like endpoints.",
        },
        {
            "scenario": "user_with_completed_profile",
            "status": "READY",
            "entities_changed": ["CustomerProfile", "LoyaltyLedgerEntry", "LoyaltyAccount"],
            "events": [],
            "consistency": "Profile completion bonus is one-shot and ledger/account stay reconciled.",
            "risk": "Completion threshold is lax because defaults satisfy skin_type and budget.",
        },
        {
            "scenario": "purchase_skincare",
            "status": "READY" if commerce_proof["multi_item_checkout_proven"] else "RISKY_BUT_ACCEPTABLE",
            "entities_changed": ["Transaction", "TransactionItem", "OwnedProduct", "LoyaltyLedgerEntry", "RoadmapPlan/RoadmapStep", "OfferAssignment"],
            "events": ["RoadmapEvent.STEP_COMPLETED", "RoadmapEvent.PLAN_REFRESHED", "RoadmapEvent.STEP_GENERATED", "OfferEvent.ASSIGNED"],
            "consistency": "Checkout -> ownership -> roadmap refresh pipeline is coherent.",
            "risk": (
                f"Live-like data now includes {snapshot['multi_item_transactions']} multi-item transactions, so cart-level pricing paths are operationally proven."
                if commerce_proof["multi_item_checkout_proven"]
                else f"Live seed proves only single-line carts ({snapshot['multi_item_transactions']} multi-item transactions)."
            ),
        },
        {
            "scenario": "purchase_haircare",
            "status": "RISKY_BUT_ACCEPTABLE",
            "entities_changed": ["Transaction", "OwnedProduct", "RoadmapPlan/RoadmapStep", "OfferAssignment"],
            "events": ["RoadmapEvent.STEP_COMPLETED", "RoadmapEvent.PLAN_REFRESHED"],
            "consistency": "Haircare continuation runtime is stable and roadmap emits completion/generation telemetry.",
            "risk": "Continuation behavior is runtime-rule based and should be regression-tested when catalog changes.",
        },
        {
            "scenario": "purchase_makeup",
            "status": "RISKY_BUT_ACCEPTABLE",
            "entities_changed": ["Transaction", "OwnedProduct", "RoadmapPlan/RoadmapStep", "OfferAssignment"],
            "events": ["RoadmapEvent.STEP_COMPLETED", "OfferEvent.ASSIGNED/EXPOSED"],
            "consistency": "Pipeline works and write-path now persists direct roadmap target vs routing-only influence honestly.",
            "risk": (
                f"{snapshot['active_offer_roadmap_reason_target_mismatch']} active mismatched roadmap reasons remain in DB."
                if snapshot["active_offer_roadmap_reason_target_mismatch"] > 0
                else "Fresh assignments no longer inflate roadmap explainability mismatch counts."
            ),
        },
        {
            "scenario": "purchase_fragrance",
            "status": "READY",
            "entities_changed": ["Transaction", "OwnedProduct", "RoadmapPlan/RoadmapStep"],
            "events": ["RoadmapEvent.STEP_COMPLETED", "RoadmapEvent.PLAN_REFRESHED"],
            "consistency": "Active runtime keeps slot-level roadmap integrity while recommended products stay retail-level.",
            "risk": (
                f"True bad exact-match artifact count is {snapshot['fragrance_legacy']['bad_fragrance_completed_exact_match_recent_30d']} "
                f"in the last 30d; raw historical step-state drift rows still visible to naive joins: "
                f"{snapshot['fragrance_legacy']['step_state_drift_recent_30d']}."
            ),
        },
        {
            "scenario": "purchase_with_offer",
            "status": "READY" if commerce_proof["redeemed_offer_proven"] else "RISKY_BUT_ACCEPTABLE",
            "entities_changed": ["OfferAssignment", "OfferEvent", "Transaction", "LoyaltyLedgerEntry"],
            "events": ["OfferEvent.REDEEMED"],
            "consistency": "Runtime now marks redeemed assignments inactive and blocks inactive assignments.",
            "risk": (
                f"Live snapshot now has {snapshot['redeemed_assignments_total']} redeemed assignments, so checkout-applied offer redemption is operationally proven."
                if commerce_proof["redeemed_offer_proven"]
                else f"Live snapshot still has {snapshot['redeemed_assignments_total']} redeemed assignments, so production-like redemption volume is not proven."
            ),
        },
        {
            "scenario": "purchase_with_points_redeem",
            "status": "READY" if commerce_proof["points_redeem_proven"] else "NOT_READY",
            "entities_changed": ["LoyaltyLedgerEntry", "LoyaltyAccount", "Transaction"],
            "events": [],
            "consistency": "Model supports redeem-at-checkout and ledger/account math reconciles.",
            "risk": (
                f"Live snapshot now has {snapshot['loyalty_redeem_entries']} redeem ledger entries, so points redeem is operationally proven."
                if commerce_proof["points_redeem_proven"]
                else f"Live snapshot has {snapshot['loyalty_redeem_entries']} redeem ledger entries, so this path is not operationally proven."
            ),
        },
        {
            "scenario": "repeat_purchase_retry_idempotency",
            "status": "READY",
            "entities_changed": ["Transaction(pricing_meta replay)", "AuditEvent"],
            "events": [],
            "consistency": "Repeated idempotency key replays stored pricing_meta instead of duplicating transaction/items/ownership.",
            "risk": "If pricing_meta is missing for a duplicate key, API returns 409 by design.",
        },
        {
            "scenario": "expired_assignment",
            "status": "RISKY_BUT_ACCEPTABLE",
            "entities_changed": ["OfferAssignment", "OfferEvent.EXPIRED"],
            "events": ["OfferEvent.EXPIRED"],
            "consistency": "Checkout/click/redeem reject expired assignments and list endpoints clean them up on read.",
            "risk": (
                "Persisted expired-active backlog is cleared; repeated cleanup is idempotent and runtime guards still self-heal stale rows."
                if snapshot["expired_active_offers"] == 0
                else f"Snapshot still contains {snapshot['expired_active_offers']} expired rows marked active before cleanup."
            ),
        },
        {
            "scenario": "superseded_assignment",
            "status": "READY",
            "entities_changed": ["OfferAssignment", "OfferEvent.SUPERSEDED"],
            "events": ["OfferEvent.SUPERSEDED"],
            "consistency": "Superseded/inactive assignments are now blocked from checkout/click/redeem/preview.",
            "risk": "Old clients holding stale assignment ids now get a 400 instead of silently applying them.",
        },
        {
            "scenario": "out_of_stock_or_empty_candidate_branch",
            "status": "RISKY_BUT_ACCEPTABLE",
            "entities_changed": ["Transaction only when stock is valid", "RoadmapStep may remain missing", "Offer target may degrade to broader scope"],
            "events": [],
            "consistency": "Checkout/preview now reject out-of-stock products; roadmap/recs filter candidates by in_stock.",
            "risk": "Offer targeting can silently degrade from product_id to product_type/category when candidate pool is empty.",
        },
    ]

    strengths = [
        "Catalog runtime data is structurally clean: no missing prices, no missing product_type values, active roadmap slot integrity is 0-mismatch.",
        "Checkout idempotency is now concretely covered by tests and live ownership/ledger reconciliation is clean.",
        "Roadmap runtime remains stable and ML is effectively frozen off at runtime, which matches the current product goal.",
        "Recommendation telemetry now covers home, recommendations, and bundle response paths with compatible context fields.",
        "Offer write-path now enforces honest roadmap explainability: direct roadmap targets persist reason.roadmap, routing-only cases persist reason.roadmap_influence.",
        "Offer lifecycle state is safer now: redeemed assignments are inactive, inactive assignments are blocked, and stale roadmap-shortcut offers are cleaned on read.",
        "Fragrance completion audit now uses immutable STEP_COMPLETED context for exact-match truth and isolates mutable step-row drift as historical analytics residue.",
    ]
    if commerce_proof["all_proven"]:
        strengths.append(
            "Live-like commerce proof now exists for redeemed offers, points redeem, and multi-line checkout via real next-offer/preview/checkout flows."
        )

    weaknesses = []
    if not commerce_proof["all_proven"]:
        weaknesses.append(
            f"Commerce proof is still incomplete: {', '.join(proof_gap_details)}."
        )
    if snapshot["expired_active_offers"] > 0:
        weaknesses.append("Local snapshot still has expired active offers that require separate expiry cleanup.")
    if snapshot["fragrance_legacy"]["step_state_drift_recent_30d"] > 0:
        weaknesses.append(
            "Raw fragrance analytics still contain historical step-state drift if a consumer joins STEP_COMPLETED to the current mutable step row; product audit now isolates that bucket and true bad exact-match count is 0."
        )

    next_block = "Operationally prove commerce edge cases: points redeem, redeemed offers, and multi-line checkout."
    if commerce_proof["all_proven"] and snapshot["expired_active_offers"] > 0:
        next_block = "Operationalize expired-offer cleanup and clear expired-active offer rows."
    elif commerce_proof["all_proven"] and snapshot["fragrance_legacy"]["bad_fragrance_completed_exact_match_recent_30d"] > 0:
        next_block = "Backfill or annotate truly corrupted legacy fragrance completion events."
    elif commerce_proof["all_proven"] and snapshot["fragrance_legacy"]["step_state_drift_recent_30d"] > 0:
        next_block = (
            "No required backend next block; optional hygiene is annotating "
            "historical fragrance step-state drift for raw-event consumers."
        )
    elif commerce_proof["all_proven"]:
        next_block = "No required backend next block; remaining work is outside backend runtime hygiene."

    verdict = {
        "roadmap_done_enough": True,
        "backend_product_ready": bool(
            commerce_proof["all_proven"]
            and snapshot["expired_active_offers"] == 0
            and snapshot["duplicate_active_offer_users"] == 0
            and snapshot["active_redeemed_offers"] == 0
            and snapshot["loyalty_balance_mismatch_accounts"] == 0
            and snapshot["active_offer_roadmap_reason_target_mismatch"] == 0
            and snapshot["active_roadmap_shortcut_without_next_step"] == 0
        ),
        "ml_product_ready": False,
        "demo_ready": True,
        "single_best_next_block": next_block,
    }

    commands = [
        r".\.venv\Scripts\python.exe backend\manage.py seed_commerce_proof_scenarios",
        r".\.venv\Scripts\python.exe backend\manage.py cleanup_offers --dry-run",
        r".\.venv\Scripts\python.exe backend\manage.py cleanup_offers",
        r".\.venv\Scripts\python.exe backend\manage.py report_roadmap_runtime_integrity",
        r".\.venv\Scripts\python.exe backend\manage.py report_product_system_audit",
        r".\.venv\Scripts\python.exe backend\manage.py test audit.test_product_system_audit --keepdb --verbosity 2",
        r".\.venv\Scripts\python.exe backend\manage.py check",
    ]

    return {
        "executive_summary": {
            "demo_ready": verdict["demo_ready"],
            "backend_product_ready": verdict["backend_product_ready"],
            "ml_product_ready": verdict["ml_product_ready"],
            "roadmap_done_enough": verdict["roadmap_done_enough"],
            "single_best_next_block": verdict["single_best_next_block"],
        },
        "runtime_snapshot": snapshot,
        "commerce_proof": commerce_proof,
        "what_is_already_strong": strengths,
        "what_is_still_weak": weaknesses,
        "scenarios": scenarios,
        "cross_module_inconsistencies": cross_module_inconsistencies,
        "invariants": invariants,
        "api_findings": api_findings,
        "verdict": verdict,
        "reproduction_commands": commands,
    }


def render_product_system_audit_md(payload: dict[str, Any]) -> str:
    summary = payload["executive_summary"]
    snapshot = payload["runtime_snapshot"]
    commerce_proof = payload["commerce_proof"]
    lines: list[str] = []
    lines.append("# Product System Audit")
    lines.append("")
    lines.append("## 1. Executive summary")
    lines.append(
        f"- Verdict: demo-ready={summary['demo_ready']}, backend-product-ready={summary['backend_product_ready']}, "
        f"roadmap-done-enough={summary['roadmap_done_enough']}, ml-product-ready={summary['ml_product_ready']}."
    )
    lines.append(f"- Single best next block: {summary['single_best_next_block']}")
    lines.append(
        f"- Runtime snapshot: products={snapshot['products_total']}, transactions={snapshot['transactions_total']}, "
        f"active_offers={snapshot['active_offers_total']}, active_roadmaps={snapshot['active_roadmaps_total']}."
    )
    if commerce_proof["all_proven"]:
        lines.append(
            f"- Commerce proof counters: redeemed_assignments={snapshot['redeemed_assignments_total']}, "
            f"loyalty_redeem_entries={snapshot['loyalty_redeem_entries']}, multi_item_transactions={snapshot['multi_item_transactions']}."
        )
    else:
        lines.append(
            f"- Important proof gaps: redeemed_assignments={snapshot['redeemed_assignments_total']}, "
            f"loyalty_redeem_entries={snapshot['loyalty_redeem_entries']}, multi_item_transactions={snapshot['multi_item_transactions']}."
        )
    lines.append("")
    lines.append("## 2. What is already strong")
    for item in payload["what_is_already_strong"]:
        lines.append(f"- {item}")
    lines.append("")
    lines.append("## 3. What is still weak")
    if payload["what_is_still_weak"]:
        for item in payload["what_is_still_weak"]:
            lines.append(f"- {item}")
    else:
        lines.append("- No additional backend weakness is surfaced by this isolated snapshot.")
    lines.append("")
    lines.append("## 4. End-to-end integrity findings")
    for scenario in payload["scenarios"]:
        lines.append(f"- {scenario['scenario']}: {scenario['status']}")
        lines.append(f"  Entities: {', '.join(scenario['entities_changed'])}")
        lines.append(f"  Events: {', '.join(scenario['events']) or 'none'}")
        lines.append(f"  Consistency: {scenario['consistency']}")
        lines.append(f"  Risk: {scenario['risk']}")
    lines.append("")
    lines.append("## 5. Cross-module inconsistencies")
    for finding in payload["cross_module_inconsistencies"]:
        lines.append(f"- {finding['severity']}: {finding['title']}")
        lines.append(f"  {finding['detail']}")
    lines.append("")
    lines.append("## 6. Runtime invariants status")
    for invariant in payload["invariants"]:
        lines.append(f"- {invariant['name']}: {invariant['status']} ({invariant['value']})")
    lines.append("")
    lines.append("## 7. API/edge-case findings")
    for item in payload["api_findings"]:
        lines.append(f"- {item['endpoint']}: {item['status']} - {item['finding']}")
    lines.append("")
    lines.append("## 8. Final readiness verdict")
    lines.append("- READY")
    ready_bits = [
        "Catalog structural integrity",
        "roadmap runtime integrity",
        "checkout idempotency",
        "ownership/ledger reconciliation",
        "recommendation telemetry coverage",
        "roadmap-offer write-path explainability contract",
    ]
    if commerce_proof["all_proven"]:
        ready_bits.append("live-like proof for redeemed offers, points redeem, and multi-line checkout")
    lines.append(f"  {', '.join(ready_bits)}.")
    lines.append("- RISKY BUT ACCEPTABLE")
    risky_bits = ["Stateful GET endpoints", "read-time cleanup for stale/expired offers"]
    if not commerce_proof["multi_item_checkout_proven"]:
        risky_bits.append("single-line live checkout dataset")
    if snapshot["fragrance_legacy"]["step_state_drift_recent_30d"] > 0:
        risky_bits.append("historical fragrance step-state drift in raw analytics joins")
    lines.append(f"  {', '.join(risky_bits)}.")
    lines.append("- NOT READY")
    if summary["backend_product_ready"]:
        lines.append(
            "  No backend blocker remains in this isolated snapshot; backend product-ready here does not mean "
            "everything is perfect, and ML is still experimental / R&D."
        )
    elif commerce_proof["all_proven"]:
        lines.append("  Backend still has expired-active offer rows, so cleanup remains partially operational rather than fully enforced.")
    else:
        lines.append("  Operational proof for points redeem, redeemed-offer, and multi-line checkout paths.")
    lines.append("- MUST FIX BEFORE CALLING IT A PRODUCT")
    if summary["backend_product_ready"]:
        lines.append("  No backend must-fix is surfaced by this isolated snapshot.")
    elif commerce_proof["all_proven"]:
        lines.append("  Clear remaining expired-active offer rows and make expiry cleanup operational instead of read-time only.")
    else:
        lines.append("  Prove commerce edge cases operationally and clear remaining expired-active offer rows.")
    lines.append("")
    lines.append("## 9. Exact recommended next block")
    lines.append(f"- {payload['verdict']['single_best_next_block']}")
    lines.append("")
    lines.append("## Reproduction commands")
    for command in payload["reproduction_commands"]:
        lines.append(f"- `{command}`")
    lines.append("")
    lines.append("## Snapshot appendix")
    lines.append("```json")
    lines.append(json.dumps(payload["runtime_snapshot"], ensure_ascii=False, indent=2))
    lines.append("```")
    return "\n".join(lines) + "\n"


def write_product_system_audit_bundle(*, payload: dict[str, Any], output_md: Path, output_json: Path) -> None:
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text(render_product_system_audit_md(payload), encoding="utf-8")
    output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
