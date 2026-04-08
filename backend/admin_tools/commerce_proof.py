from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from decimal import Decimal
import hashlib
from typing import Any

from django.contrib.auth import get_user_model
from django.core.management.base import CommandError
from django.db.models import Count, Sum
from django.utils import timezone
from rest_framework.test import APIClient

from admin_tools.demo_history_seed import (
    DEMO_PASSWORD,
    base_tier,
    ensure_demo_offers,
    reset_demo_users_and_history,
)
from catalog.models import Product
from loyalty.models import LoyaltyAccount, LoyaltyLedgerEntry
from offers.models import OfferAssignment, OfferEvent
from transactions.models import Transaction
from users_app.models import CustomerProfile


COMMERCE_PROOF_PREFIX = "demo_commerce"
COMMERCE_PROOF_SEED = 20260407
COMMERCE_PROOF_MIN_REDEEMED_ASSIGNMENTS = 2
COMMERCE_PROOF_MIN_LOYALTY_REDEEMS = 2
COMMERCE_PROOF_MIN_MULTI_ITEM_TRANSACTIONS = 2


@dataclass(frozen=True)
class CommerceProofUserSpec:
    slug: str
    first_name: str
    last_name: str
    city: str
    skin_type: str
    goals: list[str]
    budget: str


PROOF_USER_SPECS = (
    CommerceProofUserSpec(
        slug="offer",
        first_name="Amina",
        last_name="Proof",
        city="Almaty",
        skin_type=CustomerProfile.SkinType.NORMAL,
        goals=["hydration", "repair"],
        budget=CustomerProfile.Budget.MEDIUM,
    ),
    CommerceProofUserSpec(
        slug="combo",
        first_name="Dana",
        last_name="Proof",
        city="Astana",
        skin_type=CustomerProfile.SkinType.DRY,
        goals=["hydration", "radiance"],
        budget=CustomerProfile.Budget.HIGH,
    ),
    CommerceProofUserSpec(
        slug="points",
        first_name="Mira",
        last_name="Proof",
        city="Shymkent",
        skin_type=CustomerProfile.SkinType.COMBINATION,
        goals=["clarity", "hydration"],
        budget=CustomerProfile.Budget.MEDIUM,
    ),
)


def _proof_username(*, prefix: str, seed: int, slug: str) -> str:
    return f"{prefix}_s{seed}_{slug}"


def _base_product_queryset():
    return Product.objects.filter(in_stock=True).exclude(price__isnull=True).order_by("-price", "id")


def _product_or_raise(*, product_id: int) -> Product:
    product = _base_product_queryset().filter(id=int(product_id)).first()
    if product is None:
        raise CommandError(f"Product {product_id} is not available for commerce proof")
    return product


def _pick_any_product(*, exclude_ids: set[int] | None = None, category: str | None = None) -> Product:
    exclude_ids = exclude_ids or set()
    queryset = _base_product_queryset().exclude(id__in=exclude_ids)
    if category:
        product = queryset.filter(category=category).first()
        if product is not None:
            return product
    product = queryset.first()
    if product is None:
        raise CommandError("Catalog needs at least one in-stock priced product for commerce proof")
    return product


def _pick_matching_target_product(target: dict[str, Any]) -> Product:
    scope = str(target.get("scope") or "cart").strip()
    value = target.get("value")
    category = str(target.get("category") or "").strip()
    product_type = str(target.get("product_type") or "").strip()

    if scope == "product_id":
        if value in (None, ""):
            raise CommandError("Offer target has scope=product_id without value")
        return _product_or_raise(product_id=int(value))

    queryset = _base_product_queryset()
    if scope == "category":
        queryset = queryset.filter(category=str(value or category).strip())
    elif scope == "product_type":
        target_product_type = str(value or product_type).strip()
        if category:
            queryset = queryset.filter(category=category)
        queryset = queryset.filter(product_type=target_product_type)

    product = queryset.first()
    if product is None:
        raise CommandError(f"No eligible product found for offer target={target}")
    return product


def _build_multiline_items_for_target(target: dict[str, Any] | None) -> list[dict[str, int]]:
    normalized_target = target if isinstance(target, dict) else {"scope": "cart"}
    primary = _pick_matching_target_product(normalized_target)
    preferred_category = str(primary.category or "").strip() or None
    secondary = _pick_any_product(exclude_ids={int(primary.id)}, category=preferred_category)
    return [
        {"product": int(primary.id), "quantity": 1},
        {"product": int(secondary.id), "quantity": 1},
    ]


def _build_generic_multiline_items(*, exclude_ids: set[int] | None = None, preferred_category: str | None = None) -> list[dict[str, int]]:
    primary = _pick_any_product(exclude_ids=exclude_ids, category=preferred_category)
    next_exclude = set(exclude_ids or set())
    next_exclude.add(int(primary.id))
    secondary = _pick_any_product(exclude_ids=next_exclude, category=preferred_category)
    return [
        {"product": int(primary.id), "quantity": 1},
        {"product": int(secondary.id), "quantity": 1},
    ]


def _ensure_catalog_can_support_commerce_proof() -> None:
    if _base_product_queryset().count() < 2:
        raise CommandError("Commerce proof needs at least two in-stock priced products in the catalog")


def _ensure_proof_users(*, prefix: str, seed: int) -> list[Any]:
    reset_demo_users_and_history(prefix=prefix)
    User = get_user_model()
    bronze = base_tier()
    now = timezone.now()
    users: list[Any] = []
    for spec in PROOF_USER_SPECS:
        username = _proof_username(prefix=prefix, seed=seed, slug=spec.slug)
        user = User.objects.create_user(
            username=username,
            password=DEMO_PASSWORD,
            email=f"{username}@example.test",
        )
        profile, _ = CustomerProfile.objects.get_or_create(user=user)
        profile.first_name = spec.first_name
        profile.last_name = spec.last_name
        profile.city = spec.city
        profile.skin_type = spec.skin_type
        profile.goals = list(spec.goals)
        profile.avoid_flags = []
        profile.budget = spec.budget
        profile.hair_profile = {"hair_type": "wavy", "scalp_type": "normal", "hair_thickness": "medium", "concerns": ["repair"]}
        profile.makeup_profile = {"finish_pref": ["natural"], "coverage_pref": ["medium"], "undertone": "neutral"}
        profile.fragrance_profile = {"liked_families": ["fresh", "woody"], "liked_notes": ["bergamot", "cedar"], "intensity_pref": "medium"}
        profile.profile_completed_at = now
        profile.save()
        LoyaltyAccount.objects.get_or_create(user=user, defaults={"tier": bronze, "points_balance": 0})
        users.append(user)
    return users


def _authed_client(user) -> APIClient:
    client = APIClient()
    client.force_authenticate(user)
    return client


def _preview_checkout(client: APIClient, payload: dict[str, Any]) -> dict[str, Any]:
    preview = client.post("/api/checkout/preview", payload, format="json")
    if preview.status_code != 200:
        raise CommandError(f"POST /api/checkout/preview failed: status={preview.status_code} payload={payload}")
    if not bool(preview.data.get("ok")):
        raise CommandError(f"POST /api/checkout/preview returned non-ok payload={preview.data}")
    return dict(preview.data)


def _proof_idempotency_key(label: str) -> str:
    digest = hashlib.sha1(str(label).encode("utf-8")).hexdigest()
    return f"commerce-proof:{digest[:40]}"


def _commit_checkout(client: APIClient, payload: dict[str, Any]) -> dict[str, Any]:
    checkout = client.post("/api/checkout", payload, format="json")
    if checkout.status_code != 201:
        detail = getattr(checkout, "data", None)
        if detail is None:
            try:
                detail = checkout.content.decode("utf-8", errors="ignore")
            except Exception:
                detail = None
        raise CommandError(
            f"POST /api/checkout failed: status={checkout.status_code} payload={payload} detail={detail}"
        )
    if not bool(checkout.data.get("ok")):
        raise CommandError(f"POST /api/checkout returned non-ok payload={checkout.data}")
    return dict(checkout.data)


def _fetch_assignment_or_raise(client: APIClient) -> dict[str, Any]:
    response = client.get("/api/me/next-offer")
    if response.status_code != 200:
        raise CommandError(f"GET /api/me/next-offer failed: status={response.status_code}")
    assignment_id = response.data.get("assignment_id")
    if not assignment_id:
        raise CommandError(f"GET /api/me/next-offer returned no assignment: payload={response.data}")
    return dict(response.data)


def _redeem_points_request(balance: int) -> int:
    if int(balance or 0) <= 0:
        raise CommandError("Points redeem scenario needs a positive loyalty balance")
    return max(1, min(int(balance), 250))


def _assert_multiline_transaction(transaction_id: int) -> None:
    item_count = int(
        Transaction.objects.filter(id=int(transaction_id)).annotate(item_count=Count("items")).values_list("item_count", flat=True).first()
        or 0
    )
    if item_count <= 1:
        raise CommandError(f"Transaction {transaction_id} is not multi-line")


def _run_warmup_checkout(*, client: APIClient, label: str, preferred_category: str | None = None) -> dict[str, Any]:
    payload = {
        "channel": "online",
        "idempotency_key": _proof_idempotency_key(f"{label}:warmup"),
        "items": _build_generic_multiline_items(preferred_category=preferred_category),
    }
    preview = _preview_checkout(client, payload)
    checkout = _commit_checkout(client, payload)
    _assert_multiline_transaction(int(checkout["transaction_id"]))
    return {
        "type": "warmup_checkout",
        "preview": {
            "gross_total": preview["gross_total"],
            "eligible_total": preview["eligible_total"],
            "estimated_points_earned": int(preview["estimated_points_earned"]),
        },
        "checkout": {
            "transaction_id": int(checkout["transaction_id"]),
            "gross_total": checkout["gross_total"],
            "net_total": checkout["net_total"],
            "points_earned": int(checkout["points_earned"]),
            "new_balance": int(checkout["new_balance"]),
        },
    }


def _run_offer_redemption_checkout(*, user, label: str, redeem_points: int = 0) -> dict[str, Any]:
    client = _authed_client(user)
    assignment_payload = _fetch_assignment_or_raise(client)
    assignment_id = int(assignment_payload["assignment_id"])
    target = assignment_payload.get("target") or {"scope": "cart"}
    payload = {
        "channel": "online",
        "idempotency_key": _proof_idempotency_key(f"{label}:redeem-offer"),
        "apply_assignment_id": assignment_id,
        "items": _build_multiline_items_for_target(target),
    }
    if redeem_points > 0:
        payload["redeem_points"] = int(redeem_points)
    preview = _preview_checkout(client, payload)
    checkout = _commit_checkout(client, payload)
    _assert_multiline_transaction(int(checkout["transaction_id"]))

    assignment = OfferAssignment.objects.get(id=assignment_id)
    if not assignment.is_redeemed or assignment.is_active:
        raise CommandError(f"Assignment {assignment_id} was not redeemed cleanly")
    if not OfferEvent.objects.filter(assignment=assignment, event_type=OfferEvent.Type.REDEEMED).exists():
        raise CommandError(f"Assignment {assignment_id} is missing OfferEvent.REDEEMED")

    return {
        "type": "offer_redemption_checkout",
        "assignment_id": assignment_id,
        "target": target,
        "preview": {
            "gross_total": preview["gross_total"],
            "eligible_total": preview["eligible_total"],
            "offer_applied": bool(preview["offer_applied"]),
            "points_redeemed": int(preview["points_redeemed"]),
            "estimated_points_earned": int(preview["estimated_points_earned"]),
        },
        "checkout": {
            "transaction_id": int(checkout["transaction_id"]),
            "offer_assignment_id": int(checkout["offer_assignment_id"]),
            "gross_total": checkout["gross_total"],
            "net_total": checkout["net_total"],
            "points_redeemed": int(checkout["points_redeemed"]),
            "points_earned": int(checkout["points_earned"]),
            "new_balance": int(checkout["new_balance"]),
        },
    }


def _run_points_redeem_checkout(*, user, label: str) -> dict[str, Any]:
    client = _authed_client(user)
    account = LoyaltyAccount.objects.get(user=user)
    redeem_points = _redeem_points_request(int(account.points_balance))
    payload = {
        "channel": "online",
        "idempotency_key": _proof_idempotency_key(f"{label}:redeem-points"),
        "redeem_points": redeem_points,
        "items": _build_generic_multiline_items(),
    }
    preview = _preview_checkout(client, payload)
    checkout = _commit_checkout(client, payload)
    _assert_multiline_transaction(int(checkout["transaction_id"]))

    txn_id = int(checkout["transaction_id"])
    if not LoyaltyLedgerEntry.objects.filter(
        account__user=user,
        entry_type=LoyaltyLedgerEntry.Type.REDEEM,
        reference=f"checkout:txn:{txn_id}",
    ).exists():
        raise CommandError(f"Transaction {txn_id} is missing LoyaltyLedgerEntry.REDEEM")

    return {
        "type": "points_redeem_checkout",
        "preview": {
            "gross_total": preview["gross_total"],
            "eligible_total": preview["eligible_total"],
            "points_redeemed": int(preview["points_redeemed"]),
            "estimated_points_earned": int(preview["estimated_points_earned"]),
            "balance_before": int(preview["balance_before"]),
            "balance_after_estimated": int(preview["balance_after_estimated"]),
        },
        "checkout": {
            "transaction_id": txn_id,
            "gross_total": checkout["gross_total"],
            "net_total": checkout["net_total"],
            "points_redeemed": int(checkout["points_redeemed"]),
            "points_earned": int(checkout["points_earned"]),
            "new_balance": int(checkout["new_balance"]),
        },
    }


def _scenario_user_map(*, users: list[Any]) -> dict[str, Any]:
    by_slug: dict[str, Any] = {}
    for user in users:
        parts = str(user.username).split("_")
        by_slug[parts[-1]] = user
    return by_slug


def build_commerce_proof_snapshot(*, prefix: str = COMMERCE_PROOF_PREFIX, seed: int | None = None) -> dict[str, Any]:
    qs = get_user_model().objects.filter(username__startswith=f"{prefix}_", is_staff=False, is_superuser=False).order_by("username")
    if seed is not None:
        qs = qs.filter(username__contains=f"_s{seed}_")
    users = list(qs)
    user_ids = [int(user.id) for user in users]
    account_ids = list(LoyaltyAccount.objects.filter(user_id__in=user_ids).values_list("id", flat=True))
    transactions = Transaction.objects.filter(user_id__in=user_ids)
    item_counts = Counter(transactions.annotate(item_count=Count("items")).values_list("item_count", flat=True))
    return {
        "proof_users_total": int(len(users)),
        "transactions_total": int(transactions.count()),
        "single_item_transactions": int(item_counts.get(1, 0)),
        "multi_item_transactions": int(sum(count for item_count, count in item_counts.items() if int(item_count or 0) > 1)),
        "redeemed_assignments_total": int(OfferAssignment.objects.filter(user_id__in=user_ids, is_redeemed=True).count()),
        "active_redeemed_assignments": int(
            OfferAssignment.objects.filter(user_id__in=user_ids, is_redeemed=True, is_active=True).count()
        ),
        "offer_redeemed_events": int(
            OfferEvent.objects.filter(user_id__in=user_ids, event_type=OfferEvent.Type.REDEEMED).count()
        ),
        "loyalty_redeem_entries": int(
            LoyaltyLedgerEntry.objects.filter(account_id__in=account_ids, entry_type=LoyaltyLedgerEntry.Type.REDEEM).count()
        ),
        "loyalty_balance_mismatch_accounts": int(
            sum(
                1
                for account in LoyaltyAccount.objects.filter(id__in=account_ids)
                if int(account.points_balance or 0)
                != int(
                    LoyaltyLedgerEntry.objects.filter(account=account).aggregate(total=Sum("points_delta"))["total"]
                    or 0
                )
            )
        ),
        "usernames": [str(user.username) for user in users],
    }


def seed_commerce_proof_scenarios(*, seed: int = COMMERCE_PROOF_SEED, prefix: str = COMMERCE_PROOF_PREFIX) -> dict[str, Any]:
    _ensure_catalog_can_support_commerce_proof()
    ensure_demo_offers()
    users = _ensure_proof_users(prefix=prefix, seed=seed)
    user_by_slug = _scenario_user_map(users=users)

    offer_user = user_by_slug["offer"]
    combo_user = user_by_slug["combo"]
    points_user = user_by_slug["points"]

    scenario_runs = {
        "redeemed_offer_only": _run_offer_redemption_checkout(
            user=offer_user,
            label=f"{offer_user.username}:offer-only",
        ),
        "combo_offer_and_points": None,
        "points_redeem_only": None,
    }

    combo_warmup = _run_warmup_checkout(
        client=_authed_client(combo_user),
        label=f"{combo_user.username}:combo",
        preferred_category="skincare",
    )
    combo_balance = int(LoyaltyAccount.objects.get(user=combo_user).points_balance)
    combo_redeem_points = _redeem_points_request(combo_balance)
    combo_checkout = _run_offer_redemption_checkout(
        user=combo_user,
        label=f"{combo_user.username}:combo",
        redeem_points=combo_redeem_points,
    )
    scenario_runs["combo_offer_and_points"] = {
        "warmup": combo_warmup,
        "redeem": combo_checkout,
    }

    points_warmup = _run_warmup_checkout(
        client=_authed_client(points_user),
        label=f"{points_user.username}:points",
        preferred_category="haircare",
    )
    scenario_runs["points_redeem_only"] = {
        "warmup": points_warmup,
        "redeem": _run_points_redeem_checkout(
            user=points_user,
            label=f"{points_user.username}:points",
        ),
    }

    snapshot = build_commerce_proof_snapshot(prefix=prefix, seed=seed)
    if snapshot["redeemed_assignments_total"] < COMMERCE_PROOF_MIN_REDEEMED_ASSIGNMENTS:
        raise CommandError(f"Commerce proof only created {snapshot['redeemed_assignments_total']} redeemed assignments")
    if snapshot["loyalty_redeem_entries"] < COMMERCE_PROOF_MIN_LOYALTY_REDEEMS:
        raise CommandError(f"Commerce proof only created {snapshot['loyalty_redeem_entries']} loyalty redeem entries")
    if snapshot["multi_item_transactions"] < COMMERCE_PROOF_MIN_MULTI_ITEM_TRANSACTIONS:
        raise CommandError(f"Commerce proof only created {snapshot['multi_item_transactions']} multi-item transactions")
    if snapshot["active_redeemed_assignments"] > 0:
        raise CommandError(f"Commerce proof left {snapshot['active_redeemed_assignments']} redeemed assignments active")

    return {
        "seed": int(seed),
        "username_prefix": prefix,
        "password": DEMO_PASSWORD,
        "proof_users_created": int(len(users)),
        "scenarios": scenario_runs,
        "snapshot": snapshot,
        "thresholds": {
            "redeemed_assignments_total": COMMERCE_PROOF_MIN_REDEEMED_ASSIGNMENTS,
            "loyalty_redeem_entries": COMMERCE_PROOF_MIN_LOYALTY_REDEEMS,
            "multi_item_transactions": COMMERCE_PROOF_MIN_MULTI_ITEM_TRANSACTIONS,
        },
    }
