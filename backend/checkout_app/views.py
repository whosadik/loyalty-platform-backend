from __future__ import annotations

import logging
from decimal import Decimal, InvalidOperation
from datetime import timedelta

logger = logging.getLogger(__name__)

from django.db import transaction as db_tx
from django.utils import timezone
from django.db.models import Q, Sum

from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated
from rest_framework.exceptions import ValidationError

from checkout_app.serializers import (
    CheckoutCommitResponseSerializer,
    CheckoutLastResponseSerializer,
    CheckoutRequestSerializer,
)
from checkout_app.pricing import Line, apply_offer_to_totals
from transactions.models import CartItem, OwnedProduct, Transaction, TransactionItem
from transactions.serializers import TransactionSerializer
from offers.models import CampaignBudget, Offer, OfferAssignment, OfferEvent
from loyalty.models import LoyaltyAccount, LoyaltyLedgerEntry, Tier
from loyalty.points import DEFAULT_POINTS_RATE, get_effective_points_rate, get_tier_points_multiplier
from catalog.models import Product
from offers.services import expire_assignment_if_needed, get_or_assign_next_offer
from offers.events import record_offer_event
from roadmap_app.events import build_step_event_context, record_roadmap_event
from roadmap_app.models import RoadmapEvent
from roadmap_app.serializers import serialize_roadmap_step_snapshot
from roadmap_app.services import match_completed_steps_for_purchase, update_roadmap_from_purchase

from drf_spectacular.utils import OpenApiResponse, OpenApiExample, extend_schema, inline_serializer
from drf_spectacular.types import OpenApiTypes
from rest_framework import serializers

from django.db import IntegrityError
from backend.api_serializers import ApiErrorSerializer
from backend.request_language import get_request_language
from backend.throttles import CheckoutPreviewRateThrottle

from audit.logging import log_event
from audit.models import AuditEvent
from gift_cards.models import GiftCard, GiftCardLedgerEntry
from gift_cards.services import gift_card_snapshot, normalize_gift_card_code, refresh_gift_card_status
from recs_analytics.services import attribute_purchase


def _ensure_account(user) -> LoyaltyAccount:
    acc, _ = LoyaltyAccount.objects.get_or_create(user=user)
    if acc.tier_id is None:
        bronze, _ = Tier.objects.get_or_create(
            name="Bronze",
            defaults={"threshold_spend_90d": 0, "points_rate": DEFAULT_POINTS_RATE},
        )
        acc.tier = bronze
        acc.save(update_fields=["tier"])
    return acc


def _recalculate_tier(user, now) -> LoyaltyAccount:
    since = now - timedelta(days=90)
    spend = (
        Transaction.objects.filter(user=user, created_at__gte=since)
        .aggregate(s=Sum("total_amount"))["s"]
        or Decimal("0")
    )

    tiers = list(Tier.objects.all().values("id", "threshold_spend_90d"))
    tiers.sort(key=lambda t: Decimal(str(t["threshold_spend_90d"])))

    chosen_id = tiers[0]["id"] if tiers else None
    for t in tiers:
        if spend >= Decimal(str(t["threshold_spend_90d"])):
            chosen_id = t["id"]

    acc = _ensure_account(user)
    if chosen_id and acc.tier_id != chosen_id:
        acc.tier_id = chosen_id
        acc.save(update_fields=["tier"])
    return acc


def _raise_validation(message: str) -> None:
    err = ValidationError(message)
    err.detail = {"ok": False, "message": message}
    raise err


def _product_unit_price_or_raise(prod: Product) -> Decimal:
    if not bool(getattr(prod, "in_stock", False)):
        _raise_validation(f"Product {int(prod.id)} is out of stock")
    raw_price = getattr(prod, "price", None)
    if raw_price in (None, ""):
        _raise_validation(f"Product {int(prod.id)} has no valid price")
    try:
        unit_price = Decimal(str(raw_price))
    except (InvalidOperation, TypeError, ValueError):
        _raise_validation(f"Product {int(prod.id)} has no valid price")
    if not unit_price.is_finite():
        _raise_validation(f"Product {int(prod.id)} has no valid price")
    return unit_price


def _load_redeemable_gift_card_or_raise(code: str, *, lock: bool = False, now=None) -> GiftCard:
    normalized = normalize_gift_card_code(code)
    if not normalized:
        _raise_validation("Gift card code is required")

    queryset = GiftCard.objects.select_for_update() if lock else GiftCard.objects.all()
    try:
        gift_card = queryset.get(code=normalized)
    except GiftCard.DoesNotExist:
        _raise_validation("Gift card not found")

    refresh_gift_card_status(gift_card, now=now)
    if gift_card.status == GiftCard.Status.EXPIRED:
        _raise_validation("Gift card expired")
    if gift_card.status == GiftCard.Status.REFUNDED:
        _raise_validation("Gift card is no longer active")
    if gift_card.remaining_amount <= 0:
        _raise_validation("Gift card balance is empty")
    return gift_card


def _active_public_campaigns_qs(now, *, lock: bool = False):
    today = now.date()
    qs = (
        CampaignBudget.objects.filter(is_active=True, campaign_type=CampaignBudget.Type.PUBLIC)
        .filter(Q(start_date__isnull=True) | Q(start_date__lte=today))
        .filter(Q(end_date__isnull=True) | Q(end_date__gte=today))
        .order_by("priority", "id")
    )
    return qs.select_for_update() if lock else qs


def _week_start_date(now):
    return (now - timedelta(days=now.weekday())).date()


def _public_campaign_weekly_spent(campaign: CampaignBudget, now, *, reset: bool = False) -> Decimal:
    if hasattr(campaign, "week_start_date"):
        week_start = _week_start_date(now)
        if campaign.week_start_date != week_start:
            if reset:
                campaign.week_start_date = week_start
                campaign.weekly_spent = Decimal("0.00")
                campaign.save(update_fields=["week_start_date", "weekly_spent"])
            return Decimal("0.00")
    return Decimal(str(campaign.weekly_spent))


def _clean_strings(values) -> list[str]:
    out: list[str] = []
    for value in values or []:
        normalized = str(value).strip()
        if normalized and normalized not in out:
            out.append(normalized)
    return out


def _clean_ints(values) -> list[int]:
    out: list[int] = []
    for value in values or []:
        try:
            normalized = int(value)
        except (TypeError, ValueError):
            continue
        if normalized > 0 and normalized not in out:
            out.append(normalized)
    return out


def _intersect_or_union(left: list, right: list) -> list:
    if left and right:
        right_set = set(right)
        return [value for value in left if value in right_set]
    return left or right


def _intersect_brands_or_union(left: list[str], right: list[str]) -> list[str]:
    if left and right:
        right_lookup = {value.casefold() for value in right}
        return [value for value in left if value.casefold() in right_lookup]
    return left or right


def _public_offer_constraints(offer: Offer, campaign: CampaignBudget) -> dict:
    offer_categories = _clean_strings(offer.allowed_categories)
    campaign_categories = _clean_strings(campaign.allowed_categories)
    offer_product_types = _clean_strings(offer.allowed_product_types)
    campaign_product_types = _clean_strings(campaign.allowed_steps)
    offer_brands = _clean_strings(getattr(offer, "allowed_brands", []))
    campaign_brands = _clean_strings(getattr(campaign, "allowed_brands", []))
    offer_product_ids = _clean_ints(getattr(offer, "allowed_product_ids", []))
    campaign_product_ids = _clean_ints(getattr(campaign, "allowed_product_ids", []))

    categories = _intersect_or_union(offer_categories, campaign_categories)
    product_types = _intersect_or_union(offer_product_types, campaign_product_types)
    brands = _intersect_brands_or_union(offer_brands, campaign_brands)
    product_ids = _intersect_or_union(offer_product_ids, campaign_product_ids)
    impossible = (
        (offer_categories and campaign_categories and not categories)
        or (offer_product_types and campaign_product_types and not product_types)
        or (offer_brands and campaign_brands and not brands)
        or (offer_product_ids and campaign_product_ids and not product_ids)
    )
    return {
        "impossible": impossible,
        "categories": categories,
        "product_types": product_types,
        "brands": brands,
        "product_ids": product_ids,
    }


def _line_matches_constraints(line: Line, constraints: dict) -> bool:
    product = line.product
    categories = constraints.get("categories") or []
    product_types = constraints.get("product_types") or []
    brands = constraints.get("brands") or []
    product_ids = constraints.get("product_ids") or []

    if constraints.get("impossible"):
        return False

    if categories and product.category not in categories:
        return False
    if product_types and product.product_type not in product_types:
        return False
    if product_ids and int(product.id) not in product_ids:
        return False
    if brands:
        brand_lookup = {str(value).strip().casefold() for value in brands}
        if str(product.brand or "").strip().casefold() not in brand_lookup:
            return False
    return True


def _target_constraints_payload(constraints: dict) -> dict:
    payload = {}
    if constraints.get("brands"):
        payload["brands"] = constraints["brands"]
    if constraints.get("product_ids"):
        payload["product_ids"] = constraints["product_ids"]
    if constraints.get("product_types"):
        payload["product_types"] = constraints["product_types"]
    return payload


def _public_offer_targets(offer: Offer, campaign: CampaignBudget, lines: list[Line]) -> list[dict]:
    constraints = _public_offer_constraints(offer, campaign)
    eligible_lines = [line for line in lines if _line_matches_constraints(line, constraints)]
    if not eligible_lines:
        return []

    categories = constraints["categories"]
    product_types = constraints["product_types"]
    brands = constraints["brands"]
    product_ids = constraints["product_ids"]

    line_categories = sorted({str(ln.product.category) for ln in lines if getattr(ln.product, "category", None)})
    line_brands = sorted({str(ln.product.brand).strip() for ln in eligible_lines if str(ln.product.brand).strip()})
    extra_constraints = _target_constraints_payload(constraints)

    if offer.target_scope == "cart":
        if product_ids:
            return [
                {
                    "scope": "product_id",
                    "value": line.product.id,
                    "category": line.product.category,
                    "brand": line.product.brand,
                    "product_type": line.product.product_type,
                }
                for line in eligible_lines
            ]
        if brands:
            return [{"scope": "brand", "value": brand, **extra_constraints} for brand in brands]
        if categories or product_types:
            if product_types:
                return [
                    {
                        "scope": "product_type",
                        "value": product_type,
                        **({"category": category} if category else {}),
                        **extra_constraints,
                    }
                    for category in (categories or [None])
                    for product_type in product_types
                ]
            return [{"scope": "category", "value": category, **extra_constraints} for category in categories]
        return [{"scope": "cart"}]

    if offer.target_scope == "category":
        target_categories = categories or line_categories
        return [{"scope": "category", "value": category, **extra_constraints} for category in target_categories]

    if offer.target_scope == "brand":
        target_brands = brands or line_brands
        return [{"scope": "brand", "value": brand, **extra_constraints} for brand in target_brands]

    if offer.target_scope == "product_type":
        if product_types:
            category_options = categories or [None]
            return [
                {
                    "scope": "product_type",
                    "value": product_type,
                    **({"category": category} if category else {}),
                    **extra_constraints,
                }
                for category in category_options
                for product_type in product_types
            ]
        return [
            {"scope": "product_type", "category": category, **extra_constraints}
            for category in (categories or line_categories)
        ]

    if offer.target_scope == "product_id":
        targets = []
        for line in eligible_lines:
            product = line.product
            targets.append(
                {
                    "scope": "product_id",
                    "value": product.id,
                    "category": product.category,
                    "brand": product.brand,
                    "product_type": product.product_type,
                }
            )
        return targets

    return []


def _public_offer_payload(offer: Offer, campaign: CampaignBudget, target: dict) -> dict:
    return {
        "assignment_id": None,
        "source": "public_campaign",
        "public_campaign": True,
        "campaign": {
            "id": campaign.id,
            "name": campaign.name,
            "type": campaign.campaign_type,
        },
        "offer": {
            "id": offer.id,
            "name": offer.name,
            "type": offer.offer_type,
            "value": str(offer.value),
        },
        "target": target,
    }


def _personal_offer_payload(assignment: OfferAssignment, target: dict) -> dict:
    campaign = getattr(assignment.offer, "campaign", None)
    campaign_type = getattr(campaign, "campaign_type", None) if campaign else None
    return {
        "assignment_id": assignment.id,
        "source": "public_campaign" if campaign_type == CampaignBudget.Type.PUBLIC else "personal_system",
        "public_campaign": campaign_type == CampaignBudget.Type.PUBLIC,
        "campaign": (
            {
                "id": campaign.id,
                "name": campaign.name,
                "type": campaign_type,
            }
            if campaign
            else None
        ),
        "offer": {
            "id": assignment.offer.id,
            "name": assignment.offer.name,
            "type": assignment.offer.offer_type,
            "value": str(assignment.offer.value),
        },
        "target": target,
    }


def _find_public_campaign_offer(
    *,
    lines: list[Line],
    points_rate: Decimal,
    tier_points_multiplier: Decimal = Decimal("1"),
    redeem_points: int = 0,
    gift_card_balance: Decimal = Decimal("0"),
    now,
    lock: bool = False,
) -> tuple[Offer | None, CampaignBudget | None, dict | None, dict | None]:
    campaigns = list(_active_public_campaigns_qs(now, lock=lock).prefetch_related("offers"))
    best: tuple[Decimal, int, int, Offer, CampaignBudget, dict, dict] | None = None

    for campaign in campaigns:
        left = Decimal(str(campaign.weekly_limit)) - _public_campaign_weekly_spent(
            campaign,
            now,
            reset=lock,
        )
        if left <= 0:
            continue

        offers = [
            offer
            for offer in campaign.offers.all()
            if offer.is_active and offer.offer_type == Offer.Type.DISCOUNT
        ]
        for offer in offers:
            for target in _public_offer_targets(offer, campaign, lines):
                calc = apply_offer_to_totals(
                    offer_type=offer.offer_type,
                    offer_value=Decimal(str(offer.value)),
                    target=target,
                    lines=lines,
                    points_rate=points_rate,
                    tier_points_multiplier=tier_points_multiplier,
                    redeem_points=redeem_points,
                    gift_card_balance=gift_card_balance,
                )
                if not calc.get("ok"):
                    continue

                discount_amount = Decimal(str(calc.get("discount_amount") or "0"))
                if discount_amount <= 0 or discount_amount > left:
                    continue

                candidate = (
                    discount_amount,
                    -int(campaign.priority),
                    -int(offer.id),
                    offer,
                    campaign,
                    target,
                    calc,
                )
                if best is None or candidate[:3] > best[:3]:
                    best = candidate

    if best is None:
        return None, None, None, None

    _, _, _, offer, campaign, target, calc = best
    return offer, campaign, target, calc


def _list_active_personal_assignments(user, now) -> list[OfferAssignment]:
    assignments = list(
        OfferAssignment.objects.select_related("offer", "offer__campaign")
        .filter(user=user, is_active=True, is_redeemed=False)
    )

    out: list[OfferAssignment] = []
    for assignment in assignments:
        expires_at = getattr(assignment, "expires_at", None)
        if expires_at and expires_at <= now:
            continue
        if assignment.offer is None or not assignment.offer.is_active:
            continue
        campaign = getattr(assignment.offer, "campaign", None)
        if campaign is not None and not campaign.is_active:
            continue
        out.append(assignment)
    return out


def _list_applicable_public_candidates(
    *,
    lines: list[Line],
    points_rate: Decimal,
    tier_points_multiplier: Decimal,
    redeem_points: int,
    gift_card_balance: Decimal,
    now,
) -> list[dict]:
    """Return every applicable public-campaign offer with its calculated discount.

    Unlike `_find_public_campaign_offer`, this does not pick a single winner — it
    surfaces all viable options so the user can choose.
    """
    campaigns = list(_active_public_campaigns_qs(now, lock=False).prefetch_related("offers"))
    candidates: list[dict] = []
    seen_keys: set[tuple[int, int]] = set()

    for campaign in campaigns:
        weekly_limit = Decimal(str(campaign.weekly_limit or 0))
        left = weekly_limit - _public_campaign_weekly_spent(campaign, now, reset=False) if weekly_limit > 0 else None

        offers = [
            offer
            for offer in campaign.offers.all()
            if offer.is_active and offer.offer_type == Offer.Type.DISCOUNT
        ]
        for offer in offers:
            best_for_offer: tuple[Decimal, dict, dict] | None = None
            for target in _public_offer_targets(offer, campaign, lines):
                calc = apply_offer_to_totals(
                    offer_type=offer.offer_type,
                    offer_value=Decimal(str(offer.value)),
                    target=target,
                    lines=lines,
                    points_rate=points_rate,
                    tier_points_multiplier=tier_points_multiplier,
                    redeem_points=redeem_points,
                    gift_card_balance=gift_card_balance,
                )
                if not calc.get("ok"):
                    continue
                discount = Decimal(str(calc.get("discount_amount") or "0"))
                if discount <= 0:
                    continue
                if left is not None and discount > left:
                    continue
                if best_for_offer is None or discount > best_for_offer[0]:
                    best_for_offer = (discount, target, calc)

            if best_for_offer is None:
                continue

            key = (int(campaign.id), int(offer.id))
            if key in seen_keys:
                continue
            seen_keys.add(key)

            discount_amount, target, _ = best_for_offer
            candidates.append(
                {
                    "kind": "public",
                    "assignment_id": None,
                    "public_offer_id": int(offer.id),
                    "public_campaign_id": int(campaign.id),
                    "discount_amount": str(discount_amount),
                    "offer_payload": _public_offer_payload(offer, campaign, target),
                    "priority": int(campaign.priority or 0),
                }
            )
    return candidates


def _collect_available_offers(
    *,
    user,
    lines: list[Line],
    points_rate: Decimal,
    tier_points_multiplier: Decimal,
    redeem_points: int,
    gift_card_balance: Decimal,
    now,
) -> list[dict]:
    """Return every applicable offer (personal + public) ordered by discount desc.

    Each entry is a flat dict ready for the API response; the actual
    `applied_offer` payload sits under `offer_payload` (same shape produced by
    `_personal_offer_payload` / `_public_offer_payload`) so the frontend can
    reuse existing rendering logic.
    """
    candidates: list[dict] = []

    for assignment in _list_active_personal_assignments(user, now):
        target = assignment.target or {"scope": "cart"}
        calc = apply_offer_to_totals(
            offer_type=assignment.offer.offer_type,
            offer_value=Decimal(str(assignment.offer.value)),
            target=target,
            lines=lines,
            points_rate=points_rate,
            tier_points_multiplier=tier_points_multiplier,
            redeem_points=redeem_points,
            gift_card_balance=gift_card_balance,
        )
        if not calc.get("ok"):
            continue
        discount = Decimal(str(calc.get("discount_amount") or "0"))
        if discount <= 0:
            # Personal offer doesn't match current cart — skip from available list.
            continue
        candidates.append(
            {
                "kind": "personal",
                "assignment_id": int(assignment.id),
                "public_offer_id": None,
                "public_campaign_id": None,
                "discount_amount": str(discount),
                "offer_payload": _personal_offer_payload(assignment, target),
                "priority": 0,
            }
        )

    candidates.extend(
        _list_applicable_public_candidates(
            lines=lines,
            points_rate=points_rate,
            tier_points_multiplier=tier_points_multiplier,
            redeem_points=redeem_points,
            gift_card_balance=gift_card_balance,
            now=now,
        )
    )

    def _sort_key(item: dict):
        try:
            discount = Decimal(item.get("discount_amount") or "0")
        except (InvalidOperation, TypeError, ValueError):
            discount = Decimal("0")
        # Higher discount first; personal wins ties; lower campaign priority wins next tie.
        kind_rank = 0 if item.get("kind") == "personal" else 1
        return (-discount, kind_rank, item.get("priority", 0))

    candidates.sort(key=_sort_key)
    return candidates


def _resolve_public_offer_by_id(
    *,
    lines: list[Line],
    points_rate: Decimal,
    tier_points_multiplier: Decimal,
    redeem_points: int,
    gift_card_balance: Decimal,
    now,
    public_offer_id: int,
    lock: bool = False,
) -> tuple[Offer, CampaignBudget, dict, dict]:
    """Resolve a specific public offer for the cart, picking its best target.

    Raises ValidationError if the offer is not applicable to the current cart
    or its campaign is inactive / over budget.
    """
    try:
        offer = Offer.objects.select_related("campaign").get(id=public_offer_id)
    except Offer.DoesNotExist:
        _raise_validation("Public offer not found")

    campaign = offer.campaign
    if campaign is None or campaign.campaign_type != CampaignBudget.Type.PUBLIC:
        _raise_validation("Offer is not part of a public campaign")
    if not campaign.is_active or not offer.is_active:
        _raise_validation("Public offer is not active")

    today = now.date()
    if campaign.start_date and today < campaign.start_date:
        _raise_validation("Campaign has not started yet")
    if campaign.end_date and today > campaign.end_date:
        _raise_validation("Campaign has ended")

    weekly_limit = Decimal(str(campaign.weekly_limit or 0))
    left = weekly_limit - _public_campaign_weekly_spent(campaign, now, reset=lock) if weekly_limit > 0 else None

    best: tuple[Decimal, dict, dict] | None = None
    for target in _public_offer_targets(offer, campaign, lines):
        calc = apply_offer_to_totals(
            offer_type=offer.offer_type,
            offer_value=Decimal(str(offer.value)),
            target=target,
            lines=lines,
            points_rate=points_rate,
            tier_points_multiplier=tier_points_multiplier,
            redeem_points=redeem_points,
            gift_card_balance=gift_card_balance,
        )
        if not calc.get("ok"):
            continue
        discount = Decimal(str(calc.get("discount_amount") or "0"))
        if discount <= 0:
            continue
        if left is not None and discount > left:
            continue
        if best is None or discount > best[0]:
            best = (discount, target, calc)

    if best is None:
        _raise_validation("Selected promotion does not apply to this cart")

    _, target, calc = best
    return offer, campaign, target, calc


class CheckoutView(APIView):
    permission_classes = [IsAuthenticated]

    @extend_schema(
        tags=["Checkout"],
        description=(
            "Commit checkout transaction. "
            "Returns 201 for a fresh commit and 200 with idempotent_replay=true for a replayed idempotency key."
        ),
        request=CheckoutRequestSerializer,
        responses={
            201: CheckoutCommitResponseSerializer,
            200: CheckoutCommitResponseSerializer,
            400: OpenApiResponse(response=ApiErrorSerializer),
            409: OpenApiResponse(response=ApiErrorSerializer, description="Duplicate idempotency key without stored payload"),
        },
    )
    def post(self, request):
        language = get_request_language(request)
        req = CheckoutRequestSerializer(data=request.data)
        req.is_valid(raise_exception=True)
        data = req.validated_data

        idem = data.get("idempotency_key")
        if idem:
            prev = Transaction.objects.filter(user=request.user, idempotency_key=idem).first()
            if prev and (prev.pricing_meta or {}).get("type") == "gift_card_purchase":
                return Response({"ok": False, "message": "Duplicate idempotency_key"}, status=409)
            if prev and prev.pricing_meta:
                log_event(
                    request=request,
                    action=AuditEvent.Action.CHECKOUT_REPLAY,
                    entity_type="Transaction",
                    entity_id=prev.id,
                    status_code=200,
                    meta={"idempotency_key": idem},
                )
                return Response({"ok": True, "idempotent_replay": True, **prev.pricing_meta})

        now = timezone.now()

        with db_tx.atomic():
            # lock loyalty account for safe redeem/earn
            account = _ensure_account(request.user)
            account = LoyaltyAccount.objects.select_for_update().get(id=account.id)
            tier_before = account.tier.name if account.tier else None

            # 1) create transaction + items
            try:
                txn = Transaction.objects.create(
                    user=request.user,
                    channel=data.get("channel", "offline"),
                    idempotency_key=idem,
                )
            except IntegrityError:
                prev = Transaction.objects.get(user=request.user, idempotency_key=idem)
                if (prev.pricing_meta or {}).get("type") == "gift_card_purchase":
                    return Response({"ok": False, "message": "Duplicate idempotency_key"}, status=409)
                if prev.pricing_meta:
                    log_event(
                        request=request,
                        action=AuditEvent.Action.CHECKOUT_REPLAY,
                        entity_type="Transaction",
                        entity_id=prev.id,
                        status_code=200,
                        meta={"idempotency_key": idem, "source": "integrity_error"},
                    )
                    return Response({"ok": True, "idempotent_replay": True, **prev.pricing_meta})

                log_event(
                    request=request,
                    action=AuditEvent.Action.CHECKOUT_REPLAY,
                    entity_type="Transaction",
                    entity_id=prev.id,
                    status_code=409,
                    meta={"idempotency_key": idem, "source": "integrity_error", "pricing_meta_missing": True},
                )
                return Response({"ok": False, "message": "Duplicate idempotency_key"}, status=409)


            total = Decimal("0")
            created_items: list[TransactionItem] = []

            for it in data["items"]:
                product_id = it["product"]
                qty = int(it["quantity"])
                # ensure product exists
                prod = Product.objects.select_for_update().get(id=product_id)
                unit_price = _product_unit_price_or_raise(prod)

                ti = TransactionItem.objects.create(
                    transaction=txn,
                    product=prod,
                    quantity=qty,
                    unit_price=unit_price,
                )
                created_items.append(ti)
                total += unit_price * qty

                # owned update (qty/active/last_acquired)
                owned, _ = OwnedProduct.objects.get_or_create(user=request.user, product=prod)
                owned.quantity_total = int(owned.quantity_total or 0) + qty
                owned.is_active = True
                owned.last_acquired_at = now
                owned.save(update_fields=["quantity_total", "is_active", "last_acquired_at"])

            lines = [Line(product=it.product, quantity=int(it.quantity), unit_price=it.unit_price) for it in created_items]

            requested_redeem_points = int(data.get("redeem_points") or 0)
            if requested_redeem_points > account.points_balance:
                _raise_validation("Insufficient points")
            gift_card = None
            if data.get("gift_card_code"):
                gift_card = _load_redeemable_gift_card_or_raise(
                    data["gift_card_code"],
                    lock=True,
                    now=now,
                )

            points_rate = get_effective_points_rate(
                account.tier.points_rate if account.tier else DEFAULT_POINTS_RATE
            )
            tier_points_multiplier = get_tier_points_multiplier(account.tier.name if account.tier else None)

            # 2) optional offer apply (discount / points_multiplier) via shared pricing
            applied_assignment_id = None
            applied_target = None
            applied_offer_payload = None
            applied_public_offer_id = None
            applied_public_campaign_id = None
            offer_type = "discount"
            offer_value = Decimal("0")
            target = {"scope": "cart"}
            assignment = None
            public_offer = None
            public_campaign = None
            public_calc = None

            apply_assignment_id = data.get("apply_assignment_id")
            apply_public_offer_id_req = data.get("apply_public_offer_id")
            if apply_assignment_id is not None:
                assignment = (
                    OfferAssignment.objects.select_for_update()
                    .select_related("offer", "offer__campaign")
                    .get(id=apply_assignment_id, user=request.user)
                )

                if assignment.is_redeemed:
                    _raise_validation("Offer already redeemed")
                request_id = getattr(request, "request_id", None) or request.headers.get("X-Request-ID")
                if expire_assignment_if_needed(
                    assignment,
                    now=now,
                    source="checkout_offer_validation",
                    request_id=request_id,
                ):
                    _raise_validation("Offer expired")
                if not assignment.is_active:
                    _raise_validation("Offer is no longer active")

                offer_type = assignment.offer.offer_type
                offer_value = Decimal(str(assignment.offer.value))
                target = assignment.target or {"scope": "cart"}
                applied_target = target
                applied_offer_payload = _personal_offer_payload(assignment, target)
            elif apply_public_offer_id_req is not None:
                public_offer, public_campaign, public_target, public_calc = _resolve_public_offer_by_id(
                    lines=lines,
                    points_rate=points_rate,
                    tier_points_multiplier=tier_points_multiplier,
                    redeem_points=requested_redeem_points,
                    gift_card_balance=gift_card.remaining_amount if gift_card else Decimal("0"),
                    now=now,
                    public_offer_id=int(apply_public_offer_id_req),
                    lock=True,
                )
                offer_type = public_offer.offer_type
                offer_value = Decimal(str(public_offer.value))
                target = public_target
                applied_target = target
                applied_public_offer_id = public_offer.id
                applied_public_campaign_id = public_campaign.id
                applied_offer_payload = _public_offer_payload(public_offer, public_campaign, target)
            else:
                public_offer, public_campaign, public_target, public_calc = _find_public_campaign_offer(
                    lines=lines,
                    points_rate=points_rate,
                    tier_points_multiplier=tier_points_multiplier,
                    redeem_points=requested_redeem_points,
                    gift_card_balance=gift_card.remaining_amount if gift_card else Decimal("0"),
                    now=now,
                    lock=True,
                )
                if public_offer is not None and public_campaign is not None and public_target is not None:
                    offer_type = public_offer.offer_type
                    offer_value = Decimal(str(public_offer.value))
                    target = public_target
                    applied_target = target
                    applied_public_offer_id = public_offer.id
                    applied_public_campaign_id = public_campaign.id
                    applied_offer_payload = _public_offer_payload(public_offer, public_campaign, target)

            calc = public_calc or apply_offer_to_totals(
                    offer_type=offer_type,
                    offer_value=offer_value,
                    target=target,
                    lines=lines,
                    points_rate=points_rate,
                    tier_points_multiplier=tier_points_multiplier,
                    redeem_points=requested_redeem_points,
                    gift_card_balance=gift_card.remaining_amount if gift_card else Decimal("0"),
                )

            if not calc["ok"]:
                _raise_validation(calc.get("message", "Offer not applicable"))

            gross_total = Decimal(calc["gross_total"])
            discount_amount = Decimal(calc["discount_amount"])
            net_total = Decimal(calc["net_total"])
            eligible_total = Decimal(calc["eligible_total"])
            gift_card_applied_amount = Decimal(calc["gift_card_applied_amount"])
            points_redeemed = int(calc["points_redeemed"])
            base_points = int(calc["base_points"])
            points_earned = int(calc["estimated_points_earned"])
            points_multiplier = Decimal(calc["points_multiplier"])
            tier_points_multiplier = Decimal(calc["tier_points_multiplier"])
            gift_card_payload = None

            txn.total_amount = net_total
            txn.save(update_fields=["total_amount"])

            # 3) recalc tier (based on actual paid amount over the last 90 days, including this txn)
            account = _recalculate_tier(request.user, now)
            account = LoyaltyAccount.objects.select_for_update().get(id=account.id)
            recalculated_points_rate = get_effective_points_rate(
                account.tier.points_rate if account.tier else DEFAULT_POINTS_RATE
            )
            recalculated_tier_points_multiplier = get_tier_points_multiplier(account.tier.name if account.tier else None)
            if (
                recalculated_points_rate != points_rate
                or recalculated_tier_points_multiplier != tier_points_multiplier
            ):
                points_rate = recalculated_points_rate
                tier_points_multiplier = recalculated_tier_points_multiplier
                calc = apply_offer_to_totals(
                    offer_type=offer_type,
                    offer_value=offer_value,
                    target=target,
                    lines=lines,
                    points_rate=points_rate,
                    tier_points_multiplier=tier_points_multiplier,
                    redeem_points=requested_redeem_points,
                    gift_card_balance=gift_card.remaining_amount if gift_card else Decimal("0"),
                )
                if not calc["ok"]:
                    _raise_validation(calc.get("message", "Offer not applicable"))
                gross_total = Decimal(calc["gross_total"])
                discount_amount = Decimal(calc["discount_amount"])
                net_total = Decimal(calc["net_total"])
                eligible_total = Decimal(calc["eligible_total"])
                gift_card_applied_amount = Decimal(calc["gift_card_applied_amount"])
                points_redeemed = int(calc["points_redeemed"])
                base_points = int(calc["base_points"])
                points_earned = int(calc["estimated_points_earned"])
                points_multiplier = Decimal(calc["points_multiplier"])

            if assignment is not None:
                assignment.is_redeemed = True
                assignment.is_active = False
                assignment.redeemed_transaction_id = txn.id
                assignment.save(update_fields=["is_redeemed", "is_active", "redeemed_transaction_id"])
                request_id = getattr(request, "request_id", None) or request.headers.get("X-Request-ID")
                record_offer_event(
                    assignment,
                    OfferEvent.Type.REDEEMED,
                    request_id=request_id,
                    context={"endpoint": "POST /api/checkout", "variant": "v1"},
                )
                applied_assignment_id = assignment.id

            if public_campaign is not None and public_offer is not None and discount_amount > 0:
                public_campaign.weekly_spent = Decimal(str(public_campaign.weekly_spent)) + discount_amount
                public_campaign.save(update_fields=["weekly_spent"] + (["week_start_date"] if hasattr(public_campaign, "week_start_date") else []))

            if points_redeemed > 0:
                LoyaltyLedgerEntry.objects.create(
                    account=account,
                    entry_type=LoyaltyLedgerEntry.Type.REDEEM,
                    points_delta=-points_redeemed,
                    reference=f"checkout:txn:{txn.id}",
                    meta={"txn_id": txn.id, "net_total": str(net_total)},
                )
                account.points_balance -= points_redeemed

            if gift_card is not None:
                balance_before = Decimal(gift_card.remaining_amount)
                balance_after = max(balance_before - gift_card_applied_amount, Decimal("0"))
                gift_card.remaining_amount = balance_after
                gift_card.status = (
                    GiftCard.Status.EXHAUSTED if balance_after <= 0 else GiftCard.Status.ACTIVE
                )
                gift_card.save(update_fields=["remaining_amount", "status", "updated_at"])
                if gift_card_applied_amount > 0:
                    GiftCardLedgerEntry.objects.create(
                        gift_card=gift_card,
                        entry_type=GiftCardLedgerEntry.EntryType.REDEEM,
                        amount_delta=-gift_card_applied_amount,
                        transaction=txn,
                        meta={
                            "txn_id": txn.id,
                            "gross_total": str(gross_total),
                            "net_total": str(net_total),
                        },
                    )
                gift_card_payload = gift_card_snapshot(
                    gift_card,
                    applied_amount=gift_card_applied_amount,
                    balance_before=balance_before,
                    balance_after=balance_after,
                )

            LoyaltyLedgerEntry.objects.create(
                account=account,
                entry_type=LoyaltyLedgerEntry.Type.EARN,
                points_delta=points_earned,
                reference=f"checkout:txn:{txn.id}",
                meta={
                    "txn_id": txn.id,
                    "gross_total": str(gross_total),
                    "discount_amount": str(discount_amount),
                    "net_total": str(net_total),
                    "tier": account.tier.name if account.tier else None,
                    "points_rate": str(points_rate),
                    "base_points": base_points,
                    "multiplier": str(points_multiplier),
                    "tier_points_multiplier": str(tier_points_multiplier),
                    "tier_adjusted_points": int(calc.get("tier_adjusted_points") or 0),
                    "offer_assignment_id": applied_assignment_id,
                    "public_campaign_id": applied_public_campaign_id,
                    "public_offer_id": applied_public_offer_id,
                    "target": applied_target,
                    "eligible_total": str(eligible_total),
                    "points_redeemed": points_redeemed,
                    "gift_card_applied_amount": str(gift_card_applied_amount),
                },
            )

            account.points_balance += points_earned
            account.save(update_fields=["points_balance"])
            purchased_categories = sorted(list({it.product.category for it in created_items}))
            purchased_types = sorted(list({it.product.product_type for it in created_items}))
            purchased_ids = [it.product_id for it in created_items]

            post_ctx = {
                "categories": purchased_categories,
                "product_types": purchased_types,
                "product_ids": purchased_ids,
                "transaction_id": txn.id,
            }
            request_id = getattr(request, "request_id", None) or request.headers.get("X-Request-ID")
            completed_matches = []
            try:
                completed_matches = match_completed_steps_for_purchase(request.user, post_ctx)
            except Exception:
                completed_matches = []
            # Record STEP_COMPLETED before roadmap refresh. Refresh reuses the same step rows and
            # emits STEP_GENERATED, so completing after refresh would break generation->exposure
            # attribution windows for the just-finished step.
            for match in completed_matches:
                step = match.get("step")
                plan = match.get("plan")
                category = str(match.get("category") or "")
                if not step:
                    continue
                try:
                    record_roadmap_event(
                        user=request.user,
                        event_type=RoadmapEvent.Type.STEP_COMPLETED,
                        plan=plan,
                        step=step,
                        request_id=request_id,
                        context=build_step_event_context(
                            category=category,
                            step=step,
                            offer_assignment_id=applied_assignment_id,
                            transaction_id=txn.id,
                            extra={
                                "matched_by": match.get("matched_by"),
                                "match_meta": match.get("match_meta") or {},
                            },
                        ),
                    )
                except Exception:
                    pass

            roadmap_ctx = None
            next_roadmap_step = None
            try:
                roadmap_result = update_roadmap_from_purchase(request.user, post_ctx)
                roadmap_ctx = (roadmap_result or {}).get("roadmap_ctx")
                next_roadmap_step = serialize_roadmap_step_snapshot(
                    (roadmap_result or {}).get("next_missing_step"),
                    category=(roadmap_result or {}).get("category"),
                    plan_id=getattr((roadmap_result or {}).get("plan"), "id", None),
                    language=language,
                )
            except Exception:
                logger.exception(
                    "update_roadmap_from_purchase failed for user=%s txn=%s",
                    request.user.id,
                    txn.id,
                )
                roadmap_ctx = None
                next_roadmap_step = None

            # Auto-assign next offer after successful checkout
            next_assignment = get_or_assign_next_offer(
                user=request.user,
                now=now,
                context_steps=None,
                post_ctx=post_ctx,
                roadmap_ctx=roadmap_ctx,
            )

            next_offer_payload = None
            if next_assignment:
                expires_at_value = getattr(next_assignment, "expires_at", None)
                next_offer_payload = _personal_offer_payload(next_assignment, next_assignment.target)
                next_offer_payload["reason"] = next_assignment.reason
                next_offer_payload["expires_at"] = expires_at_value.isoformat() if expires_at_value else None
                next_offer_payload["offer"]["estimated_cost"] = str(
                    getattr(next_assignment.offer, "estimated_cost", "")
                )

                # AUDIT: next offer assigned
                t = next_assignment.target or {}
                log_event(
                    request=request,
                    action=AuditEvent.Action.NEXT_OFFER_ASSIGNED,
                    entity_type="OfferAssignment",
                    entity_id=next_assignment.id,
                    status_code=201,
                    meta={
                        "picked_via": t.get("picked_via"),
                        "scope": t.get("scope"),
                        "value": t.get("value"),
                        "category": t.get("category"),
                        "product_type": t.get("product_type"),
                    },
                )

            payload = {
                "transaction_id": txn.id,
                "gross_total": str(gross_total),
                "discount_amount": str(discount_amount),
                "net_total": str(net_total),
                "offer_applied": applied_assignment_id is not None or applied_public_offer_id is not None,
                "offer_assignment_id": applied_assignment_id,
                "public_campaign_id": applied_public_campaign_id,
                "public_offer_id": applied_public_offer_id,
                "applied_offer": applied_offer_payload,
                "target": applied_target,
                "eligible_total": str(eligible_total),
                "points_redeemed": points_redeemed,
                "points_earned": points_earned,
                "tier_points_multiplier": str(tier_points_multiplier),
                "new_balance": account.points_balance,
                "gift_card": gift_card_payload,
                "tier": account.tier.name if account.tier else None,
                "new_tier": account.tier.name if account.tier else None,
                "tier_upgraded": bool(account.tier and account.tier.name != tier_before),
                "next_offer": next_offer_payload,
                "next_roadmap_step": next_roadmap_step,
            }

            # сохраняем снимок результата для replay
            cart_items = {
                item.product_id: item
                for item in CartItem.objects.select_for_update().filter(
                    user=request.user,
                    product_id__in=purchased_ids,
                )
            }
            for purchased_item in created_items:
                cart_item = cart_items.get(purchased_item.product_id)
                if not cart_item:
                    continue
                remaining_quantity = int(cart_item.quantity or 0) - int(purchased_item.quantity or 0)
                if remaining_quantity <= 0:
                    cart_item.delete()
                    continue
                cart_item.quantity = remaining_quantity
                cart_item.save(update_fields=["quantity", "updated_at"])
            Transaction.objects.filter(id=txn.id).update(pricing_meta=payload)

            # AUDIT: checkout created
            log_event(
                request=request,
                action=AuditEvent.Action.CHECKOUT_CREATED,
                entity_type="Transaction",
                entity_id=txn.id,
                status_code=201,
                meta={
                    "idempotency_key": idem,
                    "gross_total": str(gross_total),
                    "net_total": str(net_total),
                    "discount_amount": str(discount_amount),
                    "offer_applied": bool(applied_assignment_id or applied_public_offer_id),
                    "offer_assignment_id": applied_assignment_id,
                    "public_campaign_id": applied_public_campaign_id,
                    "public_offer_id": applied_public_offer_id,
                    "points_redeemed": points_redeemed,
                    "gift_card_applied_amount": str(gift_card_applied_amount),
                    "points_earned": points_earned,
                    "items_count": len(created_items),
                    "channel": data.get("channel", "offline"),
                },
            )
        try:
            attribute_purchase(
                user=request.user,
                purchased_product_ids=[int(x) for x in purchased_ids], 
                window_days=7,
                request_id=getattr(request, "request_id", None),
            )
        except Exception:
            pass

        return Response({"ok": True, **payload}, status=201)


class CheckoutLastView(APIView):
    permission_classes = [IsAuthenticated]

    @extend_schema(
        tags=["Checkout"],
        description="Read-only snapshot of the latest checkout for the authenticated user.",
        responses={200: CheckoutLastResponseSerializer},
    )
    def get(self, request):
        language = get_request_language(request)
        txn = (
            Transaction.objects.filter(user=request.user)
            .filter(items__isnull=False)
            .prefetch_related("items__product")
            .order_by("-created_at", "-id")
            .distinct()
            .first()
        )
        if not txn:
            return Response({"ok": True, "checkout": None})
        return Response(
            {
                "ok": True,
                "checkout": TransactionSerializer(
                    txn,
                    context={"request": request, "language": language},
                ).data,
            }
        )

class CheckoutPreviewView(APIView):
    permission_classes = [IsAuthenticated]
    throttle_classes = [CheckoutPreviewRateThrottle]

    @extend_schema(
        tags=["Checkout"],
        description="Full checkout preview: offer + points redeem (no DB writes).",
        request=CheckoutRequestSerializer,
        responses={
            200: inline_serializer(
                name="CheckoutPreviewResponse",
                fields={
                    "ok": serializers.BooleanField(),
                    "gross_total": serializers.CharField(),
                    "discount_amount": serializers.CharField(),
                    "net_total": serializers.CharField(),
                    "offer_applied": serializers.BooleanField(),
                    "applied_offer": serializers.DictField(allow_null=True),
                    "eligible_total": serializers.CharField(),
                    "estimated_points_earned": serializers.IntegerField(),
                    "tier_points_multiplier": serializers.CharField(),
                    "points_redeemed": serializers.IntegerField(),
                    "gift_card": serializers.JSONField(allow_null=True, required=False),
                    "balance_before": serializers.IntegerField(),
                    "balance_after_estimated": serializers.IntegerField(),
                    "tier": serializers.CharField(allow_null=True),
                },
            ),
            400: OpenApiTypes.OBJECT,
        },
        examples=[
            OpenApiExample(
                "Preview with offer + redeem_points",
                request_only=True,
                value={
                    "apply_assignment_id": 4,
                    "redeem_points": 10,
                    "items": [
                        {"product": 330, "quantity": 1, "unit_price": "12.99"},
                        {"product": 322, "quantity": 1, "unit_price": "12.99"},
                    ],
                },
            ),
            OpenApiExample(
                "Preview response (sample)",
                response_only=True,
                value={
                    "ok": True,
                    "gross_total": "25.98",
                    "discount_amount": "1.30",
                    "net_total": "24.68",
                    "offer_applied": True,
                    "applied_offer": {
                        "assignment_id": 4,
                        "offer": {"id": 1, "name": "whosadik", "type": "discount", "value": "10.00"},
                        "target": {"scope": "product_id", "value": 330},
                    },
                    "eligible_total": "12.99",
                    "estimated_points_earned": 0,
                    "points_redeemed": 10,
                    "balance_before": 121,
                    "balance_after_estimated": 111,
                    "tier": "Bronze",
                },
            ),
        ],
    )
    def post(self, request):
        req = CheckoutRequestSerializer(data=request.data)
        req.is_valid(raise_exception=True)
        data = req.validated_data

        # load products
        product_ids = [it["product"] for it in data["items"]]
        products = Product.objects.in_bulk(product_ids)

        lines = []
        for it in data["items"]:
            prod = products.get(it["product"])
            if not prod:
                return Response({"ok": False, "message": f"Unknown product_id={it['product']}"}, status=400)
            lines.append(
                Line(
                    product=prod,
                    quantity=int(it["quantity"]),
                    unit_price=_product_unit_price_or_raise(prod),
                )
            )

        account, _ = LoyaltyAccount.objects.get_or_create(user=request.user)
        if account.tier_id is None:
            bronze, _ = Tier.objects.get_or_create(
                name="Bronze",
                defaults={"threshold_spend_90d": 0, "points_rate": DEFAULT_POINTS_RATE},
            )
            account.tier = bronze
            account.save(update_fields=["tier"])

        points_rate = get_effective_points_rate(
            account.tier.points_rate if account.tier else DEFAULT_POINTS_RATE
        )
        tier_points_multiplier = get_tier_points_multiplier(account.tier.name if account.tier else None)

        target = {"scope": "cart"}
        offer_applied = False
        offer_payload = None
        offer_type = "discount"
        offer_value = Decimal("0")
        public_calc = None

        apply_id = data.get("apply_assignment_id")
        apply_public_offer_id = data.get("apply_public_offer_id")
        now_value = timezone.now()
        if apply_id is not None:
            a = OfferAssignment.objects.select_related("offer", "offer__campaign").get(id=apply_id, user=request.user)
            if a.is_redeemed:
                return Response({"ok": False, "message": "Offer already redeemed"}, status=400)
            request_id = getattr(request, "request_id", None) or request.headers.get("X-Request-ID")
            if expire_assignment_if_needed(
                a,
                now=now_value,
                source="checkout_preview_offer_validation",
                request_id=request_id,
            ):
                return Response({"ok": False, "message": "Offer expired"}, status=400)
            if not a.is_active:
                return Response({"ok": False, "message": "Offer is no longer active"}, status=400)

            target = a.target or {"scope": "cart"}
            offer_type = a.offer.offer_type
            offer_value = Decimal(str(a.offer.value))
            offer_applied = True
            offer_payload = _personal_offer_payload(a, target)

        redeem_points = int(data.get("redeem_points") or 0)
        if redeem_points > account.points_balance:
            return Response({"ok": False, "message": "Insufficient points"}, status=400)
        gift_card = None
        if data.get("gift_card_code"):
            try:
                gift_card = _load_redeemable_gift_card_or_raise(data["gift_card_code"], now=now_value)
            except ValidationError as exc:
                detail = exc.detail if isinstance(exc.detail, dict) else {}
                return Response(
                    {"ok": False, "message": detail.get("message", "Gift card invalid")},
                    status=400,
                )

        if apply_id is None and apply_public_offer_id is not None:
            try:
                public_offer, public_campaign, public_target, public_calc = _resolve_public_offer_by_id(
                    lines=lines,
                    points_rate=points_rate,
                    tier_points_multiplier=tier_points_multiplier,
                    redeem_points=redeem_points,
                    gift_card_balance=gift_card.remaining_amount if gift_card else Decimal("0"),
                    now=now_value,
                    public_offer_id=int(apply_public_offer_id),
                    lock=False,
                )
            except ValidationError as exc:
                detail = exc.detail if isinstance(exc.detail, dict) else {}
                return Response(
                    {"ok": False, "message": detail.get("message", "Promotion not applicable")},
                    status=400,
                )
            target = public_target
            offer_type = public_offer.offer_type
            offer_value = Decimal(str(public_offer.value))
            offer_applied = True
            offer_payload = _public_offer_payload(public_offer, public_campaign, target)

        if apply_id is None and apply_public_offer_id is None:
            public_offer, public_campaign, public_target, public_calc = _find_public_campaign_offer(
                lines=lines,
                points_rate=points_rate,
                tier_points_multiplier=tier_points_multiplier,
                redeem_points=redeem_points,
                gift_card_balance=gift_card.remaining_amount if gift_card else Decimal("0"),
                now=now_value,
                lock=False,
            )
            if public_offer is not None and public_campaign is not None and public_target is not None:
                target = public_target
                offer_type = public_offer.offer_type
                offer_value = Decimal(str(public_offer.value))
                offer_applied = True
                offer_payload = _public_offer_payload(public_offer, public_campaign, target)

        calc = public_calc or apply_offer_to_totals(
            offer_type=offer_type,
            offer_value=offer_value,
            target=target,
            lines=lines,
            points_rate=points_rate,
            tier_points_multiplier=tier_points_multiplier,
            redeem_points=redeem_points,
            gift_card_balance=gift_card.remaining_amount if gift_card else Decimal("0"),
        )
        if not calc["ok"]:
            return Response(calc, status=400)

        gross_total = Decimal(calc["gross_total"])
        discount_amount = Decimal(calc["discount_amount"])
        net_total = Decimal(calc["net_total"])
        eligible_total = Decimal(calc["eligible_total"])
        gift_card_applied_amount = Decimal(calc["gift_card_applied_amount"])
        points_redeemed = int(calc["points_redeemed"])
        est_points = int(calc["estimated_points_earned"])

        new_balance_est = account.points_balance - points_redeemed + est_points
        gift_card_payload = None
        if gift_card is not None:
            balance_before = Decimal(gift_card.remaining_amount)
            balance_after = max(balance_before - gift_card_applied_amount, Decimal("0"))
            gift_card_payload = gift_card_snapshot(
                gift_card,
                applied_amount=gift_card_applied_amount,
                balance_before=balance_before,
                balance_after=balance_after,
            )

        available_offers = _collect_available_offers(
            user=request.user,
            lines=lines,
            points_rate=points_rate,
            tier_points_multiplier=tier_points_multiplier,
            redeem_points=redeem_points,
            gift_card_balance=gift_card.remaining_amount if gift_card else Decimal("0"),
            now=now_value,
        )

        return Response({
            "ok": True,
            "gross_total": str(gross_total),
            "discount_amount": str(discount_amount),
            "net_total": str(net_total),
            "offer_applied": offer_applied,
            "applied_offer": offer_payload,
            "available_offers": available_offers,
            "target": target,
            "eligible_total": str(eligible_total),
            "points_rate": str(points_rate),
            "tier_points_multiplier": str(calc["tier_points_multiplier"]),
            "estimated_points_earned": est_points,
            "points_redeemed": points_redeemed,
            "gift_card": gift_card_payload,
            "balance_before": account.points_balance,
            "balance_after_estimated": new_balance_est,
            "tier": account.tier.name if account.tier else None,
        })

