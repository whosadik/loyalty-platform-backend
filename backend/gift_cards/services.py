from __future__ import annotations

from decimal import Decimal, ROUND_HALF_UP
from secrets import choice
from string import ascii_uppercase, digits
from typing import Any

from django.conf import settings
from django.core.mail import send_mail
from django.utils import timezone

from .models import GiftCard


ALLOWED_GIFT_CARD_AMOUNTS = (1000, 3000, 5000, 10000)
GIFT_CARD_CODE_ALPHABET = "".join(ch for ch in f"{ascii_uppercase}{digits}" if ch not in {"0", "1", "O", "I"})


def d2(value: Decimal) -> Decimal:
    return Decimal(value).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)


def normalize_gift_card_code(raw_code: str | None) -> str:
    if not raw_code:
        return ""
    return "".join(ch for ch in str(raw_code).upper() if ch.isalnum())


def format_gift_card_code(raw_code: str | None) -> str:
    normalized = normalize_gift_card_code(raw_code)
    if not normalized:
        return ""
    return "-".join(normalized[idx: idx + 4] for idx in range(0, len(normalized), 4))


def mask_gift_card_code(raw_code: str | None) -> str:
    normalized = normalize_gift_card_code(raw_code)
    if len(normalized) <= 8:
        return normalized
    return format_gift_card_code(f"{normalized[:4]}{'*' * max(len(normalized) - 8, 0)}{normalized[-4:]}")


def generate_gift_card_code(length: int = 16) -> str:
    while True:
        code = "".join(choice(GIFT_CARD_CODE_ALPHABET) for _ in range(length))
        if not GiftCard.objects.filter(code=code).exists():
            return code


def gift_card_snapshot(
    gift_card: GiftCard,
    *,
    applied_amount: Decimal | int | str = Decimal("0"),
    balance_before: Decimal | int | str | None = None,
    balance_after: Decimal | int | str | None = None,
) -> dict[str, Any]:
    before = d2(Decimal(balance_before if balance_before is not None else gift_card.remaining_amount))
    after = d2(Decimal(balance_after if balance_after is not None else gift_card.remaining_amount))
    applied = d2(Decimal(applied_amount))
    return {
        "id": int(gift_card.id),
        "masked_code": mask_gift_card_code(gift_card.code),
        "recipient_email": gift_card.recipient_email,
        "amount": str(d2(gift_card.initial_amount)),
        "remaining_amount": str(after),
        "applied_amount": str(applied),
        "balance_before": str(before),
        "balance_after": str(after),
        "currency": gift_card.currency,
        "status": gift_card.status,
        "expires_at": gift_card.expires_at.isoformat() if gift_card.expires_at else None,
        "sent_at": gift_card.sent_at.isoformat() if gift_card.sent_at else None,
    }


def refresh_gift_card_status(gift_card: GiftCard, now=None) -> GiftCard:
    now = now or timezone.now()
    update_fields: list[str] = []

    if gift_card.expires_at and gift_card.expires_at <= now and gift_card.status != GiftCard.Status.EXPIRED:
        gift_card.status = GiftCard.Status.EXPIRED
        update_fields.append("status")
    elif gift_card.remaining_amount <= 0 and gift_card.status != GiftCard.Status.EXHAUSTED:
        gift_card.status = GiftCard.Status.EXHAUSTED
        update_fields.append("status")
    elif (
        gift_card.remaining_amount > 0
        and (not gift_card.expires_at or gift_card.expires_at > now)
        and gift_card.status != GiftCard.Status.ACTIVE
    ):
        gift_card.status = GiftCard.Status.ACTIVE
        update_fields.append("status")

    if update_fields:
        gift_card.save(update_fields=update_fields + ["updated_at"])
    return gift_card


def resolve_redeemable_gift_card(code: str, *, now=None) -> GiftCard:
    normalized = normalize_gift_card_code(code)
    if not normalized:
        raise ValueError("Gift card code is required")

    try:
        gift_card = GiftCard.objects.get(code=normalized)
    except GiftCard.DoesNotExist as exc:
        raise ValueError("Gift card not found") from exc

    refresh_gift_card_status(gift_card, now=now)
    if gift_card.status == GiftCard.Status.EXPIRED:
        raise ValueError("Gift card expired")
    if gift_card.status == GiftCard.Status.REFUNDED:
        raise ValueError("Gift card is no longer active")
    if gift_card.remaining_amount <= 0:
        raise ValueError("Gift card balance is empty")
    return gift_card


def send_gift_card_message(gift_card: GiftCard) -> None:
    formatted_code = format_gift_card_code(gift_card.code)
    send_mail(
        subject="Your Uilesim gift card",
        message=(
            f"You received a Uilesim gift card for {int(d2(gift_card.initial_amount))} KZT.\n\n"
            f"Code: {formatted_code}\n"
            f"Balance: {int(d2(gift_card.remaining_amount))} KZT\n"
            f"Use this code during checkout on {settings.FRONTEND_BASE_URL.rstrip('/')}/cart\n\n"
            f"Message: {gift_card.message or 'No message attached.'}"
        ),
        from_email=settings.DEFAULT_FROM_EMAIL,
        recipient_list=[gift_card.recipient_email],
    )

