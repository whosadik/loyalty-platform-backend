from __future__ import annotations

from typing import Any

from django.db import IntegrityError

from offers.models import OfferAssignment, OfferEvent


def record_offer_event(
    assignment: OfferAssignment,
    event_type: str,
    *,
    request_id: str | None = None,
    context: dict[str, Any] | None = None,
) -> OfferEvent | None:
    """
    Idempotent write: max 1 event per (assignment, event_type).
    """
    if not assignment:
        return None

    ctx = context or {}
    campaign_name = getattr(getattr(assignment.offer, "campaign", None), "name", None) or ""

    try:
        obj, _ = OfferEvent.objects.get_or_create(
            assignment=assignment,
            event_type=event_type,
            defaults={
                "user": assignment.user,
                "offer": assignment.offer,
                "campaign_name": campaign_name,
                "request_id": request_id,
                "context": ctx,
            },
        )
        return obj
    except IntegrityError:
        return OfferEvent.objects.filter(assignment=assignment, event_type=event_type).first()
