from __future__ import annotations

import hashlib
from typing import Any

from offers.models import OfferAssignment, OfferEvent


_ONE_SHOT_EVENT_TYPES = {
    OfferEvent.Type.ASSIGNED,
    OfferEvent.Type.REDEEMED,
    OfferEvent.Type.EXPIRED,
}


def _normalize_event_key(raw: str | None, *, assignment_id: int, event_type: str) -> str | None:
    key = (raw or "").strip()
    if not key:
        return None
    if len(key) <= 128:
        return key
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()
    prefix = f"a:{assignment_id}:t:{event_type}:sha1:"
    short = (prefix + digest)[:128]
    return short


def record_offer_event(
    assignment: OfferAssignment,
    event_type: str,
    *,
    request_id: str | None = None,
    context: dict[str, Any] | None = None,
    idempotency_key: str | None = None,
    return_created: bool = False,
) -> OfferEvent | tuple[OfferEvent | None, bool] | None:
    """
    Writes an offer event.
    Idempotency:
      - by explicit idempotency_key (preferred)
      - else by request_id for request-bound events
      - else one-shot key for assigned/redeemed/expired
    """
    if not assignment:
        return (None, False) if return_created else None

    ctx = context or {}
    campaign_name = getattr(getattr(assignment.offer, "campaign", None), "name", None) or ""

    key = (idempotency_key or "").strip() or None
    if not key and request_id:
        key = f"a:{assignment.id}:t:{event_type}:r:{request_id}"
    if not key and event_type in _ONE_SHOT_EVENT_TYPES:
        key = f"a:{assignment.id}:t:{event_type}"
    key = _normalize_event_key(key, assignment_id=assignment.id, event_type=event_type)

    if key:
        obj, created = OfferEvent.objects.get_or_create(
            event_key=key,
            defaults={
                "assignment": assignment,
                "user": assignment.user,
                "offer": assignment.offer,
                "campaign_name": campaign_name,
                "event_type": event_type,
                "request_id": request_id,
                "context": ctx,
            },
        )
        return (obj, created) if return_created else obj

    obj = OfferEvent.objects.create(
        assignment=assignment,
        user=assignment.user,
        offer=assignment.offer,
        campaign_name=campaign_name,
        event_type=event_type,
        event_key=None,
        request_id=request_id,
        context=ctx,
    )
    return (obj, True) if return_created else obj
