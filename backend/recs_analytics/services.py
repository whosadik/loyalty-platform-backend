from datetime import timedelta
from django.utils import timezone
from recs_analytics.models import RecommendationEvent


def attribute_purchase(user, purchased_product_ids: list[int], *, window_days: int = 7, request_id: str | None = None):
    since = timezone.now() - timedelta(days=window_days)

    # последние impressions по этим товарам
    qs = RecommendationEvent.objects.filter(
        user=user,
        action=RecommendationEvent.Action.IMPRESSION,
        created_at__gte=since,
        product_id__in=purchased_product_ids,
    ).order_by("-created_at")

    seen = set()
    events = []
    for ev in qs:
        if ev.product_id in seen:
            continue
        seen.add(ev.product_id)
        events.append(RecommendationEvent(
            user=user,
            action=RecommendationEvent.Action.PURCHASE_ATTRIBUTED,
            product_id=ev.product_id,
            page=ev.page,
            section_key=ev.section_key,
            request_id=request_id,
            algo_mode=ev.algo_mode,
            score=ev.score,
            components=ev.components,
            context={"attributed_from_event_id": ev.id, "window_days": window_days},
        ))

    if events:
        RecommendationEvent.objects.bulk_create(events, batch_size=500)
