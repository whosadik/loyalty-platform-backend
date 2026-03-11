from __future__ import annotations

from datetime import datetime, timedelta

from django.utils import timezone


NEW_PRODUCTS_WINDOW_DAYS = 60


def get_new_products_cutoff() -> datetime:
    return timezone.now() - timedelta(days=NEW_PRODUCTS_WINDOW_DAYS)


def created_at_is_new(created_at: datetime | None) -> bool:
    if created_at is None:
        return False
    return created_at >= get_new_products_cutoff()

