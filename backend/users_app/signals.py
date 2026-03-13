from django.conf import settings
from django.db.models.signals import post_save
from django.dispatch import receiver

from .models import CustomerProfile
from loyalty.models import LoyaltyAccount, Tier
from loyalty.points import DEFAULT_POINTS_RATE


def _ensure_default_tiers():
    # Создадим дефолтные уровни один раз (можно позже вынести в отдельную команду)
    Tier.objects.get_or_create(name="Bronze", defaults={"threshold_spend_90d": 0, "points_rate": DEFAULT_POINTS_RATE})
    Tier.objects.get_or_create(name="Silver", defaults={"threshold_spend_90d": 100, "points_rate": DEFAULT_POINTS_RATE})
    Tier.objects.get_or_create(name="Gold", defaults={"threshold_spend_90d": 250, "points_rate": DEFAULT_POINTS_RATE})


@receiver(post_save, sender=settings.AUTH_USER_MODEL)
def create_profile_for_new_user(sender, instance, created, **kwargs):
    if created:
        CustomerProfile.objects.create(user=instance)

        _ensure_default_tiers()
        bronze = Tier.objects.get(name="Bronze")
        LoyaltyAccount.objects.create(user=instance, tier=bronze)
