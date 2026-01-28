from django.db.models.signals import post_save
from django.dispatch import receiver
from django.contrib.auth import get_user_model
from admin_tools.models import StaffProfile

User = get_user_model()


@receiver(post_save, sender=User)
def ensure_staff_profile(sender, instance, created, **kwargs):
    if instance.is_staff:
        StaffProfile.objects.get_or_create(user=instance)
