from decimal import Decimal

from django.conf import settings
from django.db import models


class Tier(models.Model):
    """
    Простая система уровней.
    threshold_spend: сколько нужно потратить за 90 дней, чтобы быть в этом уровне.
    """
    name = models.CharField(max_length=50, unique=True)  # Bronze/Silver/Gold
    threshold_spend_90d = models.DecimalField(max_digits=12, decimal_places=2, default=0)
    points_rate = models.DecimalField(max_digits=10, decimal_places=2, default=Decimal("0.01"))  # 1% earn rate

    def __str__(self) -> str:
        return self.name


class LoyaltyAccount(models.Model):
    user = models.OneToOneField(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name="loyalty_account")
    tier = models.ForeignKey(Tier, on_delete=models.PROTECT, null=True, blank=True)
    points_balance = models.IntegerField(default=0)

    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self) -> str:
        return f"LoyaltyAccount(user_id={self.user_id}, points={self.points_balance})"


class LoyaltyLedgerEntry(models.Model):
    """
    Ledger = неизменяемый журнал.
    Вместо "обновить баланс" мы добавляем записи, а баланс агрегируем/кэшируем.
    """
    class Type(models.TextChoices):
        EARN = "earn", "Earn"
        REDEEM = "redeem", "Redeem"
        ADJUST = "adjust", "Adjust"

    account = models.ForeignKey(LoyaltyAccount, on_delete=models.CASCADE, related_name="ledger")
    entry_type = models.CharField(max_length=20, choices=Type.choices)

    points_delta = models.IntegerField()  # + начисление, - списание
    reference = models.CharField(max_length=100, blank=True, default="")  # например "txn:123", "offer:5"

    meta = models.JSONField(default=dict, blank=True)

    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self) -> str:
        return f"Ledger({self.entry_type}, delta={self.points_delta}, ref={self.reference})"
