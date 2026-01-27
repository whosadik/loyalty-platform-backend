from django.conf import settings
from django.db import models


class Transaction(models.Model):
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name="transactions")

    created_at = models.DateTimeField(auto_now_add=True)
    total_amount = models.DecimalField(max_digits=10, decimal_places=2, default=0)

    channel = models.CharField(max_length=20, default="offline")  # offline/online (MVP)
    idempotency_key = models.CharField(max_length=64, null=True, blank=True)
    pricing_meta = models.JSONField(default=dict, blank=True)
    
    class Meta:
        constraints = [
            models.UniqueConstraint(
                fields=["user", "idempotency_key"],
                name="uniq_txn_user_idempotency_key",
            )
        ]
    def __str__(self) -> str:
        return f"Txn(id={self.id}, user_id={self.user_id}, total={self.total_amount})"


class TransactionItem(models.Model):
    transaction = models.ForeignKey(Transaction, on_delete=models.CASCADE, related_name="items")
    product = models.ForeignKey("catalog.Product", on_delete=models.PROTECT)

    quantity = models.PositiveIntegerField(default=1)
    unit_price = models.DecimalField(max_digits=10, decimal_places=2)

    def __str__(self) -> str:
        return f"Item(txn_id={self.transaction_id}, product_id={self.product_id}, qty={self.quantity})"

from django.utils import timezone


class OwnedProduct(models.Model):
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name="owned_products")
    product = models.ForeignKey("catalog.Product", on_delete=models.PROTECT)
    quantity_total = models.PositiveIntegerField(default=0)
    is_active = models.BooleanField(default=True)  # если закончился/не использует
    last_acquired_at = models.DateTimeField(default=timezone.now)

    acquired_at = models.DateTimeField(default=timezone.now)
    source = models.CharField(max_length=30, default="transaction")  # transaction/manual/import

    class Meta:
        unique_together = ("user", "product")  # один и тот же товар считаем "есть", без дублей

    def __str__(self) -> str:
        return f"Owned(user={self.user_id}, product={self.product_id})"
