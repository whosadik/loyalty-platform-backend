from django.conf import settings
from django.db import models


class Transaction(models.Model):
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name="transactions")

    created_at = models.DateTimeField(auto_now_add=True)
    total_amount = models.DecimalField(max_digits=10, decimal_places=2, default=0)

    channel = models.CharField(max_length=20, default="offline")  # offline/online (MVP)

    def __str__(self) -> str:
        return f"Txn(id={self.id}, user_id={self.user_id}, total={self.total_amount})"


class TransactionItem(models.Model):
    transaction = models.ForeignKey(Transaction, on_delete=models.CASCADE, related_name="items")
    product = models.ForeignKey("catalog.Product", on_delete=models.PROTECT)

    quantity = models.PositiveIntegerField(default=1)
    unit_price = models.DecimalField(max_digits=10, decimal_places=2)

    def __str__(self) -> str:
        return f"Item(txn_id={self.transaction_id}, product_id={self.product_id}, qty={self.quantity})"
