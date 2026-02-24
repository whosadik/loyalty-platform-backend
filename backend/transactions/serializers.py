from decimal import Decimal

from rest_framework import serializers
from .models import Transaction, TransactionItem, OwnedProduct


class TransactionItemSerializer(serializers.ModelSerializer):
    class Meta:
        model = TransactionItem
        fields = ["product", "quantity", "unit_price"]


class TransactionSerializer(serializers.ModelSerializer):
    items = TransactionItemSerializer(many=True)

    class Meta:
        model = Transaction
        fields = ["id", "created_at", "total_amount", "channel", "items"]
        read_only_fields = ["id", "created_at", "total_amount"]

    def create(self, validated_data):
        items_data = validated_data.pop("items", [])
        user = self.context["request"].user

        txn = Transaction.objects.create(user=user, **validated_data)

        total = Decimal("0.00")
        for it in items_data:
            TransactionItem.objects.create(transaction=txn, **it)
            owned, created = OwnedProduct.objects.get_or_create(user=user, product=it["product"])
            owned.quantity_total = (owned.quantity_total or 0) + int(it["quantity"])
            owned.is_active = True
            owned.last_acquired_at = txn.created_at
            owned.save(update_fields=["quantity_total", "is_active", "last_acquired_at"])

            total += Decimal(str(it["unit_price"])) * int(it["quantity"])

        txn.total_amount = total
        txn.save(update_fields=["total_amount"])
        return txn

from catalog.serializers import ProductSerializer
from .models import OwnedProduct


class OwnedProductSerializer(serializers.ModelSerializer):
    product = ProductSerializer(read_only=True)

    class Meta:
        model = OwnedProduct
        fields = ["id", "product", "quantity_total", "is_active", "acquired_at", "last_acquired_at", "source"]
