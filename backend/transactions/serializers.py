from rest_framework import serializers
from .models import Transaction, TransactionItem


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

        total = 0
        for it in items_data:
            TransactionItem.objects.create(transaction=txn, **it)
            total += float(it["unit_price"]) * int(it["quantity"])

        txn.total_amount = total
        txn.save(update_fields=["total_amount"])
        return txn
