from decimal import Decimal

from rest_framework import serializers

from catalog.models import Product
from catalog.serializers import ProductSerializer
from .models import CartItem, OwnedProduct, Transaction, TransactionItem, WishlistItem


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


class OwnedProductSerializer(serializers.ModelSerializer):
    product = ProductSerializer(read_only=True)

    class Meta:
        model = OwnedProduct
        fields = [
            "id",
            "product",
            "quantity_total",
            "is_active",
            "notes",
            "opened_at",
            "finish_date",
            "acquired_at",
            "last_acquired_at",
            "source",
        ]
        read_only_fields = [
            "id",
            "product",
            "quantity_total",
            "acquired_at",
            "last_acquired_at",
            "source",
        ]


class ProductSummarySerializer(serializers.ModelSerializer):
    points_earned = serializers.SerializerMethodField()

    class Meta:
        model = Product
        fields = [
            "id",
            "name",
            "brand",
            "price",
            "currency",
            "category",
            "product_type",
            "in_stock",
            "image_url",
            "image_urls",
            "points_earned",
        ]

    def get_points_earned(self, obj: Product) -> int:
        try:
            price = Decimal(str(obj.price or "0"))
        except Exception:
            price = Decimal("0")
        return int(max(0, round(float(price) * 0.1)))


class WishlistItemSerializer(serializers.ModelSerializer):
    product = ProductSummarySerializer(read_only=True)

    class Meta:
        model = WishlistItem
        fields = ["product", "created_at"]
        read_only_fields = ["product", "created_at"]


class WishlistAddSerializer(serializers.Serializer):
    product_id = serializers.IntegerField(min_value=1)


class CartItemSerializer(serializers.ModelSerializer):
    product = ProductSummarySerializer(read_only=True)

    class Meta:
        model = CartItem
        fields = ["product", "quantity", "created_at", "updated_at"]
        read_only_fields = ["product", "created_at", "updated_at"]


class CartAddSerializer(serializers.Serializer):
    product_id = serializers.IntegerField(min_value=1)
    quantity = serializers.IntegerField(min_value=1, max_value=100, required=False, default=1)


class CartPatchSerializer(serializers.Serializer):
    quantity = serializers.IntegerField(min_value=0, max_value=100)
