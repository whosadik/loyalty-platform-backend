from __future__ import annotations

from decimal import Decimal

from rest_framework import serializers

from backend.request_language import AppLanguage, get_context_language
from catalog.models import Product
from catalog.sale_fields import (
    get_product_discount_percent,
    get_product_effective_price,
    get_product_original_price,
    product_has_discount,
)
from catalog.serializers import ProductSerializer
from catalog.product_metrics import get_product_points_earned
from roadmap_app.serializers import serialize_roadmap_step_snapshot
from .models import CartItem, OwnedProduct, Transaction, TransactionItem, WishlistItem


def _coerce_decimal(value, default: str = "0.00") -> Decimal:
    try:
        return Decimal(str(value))
    except Exception:
        return Decimal(default)


def _coerce_int(value, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


TRANSACTION_DESCRIPTION_COPY: dict[AppLanguage, dict[str, str]] = {
    "ru": {
        "gift_card_purchase": "Покупка подарочной карты",
        "purchase_single": "Покупка",
        "purchase_many": "Покупка, {count} товара",
    },
    "kk": {
        "gift_card_purchase": "Сыйлық картасын сатып алу",
        "purchase_single": "Сатып алу",
        "purchase_many": "Сатып алу, {count} тауар",
    },
    "en": {
        "gift_card_purchase": "Gift card purchase",
        "purchase_single": "Purchase",
        "purchase_many": "Purchase, {count} items",
    },
}


class TransactionSnapshotMixin(serializers.Serializer):
    transaction_id = serializers.SerializerMethodField()
    type = serializers.SerializerMethodField()
    description = serializers.SerializerMethodField()
    status = serializers.SerializerMethodField()
    gross_total = serializers.SerializerMethodField()
    discount_amount = serializers.SerializerMethodField()
    net_total = serializers.SerializerMethodField()
    offer_applied = serializers.SerializerMethodField()
    offer_assignment_id = serializers.SerializerMethodField()
    public_campaign_id = serializers.SerializerMethodField()
    public_offer_id = serializers.SerializerMethodField()
    applied_offer = serializers.SerializerMethodField()
    target = serializers.SerializerMethodField()
    eligible_total = serializers.SerializerMethodField()
    points_earned = serializers.SerializerMethodField()
    points_redeemed = serializers.SerializerMethodField()
    points_change = serializers.SerializerMethodField()
    new_balance = serializers.SerializerMethodField()
    tier_after = serializers.SerializerMethodField()
    new_tier = serializers.SerializerMethodField()
    tier_upgraded = serializers.SerializerMethodField()
    next_offer = serializers.SerializerMethodField()
    next_roadmap_step = serializers.SerializerMethodField()
    gift_card = serializers.SerializerMethodField()

    def _meta(self, obj: Transaction) -> dict:
        return obj.pricing_meta if isinstance(obj.pricing_meta, dict) else {}

    def _language(self) -> AppLanguage:
        return get_context_language(self.context)

    def _gross_total(self, obj: Transaction) -> Decimal:
        meta = self._meta(obj)
        return _coerce_decimal(meta.get("gross_total", obj.total_amount or "0.00"))

    def _net_total(self, obj: Transaction) -> Decimal:
        meta = self._meta(obj)
        return _coerce_decimal(meta.get("net_total", obj.total_amount or "0.00"))

    def get_transaction_id(self, obj: Transaction) -> str:
        return f"TXN-{int(obj.id):08d}"

    def get_type(self, obj: Transaction) -> str:
        return str(self._meta(obj).get("type") or "purchase")

    def get_description(self, obj: Transaction) -> str:
        copy = TRANSACTION_DESCRIPTION_COPY[self._language()]
        if self.get_type(obj) == "gift_card_purchase":
            return copy["gift_card_purchase"]

        item_count = sum(int(item.quantity or 0) for item in obj.items.all())
        if item_count <= 1:
            return copy["purchase_single"]
        return copy["purchase_many"].format(count=item_count)

    def get_status(self, obj: Transaction) -> str:
        return str(self._meta(obj).get("status") or "completed")

    def get_gross_total(self, obj: Transaction) -> str:
        return str(self._gross_total(obj))

    def get_discount_amount(self, obj: Transaction) -> str:
        meta = self._meta(obj)
        return str(_coerce_decimal(meta.get("discount_amount", "0.00")))

    def get_net_total(self, obj: Transaction) -> str:
        return str(self._net_total(obj))

    def get_offer_applied(self, obj: Transaction) -> bool:
        return bool(self._meta(obj).get("offer_applied", False))

    def get_offer_assignment_id(self, obj: Transaction):
        return self._meta(obj).get("offer_assignment_id")

    def get_public_campaign_id(self, obj: Transaction):
        return self._meta(obj).get("public_campaign_id")

    def get_public_offer_id(self, obj: Transaction):
        return self._meta(obj).get("public_offer_id")

    def get_applied_offer(self, obj: Transaction):
        return self._meta(obj).get("applied_offer")

    def get_target(self, obj: Transaction):
        return self._meta(obj).get("target")

    def get_eligible_total(self, obj: Transaction) -> str:
        meta = self._meta(obj)
        return str(_coerce_decimal(meta.get("eligible_total", self._gross_total(obj))))

    def get_points_earned(self, obj: Transaction) -> int:
        return _coerce_int(self._meta(obj).get("points_earned"), default=0)

    def get_points_redeemed(self, obj: Transaction) -> int:
        return _coerce_int(self._meta(obj).get("points_redeemed"), default=0)

    def get_points_change(self, obj: Transaction) -> int:
        return self.get_points_earned(obj) - self.get_points_redeemed(obj)

    def get_new_balance(self, obj: Transaction):
        value = self._meta(obj).get("new_balance")
        if value is None:
            return None
        return _coerce_int(value, default=0)

    def get_tier_after(self, obj: Transaction):
        meta = self._meta(obj)
        return meta.get("new_tier") or meta.get("tier")

    def get_new_tier(self, obj: Transaction):
        return self.get_tier_after(obj)

    def get_tier_upgraded(self, obj: Transaction) -> bool:
        return bool(self._meta(obj).get("tier_upgraded", False))

    def get_next_offer(self, obj: Transaction):
        return self._meta(obj).get("next_offer")

    def get_next_roadmap_step(self, obj: Transaction):
        value = self._meta(obj).get("next_roadmap_step")
        if isinstance(value, dict):
            return serialize_roadmap_step_snapshot(value, language=self._language())
        return value

    def get_gift_card(self, obj: Transaction):
        return self._meta(obj).get("gift_card")


class TransactionItemSerializer(serializers.ModelSerializer):
    product_summary = serializers.SerializerMethodField()

    def get_product_summary(self, obj: TransactionItem):
        return ProductSummarySerializer(obj.product, context=self.context).data

    class Meta:
        model = TransactionItem
        fields = ["product", "quantity", "unit_price", "product_summary"]


class TransactionSerializer(TransactionSnapshotMixin, serializers.ModelSerializer):
    items = TransactionItemSerializer(many=True)

    class Meta:
        model = Transaction
        fields = [
            "id",
            "transaction_id",
            "created_at",
            "type",
            "description",
            "status",
            "channel",
            "total_amount",
            "gross_total",
            "discount_amount",
            "net_total",
            "offer_applied",
            "offer_assignment_id",
            "public_campaign_id",
            "public_offer_id",
            "applied_offer",
            "target",
            "eligible_total",
            "points_earned",
            "points_redeemed",
            "points_change",
            "new_balance",
            "tier_after",
            "new_tier",
            "tier_upgraded",
            "next_offer",
            "next_roadmap_step",
            "gift_card",
            "items",
        ]
        read_only_fields = [
            "id",
            "transaction_id",
            "created_at",
            "type",
            "description",
            "status",
            "total_amount",
            "gross_total",
            "discount_amount",
            "net_total",
            "offer_applied",
            "offer_assignment_id",
            "public_campaign_id",
            "public_offer_id",
            "applied_offer",
            "target",
            "eligible_total",
            "points_earned",
            "points_redeemed",
            "points_change",
            "new_balance",
            "tier_after",
            "new_tier",
            "tier_upgraded",
            "next_offer",
            "next_roadmap_step",
            "gift_card",
        ]

    def create(self, validated_data):
        items_data = validated_data.pop("items", [])
        user = self.context["request"].user

        txn = Transaction.objects.create(user=user, **validated_data)

        total = Decimal("0.00")
        for item in items_data:
            TransactionItem.objects.create(transaction=txn, **item)
            owned, _ = OwnedProduct.objects.get_or_create(user=user, product=item["product"])
            owned.quantity_total = (owned.quantity_total or 0) + int(item["quantity"])
            owned.is_active = True
            owned.last_acquired_at = txn.created_at
            owned.save(update_fields=["quantity_total", "is_active", "last_acquired_at"])

            total += Decimal(str(item["unit_price"])) * int(item["quantity"])

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
    price = serializers.SerializerMethodField()
    original_price = serializers.SerializerMethodField()
    discount = serializers.SerializerMethodField()
    has_discount = serializers.SerializerMethodField()
    points_earned = serializers.SerializerMethodField()
    image_url = serializers.SerializerMethodField()

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
            "original_price",
            "discount",
            "has_discount",
            "points_earned",
        ]

    def get_price(self, obj: Product) -> str | None:
        price = get_product_effective_price(obj)
        return str(price) if price is not None else None

    def get_original_price(self, obj: Product) -> str | None:
        original_price = get_product_original_price(obj)
        return str(original_price) if original_price is not None else None

    def get_discount(self, obj: Product) -> int | None:
        return get_product_discount_percent(obj)

    def get_has_discount(self, obj: Product) -> bool:
        return product_has_discount(obj)

    def get_image_url(self, obj: Product) -> str:
        if obj.image:
            request = self.context.get("request")
            if request is not None:
                return request.build_absolute_uri(obj.image.url)
            return obj.image.url
        return obj.image_url or ""

    def get_points_earned(self, obj: Product) -> int:
        request = self.context.get("request")
        user = getattr(request, "user", None) if request is not None else None
        return get_product_points_earned(obj, user=user)


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
