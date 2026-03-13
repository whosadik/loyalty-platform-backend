from rest_framework import serializers

from .models import GiftCard
from .services import ALLOWED_GIFT_CARD_AMOUNTS, format_gift_card_code, gift_card_snapshot


class GiftCardPurchaseRequestSerializer(serializers.Serializer):
    amount = serializers.IntegerField()
    recipient_email = serializers.EmailField()
    message = serializers.CharField(required=False, allow_blank=True, max_length=500)
    idempotency_key = serializers.CharField(required=False, allow_blank=False, max_length=64)
    channel = serializers.CharField(required=False, default="online")

    def validate_amount(self, value: int) -> int:
        if value not in ALLOWED_GIFT_CARD_AMOUNTS:
            raise serializers.ValidationError(
                f"Allowed gift card amounts: {', '.join(str(item) for item in ALLOWED_GIFT_CARD_AMOUNTS)}"
            )
        return int(value)


class GiftCardSnapshotSerializer(serializers.Serializer):
    id = serializers.IntegerField()
    masked_code = serializers.CharField()
    recipient_email = serializers.EmailField()
    amount = serializers.CharField()
    remaining_amount = serializers.CharField()
    applied_amount = serializers.CharField(required=False)
    balance_before = serializers.CharField(required=False)
    balance_after = serializers.CharField(required=False)
    currency = serializers.CharField()
    status = serializers.CharField()
    expires_at = serializers.DateTimeField(allow_null=True)
    sent_at = serializers.DateTimeField(allow_null=True)


class GiftCardPurchaseResponseSerializer(serializers.Serializer):
    ok = serializers.BooleanField()
    idempotent_replay = serializers.BooleanField(required=False)
    transaction_id = serializers.IntegerField()
    gross_total = serializers.CharField()
    discount_amount = serializers.CharField()
    net_total = serializers.CharField()
    gift_card = GiftCardSnapshotSerializer()
    email_sent = serializers.BooleanField()
    new_balance = serializers.IntegerField(required=False, allow_null=True)


class GiftCardSentItemSerializer(serializers.ModelSerializer):
    snapshot = serializers.SerializerMethodField()

    class Meta:
        model = GiftCard
        fields = [
            "id",
            "recipient_email",
            "message",
            "status",
            "created_at",
            "snapshot",
        ]

    def get_snapshot(self, obj: GiftCard):
        return gift_card_snapshot(obj)


class GiftCardReceivedItemSerializer(serializers.ModelSerializer):
    snapshot = serializers.SerializerMethodField()
    sender_name = serializers.SerializerMethodField()
    sender_email = serializers.EmailField(source="purchaser.email", read_only=True)
    code = serializers.SerializerMethodField()

    class Meta:
        model = GiftCard
        fields = [
            "id",
            "message",
            "status",
            "created_at",
            "sender_name",
            "sender_email",
            "code",
            "snapshot",
        ]

    def get_snapshot(self, obj: GiftCard):
        return gift_card_snapshot(obj)

    def get_code(self, obj: GiftCard) -> str:
        return format_gift_card_code(obj.code)

    def get_sender_name(self, obj: GiftCard) -> str:
        full_name = obj.purchaser.get_full_name().strip()
        if full_name:
            return full_name
        username = getattr(obj.purchaser, "username", "") or ""
        if username:
            return username
        return obj.purchaser.email or "Uilesim"


class GiftCardSentListResponseSerializer(serializers.Serializer):
    ok = serializers.BooleanField()
    count = serializers.IntegerField()
    items = GiftCardSentItemSerializer(many=True)


class GiftCardReceivedListResponseSerializer(serializers.Serializer):
    ok = serializers.BooleanField()
    count = serializers.IntegerField()
    items = GiftCardReceivedItemSerializer(many=True)
