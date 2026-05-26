from rest_framework import serializers

from backend.request_language import AppLanguage, get_context_language
from .models import LoyaltyLedgerEntry


class RedeemPointsRequestSerializer(serializers.Serializer):
    points = serializers.IntegerField(min_value=1)
    reference = serializers.CharField(required=False, allow_blank=True, default="")


class MeLoyaltyResponseSerializer(serializers.Serializer):
    tier = serializers.CharField(allow_null=True)
    points_balance = serializers.IntegerField()


class RedeemPointsResponseSerializer(serializers.Serializer):
    ok = serializers.BooleanField()
    new_balance = serializers.IntegerField()


LEDGER_DESCRIPTION_COPY: dict[AppLanguage, dict[str, str]] = {
    "ru": {
        "profile_completion": "Бонус за заполнение профиля",
        "roadmap_step": "Бонус за шаг роадмапа",
        "txn_earn": "Начисление за покупку",
        "txn_redeem": "Списание при оплате заказа",
        "manual_redeem": "Списание баллов",
        "offer": "Бонус по офферу",
        "gift_card": "Операция с подарочной картой",
        "adjust": "Корректировка баланса",
        "earn": "Начисление баллов",
        "redeem": "Списание баллов",
    },
    "kk": {
        "profile_completion": "Профильді толтыру бонусы",
        "roadmap_step": "Roadmap қадамы бонусы",
        "txn_earn": "Сатып алу үшін есептеу",
        "txn_redeem": "Тапсырысты төлеу кезінде шегеру",
        "manual_redeem": "Ұпайларды шегеру",
        "offer": "Оффер бойынша бонус",
        "gift_card": "Сыйлық картасы операциясы",
        "adjust": "Балансты түзету",
        "earn": "Ұпайларды есептеу",
        "redeem": "Ұпайларды шегеру",
    },
    "en": {
        "profile_completion": "Profile completion bonus",
        "roadmap_step": "Roadmap step bonus",
        "txn_earn": "Points earned for purchase",
        "txn_redeem": "Points redeemed on order",
        "manual_redeem": "Manual points redemption",
        "offer": "Offer bonus",
        "gift_card": "Gift card operation",
        "adjust": "Balance adjustment",
        "earn": "Points earned",
        "redeem": "Points redeemed",
    },
}


def _ledger_kind_from_reference(reference: str, entry_type: str) -> str:
    ref = (reference or "").strip().lower()
    if ref.startswith("profile_completion"):
        return "profile_completion"
    if ref.startswith("roadmap_step") or ref.startswith("roadmap:step"):
        return "roadmap_step"
    if ref.startswith("txn:") and entry_type == LoyaltyLedgerEntry.Type.EARN:
        return "txn_earn"
    if ref.startswith("txn:") and entry_type == LoyaltyLedgerEntry.Type.REDEEM:
        return "txn_redeem"
    if ref == "manual_redeem":
        return "manual_redeem"
    if ref.startswith("offer"):
        return "offer"
    if ref.startswith("gift_card") or ref.startswith("giftcard"):
        return "gift_card"
    if entry_type == LoyaltyLedgerEntry.Type.ADJUST:
        return "adjust"
    if entry_type == LoyaltyLedgerEntry.Type.EARN:
        return "earn"
    if entry_type == LoyaltyLedgerEntry.Type.REDEEM:
        return "redeem"
    return "adjust"


class LoyaltyLedgerEntrySerializer(serializers.ModelSerializer):
    description = serializers.SerializerMethodField()
    kind = serializers.SerializerMethodField()
    transaction_id = serializers.SerializerMethodField()

    class Meta:
        model = LoyaltyLedgerEntry
        fields = [
            "id",
            "entry_type",
            "points_delta",
            "reference",
            "kind",
            "description",
            "transaction_id",
            "meta",
            "created_at",
        ]

    def _kind(self, obj: LoyaltyLedgerEntry) -> str:
        return _ledger_kind_from_reference(obj.reference or "", obj.entry_type)

    def get_kind(self, obj: LoyaltyLedgerEntry) -> str:
        return self._kind(obj)

    def get_description(self, obj: LoyaltyLedgerEntry) -> str:
        language = get_context_language(self.context)
        copy = LEDGER_DESCRIPTION_COPY.get(language, LEDGER_DESCRIPTION_COPY["ru"])
        return copy.get(self._kind(obj), copy["adjust"])

    def get_transaction_id(self, obj: LoyaltyLedgerEntry):
        ref = (obj.reference or "").strip().lower()
        if ref.startswith("txn:"):
            tail = ref.split(":", 1)[1]
            try:
                return int(tail)
            except (TypeError, ValueError):
                return None
        return None
