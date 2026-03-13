from django.urls import path

from .views import GiftCardPurchaseView, MyReceivedGiftCardsView, MySentGiftCardsView


urlpatterns = [
    path("gift-cards/purchase", GiftCardPurchaseView.as_view(), name="gift-card-purchase"),
    path("me/gift-cards/sent", MySentGiftCardsView.as_view(), name="me-gift-cards-sent"),
    path("me/gift-cards/received", MyReceivedGiftCardsView.as_view(), name="me-gift-cards-received"),
]
