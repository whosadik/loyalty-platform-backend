from django.urls import path
from .views import MeNextOfferView, RedeemOfferView

urlpatterns = [
    path("me/next-offer", MeNextOfferView.as_view(), name="me-next-offer"),
    path("offers/redeem", RedeemOfferView.as_view(), name="offers-redeem"),
]
