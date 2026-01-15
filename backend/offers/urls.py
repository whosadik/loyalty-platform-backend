from django.urls import path
from .views import MeNextOfferView, RedeemOfferView, MeOffersView, OfferPreviewView

urlpatterns = [
    path("me/next-offer", MeNextOfferView.as_view(), name="me-next-offer"),
    path("offers/redeem", RedeemOfferView.as_view(), name="offers-redeem"),
    path("me/offers", MeOffersView.as_view(), name="me-offers"),
    path("offers/preview", OfferPreviewView.as_view(), name="offers-preview"),
]
