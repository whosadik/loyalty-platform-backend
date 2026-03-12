from django.urls import path
from .views import HomePromotionsView, MeNextOfferView, RedeemOfferView, MeOffersView, OfferPreviewView, OfferClickView
from offers.views_admin import AdminCacheInvalidateView
urlpatterns = [
    path("me/next-offer", MeNextOfferView.as_view(), name="me-next-offer"),
    path("me/home-promotions", HomePromotionsView.as_view(), name="me-home-promotions"),
    path("offers/redeem", RedeemOfferView.as_view(), name="offers-redeem"),
    path("offers/click", OfferClickView.as_view(), name="offers-click"),
    path("me/offers", MeOffersView.as_view(), name="me-offers"),
    path("offers/preview", OfferPreviewView.as_view(), name="offers-preview"),
    path("admin/cache/invalidate", AdminCacheInvalidateView.as_view(), name="admin-cache-invalidate"),
]
