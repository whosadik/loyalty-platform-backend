from django.urls import path
from rest_framework.routers import DefaultRouter
from .views import (
    MeCartItemView,
    MeCartView,
    MeWishlistItemView,
    MeWishlistView,
    OwnedProductViewSet,
    TransactionViewSet,
)

router = DefaultRouter()

router.register(r"transactions", TransactionViewSet, basename="transactions")
router.register(r"me/owned-products", OwnedProductViewSet, basename="owned-products")


urlpatterns = [
    *router.urls,
    path("me/wishlist", MeWishlistView.as_view(), name="me-wishlist"),
    path("me/wishlist/<int:product_id>", MeWishlistItemView.as_view(), name="me-wishlist-item"),
    path("me/cart", MeCartView.as_view(), name="me-cart"),
    path("me/cart/<int:product_id>", MeCartItemView.as_view(), name="me-cart-item"),
]
