from django.urls import path
from .views import CheckoutLastView, CheckoutPreviewView, CheckoutView

urlpatterns = [
    path("checkout", CheckoutView.as_view(), name="checkout"),
    path("checkout/last", CheckoutLastView.as_view(), name="checkout-last"),
    path("checkout/preview", CheckoutPreviewView.as_view(), name="checkout-preview"),
]
