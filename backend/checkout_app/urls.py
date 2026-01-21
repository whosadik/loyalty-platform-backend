from django.urls import path
from .views import CheckoutView, CheckoutPreviewView

urlpatterns = [
    path("checkout", CheckoutView.as_view(), name="checkout"),
    path("checkout/preview", CheckoutPreviewView.as_view(), name="checkout-preview"),
]