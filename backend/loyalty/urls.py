from django.urls import path
from .views import MeLoyaltyStatusView, RedeemPointsView

urlpatterns = [
    path("me/loyalty", MeLoyaltyStatusView.as_view(), name="me-loyalty"),
    path("loyalty/redeem-points", RedeemPointsView.as_view(), name="loyalty-redeem-points"),
]
