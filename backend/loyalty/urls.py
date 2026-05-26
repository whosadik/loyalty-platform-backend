from django.urls import path
from .views import MeLoyaltyHistoryView, MeLoyaltyStatusView, RedeemPointsView

urlpatterns = [
    path("me/loyalty", MeLoyaltyStatusView.as_view(), name="me-loyalty"),
    path("me/loyalty/history", MeLoyaltyHistoryView.as_view(), name="me-loyalty-history"),
    path("loyalty/redeem-points", RedeemPointsView.as_view(), name="loyalty-redeem-points"),
]
