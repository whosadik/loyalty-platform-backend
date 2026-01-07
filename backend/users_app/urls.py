from django.urls import path
from .views import MeProfileView

urlpatterns = [
    path("me/profile", MeProfileView.as_view(), name="me-profile"),
]
