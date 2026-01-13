from django.urls import path
from .views import MeRecommendationsView, MeBundleView

urlpatterns = [
    path("me/recommendations", MeRecommendationsView.as_view(), name="me-recommendations"),
    path("me/recommendations/bundle", MeBundleView.as_view(), name="me-recommendations-bundle"),
]
