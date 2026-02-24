from django.urls import path
from .views import MeFavoriteCategoryView, MeProfileView

urlpatterns = [
    path("me/profile", MeProfileView.as_view(), name="me-profile"),
    path("me/favorite-category", MeFavoriteCategoryView.as_view(), name="me-favorite-category"),
]
