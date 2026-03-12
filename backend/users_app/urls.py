from django.urls import path
from .views import MeFavoriteCategoryView, MeProfileTaxonomyView, MeProfileView

urlpatterns = [
    path("me/profile", MeProfileView.as_view(), name="me-profile"),
    path("me/profile-taxonomy", MeProfileTaxonomyView.as_view(), name="me-profile-taxonomy"),
    path("me/favorite-category", MeFavoriteCategoryView.as_view(), name="me-favorite-category"),
]
