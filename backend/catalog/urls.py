from django.urls import path
from rest_framework.routers import DefaultRouter
from .views import BrandDetailView, BrandListView, ProductViewSet

router = DefaultRouter()
router.register(r"products", ProductViewSet, basename="products")

urlpatterns = [
    path("brands/", BrandListView.as_view()),
    path("brands/<str:brand_slug>/", BrandDetailView.as_view()),
    *router.urls,
]
