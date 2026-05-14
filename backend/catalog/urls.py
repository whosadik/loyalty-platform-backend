from django.urls import path
from rest_framework.routers import DefaultRouter
from .views import BrandDetailView, BrandListView, HomeHeroView, ProductViewSet
from .manual_images import AttachImageByUrlView, manual_images_page

router = DefaultRouter()
router.register(r"products", ProductViewSet, basename="products")

urlpatterns = [
    path("home/hero", HomeHeroView.as_view()),
    path("brands/", BrandListView.as_view()),
    path("brands/<str:brand_slug>/", BrandDetailView.as_view()),
    path("admin/manual-images/", manual_images_page, name="manual-images"),
    path("admin/attach_image_by_url/", AttachImageByUrlView.as_view(), name="attach-image-by-url"),
    *router.urls,
]
