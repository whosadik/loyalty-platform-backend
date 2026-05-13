from django.urls import path
from rest_framework.routers import DefaultRouter

from admin_tools.views import AdminHealthView, AdminOverviewView, AdminRecsExperimentsView
from catalog.views_admin import AdminBrandViewSet, AdminProductViewSet
from offers.views_admin_campaigns import (
    AdminCampaignBannerUploadView,
    AdminCampaignDetailView,
    AdminCampaignListCreateView,
    AdminCampaignPublishView,
    AdminCampaignRecommendationsView,
)
from offers.views_admin_offers import (
    AdminOfferDetailView,
    AdminOfferListCreateView,
)


router = DefaultRouter()
router.register(r"admin/brands", AdminBrandViewSet, basename="admin-brands")
router.register(r"admin/products", AdminProductViewSet, basename="admin-products")


urlpatterns = [
    path("admin/health", AdminHealthView.as_view(), name="admin-health"),
    path("admin/overview", AdminOverviewView.as_view(), name="admin-overview"),
    path("admin/recs/experiments", AdminRecsExperimentsView.as_view(), name="admin-recs-experiments"),
    path("admin/campaigns", AdminCampaignListCreateView.as_view(), name="admin-campaigns"),
    path("admin/campaigns/<int:pk>", AdminCampaignDetailView.as_view(), name="admin-campaign-detail"),
    path("admin/campaigns/<int:pk>/publish", AdminCampaignPublishView.as_view(), name="admin-campaign-publish"),
    path("admin/campaigns/<int:pk>/banner", AdminCampaignBannerUploadView.as_view(), name="admin-campaign-banner"),
    path(
        "admin/campaigns/<int:pk>/recommendations",
        AdminCampaignRecommendationsView.as_view(),
        name="admin-campaign-recommendations",
    ),
    path("admin/offers", AdminOfferListCreateView.as_view(), name="admin-offers"),
    path("admin/offers/<int:pk>", AdminOfferDetailView.as_view(), name="admin-offer-detail"),
    *router.urls,
]
