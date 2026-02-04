from django.urls import path
from admin_tools.views import AdminHealthView, AdminOverviewView
from offers.views_admin_campaigns import AdminCampaignListCreateView, AdminCampaignDetailView

urlpatterns = [
    path("admin/health", AdminHealthView.as_view(), name="admin-health"),
    path("admin/overview", AdminOverviewView.as_view(), name="admin-overview"),
    path("admin/campaigns", AdminCampaignListCreateView.as_view(), name="admin-campaigns"),
    path("admin/campaigns/<int:pk>", AdminCampaignDetailView.as_view(), name="admin-campaign-detail"),
]
