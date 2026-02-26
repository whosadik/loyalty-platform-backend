from django.urls import path
from admin_tools.views import AdminHealthView, AdminOverviewView, AdminRecsExperimentsView
from offers.views_admin_campaigns import AdminCampaignListCreateView, AdminCampaignDetailView

urlpatterns = [
    path("admin/health", AdminHealthView.as_view(), name="admin-health"),
    path("admin/overview", AdminOverviewView.as_view(), name="admin-overview"),
    path("admin/recs/experiments", AdminRecsExperimentsView.as_view(), name="admin-recs-experiments"),
    path("admin/campaigns", AdminCampaignListCreateView.as_view(), name="admin-campaigns"),
    path("admin/campaigns/<int:pk>", AdminCampaignDetailView.as_view(), name="admin-campaign-detail"),
]
