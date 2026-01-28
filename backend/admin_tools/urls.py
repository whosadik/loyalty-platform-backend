from django.urls import path
from admin_tools.views import AdminHealthView, AdminOverviewView

urlpatterns = [
    path("admin/health", AdminHealthView.as_view(), name="admin-health"),
    path("admin/overview", AdminOverviewView.as_view(), name="admin-overview"),
]
