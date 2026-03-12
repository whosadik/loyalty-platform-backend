from django.urls import path
from .views import AdminMetricsExportCsvView, AdminMetricsView

urlpatterns = [
    path("admin/metrics", AdminMetricsView.as_view(), name="admin-metrics"),
    path("admin/metrics/export", AdminMetricsExportCsvView.as_view(), name="admin-metrics-export"),
]
