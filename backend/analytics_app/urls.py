from django.urls import path
from .views import AdminMetricsView

urlpatterns = [
    path("admin/metrics", AdminMetricsView.as_view(), name="admin-metrics"),
]
