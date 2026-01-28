from django.urls import path
from audit.views_admin import AdminAuditListView

urlpatterns = [
    path("admin/audit", AdminAuditListView.as_view(), name="admin-audit"),
]
