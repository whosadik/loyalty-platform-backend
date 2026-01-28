from django.urls import path
from audit.views_admin import AdminAuditExportCsvView, AdminAuditListView

urlpatterns = [
    path("admin/audit", AdminAuditListView.as_view(), name="admin-audit"),
    path("admin/audit/export.csv", AdminAuditExportCsvView.as_view(), name="admin-audit-export"),

]
