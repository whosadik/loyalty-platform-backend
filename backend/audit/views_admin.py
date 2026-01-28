from datetime import datetime
from django.utils.dateparse import parse_datetime
from django.db.models import Q

from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.permissions import IsAdminUser

from audit.models import AuditEvent
from audit.serializers import AuditEventSerializer
from backend.pagination import AdminAuditPagination
import csv
from django.http import StreamingHttpResponse


class AdminAuditListView(APIView):
    permission_classes = [IsAdminUser]

    def get(self, request):
        qs = AuditEvent.objects.all().order_by("-created_at")

        # filters
        action = request.query_params.get("action")
        if action:
            qs = qs.filter(action=action)

        user_id = request.query_params.get("user_id")
        if user_id:
            qs = qs.filter(user_id=user_id)

        request_id = request.query_params.get("request_id")
        if request_id:
            qs = qs.filter(request_id=request_id)

        entity_type = request.query_params.get("entity_type")
        if entity_type:
            qs = qs.filter(entity_type=entity_type)

        entity_id = request.query_params.get("entity_id")
        if entity_id:
            qs = qs.filter(entity_id=str(entity_id))

        path = request.query_params.get("path")
        if path:
            qs = qs.filter(path__icontains=path)

        status_code = request.query_params.get("status_code")
        if status_code:
            qs = qs.filter(status_code=status_code)

        since = request.query_params.get("since")
        if since:
            dt = parse_datetime(since)
            if dt:
                qs = qs.filter(created_at__gte=dt)

        until = request.query_params.get("until")
        if until:
            dt = parse_datetime(until)
            if dt:
                qs = qs.filter(created_at__lte=dt)

        paginator = AdminAuditPagination()
        page = paginator.paginate_queryset(qs, request)
        ser = AuditEventSerializer(page, many=True)
        return paginator.get_paginated_response(ser.data)

class Echo:
    def write(self, value):
        return value


class AdminAuditExportCsvView(APIView):
    permission_classes = [IsAdminUser]

    def get(self, request):
        # используем те же фильтры, что и list view
        qs = AuditEvent.objects.all().order_by("-created_at")

        action = request.query_params.get("action")
        if action:
            qs = qs.filter(action=action)

        user_id = request.query_params.get("user_id")
        if user_id:
            qs = qs.filter(user_id=user_id)

        request_id = request.query_params.get("request_id")
        if request_id:
            qs = qs.filter(request_id=request_id)

        entity_type = request.query_params.get("entity_type")
        if entity_type:
            qs = qs.filter(entity_type=entity_type)

        entity_id = request.query_params.get("entity_id")
        if entity_id:
            qs = qs.filter(entity_id=str(entity_id))

        path = request.query_params.get("path")
        if path:
            qs = qs.filter(path__icontains=path)

        status_code = request.query_params.get("status_code")
        if status_code:
            qs = qs.filter(status_code=status_code)

        since = request.query_params.get("since")
        if since:
            dt = parse_datetime(since)
            if dt:
                qs = qs.filter(created_at__gte=dt)

        until = request.query_params.get("until")
        if until:
            dt = parse_datetime(until)
            if dt:
                qs = qs.filter(created_at__lte=dt)

        # CSV streaming
        pseudo_buffer = Echo()
        writer = csv.writer(pseudo_buffer)

        header = [
            "id", "created_at", "action", "user_id",
            "entity_type", "entity_id",
            "request_id", "path", "method",
            "status_code", "ip", "meta",
        ]

        def row_iter():
            yield writer.writerow(header)
            for a in qs.iterator(chunk_size=2000):
                yield writer.writerow([
                    a.id,
                    a.created_at.isoformat(),
                    a.action,
                    a.user_id or "",
                    a.entity_type or "",
                    a.entity_id or "",
                    a.request_id or "",
                    a.path or "",
                    a.method or "",
                    a.status_code or "",
                    a.ip or "",
                    a.meta,
                ])

        resp = StreamingHttpResponse(row_iter(), content_type="text/csv")
        resp["Content-Disposition"] = 'attachment; filename="audit_export.csv"'
        return resp
