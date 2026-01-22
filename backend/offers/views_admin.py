from django.core.cache import cache
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.permissions import IsAdminUser


class AdminCacheInvalidateView(APIView):
    permission_classes = [IsAdminUser]

    def post(self, request):
        keys = ["recs:products:v1", "recs:cooc90d:v1"]
        deleted = 0
        for k in keys:
            if cache.delete(k):
                deleted += 1
        return Response({"ok": True, "deleted": deleted, "keys": keys})
