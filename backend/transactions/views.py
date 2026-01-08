from rest_framework import viewsets
from rest_framework.decorators import action
from rest_framework.exceptions import MethodNotAllowed
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response

from .models import Transaction, OwnedProduct
from .serializers import TransactionSerializer, OwnedProductSerializer


class TransactionViewSet(viewsets.ModelViewSet):
    serializer_class = TransactionSerializer
    permission_classes = [IsAuthenticated]

    def get_queryset(self):
        return Transaction.objects.filter(user=self.request.user).order_by("-created_at")


class OwnedProductViewSet(viewsets.ModelViewSet):
    permission_classes = [IsAuthenticated]
    serializer_class = OwnedProductSerializer
    http_method_names = ["get", "post", "patch", "head", "options"]

    def get_queryset(self):
        return OwnedProduct.objects.filter(user=self.request.user).select_related("product").order_by("-last_acquired_at", "-id")

    def create(self, request, *args, **kwargs):
        raise MethodNotAllowed("post")

    @action(detail=True, methods=["post"])
    def deactivate(self, request, pk=None):
        obj = self.get_object()
        obj.is_active = False
        obj.save(update_fields=["is_active"])
        return Response({"ok": True, "id": obj.id, "is_active": obj.is_active})

    @action(detail=True, methods=["post"])
    def activate(self, request, pk=None):
        obj = self.get_object()
        obj.is_active = True
        obj.save(update_fields=["is_active"])
        return Response({"ok": True, "id": obj.id, "is_active": obj.is_active})

