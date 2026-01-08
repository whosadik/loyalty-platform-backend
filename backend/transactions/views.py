from rest_framework import viewsets
from rest_framework.permissions import IsAuthenticated

from .models import Transaction, OwnedProduct
from .serializers import TransactionSerializer, OwnedProductSerializer


class TransactionViewSet(viewsets.ModelViewSet):
    serializer_class = TransactionSerializer
    permission_classes = [IsAuthenticated]

    def get_queryset(self):
        return Transaction.objects.filter(user=self.request.user).order_by("-created_at")


class OwnedProductViewSet(viewsets.ReadOnlyModelViewSet):
    permission_classes = [IsAuthenticated]
    serializer_class = OwnedProductSerializer

    def get_queryset(self):
        return OwnedProduct.objects.filter(user=self.request.user).select_related("product").order_by("-acquired_at")
