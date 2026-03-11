from django.db.models import Sum
from django.shortcuts import get_object_or_404
from rest_framework import status, viewsets
from rest_framework.decorators import action
from rest_framework.exceptions import MethodNotAllowed
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework.views import APIView
from drf_spectacular.utils import extend_schema, extend_schema_view

from catalog.models import Product
from .models import CartItem, OwnedProduct, Transaction, WishlistItem
from .serializers import (
    CartAddSerializer,
    CartItemSerializer,
    CartPatchSerializer,
    OwnedProductSerializer,
    TransactionSerializer,
    WishlistAddSerializer,
    WishlistItemSerializer,
)


class TransactionViewSet(viewsets.ReadOnlyModelViewSet):
    serializer_class = TransactionSerializer
    permission_classes = [IsAuthenticated]

    def get_queryset(self):
        return (
            Transaction.objects.filter(user=self.request.user)
            .prefetch_related("items__product")
            .order_by("-created_at", "-id")
        )


@extend_schema_view(
    create=extend_schema(exclude=True),
)
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
        return Response(
            {
                "ok": True,
                "id": obj.id,
                "is_active": obj.is_active,
                "owned_product": self.get_serializer(obj).data,
            }
        )

    @action(detail=True, methods=["post"])
    def activate(self, request, pk=None):
        obj = self.get_object()
        obj.is_active = True
        obj.save(update_fields=["is_active"])
        return Response(
            {
                "ok": True,
                "id": obj.id,
                "is_active": obj.is_active,
                "owned_product": self.get_serializer(obj).data,
            }
        )


class MeWishlistView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request):
        qs = WishlistItem.objects.filter(user=request.user).select_related("product").order_by("-created_at", "-id")
        items = WishlistItemSerializer(qs, many=True).data
        return Response({"ok": True, "count": qs.count(), "items": items})

    def post(self, request):
        serializer = WishlistAddSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)

        product_id = serializer.validated_data["product_id"]
        product = get_object_or_404(Product, id=product_id)
        item, created = WishlistItem.objects.get_or_create(user=request.user, product=product)

        count = WishlistItem.objects.filter(user=request.user).count()
        return Response(
            {
                "ok": True,
                "created": created,
                "count": count,
                "item": WishlistItemSerializer(item).data,
            },
            status=status.HTTP_201_CREATED if created else status.HTTP_200_OK,
        )


class MeWishlistItemView(APIView):
    permission_classes = [IsAuthenticated]

    def delete(self, request, product_id: int):
        deleted, _ = WishlistItem.objects.filter(user=request.user, product_id=product_id).delete()
        count = WishlistItem.objects.filter(user=request.user).count()
        return Response(
            {
                "ok": True,
                "product_id": int(product_id),
                "deleted": int(bool(deleted)),
                "count": count,
            }
        )


class MeCartView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request):
        qs = CartItem.objects.filter(user=request.user).select_related("product").order_by("-updated_at", "-id")
        items = CartItemSerializer(qs, many=True).data
        total_quantity = int(qs.aggregate(total=Sum("quantity"))["total"] or 0)
        return Response(
            {
                "ok": True,
                "count": qs.count(),
                "total_quantity": total_quantity,
                "items": items,
            }
        )

    def post(self, request):
        serializer = CartAddSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)

        product_id = serializer.validated_data["product_id"]
        quantity = serializer.validated_data.get("quantity", 1)

        product = get_object_or_404(Product, id=product_id)
        item, created = CartItem.objects.get_or_create(
            user=request.user,
            product=product,
            defaults={"quantity": int(quantity)},
        )
        if not created:
            item.quantity = int(item.quantity or 0) + int(quantity)
            item.save(update_fields=["quantity", "updated_at"])

        qs = CartItem.objects.filter(user=request.user)
        total_quantity = int(qs.aggregate(total=Sum("quantity"))["total"] or 0)
        return Response(
            {
                "ok": True,
                "created": created,
                "count": qs.count(),
                "total_quantity": total_quantity,
                "item": CartItemSerializer(item).data,
            },
            status=status.HTTP_201_CREATED if created else status.HTTP_200_OK,
        )


class MeCartItemView(APIView):
    permission_classes = [IsAuthenticated]

    def patch(self, request, product_id: int):
        serializer = CartPatchSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        quantity = serializer.validated_data["quantity"]

        item = get_object_or_404(CartItem, user=request.user, product_id=product_id)

        if int(quantity) <= 0:
            item.delete()
            qs = CartItem.objects.filter(user=request.user)
            total_quantity = int(qs.aggregate(total=Sum("quantity"))["total"] or 0)
            return Response(
                {
                    "ok": True,
                    "product_id": int(product_id),
                    "deleted": 1,
                    "count": qs.count(),
                    "total_quantity": total_quantity,
                }
            )

        item.quantity = int(quantity)
        item.save(update_fields=["quantity", "updated_at"])

        qs = CartItem.objects.filter(user=request.user)
        total_quantity = int(qs.aggregate(total=Sum("quantity"))["total"] or 0)
        return Response(
            {
                "ok": True,
                "count": qs.count(),
                "total_quantity": total_quantity,
                "item": CartItemSerializer(item).data,
            }
        )

    def delete(self, request, product_id: int):
        deleted, _ = CartItem.objects.filter(user=request.user, product_id=product_id).delete()
        qs = CartItem.objects.filter(user=request.user)
        total_quantity = int(qs.aggregate(total=Sum("quantity"))["total"] or 0)
        return Response(
            {
                "ok": True,
                "product_id": int(product_id),
                "deleted": int(bool(deleted)),
                "count": qs.count(),
                "total_quantity": total_quantity,
            }
        )

