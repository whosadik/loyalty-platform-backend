from rest_framework.routers import DefaultRouter
from .views import TransactionViewSet

router = DefaultRouter()
from .views import TransactionViewSet, OwnedProductViewSet

router.register(r"transactions", TransactionViewSet, basename="transactions")
router.register(r"me/owned-products", OwnedProductViewSet, basename="owned-products")


urlpatterns = router.urls