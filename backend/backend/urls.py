from django.contrib import admin
from django.urls import path, include
from drf_spectacular.views import SpectacularAPIView, SpectacularSwaggerView
urlpatterns = [
    path("admin/", admin.site.urls),

    path("api/", include("catalog.urls")),
    path("api/", include("users_app.urls")),
    path("api/", include("routines.urls")),
    path("api/", include("transactions.urls")),
    path("api/", include("offers.urls")),
    path("api/", include("loyalty.urls")),
    path("api/", include("analytics_app.urls")),
    path("api/schema/", SpectacularAPIView.as_view(), name="schema"),
    path("api/docs/", SpectacularSwaggerView.as_view(url_name="schema"), name="swagger-ui"),
    path("api/", include("recs_app.urls")),
    path("api/", include("checkout_app.urls")),
    path("api/", include("audit.urls")),
    path("api/", include("admin_tools.urls")),
    path("api/", include("recs_analytics.urls")),
]
