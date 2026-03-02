from django.urls import path, include

urlpatterns = [
    # подключи сюда все свои app urls, как они были подключены в основном urls.py под /api/
    path("", include("catalog.urls")),
    path("", include("users_app.urls")),
    path("", include("routines.urls")),
    path("", include("offers.urls")),
    path("", include("checkout_app.urls")),
    path("", include("roadmap_app.urls")),
]
