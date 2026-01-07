from django.urls import path
from .views import RoutineGenerateView, RoutineValidateView

urlpatterns = [
    path("routine/generate", RoutineGenerateView.as_view(), name="routine-generate"),
    path("routine/validate", RoutineValidateView.as_view(), name="routine-validate"),
]
