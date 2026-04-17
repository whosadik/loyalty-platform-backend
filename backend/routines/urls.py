from django.urls import path
from .views import (
    RoutineGenerateView,
    RoutineHistoryView,
    RoutineValidateView,
    SavedRoutineView,
)

urlpatterns = [
    path("routine/generate", RoutineGenerateView.as_view(), name="routine-generate"),
    path("routine/validate", RoutineValidateView.as_view(), name="routine-validate"),
    path("routine/saved", SavedRoutineView.as_view(), name="routine-saved"),
    path("routine/history", RoutineHistoryView.as_view(), name="routine-history"),
]
