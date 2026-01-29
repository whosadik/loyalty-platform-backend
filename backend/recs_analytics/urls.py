from django.urls import path
from recs_analytics.views import RecEventCreateView

urlpatterns = [
    path("me/recommendations/event", RecEventCreateView.as_view(), name="me-recs-event"),
]
