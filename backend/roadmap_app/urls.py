from django.urls import path

from roadmap_app.views import MeRoadmapRefreshView, MeRoadmapStepPatchView, MeRoadmapView

urlpatterns = [
    path("me/roadmap", MeRoadmapView.as_view(), name="me-roadmap"),
    path("me/roadmap/refresh", MeRoadmapRefreshView.as_view(), name="me-roadmap-refresh"),
    path("me/roadmap/steps/<int:step_id>", MeRoadmapStepPatchView.as_view(), name="me-roadmap-step-patch"),
]
