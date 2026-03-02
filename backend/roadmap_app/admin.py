from django.contrib import admin

from roadmap_app.models import RoadmapEvent, RoadmapPlan, RoadmapStep


@admin.register(RoadmapPlan)
class RoadmapPlanAdmin(admin.ModelAdmin):
    list_display = ("id", "user", "category", "is_active", "version", "updated_at")
    list_filter = ("category", "is_active")
    search_fields = ("user__username", "user__email")


@admin.register(RoadmapStep)
class RoadmapStepAdmin(admin.ModelAdmin):
    list_display = ("id", "plan", "step_index", "product_type", "status", "recommended_product")
    list_filter = ("status", "plan__category")
    search_fields = ("plan__user__username", "product_type")


@admin.register(RoadmapEvent)
class RoadmapEventAdmin(admin.ModelAdmin):
    list_display = ("id", "created_at", "user", "event_type", "plan", "step")
    list_filter = ("event_type",)
    search_fields = ("user__username", "request_id")
