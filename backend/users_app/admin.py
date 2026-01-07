from django.contrib import admin
from .models import CustomerProfile

@admin.register(CustomerProfile)
class CustomerProfileAdmin(admin.ModelAdmin):
    list_display = ("id", "user", "skin_type", "budget", "updated_at")
    list_filter = ("skin_type", "budget")
    search_fields = ("user__username", "user__email")
