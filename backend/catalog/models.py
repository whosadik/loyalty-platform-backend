from django.db import models


class Product(models.Model):
    class Category(models.TextChoices):
        SKINCARE = "skincare", "Skincare"
        HAIRCARE = "haircare", "Haircare"
        MAKEUP = "makeup", "Makeup"
        FRAGRANCE = "fragrance", "Fragrance"

    class Step(models.TextChoices):
        # Legacy routine steps for skincare compatibility.
        CLEANSER = "cleanser", "Cleanser"
        TONER = "toner", "Toner"
        SERUM = "serum", "Serum"
        MOISTURIZER = "moisturizer", "Moisturizer"
        SPF = "spf", "SPF"
        MASK = "mask", "Mask"

    class Strength(models.TextChoices):
        LOW = "low", "Low"
        MEDIUM = "medium", "Medium"
        HIGH = "high", "High"

    name = models.CharField(max_length=200)
    brand = models.CharField(max_length=120, blank=True, default="")
    price = models.DecimalField(max_digits=10, decimal_places=2, null=True, blank=True)
    source_product_id = models.CharField(max_length=64, blank=True, default="", db_index=True)
    currency = models.CharField(max_length=8, blank=True, default="")

    # Unified taxonomy fields.
    category = models.CharField(max_length=30, choices=Category.choices, default=Category.SKINCARE, db_index=True)
    product_type = models.CharField(max_length=50, db_index=True)

    # Generic recommendation fields.
    concerns = models.JSONField(default=list, blank=True)
    attrs = models.JSONField(default=dict, blank=True)

    # Legacy skincare-specific fields.
    step = models.CharField(max_length=30, choices=Step.choices, blank=True, default="")
    actives = models.JSONField(default=list, blank=True)
    flags = models.JSONField(default=list, blank=True)
    supported_skin_types = models.JSONField(default=list, blank=True)
    strength = models.CharField(max_length=20, choices=Strength.choices, default=Strength.LOW)

    in_stock = models.BooleanField(default=True)

    # PDP/content fields from external catalog.
    image_url = models.URLField(max_length=500, blank=True, default="")
    image_urls = models.JSONField(default=list, blank=True)
    description = models.TextField(blank=True, default="")
    application_text = models.TextField(blank=True, default="")
    ingredients_inci = models.TextField(blank=True, default="")
    volume_raw = models.TextField(blank=True, default="")
    raw_meta = models.JSONField(default=dict, blank=True)

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self) -> str:
        return f"{self.brand} {self.name}".strip()
