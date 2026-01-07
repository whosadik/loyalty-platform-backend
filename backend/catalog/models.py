from django.db import models


class Product(models.Model):
    class Step(models.TextChoices):
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

    step = models.CharField(max_length=30, choices=Step.choices)
    actives = models.JSONField(default=list, blank=True)          # пример: ["bha", "niacinamide"]
    flags = models.JSONField(default=list, blank=True)            # пример: ["fragrance", "alcohol"]
    supported_skin_types = models.JSONField(default=list, blank=True)  # пример: ["oily", "sensitive"]
    strength = models.CharField(max_length=20, choices=Strength.choices, default=Strength.LOW)

    in_stock = models.BooleanField(default=True)

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self) -> str:
        return f"{self.brand} {self.name}".strip()
