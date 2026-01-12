from django.db import models


class Product(models.Model):
    class Category(models.TextChoices):
        SKINCARE = "skincare", "Skincare"
        HAIRCARE = "haircare", "Haircare"
        MAKEUP = "makeup", "Makeup"
        FRAGRANCE = "fragrance", "Fragrance"

    class Step(models.TextChoices):
        # legacy: оставляем для совместимости (skincare routine)
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

    # NEW: универсальная таксономия
    category = models.CharField(max_length=30, choices=Category.choices, default=Category.SKINCARE, db_index=True)
    product_type = models.CharField(max_length=50, db_index=True)  # cleanser/lipstick/mascara/edp/shampoo/etc

    # NEW: универсальные поля для рекомендаций
    concerns = models.JSONField(default=list, blank=True)  # ["hydration","long_wear","anti_frizz"]
    attrs = models.JSONField(default=dict, blank=True)     # category-specific attributes

    # legacy для skincare (можно оставить, не использовать для новых категорий)
    step = models.CharField(max_length=30, choices=Step.choices, blank=True, default="")
    actives = models.JSONField(default=list, blank=True)
    flags = models.JSONField(default=list, blank=True)
    supported_skin_types = models.JSONField(default=list, blank=True)  # пустой список = подходит всем
    strength = models.CharField(max_length=20, choices=Strength.choices, default=Strength.LOW)

    in_stock = models.BooleanField(default=True)

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self) -> str:
        return f"{self.brand} {self.name}".strip()
