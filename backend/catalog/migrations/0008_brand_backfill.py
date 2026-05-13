from django.db import migrations
from django.utils.text import slugify


def _unique_slug(Brand, base_slug: str) -> str:
    candidate = base_slug or "brand"
    suffix = 2
    while Brand.objects.filter(slug=candidate).exists():
        candidate = f"{base_slug}-{suffix}"
        suffix += 1
    return candidate


def populate_brand_ref(apps, schema_editor):
    Brand = apps.get_model("catalog", "Brand")
    Product = apps.get_model("catalog", "Product")

    name_to_brand: dict[str, object] = {}

    distinct_names = (
        Product.objects.exclude(brand="")
        .values_list("brand", flat=True)
        .distinct()
    )

    for raw_name in distinct_names:
        normalized_name = (raw_name or "").strip()
        if not normalized_name:
            continue

        existing = Brand.objects.filter(name__iexact=normalized_name).first()
        if existing is not None:
            name_to_brand[normalized_name.lower()] = existing
            continue

        base_slug = slugify(normalized_name, allow_unicode=True)
        slug = _unique_slug(Brand, base_slug)
        brand = Brand.objects.create(name=normalized_name, slug=slug)
        name_to_brand[normalized_name.lower()] = brand

    for product in Product.objects.exclude(brand="").only("id", "brand", "brand_ref_id").iterator():
        key = (product.brand or "").strip().lower()
        brand = name_to_brand.get(key)
        if brand is None:
            continue
        if product.brand_ref_id == brand.id:
            continue
        product.brand_ref_id = brand.id
        product.save(update_fields=["brand_ref"])


def reverse_populate(apps, schema_editor):
    Product = apps.get_model("catalog", "Product")
    Product.objects.exclude(brand_ref__isnull=True).update(brand_ref=None)
    Brand = apps.get_model("catalog", "Brand")
    Brand.objects.all().delete()


class Migration(migrations.Migration):

    dependencies = [
        ("catalog", "0007_brand_model"),
    ]

    operations = [
        migrations.RunPython(populate_brand_ref, reverse_code=reverse_populate),
    ]
