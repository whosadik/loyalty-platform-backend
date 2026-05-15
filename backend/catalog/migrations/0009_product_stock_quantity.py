from django.db import migrations, models


DEFAULT_BACKFILL_QUANTITY = 100


def backfill_stock_quantity(apps, schema_editor):
    Product = apps.get_model("catalog", "Product")
    Product.objects.filter(in_stock=True, stock_quantity=0).update(
        stock_quantity=DEFAULT_BACKFILL_QUANTITY,
    )


def reverse_noop(apps, schema_editor):
    pass


class Migration(migrations.Migration):

    dependencies = [
        ("catalog", "0008_brand_backfill"),
    ]

    operations = [
        migrations.AddField(
            model_name="product",
            name="stock_quantity",
            field=models.PositiveIntegerField(default=0),
        ),
        migrations.RunPython(backfill_stock_quantity, reverse_code=reverse_noop),
    ]
