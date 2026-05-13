from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("offers", "0013_campaignbudget_start_date"),
    ]

    operations = [
        migrations.AddField(
            model_name="campaignbudget",
            name="allowed_brands",
            field=models.JSONField(blank=True, default=list),
        ),
        migrations.AddField(
            model_name="campaignbudget",
            name="allowed_product_ids",
            field=models.JSONField(blank=True, default=list),
        ),
        migrations.AddField(
            model_name="campaignbudget",
            name="campaign_type",
            field=models.CharField(
                choices=[("personal", "Personal"), ("public", "Public")],
                db_index=True,
                default="personal",
                max_length=20,
            ),
        ),
        migrations.AddField(
            model_name="campaignbudget",
            name="recommendation_rules",
            field=models.JSONField(blank=True, default=dict),
        ),
        migrations.AddField(
            model_name="offer",
            name="allowed_brands",
            field=models.JSONField(blank=True, default=list),
        ),
        migrations.AddField(
            model_name="offer",
            name="allowed_product_ids",
            field=models.JSONField(blank=True, default=list),
        ),
        migrations.AlterField(
            model_name="offer",
            name="target_scope",
            field=models.CharField(
                choices=[
                    ("cart", "Cart"),
                    ("category", "Category"),
                    ("brand", "Brand"),
                    ("product_type", "Product type"),
                    ("product_id", "Product"),
                ],
                default="cart",
                max_length=20,
            ),
        ),
    ]
