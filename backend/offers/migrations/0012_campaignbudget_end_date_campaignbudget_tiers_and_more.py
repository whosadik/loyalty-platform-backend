from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("offers", "0011_alter_offerevent_event_type"),
    ]

    operations = [
        migrations.AddField(
            model_name="campaignbudget",
            name="end_date",
            field=models.DateField(blank=True, null=True),
        ),
        migrations.AddField(
            model_name="campaignbudget",
            name="tiers",
            field=models.JSONField(blank=True, default=list),
        ),
        migrations.AddField(
            model_name="campaignbudget",
            name="promo_text",
            field=models.TextField(blank=True, default=""),
        ),
        migrations.AddField(
            model_name="campaignbudget",
            name="banner_url",
            field=models.URLField(blank=True, default=""),
        ),
    ]
