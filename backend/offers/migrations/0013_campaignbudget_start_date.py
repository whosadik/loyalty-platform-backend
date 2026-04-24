from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("offers", "0012_campaignbudget_end_date_campaignbudget_tiers_and_more"),
    ]

    operations = [
        migrations.AddField(
            model_name="campaignbudget",
            name="start_date",
            field=models.DateField(blank=True, null=True),
        ),
    ]
