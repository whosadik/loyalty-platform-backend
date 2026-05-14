from decimal import Decimal

from django.db import migrations


NEW_THRESHOLDS = {
    "Bronze": Decimal("0"),
    "Silver": Decimal("50000"),
    "Gold": Decimal("300000"),
}

OLD_THRESHOLDS = {
    "Bronze": Decimal("0"),
    "Silver": Decimal("100"),
    "Gold": Decimal("250"),
}


def _apply(thresholds):
    def _runner(apps, schema_editor):
        Tier = apps.get_model("loyalty", "Tier")
        for name, value in thresholds.items():
            Tier.objects.filter(name=name).update(threshold_spend_90d=value)
    return _runner


class Migration(migrations.Migration):
    dependencies = [
        ("loyalty", "0003_points_rate_one_percent"),
    ]

    operations = [
        migrations.RunPython(_apply(NEW_THRESHOLDS), _apply(OLD_THRESHOLDS)),
    ]
