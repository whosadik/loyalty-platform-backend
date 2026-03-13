from decimal import Decimal, ROUND_HALF_UP

from django.db import migrations, models


DEFAULT_POINTS_RATE = Decimal("0.10")


def normalize_points_rates(apps, schema_editor):
    Tier = apps.get_model("loyalty", "Tier")

    for tier in Tier.objects.all():
        try:
            rate = Decimal(str(tier.points_rate))
        except Exception:
            rate = DEFAULT_POINTS_RATE

        if not rate.is_finite() or rate <= 0:
            rate = DEFAULT_POINTS_RATE
        elif rate >= Decimal("1"):
            rate = (rate / Decimal("10")).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

        if rate > DEFAULT_POINTS_RATE:
            rate = DEFAULT_POINTS_RATE

        rate = rate.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
        if tier.points_rate != rate:
            tier.points_rate = rate
            tier.save(update_fields=["points_rate"])


def noop_reverse(apps, schema_editor):
    return


class Migration(migrations.Migration):
    dependencies = [
        ("loyalty", "0001_initial"),
    ]

    operations = [
        migrations.AlterField(
            model_name="tier",
            name="points_rate",
            field=models.DecimalField(decimal_places=2, default=Decimal("0.10"), max_digits=10),
        ),
        migrations.RunPython(normalize_points_rates, noop_reverse),
    ]
