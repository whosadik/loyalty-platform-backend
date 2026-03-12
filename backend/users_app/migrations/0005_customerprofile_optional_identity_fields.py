from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("users_app", "0004_customerprofile_email_verification_fields"),
    ]

    operations = [
        migrations.AddField(
            model_name="customerprofile",
            name="city",
            field=models.CharField(blank=True, default="", max_length=120),
        ),
        migrations.AddField(
            model_name="customerprofile",
            name="first_name",
            field=models.CharField(blank=True, default="", max_length=150),
        ),
        migrations.AddField(
            model_name="customerprofile",
            name="last_name",
            field=models.CharField(blank=True, default="", max_length=150),
        ),
        migrations.AddField(
            model_name="customerprofile",
            name="phone",
            field=models.CharField(blank=True, default="", max_length=32),
        ),
    ]
