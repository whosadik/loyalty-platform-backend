from __future__ import annotations

from decimal import Decimal
from io import StringIO

from django.contrib.auth import get_user_model
from django.core.management import call_command
from django.db import IntegrityError, connection, transaction
from django.test import TestCase

from catalog.models import Product
from transactions.models import OwnedProduct


class FixPostgresSequencesTests(TestCase):
    def test_fix_ownedproduct_sequence_drift(self):
        if connection.vendor != "postgresql":
            self.skipTest("PostgreSQL-specific sequence test")

        User = get_user_model()
        u1 = User.objects.create_user(username="seqfix_u1", password="pass12345")
        u2 = User.objects.create_user(username="seqfix_u2", password="pass12345")
        u3 = User.objects.create_user(username="seqfix_u3", password="pass12345")
        u4 = User.objects.create_user(username="seqfix_u4", password="pass12345")

        p1 = Product.objects.create(
            name="SeqFix Product 1",
            brand="B",
            category="skincare",
            product_type="serum",
            price=Decimal("10.00"),
            in_stock=True,
        )
        p2 = Product.objects.create(
            name="SeqFix Product 2",
            brand="B",
            category="skincare",
            product_type="cleanser",
            price=Decimal("12.00"),
            in_stock=True,
        )
        p3 = Product.objects.create(
            name="SeqFix Product 3",
            brand="B",
            category="skincare",
            product_type="moisturizer",
            price=Decimal("15.00"),
            in_stock=True,
        )
        p4 = Product.objects.create(
            name="SeqFix Product 4",
            brand="B",
            category="skincare",
            product_type="spf",
            price=Decimal("14.00"),
            in_stock=True,
        )

        a = OwnedProduct.objects.create(user=u1, product=p1, quantity_total=1, source="manual")
        b = OwnedProduct.objects.create(user=u2, product=p2, quantity_total=1, source="manual")
        self.assertGreater(int(b.id), int(a.id))

        table_name = OwnedProduct._meta.db_table
        with connection.cursor() as cursor:
            cursor.execute("SELECT pg_get_serial_sequence(%s, %s)", [table_name, "id"])
            seq_name = cursor.fetchone()[0]
            self.assertTrue(seq_name)
            # Make nextval return an already used id.
            cursor.execute("SELECT setval(%s, %s, true)", [seq_name, int(a.id)])

        with transaction.atomic():
            with self.assertRaises(IntegrityError):
                OwnedProduct.objects.create(user=u3, product=p3, quantity_total=1, source="manual")

        out = StringIO()
        call_command("fix_postgres_sequences", apply=True, tables=[table_name], stdout=out)
        self.assertIn("applied", out.getvalue())

        created = OwnedProduct.objects.create(user=u4, product=p4, quantity_total=1, source="manual")
        self.assertGreater(int(created.id), int(b.id))
