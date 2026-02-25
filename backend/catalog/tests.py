from __future__ import annotations

import tempfile
from decimal import Decimal
from pathlib import Path

from django.core.management import call_command
from django.test import TestCase

from catalog.models import Product

try:
    from openpyxl import Workbook
except Exception:  # pragma: no cover
    Workbook = None


class ImportProductsXlsxTests(TestCase):
    def test_import_products_xlsx_replace(self):
        if Workbook is None:
            self.skipTest("openpyxl is not installed")

        Product.objects.create(
            name="Old Product",
            brand="Old Brand",
            price=Decimal("10.00"),
            category=Product.Category.SKINCARE,
            product_type="cleanser",
            concerns=[],
            attrs={},
            actives=[],
            flags=[],
            supported_skin_types=[],
            strength=Product.Strength.LOW,
            in_stock=True,
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            xlsx_path = Path(tmp_dir) / "catalog.xlsx"
            wb = Workbook()
            ws = wb.active
            ws.title = "Sheet1"
            ws.append(
                [
                    "id",
                    "name",
                    "brand",
                    "price",
                    "currency",
                    "category",
                    "product_type",
                    "concerns",
                    "attrs",
                    "actives",
                    "flags",
                    "supported_skin_types",
                    "strength",
                    "in_stock",
                    "step",
                    "photo_url_primary",
                    "photo_urls",
                    "application_text",
                    "ingredients_inci",
                    "description_text",
                    "product_type_raw",
                    "concerns_raw",
                    "supported_skin_types_raw",
                    "area_raw",
                    "volume_raw",
                ]
            )
            ws.append(
                [
                    "ga-1",
                    "Imported Product",
                    "Imported Brand",
                    "1999",
                    "KZT",
                    "skincare",
                    "serum",
                    '["hydration"]',
                    '{"area": "face", "volume_ml": 30}',
                    '["niacinamide"]',
                    '["fragrance"]',
                    '["all"]',
                    "medium",
                    1,
                    "3",
                    "https://example.com/img.jpg",
                    '["https://example.com/img.jpg"]',
                    "Apply daily",
                    "Aqua, Niacinamide",
                    "Product description",
                    "сыворотка",
                    "увлажнение",
                    "для всех типов кожи",
                    "лицо",
                    "30 мл",
                ]
            )
            wb.save(xlsx_path)

            call_command("import_products_xlsx", path=str(xlsx_path), replace=True)

        self.assertEqual(Product.objects.count(), 1)
        p = Product.objects.get()
        self.assertEqual(p.source_product_id, "ga-1")
        self.assertEqual(p.name, "Imported Product")
        self.assertEqual(p.brand, "Imported Brand")
        self.assertEqual(p.price, Decimal("1999"))
        self.assertEqual(p.currency, "KZT")
        self.assertEqual(p.category, Product.Category.SKINCARE)
        self.assertEqual(p.product_type, "serum")
        self.assertEqual(p.step, "serum")
        self.assertEqual(p.supported_skin_types, [])
        self.assertEqual(p.image_url, "https://example.com/img.jpg")
        self.assertEqual(p.description, "Product description")
