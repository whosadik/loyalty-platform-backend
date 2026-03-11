from __future__ import annotations

import tempfile
from decimal import Decimal
from pathlib import Path

from django.contrib.auth import get_user_model
from django.core.management import call_command
from django.test import TestCase
from rest_framework.test import APITestCase

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
                    "original_price",
                    "discount",
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
                    "serum",
                    "hydration",
                    "all",
                    "face",
                    "30 ml",
                    "2499",
                    "20",
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
        self.assertEqual(p.raw_meta["original_price"], "2499")
        self.assertEqual(p.raw_meta["discount"], 20)


class ProductSearchApiTests(APITestCase):
    def setUp(self):
        user_model = get_user_model()
        self.user = user_model.objects.create_user(username="catalog_search_u1", password="pass12345")
        self.client.force_authenticate(self.user)

        Product.objects.create(
            name="Ultra Hydration Cream",
            brand="DermaLab",
            price=Decimal("100.00"),
            category=Product.Category.SKINCARE,
            product_type="cream",
            concerns=[],
            attrs={},
            actives=[],
            flags=[],
            supported_skin_types=[],
            strength=Product.Strength.LOW,
            in_stock=True,
        )
        Product.objects.create(
            name="Color Pop Lipstick",
            brand="Glowify",
            price=Decimal("120.00"),
            category=Product.Category.MAKEUP,
            product_type="lipstick",
            concerns=[],
            attrs={},
            actives=[],
            flags=[],
            supported_skin_types=[],
            strength=Product.Strength.LOW,
            in_stock=True,
        )
        Product.objects.create(
            name="Ocean Breeze Perfume",
            brand="Aurum",
            price=Decimal("220.00"),
            category=Product.Category.FRAGRANCE,
            product_type="edp",
            concerns=[],
            attrs={},
            actives=[],
            flags=[],
            supported_skin_types=[],
            strength=Product.Strength.LOW,
            in_stock=True,
        )

    def test_search_filters_products_by_name_or_brand(self):
        resp = self.client.get("/api/products/?search=lip")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(len(resp.data), 1)
        self.assertEqual(resp.data[0]["name"], "Color Pop Lipstick")

        resp = self.client.get("/api/products/?search=dermalab")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(len(resp.data), 1)
        self.assertEqual(resp.data[0]["brand"], "DermaLab")


class ProductSaleApiTests(APITestCase):
    def setUp(self):
        user_model = get_user_model()
        self.user = user_model.objects.create_user(username="catalog_sale_u1", password="pass12345")
        self.client.force_authenticate(self.user)

        self.discounted = Product.objects.create(
            name="Sale Serum",
            brand="DermaLab",
            price=Decimal("100.00"),
            category=Product.Category.SKINCARE,
            product_type="serum",
            concerns=[],
            attrs={},
            actives=[],
            flags=[],
            supported_skin_types=[],
            strength=Product.Strength.LOW,
            in_stock=True,
            raw_meta={"original_price": "125.00"},
        )
        self.discount_only = Product.objects.create(
            name="Discount Mask",
            brand="Glowify",
            price=Decimal("80.00"),
            category=Product.Category.SKINCARE,
            product_type="mask",
            concerns=[],
            attrs={},
            actives=[],
            flags=[],
            supported_skin_types=[],
            strength=Product.Strength.LOW,
            in_stock=True,
            raw_meta={"discount": 20},
        )
        Product.objects.create(
            name="Regular Cleanser",
            brand="Aurum",
            price=Decimal("70.00"),
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

    def test_product_serializer_exposes_sale_fields(self):
        resp = self.client.get(f"/api/products/{self.discounted.id}/")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.data["original_price"], "125.00")
        self.assertEqual(resp.data["discount"], 20)
        self.assertTrue(resp.data["has_discount"])

        resp = self.client.get(f"/api/products/{self.discount_only.id}/")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.data["discount"], 20)
        self.assertEqual(resp.data["original_price"], "100.00")
        self.assertTrue(resp.data["has_discount"])

    def test_sale_query_param_returns_only_discounted_products(self):
        resp = self.client.get("/api/products/?sale=true")
        self.assertEqual(resp.status_code, 200)

        names = {item["name"] for item in resp.data}
        self.assertEqual(names, {"Sale Serum", "Discount Mask"})
