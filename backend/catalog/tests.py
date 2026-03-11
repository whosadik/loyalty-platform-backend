from __future__ import annotations

import tempfile
from datetime import timedelta
from decimal import Decimal
from pathlib import Path

from django.contrib.auth import get_user_model
from django.core.management import call_command
from django.test import TestCase
from django.utils import timezone
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
                    "rating",
                    "reviews_count",
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
                    "4.8",
                    "56",
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
        self.assertEqual(p.raw_meta["rating"], "4.8")
        self.assertEqual(p.raw_meta["reviews_count"], 56)


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

    def test_search_filters_products_by_category(self):
        resp = self.client.get("/api/products/?search=skincare")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(len(resp.data), 1)
        self.assertEqual(resp.data[0]["category"], Product.Category.SKINCARE)

    def test_products_list_supports_opt_in_pagination(self):
        for index in range(15):
            Product.objects.create(
                name=f"Paged Product {index}",
                brand="Paged Brand",
                price=Decimal("90.00"),
                category=Product.Category.SKINCARE,
                product_type="serum",
                concerns=[],
                attrs={},
                actives=[],
                flags=[],
                supported_skin_types=[],
                strength=Product.Strength.LOW,
                in_stock=True,
            )

        resp = self.client.get("/api/products/?page=1&page_size=12")
        self.assertEqual(resp.status_code, 200)
        self.assertIsInstance(resp.data, dict)
        self.assertEqual(resp.data["count"], 18)
        self.assertEqual(len(resp.data["results"]), 12)
        self.assertIsNotNone(resp.data["next"])

    def test_products_list_without_page_keeps_legacy_array_shape(self):
        resp = self.client.get("/api/products/")
        self.assertEqual(resp.status_code, 200)
        self.assertIsInstance(resp.data, list)

    def test_new_query_param_returns_only_recent_products(self):
        fresh = Product.objects.create(
            name="Fresh Serum",
            brand="Fresh Lab",
            price=Decimal("99.00"),
            category=Product.Category.SKINCARE,
            product_type="serum",
            concerns=[],
            attrs={},
            actives=[],
            flags=[],
            supported_skin_types=[],
            strength=Product.Strength.LOW,
            in_stock=True,
        )
        old = Product.objects.create(
            name="Old Mask",
            brand="Archive Lab",
            price=Decimal("89.00"),
            category=Product.Category.SKINCARE,
            product_type="mask",
            concerns=[],
            attrs={},
            actives=[],
            flags=[],
            supported_skin_types=[],
            strength=Product.Strength.LOW,
            in_stock=True,
        )
        Product.objects.filter(id=old.id).update(created_at=timezone.now() - timedelta(days=90))
        old.refresh_from_db()

        resp = self.client.get("/api/products/?new=true")
        self.assertEqual(resp.status_code, 200)

        names = {item["name"] for item in resp.data}
        self.assertIn(fresh.name, names)
        self.assertNotIn(old.name, names)

        fresh_payload = next(item for item in resp.data if item["name"] == fresh.name)
        self.assertTrue(fresh_payload["is_new"])


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
        self.assertEqual(resp.data["brand_slug"], "dermalab")
        self.assertEqual(resp.data["points_earned"], 10)

        resp = self.client.get(f"/api/products/{self.discount_only.id}/")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.data["discount"], 20)
        self.assertEqual(resp.data["original_price"], "100.00")
        self.assertTrue(resp.data["has_discount"])

    def test_product_serializer_exposes_social_proof_fields(self):
        self.discounted.raw_meta = {
            **self.discounted.raw_meta,
            "rating": "4.7",
            "reviews_count": 42,
        }
        self.discounted.save(update_fields=["raw_meta", "updated_at"])

        resp = self.client.get(f"/api/products/{self.discounted.id}/")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.data["rating"], 4.7)
        self.assertEqual(resp.data["reviews_count"], 42)

    def test_sale_query_param_returns_only_discounted_products(self):
        resp = self.client.get("/api/products/?sale=true")
        self.assertEqual(resp.status_code, 200)

        names = {item["name"] for item in resp.data}
        self.assertEqual(names, {"Sale Serum", "Discount Mask"})


class BrandApiTests(APITestCase):
    def setUp(self):
        user_model = get_user_model()
        self.user = user_model.objects.create_user(username="catalog_brand_u1", password="pass12345")
        self.client.force_authenticate(self.user)

        Product.objects.create(
            name="Bright Serum",
            brand="Glow Lab",
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
            raw_meta={"discount": 15},
        )
        Product.objects.create(
            name="Soft Cleanser",
            brand="Glow Lab",
            price=Decimal("80.00"),
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
        Product.objects.create(
            name="Night Cream",
            brand="Derma Stories",
            price=Decimal("120.00"),
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

    def test_brands_list_returns_slug_and_product_count(self):
        resp = self.client.get("/api/brands/")
        self.assertEqual(resp.status_code, 200)

        glow_lab = next((item for item in resp.data if item["name"] == "Glow Lab"), None)
        self.assertIsNotNone(glow_lab)
        self.assertEqual(glow_lab["slug"], "glow-lab")
        self.assertEqual(glow_lab["product_count"], 2)

    def test_brand_detail_returns_meta_for_slug(self):
        resp = self.client.get("/api/brands/glow-lab/")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.data["name"], "Glow Lab")
        self.assertEqual(resp.data["product_count"], 2)
        self.assertEqual(resp.data["sale_products_count"], 1)
        self.assertIn("skincare", resp.data["categories"])
        self.assertIn("serum", resp.data["top_product_types"])
        self.assertTrue(resp.data["description"].startswith("Glow Lab"))
