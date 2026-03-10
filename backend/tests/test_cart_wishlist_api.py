from django.contrib.auth import get_user_model
from rest_framework.test import APITestCase

from catalog.models import Product


class CartWishlistApiTests(APITestCase):
    def setUp(self):
        User = get_user_model()
        self.user = User.objects.create_user(username="cartwish_u1", password="pass12345")
        self.client.force_authenticate(self.user)

        self.product_1 = Product.objects.create(name="Serum A", product_type="serum")
        self.product_2 = Product.objects.create(name="Cream B", product_type="moisturizer")

    def test_wishlist_flow(self):
        empty = self.client.get("/api/me/wishlist")
        self.assertEqual(empty.status_code, 200)
        self.assertEqual(empty.data["count"], 0)
        self.assertEqual(empty.data["items"], [])

        created = self.client.post("/api/me/wishlist", {"product_id": self.product_1.id}, format="json")
        self.assertEqual(created.status_code, 201)
        self.assertTrue(created.data["ok"])
        self.assertTrue(created.data["created"])
        self.assertEqual(created.data["count"], 1)
        self.assertEqual(created.data["item"]["product"]["id"], self.product_1.id)

        duplicate = self.client.post("/api/me/wishlist", {"product_id": self.product_1.id}, format="json")
        self.assertEqual(duplicate.status_code, 200)
        self.assertTrue(duplicate.data["ok"])
        self.assertFalse(duplicate.data["created"])
        self.assertEqual(duplicate.data["count"], 1)

        listed = self.client.get("/api/me/wishlist")
        self.assertEqual(listed.status_code, 200)
        self.assertEqual(listed.data["count"], 1)
        self.assertEqual(listed.data["items"][0]["product"]["id"], self.product_1.id)

        removed = self.client.delete(f"/api/me/wishlist/{self.product_1.id}")
        self.assertEqual(removed.status_code, 200)
        self.assertTrue(removed.data["ok"])
        self.assertEqual(removed.data["deleted"], 1)
        self.assertEqual(removed.data["count"], 0)

    def test_cart_flow(self):
        created = self.client.post(
            "/api/me/cart",
            {"product_id": self.product_1.id, "quantity": 2},
            format="json",
        )
        self.assertEqual(created.status_code, 201)
        self.assertTrue(created.data["ok"])
        self.assertTrue(created.data["created"])
        self.assertEqual(created.data["count"], 1)
        self.assertEqual(created.data["total_quantity"], 2)
        self.assertEqual(created.data["item"]["quantity"], 2)

        merged = self.client.post(
            "/api/me/cart",
            {"product_id": self.product_1.id, "quantity": 3},
            format="json",
        )
        self.assertEqual(merged.status_code, 200)
        self.assertFalse(merged.data["created"])
        self.assertEqual(merged.data["count"], 1)
        self.assertEqual(merged.data["total_quantity"], 5)
        self.assertEqual(merged.data["item"]["quantity"], 5)

        second = self.client.post("/api/me/cart", {"product_id": self.product_2.id}, format="json")
        self.assertEqual(second.status_code, 201)
        self.assertEqual(second.data["count"], 2)
        self.assertEqual(second.data["total_quantity"], 6)

        listed = self.client.get("/api/me/cart")
        self.assertEqual(listed.status_code, 200)
        self.assertEqual(listed.data["count"], 2)
        self.assertEqual(listed.data["total_quantity"], 6)

        patched = self.client.patch(f"/api/me/cart/{self.product_1.id}", {"quantity": 1}, format="json")
        self.assertEqual(patched.status_code, 200)
        self.assertEqual(patched.data["item"]["quantity"], 1)
        self.assertEqual(patched.data["total_quantity"], 2)

        deleted_by_patch = self.client.patch(
            f"/api/me/cart/{self.product_1.id}",
            {"quantity": 0},
            format="json",
        )
        self.assertEqual(deleted_by_patch.status_code, 200)
        self.assertEqual(deleted_by_patch.data["deleted"], 1)
        self.assertEqual(deleted_by_patch.data["count"], 1)
        self.assertEqual(deleted_by_patch.data["total_quantity"], 1)

        deleted = self.client.delete(f"/api/me/cart/{self.product_2.id}")
        self.assertEqual(deleted.status_code, 200)
        self.assertEqual(deleted.data["deleted"], 1)
        self.assertEqual(deleted.data["count"], 0)
        self.assertEqual(deleted.data["total_quantity"], 0)
