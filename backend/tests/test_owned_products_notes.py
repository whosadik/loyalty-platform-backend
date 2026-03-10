from django.contrib.auth import get_user_model
from rest_framework.test import APITestCase

from catalog.models import Product
from transactions.models import OwnedProduct


class OwnedProductsNotesContractTests(APITestCase):
    def setUp(self):
        User = get_user_model()
        self.user = User.objects.create_user(username="owned_u1", password="pass12345")
        self.client.force_authenticate(self.user)

        product = Product.objects.create(
            name="Owned Product",
            product_type="serum",
        )
        self.owned = OwnedProduct.objects.create(
            user=self.user,
            product=product,
            quantity_total=1,
            is_active=True,
        )

    def test_patch_owned_product_updates_notes_and_dates(self):
        payload = {
            "notes": "Use in PM only",
            "opened_at": "2026-03-01",
            "finish_date": "2026-04-01",
        }
        resp = self.client.patch(f"/api/me/owned-products/{self.owned.id}/", payload, format="json")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.data["notes"], "Use in PM only")
        self.assertEqual(resp.data["opened_at"], "2026-03-01")
        self.assertEqual(resp.data["finish_date"], "2026-04-01")

    def test_activate_deactivate_returns_owned_product_payload(self):
        deactivate = self.client.post(f"/api/me/owned-products/{self.owned.id}/deactivate/", {}, format="json")
        self.assertEqual(deactivate.status_code, 200)
        self.assertTrue(deactivate.data["ok"])
        self.assertIn("owned_product", deactivate.data)
        self.assertEqual(deactivate.data["id"], self.owned.id)
        self.assertFalse(deactivate.data["is_active"])
        self.assertFalse(deactivate.data["owned_product"]["is_active"])

        activate = self.client.post(f"/api/me/owned-products/{self.owned.id}/activate/", {}, format="json")
        self.assertEqual(activate.status_code, 200)
        self.assertTrue(activate.data["ok"])
        self.assertIn("owned_product", activate.data)
        self.assertEqual(activate.data["id"], self.owned.id)
        self.assertTrue(activate.data["is_active"])
        self.assertTrue(activate.data["owned_product"]["is_active"])
