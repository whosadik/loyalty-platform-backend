import yaml
from django.contrib.auth import get_user_model
from rest_framework.test import APIClient, APITestCase


class AuthApiFlowTests(APITestCase):
    def setUp(self):
        User = get_user_model()
        self.user = User.objects.create_user(username="auth_u1", password="pass12345")
        self.client = APIClient(enforce_csrf_checks=True)

    def _fetch_csrf(self) -> str:
        resp = self.client.get("/api/auth/csrf")
        self.assertEqual(resp.status_code, 200)
        self.assertTrue(resp.data["ok"])
        self.assertIn("csrftoken", self.client.cookies)
        token = resp.data.get("csrfToken")
        self.assertTrue(token)
        return token

    def test_csrf_endpoint_sets_cookie(self):
        self._fetch_csrf()

    def test_login_without_csrf_fails_with_error_envelope(self):
        resp = self.client.post(
            "/api/auth/login",
            {"username": "auth_u1", "password": "pass12345"},
            format="json",
        )
        self.assertIn(resp.status_code, {400, 403})
        self.assertFalse(resp.data["ok"])
        self.assertIn("code", resp.data)
        self.assertIn("message", resp.data)
        self.assertIn("request_id", resp.data)

    def test_login_invalid_credentials_returns_400(self):
        csrf = self._fetch_csrf()
        resp = self.client.post(
            "/api/auth/login",
            {"username": "auth_u1", "password": "bad-password"},
            format="json",
            HTTP_X_CSRFTOKEN=csrf,
        )
        self.assertEqual(resp.status_code, 400)
        self.assertFalse(resp.data["ok"])
        self.assertEqual(resp.data["code"], "invalid_credentials")
        self.assertIn("request_id", resp.data)

    def test_login_me_logout_flow(self):
        csrf = self._fetch_csrf()
        login_resp = self.client.post(
            "/api/auth/login",
            {"username": "auth_u1", "password": "pass12345"},
            format="json",
            HTTP_X_CSRFTOKEN=csrf,
        )
        self.assertEqual(login_resp.status_code, 200)
        self.assertTrue(login_resp.data["ok"])
        self.assertEqual(login_resp.data["user"]["username"], "auth_u1")
        self.assertIn("sessionid", self.client.cookies)

        me_resp = self.client.get("/api/auth/me")
        self.assertEqual(me_resp.status_code, 200)
        self.assertTrue(me_resp.data["ok"])
        self.assertEqual(me_resp.data["user"]["username"], "auth_u1")

        csrf_after_login = self.client.cookies["csrftoken"].value
        logout_resp = self.client.post("/api/auth/logout", {}, format="json", HTTP_X_CSRFTOKEN=csrf_after_login)
        self.assertEqual(logout_resp.status_code, 200)
        self.assertTrue(logout_resp.data["ok"])

        me_after_logout = self.client.get("/api/auth/me")
        self.assertIn(me_after_logout.status_code, {401, 403})


class CatalogWritePermissionTests(APITestCase):
    def setUp(self):
        User = get_user_model()
        self.regular_user = User.objects.create_user(username="catalog_regular", password="pass12345")
        self.staff_user = User.objects.create_user(
            username="catalog_staff",
            password="pass12345",
            is_staff=True,
        )
        self.payload = {
            "name": "Permission Test Product",
            "product_type": "serum",
            "category": "skincare",
        }

    def test_regular_user_cannot_create_product(self):
        client = APIClient()
        client.force_authenticate(self.regular_user)
        resp = client.post("/api/products/", self.payload, format="json")
        self.assertEqual(resp.status_code, 403)

    def test_staff_user_can_create_product(self):
        client = APIClient()
        client.force_authenticate(self.staff_user)
        resp = client.post("/api/products/", self.payload, format="json")
        self.assertEqual(resp.status_code, 201)


class SchemaAndOwnedProductsContractTests(APITestCase):
    def setUp(self):
        User = get_user_model()
        self.user = User.objects.create_user(username="schema_u1", password="pass12345")
        self.client.force_authenticate(self.user)

    def test_owned_products_post_returns_405(self):
        resp = self.client.post("/api/me/owned-products/", {}, format="json")
        self.assertEqual(resp.status_code, 405)

    def test_schema_contains_auth_paths_and_no_owned_products_post(self):
        resp = self.client.get("/api/schema/?format=json")
        self.assertEqual(resp.status_code, 200)

        if isinstance(resp.data, dict):
            schema = resp.data
        else:
            schema = yaml.safe_load(resp.content.decode("utf-8"))

        paths = schema.get("paths", {})
        self.assertIn("/api/auth/csrf", paths)
        self.assertIn("/api/auth/login", paths)
        self.assertIn("/api/auth/logout", paths)
        self.assertIn("/api/auth/me", paths)

        owned_products_ops = paths.get("/api/me/owned-products/", {})
        self.assertNotIn("post", {k.lower(): v for k, v in owned_products_ops.items()})
