import yaml
from copy import deepcopy
from contextlib import contextmanager

from django.conf import settings
from django.contrib.auth import get_user_model
from django.core import mail
from django.core.cache import cache
from django.test import override_settings
from rest_framework.settings import api_settings
from rest_framework.throttling import SimpleRateThrottle
from rest_framework.test import APIClient, APITestCase

from backend.auth_email import (
    build_email_verification_token,
    build_password_reset_token,
    build_password_reset_uid,
    send_email_verification_message,
)
from loyalty.models import LoyaltyAccount
from users_app.models import CustomerProfile


@contextmanager
def override_rest_framework_throttle_rates(**rates):
    rest_framework = deepcopy(settings.REST_FRAMEWORK)
    rest_framework["DEFAULT_THROTTLE_RATES"] = {
        **rest_framework.get("DEFAULT_THROTTLE_RATES", {}),
        **rates,
    }
    with override_settings(REST_FRAMEWORK=rest_framework):
        api_settings.reload()
        SimpleRateThrottle.THROTTLE_RATES = api_settings.DEFAULT_THROTTLE_RATES
        try:
            yield
        finally:
            api_settings.reload()
            SimpleRateThrottle.THROTTLE_RATES = api_settings.DEFAULT_THROTTLE_RATES


@override_settings(
    EMAIL_BACKEND="django.core.mail.backends.locmem.EmailBackend",
    FRONTEND_BASE_URL="http://localhost:5173",
    EMAIL_VERIFICATION_RESEND_COOLDOWN_SECONDS=60,
)
class AuthApiFlowTests(APITestCase):
    def setUp(self):
        User = get_user_model()
        self.user = User.objects.create_user(username="auth_u1", email="auth_u1@example.com", password="pass12345")
        self.client = APIClient(enforce_csrf_checks=True)
        cache.clear()

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
            {"email": "auth_u1@example.com", "password": "pass12345"},
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
            {"email": "auth_u1@example.com", "password": "bad-password"},
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
            {"email": "AUTH_U1@example.com", "password": "pass12345"},
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

    def test_login_with_username_is_not_supported(self):
        csrf = self._fetch_csrf()
        resp = self.client.post(
            "/api/auth/login",
            {"email": "auth_u1", "password": "pass12345"},
            format="json",
            HTTP_X_CSRFTOKEN=csrf,
        )
        self.assertEqual(resp.status_code, 400)
        self.assertFalse(resp.data["ok"])
        self.assertEqual(resp.data["code"], "validation_error")

    def test_login_rate_limit_returns_429_with_retry_after(self):
        cache.clear()
        csrf = self._fetch_csrf()

        with override_rest_framework_throttle_rates(auth_login="2/min"):
            for _ in range(2):
                resp = self.client.post(
                    "/api/auth/login",
                    {"email": "auth_u1@example.com", "password": "bad-password"},
                    format="json",
                    HTTP_X_CSRFTOKEN=csrf,
                )
                self.assertEqual(resp.status_code, 400)

            throttled = self.client.post(
                "/api/auth/login",
                {"email": "auth_u1@example.com", "password": "bad-password"},
                format="json",
                HTTP_X_CSRFTOKEN=csrf,
            )

        self.assertEqual(throttled.status_code, 429)
        self.assertFalse(throttled.data["ok"])
        self.assertEqual(throttled.data["code"], "rate_limited")
        self.assertIn("retry_after_seconds", throttled.data["details"])
        self.assertGreater(throttled.data["details"]["retry_after_seconds"], 0)

    def test_register_creates_session_profile_and_loyalty(self):
        csrf = self._fetch_csrf()
        register_resp = self.client.post(
            "/api/auth/register",
            {
                "email": "fresh_user@example.com",
                "password": "pass12345!",
                "password_confirm": "pass12345!",
            },
            format="json",
            HTTP_X_CSRFTOKEN=csrf,
        )
        self.assertEqual(register_resp.status_code, 200)
        self.assertTrue(register_resp.data["ok"])
        self.assertEqual(register_resp.data["user"]["username"], "fresh_user")
        self.assertEqual(register_resp.data["user"]["email"], "fresh_user@example.com")
        self.assertFalse(register_resp.data["user"]["email_verified"])
        self.assertEqual(register_resp.data["verification_email"], "fresh_user@example.com")
        self.assertTrue(register_resp.data["verification_email_sent"])
        self.assertEqual(register_resp.data["resend_available_in_seconds"], 60)
        self.assertIn("sessionid", self.client.cookies)
        self.assertEqual(len(mail.outbox), 1)
        self.assertEqual(mail.outbox[0].to, ["fresh_user@example.com"])
        self.assertIn("/verify-email?token=", mail.outbox[0].body)

        me_resp = self.client.get("/api/auth/me")
        self.assertEqual(me_resp.status_code, 200)
        self.assertEqual(me_resp.data["user"]["username"], "fresh_user")
        self.assertFalse(me_resp.data["user"]["email_verified"])

        User = get_user_model()
        created_user = User.objects.get(username="fresh_user")
        self.assertEqual(created_user.email, "fresh_user@example.com")
        self.assertTrue(CustomerProfile.objects.filter(user=created_user).exists())
        self.assertTrue(LoyaltyAccount.objects.filter(user=created_user).exists())
        self.assertIsNotNone(CustomerProfile.objects.get(user=created_user).email_verification_sent_at)

    def test_register_auto_generates_unique_username_when_local_part_is_taken(self):
        csrf = self._fetch_csrf()
        get_user_model().objects.create_user(
            username="same",
            email="same@old-example.com",
            password="pass12345!",
        )

        register_resp = self.client.post(
            "/api/auth/register",
            {
                "email": "same@example.com",
                "password": "pass12345!",
                "password_confirm": "pass12345!",
            },
            format="json",
            HTTP_X_CSRFTOKEN=csrf,
        )
        self.assertEqual(register_resp.status_code, 200)
        self.assertTrue(register_resp.data["ok"])
        self.assertEqual(register_resp.data["user"]["email"], "same@example.com")
        self.assertEqual(register_resp.data["user"]["username"], "same-2")

    def test_register_rate_limit_returns_429_with_retry_after(self):
        cache.clear()

        with override_rest_framework_throttle_rates(auth_register="1/min"):
            first_client = APIClient(enforce_csrf_checks=True)
            first_csrf = first_client.get("/api/auth/csrf").data["csrfToken"]
            first = first_client.post(
                "/api/auth/register",
                {
                    "email": "limited-one@example.com",
                    "password": "pass12345!",
                    "password_confirm": "pass12345!",
                },
                format="json",
                HTTP_X_CSRFTOKEN=first_csrf,
            )
            self.assertEqual(first.status_code, 200)

            second_client = APIClient(enforce_csrf_checks=True)
            second_csrf = second_client.get("/api/auth/csrf").data["csrfToken"]
            throttled = second_client.post(
                "/api/auth/register",
                {
                    "email": "limited-two@example.com",
                    "password": "pass12345!",
                    "password_confirm": "pass12345!",
                },
                format="json",
                HTTP_X_CSRFTOKEN=second_csrf,
            )

        self.assertEqual(throttled.status_code, 429)
        self.assertFalse(throttled.data["ok"])
        self.assertEqual(throttled.data["code"], "rate_limited")
        self.assertIn("retry_after_seconds", throttled.data["details"])

    def test_register_duplicate_username_returns_validation_error(self):
        csrf = self._fetch_csrf()
        resp = self.client.post(
            "/api/auth/register",
            {
                "username": "auth_u1",
                "email": "other@example.com",
                "password": "pass12345!",
                "password_confirm": "pass12345!",
            },
            format="json",
            HTTP_X_CSRFTOKEN=csrf,
        )
        self.assertEqual(resp.status_code, 400)
        self.assertFalse(resp.data["ok"])
        self.assertEqual(resp.data["code"], "validation_error")
        self.assertIn("username", resp.data["details"])

    def test_register_duplicate_email_returns_validation_error(self):
        csrf = self._fetch_csrf()
        resp = self.client.post(
            "/api/auth/register",
            {
                "username": "auth_u2",
                "email": "AUTH_U1@example.com",
                "password": "pass12345!",
                "password_confirm": "pass12345!",
            },
            format="json",
            HTTP_X_CSRFTOKEN=csrf,
        )
        self.assertEqual(resp.status_code, 400)
        self.assertFalse(resp.data["ok"])
        self.assertEqual(resp.data["code"], "validation_error")
        self.assertIn("email", resp.data["details"])

    def test_verify_email_marks_profile_and_auth_me(self):
        self.client.force_login(self.user)
        token = build_email_verification_token(self.user)

        verify_resp = self.client.post(
            "/api/auth/verify-email",
            {"token": token},
            format="json",
        )
        self.assertEqual(verify_resp.status_code, 200)
        self.assertTrue(verify_resp.data["ok"])
        self.assertTrue(verify_resp.data["email_verified"])
        self.assertFalse(verify_resp.data["already_verified"])
        self.assertEqual(verify_resp.data["email"], "auth_u1@example.com")

        profile = CustomerProfile.objects.get(user=self.user)
        self.assertIsNotNone(profile.email_verified_at)

        me_resp = self.client.get("/api/auth/me")
        self.assertEqual(me_resp.status_code, 200)
        self.assertTrue(me_resp.data["user"]["email_verified"])

    def test_verification_status_returns_email_verified_and_cooldown(self):
        self.client.force_login(self.user)
        send_email_verification_message(self.user)

        status_resp = self.client.get("/api/auth/verification-status")
        self.assertEqual(status_resp.status_code, 200)
        self.assertTrue(status_resp.data["ok"])
        self.assertEqual(status_resp.data["email"], "auth_u1@example.com")
        self.assertFalse(status_resp.data["email_verified"])
        self.assertGreaterEqual(status_resp.data["resend_available_in_seconds"], 0)
        self.assertLessEqual(status_resp.data["resend_available_in_seconds"], 60)

    def test_resend_verification_sends_email_for_authenticated_user(self):
        self.client.force_login(self.user)
        csrf = self._fetch_csrf()

        resend_resp = self.client.post(
            "/api/auth/verify-email/resend",
            {},
            format="json",
            HTTP_X_CSRFTOKEN=csrf,
        )
        self.assertEqual(resend_resp.status_code, 200)
        self.assertTrue(resend_resp.data["ok"])
        self.assertTrue(resend_resp.data["sent"])
        self.assertFalse(resend_resp.data["already_verified"])
        self.assertEqual(resend_resp.data["email"], "auth_u1@example.com")
        self.assertEqual(resend_resp.data["resend_available_in_seconds"], 60)
        self.assertEqual(len(mail.outbox), 1)
        self.assertEqual(mail.outbox[0].to, ["auth_u1@example.com"])
        self.assertIn("/verify-email?token=", mail.outbox[0].body)

    def test_resend_verification_respects_cooldown(self):
        self.client.force_login(self.user)
        send_email_verification_message(self.user)
        csrf = self._fetch_csrf()

        resend_resp = self.client.post(
            "/api/auth/verify-email/resend",
            {},
            format="json",
            HTTP_X_CSRFTOKEN=csrf,
        )
        self.assertEqual(resend_resp.status_code, 400)
        self.assertFalse(resend_resp.data["ok"])
        self.assertIn("resend_available_in_seconds", resend_resp.data["details"])

    def test_unverified_user_cannot_access_profile_api(self):
        self.client.force_login(self.user)

        resp = self.client.get("/api/me/profile")
        payload = resp.json()
        self.assertEqual(resp.status_code, 403)
        self.assertFalse(payload["ok"])
        self.assertEqual(payload["code"], "email_not_verified")
        self.assertEqual(payload["details"]["email"], "auth_u1@example.com")
        self.assertIn("resend_available_in_seconds", payload["details"])

    def test_unverified_user_cannot_access_catalog_api(self):
        self.client.force_login(self.user)

        resp = self.client.get("/api/products/")
        payload = resp.json()
        self.assertEqual(resp.status_code, 403)
        self.assertFalse(payload["ok"])
        self.assertEqual(payload["code"], "email_not_verified")

    def test_verified_user_can_access_profile_api(self):
        self.client.force_login(self.user)
        token = build_email_verification_token(self.user)
        verify_resp = self.client.post(
            "/api/auth/verify-email",
            {"token": token},
            format="json",
        )
        self.assertEqual(verify_resp.status_code, 200)

        resp = self.client.get("/api/me/profile")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.data["first_name"], "")
        self.assertEqual(resp.data["last_name"], "")
        self.assertEqual(resp.data["phone"], "")
        self.assertEqual(resp.data["city"], "")

    def test_verified_user_can_update_optional_profile_fields(self):
        self.client.force_login(self.user)
        token = build_email_verification_token(self.user)
        verify_resp = self.client.post(
            "/api/auth/verify-email",
            {"token": token},
            format="json",
        )
        self.assertEqual(verify_resp.status_code, 200)

        csrf = self._fetch_csrf()
        resp = self.client.put(
            "/api/me/profile",
            {
                "first_name": "Adik",
                "last_name": "User",
                "phone": "+7 777 123 45 67",
                "city": "Almaty",
            },
            format="json",
            HTTP_X_CSRFTOKEN=csrf,
        )
        self.assertEqual(resp.status_code, 200)
        self.assertTrue(resp.data["ok"])
        self.assertEqual(resp.data["profile"]["first_name"], "Adik")
        self.assertEqual(resp.data["profile"]["last_name"], "User")
        self.assertEqual(resp.data["profile"]["phone"], "+7 777 123 45 67")
        self.assertEqual(resp.data["profile"]["city"], "Almaty")

    def test_password_reset_request_sends_email_for_existing_user(self):
        csrf = self._fetch_csrf()
        resp = self.client.post(
            "/api/auth/password-reset/request",
            {"email": "auth_u1@example.com"},
            format="json",
            HTTP_X_CSRFTOKEN=csrf,
        )
        self.assertEqual(resp.status_code, 200)
        self.assertTrue(resp.data["ok"])
        self.assertTrue(resp.data["sent"])
        self.assertEqual(resp.data["email"], "auth_u1@example.com")
        self.assertEqual(len(mail.outbox), 1)
        self.assertEqual(mail.outbox[0].to, ["auth_u1@example.com"])
        self.assertIn("/reset-password?uid=", mail.outbox[0].body)
        self.assertIn("&token=", mail.outbox[0].body)

    def test_password_reset_request_is_generic_for_unknown_email(self):
        csrf = self._fetch_csrf()
        resp = self.client.post(
            "/api/auth/password-reset/request",
            {"email": "unknown@example.com"},
            format="json",
            HTTP_X_CSRFTOKEN=csrf,
        )
        self.assertEqual(resp.status_code, 200)
        self.assertTrue(resp.data["ok"])
        self.assertTrue(resp.data["sent"])
        self.assertEqual(resp.data["email"], "unknown@example.com")
        self.assertEqual(len(mail.outbox), 0)

    def test_password_reset_request_rate_limit_returns_429_with_retry_after(self):
        cache.clear()
        csrf = self._fetch_csrf()

        with override_rest_framework_throttle_rates(auth_password_reset_request="1/min"):
            first = self.client.post(
                "/api/auth/password-reset/request",
                {"email": "auth_u1@example.com"},
                format="json",
                HTTP_X_CSRFTOKEN=csrf,
            )
            self.assertEqual(first.status_code, 200)

            throttled = self.client.post(
                "/api/auth/password-reset/request",
                {"email": "auth_u1@example.com"},
                format="json",
                HTTP_X_CSRFTOKEN=csrf,
            )

        self.assertEqual(throttled.status_code, 429)
        self.assertFalse(throttled.data["ok"])
        self.assertEqual(throttled.data["code"], "rate_limited")
        self.assertIn("retry_after_seconds", throttled.data["details"])

    def test_password_reset_validate_returns_valid_for_good_link(self):
        resp = self.client.post(
            "/api/auth/password-reset/validate",
            {
                "uid": build_password_reset_uid(self.user),
                "token": build_password_reset_token(self.user),
            },
            format="json",
        )
        self.assertEqual(resp.status_code, 200)
        self.assertTrue(resp.data["ok"])
        self.assertTrue(resp.data["valid"])

    def test_password_reset_confirm_updates_password(self):
        uid = build_password_reset_uid(self.user)
        token = build_password_reset_token(self.user)
        resp = self.client.post(
            "/api/auth/password-reset/confirm",
            {
                "uid": uid,
                "token": token,
                "password": "new-pass-12345!",
                "password_confirm": "new-pass-12345!",
            },
            format="json",
        )
        self.assertEqual(resp.status_code, 200)
        self.assertTrue(resp.data["ok"])
        self.user.refresh_from_db()
        self.assertTrue(self.user.check_password("new-pass-12345!"))

    def test_resend_verification_rate_limit_returns_429_with_retry_after(self):
        cache.clear()
        self.client.force_login(self.user)
        csrf = self._fetch_csrf()

        with override_rest_framework_throttle_rates(auth_verify_email_resend="1/min"), override_settings(
            EMAIL_VERIFICATION_RESEND_COOLDOWN_SECONDS=0
        ):
            first = self.client.post(
                "/api/auth/verify-email/resend",
                {},
                format="json",
                HTTP_X_CSRFTOKEN=csrf,
            )
            self.assertEqual(first.status_code, 200)

            throttled = self.client.post(
                "/api/auth/verify-email/resend",
                {},
                format="json",
                HTTP_X_CSRFTOKEN=csrf,
            )

        self.assertEqual(throttled.status_code, 429)
        self.assertFalse(throttled.data["ok"])
        self.assertEqual(throttled.data["code"], "rate_limited")
        self.assertIn("retry_after_seconds", throttled.data["details"])


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
        self.assertIn("/api/auth/password-reset/request", paths)
        self.assertIn("/api/auth/password-reset/validate", paths)
        self.assertIn("/api/auth/password-reset/confirm", paths)
        self.assertIn("/api/auth/register", paths)
        self.assertIn("/api/auth/verify-email", paths)
        self.assertIn("/api/auth/verify-email/resend", paths)
        self.assertIn("/api/auth/verification-status", paths)
        self.assertIn("/api/auth/logout", paths)
        self.assertIn("/api/auth/me", paths)

        owned_products_ops = paths.get("/api/me/owned-products/", {})
        self.assertNotIn("post", {k.lower(): v for k, v in owned_products_ops.items()})
