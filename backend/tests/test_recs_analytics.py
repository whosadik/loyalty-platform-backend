from decimal import Decimal
from django.contrib.auth import get_user_model
from rest_framework.test import APITestCase

from catalog.models import Product
from users_app.models import CustomerProfile
from recs_analytics.models import RecommendationEvent


class RecsAnalyticsTests(APITestCase):
    def setUp(self):
        User = get_user_model()
        self.user = User.objects.create_user(username="u3", password="pass12345")
        self.client.force_authenticate(self.user)
        CustomerProfile.objects.get_or_create(user=self.user)

        self.p = Product.objects.create(
            name="Lipstick", brand="B", price=Decimal("9.99"),
            category="makeup", product_type="lipstick",
            concerns=[], attrs={"finish":"matte"}, actives=[], flags=[],
            supported_skin_types=[], strength="low", in_stock=True,
        )

    def test_impressions_written_on_home(self):
        r = self.client.get("/api/me/recommendations/home?limit=5")
        self.assertEqual(r.status_code, 200)
        # не гарантируем, что будет lipstick, но impressions должны писаться на те, что вернулись
        self.assertTrue(RecommendationEvent.objects.filter(user=self.user, action="impression").exists())
