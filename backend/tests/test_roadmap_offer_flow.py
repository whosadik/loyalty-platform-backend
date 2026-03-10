from decimal import Decimal

from django.contrib.auth import get_user_model
from django.db.models import Count
from django.utils import timezone
from rest_framework.test import APITestCase

from catalog.models import Product
from loyalty.models import LoyaltyAccount, Tier
from offers.models import CampaignBudget, Offer, OfferAssignment, OfferEvent
from offers.services import _pick_target_for_offer
from roadmap_app.fragrance_slots import slot_of_fragrance
from roadmap_app.models import RoadmapPlan, RoadmapStep
from transactions.models import OwnedProduct
from users_app.models import CustomerProfile


class RoadmapOfferFlowTests(APITestCase):
    def setUp(self):
        User = get_user_model()
        self.user = User.objects.create_user(username="roadmap_u1", password="pass12345")
        self.client.force_authenticate(self.user)

        CustomerProfile.objects.get_or_create(user=self.user)
        bronze, _ = Tier.objects.get_or_create(
            name="Bronze",
            defaults={"threshold_spend_90d": 0, "points_rate": 1.0},
        )
        LoyaltyAccount.objects.get_or_create(
            user=self.user,
            defaults={"tier": bronze, "points_balance": 0},
        )

        self.p_shampoo = Product.objects.create(
            name="Shampoo A",
            brand="B",
            price=Decimal("8.00"),
            category="haircare",
            product_type="shampoo",
            concerns=[],
            attrs={},
            actives=[],
            flags=[],
            supported_skin_types=["normal"],
            strength="low",
            in_stock=True,
        )
        self.p_conditioner = Product.objects.create(
            name="Conditioner A",
            brand="B",
            price=Decimal("9.00"),
            category="haircare",
            product_type="conditioner",
            concerns=[],
            attrs={},
            actives=[],
            flags=[],
            supported_skin_types=["normal"],
            strength="low",
            in_stock=True,
        )
        Product.objects.create(
            name="Hair Mask A",
            brand="B",
            price=Decimal("11.00"),
            category="haircare",
            product_type="hair_mask",
            concerns=[],
            attrs={},
            actives=[],
            flags=[],
            supported_skin_types=["normal"],
            strength="low",
            in_stock=True,
        )
        Product.objects.create(
            name="Hair Oil A",
            brand="B",
            price=Decimal("10.00"),
            category="haircare",
            product_type="hair_oil",
            concerns=[],
            attrs={},
            actives=[],
            flags=[],
            supported_skin_types=["normal"],
            strength="low",
            in_stock=True,
        )

        self.default_campaign, _ = CampaignBudget.objects.get_or_create(
            name="default",
            defaults={
                "weekly_limit": Decimal("1000.00"),
                "weekly_spent": Decimal("0.00"),
                "priority": 100,
                "is_active": True,
            },
        )
        Offer.objects.create(
            name="Roadmap Conditioner Offer",
            offer_type=Offer.Type.DISCOUNT,
            value=Decimal("10.00"),
            estimated_cost=Decimal("4.00"),
            is_active=True,
            target_scope="product_type",
            cooldown_days=0,
            expires_in_days=7,
            allowed_categories=["haircare"],
            allowed_product_types=["conditioner", "hair_mask", "hair_oil"],
            campaign=self.default_campaign,
        )

    def _checkout_shampoo(self):
        payload = {
            "channel": "offline",
            "items": [{"product": self.p_shampoo.id, "quantity": 1}],
        }
        return self.client.post("/api/checkout", payload, format="json")

    def test_haircare_roadmap_after_shampoo_purchase_and_next_offer_target(self):
        checkout = self._checkout_shampoo()
        self.assertEqual(checkout.status_code, 201)

        roadmap = self.client.get("/api/me/roadmap?category=haircare")
        self.assertEqual(roadmap.status_code, 200)

        steps = roadmap.data.get("steps") or []
        self.assertGreaterEqual(len(steps), 4)

        shampoo_steps = [x for x in steps if x.get("product_type") == "shampoo"]
        self.assertTrue(shampoo_steps)
        self.assertIn(shampoo_steps[0].get("status"), {"owned", "completed"})

        summary = roadmap.data.get("summary") or {}
        next_step = summary.get("next_step") or {}
        self.assertEqual(next_step.get("product_type"), "conditioner")

        next_offer = self.client.get("/api/me/next-offer")
        self.assertEqual(next_offer.status_code, 200)
        reason = (next_offer.data.get("reason") or {}).get("roadmap") or {}
        target = next_offer.data.get("target") or {}

        target_product_type = target.get("product_type")
        if target.get("scope") == "product_type":
            target_product_type = target.get("value")

        self.assertTrue(
            reason.get("next_product_type") == "conditioner" or target_product_type == "conditioner"
        )

    def test_roadmap_refresh_is_idempotent_and_no_duplicate_step_indexes(self):
        checkout = self._checkout_shampoo()
        self.assertEqual(checkout.status_code, 201)

        body = {"category": "haircare"}
        first = self.client.post("/api/me/roadmap/refresh", body, format="json")
        second = self.client.post("/api/me/roadmap/refresh", body, format="json")
        self.assertEqual(first.status_code, 200)
        self.assertEqual(second.status_code, 200)
        self.assertEqual(first.data.get("id"), second.data.get("id"))

        plan_id = int(first.data["id"])
        rows = (
            RoadmapStep.objects.filter(plan_id=plan_id)
            .values("step_index")
            .annotate(c=Count("id"))
            .order_by("step_index")
        )
        self.assertTrue(rows)
        self.assertEqual(sum(int(r["c"]) for r in rows), len(rows))
        self.assertEqual(
            RoadmapPlan.objects.filter(user=self.user, category="haircare", is_active=True).count(),
            1,
        )

    def test_recommended_products_are_unique_and_in_stock(self):
        checkout = self._checkout_shampoo()
        self.assertEqual(checkout.status_code, 201)

        roadmap = self.client.get("/api/me/roadmap?category=haircare")
        self.assertEqual(roadmap.status_code, 200)
        steps = roadmap.data.get("steps") or []

        rec_ids = []
        for step in steps:
            rp = step.get("recommended_product")
            if not rp:
                continue
            self.assertEqual(step.get("recommended_product_id"), int(rp["id"]))
            rec_ids.append(int(rp["id"]))
            self.assertTrue(bool(rp.get("in_stock")))
        self.assertEqual(len(rec_ids), len(set(rec_ids)))

    def test_not_owned_products_are_not_recommended_or_suggested(self):
        OwnedProduct.objects.create(user=self.user, product=self.p_conditioner, quantity_total=1, is_active=True)

        refresh = self.client.post("/api/me/roadmap/refresh", {"category": "haircare"}, format="json")
        self.assertEqual(refresh.status_code, 200)
        steps = refresh.data.get("steps") or []

        for step in steps:
            rp = step.get("recommended_product")
            if rp:
                self.assertNotEqual(int(rp["id"]), self.p_conditioner.id)
            suggestions = [int(x) for x in (step.get("suggestions") or [])]
            self.assertNotIn(self.p_conditioner.id, suggestions)

    def test_instock_fallback_does_not_break_when_slot_empty(self):
        Product.objects.filter(category="haircare").exclude(id=self.p_shampoo.id).update(in_stock=False)

        checkout = self._checkout_shampoo()
        self.assertEqual(checkout.status_code, 201)
        roadmap = self.client.get("/api/me/roadmap?category=haircare")
        self.assertEqual(roadmap.status_code, 200)

        for step in roadmap.data.get("steps") or []:
            rp = step.get("recommended_product")
            if rp:
                self.assertTrue(bool(rp.get("in_stock")))


class RoadmapSupersedeTests(APITestCase):
    def setUp(self):
        User = get_user_model()
        self.user = User.objects.create_user(username="roadmap_sup_u1", password="pass12345")
        self.client.force_authenticate(self.user)

        CustomerProfile.objects.get_or_create(user=self.user)
        bronze, _ = Tier.objects.get_or_create(
            name="Bronze",
            defaults={"threshold_spend_90d": 0, "points_rate": 1.0},
        )
        LoyaltyAccount.objects.get_or_create(
            user=self.user,
            defaults={"tier": bronze, "points_balance": 0},
        )

        self.p_cleanser = Product.objects.create(
            name="Cleanser Seed",
            brand="B",
            price=Decimal("7.00"),
            category="skincare",
            product_type="cleanser",
            concerns=[],
            attrs={},
            actives=[],
            flags=[],
            supported_skin_types=["normal"],
            strength="low",
            in_stock=True,
        )
        self.p_shampoo_883 = Product.objects.create(
            id=883,
            name="Shampoo 883",
            brand="B",
            price=Decimal("8.00"),
            category="haircare",
            product_type="shampoo",
            concerns=[],
            attrs={},
            actives=[],
            flags=[],
            supported_skin_types=["normal"],
            strength="low",
            in_stock=True,
        )
        self.p_conditioner = Product.objects.create(
            name="Conditioner Sup",
            brand="B",
            price=Decimal("9.00"),
            category="haircare",
            product_type="conditioner",
            concerns=[],
            attrs={},
            actives=[],
            flags=[],
            supported_skin_types=["normal"],
            strength="low",
            in_stock=True,
        )
        Product.objects.create(
            name="Hair Mask Sup",
            brand="B",
            price=Decimal("10.00"),
            category="haircare",
            product_type="hair_mask",
            concerns=[],
            attrs={},
            actives=[],
            flags=[],
            supported_skin_types=["normal"],
            strength="low",
            in_stock=True,
        )
        Product.objects.create(
            name="Hair Oil Sup",
            brand="B",
            price=Decimal("11.00"),
            category="haircare",
            product_type="hair_oil",
            concerns=[],
            attrs={},
            actives=[],
            flags=[],
            supported_skin_types=["normal"],
            strength="low",
            in_stock=True,
        )

        self.default_campaign, _ = CampaignBudget.objects.get_or_create(
            name="default",
            defaults={
                "weekly_limit": Decimal("1000.00"),
                "weekly_spent": Decimal("0.00"),
                "priority": 100,
                "is_active": True,
            },
        )
        self.haircare_campaign = CampaignBudget.objects.create(
            name="haircare_roadmap_campaign",
            weekly_limit=Decimal("1000.00"),
            weekly_spent=Decimal("0.00"),
            priority=10,
            is_active=True,
            allowed_categories=["haircare"],
        )

        self.offer_a = Offer.objects.create(
            name="Seed Skincare Offer",
            offer_type=Offer.Type.DISCOUNT,
            value=Decimal("5.00"),
            estimated_cost=Decimal("1.00"),
            is_active=True,
            target_scope="product_type",
            cooldown_days=0,
            expires_in_days=7,
            allowed_categories=["skincare"],
            allowed_product_types=["moisturizer"],
            campaign=self.default_campaign,
        )
        self.offer_b = Offer.objects.create(
            name="Roadmap Conditioner Offer",
            offer_type=Offer.Type.DISCOUNT,
            value=Decimal("10.00"),
            estimated_cost=Decimal("3.00"),
            is_active=True,
            target_scope="product_type",
            cooldown_days=0,
            expires_in_days=7,
            allowed_categories=["haircare"],
            allowed_product_types=["conditioner"],
            campaign=self.haircare_campaign,
        )

    def _checkout(self, product_id: int):
        return self.client.post(
            "/api/checkout",
            {"channel": "offline", "items": [{"product": product_id, "quantity": 1}]},
            format="json",
        )

    def test_roadmap_soft_supersede_on_checkout(self):
        first_checkout = self._checkout(self.p_cleanser.id)
        self.assertEqual(first_checkout.status_code, 201)
        a_id = int((first_checkout.data.get("next_offer") or {}).get("assignment_id"))
        self.assertTrue(a_id)

        self.client.get("/api/me/next-offer")

        second_checkout = self._checkout(883)
        self.assertEqual(second_checkout.status_code, 201)
        next_offer = second_checkout.data.get("next_offer") or {}
        b_id = int(next_offer.get("assignment_id"))
        self.assertNotEqual(a_id, b_id)
        reason_roadmap = (next_offer.get("reason") or {}).get("roadmap") or {}
        target = next_offer.get("target") or {}
        target_pt = target.get("product_type")
        if target.get("scope") == "product_type":
            target_pt = target.get("value")

        self.assertTrue(
            reason_roadmap.get("next_product_type") == "conditioner" or target_pt == "conditioner"
        )

        a = OfferAssignment.objects.get(id=a_id)
        self.assertFalse(a.is_active)
        self.assertIsNotNone(a.superseded_at)
        self.assertEqual(a.superseded_by_id, b_id)
        self.assertTrue(
            OfferEvent.objects.filter(
                assignment_id=a_id,
                event_type=OfferEvent.Type.SUPERSEDED,
            ).exists()
        )

    def test_no_supersede_if_clicked(self):
        first_checkout = self._checkout(self.p_cleanser.id)
        self.assertEqual(first_checkout.status_code, 201)
        a_id = int((first_checkout.data.get("next_offer") or {}).get("assignment_id"))
        self.assertTrue(a_id)

        click = self.client.post("/api/offers/click", {"assignment_id": a_id}, format="json")
        self.assertEqual(click.status_code, 200)

        second_checkout = self._checkout(883)
        self.assertEqual(second_checkout.status_code, 201)
        next_offer = second_checkout.data.get("next_offer") or {}
        self.assertEqual(int(next_offer.get("assignment_id")), a_id)

        a = OfferAssignment.objects.get(id=a_id)
        self.assertTrue(a.is_active)
        self.assertIsNone(a.superseded_at)
        self.assertFalse(
            OfferEvent.objects.filter(
                assignment_id=a_id,
                event_type=OfferEvent.Type.SUPERSEDED,
            ).exists()
        )


class RoadmapFragranceTargetRegressionTests(APITestCase):
    def setUp(self):
        User = get_user_model()
        self.user = User.objects.create_user(username="roadmap_frag_slot_u1", password="pass12345")
        self.client.force_authenticate(self.user)

        CustomerProfile.objects.get_or_create(user=self.user)
        bronze, _ = Tier.objects.get_or_create(
            name="Bronze",
            defaults={"threshold_spend_90d": 0, "points_rate": 1.0},
        )
        LoyaltyAccount.objects.get_or_create(
            user=self.user,
            defaults={"tier": bronze, "points_balance": 0},
        )

        self.default_campaign, _ = CampaignBudget.objects.get_or_create(
            name="default",
            defaults={
                "weekly_limit": Decimal("1000.00"),
                "weekly_spent": Decimal("0.00"),
                "priority": 100,
                "is_active": True,
            },
        )
        self.offer = Offer.objects.create(
            name="Fragrance Slot Product Offer",
            offer_type=Offer.Type.DISCOUNT,
            value=Decimal("10.00"),
            estimated_cost=Decimal("2.00"),
            is_active=True,
            target_scope="product_id",
            cooldown_days=0,
            expires_in_days=7,
            allowed_categories=["fragrance"],
            allowed_product_types=["edp"],
            campaign=self.default_campaign,
        )

        self.p_warm_day = Product.objects.create(
            name="Citrus Day EDP",
            brand="F",
            price=Decimal("40.00"),
            category="fragrance",
            product_type="edp",
            concerns=[],
            attrs={"scent_family": "citrus", "notes": ["bergamot"], "intensity": "soft"},
            actives=[],
            flags=[],
            supported_skin_types=["normal"],
            strength="medium",
            in_stock=True,
        )
        self.p_warm_evening = Product.objects.create(
            name="Citrus Night EDP",
            brand="F",
            price=Decimal("45.00"),
            category="fragrance",
            product_type="edp",
            concerns=[],
            attrs={"scent_family": "citrus", "notes": ["neroli"], "intensity": "strong"},
            actives=[],
            flags=[],
            supported_skin_types=["normal"],
            strength="high",
            in_stock=True,
        )

    def test_fragrance_slot_roadmap_ctx_product_id_target_scope(self):
        target = _pick_target_for_offer(
            self.user,
            self.offer,
            timezone.now(),
            context_steps=None,
            post_ctx={
                "categories": ["fragrance"],
                "product_types": ["edp"],
                "product_ids": [self.p_warm_day.id],
            },
            campaign=self.default_campaign,
            roadmap_ctx={
                "category": "fragrance",
                "next_product_type": "warm_evening",
            },
        )

        self.assertIsInstance(target, dict)
        self.assertEqual(target.get("scope"), "product_id")
        self.assertEqual(target.get("category"), "fragrance")
        pid = int(target["value"])
        self.assertTrue(pid > 0)

        prod = Product.objects.get(id=pid)
        slot = slot_of_fragrance(prod.attrs or {}, raw_meta=prod.raw_meta or {})
        self.assertEqual(slot, "warm_evening")
        self.assertEqual(target.get("product_type"), "warm_evening")
        self.assertEqual(target.get("actual_product_type"), prod.product_type)
