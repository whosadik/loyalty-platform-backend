from decimal import Decimal

from django.contrib.auth import get_user_model
from rest_framework.test import APITestCase

from catalog.models import Product
from loyalty.models import LoyaltyAccount, Tier
from offers.models import CampaignBudget, Offer, OfferAssignment, OfferEvent
from roadmap_app.models import RoadmapEvent, RoadmapPlan, RoadmapStep
from users_app.models import CustomerProfile


class RoadmapTelemetryTests(APITestCase):
    def setUp(self):
        User = get_user_model()
        self.user = User.objects.create_user(username="roadmap_tele_u1", password="pass12345")
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
            name="Telemetry Shampoo",
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
            name="Telemetry Conditioner",
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
            name="Telemetry Hair Mask",
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
            name="Telemetry Hair Oil",
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

    def _checkout(self, product_id: int):
        return self.client.post(
            "/api/checkout",
            {"channel": "offline", "items": [{"product": product_id, "quantity": 1}]},
            format="json",
        )

    def test_roadmap_exposed_skipped_and_clicked_events(self):
        c = self._checkout(self.p_shampoo.id)
        self.assertEqual(c.status_code, 201)

        roadmap = self.client.get("/api/me/roadmap?category=haircare")
        self.assertEqual(roadmap.status_code, 200)
        next_step = (roadmap.data.get("summary") or {}).get("next_step") or {}
        step_id = int(next_step["id"])

        exposed = RoadmapEvent.objects.filter(
            user=self.user,
            step_id=step_id,
            event_type=RoadmapEvent.Type.STEP_EXPOSED,
        ).first()
        self.assertIsNotNone(exposed)
        self.assertIn("category", exposed.context)
        self.assertIn("step_index", exposed.context)
        self.assertIn("product_type", exposed.context)

        patch = self.client.patch(
            f"/api/me/roadmap/steps/{step_id}",
            {"status": "skipped"},
            format="json",
        )
        self.assertEqual(patch.status_code, 200)
        self.assertTrue(
            RoadmapEvent.objects.filter(
                user=self.user,
                step_id=step_id,
                event_type=RoadmapEvent.Type.STEP_SKIPPED,
            ).exists()
        )

        click = self.client.post(f"/api/me/roadmap/steps/{step_id}/click", format="json")
        self.assertEqual(click.status_code, 200)
        self.assertTrue(
            RoadmapEvent.objects.filter(
                user=self.user,
                step_id=step_id,
                event_type=RoadmapEvent.Type.STEP_CLICKED,
            ).exists()
        )

    def test_refresh_endpoint_emits_generation_events_without_fake_exposure(self):
        response = self.client.post(
            "/api/me/roadmap/refresh",
            {"category": "haircare"},
            format="json",
        )
        self.assertEqual(response.status_code, 200)

        plan = RoadmapPlan.objects.get(user=self.user, category="haircare", is_active=True)
        next_step = (
            RoadmapStep.objects.filter(
                plan=plan,
                status__in=[RoadmapStep.Status.MISSING, RoadmapStep.Status.RECOMMENDED],
            )
            .order_by("step_index")
            .first()
        )
        self.assertIsNotNone(next_step)

        plan_event = RoadmapEvent.objects.filter(
            user=self.user,
            plan=plan,
            event_type=RoadmapEvent.Type.PLAN_REFRESHED,
        ).order_by("-id").first()
        self.assertIsNotNone(plan_event)
        self.assertEqual(int((plan_event.context or {}).get("plan_id")), int(plan.id))
        self.assertEqual(str((plan_event.context or {}).get("category")), "haircare")
        self.assertEqual(str((plan_event.context or {}).get("source")), "roadmap_v1")
        self.assertEqual(int((plan_event.context or {}).get("steps_total")), int(plan.steps.count()))
        self.assertEqual(int((plan_event.context or {}).get("next_step_id")), int(next_step.id))
        self.assertTrue(str(((plan_event.context or {}).get("ml") or {}).get("decision") or "").strip())
        self.assertIn("planner", plan_event.context or {})

        generated_events = RoadmapEvent.objects.filter(
            user=self.user,
            plan=plan,
            event_type=RoadmapEvent.Type.STEP_GENERATED,
        ).order_by("step__step_index", "id")
        self.assertEqual(generated_events.count(), plan.steps.count())
        generated = generated_events.first()
        self.assertIsNotNone(generated)
        self.assertEqual(int((generated.context or {}).get("plan_id")), int(plan.id))
        self.assertEqual(int((generated.context or {}).get("step_id")), int(generated.step_id))
        self.assertEqual(str((generated.context or {}).get("plan_source")), "roadmap_v1")
        self.assertIn(str((generated.context or {}).get("source") or ""), {"rules", "ml_next_step", "planner", "planner_fallback", "user_state", ""})
        self.assertIn("ml", generated.context or {})
        self.assertIn("planner", generated.context or {})

        self.assertFalse(
            RoadmapEvent.objects.filter(
                user=self.user,
                plan=plan,
                event_type=RoadmapEvent.Type.STEP_EXPOSED,
            ).exists()
        )

    def test_checkout_logs_roadmap_step_completed(self):
        first = self._checkout(self.p_shampoo.id)
        self.assertEqual(first.status_code, 201)

        roadmap = self.client.get("/api/me/roadmap?category=haircare")
        self.assertEqual(roadmap.status_code, 200)
        next_step = (roadmap.data.get("summary") or {}).get("next_step") or {}
        step_id = int(next_step["id"])

        second = self._checkout(self.p_conditioner.id)
        self.assertEqual(second.status_code, 201)
        txn_id = int(second.data["transaction_id"])

        event = RoadmapEvent.objects.filter(
            user=self.user,
            step_id=step_id,
            event_type=RoadmapEvent.Type.STEP_COMPLETED,
        ).order_by("-id").first()
        self.assertIsNotNone(event)
        self.assertEqual(int(event.context.get("transaction_id")), txn_id)
        self.assertIn(event.context.get("matched_by"), {"product_type", "recommended_product_id"})
        latest_generated_same_step = RoadmapEvent.objects.filter(
            user=self.user,
            step_id=step_id,
            event_type=RoadmapEvent.Type.STEP_GENERATED,
        ).order_by("-id").first()
        self.assertIsNotNone(latest_generated_same_step)
        self.assertLess(int(event.id), int(latest_generated_same_step.id))

    def test_checkout_path_emits_generation_events_before_any_exposure(self):
        response = self._checkout(self.p_shampoo.id)
        self.assertEqual(response.status_code, 201)

        plan = RoadmapPlan.objects.get(user=self.user, category="haircare", is_active=True)
        self.assertTrue(
            RoadmapEvent.objects.filter(
                user=self.user,
                plan=plan,
                event_type=RoadmapEvent.Type.PLAN_REFRESHED,
            ).exists()
        )
        self.assertEqual(
            RoadmapEvent.objects.filter(
                user=self.user,
                plan=plan,
                event_type=RoadmapEvent.Type.STEP_GENERATED,
            ).count(),
            plan.steps.count(),
        )
        self.assertFalse(
            RoadmapEvent.objects.filter(
                user=self.user,
                plan=plan,
                event_type=RoadmapEvent.Type.STEP_EXPOSED,
            ).exists()
        )

    def test_roadmap_exposed_daily_dedup_on_get(self):
        first = self._checkout(self.p_shampoo.id)
        self.assertEqual(first.status_code, 201)

        r1 = self.client.get("/api/me/roadmap?category=haircare")
        r2 = self.client.get("/api/me/roadmap?category=haircare")
        self.assertEqual(r1.status_code, 200)
        self.assertEqual(r2.status_code, 200)

        next_step = (r1.data.get("summary") or {}).get("next_step") or {}
        step_id = int(next_step["id"])
        self.assertEqual(
            RoadmapEvent.objects.filter(
                user=self.user,
                step_id=step_id,
                event_type=RoadmapEvent.Type.STEP_EXPOSED,
            ).count(),
            1,
        )

    def test_offer_exposed_writes_roadmap_exposed_when_roadmap_shortcut(self):
        default_campaign, _ = CampaignBudget.objects.get_or_create(
            name="default",
            defaults={
                "weekly_limit": Decimal("1000.00"),
                "weekly_spent": Decimal("0.00"),
                "priority": 100,
                "is_active": True,
            },
        )
        Offer.objects.create(
            name="Telemetry Haircare Roadmap Offer",
            offer_type=Offer.Type.DISCOUNT,
            value=Decimal("10.00"),
            estimated_cost=Decimal("2.00"),
            is_active=True,
            target_scope="product_type",
            cooldown_days=0,
            expires_in_days=7,
            allowed_categories=["haircare"],
            allowed_product_types=["conditioner", "hair_mask", "hair_oil"],
            campaign=default_campaign,
        )

        checkout = self._checkout(self.p_shampoo.id)
        self.assertEqual(checkout.status_code, 201)
        checkout_next_offer = checkout.data.get("next_offer") or {}
        assignment_id = int(checkout_next_offer.get("assignment_id"))

        plan = RoadmapPlan.objects.get(user=self.user, category="haircare", is_active=True)
        next_step = (
            RoadmapStep.objects.filter(
                plan=plan,
                status__in=[RoadmapStep.Status.MISSING, RoadmapStep.Status.RECOMMENDED],
            )
            .order_by("step_index")
            .first()
        )
        self.assertIsNotNone(next_step)
        step_id = int(next_step.id)
        self.assertEqual(
            RoadmapEvent.objects.filter(
                user=self.user,
                step=next_step,
                event_type=RoadmapEvent.Type.STEP_EXPOSED,
            ).count(),
            0,
        )

        r1 = self.client.get("/api/me/next-offer")
        self.assertEqual(r1.status_code, 200)
        target = r1.data.get("target") or {}
        self.assertTrue(str(target.get("picked_via") or "").startswith("roadmap_shortcut"))

        self.assertEqual(
            OfferEvent.objects.filter(
                assignment_id=assignment_id,
                event_type=OfferEvent.Type.EXPOSED,
            ).count(),
            1,
        )
        self.assertEqual(
            RoadmapEvent.objects.filter(
                user=self.user,
                step_id=step_id,
                event_type=RoadmapEvent.Type.STEP_EXPOSED,
            ).count(),
            1,
        )

    def test_offer_exposed_merges_into_existing_roadmap_exposed(self):
        default_campaign, _ = CampaignBudget.objects.get_or_create(
            name="default",
            defaults={
                "weekly_limit": Decimal("1000.00"),
                "weekly_spent": Decimal("0.00"),
                "priority": 100,
                "is_active": True,
            },
        )
        Offer.objects.create(
            name="Telemetry Haircare Roadmap Offer Merge",
            offer_type=Offer.Type.DISCOUNT,
            value=Decimal("10.00"),
            estimated_cost=Decimal("2.00"),
            is_active=True,
            target_scope="product_type",
            cooldown_days=0,
            expires_in_days=7,
            allowed_categories=["haircare"],
            allowed_product_types=["conditioner", "hair_mask", "hair_oil"],
            campaign=default_campaign,
        )

        checkout = self._checkout(self.p_shampoo.id)
        self.assertEqual(checkout.status_code, 201)
        seed_assignment_id = int((checkout.data.get("next_offer") or {}).get("assignment_id"))
        OfferAssignment.objects.filter(id=seed_assignment_id).update(is_active=False)

        roadmap = self.client.get("/api/me/roadmap?category=haircare")
        self.assertEqual(roadmap.status_code, 200)
        next_step = (roadmap.data.get("summary") or {}).get("next_step") or {}
        step_id = int(next_step["id"])

        exposed_before = RoadmapEvent.objects.filter(
            user=self.user,
            step_id=step_id,
            event_type=RoadmapEvent.Type.STEP_EXPOSED,
        ).order_by("-id").first()
        self.assertIsNotNone(exposed_before)
        self.assertIn((exposed_before.context or {}).get("offer_assignment_id"), [None, ""])

        OfferAssignment.objects.filter(id=seed_assignment_id).update(is_active=True)
        next_offer = self.client.get("/api/me/next-offer")
        self.assertEqual(next_offer.status_code, 200)
        assignment_id = int(next_offer.data["assignment_id"])
        target = next_offer.data.get("target") or {}
        self.assertTrue(str(target.get("picked_via") or "").startswith("roadmap_shortcut"))

        self.assertEqual(
            OfferEvent.objects.filter(
                assignment_id=assignment_id,
                event_type=OfferEvent.Type.EXPOSED,
            ).count(),
            1,
        )
        self.assertEqual(
            RoadmapEvent.objects.filter(
                user=self.user,
                step_id=step_id,
                event_type=RoadmapEvent.Type.STEP_EXPOSED,
            ).count(),
            1,
        )

        exposed_after = RoadmapEvent.objects.filter(
            user=self.user,
            step_id=step_id,
            event_type=RoadmapEvent.Type.STEP_EXPOSED,
        ).order_by("-id").first()
        self.assertIsNotNone(exposed_after)
        self.assertEqual((exposed_after.context or {}).get("offer_assignment_id"), assignment_id)
        sources = (exposed_after.context or {}).get("sources") or []
        self.assertIn("offers", sources)

        r2 = self.client.get("/api/me/next-offer")
        self.assertEqual(r2.status_code, 200)
        self.assertEqual(
            OfferEvent.objects.filter(
                assignment_id=assignment_id,
                event_type=OfferEvent.Type.EXPOSED,
            ).count(),
            1,
        )
        self.assertEqual(
            RoadmapEvent.objects.filter(
                user=self.user,
                step_id=step_id,
                event_type=RoadmapEvent.Type.STEP_EXPOSED,
            ).count(),
            1,
        )
