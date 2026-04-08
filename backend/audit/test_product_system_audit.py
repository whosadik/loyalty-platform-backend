from __future__ import annotations

from datetime import timedelta
from decimal import Decimal

from django.contrib.auth import get_user_model
from django.core.management import call_command
from django.test import override_settings
from django.utils import timezone
from rest_framework.test import APITestCase

from admin_tools.product_system_audit import build_product_system_audit_payload
from catalog.models import Product
from loyalty.models import LoyaltyLedgerEntry
from offers.models import Offer, OfferAssignment, OfferEvent
from roadmap_app.models import RoadmapEvent, RoadmapPlan, RoadmapStep
from transactions.models import OwnedProduct, Transaction, TransactionItem


@override_settings(
    CACHES={"default": {"BACKEND": "django.core.cache.backends.locmem.LocMemCache"}},
)
class ProductSystemAuditRuntimeTests(APITestCase):
    def setUp(self):
        user_model = get_user_model()
        self.user = user_model.objects.create_user(username="audit_runtime_u1", password="pass12345")
        self.client.force_authenticate(self.user)

    def _create_product(
        self,
        *,
        name: str,
        category: str = Product.Category.SKINCARE,
        product_type: str = "cleanser",
        price: str = "100.00",
        in_stock: bool = True,
    ) -> Product:
        return Product.objects.create(
            name=name,
            brand="Audit Lab",
            price=Decimal(price),
            currency="KZT",
            category=category,
            product_type=product_type,
            concerns=[],
            attrs={},
            actives=[],
            flags=[],
            supported_skin_types=[],
            strength=Product.Strength.LOW,
            in_stock=in_stock,
        )

    def _create_assignment(
        self,
        *,
        offer_type: str = Offer.Type.DISCOUNT,
        value: str = "10.00",
        target: dict | None = None,
        is_active: bool = True,
        is_redeemed: bool = False,
        expires_in_days: int = 7,
        reason: dict | None = None,
    ) -> OfferAssignment:
        offer_expires_in_days = max(1, int(expires_in_days or 0))
        offer = Offer.objects.create(
            is_active=True,
            name=f"Audit {offer_type}",
            offer_type=offer_type,
            value=Decimal(value),
            estimated_cost=Decimal("1.00"),
            expires_in_days=offer_expires_in_days,
        )
        return OfferAssignment.objects.create(
            user=self.user,
            offer=offer,
            target=target or {"scope": "cart"},
            reason=reason or {},
            is_active=is_active,
            is_redeemed=is_redeemed,
            expires_at=timezone.now() + timedelta(days=expires_in_days),
        )

    def _create_transaction_for_product(self, product: Product) -> Transaction:
        txn = Transaction.objects.create(user=self.user, total_amount=Decimal(str(product.price or "0")))
        TransactionItem.objects.create(
            transaction=txn,
            product=product,
            quantity=1,
            unit_price=Decimal(str(product.price)),
        )
        return txn

    def test_checkout_idempotency_preserves_single_transaction_and_owned_quantity(self):
        product = self._create_product(name="Idempotent Cleanser")

        first = self.client.post(
            "/api/checkout",
            {
                "idempotency_key": "idem-audit-1",
                "items": [{"product": product.id, "quantity": 2}],
            },
            format="json",
        )
        self.assertEqual(first.status_code, 201)
        self.assertTrue(first.data["ok"])
        self.assertEqual(Transaction.objects.count(), 1)
        self.assertEqual(TransactionItem.objects.count(), 1)
        self.assertEqual(OwnedProduct.objects.get(user=self.user, product=product).quantity_total, 2)
        self.assertEqual(
            LoyaltyLedgerEntry.objects.filter(account__user=self.user).count(),
            1,
        )

        replay = self.client.post(
            "/api/checkout",
            {
                "idempotency_key": "idem-audit-1",
                "items": [{"product": product.id, "quantity": 2}],
            },
            format="json",
        )
        self.assertEqual(replay.status_code, 200)
        self.assertTrue(replay.data["idempotent_replay"])
        self.assertEqual(Transaction.objects.count(), 1)
        self.assertEqual(TransactionItem.objects.count(), 1)
        self.assertEqual(OwnedProduct.objects.get(user=self.user, product=product).quantity_total, 2)
        self.assertEqual(
            LoyaltyLedgerEntry.objects.filter(account__user=self.user).count(),
            1,
        )

    def test_checkout_marks_redeemed_assignment_inactive(self):
        product = self._create_product(name="Discount Serum", product_type="serum")
        assignment = self._create_assignment(
            target={"scope": "product_id", "value": product.id, "category": product.category, "product_type": product.product_type},
        )

        resp = self.client.post(
            "/api/checkout",
            {
                "items": [{"product": product.id, "quantity": 1}],
                "apply_assignment_id": assignment.id,
            },
            format="json",
        )
        self.assertEqual(resp.status_code, 201)

        assignment.refresh_from_db()
        self.assertTrue(assignment.is_redeemed)
        self.assertFalse(assignment.is_active)
        self.assertIsNotNone(assignment.redeemed_transaction_id)
        self.assertTrue(
            OfferEvent.objects.filter(
                assignment=assignment,
                event_type=OfferEvent.Type.REDEEMED,
            ).exists()
        )

    def test_inactive_assignment_is_rejected_by_checkout_and_offer_endpoints(self):
        product = self._create_product(name="Inactive Assignment Product", product_type="mask")
        inactive_assignment = self._create_assignment(
            offer_type=Offer.Type.POINTS_MULTIPLIER,
            value="2.00",
            target={"scope": "product_id", "value": product.id, "category": product.category, "product_type": product.product_type},
            is_active=False,
        )
        txn = self._create_transaction_for_product(product)

        checkout_resp = self.client.post(
            "/api/checkout",
            {
                "items": [{"product": product.id, "quantity": 1}],
                "apply_assignment_id": inactive_assignment.id,
            },
            format="json",
        )
        self.assertEqual(checkout_resp.status_code, 400)
        self.assertEqual(checkout_resp.data["message"], "Validation error")
        self.assertEqual(checkout_resp.data["details"]["message"], "Offer is no longer active")

        preview_resp = self.client.post(
            "/api/offers/preview",
            {
                "assignment_id": inactive_assignment.id,
                "items": [{"product": product.id, "quantity": 1}],
            },
            format="json",
        )
        self.assertEqual(preview_resp.status_code, 400)
        self.assertEqual(preview_resp.data["message"], "Offer is no longer active")

        click_resp = self.client.post(
            "/api/offers/click",
            {"assignment_id": inactive_assignment.id},
            format="json",
        )
        self.assertEqual(click_resp.status_code, 400)
        self.assertEqual(click_resp.data["message"], "Offer is no longer active")

        redeem_resp = self.client.post(
            "/api/offers/redeem",
            {"assignment_id": inactive_assignment.id, "transaction_id": txn.id},
            format="json",
        )
        self.assertEqual(redeem_resp.status_code, 400)
        self.assertEqual(redeem_resp.data["message"], "Offer is no longer active")

    def test_checkout_and_preview_reject_out_of_stock_product(self):
        product = self._create_product(
            name="Out Of Stock SPF",
            product_type="spf",
            in_stock=False,
        )

        preview_resp = self.client.post(
            "/api/checkout/preview",
            {"items": [{"product": product.id, "quantity": 1}]},
            format="json",
        )
        self.assertEqual(preview_resp.status_code, 400)
        self.assertEqual(preview_resp.data["message"], "Validation error")
        self.assertEqual(preview_resp.data["details"]["message"], f"Product {product.id} is out of stock")

        checkout_resp = self.client.post(
            "/api/checkout",
            {"items": [{"product": product.id, "quantity": 1}]},
            format="json",
        )
        self.assertEqual(checkout_resp.status_code, 400)
        self.assertEqual(checkout_resp.data["message"], "Validation error")
        self.assertEqual(checkout_resp.data["details"]["message"], f"Product {product.id} is out of stock")
        self.assertEqual(Transaction.objects.count(), 0)

    def test_stale_roadmap_shortcut_offer_is_not_returned_after_roadmap_completion(self):
        plan = RoadmapPlan.objects.create(
            user=self.user,
            category=RoadmapPlan.Category.MAKEUP,
            is_active=True,
            meta={},
        )
        completed_step = RoadmapStep.objects.create(
            plan=plan,
            step_index=1,
            product_type="foundation",
            status=RoadmapStep.Status.COMPLETED,
            suggestions=[],
            why=["picked via rules"],
        )
        stale_assignment = self._create_assignment(
            target={
                "scope": "product_type",
                "value": "blush",
                "category": "makeup",
                "picked_via": "roadmap_shortcut",
            },
            reason={
                "roadmap": {
                    "plan_id": plan.id,
                    "step_id": completed_step.id,
                    "category": "makeup",
                    "step_index": completed_step.step_index,
                    "next_product_type": "blush",
                }
            },
        )

        roadmap_resp = self.client.get("/api/me/roadmap?category=makeup")
        self.assertEqual(roadmap_resp.status_code, 200)
        self.assertIsNone(roadmap_resp.data["summary"]["next_step"])

        next_offer_resp = self.client.get("/api/me/next-offer")
        self.assertEqual(next_offer_resp.status_code, 200)
        self.assertIsNone(next_offer_resp.data["offer"])

        stale_assignment.refresh_from_db()
        self.assertFalse(stale_assignment.is_active)

        offers_resp = self.client.get("/api/me/offers")
        self.assertEqual(offers_resp.status_code, 200)
        self.assertEqual(offers_resp.json(), [])

    def test_cleanup_offers_deactivates_expired_active_assignment_idempotently(self):
        assignment = self._create_assignment(expires_in_days=-1)

        call_command("cleanup_offers")

        assignment.refresh_from_db()
        self.assertFalse(assignment.is_active)
        self.assertFalse(assignment.is_redeemed)
        self.assertEqual(
            OfferEvent.objects.filter(
                assignment=assignment,
                event_type=OfferEvent.Type.EXPIRED,
            ).count(),
            1,
        )

        call_command("cleanup_offers")

        assignment.refresh_from_db()
        self.assertFalse(assignment.is_active)
        self.assertEqual(
            OfferEvent.objects.filter(
                assignment=assignment,
                event_type=OfferEvent.Type.EXPIRED,
            ).count(),
            1,
        )

    def test_expired_assignment_is_not_returned_by_active_offer_read_path(self):
        expired_assignment = self._create_assignment(expires_in_days=-1)

        offers_resp = self.client.get("/api/me/offers")
        self.assertEqual(offers_resp.status_code, 200)
        self.assertEqual(offers_resp.json(), [])

        expired_assignment.refresh_from_db()
        self.assertFalse(expired_assignment.is_active)
        self.assertEqual(
            OfferEvent.objects.filter(
                assignment=expired_assignment,
                event_type=OfferEvent.Type.EXPIRED,
            ).count(),
            1,
        )

    def test_expired_assignment_is_rejected_consistently_across_offer_endpoints(self):
        product = self._create_product(name="Expired Offer Product", product_type="essence")
        assignment = self._create_assignment(
            offer_type=Offer.Type.POINTS_MULTIPLIER,
            value="2.00",
            target={"scope": "product_id", "value": product.id, "category": product.category, "product_type": product.product_type},
            expires_in_days=-1,
        )
        txn = self._create_transaction_for_product(product)

        preview_resp = self.client.post(
            "/api/offers/preview",
            {
                "assignment_id": assignment.id,
                "items": [{"product": product.id, "quantity": 1}],
            },
            format="json",
        )
        self.assertEqual(preview_resp.status_code, 400)
        self.assertEqual(preview_resp.data["message"], "Offer expired")

        click_resp = self.client.post(
            "/api/offers/click",
            {"assignment_id": assignment.id},
            format="json",
        )
        self.assertEqual(click_resp.status_code, 400)
        self.assertEqual(click_resp.data["message"], "Offer expired")

        redeem_resp = self.client.post(
            "/api/offers/redeem",
            {"assignment_id": assignment.id, "transaction_id": txn.id},
            format="json",
        )
        self.assertEqual(redeem_resp.status_code, 400)
        self.assertEqual(redeem_resp.data["message"], "Offer expired")

        checkout_resp = self.client.post(
            "/api/checkout",
            {
                "items": [{"product": product.id, "quantity": 1}],
                "apply_assignment_id": assignment.id,
            },
            format="json",
        )
        self.assertEqual(checkout_resp.status_code, 400)
        self.assertEqual(checkout_resp.data["message"], "Validation error")
        self.assertEqual(checkout_resp.data["details"]["message"], "Offer expired")

        preview_checkout_resp = self.client.post(
            "/api/checkout/preview",
            {
                "items": [{"product": product.id, "quantity": 1}],
                "apply_assignment_id": assignment.id,
            },
            format="json",
        )
        self.assertEqual(preview_checkout_resp.status_code, 400)
        self.assertEqual(preview_checkout_resp.data["message"], "Offer expired")

        assignment.refresh_from_db()
        self.assertFalse(assignment.is_active)
        self.assertEqual(
            OfferEvent.objects.filter(
                assignment=assignment,
                event_type=OfferEvent.Type.EXPIRED,
            ).count(),
            1,
        )

    def test_checkout_rejects_too_long_idempotency_key_with_validation_error(self):
        product = self._create_product(name="Long Idempotency Product", product_type="mask")
        payload = {
            "idempotency_key": "x" * 65,
            "items": [{"product": product.id, "quantity": 1}],
        }

        preview_resp = self.client.post("/api/checkout/preview", payload, format="json")
        self.assertEqual(preview_resp.status_code, 400)
        self.assertEqual(preview_resp.data["code"], "validation_error")
        self.assertIn("idempotency_key", preview_resp.data["details"])

        checkout_resp = self.client.post("/api/checkout", payload, format="json")
        self.assertEqual(checkout_resp.status_code, 400)
        self.assertEqual(checkout_resp.data["code"], "validation_error")
        self.assertIn("idempotency_key", checkout_resp.data["details"])
        self.assertEqual(Transaction.objects.count(), 0)

    def test_product_system_audit_separates_fragrance_step_state_drift_from_runtime_failure(self):
        warm_day = Product.objects.create(
            name="Audit Warm Day",
            brand="Audit Lab",
            price=Decimal("140.00"),
            currency="KZT",
            category=Product.Category.FRAGRANCE,
            product_type="edp",
            concerns=[],
            attrs={"scent_family": "citrus", "notes": ["bergamot"], "intensity": "soft"},
            actives=[],
            flags=[],
            supported_skin_types=[],
            strength=Product.Strength.LOW,
            in_stock=True,
        )
        plan = RoadmapPlan.objects.create(
            user=self.user,
            category=RoadmapPlan.Category.FRAGRANCE,
            is_active=False,
            meta={},
        )
        step = RoadmapStep.objects.create(
            plan=plan,
            step_index=1,
            product_type="cold_day",
            status=RoadmapStep.Status.COMPLETED,
            recommended_product=warm_day,
            suggestions=[warm_day.id],
            why=["picked via rules"],
        )
        RoadmapEvent.objects.create(
            user=self.user,
            plan=plan,
            step=step,
            event_type=RoadmapEvent.Type.STEP_COMPLETED,
            context={
                "category": "fragrance",
                "product_type": "warm_day",
                "matched_by": "recommended_product_id",
                "recommended_product_id": warm_day.id,
                "match_meta": {
                    "recommended_product_id": warm_day.id,
                    "purchased_product_id": warm_day.id,
                    "purchased_product_type": "warm_day",
                },
            },
        )

        payload = build_product_system_audit_payload()
        invariants = {item["name"]: item for item in payload["invariants"]}
        fragrance_legacy = payload["runtime_snapshot"]["fragrance_legacy"]

        self.assertEqual(invariants["fragrance_slot_level_runtime_integrity"]["status"], "PASS")
        self.assertEqual(
            invariants["legacy_fragrance_completion_noise_isolated_from_current_runtime_truth"]["status"],
            "PASS",
        )
        self.assertEqual(int(fragrance_legacy["bad_fragrance_completed_exact_match_recent_30d"]), 0)
        self.assertEqual(int(fragrance_legacy["step_state_drift_recent_30d"]), 1)
