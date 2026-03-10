from __future__ import annotations

from collections import Counter
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import patch

from django.contrib.auth import get_user_model
from django.test import TestCase

from admin_tools.management.commands import simulate_roadmap_sessions as sim_cmd
from catalog.models import Product
from roadmap_app.models import RoadmapEvent, RoadmapPlan, RoadmapStep


class SimulatorFreshUsersTests(TestCase):
    def setUp(self):
        self.command = sim_cmd.Command()
        self.command._fresh_user_token = "tok123"

    def test_select_or_create_users_fresh_ga_batch_does_not_reuse_existing(self):
        User = get_user_model()
        existing = User.objects.create_user(username="ga_existing_000001", password="pass12345")

        selected = self.command._select_or_create_users(
            users_n=2,
            include_ga=True,
            batch_size=10,
            fresh_users_only=True,
        )

        usernames = [str(user.username) for user in selected]
        self.assertEqual(len(usernames), 2)
        self.assertNotIn(existing.username, usernames)
        self.assertTrue(all(name.startswith("ga_fresh_tok123_") for name in usernames))

    def test_select_or_create_users_reuses_existing_ga_users_by_default(self):
        User = get_user_model()
        first = User.objects.create_user(username="ga_existing_000001", password="pass12345")
        second = User.objects.create_user(username="ga_existing_000002", password="pass12345")

        selected = self.command._select_or_create_users(
            users_n=2,
            include_ga=True,
            batch_size=10,
            fresh_users_only=False,
        )

        self.assertEqual([int(user.id) for user in selected], [int(first.id), int(second.id)])


class SimulatorPersistedExposureCountTests(TestCase):
    def setUp(self):
        self.command = sim_cmd.Command()
        User = get_user_model()
        self.user = User.objects.create_user(username="sim_exposure_u1", password="pass12345")
        self.other_user = User.objects.create_user(username="sim_exposure_u2", password="pass12345")

        product = Product.objects.create(
            name="Exposure Count Product",
            brand="B",
            price=Decimal("10.00"),
            category="skincare",
            product_type="serum",
            in_stock=True,
        )
        self.plan = RoadmapPlan.objects.create(user=self.user, category="skincare", is_active=True, meta={})
        self.step = RoadmapStep.objects.create(
            plan=self.plan,
            step_index=1,
            product_type="serum",
            status=RoadmapStep.Status.RECOMMENDED,
            recommended_product=product,
        )

    def _event(self, *, user, created_at: datetime) -> None:
        event = RoadmapEvent.objects.create(
            user=user,
            plan=self.plan if user == self.user else None,
            step=self.step if user == self.user else None,
            event_type=RoadmapEvent.Type.STEP_EXPOSED,
            context={},
        )
        RoadmapEvent.objects.filter(id=event.id).update(created_at=created_at)

    def test_count_user_step_exposed_for_utc_day_counts_only_matching_user_and_day(self):
        ref_dt = datetime(2026, 3, 10, 12, 0, tzinfo=timezone.utc)
        self._event(user=self.user, created_at=ref_dt - timedelta(hours=1))
        self._event(user=self.user, created_at=ref_dt - timedelta(days=1))
        self._event(user=self.other_user, created_at=ref_dt - timedelta(minutes=10))

        count = self.command._count_user_step_exposed_for_utc_day(
            user_id=int(self.user.id),
            ref_dt=ref_dt,
        )

        self.assertEqual(count, 1)


class SimulatorNextStepPayloadTests(TestCase):
    def test_payload_matches_next_step_detects_target_and_recommended_product(self):
        command = sim_cmd.Command()

        class FakeProductCache:
            def product_meta(self, pid):
                if int(pid) == 1001:
                    return {
                        "in_stock": True,
                        "category": "makeup",
                        "product_type": "foundation",
                    }
                if int(pid) == 1002:
                    return {
                        "in_stock": True,
                        "category": "makeup",
                        "product_type": "primer",
                    }
                return None

        has_step_target, has_recommended = command._payload_matches_next_step(
            items=[{"product": 1002, "quantity": 1}, {"product": 1001, "quantity": 1}],
            category="makeup",
            step_product_type="foundation",
            recommended_product_id=1001,
            product_cache=FakeProductCache(),
        )

        self.assertTrue(has_step_target)
        self.assertTrue(has_recommended)

    def test_recommended_product_id_from_step_falls_back_to_nested_product_object(self):
        command = sim_cmd.Command()

        rec_pid = command._recommended_product_id_from_step(
            next_step={"recommended_product_id": 1001},
            next_step_row={"recommended_product": {"id": 1002, "name": "Conditioner X"}},
        )
        self.assertEqual(rec_pid, 1002)

        rec_pid = command._recommended_product_id_from_step(
            next_step={"recommended_product": {"id": 1003}},
            next_step_row={},
        )
        self.assertEqual(rec_pid, 1003)


class SimulatorRepeatFlowTests(TestCase):
    def test_repeat_after_step1_completion_exposes_step2_and_updates_counters(self):
        command = sim_cmd.Command()

        class FakeRng:
            def __init__(self, values):
                self.values = list(values)

            def random(self):
                if not self.values:
                    return 0.99
                return float(self.values.pop(0))

            def randint(self, a, b):
                return int(a)

        rng = FakeRng(
            [
                0.99,  # skip refresh
                0.0,   # initial roadmap get
                0.99,  # skip next-offer get
                0.99,  # no step click/skip
                0.0,   # do place order
                0.0,   # should_complete=True
                0.0,   # return after step_1 completion
                0.0,   # do follow-up checkout after repeat
            ]
        )
        counters: Counter[str] = Counter()
        warnings: list[str] = []
        state = sim_cmd.UserState(
            segment=sim_cmd.SEGMENT_CONFIGS["active"],
            favorite_category="haircare",
            profile=None,
        )

        roadmap_payload_step1 = {
            "summary": {
                "next_step": {"id": 101, "step_index": 1, "product_type": "shampoo", "recommended_product_id": 1001}
            },
            "steps": [
                {"id": 101, "step_index": 1, "product_type": "shampoo", "recommended_product_id": 1001},
                {"id": 102, "step_index": 2, "product_type": "conditioner", "recommended_product_id": 1002},
            ],
        }
        roadmap_payload_step2 = {
            "summary": {
                "next_step": {"id": 102, "step_index": 2, "product_type": "conditioner", "recommended_product_id": 1002}
            },
            "steps": [
                {"id": 101, "step_index": 1, "product_type": "shampoo", "recommended_product_id": 1001},
                {"id": 102, "step_index": 2, "product_type": "conditioner", "recommended_product_id": 1002},
            ],
        }

        get_responses = [
            SimpleNamespace(status_code=200, data=roadmap_payload_step1, content=b""),
            SimpleNamespace(status_code=200, data=roadmap_payload_step2, content=b""),
        ]

        def fake_api_get(**kwargs):
            return get_responses.pop(0)

        def fake_api_post(**kwargs):
            return SimpleNamespace(status_code=201, data={"transaction_id": 1}, content=b"")

        with patch.object(command, "_choose_category", return_value="haircare"), patch.object(
            command,
            "_api_get",
            side_effect=fake_api_get,
        ), patch.object(
            command,
            "_api_post",
            side_effect=fake_api_post,
        ), patch.object(
            command,
            "_build_checkout_items",
            side_effect=[
                ([{"product": 1001, "quantity": 1}], [1001], 1001),
                ([{"product": 1002, "quantity": 1}], [1002], 1002),
            ],
        ), patch.object(
            command,
            "_sanitize_checkout_items",
            side_effect=lambda items, **kwargs: items,
        ), patch.object(
            command,
            "_build_checkout_payload",
            side_effect=[
                ({"channel": "web", "items": [{"product": 1001, "quantity": 1}]}, False),
                ({"channel": "web", "items": [{"product": 1002, "quantity": 1}]}, False),
            ],
        ), patch.object(
            command,
            "_payload_matches_next_step",
            side_effect=[(True, True), (True, True)],
        ), patch.object(
            command,
            "_count_user_step_exposed_for_utc_day",
            side_effect=[0, 1, 1, 2],
        ), patch.object(
            command,
            "_count_user_specific_step_event_for_utc_day",
            side_effect=[0, 1, 0, 1],
        ):
            idem_counter = command._simulate_single_session(
                client=None,
                user_id=123,
                day_index=0,
                session_index=0,
                sim_now=datetime(2026, 3, 10, 12, 0, tzinfo=timezone.utc),
                state=state,
                max_orders_per_session=1,
                max_items=1,
                p_roadmap_get=1.0,
                p_next_offer_get=0.0,
                p_step_click=0.0,
                p_step_skip=0.0,
                p_complete_next_step=1.0,
                p_redeem_offer=0.0,
                p_return_after_step1_exposure=0.0,
                p_return_after_step1_completion=1.0,
                p_followup_checkout_after_repeat=1.0,
                counters=counters,
                warnings=warnings,
                product_cache=None,
                rng=rng,
                idem_counter=0,
                run_nonce="r1",
            )

        self.assertEqual(idem_counter, 2)
        self.assertEqual(counters["roadmap_get_ok"], 2)
        self.assertEqual(counters["roadmap_step_exposed_persisted"], 2)
        self.assertEqual(counters["step_1_exposed"], 1)
        self.assertEqual(counters["step_1_completed"], 1)
        self.assertEqual(counters["returned_after_step_1_completion"], 1)
        self.assertEqual(counters["step_2_exposed"], 1)
        self.assertEqual(counters["step_2_completed"], 1)
        self.assertEqual(counters["checkout_targeted_next_step"], 2)
        self.assertEqual(counters["checkout_targeted_recommended_product"], 2)
        self.assertEqual(counters["followup_checkout_attempts"], 1)
        self.assertEqual(counters["followup_checkout_success"], 1)
        self.assertEqual(counters["followup_step_completed"], 1)
        self.assertEqual(warnings, [])
