from __future__ import annotations

from datetime import timedelta
from decimal import Decimal
from unittest.mock import patch

from django.contrib.auth import get_user_model
from django.test import TestCase
from django.test.utils import override_settings
from django.utils import timezone

from offers.services import _build_rec_profile
from roadmap_app.models import RoadmapEvent, RoadmapPlan, RoadmapStep
from roadmap_app.sku_ranking import _recent_recommended_product_health_map
from roadmap_app.services import _recommend_candidates_for_type
from catalog.models import Product
from users_app.models import CustomerProfile


class RoadmapSkuRankingTests(TestCase):
    def setUp(self):
        User = get_user_model()
        self.user = User.objects.create_user(username="roadmap_sku_u1", password="pass12345")

    def _profile(self, *, goals: list[str], avoid_flags: list[str], hair_profile: dict) -> CustomerProfile:
        cp, _ = CustomerProfile.objects.get_or_create(user=self.user)
        cp.goals = list(goals)
        cp.avoid_flags = list(avoid_flags)
        cp.hair_profile = dict(hair_profile)
        cp.budget = CustomerProfile.Budget.HIGH
        cp.save(update_fields=["goals", "avoid_flags", "hair_profile", "budget", "updated_at"])
        return cp

    def _product_row(
        self,
        *,
        pid: int,
        name: str,
        product_type: str,
        concerns: list[str],
        actives: list[str],
        flags: list[str],
        attrs: dict,
        raw_meta: dict,
        ingredients_inci: str,
    ) -> dict:
        return {
            "id": pid,
            "name": name,
            "brand": "RankLab",
            "price": Decimal("10000.00"),
            "category": "haircare",
            "product_type": product_type,
            "concerns": list(concerns),
            "attrs": dict(attrs),
            "raw_meta": dict(raw_meta),
            "actives": list(actives),
            "flags": list(flags),
            "supported_skin_types": [],
            "strength": "low",
            "in_stock": True,
            "ingredients_inci": ingredients_inci,
        }

    def _roadmap_event(
        self,
        *,
        user,
        plan: RoadmapPlan | None,
        step: RoadmapStep | None,
        event_type: str,
        created_at,
        context: dict,
    ) -> RoadmapEvent:
        event = RoadmapEvent.objects.create(
            user=user,
            plan=plan,
            step=step,
            event_type=event_type,
            context=context,
        )
        RoadmapEvent.objects.filter(id=event.id).update(created_at=created_at)
        event.refresh_from_db()
        return event

    def test_haircare_leavein_reranker_prefers_lightweight_leavein_for_fine_profile(self):
        cp = self._profile(
            goals=["volume", "lightweight_care"],
            avoid_flags=["heavy_oils"],
            hair_profile={
                "hair_type": "wavy",
                "scalp_type": "normal",
                "hair_thickness": "fine",
                "concerns": ["frizz", "definition"],
            },
        )
        prof = _build_rec_profile(cp)

        anchor_mask = self._product_row(
            pid=9001,
            name="Anchor Hair Mask",
            product_type="hair_mask",
            concerns=["frizz", "dryness", "definition"],
            actives=["glycerin", "panthenol"],
            flags=[],
            attrs={"hair_type": "wavy", "hair_thickness": "fine"},
            raw_meta={"finish": "soft"},
            ingredients_inci="Aqua, Glycerin, Panthenol, Aloe",
        )
        rich_leave_in = self._product_row(
            pid=9101,
            name="Rich Leave In",
            product_type="leave_in",
            concerns=["definition", "dryness"],
            actives=["coconut_oil", "shea_butter", "glycerin"],
            flags=["heavy_oils"],
            attrs={"hair_type": "curly", "hair_thickness": "thick"},
            raw_meta={"finish": "rich"},
            ingredients_inci="Aqua, Coconut Oil, Shea Butter, Glycerin",
        )
        light_leave_in = self._product_row(
            pid=9102,
            name="Light Leave In",
            product_type="leave_in",
            concerns=["frizz", "definition", "volume", "lightweight_care"],
            actives=["panthenol", "aloe", "rice_protein"],
            flags=[],
            attrs={"hair_type": "wavy", "hair_thickness": "fine"},
            raw_meta={"finish": "airy"},
            ingredients_inci="Aqua, Panthenol, Aloe, Rice Protein",
        )

        rec_rows = [
            {"product": rich_leave_in, "score": 0.91, "components": {"cooccurrence": 3}, "why": ["base order"]},
            {"product": light_leave_in, "score": 0.84, "components": {"cooccurrence": 1}, "why": ["base order"]},
        ]

        with patch("roadmap_app.services.rec_recommend", return_value=rec_rows):
            ranked = _recommend_candidates_for_type(
                user=self.user,
                category="haircare",
                product_type="leave_in",
                context_product_ids=[anchor_mask["id"]],
                context_products=[anchor_mask],
                owned_product_ids=set(),
                used_recommended_ids=set(),
                prof=prof,
                products_for_recs=[anchor_mask, rich_leave_in, light_leave_in],
                co_map={},
            )

        self.assertEqual(int(ranked[0]["product"]["id"]), int(light_leave_in["id"]))
        self.assertIn("roadmap_rerank", ranked[0]["components"])
        self.assertGreater(float(ranked[0]["score"]), float(ranked[1]["score"]))

    def test_haircare_scalp_serum_reranker_prefers_oil_control_variant_for_oily_scalp(self):
        cp = self._profile(
            goals=["scalp_balance"],
            avoid_flags=[],
            hair_profile={
                "hair_type": "straight",
                "scalp_type": "oily",
                "hair_thickness": "medium",
                "concerns": ["oiliness", "flakes", "build_up"],
            },
        )
        prof = _build_rec_profile(cp)

        anchor_shampoo = self._product_row(
            pid=9201,
            name="Scalp Reset Shampoo",
            product_type="shampoo",
            concerns=["oiliness", "build_up"],
            actives=["salicylic_acid"],
            flags=[],
            attrs={"scalp_type": "oily"},
            raw_meta={"finish": "fresh"},
            ingredients_inci="Aqua, Salicylic Acid, Zinc PCA",
        )
        sensitive_serum = self._product_row(
            pid=9301,
            name="Sensitive Scalp Serum",
            product_type="scalp_serum",
            concerns=["scalp_health"],
            actives=["aloe", "panthenol"],
            flags=[],
            attrs={"scalp_type": "sensitive"},
            raw_meta={"finish": "soft"},
            ingredients_inci="Aqua, Aloe, Panthenol",
        )
        oil_control_serum = self._product_row(
            pid=9302,
            name="Oil Control Scalp Serum",
            product_type="scalp_serum",
            concerns=["oiliness", "flakes", "scalp_balance"],
            actives=["salicylic_acid", "tea_tree", "zinc_pca"],
            flags=[],
            attrs={"scalp_type": "oily"},
            raw_meta={"finish": "fresh"},
            ingredients_inci="Aqua, Salicylic Acid, Tea Tree, Zinc PCA",
        )

        rec_rows = [
            {"product": sensitive_serum, "score": 0.88, "components": {"cooccurrence": 2}, "why": ["base order"]},
            {"product": oil_control_serum, "score": 0.8, "components": {"cooccurrence": 1}, "why": ["base order"]},
        ]

        with patch("roadmap_app.services.rec_recommend", return_value=rec_rows):
            ranked = _recommend_candidates_for_type(
                user=self.user,
                category="haircare",
                product_type="scalp_serum",
                context_product_ids=[anchor_shampoo["id"]],
                context_products=[anchor_shampoo],
                owned_product_ids=set(),
                used_recommended_ids=set(),
                prof=prof,
                products_for_recs=[anchor_shampoo, sensitive_serum, oil_control_serum],
                co_map={},
            )

        self.assertEqual(int(ranked[0]["product"]["id"]), int(oil_control_serum["id"]))
        self.assertIn("roadmap_rerank", ranked[0]["components"])
        self.assertGreater(float(ranked[0]["score"]), float(ranked[1]["score"]))

    @override_settings(
        ROADMAP_SKU_HEALTH_PENALTY_ENABLED=True,
        ROADMAP_SKU_HEALTH_PENALTY_CATEGORIES=["haircare"],
        ROADMAP_SKU_HEALTH_PENALTY_PRODUCT_TYPES=["shampoo", "conditioner"],
        ROADMAP_SKU_HEALTH_PENALTY_MIN_RECOMMENDED_STEPS=20,
        ROADMAP_SKU_HEALTH_PENALTY_MAX_EXACT_ADOPTION_RATE=0.01,
        ROADMAP_SKU_HEALTH_PENALTY_MAX_EFFECTIVE_ADOPTION_RATE=0.03,
        ROADMAP_SKU_HEALTH_PENALTY_SEMANTIC_WEIGHT=0.25,
        ROADMAP_SKU_HEALTH_PENALTY_MAX_VALUE=0.65,
    )
    def test_haircare_shampoo_health_penalty_demotes_dead_sku(self):
        cp = self._profile(
            goals=["cleanse"],
            avoid_flags=[],
            hair_profile={
                "hair_type": "straight",
                "scalp_type": "normal",
                "hair_thickness": "medium",
                "concerns": ["shine"],
            },
        )
        prof = _build_rec_profile(cp)

        dominant_shampoo = self._product_row(
            pid=9401,
            name="Dominant Shampoo",
            product_type="shampoo",
            concerns=["shine"],
            actives=["aloe"],
            flags=[],
            attrs={"hair_type": "straight", "scalp_type": "normal", "hair_thickness": "medium"},
            raw_meta={"finish": "fresh"},
            ingredients_inci="Aqua, Aloe",
        )
        healthy_shampoo = self._product_row(
            pid=9402,
            name="Healthy Shampoo",
            product_type="shampoo",
            concerns=["shine"],
            actives=["aloe"],
            flags=[],
            attrs={"hair_type": "straight", "scalp_type": "normal", "hair_thickness": "medium"},
            raw_meta={"finish": "fresh"},
            ingredients_inci="Aqua, Aloe",
        )
        rec_rows = [
            {"product": dominant_shampoo, "score": 0.91, "components": {"cooccurrence": 3}, "why": ["base order"]},
            {"product": healthy_shampoo, "score": 0.62, "components": {"cooccurrence": 1}, "why": ["base order"]},
        ]
        health_map = {
            int(dominant_shampoo["id"]): {
                "recommended_steps": 61,
                "exact_adoption_rate": 0.0,
                "semantic_alternative_rate": 0.0,
                "effective_adoption_rate": 0.0,
            },
            int(healthy_shampoo["id"]): {
                "recommended_steps": 24,
                "exact_adoption_rate": 0.08,
                "semantic_alternative_rate": 0.0,
                "effective_adoption_rate": 0.08,
            },
        }

        with (
            patch("roadmap_app.services.rec_recommend", return_value=rec_rows),
            patch("roadmap_app.sku_ranking._recent_recommended_product_health_map", return_value=health_map),
        ):
            ranked = _recommend_candidates_for_type(
                user=self.user,
                category="haircare",
                product_type="shampoo",
                context_product_ids=[],
                context_products=[],
                owned_product_ids=set(),
                used_recommended_ids=set(),
                prof=prof,
                products_for_recs=[dominant_shampoo, healthy_shampoo],
                co_map={},
            )

        self.assertEqual(int(ranked[0]["product"]["id"]), int(healthy_shampoo["id"]))
        sku_health = ranked[1]["components"]["roadmap_rerank"]["sku_health"]
        self.assertTrue(bool(sku_health.get("eligible")))
        self.assertLess(float(sku_health.get("penalty") or 0.0), 0.0)
        self.assertIn("low recent SKU adoption signal", list(ranked[1]["why"]))

    @override_settings(
        ROADMAP_SKU_HEALTH_PENALTY_ENABLED=True,
        ROADMAP_SKU_HEALTH_PENALTY_CATEGORIES=["haircare"],
        ROADMAP_SKU_HEALTH_PENALTY_PRODUCT_TYPES=["shampoo"],
        ROADMAP_SKU_HEALTH_PENALTY_WINDOW_DAYS=30,
        ROADMAP_SKU_HEALTH_PENALTY_CACHE_TTL_SECONDS=60,
        ROADMAP_SKU_HEALTH_PENALTY_INCLUDE_GA=True,
        ROADMAP_SKU_HEALTH_PENALTY_SEMANTIC_WEIGHT=0.25,
    )
    def test_recent_recommended_product_health_map_aggregates_next_step_only_instances(self):
        from django.core.cache import cache

        cache.clear()
        base = timezone.now() - timedelta(days=1)
        rec_bad = Product.objects.create(
            name="Bad Shampoo",
            brand="RankLab",
            price=Decimal("100.00"),
            category="haircare",
            product_type="shampoo",
            in_stock=True,
        )
        rec_good = Product.objects.create(
            name="Good Shampoo",
            brand="RankLab",
            price=Decimal("100.00"),
            category="haircare",
            product_type="shampoo",
            in_stock=True,
        )

        users = [
            get_user_model().objects.create_user(username=f"sku_health_u{i}", password="pass12345")
            for i in range(1, 4)
        ]
        plan_bad_1 = RoadmapPlan.objects.create(user=users[0], category="haircare", is_active=True)
        step_bad_1 = RoadmapStep.objects.create(
            plan=plan_bad_1,
            step_index=1,
            product_type="shampoo",
            status=RoadmapStep.Status.RECOMMENDED,
            recommended_product=rec_bad,
        )
        plan_bad_2 = RoadmapPlan.objects.create(user=users[1], category="haircare", is_active=True)
        step_bad_2 = RoadmapStep.objects.create(
            plan=plan_bad_2,
            step_index=1,
            product_type="shampoo",
            status=RoadmapStep.Status.RECOMMENDED,
            recommended_product=rec_bad,
        )
        plan_good = RoadmapPlan.objects.create(user=users[2], category="haircare", is_active=True)
        step_good = RoadmapStep.objects.create(
            plan=plan_good,
            step_index=1,
            product_type="shampoo",
            status=RoadmapStep.Status.RECOMMENDED,
            recommended_product=rec_good,
        )

        for offset_minutes, plan, step, user in [
            (1, plan_bad_1, step_bad_1, users[0]),
            (5, plan_bad_2, step_bad_2, users[1]),
            (9, plan_good, step_good, users[2]),
        ]:
            self._roadmap_event(
                user=user,
                plan=plan,
                step=None,
                event_type=RoadmapEvent.Type.PLAN_REFRESHED,
                created_at=base + timedelta(minutes=offset_minutes),
                context={
                    "plan_id": plan.id,
                    "category": "haircare",
                    "next_step_id": step.id,
                },
            )
            self._roadmap_event(
                user=user,
                plan=plan,
                step=step,
                event_type=RoadmapEvent.Type.STEP_GENERATED,
                created_at=base + timedelta(minutes=offset_minutes + 1),
                context={
                    "plan_id": plan.id,
                    "step_id": step.id,
                    "category": "haircare",
                    "product_type": "shampoo",
                    "recommended_product_id": step.recommended_product_id,
                    "has_recommendation": True,
                },
            )

        self._roadmap_event(
            user=users[2],
            plan=plan_good,
            step=step_good,
            event_type=RoadmapEvent.Type.STEP_COMPLETED,
            created_at=base + timedelta(minutes=11),
            context={
                "category": "haircare",
                "matched_by": "recommended_product_id",
                "match_meta": {
                    "recommended_product_id": rec_good.id,
                    "purchased_product_id": rec_good.id,
                    "purchased_product_type": "shampoo",
                },
            },
        )

        health_map = _recent_recommended_product_health_map(category="haircare", product_type="shampoo")

        self.assertEqual(int(health_map[int(rec_bad.id)]["recommended_steps"]), 2)
        self.assertAlmostEqual(float(health_map[int(rec_bad.id)]["exact_adoption_rate"]), 0.0, places=6)
        self.assertEqual(int(health_map[int(rec_good.id)]["recommended_steps"]), 1)
        self.assertAlmostEqual(float(health_map[int(rec_good.id)]["exact_adoption_rate"]), 1.0, places=6)
