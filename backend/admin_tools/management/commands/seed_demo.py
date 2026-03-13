import random
from datetime import timedelta
from decimal import Decimal

from django.contrib.auth import get_user_model
from django.core.management import call_command
from django.core.management.base import BaseCommand
from django.db import transaction as db_tx
from django.utils import timezone

from catalog.models import Product
from loyalty.models import LoyaltyAccount, LoyaltyLedgerEntry, Tier
from loyalty.points import DEFAULT_POINTS_RATE, get_effective_points_rate
from offers.models import CampaignBudget, Offer, OfferAssignment
from offers.services import get_or_assign_next_offer
from transactions.models import OwnedProduct, Transaction, TransactionItem
from users_app.models import CustomerProfile


DEMO_PREFIX = "demo"


def pick_product(pool, fallback_pool):
    if pool:
        return random.choice(pool)
    return random.choice(fallback_pool) if fallback_pool else None


class Command(BaseCommand):
    help = "Seed demo users, deterministic scenarios, transactions, owned products and optional offers."

    def add_arguments(self, parser):
        parser.add_argument("--users", type=int, default=30)
        parser.add_argument("--txns", type=int, default=300)
        parser.add_argument("--days", type=int, default=90)
        parser.add_argument("--prefix", type=str, default=DEMO_PREFIX)
        parser.add_argument("--clear", action="store_true", help="Delete demo users (prefix_*) before seeding")
        parser.add_argument("--warm-cache", action="store_true", help="Run build_recs_cache after seeding")
        parser.add_argument("--assign-offers", action="store_true", help="Assign next offer for each demo user")

    def handle(self, *args, **opt):
        users_n = int(opt["users"])
        txns_n = int(opt["txns"])
        days = int(opt["days"])
        prefix = opt["prefix"]

        random.seed(42)
        now = timezone.now()
        User = get_user_model()

        if opt["clear"]:
            demo_users_qs = User.objects.filter(username__startswith=f"{prefix}_")
            cnt = demo_users_qs.count()
            demo_users_qs.delete()
            self.stdout.write(self.style.WARNING(f"Deleted {cnt} demo users (username startswith {prefix}_)"))

        products = list(Product.objects.filter(in_stock=True))
        if not products:
            self.stdout.write(self.style.ERROR("No products in catalog. Seed catalog first."))
            return

        by_cat_pt = {}
        for p in products:
            by_cat_pt.setdefault((p.category, p.product_type), []).append(p)

        def pool(cat, pt):
            return by_cat_pt.get((cat, pt), [])

        def ensure_product(category: str, product_type: str, name: str, price: str = "19.99") -> Product:
            existing = Product.objects.filter(category=category, product_type=product_type, in_stock=True).order_by("id").first()
            if existing:
                return existing
            created = Product.objects.create(
                name=name,
                brand="DEMO",
                price=Decimal(price),
                category=category,
                product_type=product_type,
                concerns=[],
                attrs={},
                actives=[],
                flags=[],
                supported_skin_types=["normal"],
                strength="low",
                in_stock=True,
            )
            products.append(created)
            by_cat_pt.setdefault((created.category, created.product_type), []).append(created)
            return created

        def ensure_campaign(name: str, *, weekly_limit: str, priority: int, allowed_categories=None, allowed_steps=None):
            c, _ = CampaignBudget.objects.get_or_create(
                name=name,
                defaults={
                    "weekly_limit": Decimal(weekly_limit),
                    "weekly_spent": Decimal("0.0"),
                    "priority": priority,
                    "is_active": True,
                    "allowed_categories": allowed_categories or [],
                    "allowed_steps": allowed_steps or [],
                },
            )
            changed = []
            if c.weekly_limit < Decimal(weekly_limit):
                c.weekly_limit = Decimal(weekly_limit)
                changed.append("weekly_limit")
            if c.priority != priority:
                c.priority = priority
                changed.append("priority")
            if not c.is_active:
                c.is_active = True
                changed.append("is_active")
            if (c.allowed_categories or []) != (allowed_categories or []):
                c.allowed_categories = allowed_categories or []
                changed.append("allowed_categories")
            if (c.allowed_steps or []) != (allowed_steps or []):
                c.allowed_steps = allowed_steps or []
                changed.append("allowed_steps")
            if changed:
                c.save(update_fields=changed)
            return c

        def ensure_offer(campaign: CampaignBudget, name: str, **defaults):
            Offer.objects.update_or_create(campaign=campaign, name=name, defaults=defaults)

        bronze, _ = Tier.objects.get_or_create(
            name="Bronze",
            defaults={"threshold_spend_90d": 0, "points_rate": DEFAULT_POINTS_RATE},
        )

        default_campaign = ensure_campaign("default", weekly_limit="100000.0", priority=100)
        onboarding_campaign = ensure_campaign("onboarding_first_order", weekly_limit="50000.0", priority=5)
        winback_campaign = ensure_campaign("winback_30d", weekly_limit="50000.0", priority=10)
        favorite_campaign = ensure_campaign("favorite_category", weekly_limit="50000.0", priority=20)
        makeup_campaign = ensure_campaign("makeup_push", weekly_limit="50000.0", priority=25, allowed_categories=["makeup"])
        fragrance_campaign = ensure_campaign(
            "fragrance_crosssell",
            weekly_limit="50000.0",
            priority=15,
            allowed_categories=["fragrance"],
        )
        haircare_campaign = ensure_campaign("haircare_push", weekly_limit="50000.0", priority=30, allowed_categories=["haircare"])

        ensure_offer(
            default_campaign,
            "[DEMO] Default x2 points",
            offer_type=Offer.Type.POINTS_MULTIPLIER,
            value=Decimal("2.00"),
            estimated_cost=Decimal("3.00"),
            is_active=True,
            target_scope="cart",
            cooldown_days=7,
            expires_in_days=7,
            allowed_categories=[],
            allowed_product_types=[],
        )
        ensure_offer(
            onboarding_campaign,
            "[DEMO] First order -10%",
            offer_type=Offer.Type.DISCOUNT,
            value=Decimal("10.00"),
            estimated_cost=Decimal("8.00"),
            is_active=True,
            target_scope="cart",
            cooldown_days=365,
            expires_in_days=7,
            allowed_categories=[],
            allowed_product_types=[],
        )
        ensure_offer(
            winback_campaign,
            "[DEMO] Win-back -15%",
            offer_type=Offer.Type.DISCOUNT,
            value=Decimal("15.00"),
            estimated_cost=Decimal("12.00"),
            is_active=True,
            target_scope="cart",
            cooldown_days=30,
            expires_in_days=10,
            allowed_categories=[],
            allowed_product_types=[],
        )
        ensure_offer(
            favorite_campaign,
            "[DEMO] Favorite category -12%",
            offer_type=Offer.Type.DISCOUNT,
            value=Decimal("12.00"),
            estimated_cost=Decimal("9.00"),
            is_active=True,
            target_scope="category",
            cooldown_days=14,
            expires_in_days=10,
            allowed_categories=[],
            allowed_product_types=[],
        )
        ensure_offer(
            makeup_campaign,
            "[DEMO] Makeup bundle -8%",
            offer_type=Offer.Type.DISCOUNT,
            value=Decimal("8.00"),
            estimated_cost=Decimal("6.00"),
            is_active=True,
            target_scope="product_id",
            cooldown_days=3,
            expires_in_days=7,
            allowed_categories=["makeup"],
            allowed_product_types=[],
        )
        ensure_offer(
            fragrance_campaign,
            "[DEMO] Fragrance cross-sell -10%",
            offer_type=Offer.Type.DISCOUNT,
            value=Decimal("10.00"),
            estimated_cost=Decimal("7.00"),
            is_active=True,
            target_scope="product_id",
            cooldown_days=3,
            expires_in_days=7,
            allowed_categories=["fragrance"],
            allowed_product_types=[],
        )
        ensure_offer(
            haircare_campaign,
            "[DEMO] Haircare bundle -8%",
            offer_type=Offer.Type.DISCOUNT,
            value=Decimal("8.00"),
            estimated_cost=Decimal("6.00"),
            is_active=True,
            target_scope="product_id",
            cooldown_days=3,
            expires_in_days=7,
            allowed_categories=["haircare"],
            allowed_product_types=[],
        )

        skin_types = ["dry", "oily", "combination", "normal", "sensitive"]
        goals = ["acne", "hydration", "anti_aging", "brightening", "soothing"]
        avoid_flags = ["alcohol", "fragrance", "silicones", "parabens"]
        hair_types = ["straight", "wavy", "curly", "coily"]
        hair_concerns = ["frizz", "dryness", "oiliness", "damage"]
        makeup_prefs = ["matte", "dewy", "long_wear", "waterproof"]
        frag_families = ["fresh", "floral", "woody", "oriental", "gourmand"]

        def ensure_loyalty(user):
            la, _ = LoyaltyAccount.objects.get_or_create(user=user, defaults={"tier": bronze, "points_balance": 0})
            changed = []
            if la.tier_id is None:
                la.tier = bronze
                changed.append("tier")
            if changed:
                la.save(update_fields=changed)
            return la

        def ensure_profile(user, randomize: bool = False):
            cp, _ = CustomerProfile.objects.get_or_create(user=user)
            if randomize:
                cp.skin_type = random.choice(skin_types)
                cp.goals = random.sample(goals, k=random.randint(1, 2))
                cp.avoid_flags = random.sample(avoid_flags, k=random.randint(0, 2))
                cp.budget = random.choice(["low", "medium", "high"])
                cp.hair_profile = {
                    "hair_type": random.choice(hair_types),
                    "scalp_type": random.choice(["dry", "normal", "oily"]),
                    "hair_thickness": random.choice(["fine", "medium", "thick"]),
                    "concerns": random.sample(hair_concerns, k=random.randint(1, 2)),
                }
                cp.makeup_profile = {
                    "finish_pref": random.sample(["matte", "dewy", "natural"], k=random.randint(1, 2)),
                    "coverage_pref": random.sample(["light", "medium", "full"], k=random.randint(1, 2)),
                    "concerns": random.sample(makeup_prefs, k=random.randint(1, 2)),
                }
                cp.fragrance_profile = {
                    "liked_families": random.sample(frag_families, k=random.randint(1, 2)),
                    "disliked_families": random.sample(frag_families, k=random.randint(0, 1)),
                    "intensity_pref": random.choice(["light", "medium", "strong"]),
                }
            else:
                cp.skin_type = "normal"
                cp.goals = ["hydration"]
                cp.avoid_flags = []
                cp.budget = "medium"
                cp.hair_profile = {"hair_type": "straight", "concerns": ["dryness"]}
                cp.makeup_profile = {"finish_pref": ["natural"], "coverage_pref": ["medium"]}
                cp.fragrance_profile = {"liked_families": ["fresh"], "intensity_pref": "medium"}
            cp.save()

        def reset_user_state(user):
            OfferAssignment.objects.filter(user=user).delete()
            Transaction.objects.filter(user=user).delete()
            OwnedProduct.objects.filter(user=user).delete()
            la = ensure_loyalty(user)
            LoyaltyLedgerEntry.objects.filter(account=la).delete()
            la.points_balance = 0
            la.tier = bronze
            la.save(update_fields=["points_balance", "tier"])

        def upsert_owned(user, product, qty, acquired_at):
            op, created_op = OwnedProduct.objects.get_or_create(
                user=user,
                product=product,
                defaults={"acquired_at": acquired_at, "source": "transaction", "is_active": True},
            )
            op.quantity_total = int(op.quantity_total or 0) + qty
            op.is_active = True
            op.last_acquired_at = acquired_at
            updates = ["quantity_total", "is_active", "last_acquired_at"]
            if hasattr(op, "acquired_at") and (created_op or op.acquired_at is None or op.acquired_at < acquired_at):
                op.acquired_at = acquired_at
                updates.append("acquired_at")
            op.save(update_fields=list(dict.fromkeys(updates)))

        def create_seed_transaction(user, *, created_at, channel, items):
            with db_tx.atomic():
                txn = Transaction.objects.create(user=user, channel=channel, total_amount=Decimal("0.00"))
                total = Decimal("0.00")
                for prod, qty in items:
                    qty = int(qty)
                    unit_price = Decimal(str(prod.price))
                    TransactionItem.objects.create(
                        transaction=txn,
                        product=prod,
                        quantity=qty,
                        unit_price=unit_price,
                    )
                    total += unit_price * qty
                    upsert_owned(user, prod, qty, acquired_at=created_at)

                txn.total_amount = total
                txn.save(update_fields=["total_amount"])
                Transaction.objects.filter(id=txn.id).update(created_at=created_at)

                la = LoyaltyAccount.objects.select_for_update().get(user=user)
                rate = get_effective_points_rate(
                    la.tier.points_rate if la.tier else DEFAULT_POINTS_RATE
                )
                points = int(round(float(total * rate)))
                entry = LoyaltyLedgerEntry.objects.create(
                    account=la,
                    entry_type=LoyaltyLedgerEntry.Type.EARN,
                    points_delta=points,
                    reference=f"txn:{txn.id}",
                    meta={"source": "seed_demo", "transaction_id": txn.id},
                )
                LoyaltyLedgerEntry.objects.filter(id=entry.id).update(created_at=created_at)
                la.points_balance = int(la.points_balance) + points
                la.save(update_fields=["points_balance"])
                return txn

        # random demo users
        demo_users = []
        for i in range(1, users_n + 1):
            username = f"{prefix}_{i:03d}"
            user, created = User.objects.get_or_create(username=username, defaults={"is_active": True})
            if created:
                user.set_password("demo12345")
                user.save(update_fields=["password"])
            ensure_profile(user, randomize=True)
            ensure_loyalty(user)
            demo_users.append(user)

        self.stdout.write(self.style.SUCCESS(f"Created/updated {len(demo_users)} random demo users + profiles"))

        bundle_patterns = [
            [("skincare", "cleanser"), ("skincare", "serum"), ("skincare", "moisturizer"), ("skincare", "spf")],
            [("makeup", "foundation"), ("makeup", "mascara"), ("makeup", "blush")],
            [("makeup", "lipstick"), ("makeup", "blush"), ("makeup", "eyeshadow")],
            [("haircare", "shampoo"), ("haircare", "conditioner")],
            [("haircare", "conditioner"), ("haircare", "hair_mask")],
            [("fragrance", "edp"), ("fragrance", "body_mist")],
            [("fragrance", "edt"), ("fragrance", "body_mist")],
        ]

        # random transactions
        for _ in range(txns_n):
            user = random.choice(demo_users)
            created_at = now - timedelta(days=random.randint(0, days), hours=random.randint(0, 23), minutes=random.randint(0, 59))
            channel = random.choice(["offline", "online"])

            pat = random.choice(bundle_patterns)
            chosen_products = []
            for cat, pt in pat:
                p = pick_product(pool(cat, pt), products)
                if p:
                    chosen_products.append(p)

            if random.random() < 0.3:
                for _ in range(random.randint(1, 2)):
                    chosen_products.append(random.choice(products))

            unique = []
            seen = set()
            for p in chosen_products:
                if p.id in seen:
                    continue
                seen.add(p.id)
                unique.append(p)
            unique = unique[:5] or [random.choice(products)]

            items = [(p, 1 if random.random() < 0.9 else 2) for p in unique]
            create_seed_transaction(user, created_at=created_at, channel=channel, items=items)

        self.stdout.write(self.style.SUCCESS(f"Generated {txns_n} random transactions + items + owned products + points"))

        # deterministic scenario users
        scenario_users = []

        def ensure_scenario_user(suffix: str):
            username = f"{prefix}_{suffix}"
            user, created = User.objects.get_or_create(username=username, defaults={"is_active": True})
            if created:
                user.set_password("demo12345")
                user.save(update_fields=["password"])
            ensure_profile(user, randomize=False)
            ensure_loyalty(user)
            reset_user_state(user)
            scenario_users.append(user)
            return user

        p_foundation = ensure_product("makeup", "foundation", "Scenario Foundation")
        p_mascara = ensure_product("makeup", "mascara", "Scenario Mascara")
        p_lipstick = ensure_product("makeup", "lipstick", "Scenario Lipstick")
        p_blush = ensure_product("makeup", "blush", "Scenario Blush")
        p_edp = ensure_product("fragrance", "edp", "Scenario EDP")
        p_body_mist = ensure_product("fragrance", "body_mist", "Scenario Body Mist")
        p_shampoo = ensure_product("haircare", "shampoo", "Scenario Shampoo")
        p_conditioner = ensure_product("haircare", "conditioner", "Scenario Conditioner")
        p_cleanser = ensure_product("skincare", "cleanser", "Scenario Cleanser")
        p_serum = ensure_product("skincare", "serum", "Scenario Serum")

        # 1) first order: no txns
        ensure_scenario_user("first_order")

        # 2) winback: last txn 40 days ago
        u_winback = ensure_scenario_user("winback")
        create_seed_transaction(
            u_winback,
            created_at=now - timedelta(days=40),
            channel="offline",
            items=[(p_cleanser, 1), (p_serum, 1)],
        )

        # 3) favorite makeup: makeup-heavy in last 90 days
        u_fav = ensure_scenario_user("favorite_makeup")
        create_seed_transaction(u_fav, created_at=now - timedelta(days=20), channel="offline", items=[(p_foundation, 2), (p_mascara, 1)])
        create_seed_transaction(u_fav, created_at=now - timedelta(days=10), channel="offline", items=[(p_lipstick, 2), (p_blush, 1)])
        create_seed_transaction(u_fav, created_at=now - timedelta(days=6), channel="offline", items=[(p_foundation, 1), (p_mascara, 1)])
        create_seed_transaction(u_fav, created_at=now - timedelta(days=5), channel="offline", items=[(p_cleanser, 1)])

        # 4) bundle foundation -> mascara tendency
        u_bundle_foundation = ensure_scenario_user("bundle_foundation")
        create_seed_transaction(
            u_bundle_foundation,
            created_at=now - timedelta(days=2),
            channel="offline",
            items=[(p_foundation, 1)],
        )

        # 5) bundle edp -> body_mist tendency
        u_bundle_edp = ensure_scenario_user("bundle_edp")
        create_seed_transaction(
            u_bundle_edp,
            created_at=now - timedelta(days=2),
            channel="offline",
            items=[(p_edp, 1)],
        )

        # 6) bundle shampoo -> conditioner tendency
        u_bundle_shampoo = ensure_scenario_user("bundle_shampoo")
        create_seed_transaction(
            u_bundle_shampoo,
            created_at=now - timedelta(days=2),
            channel="offline",
            items=[(p_shampoo, 1)],
        )

        self.stdout.write(
            self.style.SUCCESS(
                "Scenario users ready: "
                f"{prefix}_first_order, {prefix}_winback, {prefix}_favorite_makeup, "
                f"{prefix}_bundle_foundation, {prefix}_bundle_edp, {prefix}_bundle_shampoo"
            )
        )

        all_demo_users = demo_users + scenario_users

        if opt["warm_cache"]:
            call_command("build_recs_cache", clear=True)
            self.stdout.write(self.style.SUCCESS("Warmed recs cache (products + cooccurrence)"))

        if opt["assign_offers"]:
            assigned = 0
            for u in all_demo_users:
                last_txn = Transaction.objects.filter(user=u).order_by("-created_at").first()
                post_ctx = None
                if last_txn:
                    items = list(TransactionItem.objects.filter(transaction=last_txn).select_related("product"))
                    post_ctx = {
                        "product_ids": [it.product_id for it in items],
                        "categories": sorted({it.product.category for it in items}),
                        "product_types": sorted({it.product.product_type for it in items}),
                    }

                # Keep scenario assignment deterministic:
                # favorite_category should be assigned from baseline next-offer flow (without post_ctx).
                if u.username == f"{prefix}_favorite_makeup":
                    post_ctx = None

                with db_tx.atomic():
                    a = get_or_assign_next_offer(user=u, now=timezone.now(), context_steps=None, post_ctx=post_ctx)
                    if a:
                        assigned += 1

            self.stdout.write(self.style.SUCCESS(f"Assigned next offers for {assigned} users"))

        self.stdout.write(self.style.SUCCESS("seed_demo done"))
