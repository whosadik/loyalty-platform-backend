import random
from decimal import Decimal
from datetime import timedelta

from django.core.management.base import BaseCommand
from django.core.management import call_command
from django.db import transaction as db_tx
from django.utils import timezone
from django.contrib.auth import get_user_model

from catalog.models import Product
from transactions.models import Transaction, TransactionItem, OwnedProduct
from users_app.models import CustomerProfile

from loyalty.models import LoyaltyAccount, Tier
from loyalty.models import LoyaltyLedgerEntry

from offers.models import CampaignBudget
from offers.services import get_or_assign_next_offer


DEMO_PREFIX = "demo"


def pick_product(pool, fallback_pool):
    if pool:
        return random.choice(pool)
    return random.choice(fallback_pool) if fallback_pool else None


class Command(BaseCommand):
    help = "Seed demo users, profiles, transactions, owned products, points, and warm recs cache."

    def add_arguments(self, parser):
        parser.add_argument("--users", type=int, default=30)
        parser.add_argument("--txns", type=int, default=300)
        parser.add_argument("--days", type=int, default=90)
        parser.add_argument("--prefix", type=str, default=DEMO_PREFIX)
        parser.add_argument("--clear", action="store_true", help="Delete demo users (prefix_*) before seeding")
        parser.add_argument("--warm-cache", action="store_true", help="Run build_recs_cache after seeding")
        parser.add_argument("--assign-offers", action="store_true", help="Assign next offer for each demo user")

    def create_ledger_entry(*, user, account, delta: int, kind: str, meta: dict, created_at):
        M = LoyaltyLedgerEntry
        fields = {f.name for f in M._meta.fields}

        data = {}

        # owner
        if "user" in fields:
            data["user"] = user
        if "account" in fields:
            data["account"] = account
        elif "loyalty_account" in fields:
            data["loyalty_account"] = account

        # type/kind
        if "event_type" in fields:
            data["event_type"] = kind
        elif "entry_type" in fields:
            data["entry_type"] = kind
        elif "kind" in fields:
            data["kind"] = kind
        elif "action" in fields:
            data["action"] = kind
        elif "type" in fields:
            data["type"] = kind

        # delta/points
        if "points_delta" in fields:
            data["points_delta"] = delta
        elif "delta" in fields:
            data["delta"] = delta
        elif "points" in fields:
            data["points"] = delta
        elif "amount" in fields:
            data["amount"] = delta

        # meta
        if "meta" in fields:
            data["meta"] = meta

        # created_at: если поле существует и не auto_now_add — выставим
        if "created_at" in fields:
            f = M._meta.get_field("created_at")
            if not getattr(f, "auto_now_add", False) and not getattr(f, "auto_now", False):
                data["created_at"] = created_at

        return M.objects.create(**data)
    def ledger_add(account, entry_type, delta: int, reference: str, meta: dict, created_at):
        entry = LoyaltyLedgerEntry.objects.create(
            account=account,
            entry_type=entry_type,
            points_delta=delta,
            reference=reference,
            meta=meta,
        )
        LoyaltyLedgerEntry.objects.filter(id=entry.id).update(created_at=created_at)
        return entry
    
    def handle(self, *args, **opt):
        users_n = int(opt["users"])
        txns_n = int(opt["txns"])
        days = int(opt["days"])
        prefix = opt["prefix"]

        User = get_user_model()

        if opt["clear"]:
            demo_users = User.objects.filter(username__startswith=f"{prefix}_")
            cnt = demo_users.count()
            demo_users.delete()
            self.stdout.write(self.style.WARNING(f"Deleted {cnt} demo users (username startswith {prefix}_)"))

        products = list(Product.objects.filter(in_stock=True))
        if not products:
            self.stdout.write(self.style.ERROR("No products in catalog. Seed catalog first."))
            return

        # pools by (category, product_type)
        by_cat_pt = {}
        for p in products:
            by_cat_pt.setdefault((p.category, p.product_type), []).append(p)

        def pool(cat, pt):
            return by_cat_pt.get((cat, pt), [])

        # “паттерны корзин” для сильного co-occurrence
        bundle_patterns = [
            # skincare
            [("skincare", "cleanser"), ("skincare", "serum"), ("skincare", "moisturizer"), ("skincare", "spf")],
            # makeup
            [("makeup", "foundation"), ("makeup", "mascara"), ("makeup", "blush")],
            [("makeup", "lipstick"), ("makeup", "blush"), ("makeup", "eyeshadow")],
            # haircare
            [("haircare", "shampoo"), ("haircare", "conditioner")],
            [("haircare", "conditioner"), ("haircare", "hair_mask")],
            # fragrance
            [("fragrance", "edp"), ("fragrance", "body_mist")],
            [("fragrance", "edt"), ("fragrance", "body_mist")],
        ]

        skin_types = ["dry", "oily", "combination", "normal", "sensitive"]
        goals = ["acne", "hydration", "anti_aging", "brightening", "soothing"]
        avoid_flags = ["alcohol", "fragrance", "silicones", "parabens"]

        hair_types = ["straight", "wavy", "curly", "coily"]
        hair_concerns = ["frizz", "dryness", "oiliness", "damage"]
        makeup_prefs = ["matte", "dewy", "long_wear", "waterproof"]
        frag_families = ["fresh", "floral", "woody", "oriental", "gourmand"]

        # Ensure tier + budget exist
        bronze, _ = Tier.objects.get_or_create(name="Bronze", defaults={"threshold_spend_90d": 0, "points_rate": 1.0})
        b = CampaignBudget.objects.filter(name="default").first()
        if b:
            # чтобы assign-offers не упёрся в бюджет при демо
            b.weekly_limit = Decimal("100000.0")
            b.save(update_fields=["weekly_limit"])

        # create users + profiles
        demo_users = []
        for i in range(1, users_n + 1):
            username = f"{prefix}_{i:03d}"
            u, created = User.objects.get_or_create(username=username, defaults={"is_active": True})
            if created:
                u.set_password("demo12345")
                u.save(update_fields=["password"])

            cp, _ = CustomerProfile.objects.get_or_create(user=u)
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
            cp.save()

            la, _ = LoyaltyAccount.objects.get_or_create(user=u, defaults={"tier": bronze, "points_balance": 0})
            if la.tier_id is None:
                la.tier = bronze
                la.save(update_fields=["tier"])

            demo_users.append(u)

        self.stdout.write(self.style.SUCCESS(f"Created/updated {len(demo_users)} demo users + profiles"))

        # generate transactions
        now = timezone.now()
        for t in range(txns_n):
            user = random.choice(demo_users)
            created_at = now - timedelta(days=random.randint(0, days), hours=random.randint(0, 23), minutes=random.randint(0, 59))
            channel = random.choice(["offline", "online"])

            # pick pattern to strengthen co-occurrence
            pat = random.choice(bundle_patterns)
            chosen_products = []
            for (cat, pt) in pat:
                p = pick_product(pool(cat, pt), products)
                if p:
                    chosen_products.append(p)

            # add noise items sometimes
            if random.random() < 0.3:
                for _ in range(random.randint(1, 2)):
                    chosen_products.append(random.choice(products))

            # unique and limit
            chosen_products = list({p.id: p for p in chosen_products}.values())[:5]
            if not chosen_products:
                chosen_products = [random.choice(products)]

            with db_tx.atomic():
                txn = Transaction.objects.create(
                    user=user,
                    channel=channel,
                    total_amount=Decimal("0.00"),
                    created_at=created_at,
                )

                total = Decimal("0.00")
                item_payload = []
                for p in chosen_products:
                    qty = 1 if random.random() < 0.9 else 2
                    unit_price = Decimal(str(p.price))
                    total += unit_price * qty
                    TransactionItem.objects.create(
                        transaction=txn,
                        product=p,
                        quantity=qty,
                        unit_price=unit_price,
                    )
                    item_payload.append((p, qty, unit_price))

                    # owned product
                    # owned product (upsert because (user, product) is unique)
                    defaults = {}
                    if hasattr(OwnedProduct, "acquired_at"):
                        defaults["acquired_at"] = created_at
                    if hasattr(OwnedProduct, "source"):
                        defaults["source"] = "transaction"
                    if hasattr(OwnedProduct, "is_active"):
                        defaults["is_active"] = True

                    op, created_op = OwnedProduct.objects.get_or_create(
                        user=user,
                        product=p,
                        defaults=defaults,
                    )

                    # если уже был — обновим полезные поля (например, last acquired_at)
                    updates = {}
                    if not created_op:
                        if hasattr(OwnedProduct, "acquired_at"):
                            cur = getattr(op, "acquired_at", None)
                            if cur is None or cur < created_at:
                                updates["acquired_at"] = created_at
                        if hasattr(OwnedProduct, "is_active") and getattr(op, "is_active", True) is False:
                            updates["is_active"] = True
                        if hasattr(OwnedProduct, "source") and (not getattr(op, "source", "")):
                            updates["source"] = "transaction"

                        if updates:
                            OwnedProduct.objects.filter(id=op.id).update(**updates)


                txn.total_amount = total
                txn.save(update_fields=["total_amount"])

                # points earn (simple)
                la = LoyaltyAccount.objects.select_for_update().get(user=user)
                rate = Decimal(str(la.tier.points_rate if la.tier else 1.0))
                points = int(round(float(total * rate)))

                # ledger + balance
                entry = LoyaltyLedgerEntry.objects.create(
                    account=la,
                    entry_type=LoyaltyLedgerEntry.Type.EARN,     # важно: это "earn", не "EARN"
                    points_delta=points,
                    reference=f"txn:{txn.id}",
                    meta={"source": "seed_demo", "transaction_id": txn.id},
                )

                # created_at у тебя auto_now_add=True, поэтому на create не задаётся
                # но можно проставить постфактум (для реалистичных 90 дней)
                LoyaltyLedgerEntry.objects.filter(id=entry.id).update(created_at=created_at)

                la.points_balance = int(la.points_balance) + points
                la.save(update_fields=["points_balance"])

        self.stdout.write(self.style.SUCCESS(f"Generated {txns_n} transactions + items + owned products + points"))

        # warm cache
        if opt["warm_cache"]:
            call_command("build_recs_cache", clear=True)
            self.stdout.write(self.style.SUCCESS("Warmed recs cache (products + cooccurrence)"))

        # assign next offers (post_ctx from last txn items)
        if opt["assign_offers"]:
            for u in demo_users:
                last_txn = Transaction.objects.filter(user=u).order_by("-created_at").first()
                if not last_txn:
                    continue
                items = list(TransactionItem.objects.filter(transaction=last_txn).select_related("product"))
                post_ctx = {
                    "product_ids": [it.product_id for it in items],
                    "categories": sorted({it.product.category for it in items}),
                    "product_types": sorted({it.product.product_type for it in items}),
                }
                with db_tx.atomic():
                    get_or_assign_next_offer(user=u, now=timezone.now(), context_steps=None, post_ctx=post_ctx)

            self.stdout.write(self.style.SUCCESS(f"Assigned next offers for {len(demo_users)} demo users"))

        self.stdout.write(self.style.SUCCESS("seed_demo done"))
