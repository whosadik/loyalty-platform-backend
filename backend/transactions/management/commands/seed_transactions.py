import random
from decimal import Decimal
from datetime import timedelta

from django.core.management.base import BaseCommand
from django.contrib.auth import get_user_model
from django.utils import timezone
from django.db import transaction as db_tx
from django.db.models import Sum

from catalog.models import Product
from transactions.models import Transaction, TransactionItem, OwnedProduct
from loyalty.models import LoyaltyAccount, LoyaltyLedgerEntry, Tier


class Command(BaseCommand):
    help = "Seed synthetic transactions with baskets (2-6 items) to build co-occurrence & realistic metrics."

    def add_arguments(self, parser):
        parser.add_argument("--reset", action="store_true", help="Delete all transactions/items before seeding")
        parser.add_argument("--transactions", type=int, default=500, help="How many transactions to create")
        parser.add_argument("--users", type=int, default=3, help="How many users to use/create")
        parser.add_argument("--min_items", type=int, default=2, help="Min items in basket")
        parser.add_argument("--max_items", type=int, default=6, help="Max items in basket")
        parser.add_argument("--days", type=int, default=90, help="Spread purchases across last N days")
        parser.add_argument("--with_loyalty", action="store_true", help="Also write loyalty ledger earn entries")
        parser.add_argument("--recalc_tiers", action="store_true", help="Recalculate tiers at end (based on spend 90d)")

    def handle(self, *args, **opt):
        User = get_user_model()

        if opt["reset"]:
            TransactionItem.objects.all().delete()
            Transaction.objects.all().delete()
            OwnedProduct.objects.all().delete()
            # ledger не трогаем (можно отдельно чистить при необходимости)
            self.stdout.write(self.style.WARNING("Deleted transactions, items, owned_products."))

        # products pool
        products = list(Product.objects.filter(in_stock=True).values("id", "category", "product_type", "price"))
        if len(products) < 30:
            self.stdout.write(self.style.ERROR("Not enough products. Run seed_products first."))
            return

        by_cat = {}
        by_cat_type = {}
        for p in products:
            by_cat.setdefault(p["category"], []).append(p)
            by_cat_type.setdefault((p["category"], p["product_type"]), []).append(p)

        # ensure tiers exist
        Tier.objects.get_or_create(name="Bronze", defaults={"threshold_spend_90d": 0, "points_rate": 1.0})
        Tier.objects.get_or_create(name="Silver", defaults={"threshold_spend_90d": 100, "points_rate": 1.2})
        Tier.objects.get_or_create(name="Gold", defaults={"threshold_spend_90d": 250, "points_rate": 1.5})

        # users
        users = list(User.objects.all().order_by("id")[: opt["users"]])
        while len(users) < opt["users"]:
            idx = len(users) + 1
            u = User.objects.create_user(username=f"demo{idx}", password="demo12345")
            users.append(u)

        # ensure loyalty accounts
        bronze = Tier.objects.get(name="Bronze")
        for u in users:
            acc, _ = LoyaltyAccount.objects.get_or_create(user=u)
            if acc.tier_id is None:
                acc.tier = bronze
                acc.save(update_fields=["tier"])

        def rand_dt():
            # random datetime in last N days
            return timezone.now() - timedelta(days=random.randint(0, opt["days"]), hours=random.randint(0, 23), minutes=random.randint(0, 59))

        # category weights (tune as you like)
        cat_weights = [
            ("skincare", 0.35),
            ("makeup", 0.25),
            ("haircare", 0.20),
            ("fragrance", 0.20),
        ]

        def pick_category():
            r = random.random()
            s = 0.0
            for c, w in cat_weights:
                s += w
                if r <= s:
                    return c
            return "skincare"

        def choose_basket():
            """
            Делает корзину 2-6 товаров + иногда добавляет “типичные пары” для сильного co-occurrence.
            """
            n = random.randint(opt["min_items"], opt["max_items"])
            basket = []

            # 60%: start from one category, then mix
            base_cat = pick_category() if random.random() < 0.6 else None

            def add_from(cat, ptype=None):
                pool = by_cat_type.get((cat, ptype)) if ptype else by_cat.get(cat)
                if not pool:
                    return
                basket.append(random.choice(pool))

            # add typical pairs with some probability
            if random.random() < 0.35:
                # skincare trio
                add_from("skincare", "cleanser")
                add_from("skincare", "moisturizer")
                if random.random() < 0.7:
                    add_from("skincare", "spf")
            if random.random() < 0.25:
                # makeup pair
                add_from("makeup", "lipstick")
                add_from("makeup", "mascara")
            if random.random() < 0.20:
                # haircare pair
                add_from("haircare", "shampoo")
                add_from("haircare", "conditioner")
            if random.random() < 0.20:
                # fragrance pair
                add_from("fragrance", random.choice(["edp", "edt"]))
                if random.random() < 0.5:
                    add_from("fragrance", "body_mist")

            # fill remaining
            while len(basket) < n:
                cat = base_cat if base_cat and random.random() < 0.65 else pick_category()
                add_from(cat)

            # unique by product id
            uniq = {}
            for p in basket:
                uniq[p["id"]] = p
            return list(uniq.values())

        created = 0

        for _ in range(opt["transactions"]):
            user = random.choice(users)
            when = rand_dt()
            basket = choose_basket()
            channel = random.choice(["offline", "online"])

            with db_tx.atomic():
                txn = Transaction.objects.create(user=user, channel=channel)
                # принудительно проставим created_at (если поле auto_now_add)
                Transaction.objects.filter(id=txn.id).update(created_at=when)

                total = Decimal("0")
                for p in basket:
                    qty = random.randint(1, 2)
                    price = p["price"]
                    unit_price = Decimal(str(price)) if price is not None else Decimal(str(random.choice([9.99, 12.99, 19.99, 29.99])))

                    TransactionItem.objects.create(
                        transaction=txn,
                        product_id=p["id"],
                        quantity=qty,
                        unit_price=unit_price,
                    )
                    total += unit_price * qty

                    owned, _ = OwnedProduct.objects.get_or_create(user=user, product_id=p["id"])
                    owned.quantity_total = int(owned.quantity_total or 0) + qty
                    owned.is_active = True
                    owned.last_acquired_at = when
                    owned.save(update_fields=["quantity_total", "is_active", "last_acquired_at"])

                Transaction.objects.filter(id=txn.id).update(total_amount=total)

                if opt["with_loyalty"]:
                    acc = LoyaltyAccount.objects.select_for_update().get(user=user)
                    points_rate = Decimal(str(acc.tier.points_rate if acc.tier else 1.0))
                    earned = int(round(float(total * points_rate)))

                    LoyaltyLedgerEntry.objects.create(
                        account=acc,
                        entry_type=LoyaltyLedgerEntry.Type.EARN,
                        points_delta=earned,
                        reference=f"seed:txn:{txn.id}",
                        meta={"txn_id": txn.id, "total": str(total), "points_rate": str(points_rate)},
                    )
                    acc.points_balance += earned
                    acc.save(update_fields=["points_balance"])

            created += 1

        if opt["recalc_tiers"]:
            now = timezone.now()
            since = now - timedelta(days=90)
            tiers = list(Tier.objects.all().values("id", "threshold_spend_90d"))
            tiers.sort(key=lambda t: Decimal(str(t["threshold_spend_90d"])))

            for u in users:
                spend = (
                    Transaction.objects.filter(user=u, created_at__gte=since)
                    .aggregate(s=Sum("total_amount"))["s"]
                    or Decimal("0")
                )
                chosen_id = tiers[0]["id"]
                for t in tiers:
                    if spend >= Decimal(str(t["threshold_spend_90d"])):
                        chosen_id = t["id"]

                LoyaltyAccount.objects.filter(user=u).update(tier_id=chosen_id)

        self.stdout.write(self.style.SUCCESS(f"Created {created} transactions for {len(users)} users."))
        self.stdout.write(self.style.SUCCESS("Now bundle/co-occurrence should start returning results."))
