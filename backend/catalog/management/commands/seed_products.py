import random
from django.core.management.base import BaseCommand

from catalog.models import Product


class Command(BaseCommand):
    help = "Seed demo products for routine builder / recommendations"

    def handle(self, *args, **options):
        if Product.objects.exists():
            self.stdout.write(self.style.WARNING("Products already exist. Skipping."))
            return

        # Набор "активов" и "флагов" для демо
        actives_pool = ["bha", "aha", "retinoid", "vitamin_c", "niacinamide", "ceramides", "hyaluronic"]
        flags_pool = ["fragrance", "alcohol", "essential_oils"]

        skin_types = ["dry", "oily", "combination", "sensitive", "normal"]

        # (step, count, typical_actives)
        plan = [
            ("cleanser", 20, []),
            ("moisturizer", 20, ["ceramides", "hyaluronic", "niacinamide"]),
            ("spf", 15, []),
            ("serum", 30, actives_pool),
            ("toner", 10, ["aha", "bha"]),
            ("mask", 10, ["aha", "bha", "hyaluronic"]),
        ]

        brands = ["DermaLab", "Glowify", "SkinNova", "CosmoCare", "PureDerm", "Aurum"]

        created = 0
        for step, count, typical_actives in plan:
            for i in range(count):
                brand = random.choice(brands)
                name = f"{step.title()} {i+1}"

                # supported skin types: 2-4 типа
                supported = random.sample(skin_types, k=random.randint(2, 4))

                # actives: для сывороток/тоников/масок - иногда
                actives = []
                if typical_actives and random.random() < 0.7:
                    k = 1 if random.random() < 0.75 else 2
                    actives = random.sample(typical_actives, k=k)

                # flags: иногда есть "нежелательные"
                flags = []
                if random.random() < 0.25:
                    flags = random.sample(flags_pool, k=1)

                strength = "low"
                if any(a in actives for a in ["aha", "bha", "retinoid", "vitamin_c"]):
                    strength = random.choice(["low", "medium"])

                price = random.choice([9.99, 12.99, 15.99, 19.99, 24.99, 29.99])

                Product.objects.create(
                    name=name,
                    brand=brand,
                    price=price,
                    step=step,
                    actives=actives,
                    flags=flags,
                    supported_skin_types=supported,
                    strength=strength,
                    in_stock=True,
                )
                created += 1

        self.stdout.write(self.style.SUCCESS(f"Seeded {created} products."))
