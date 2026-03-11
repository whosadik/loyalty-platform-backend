import random

from django.core.management.base import BaseCommand

from catalog.models import Product


ALL_SKIN_TYPES = ["dry", "oily", "combination", "sensitive", "normal"]


class Command(BaseCommand):
    help = "Seed demo products across categories WITHOUT deleting existing data (safe reseed)."

    def add_arguments(self, parser):
        parser.add_argument(
            "--only",
            nargs="*",
            choices=["skincare", "haircare", "makeup", "fragrance"],
            help="Seed only selected categories (default: seed missing categories).",
        )

    def handle(self, *args, **options):
        random.seed(42)

        brands = ["DermaLab", "Glowify", "SkinNova", "CosmoCare", "PureDerm", "Aurum"]

        def mk_common():
            price = random.choice([9.99, 12.99, 15.99, 19.99, 24.99, 29.99, 39.99])
            raw_meta = {
                "rating": f"{random.choice([4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 4.8]):.1f}",
                "reviews_count": random.randint(12, 120),
            }
            if random.random() < 0.25:
                discount = random.choice([10, 15, 20, 25, 30])
                original_price = round(price / (1 - discount / 100), 2)
                raw_meta.update({
                    "original_price": f"{original_price:.2f}",
                    "discount": discount,
                })
            return {
                "brand": random.choice(brands),
                "price": price,
                "flags": random.sample(["fragrance", "alcohol", "essential_oils"], k=1) if random.random() < 0.2 else [],
                "in_stock": True,
                "actives": [],
                "strength": "low",
                "supported_skin_types": [],
                "concerns": [],
                "attrs": {},
                "step": "",
                "raw_meta": raw_meta,
            }

        existing_categories = set(Product.objects.values_list("category", flat=True).distinct())
        requested = options.get("only") or []

        if requested:
            categories_to_seed = [c for c in requested if c not in existing_categories]
        else:
            categories_to_seed = [c for c in ["skincare", "haircare", "makeup", "fragrance"] if c not in existing_categories]

        if not categories_to_seed:
            self.stdout.write(self.style.WARNING(f"Nothing to seed. Existing categories: {sorted(existing_categories)}"))
            return

        created = 0

        def safe_create(category: str, product_type: str, name: str, defaults: dict):
            nonlocal created
            obj, was_created = Product.objects.get_or_create(
                category=category,
                product_type=product_type,
                name=name,
                defaults=defaults,
            )
            if was_created:
                created += 1
            return obj, was_created

        if "skincare" in categories_to_seed:
            skincare_types = ["cleanser", "moisturizer", "spf", "serum", "toner", "mask"]
            actives_pool = ["bha", "aha", "retinoid", "vitamin_c", "niacinamide", "ceramides", "hyaluronic"]
            skincare_concerns = ["acne", "hydration", "anti_aging", "sensitivity", "brightening"]

            for pt in skincare_types:
                count = 25 if pt in {"cleanser", "moisturizer", "serum"} else 15
                for i in range(count):
                    common = mk_common()

                    actives = []
                    if pt in {"serum", "toner", "mask"} and random.random() < 0.7:
                        actives = random.sample(actives_pool, k=1 if random.random() < 0.8 else 2)

                    strength = "low"
                    if any(a in actives for a in ["aha", "bha", "retinoid", "vitamin_c"]):
                        strength = random.choice(["low", "medium"])

                    defaults = {
                        **common,
                        "brand": random.choice(brands),
                        "price": common["price"],
                        "category": "skincare",
                        "product_type": pt,
                        "step": pt,
                        "actives": actives,
                        "supported_skin_types": random.sample(ALL_SKIN_TYPES, k=random.randint(2, 5)),
                        "strength": strength,
                        "concerns": random.sample(skincare_concerns, k=random.randint(1, 2)),
                        "attrs": {},
                    }

                    safe_create("skincare", pt, f"{pt.title()} {i+1}", defaults)

        if "haircare" in categories_to_seed:
            hair_types = ["shampoo", "conditioner", "hair_mask", "hair_oil", "scalp_serum"]
            hair_concerns = ["anti_frizz", "volume", "repair", "color_safe", "sensitive_scalp"]

            for pt in hair_types:
                for i in range(20):
                    common = mk_common()
                    attrs = {
                        "hair_type": random.choice(["straight", "wavy", "curly", "coily"]),
                        "scalp_type": random.choice(["oily", "dry", "sensitive", "normal"]),
                        "hair_thickness": random.choice(["fine", "medium", "thick"]),
                    }
                    defaults = {
                        **common,
                        "category": "haircare",
                        "product_type": pt,
                        "concerns": random.sample(hair_concerns, k=1),
                        "attrs": attrs,
                    }
                    safe_create("haircare", pt, f"{pt.replace('_', ' ').title()} {i+1}", defaults)

        if "makeup" in categories_to_seed:
            makeup_types = ["lipstick", "mascara", "foundation", "blush", "eyeshadow"]
            makeup_concerns = ["long_wear", "natural_finish", "full_coverage", "waterproof", "sensitive_eyes"]

            for pt in makeup_types:
                for i in range(25):
                    common = mk_common()
                    if pt == "foundation":
                        attrs = {
                            "finish": random.choice(["matte", "natural", "dewy"]),
                            "coverage": random.choice(["light", "medium", "full"]),
                            "undertone": random.choice(["warm", "cool", "neutral"]),
                            "tone_family": random.choice(["light", "medium", "deep"]),
                        }
                    elif pt == "mascara":
                        attrs = {
                            "waterproof": random.choice([True, False]),
                            "effect": random.choice(["volume", "length", "curl"]),
                        }
                    elif pt == "lipstick":
                        attrs = {
                            "finish": random.choice(["matte", "satin", "gloss"]),
                            "shade_family": random.choice(["nude", "red", "berry", "pink"]),
                        }
                    else:
                        attrs = {"finish": random.choice(["matte", "satin", "shimmer"])}

                    defaults = {
                        **common,
                        "category": "makeup",
                        "product_type": pt,
                        "concerns": random.sample(makeup_concerns, k=1),
                        "attrs": attrs,
                    }
                    safe_create("makeup", pt, f"{pt.title()} {i+1}", defaults)

        if "fragrance" in categories_to_seed:
            frag_types = ["edp", "edt", "body_mist"]
            scent_families = ["floral", "woody", "fresh", "oriental", "gourmand"]
            notes_pool = ["vanilla", "rose", "citrus", "musk", "amber", "jasmine", "bergamot", "patchouli"]

            for pt in frag_types:
                for i in range(20):
                    common = mk_common()
                    attrs = {
                        "scent_family": random.choice(scent_families),
                        "intensity": random.choice(["light", "medium", "strong"]),
                        "notes": random.sample(notes_pool, k=3),
                    }
                    defaults = {
                        **common,
                        "category": "fragrance",
                        "product_type": pt,
                        "concerns": ["gift"] if random.random() < 0.2 else [],
                        "attrs": attrs,
                    }
                    safe_create("fragrance", pt, f"{pt.upper()} {i+1}", defaults)

        self.stdout.write(self.style.SUCCESS(f"Seeded {created} new products. Categories added: {categories_to_seed}"))
