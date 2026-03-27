from __future__ import annotations

from django.core.management.base import BaseCommand

from admin_tools.demo_history_seed import DEMO_USER_PREFIX, parse_yes_no, seed_demo_purchase_history


class Command(BaseCommand):
    help = "Seed deterministic demo purchase history on top of the curated real catalog."

    def add_arguments(self, parser):
        parser.add_argument("--seed", type=int, default=20260326, help="Deterministic seed.")
        parser.add_argument("--users", type=int, default=180, help="How many demo users to seed.")
        parser.add_argument(
            "--max-transactions-per-user",
            type=int,
            default=8,
            help="Upper bound for planned transactions per demo user.",
        )
        parser.add_argument(
            "--days-span",
            type=int,
            default=240,
            help="How far back in time the seeded history should span.",
        )
        parser.add_argument(
            "--use-checkout-path",
            type=str,
            default="yes",
            choices=["yes", "no"],
            help="Route the last primary-category purchases through /api/checkout to create runtime telemetry.",
        )
        parser.add_argument(
            "--include-offer-recs-telemetry",
            type=str,
            default="yes",
            choices=["yes", "no"],
            help="Exercise roadmap/offers/recs endpoints around live-tail checkouts.",
        )
        parser.add_argument(
            "--username-prefix",
            type=str,
            default=DEMO_USER_PREFIX,
            help="Username prefix for demo users.",
        )

    def handle(self, *args, **options):
        summary = seed_demo_purchase_history(
            seed=int(options["seed"]),
            users=int(options["users"]),
            max_transactions_per_user=int(options["max_transactions_per_user"]),
            days_span=int(options["days_span"]),
            use_checkout_path=parse_yes_no(options["use_checkout_path"]),
            include_offer_recs_telemetry=parse_yes_no(options["include_offer_recs_telemetry"]),
            prefix=str(options["username_prefix"]).strip(),
        )
        self.stdout.write(f"users_seeded={summary['users_seeded']}")
        self.stdout.write(f"transactions_created={summary['transactions_created']}")
        self.stdout.write(f"direct_transactions_created={summary['direct_transactions_created']}")
        self.stdout.write(f"live_tail_transactions_created={summary['live_tail_transactions_created']}")
        self.stdout.write(f"owned_products_total={summary['owned_products_total']}")
        self.stdout.write(f"per_category_transactions={summary['per_category_transactions']}")
        self.stdout.write(f"use_checkout_path={summary['use_checkout_path']}")
        self.stdout.write(f"include_offer_recs_telemetry={summary['include_offer_recs_telemetry']}")
