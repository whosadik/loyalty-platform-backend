from __future__ import annotations

import json

from django.core.management.base import BaseCommand

from admin_tools.commerce_proof import (
    COMMERCE_PROOF_PREFIX,
    COMMERCE_PROOF_SEED,
    seed_commerce_proof_scenarios,
)


class Command(BaseCommand):
    help = "Create live-like commerce proof scenarios via real next-offer and checkout flows."

    def add_arguments(self, parser):
        parser.add_argument("--seed", type=int, default=COMMERCE_PROOF_SEED)
        parser.add_argument("--username-prefix", type=str, default=COMMERCE_PROOF_PREFIX)

    def handle(self, *args, **options):
        summary = seed_commerce_proof_scenarios(
            seed=int(options["seed"]),
            prefix=str(options["username_prefix"]).strip(),
        )
        snapshot = summary["snapshot"]
        self.stdout.write(f"seed={summary['seed']}")
        self.stdout.write(f"username_prefix={summary['username_prefix']}")
        self.stdout.write(f"proof_users_created={summary['proof_users_created']}")
        self.stdout.write(f"redeemed_assignments_total={snapshot['redeemed_assignments_total']}")
        self.stdout.write(f"loyalty_redeem_entries={snapshot['loyalty_redeem_entries']}")
        self.stdout.write(f"multi_item_transactions={snapshot['multi_item_transactions']}")
        self.stdout.write(f"active_redeemed_assignments={snapshot['active_redeemed_assignments']}")
        self.stdout.write(f"usernames={json.dumps(snapshot['usernames'], ensure_ascii=False)}")
