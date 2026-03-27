from __future__ import annotations

from django.core.management.base import BaseCommand

from admin_tools.demo_history_seed import DEMO_USER_PREFIX, seed_demo_users


class Command(BaseCommand):
    help = "Create deterministic demo users and correlated profiles without touching staff/admin users."

    def add_arguments(self, parser):
        parser.add_argument("--seed", type=int, default=20260326, help="Deterministic seed.")
        parser.add_argument("--users", type=int, default=180, help="Number of demo users to create.")
        parser.add_argument(
            "--username-prefix",
            type=str,
            default=DEMO_USER_PREFIX,
            help="Username prefix for disposable demo users.",
        )

    def handle(self, *args, **options):
        summary = seed_demo_users(
            seed=int(options["seed"]),
            total_users=int(options["users"]),
            prefix=str(options["username_prefix"]).strip(),
        )
        self.stdout.write(f"seed={summary['seed']}")
        self.stdout.write(f"users_created={summary['users_created']}")
        self.stdout.write(f"deleted_previous_demo_users={summary['deleted_previous_demo_users']}")
        self.stdout.write(f"cohorts={summary['cohorts']}")
        self.stdout.write(f"username_prefix={summary['username_prefix']}")
        self.stdout.write(f"password={summary['password']}")
