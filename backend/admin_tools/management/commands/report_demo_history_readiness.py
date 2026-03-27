from __future__ import annotations

from django.core.management.base import BaseCommand

from admin_tools.demo_history_seed import (
    DEMO_USER_PREFIX,
    REPORTS_DIR,
    build_demo_history_readiness_md,
    build_demo_history_readiness_report,
    write_report_files,
)


class Command(BaseCommand):
    help = "Audit whether the current deterministic demo history is good enough for roadmap dataset rebuilds."

    def add_arguments(self, parser):
        parser.add_argument(
            "--seed",
            type=int,
            default=20260326,
            help="Restrict audit to demo users for a specific seed tag.",
        )
        parser.add_argument(
            "--username-prefix",
            type=str,
            default=DEMO_USER_PREFIX,
            help="Username prefix for demo users.",
        )
        parser.add_argument(
            "--output-md",
            type=str,
            default=str(REPORTS_DIR / "report_demo_history_readiness.md"),
            help="Markdown report path.",
        )
        parser.add_argument(
            "--output-json",
            type=str,
            default=str(REPORTS_DIR / "report_demo_history_readiness.json"),
            help="JSON report path.",
        )

    def handle(self, *args, **options):
        report = build_demo_history_readiness_report(
            prefix=str(options["username_prefix"]).strip(),
            seed=int(options["seed"]),
        )
        write_report_files(
            report=report,
            md_path=options["output_md"],
            json_path=options["output_json"],
            md_builder=build_demo_history_readiness_md,
        )
        verdict = report["verdict"]
        self.stdout.write(f"demo_users_total={report['demo_users_total']}")
        self.stdout.write(f"transactions_total={report['transactions_total']}")
        self.stdout.write(f"initial_planner_anchor_counts={report['initial_planner_anchor_counts']}")
        self.stdout.write(f"fragrance_slot_transitions={report['fragrance_slot_transitions']}")
        self.stdout.write(f"safe_for_dataset_rebuild={verdict['safe_for_dataset_rebuild']}")
        self.stdout.write(f"enough_for_initial_planner_dataset={verdict['enough_for_initial_planner_dataset']}")
        self.stdout.write(f"enough_for_transition_dataset={verdict['enough_for_transition_dataset']}")
        self.stdout.write(f"report_md={options['output_md']}")
        self.stdout.write(f"report_json={options['output_json']}")
