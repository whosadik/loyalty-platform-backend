from __future__ import annotations

from django.core.management.base import BaseCommand

from admin_tools.demo_history_seed import (
    REPORTS_DIR,
    build_demo_seed_catalog_coverage_md,
    build_demo_seed_catalog_coverage_report,
    write_report_files,
)


class Command(BaseCommand):
    help = "Report current curated catalog coverage and usable in-stock seed pools for deterministic demo seeding."

    def add_arguments(self, parser):
        parser.add_argument(
            "--output-md",
            type=str,
            default=str(REPORTS_DIR / "report_demo_seed_catalog_coverage.md"),
            help="Markdown report path.",
        )
        parser.add_argument(
            "--output-json",
            type=str,
            default=str(REPORTS_DIR / "report_demo_seed_catalog_coverage.json"),
            help="JSON report path.",
        )

    def handle(self, *args, **options):
        report = build_demo_seed_catalog_coverage_report()
        write_report_files(
            report=report,
            md_path=options["output_md"],
            json_path=options["output_json"],
            md_builder=build_demo_seed_catalog_coverage_md,
        )
        self.stdout.write(f"products_total={report['counts']['products_total']}")
        self.stdout.write(f"by_category={report['counts']['by_category']}")
        self.stdout.write(f"fragrance_slots={report['counts']['fragrance_slots']}")
        self.stdout.write(f"missing_types={len(report['missing_types'])}")
        self.stdout.write(f"ready_for_demo_seeding={report['ready_for_demo_seeding']}")
        self.stdout.write(f"report_md={options['output_md']}")
        self.stdout.write(f"report_json={options['output_json']}")
