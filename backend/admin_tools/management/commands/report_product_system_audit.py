from __future__ import annotations

from pathlib import Path

from django.core.management.base import BaseCommand

from admin_tools.product_system_audit import (
    PROJECT_ROOT,
    build_product_system_audit_payload,
    write_product_system_audit_bundle,
)


class Command(BaseCommand):
    help = "Generate a full-system product audit report for catalog -> profile -> recommendations -> roadmap -> offers -> checkout -> loyalty -> analytics."

    def add_arguments(self, parser):
        parser.add_argument(
            "--output-md",
            type=str,
            default=str((PROJECT_ROOT / "reports" / "product_system_audit.md").resolve()),
        )
        parser.add_argument(
            "--output-json",
            type=str,
            default=str((PROJECT_ROOT / "reports" / "product_system_audit.json").resolve()),
        )

    def handle(self, *args, **options):
        payload = build_product_system_audit_payload()
        output_md = Path(str(options["output_md"]).strip()).resolve()
        output_json = Path(str(options["output_json"]).strip()).resolve()
        write_product_system_audit_bundle(
            payload=payload,
            output_md=output_md,
            output_json=output_json,
        )
        summary = payload["executive_summary"]
        self.stdout.write(f"demo_ready={summary['demo_ready']}")
        self.stdout.write(f"backend_product_ready={summary['backend_product_ready']}")
        self.stdout.write(f"roadmap_done_enough={summary['roadmap_done_enough']}")
        self.stdout.write(f"ml_product_ready={summary['ml_product_ready']}")
        self.stdout.write(f"single_best_next_block={summary['single_best_next_block']}")
        self.stdout.write(f"report_md={output_md}")
        self.stdout.write(f"report_json={output_json}")
