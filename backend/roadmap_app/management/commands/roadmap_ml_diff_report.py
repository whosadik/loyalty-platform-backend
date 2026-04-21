from __future__ import annotations

import json

from django.core.management.base import BaseCommand

from roadmap_app.ml_diff_report import (
    DEFAULT_TOP_DIVERGENCES,
    DEFAULT_WINDOW_MINUTES,
    build_control_vs_ml_diff_report,
)


class Command(BaseCommand):
    help = (
        "Produce a control-vs-ML diff report from the RoadmapMLInvocation log. "
        "Compares what was served (planned_target_product_type) against the active "
        "ML top pick and the shadow ML top pick per category, with agreement rates "
        "and top divergence pairs."
    )

    def add_arguments(self, parser) -> None:
        parser.add_argument("--window-minutes", type=int, default=DEFAULT_WINDOW_MINUTES)
        parser.add_argument(
            "--category",
            action="append",
            default=[],
            help="Filter to specific category (repeatable). Default: all categories present.",
        )
        parser.add_argument(
            "--top-divergences",
            type=int,
            default=DEFAULT_TOP_DIVERGENCES,
        )
        parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON output.")

    def handle(self, *args, **options) -> None:
        report = build_control_vs_ml_diff_report(
            window_minutes=options["window_minutes"],
            categories=options["category"] or None,
            top_divergences=options["top_divergences"],
        )
        if options["json"]:
            self.stdout.write(
                json.dumps(report, indent=2, sort_keys=True, default=str)
            )
            return

        self.stdout.write(
            f"window={report['window_minutes']}m  cutoff={report['cutoff_utc']}  "
            f"total={report['total_invocations']}"
        )
        if not report["per_category"]:
            self.stdout.write("  (no invocations in window)")
            return
        for cat in sorted(report["per_category"]):
            payload = report["per_category"][cat]
            sva = payload["agreement"]["served_vs_active"]
            svs = payload["agreement"]["served_vs_shadow"]
            avs = payload["agreement"]["active_vs_shadow"]
            self.stdout.write(
                f"  [{cat}] total={payload['total']} "
                f"rollout_sel={payload['rollout_selected_count']} "
                f"decisions={payload['decision_counts']}"
            )
            self.stdout.write(
                f"    served vs active: {sva['matches']}/{sva['compared']} "
                f"({sva['agreement_pct']:.2f}%)"
            )
            self.stdout.write(
                f"    served vs shadow: {svs['matches']}/{svs['compared']} "
                f"({svs['agreement_pct']:.2f}%)"
            )
            self.stdout.write(
                f"    active vs shadow: {avs['matches']}/{avs['compared']} "
                f"({avs['agreement_pct']:.2f}%)"
            )
            if payload["fallback_reason_counts"]:
                self.stdout.write(
                    f"    fallback_reasons={payload['fallback_reason_counts']}"
                )
            for label in ("served_vs_active", "served_vs_shadow", "active_vs_shadow"):
                top = payload["top_divergences"].get(label) or []
                if top:
                    self.stdout.write(f"    top divergences ({label}):")
                    for item in top:
                        self.stdout.write(f"      {item}")
