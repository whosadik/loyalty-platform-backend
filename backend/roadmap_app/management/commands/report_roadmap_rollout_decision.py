from __future__ import annotations

from django.core.management.base import BaseCommand

from roadmap_app.ml_next_step import v4_category_staged_rollout_status


class Command(BaseCommand):
    help = "Read-only staged rollout decision for Roadmap NextStep v4 by category."

    def add_arguments(self, parser):
        parser.add_argument(
            "--include-ga",
            action="store_true",
            default=False,
            help="Compatibility flag for report pipelines; rollout decision uses configured uplift artifacts.",
        )

    def handle(self, *args, **options):
        include_ga = bool(options.get("include_ga"))
        categories = ["skincare", "haircare", "makeup", "fragrance"]
        statuses = [v4_category_staged_rollout_status(cat) for cat in categories]

        self.stdout.write("# Roadmap Rollout Decision")
        self.stdout.write("")
        self.stdout.write(f"- include_ga flag: `{include_ga}`")
        if statuses:
            self.stdout.write(
                f"- source_report_7d: `{str(statuses[0].get('source_report_path_7d') or '')}`"
            )
            self.stdout.write(
                f"- source_report_30d: `{str(statuses[0].get('source_report_path_30d') or '')}`"
            )
        self.stdout.write("")
        self.stdout.write(
            "| category | current decision | 7d recommendation | 30d recommendation | final rollout status | reason |"
        )
        self.stdout.write("| --- | --- | --- | --- | --- | --- |")
        for status in statuses:
            category = str(status.get("category") or "")
            current_decision = str(status.get("current_decision") or status.get("final_status") or "")
            recommendation_7d = str(status.get("recommendation_7d") or "")
            recommendation_30d = str(status.get("recommendation_30d") or "")
            final_status = str(status.get("final_status") or "")
            reason = str(status.get("reason") or "")
            hold_reason = str(status.get("hold_reason") or "")
            if final_status == "HOLD" and hold_reason:
                reason = f"{reason} ({hold_reason})"
            self.stdout.write(
                f"| {category} | {current_decision} | {recommendation_7d} | "
                f"{recommendation_30d} | {final_status} | {reason} |"
            )
