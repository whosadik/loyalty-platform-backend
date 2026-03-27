from __future__ import annotations

import json
from pathlib import Path

from django.core.management.base import BaseCommand

from admin_tools.roadmap_live_initial_diagnostics import (
    build_live_initial_diagnostics_report,
    diagnostics_markdown,
    selected_categories,
    selected_split_schemes,
)


class Command(BaseCommand):
    help = "Build diagnostic report for live initial planner first-step view and time-split gap analysis."

    def add_arguments(self, parser):
        parser.add_argument("--data-dir", type=str, default="tmp/roadmap_planner_live_initial_v1")
        parser.add_argument("--categories", type=str, default="haircare,skincare,fragrance")
        parser.add_argument("--split-schemes", type=str, default="time,user")
        parser.add_argument("--seed", type=int, default=42)
        parser.add_argument("--scenario-seed", type=int, default=20260326)
        parser.add_argument("--scenario-users", type=int, default=180)
        parser.add_argument("--scenario-max-transactions-per-user", type=int, default=10)
        parser.add_argument("--scenario-prefix", type=str, default="ga_demo")
        parser.add_argument("--runtime-shadow-time-report", type=str, default="reports/roadmap_live_planner_shadow.json")
        parser.add_argument("--runtime-shadow-user-report", type=str, default="reports/roadmap_live_planner_shadow_user.json")
        parser.add_argument("--report-md", type=str, default="reports/roadmap_live_initial_diagnostics.md")
        parser.add_argument("--report-json", type=str, default="reports/roadmap_live_initial_diagnostics.json")

    def handle(self, *args, **options):
        categories = selected_categories(str(options["categories"]))
        split_schemes = selected_split_schemes(str(options["split_schemes"]))
        report = build_live_initial_diagnostics_report(
            data_dir=str(options["data_dir"]),
            categories=categories,
            split_schemes=split_schemes,
            seed=int(options["seed"]),
            scenario_seed=int(options["scenario_seed"]),
            scenario_users=int(options["scenario_users"]),
            scenario_max_transactions_per_user=int(options["scenario_max_transactions_per_user"]),
            scenario_prefix=str(options["scenario_prefix"]),
            runtime_shadow_time_report=str(options["runtime_shadow_time_report"]),
            runtime_shadow_user_report=str(options["runtime_shadow_user_report"]),
        )
        repo_root = Path(__file__).resolve().parents[4]
        report_md = (repo_root / Path(str(options["report_md"]))).resolve()
        report_json = (repo_root / Path(str(options["report_json"]))).resolve()
        report_md.parent.mkdir(parents=True, exist_ok=True)
        report_json.parent.mkdir(parents=True, exist_ok=True)
        report_json.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        report_md.write_text(diagnostics_markdown(report), encoding="utf-8")
        self.stdout.write(f"[report_roadmap_live_initial_diagnostics] json={report_json}")
        self.stdout.write(f"[report_roadmap_live_initial_diagnostics] md={report_md}")
