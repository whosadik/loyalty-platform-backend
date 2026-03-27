from __future__ import annotations

import json
from pathlib import Path

from django.core.management.base import BaseCommand

from admin_tools.roadmap_live_initial_diagnostics import (
    baseline_compare_markdown,
    build_live_initial_baseline_compare_report,
    selected_categories,
    selected_split_schemes,
)


class Command(BaseCommand):
    help = "Compare live initial first-step baselines and write a separate ablation report."

    def add_arguments(self, parser):
        parser.add_argument("--data-dir", type=str, default="tmp/roadmap_planner_live_initial_v1")
        parser.add_argument("--transitions-dir", type=str, default="tmp/roadmap_planner_live_transitions_v1")
        parser.add_argument("--model-root", type=str, default="models/roadmap_live_initial_planner_v1")
        parser.add_argument("--categories", type=str, default="haircare,skincare,fragrance")
        parser.add_argument("--split-schemes", type=str, default="time,user")
        parser.add_argument("--seed", type=int, default=42)
        parser.add_argument("--estimator", type=str, default="lightgbm")
        parser.add_argument("--runtime-shadow-time-report", type=str, default="reports/roadmap_live_planner_shadow.json")
        parser.add_argument("--runtime-shadow-user-report", type=str, default="reports/roadmap_live_planner_shadow_user.json")
        parser.add_argument("--report-md", type=str, default="reports/roadmap_live_initial_baseline_compare.md")
        parser.add_argument("--report-json", type=str, default="reports/roadmap_live_initial_baseline_compare.json")
        parser.add_argument("--ablation-md", type=str, default="reports/roadmap_live_initial_ablation.md")
        parser.add_argument("--ablation-json", type=str, default="reports/roadmap_live_initial_ablation.json")

    def handle(self, *args, **options):
        categories = selected_categories(str(options["categories"]))
        split_schemes = selected_split_schemes(str(options["split_schemes"]))
        report, ablation_report = build_live_initial_baseline_compare_report(
            data_dir=str(options["data_dir"]),
            transitions_dir=str(options["transitions_dir"]),
            model_root=str(options["model_root"]),
            categories=categories,
            split_schemes=split_schemes,
            seed=int(options["seed"]),
            estimator_name=str(options["estimator"]),
            runtime_shadow_time_report=str(options["runtime_shadow_time_report"]),
            runtime_shadow_user_report=str(options["runtime_shadow_user_report"]),
        )
        report_md, ablation_md = baseline_compare_markdown(report, ablation_report)

        repo_root = Path(__file__).resolve().parents[4]
        report_md_path = (repo_root / Path(str(options["report_md"]))).resolve()
        report_json_path = (repo_root / Path(str(options["report_json"]))).resolve()
        ablation_md_path = (repo_root / Path(str(options["ablation_md"]))).resolve()
        ablation_json_path = (repo_root / Path(str(options["ablation_json"]))).resolve()
        for path in (report_md_path, report_json_path, ablation_md_path, ablation_json_path):
            path.parent.mkdir(parents=True, exist_ok=True)

        report_json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        ablation_json_path.write_text(json.dumps(ablation_report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        report_md_path.write_text(report_md, encoding="utf-8")
        ablation_md_path.write_text(ablation_md, encoding="utf-8")
        self.stdout.write(f"[report_roadmap_live_initial_baseline_compare] json={report_json_path}")
        self.stdout.write(f"[report_roadmap_live_initial_baseline_compare] md={report_md_path}")
        self.stdout.write(f"[report_roadmap_live_initial_baseline_compare] ablation_json={ablation_json_path}")
        self.stdout.write(f"[report_roadmap_live_initial_baseline_compare] ablation_md={ablation_md_path}")
