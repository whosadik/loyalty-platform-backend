from __future__ import annotations

import json

from django.core.management.base import BaseCommand, CommandError

from admin_tools.roadmap_continuation_alignment import (
    continuation_categories,
    disagreement_summary,
    load_eval_rows_from_dataset,
    resolve_path,
    sample_disagreement_cases,
    selected_split_schemes,
)


def _markdown(report: dict) -> str:
    lines = [
        "# Roadmap Continuation Disagreement Cases",
        "",
        f"- dataset_dir: `{report['dataset_dir']}`",
        f"- model_root: `{report['model_root']}`",
        f"- split_schemes: `{report['split_schemes']}`",
        "",
    ]
    for scheme, payload in sorted((report.get("by_split_scheme") or {}).items()):
        lines.extend([f"## {scheme}", ""])
        for category, summary in sorted((payload.get("categories") or {}).items()):
            reason_counts = summary.get("reason_counts") or {}
            lines.extend(
                [
                    f"### {category}",
                    f"- rows_total: **{summary.get('rows_total', 0)}**",
                    f"- disagreements_total: **{summary.get('disagreements_total', 0)}**",
                    f"- ml_stop_runtime_continue: `{reason_counts.get('ml_stop_runtime_continue', 0)}`",
                    f"- ml_continue_runtime_stop: `{reason_counts.get('ml_continue_runtime_stop', 0)}`",
                    f"- both_continue_but_different_action: `{reason_counts.get('both_continue_but_different_action', 0)}`",
                    f"- low_confidence_ml_disagreement: `{reason_counts.get('low_confidence_ml_disagreement', 0)}`",
                    f"- fragrance_slot_conflicts: `{reason_counts.get('fragrance_slot_conflicts', 0)}`",
                ]
            )
        lines.append("")
    lines.append("## Sample Cases")
    for case in report.get("sample_cases") or []:
        lines.append(
            f"- {case['split_scheme']} | {case['category']} | user={case['user_id']} | decision={case['decision_id']} | "
            f"truth={case['truth_label']} | runtime={case['runtime_action']} | "
            f"ml={case['ml_top_predictions'][0]['action'] if case['ml_top_predictions'] else '__stop__'} | "
            f"reason={case['reason_code']} | noise={case['suspected_label_noise_reason'] or 'none'}"
        )
    return "\n".join(lines).rstrip() + "\n"


class Command(BaseCommand):
    help = "Case-level disagreement audit between continuation ML and current runtime continuation behavior."

    def add_arguments(self, parser):
        parser.add_argument("--data-dir", type=str, default="tmp/roadmap_continuation_dataset_v1")
        parser.add_argument("--model-root", type=str, default="models/roadmap_continuation_planner")
        parser.add_argument("--categories", type=str, default="haircare,skincare,fragrance")
        parser.add_argument("--split-schemes", type=str, default="time,user")
        parser.add_argument("--seed", type=int, default=42)
        parser.add_argument("--sample-per-category", type=int, default=10)
        parser.add_argument("--report-md", type=str, default="reports/roadmap_continuation_disagreement_cases.md")
        parser.add_argument("--report-json", type=str, default="reports/roadmap_continuation_disagreement_cases.json")

    def handle(self, *args, **options):
        categories = continuation_categories(str(options["categories"]))
        split_schemes = selected_split_schemes(str(options["split_schemes"]))
        report_md = resolve_path(str(options["report_md"]))
        report_json = resolve_path(str(options["report_json"]))
        report_md.parent.mkdir(parents=True, exist_ok=True)
        report_json.parent.mkdir(parents=True, exist_ok=True)

        try:
            rows, _metadata = load_eval_rows_from_dataset(
                data_dir=str(options["data_dir"]),
                model_root=str(options["model_root"]),
                categories=categories,
                split_schemes=split_schemes,
                seed=int(options["seed"]),
                eval_split="test",
            )
        except Exception as exc:
            raise CommandError(str(exc)) from exc

        by_split_scheme: dict[str, dict] = {}
        for scheme in split_schemes:
            scheme_rows = rows[rows["split_scheme"].astype(str) == scheme].copy() if not rows.empty else rows
            by_split_scheme[scheme] = {"categories": {}}
            for category in categories:
                category_rows = scheme_rows[scheme_rows["category"].astype(str) == category].copy() if not scheme_rows.empty else scheme_rows
                by_split_scheme[scheme]["categories"][category] = disagreement_summary(category_rows)

        report = {
            "dataset_dir": str(resolve_path(str(options["data_dir"]))),
            "model_root": str(resolve_path(str(options["model_root"]))),
            "split_schemes": split_schemes,
            "by_split_scheme": by_split_scheme,
            "sample_cases": sample_disagreement_cases(rows, per_category=int(options["sample_per_category"])),
        }
        report_json.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        report_md.write_text(_markdown(report), encoding="utf-8")
        self.stdout.write(f"[report_roadmap_continuation_disagreement_cases] json={report_json}")
        self.stdout.write(f"[report_roadmap_continuation_disagreement_cases] md={report_md}")
