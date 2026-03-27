from __future__ import annotations

import json

from django.core.management.base import BaseCommand, CommandError

from admin_tools.roadmap_continuation_alignment import (
    aggregate_alignment,
    continuation_categories,
    load_eval_rows_from_dataset,
    resolve_path,
    selected_split_schemes,
    top_alignment_examples,
    window_sensitivity_report,
)


def _markdown(report: dict) -> str:
    lines = [
        "# Roadmap Continuation Truth Alignment",
        "",
        f"- dataset_dir: `{report['dataset_dir']}`",
        f"- model_root: `{report['model_root']}`",
        f"- split_schemes: `{report['split_schemes']}`",
        "",
    ]
    for scheme, payload in sorted((report.get("alignment") or {}).items()):
        lines.extend([f"## {scheme}", ""])
        lines.append(f"- overall_by_category: `{payload.get('by_category')}`")
        lines.append(f"- overall_by_decision_type: `{payload.get('by_decision_type')}`")
        lines.append(f"- overall_by_confidence_bucket: `{payload.get('by_confidence_bucket')}`")
        lines.append(f"- overall_by_suffix_length_bucket: `{payload.get('by_suffix_length_bucket')}`")
        lines.append("")

    lines.extend(["## Window Sensitivity"])
    for window, payload in sorted((report.get("window_sensitivity") or {}).items(), key=lambda item: int(item[0])):
        lines.append(
            f"- {window}d: decisions={payload['trusted_decisions_total']}, non_stop={payload['non_stop_positives_total']}, "
            f"stop_rate={payload['stop_rate']:.4f}, runtime_acc={payload['runtime_accuracy_overall']:.4f}, "
            f"ml_acc={payload['ml_accuracy_overall']:.4f}, runtime_continue_when_truth_stop_rate={payload['runtime_continue_when_truth_stop_rate']:.4f}"
        )
    lines.extend(["", "## Example Cases"])
    examples = report.get("examples") or {}
    for key in ("runtime_right_ml_wrong", "ml_right_runtime_wrong", "suspected_label_noise"):
        lines.append(f"### {key}")
        for case in examples.get(key) or []:
            lines.append(
                f"- {case['split_scheme']} | {case['category']} | truth={case['truth_label']} | runtime={case['runtime_action']} | "
                f"ml={case['ml_action']} | conf={case['ml_top_probability']:.4f} | noise={case['suspected_label_noise_reason'] or 'none'}"
            )
    return "\n".join(lines).rstrip() + "\n"


class Command(BaseCommand):
    help = "Truth-alignment audit for continuation ML vs current runtime continuation rules."

    def add_arguments(self, parser):
        parser.add_argument("--data-dir", type=str, default="tmp/roadmap_continuation_dataset_v1")
        parser.add_argument("--model-root", type=str, default="models/roadmap_continuation_planner")
        parser.add_argument("--categories", type=str, default="haircare,skincare,fragrance")
        parser.add_argument("--split-schemes", type=str, default="time,user")
        parser.add_argument("--seed", type=int, default=42)
        parser.add_argument("--window-days-list", type=str, default="3,7,14")
        parser.add_argument("--model-split-scheme", type=str, default="user")
        parser.add_argument("--sample-per-category", type=int, default=5)
        parser.add_argument("--report-md", type=str, default="reports/roadmap_continuation_truth_alignment.md")
        parser.add_argument("--report-json", type=str, default="reports/roadmap_continuation_truth_alignment.json")

    def handle(self, *args, **options):
        categories = continuation_categories(str(options["categories"]))
        split_schemes = selected_split_schemes(str(options["split_schemes"]))
        window_days = [int(item.strip()) for item in str(options["window_days_list"]).split(",") if str(item).strip()]
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

        alignment: dict[str, dict] = {}
        for scheme in split_schemes:
            scheme_rows = rows[rows["split_scheme"].astype(str) == scheme].copy() if not rows.empty else rows
            alignment[scheme] = {
                "by_category": aggregate_alignment(scheme_rows, group_fields=["category"]),
                "by_decision_type": aggregate_alignment(scheme_rows, group_fields=["category", "decision_type"]),
                "by_label": aggregate_alignment(scheme_rows, group_fields=["category", "truth_label"]),
                "by_confidence_bucket": aggregate_alignment(scheme_rows, group_fields=["category", "ml_confidence_bucket"]),
                "by_suffix_length_bucket": aggregate_alignment(scheme_rows, group_fields=["category", "suffix_length_bucket"]),
            }

        try:
            sensitivity = window_sensitivity_report(
                categories=categories,
                model_root=str(options["model_root"]),
                model_split_scheme=str(options["model_split_scheme"]),
                seed=int(options["seed"]),
                window_days_list=window_days,
            )
        except Exception as exc:
            raise CommandError(str(exc)) from exc

        report = {
            "dataset_dir": str(resolve_path(str(options["data_dir"]))),
            "model_root": str(resolve_path(str(options["model_root"]))),
            "split_schemes": split_schemes,
            "alignment": alignment,
            "window_sensitivity": sensitivity,
            "examples": top_alignment_examples(rows, sample_per_category=int(options["sample_per_category"])),
        }
        report_json.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        report_md.write_text(_markdown(report), encoding="utf-8")
        self.stdout.write(f"[report_roadmap_continuation_truth_alignment] json={report_json}")
        self.stdout.write(f"[report_roadmap_continuation_truth_alignment] md={report_md}")
