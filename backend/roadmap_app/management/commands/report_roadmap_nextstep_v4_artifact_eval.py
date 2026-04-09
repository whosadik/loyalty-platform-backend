from __future__ import annotations

from pathlib import Path

from django.conf import settings
from django.core.management.base import BaseCommand, CommandError

from roadmap_app.nextstep_artifact_eval import write_nextstep_v4_artifact_eval_report


class Command(BaseCommand):
    help = "Score the exact configured Roadmap NextStep v4 artifact on its preserved dataset/splits and write artifact-local eval_report."

    def add_arguments(self, parser):
        parser.add_argument(
            "--model-path",
            type=str,
            default=str(getattr(settings, "ROADMAP_NEXTSTEP_V4_MODEL_PATH", "") or "").strip(),
        )
        parser.add_argument(
            "--data-dir",
            type=str,
            default="",
            help="Optional override for the dataset directory containing dataset.parquet, splits.json and metadata.json.",
        )
        parser.add_argument(
            "--output-json",
            type=str,
            default="",
            help="Optional override for eval_report.json. Defaults to <artifact_dir>/eval_report.json.",
        )
        parser.add_argument(
            "--output-md",
            type=str,
            default="",
            help="Optional override for eval_report.md. Defaults to <artifact_dir>/eval_report.md.",
        )

    def handle(self, *args, **options):
        model_path = str(options.get("model_path") or "").strip()
        if not model_path:
            raise CommandError("--model-path must not be empty")

        data_dir = str(options.get("data_dir") or "").strip() or None
        output_json = str(options.get("output_json") or "").strip() or None
        output_md = str(options.get("output_md") or "").strip() or None

        report, json_path, md_path = write_nextstep_v4_artifact_eval_report(
            model_path=model_path,
            data_dir=data_dir,
            output_json=output_json,
            output_md=output_md,
        )

        self.stdout.write(
            f"[report_roadmap_nextstep_v4_artifact_eval] model_path={Path(model_path).expanduser().resolve()}"
        )
        self.stdout.write(f"[report_roadmap_nextstep_v4_artifact_eval] eval_json={json_path}")
        self.stdout.write(f"[report_roadmap_nextstep_v4_artifact_eval] eval_md={md_path}")
        self.stdout.write(
            "[report_roadmap_nextstep_v4_artifact_eval] "
            f"test_recall@1={float(report['metrics_test']['recall_at_1']):.4f} "
            f"test_ndcg@5={float(report['metrics_test']['ndcg_at_5']):.4f}"
        )
