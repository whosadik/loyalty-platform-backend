from __future__ import annotations

import json

from django.core.management.base import BaseCommand, CommandError

from admin_tools.roadmap_continuation_dataset import (
    build_continuation_bundle,
    resolve_repo_path,
    selected_categories,
    write_continuation_summary_md,
)


class Command(BaseCommand):
    help = "Build continuation-only roadmap planner dataset from trusted completed/skipped transitions."

    def add_arguments(self, parser):
        parser.add_argument("--days", type=int, default=365)
        parser.add_argument("--out-dir", type=str, default="tmp/roadmap_continuation_dataset_v1")
        parser.add_argument("--label-window-days", type=int, default=3)
        parser.add_argument("--include-ga", action="store_true", default=False)
        parser.add_argument("--seed", type=int, default=42)
        parser.add_argument("--categories", type=str, default="haircare,skincare,fragrance")

    def handle(self, *args, **options):
        out_dir = resolve_repo_path(str(options["out_dir"]))
        out_dir.mkdir(parents=True, exist_ok=True)
        categories = selected_categories(str(options["categories"]))

        try:
            bundle = build_continuation_bundle(
                days=int(options["days"]),
                label_window_days=int(options["label_window_days"]),
                include_ga=bool(options["include_ga"]),
                seed=int(options["seed"]),
                categories=categories,
            )
        except Exception as exc:
            raise CommandError(str(exc)) from exc

        frame = bundle["frame"]
        metadata = bundle["metadata"]
        splits_payload = bundle["splits"]

        dataset_format = "parquet"
        dataset_path = out_dir / "dataset.parquet"
        try:
            frame.to_parquet(dataset_path, index=False)
        except Exception:
            dataset_format = "csv"
            dataset_path = out_dir / "dataset.csv"
            frame.to_csv(dataset_path, index=False)

        metadata = dict(metadata)
        metadata["dataset_format"] = dataset_format
        metadata["dataset_file"] = str(dataset_path)

        metadata_path = out_dir / "metadata.json"
        metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        splits_path = out_dir / "splits.json"
        splits_path.write_text(json.dumps(splits_payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        summary_path = write_continuation_summary_md(out_dir=out_dir, metadata=metadata)

        self.stdout.write(f"[build_roadmap_continuation_dataset] dataset={dataset_path}")
        self.stdout.write(f"[build_roadmap_continuation_dataset] metadata={metadata_path}")
        self.stdout.write(f"[build_roadmap_continuation_dataset] splits={splits_path}")
        self.stdout.write(f"[build_roadmap_continuation_dataset] summary={summary_path}")

