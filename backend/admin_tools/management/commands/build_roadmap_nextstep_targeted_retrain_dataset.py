from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from django.core.management.base import BaseCommand, CommandError

from roadmap_app.nextstep_targeted_retrain import (
    DEFAULT_TARGETED_DATA_DIR,
    POLICY_VERSION,
    apply_targeted_retrain_weights,
    targeted_retrain_policy_payload,
)


class Command(BaseCommand):
    help = "Build a targeted-retrain overlay dataset for roadmap nextstep_v4 without changing features or labels."

    def add_arguments(self, parser):
        parser.add_argument("--source-data-dir", default="data/ml/roadmap_nextstep_v4")
        parser.add_argument("--out-dir", default=str(DEFAULT_TARGETED_DATA_DIR))

    def handle(self, *args, **options):
        source_dir = Path(str(options.get("source_data_dir") or "")).expanduser().resolve()
        out_dir = Path(str(options.get("out_dir") or "")).expanduser().resolve()
        dataset_path = source_dir / "dataset.parquet"
        if not dataset_path.exists():
            raise CommandError(f"Missing source dataset: {dataset_path}")
        metadata_path = source_dir / "metadata.json"
        splits_path = source_dir / "splits.json"
        if not metadata_path.exists() or not splits_path.exists():
            raise CommandError(f"Missing metadata.json or splits.json in {source_dir}")

        df = pd.read_parquet(dataset_path)
        weighted_df, summary = apply_targeted_retrain_weights(df)
        out_dir.mkdir(parents=True, exist_ok=True)
        weighted_df.to_parquet(out_dir / "dataset.parquet", index=False)
        (out_dir / "splits.json").write_text(splits_path.read_text(encoding="utf-8"), encoding="utf-8")

        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata["targeted_retrain_policy_version"] = POLICY_VERSION
        metadata["targeted_retrain_policy"] = targeted_retrain_policy_payload()
        metadata["targeted_retrain_summary"] = summary
        metadata["source_dataset_dir"] = str(source_dir)
        metadata["source_dataset_file"] = str(dataset_path)
        metadata["dataset_file"] = str((out_dir / "dataset.parquet").resolve())
        metadata["sample_weight_policy"] = {
            "type": "targeted_retrain_overlay",
            "policy_version": POLICY_VERSION,
            "rows_reweighted_total": int(summary["rows_reweighted_total"]),
        }
        (out_dir / "metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        self.stdout.write("[build_roadmap_nextstep_targeted_retrain_dataset] done")
        self.stdout.write(f"[build_roadmap_nextstep_targeted_retrain_dataset] source={source_dir}")
        self.stdout.write(f"[build_roadmap_nextstep_targeted_retrain_dataset] out={out_dir}")
        self.stdout.write(
            "[build_roadmap_nextstep_targeted_retrain_dataset] "
            f"rows_reweighted={summary['rows_reweighted_total']} policy={POLICY_VERSION}"
        )
