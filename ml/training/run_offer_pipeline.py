from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def run(cmd: list[str]) -> None:
    print("RUN:", " ".join(cmd))
    proc = subprocess.run(cmd, check=False)
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--processed_dir", default="data/processed/project")
    ap.add_argument("--models_dir", default="models/offer_redemption_lr_v1")
    ap.add_argument("--days", type=int, default=180)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    Path(args.processed_dir).mkdir(parents=True, exist_ok=True)
    Path(args.models_dir).mkdir(parents=True, exist_ok=True)

    run(
        [
            args.python,
            "ml/training/export_project_training_data.py",
            "--out_dir",
            args.processed_dir,
            "--days",
            str(args.days),
        ]
    )

    run(
        [
            args.python,
            "ml/training/train_offer_redemption_lr.py",
            "--dataset",
            str(Path(args.processed_dir) / "offer_train.parquet"),
            "--out_dir",
            args.models_dir,
            "--seed",
            str(args.seed),
        ]
    )

    print("DONE")
    print("offer model:", str(Path(args.models_dir) / "model.pkl"))


if __name__ == "__main__":
    main()
