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
    ap.add_argument(
        "--data_source",
        choices=["processed", "cosmetics_raw", "project_db"],
        default="processed",
        help="processed: use existing parquet, cosmetics_raw: build from raw CSV, project_db: export from backend DB tables",
    )
    ap.add_argument("--raw_glob", default="")
    ap.add_argument("--db_days", type=int, default=180, help="Window for project_db export")
    ap.add_argument("--processed_dir", default="data/processed/cosmetics")
    ap.add_argument("--models_dir", default="models/recs_reranker_v2")
    ap.add_argument("--reports_dir", default="reports")
    ap.add_argument("--top_m", type=int, default=1500)
    ap.add_argument("--context_k", type=int, default=10)
    ap.add_argument("--product_type_fallback_topn", type=int, default=400)
    ap.add_argument("--category_fallback_topn", type=int, default=400)
    ap.add_argument("--brand_fallback_topn", type=int, default=400)
    ap.add_argument(
        "--behavior_event_types",
        default="add_to_cart,click,purchase_attributed",
        help="Comma-separated event types for additional behavior transitions",
    )
    ap.add_argument("--behavior_weight", type=float, default=0.25)
    ap.add_argument("--neg_per_pos", type=int, default=20)
    ap.add_argument("--estimator", choices=["lr", "hgb"], default="hgb")
    ap.add_argument("--model_version", default="recs_reranker_v3")
    ap.add_argument("--test_size", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    processed = Path(args.processed_dir)
    processed.mkdir(parents=True, exist_ok=True)
    Path(args.models_dir).mkdir(parents=True, exist_ok=True)
    Path(args.reports_dir).mkdir(parents=True, exist_ok=True)

    interactions = processed / "interactions.parquet"
    items = processed / "items.parquet"
    ds = processed / "next_purchase_ds.parquet"

    source = args.data_source
    if source == "processed" and args.raw_glob:
        # Backward compatibility with old commands.
        source = "cosmetics_raw"

    if source == "cosmetics_raw":
        if not args.raw_glob:
            raise SystemExit("--raw_glob is required for --data_source cosmetics_raw")
        run(
            [
                args.python,
                "ml/datasets/convert_cosmetics.py",
                "--input_glob",
                args.raw_glob,
                "--out_dir",
                str(processed),
            ]
        )
    elif source == "project_db":
        run(
            [
                args.python,
                "ml/training/export_project_training_data.py",
                "--out_dir",
                str(processed),
                "--days",
                str(args.db_days),
            ]
        )

    if not interactions.exists():
        raise SystemExit(f"Missing interactions parquet: {interactions}")
    if not items.exists():
        raise SystemExit(f"Missing items parquet: {items}")

    run(
        [
            args.python,
            "ml/training/build_next_purchase_dataset.py",
            "--interactions",
            str(interactions),
            "--items",
            str(items),
            "--out",
            str(ds),
            "--context_k",
            str(args.context_k),
        ]
    )

    run(
        [
            args.python,
            "ml/training/train_reranker_lr.py",
            "--interactions",
            str(interactions),
            "--items",
            str(items),
            "--ds",
            str(ds),
            "--out_dir",
            str(args.models_dir),
            "--top_m",
            str(args.top_m),
            "--context_k",
            str(args.context_k),
            "--product_type_fallback_topn",
            str(args.product_type_fallback_topn),
            "--category_fallback_topn",
            str(args.category_fallback_topn),
            "--brand_fallback_topn",
            str(args.brand_fallback_topn),
            "--estimator",
            str(args.estimator),
            "--model_version",
            str(args.model_version),
            "--behavior_event_types",
            str(args.behavior_event_types),
            "--behavior_weight",
            str(args.behavior_weight),
            "--neg_per_pos",
            str(args.neg_per_pos),
            "--test_size",
            str(args.test_size),
            "--seed",
            str(args.seed),
        ]
    )

    train_users = str(Path(args.models_dir) / "train_users.txt")
    test_users = str(Path(args.models_dir) / "test_users.txt")

    run(
        [
            args.python,
            "ml/training/eval_cooc_baseline.py",
            "--interactions",
            str(interactions),
            "--items",
            str(items),
            "--ds",
            str(ds),
            "--train_users",
            train_users,
            "--eval_users",
            test_users,
            "--top_m",
            str(args.top_m),
            "--context_k",
            str(args.context_k),
            "--product_type_fallback_topn",
            str(args.product_type_fallback_topn),
            "--category_fallback_topn",
            str(args.category_fallback_topn),
            "--brand_fallback_topn",
            str(args.brand_fallback_topn),
            "--behavior_event_types",
            str(args.behavior_event_types),
            "--behavior_weight",
            str(args.behavior_weight),
            "--out_report",
            str(Path(args.reports_dir) / "cooc_baseline_test.txt"),
            "--out_json",
            str(Path(args.reports_dir) / "cooc_baseline_test.json"),
        ]
    )

    run(
        [
            args.python,
            "ml/training/eval_reranker.py",
            "--interactions",
            str(interactions),
            "--items",
            str(items),
            "--ds",
            str(ds),
            "--model",
            str(Path(args.models_dir) / "model.pkl"),
            "--train_users",
            train_users,
            "--eval_users",
            test_users,
            "--top_m",
            str(args.top_m),
            "--context_k",
            str(args.context_k),
            "--product_type_fallback_topn",
            str(args.product_type_fallback_topn),
            "--category_fallback_topn",
            str(args.category_fallback_topn),
            "--brand_fallback_topn",
            str(args.brand_fallback_topn),
            "--behavior_event_types",
            str(args.behavior_event_types),
            "--behavior_weight",
            str(args.behavior_weight),
            "--out_report",
            str(Path(args.reports_dir) / "reranker_test.txt"),
            "--out_json",
            str(Path(args.reports_dir) / "reranker_test.json"),
        ]
    )

    print("DONE")
    print("model:", str(Path(args.models_dir) / "model.pkl"))
    print("baseline report:", str(Path(args.reports_dir) / "cooc_baseline_test.txt"))
    print("reranker report:", str(Path(args.reports_dir) / "reranker_test.txt"))


if __name__ == "__main__":
    main()
