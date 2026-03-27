from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

try:
    from .roadmap_continuation_planner_common import (
        DEFAULT_CATEGORIES,
        build_continuation_decision_dataframe,
        continuation_categories,
        decision_metrics,
        ensure_dependencies,
        load_live_dataset_bundle,
        load_model_artifact,
        resolve_path,
        selected_split_schemes,
        split_frames,
        split_user_overlap,
        suffix_metrics,
        suffix_targets_from_decisions,
    )
except ImportError:  # pragma: no cover
    from roadmap_continuation_planner_common import (
        DEFAULT_CATEGORIES,
        build_continuation_decision_dataframe,
        continuation_categories,
        decision_metrics,
        ensure_dependencies,
        load_live_dataset_bundle,
        load_model_artifact,
        resolve_path,
        selected_split_schemes,
        split_frames,
        split_user_overlap,
        suffix_metrics,
        suffix_targets_from_decisions,
    )

REPO_ROOT = Path(__file__).resolve().parents[2]
BACKEND_DIR = REPO_ROOT / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from roadmap_app.ml_continuation_planner import rollout_continuation_suffix  # noqa: E402
from ml.training.roadmap_live_planner_common import predict_bundle_probabilities  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="tmp/roadmap_continuation_dataset_v1")
    parser.add_argument("--model-root", type=str, default="models/roadmap_continuation_planner")
    parser.add_argument("--report-md", type=str, default="reports/roadmap_continuation_planner_eval.md")
    parser.add_argument("--report-json", type=str, default="reports/roadmap_continuation_planner_eval.json")
    parser.add_argument("--categories", type=str, default="haircare,skincare,fragrance")
    parser.add_argument("--split-schemes", type=str, default="time,user")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _scheme_root(model_root: Path, scheme: str) -> Path:
    return model_root / str(scheme or "time").strip().lower()


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Roadmap Continuation Planner Evaluation",
        "",
        f"- dataset_dir: `{report['dataset_dir']}`",
        f"- model_root: `{report['model_root']}`",
        "",
    ]
    for scheme, scheme_payload in sorted((report.get("split_schemes") or {}).items()):
        lines.extend([f"## {scheme}", ""])
        lines.append(f"- user_overlap: `{scheme_payload['user_overlap']}`")
        for category, payload in sorted((scheme_payload.get("categories") or {}).items()):
            lines.extend(
                [
                    f"### {category}",
                    f"- train/val/test rows: `{payload['rows']['train']}/{payload['rows']['val']}/{payload['rows']['test']}`",
                    f"- val acc@1/recall@3/stop_acc/macro_f1: `{payload['decision_metrics']['val']['acc_at_1']:.4f}` / `{payload['decision_metrics']['val']['recall_at_3']:.4f}` / `{payload['decision_metrics']['val']['stop_acc']:.4f}` / `{payload['decision_metrics']['val']['macro_f1']:.4f}`",
                    f"- test acc@1/recall@3/stop_acc/macro_f1: `{payload['decision_metrics']['test']['acc_at_1']:.4f}` / `{payload['decision_metrics']['test']['recall_at_3']:.4f}` / `{payload['decision_metrics']['test']['stop_acc']:.4f}` / `{payload['decision_metrics']['test']['macro_f1']:.4f}`",
                    f"- val suffix exact/prefix/length_mae/first_error_mean: `{payload['suffix_metrics']['val']['exact_full_suffix_match']:.4f}` / `{payload['suffix_metrics']['val']['prefix_match_rate']:.4f}` / `{payload['suffix_metrics']['val']['suffix_length_mae']:.4f}` / `{payload['suffix_metrics']['val']['first_error_position_mean']:.4f}`",
                    f"- test suffix exact/prefix/length_mae/first_error_mean: `{payload['suffix_metrics']['test']['exact_full_suffix_match']:.4f}` / `{payload['suffix_metrics']['test']['prefix_match_rate']:.4f}` / `{payload['suffix_metrics']['test']['suffix_length_mae']:.4f}` / `{payload['suffix_metrics']['test']['first_error_position_mean']:.4f}`",
                ]
            )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    ensure_dependencies()
    args = parse_args()
    data_dir = resolve_path(str(args.data_dir))
    model_root = resolve_path(str(args.model_root))
    report_md_path = resolve_path(str(args.report_md))
    report_json_path = resolve_path(str(args.report_json))
    report_md_path.parent.mkdir(parents=True, exist_ok=True)
    report_json_path.parent.mkdir(parents=True, exist_ok=True)

    dataset_df, dataset_metadata, _ = load_live_dataset_bundle(data_dir)
    categories = continuation_categories(args.categories)
    if not categories:
        categories = list(DEFAULT_CATEGORIES)
    split_schemes = selected_split_schemes(args.split_schemes)

    report: dict[str, Any] = {
        "dataset_dir": str(data_dir),
        "model_root": str(model_root),
        "dataset_version": str(dataset_metadata.get("version") or ""),
        "split_schemes": {},
    }

    for scheme in split_schemes:
        scheme_root = _scheme_root(model_root, scheme)
        scheme_report: dict[str, Any] = {"categories": {}}
        all_frames: list[pd.DataFrame] = []
        for category in categories:
            decisions_df = build_continuation_decision_dataframe(
                dataset_df=dataset_df,
                category=category,
                split_scheme=scheme,
                seed=int(args.seed),
            )
            if decisions_df.empty:
                continue
            all_frames.append(decisions_df)
            bundle = load_model_artifact(category, model_root=scheme_root)
            split_map = split_frames(decisions_df)
            action_space = list(bundle.get("action_space") or [])
            decision_suffix_targets = suffix_targets_from_decisions(dataset_df, category=category)
            category_report: dict[str, Any] = {
                "rows": {name: int(len(frame)) for name, frame in split_map.items()},
                "decision_metrics": {},
                "suffix_metrics": {},
            }
            for split_name in ("val", "test"):
                frame = split_map[split_name].sort_values(["t0_utc", "decision_id"]).reset_index(drop=True)
                prob_df = predict_bundle_probabilities(bundle, frame) if not frame.empty else pd.DataFrame(columns=action_space)
                predictions = prob_df.idxmax(axis=1).tolist() if not prob_df.empty else []
                category_report["decision_metrics"][split_name] = decision_metrics(frame, predictions, prob_df, action_space)
                category_report["suffix_metrics"][split_name] = suffix_metrics(
                    frame,
                    category=category,
                    rollout_fn=rollout_continuation_suffix,
                    model_root=scheme_root,
                    decision_suffix_targets=decision_suffix_targets,
                )
            scheme_report["categories"][category] = category_report
        combined = pd.concat(all_frames, ignore_index=True) if all_frames else pd.DataFrame(columns=["user_id", "eval_split"])
        scheme_report["user_overlap"] = split_user_overlap(combined) if not combined.empty else {"train_val": 0, "train_test": 0, "val_test": 0}
        report["split_schemes"][scheme] = scheme_report

    report_json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    report_md_path.write_text(_markdown(report), encoding="utf-8")
    print(f"[eval_roadmap_continuation_planner] json={report_json_path}")
    print(f"[eval_roadmap_continuation_planner] md={report_md_path}")


if __name__ == "__main__":
    main()

