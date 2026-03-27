from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    from .roadmap_initial_planner_common import longest_common_prefix_rate, sequence_exact_match
    from .roadmap_live_planner_common import (
        ALLOWED_CATEGORIES,
        accuracy,
        build_live_decision_dataframe,
        confusion_matrix,
        ensure_dependencies,
        episode_targets_from_transitions,
        load_live_dataset_bundle,
        load_model_artifact,
        per_label_stats,
        predict_bundle_probabilities,
        recall_at_k,
        resolve_path,
        selected_categories,
        selected_split_schemes,
        split_frames,
        split_user_overlap,
    )
except ImportError:  # pragma: no cover
    from roadmap_initial_planner_common import longest_common_prefix_rate, sequence_exact_match
    from roadmap_live_planner_common import (
        ALLOWED_CATEGORIES,
        accuracy,
        build_live_decision_dataframe,
        confusion_matrix,
        ensure_dependencies,
        episode_targets_from_transitions,
        load_live_dataset_bundle,
        load_model_artifact,
        per_label_stats,
        predict_bundle_probabilities,
        recall_at_k,
        resolve_path,
        selected_categories,
        selected_split_schemes,
        split_frames,
        split_user_overlap,
    )

REPO_ROOT = Path(__file__).resolve().parents[2]
BACKEND_DIR = REPO_ROOT / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from roadmap_app.ml_live_planner import rollout_live_plan  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="tmp/roadmap_planner_live_initial_v1")
    parser.add_argument("--transitions-dir", type=str, default="tmp/roadmap_planner_live_transitions_v1")
    parser.add_argument("--model-root", type=str, default="models/roadmap_live_initial_planner_v1")
    parser.add_argument("--report-md", type=str, default="reports/roadmap_live_initial_planner_eval.md")
    parser.add_argument("--report-json", type=str, default="reports/roadmap_live_initial_planner_eval.json")
    parser.add_argument("--categories", type=str, default="haircare,skincare,fragrance")
    parser.add_argument("--split-schemes", type=str, default="time,user")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _split_metrics(frame: pd.DataFrame, predictions: list[str], prob_df: pd.DataFrame, labels: list[str]) -> dict[str, Any]:
    y_true = [str(item) for item in frame["label"].astype(str).tolist()]
    stop_indices = [idx for idx, value in enumerate(y_true) if value == "__stop__"]
    continue_indices = [idx for idx, value in enumerate(y_true) if value != "__stop__"]
    stop_true = [y_true[idx] for idx in stop_indices]
    stop_pred = [predictions[idx] for idx in stop_indices]
    cont_true = [y_true[idx] for idx in continue_indices]
    cont_pred = [predictions[idx] for idx in continue_indices]
    return {
        "rows": int(len(frame)),
        "acc_at_1": round(accuracy(y_true, predictions), 6),
        "recall_at_3": round(recall_at_k(prob_df, y_true, 3), 6),
        "stop_acc": round(accuracy(stop_true, stop_pred), 6) if stop_true else 0.0,
        "per_label": per_label_stats(y_true, predictions, labels),
        "confusion_matrix": confusion_matrix(y_true, predictions, labels),
        "non_stop_accuracy": round(accuracy(cont_true, cont_pred), 6) if cont_true else 0.0,
    }


def _sequence_metrics(frame: pd.DataFrame, *, category: str, model_root: Path, episode_targets: dict[int, list[str]]) -> dict[str, Any]:
    if frame.empty:
        return {
            "rows": 0,
            "exact_full_plan_match": 0.0,
            "prefix_match_rate": 0.0,
            "length_mae": 0.0,
        }
    exact: list[int] = []
    prefix: list[float] = []
    length_mae: list[float] = []
    per_position_hits: dict[int, list[int]] = defaultdict(list)
    for row in frame.itertuples(index=False):
        target = list(episode_targets.get(int(getattr(row, "episode_id")), []))
        predicted = rollout_live_plan(category, row._asdict(), model_root=model_root)
        exact.append(sequence_exact_match(predicted, target))
        prefix.append(longest_common_prefix_rate(predicted, target))
        length_mae.append(abs(len(predicted) - len(target)))
        width = max(len(predicted), len(target))
        for idx in range(width):
            pred = predicted[idx] if idx < len(predicted) else "__stop__"
            truth = target[idx] if idx < len(target) else "__stop__"
            per_position_hits[idx + 1].append(int(str(pred) == str(truth)))
    return {
        "rows": int(len(frame)),
        "exact_full_plan_match": round(float(np.mean(exact)), 6),
        "prefix_match_rate": round(float(np.mean(prefix)), 6),
        "length_mae": round(float(np.mean(length_mae)), 6),
        "per_position_accuracy": {
            str(position): round(float(np.mean(values)), 6)
            for position, values in sorted(per_position_hits.items())
        },
    }


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Roadmap Live Initial Planner Evaluation",
        "",
        f"- dataset_dir: `{report['dataset_dir']}`",
        f"- transitions_dir: `{report['transitions_dir']}`",
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
                    f"- val acc@1/recall@3/stop_acc: `{payload['decision_metrics']['val']['acc_at_1']:.4f}` / `{payload['decision_metrics']['val']['recall_at_3']:.4f}` / `{payload['decision_metrics']['val']['stop_acc']:.4f}`",
                    f"- test acc@1/recall@3/stop_acc: `{payload['decision_metrics']['test']['acc_at_1']:.4f}` / `{payload['decision_metrics']['test']['recall_at_3']:.4f}` / `{payload['decision_metrics']['test']['stop_acc']:.4f}`",
                    f"- val exact/prefix/length_mae: `{payload['sequence_metrics']['val']['exact_full_plan_match']:.4f}` / `{payload['sequence_metrics']['val']['prefix_match_rate']:.4f}` / `{payload['sequence_metrics']['val']['length_mae']:.4f}`",
                    f"- test exact/prefix/length_mae: `{payload['sequence_metrics']['test']['exact_full_plan_match']:.4f}` / `{payload['sequence_metrics']['test']['prefix_match_rate']:.4f}` / `{payload['sequence_metrics']['test']['length_mae']:.4f}`",
                ]
            )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    ensure_dependencies()
    args = parse_args()
    data_dir = resolve_path(str(args.data_dir))
    transitions_dir = resolve_path(str(args.transitions_dir))
    model_root = resolve_path(str(args.model_root))
    report_md_path = resolve_path(str(args.report_md))
    report_json_path = resolve_path(str(args.report_json))
    report_md_path.parent.mkdir(parents=True, exist_ok=True)
    report_json_path.parent.mkdir(parents=True, exist_ok=True)

    initial_df, initial_metadata, _ = load_live_dataset_bundle(data_dir)
    transitions_df, _transition_metadata, _ = load_live_dataset_bundle(transitions_dir)
    categories = selected_categories(args.categories, allowed=ALLOWED_CATEGORIES)
    split_schemes = selected_split_schemes(args.split_schemes)

    report: dict[str, Any] = {
        "dataset_dir": str(data_dir),
        "transitions_dir": str(transitions_dir),
        "model_root": str(model_root),
        "dataset_version": str(initial_metadata.get("version") or ""),
        "split_schemes": {},
    }

    for scheme in split_schemes:
        scheme_root = model_root / scheme
        scheme_report: dict[str, Any] = {"categories": {}}
        all_frames: list[pd.DataFrame] = []
        for category in categories:
            decisions_df = build_live_decision_dataframe(
                dataset_df=initial_df,
                category=category,
                split_scheme=scheme,
                seed=int(args.seed),
                continuation_only=False,
            )
            if decisions_df.empty:
                continue
            all_frames.append(decisions_df)
            bundle = load_model_artifact(category, model_root=scheme_root)
            split_map = split_frames(decisions_df)
            action_space = list(bundle.get("action_space") or [])
            category_report: dict[str, Any] = {
                "rows": {name: int(len(frame)) for name, frame in split_map.items()},
                "decision_metrics": {},
                "sequence_metrics": {},
            }
            episode_targets, _decision_suffix = episode_targets_from_transitions(transitions_df, category=category)
            for split_name in ("val", "test"):
                frame = split_map[split_name].sort_values(["t0_utc", "decision_id"]).reset_index(drop=True)
                prob_df = predict_bundle_probabilities(bundle, frame) if not frame.empty else pd.DataFrame(columns=action_space)
                predictions = prob_df.idxmax(axis=1).tolist() if not prob_df.empty else []
                category_report["decision_metrics"][split_name] = _split_metrics(frame, predictions, prob_df, action_space)
                category_report["sequence_metrics"][split_name] = _sequence_metrics(
                    frame,
                    category=category,
                    model_root=scheme_root,
                    episode_targets=episode_targets,
                )
            scheme_report["categories"][category] = category_report
        combined = pd.concat(all_frames, ignore_index=True) if all_frames else pd.DataFrame(columns=["user_id", "eval_split"])
        scheme_report["user_overlap"] = split_user_overlap(combined) if not combined.empty else {"train_val": 0, "train_test": 0, "val_test": 0}
        report["split_schemes"][scheme] = scheme_report

    report_json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    report_md_path.write_text(_markdown(report), encoding="utf-8")
    print(f"[eval_roadmap_live_initial_planner] json={report_json_path}")
    print(f"[eval_roadmap_live_initial_planner] md={report_md_path}")


if __name__ == "__main__":
    main()
