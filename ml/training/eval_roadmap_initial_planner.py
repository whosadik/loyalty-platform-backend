from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    from .roadmap_initial_planner_common import (
        ACTION_SPACE_BY_CATEGORY,
        STOP_TOKEN,
        build_decision_state_dataframe,
        load_teacher_dataset,
        longest_common_prefix_rate,
        majority_label_baseline,
        predict_action_probabilities,
        previous_step_prior_map,
        resolve_path,
        sequence_exact_match,
    )
except ImportError:  # pragma: no cover
    from roadmap_initial_planner_common import (
        ACTION_SPACE_BY_CATEGORY,
        STOP_TOKEN,
        build_decision_state_dataframe,
        load_teacher_dataset,
        longest_common_prefix_rate,
        majority_label_baseline,
        predict_action_probabilities,
        previous_step_prior_map,
        resolve_path,
        sequence_exact_match,
    )

REPO_ROOT = Path(__file__).resolve().parents[2]
BACKEND_DIR = REPO_ROOT / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from roadmap_app.ml_initial_planner import load_initial_planner, rollout_initial_plan  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="tmp/roadmap_teacher_v1")
    parser.add_argument("--model-root", type=str, default="models/roadmap_initial_planner")
    parser.add_argument("--report-md", type=str, default="reports/roadmap_initial_planner_eval.md")
    parser.add_argument("--report-json", type=str, default="reports/roadmap_initial_planner_eval.json")
    parser.add_argument("--categories", type=str, default="")
    return parser.parse_args()


def selected_categories(raw: str) -> list[str]:
    if not str(raw or "").strip():
        return list(ACTION_SPACE_BY_CATEGORY.keys())
    out: list[str] = []
    for item in str(raw).split(","):
        token = str(item or "").strip().lower()
        if token and token not in out:
            out.append(token)
    return out


def _accuracy(y_true: list[str], y_pred: list[str]) -> float:
    if not y_true:
        return 0.0
    hits = sum(int(str(a) == str(b)) for a, b in zip(y_true, y_pred))
    return float(hits / max(1, len(y_true)))


def _recall_at_k(prob_df: pd.DataFrame, y_true: list[str], k: int) -> float:
    if prob_df.empty:
        return 0.0
    columns = list(prob_df.columns)
    values = prob_df.to_numpy(dtype=float)
    out = 0
    width = max(1, min(int(k), values.shape[1]))
    order = np.argsort(-values, axis=1)[:, :width]
    for idx, truth in enumerate(y_true):
        labels = {columns[pos] for pos in order[idx]}
        out += int(str(truth) in labels)
    return float(out / max(1, len(y_true)))


def _per_position_accuracy(df: pd.DataFrame, predictions: list[str]) -> dict[str, float]:
    grouped: dict[int, list[tuple[str, str]]] = defaultdict(list)
    for row, pred in zip(df.itertuples(index=False), predictions):
        grouped[int(getattr(row, "position", 0) or 0)].append((str(getattr(row, "label", "")), str(pred)))
    return {
        str(position): _accuracy([truth for truth, _pred in pairs], [pred for _truth, pred in pairs])
        for position, pairs in sorted(grouped.items())
    }


def _confusion(y_true: list[str], y_pred: list[str], labels: list[str]) -> dict[str, dict[str, int]]:
    matrix: dict[str, dict[str, int]] = {
        truth: {pred: 0 for pred in labels}
        for truth in labels
    }
    for truth, pred in zip(y_true, y_pred):
        truth_key = str(truth)
        pred_key = str(pred)
        matrix.setdefault(truth_key, {key: 0 for key in labels})
        matrix[truth_key].setdefault(pred_key, 0)
        matrix[truth_key][pred_key] += 1
    return matrix


def _split_metrics(df: pd.DataFrame, predictions: list[str], prob_df: pd.DataFrame, labels: list[str]) -> dict[str, Any]:
    y_true = [str(item) for item in df["label"].astype(str).tolist()]
    stop_rows = [idx for idx, label in enumerate(y_true) if label == STOP_TOKEN]
    stop_true = [y_true[idx] for idx in stop_rows]
    stop_pred = [predictions[idx] for idx in stop_rows]
    return {
        "rows": int(len(df)),
        "next_action_accuracy_at_1": round(_accuracy(y_true, predictions), 6),
        "next_action_recall_at_3": round(_recall_at_k(prob_df, y_true, 3), 6),
        "stop_accuracy": round(_accuracy(stop_true, stop_pred), 6) if stop_true else 0.0,
        "per_position_accuracy": _per_position_accuracy(df, predictions),
        "confusion_matrix": _confusion(y_true, predictions, labels),
    }


def _baseline_predictions(df: pd.DataFrame, *, train_df: pd.DataFrame) -> dict[str, list[str]]:
    majority = majority_label_baseline(train_df)
    prev_map = previous_step_prior_map(train_df)
    majority_preds = [majority for _ in range(len(df))]
    prev_preds = [prev_map.get(str(value or "__none__"), majority) for value in df["prev_step_1"].astype(str).tolist()]
    return {
        "majority_action": majority_preds,
        "prev_step_prior": prev_preds,
    }


def _sequence_rollout_metrics(sequence_df: pd.DataFrame, *, category: str, model_root: Path) -> dict[str, Any]:
    if sequence_df.empty:
        return {"rows": 0, "target_length_mae": 0.0, "exact_full_sequence_match_rate": 0.0, "prefix_match_rate": 0.0}
    target_lengths: list[int] = []
    predicted_lengths: list[int] = []
    exact: list[int] = []
    prefix_rates: list[float] = []
    per_position_hits: dict[int, list[int]] = defaultdict(list)
    for row in sequence_df.itertuples(index=False):
        target = json.loads(str(getattr(row, "target_sequence_json")))
        predicted = rollout_initial_plan(category, row._asdict(), model_root=model_root)
        target_lengths.append(len(target))
        predicted_lengths.append(len(predicted))
        exact.append(sequence_exact_match(predicted, target))
        prefix_rates.append(longest_common_prefix_rate(predicted, target))
        for position in range(max(len(target), len(predicted))):
            target_step = target[position] if position < len(target) else STOP_TOKEN
            pred_step = predicted[position] if position < len(predicted) else STOP_TOKEN
            per_position_hits[position + 1].append(int(str(target_step) == str(pred_step)))
    return {
        "rows": int(len(sequence_df)),
        "target_length_mae": round(float(np.mean(np.abs(np.asarray(predicted_lengths) - np.asarray(target_lengths)))), 6),
        "exact_full_sequence_match_rate": round(float(np.mean(exact)), 6),
        "prefix_match_rate": round(float(np.mean(prefix_rates)), 6),
        "per_position_sequence_accuracy": {
            str(position): round(float(np.mean(values)), 6)
            for position, values in sorted(per_position_hits.items())
        },
    }


def _baseline_sequence_rollout_metrics(sequence_df: pd.DataFrame, *, category: str, train_df: pd.DataFrame) -> dict[str, Any]:
    if sequence_df.empty:
        return {
            "majority_action": {"rows": 0, "exact_full_sequence_match_rate": 0.0, "prefix_match_rate": 0.0},
            "prev_step_prior": {"rows": 0, "exact_full_sequence_match_rate": 0.0, "prefix_match_rate": 0.0},
        }
    position_majority: dict[int, str] = {}
    grouped = train_df.groupby("position")["label"].apply(lambda series: Counter(series.astype(str)).most_common(1)[0][0] if len(series) else STOP_TOKEN)
    for position, label in grouped.items():
        position_majority[int(position)] = str(label)
    prev_map = previous_step_prior_map(train_df)
    overall_majority = majority_label_baseline(train_df)
    action_space = [token for token in ACTION_SPACE_BY_CATEGORY.get(category, []) if token != STOP_TOKEN]

    def _rollout(mode: str, row_dict: dict[str, Any]) -> list[str]:
        prefix: list[str] = []
        max_steps = len(action_space)
        for position in range(1, max_steps + 1):
            if mode == "majority_action":
                candidate = position_majority.get(position, overall_majority)
            else:
                prev = prefix[-1] if prefix else "__none__"
                candidate = prev_map.get(prev, position_majority.get(position, overall_majority))
            if candidate == STOP_TOKEN:
                break
            if candidate not in action_space or candidate in prefix:
                break
            prefix.append(candidate)
        return prefix

    report: dict[str, Any] = {}
    for mode in ["majority_action", "prev_step_prior"]:
        exact: list[int] = []
        prefix_rates: list[float] = []
        for row in sequence_df.itertuples(index=False):
            target = json.loads(str(getattr(row, "target_sequence_json")))
            predicted = _rollout(mode, row._asdict())
            exact.append(sequence_exact_match(predicted, target))
            prefix_rates.append(longest_common_prefix_rate(predicted, target))
        report[mode] = {
            "rows": int(len(sequence_df)),
            "exact_full_sequence_match_rate": round(float(np.mean(exact)), 6),
            "prefix_match_rate": round(float(np.mean(prefix_rates)), 6),
        }
    return report


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Roadmap Initial Planner Evaluation",
        "",
        f"- dataset_dir: `{report['dataset_dir']}`",
        f"- model_root: `{report['model_root']}`",
        "",
    ]
    for category, payload in sorted((report.get("categories") or {}).items()):
        lines.append(f"## {category}")
        lines.append(f"- train/val/test decision rows: `{payload['rows']['train']}/{payload['rows']['val']}/{payload['rows']['test']}`")
        for split_name in ["val", "test"]:
            row = payload["action_metrics"][split_name]
            lines.append(
                f"- {split_name}: acc@1=`{row['next_action_accuracy_at_1']:.4f}` "
                f"recall@3=`{row['next_action_recall_at_3']:.4f}` stop_acc=`{row['stop_accuracy']:.4f}`"
            )
        for split_name in ["val", "test"]:
            row = payload["sequence_metrics"][split_name]
            lines.append(
                f"- {split_name} sequence: exact=`{row['exact_full_sequence_match_rate']:.4f}` "
                f"prefix=`{row['prefix_match_rate']:.4f}` length_mae=`{row['target_length_mae']:.4f}`"
            )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    args = parse_args()
    data_dir = resolve_path(str(args.data_dir))
    model_root = resolve_path(str(args.model_root))
    report_md_path = resolve_path(str(args.report_md))
    report_json_path = resolve_path(str(args.report_json))
    report_md_path.parent.mkdir(parents=True, exist_ok=True)
    report_json_path.parent.mkdir(parents=True, exist_ok=True)

    stepwise_df, sequence_df, dataset_metadata, _splits = load_teacher_dataset(data_dir)
    categories = selected_categories(args.categories)
    report: dict[str, Any] = {
        "dataset_dir": str(data_dir),
        "model_root": str(model_root),
        "categories": {},
        "notes": [
            "teacher dataset evaluation only; no runtime telemetry used",
            "teacher-policy oracle baseline is omitted because it would trivially match labels by construction",
        ],
    }

    for category in categories:
        decisions_df = build_decision_state_dataframe(stepwise_df=stepwise_df, category=category, metadata=dataset_metadata)
        if decisions_df.empty:
            continue
        bundle = load_initial_planner(category, model_root=model_root)
        split_map = {
            "train": decisions_df[decisions_df["split"].astype(str) == "train"].copy(),
            "val": decisions_df[decisions_df["split"].astype(str) == "val"].copy(),
            "test": decisions_df[decisions_df["split"].astype(str) == "test"].copy(),
        }
        feature_columns = list(bundle.get("feature_columns") or [])
        action_space = list(bundle.get("action_space") or ACTION_SPACE_BY_CATEGORY.get(category) or [])
        category_report: dict[str, Any] = {
            "rows": {name: int(len(frame)) for name, frame in split_map.items()},
            "action_space": action_space,
            "action_metrics": {},
            "baseline_action_metrics": {},
            "sequence_metrics": {},
            "baseline_sequence_metrics": {},
        }

        for split_name in ["val", "test"]:
            frame = split_map[split_name].sort_values(["planning_id", "position"]).reset_index(drop=True)
            prob_df = predict_action_probabilities(bundle, frame[feature_columns]) if not frame.empty else pd.DataFrame(columns=action_space)
            predictions = prob_df.idxmax(axis=1).tolist() if not prob_df.empty else []
            category_report["action_metrics"][split_name] = _split_metrics(frame, predictions, prob_df, action_space)
            baseline_predictions = _baseline_predictions(frame, train_df=split_map["train"])
            category_report["baseline_action_metrics"][split_name] = {
                name: {
                    "next_action_accuracy_at_1": round(_accuracy(frame["label"].astype(str).tolist(), preds), 6),
                    "stop_accuracy": round(
                        _accuracy(
                            [label for label in frame["label"].astype(str).tolist() if label == STOP_TOKEN],
                            [pred for label, pred in zip(frame["label"].astype(str).tolist(), preds) if label == STOP_TOKEN],
                        ),
                        6,
                    )
                    if len(frame)
                    else 0.0,
                }
                for name, preds in baseline_predictions.items()
            }

            seq_frame = sequence_df[
                (sequence_df["category"].astype(str).str.lower() == category)
                & (sequence_df["split"].astype(str) == split_name)
            ].copy()
            category_report["sequence_metrics"][split_name] = _sequence_rollout_metrics(
                seq_frame,
                category=category,
                model_root=model_root,
            )
            category_report["baseline_sequence_metrics"][split_name] = _baseline_sequence_rollout_metrics(
                seq_frame,
                category=category,
                train_df=split_map["train"],
            )
            if category == "fragrance":
                category_report["fragrance_slot_confusion_matrix"] = category_report["action_metrics"][split_name]["confusion_matrix"]

        report["categories"][category] = category_report

    report_json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    report_md_path.write_text(_markdown(report), encoding="utf-8")
    print(f"[eval_roadmap_initial_planner] json={report_json_path}")
    print(f"[eval_roadmap_initial_planner] md={report_md_path}")


if __name__ == "__main__":
    main()
