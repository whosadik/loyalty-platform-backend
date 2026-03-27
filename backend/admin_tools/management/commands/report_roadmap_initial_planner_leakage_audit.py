from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

try:
    import pandas as pd
except Exception:  # pragma: no cover
    pd = None

from django.core.management.base import BaseCommand, CommandError

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ml.training.roadmap_initial_planner_common import (  # noqa: E402
    ACTION_SPACE_BY_CATEGORY,
    build_decision_state_dataframe,
    load_teacher_dataset,
    predict_action_probabilities,
    previous_step_prior_map,
    resolve_path,
)
from roadmap_app.ml_initial_planner import load_initial_planner  # noqa: E402

DIRECT_LEAK_COLUMNS = {
    "target_sequence_json",
    "target_length",
    "teacher_target_at_position",
    "teacher_policy_version",
    "teacher_policy_trace_json",
    "teacher_seed_in_target",
    "teacher_seed_target_position",
    "target_source",
    "y",
    "label",
    "candidate_type",
}
DIRECT_LEAK_PREFIXES = ("target_", "teacher_")
SUSPICIOUS_PREFIX_FEATURES = {"position", "prefix_length", "prev_step_1", "prev_step_2", "remaining_action_count"}


def _selected_categories(raw: str) -> list[str]:
    if not str(raw or "").strip():
        return list(ACTION_SPACE_BY_CATEGORY.keys())
    out: list[str] = []
    for item in str(raw).split(","):
        token = str(item or "").strip().lower()
        if token and token not in out:
            out.append(token)
    return out


def _is_direct_leak_feature(name: str) -> bool:
    token = str(name or "").strip().lower()
    return token in DIRECT_LEAK_COLUMNS or token.startswith(DIRECT_LEAK_PREFIXES)


def _accuracy(y_true: list[str], y_pred: list[str]) -> float:
    if not y_true:
        return 0.0
    hits = sum(int(str(a) == str(b)) for a, b in zip(y_true, y_pred))
    return float(hits / max(1, len(y_true)))


def _group_majority_accuracy(train_df: "pd.DataFrame", eval_df: "pd.DataFrame", keys: list[str]) -> dict[str, Any]:
    if eval_df.empty:
        return {"accuracy": 0.0, "coverage": 0.0, "group_count": 0}
    if not keys:
        majority = str(train_df["label"].astype(str).mode().iloc[0]) if not train_df.empty else "__stop__"
        return {
            "accuracy": _accuracy(eval_df["label"].astype(str).tolist(), [majority] * len(eval_df)),
            "coverage": 1.0,
            "group_count": 1,
        }

    def _row_key(row: Any) -> tuple[Any, ...]:
        return tuple(getattr(row, key) for key in keys)

    grouped: dict[tuple[Any, ...], Counter[str]] = defaultdict(Counter)
    for row in train_df[keys + ["label"]].itertuples(index=False):
        key = _row_key(row)
        grouped[key][str(getattr(row, "label"))] += 1
    default_label = str(train_df["label"].astype(str).mode().iloc[0]) if not train_df.empty else "__stop__"
    truth: list[str] = []
    pred: list[str] = []
    covered = 0
    for row in eval_df[keys + ["label"]].itertuples(index=False):
        key = _row_key(row)
        counts = grouped.get(key)
        label = default_label
        if counts:
            label = str(max(counts.items(), key=lambda item: (item[1], item[0]))[0])
            covered += 1
        truth.append(str(getattr(row, "label")))
        pred.append(label)
    return {
        "accuracy": _accuracy(truth, pred),
        "coverage": float(covered / max(1, len(eval_df))),
        "group_count": int(len(grouped)),
    }


def _prefix_signature_columns(decisions_df: "pd.DataFrame", category: str) -> list[str]:
    del category
    cols = [name for name in ["position", "prev_step_1", "prev_step_2"] if name in decisions_df.columns]
    cols.extend(sorted(col for col in decisions_df.columns if col.startswith("seen_")))
    return cols


def _deterministic_prefix_share(train_df: "pd.DataFrame", keys: list[str]) -> float:
    if train_df.empty or not keys:
        return 0.0
    grouped = train_df.groupby(keys)["label"].nunique(dropna=False)
    if grouped.empty:
        return 0.0
    deterministic_keys = set(grouped[grouped == 1].index.tolist())

    def _normalize_key(raw: Any) -> tuple[Any, ...]:
        if isinstance(raw, tuple):
            return raw
        return (raw,)

    normalized = {_normalize_key(item) for item in deterministic_keys}
    hits = 0
    for row in train_df[keys].itertuples(index=False):
        if tuple(row) in normalized:
            hits += 1
    return float(hits / max(1, len(train_df)))


def _temporal_profile_finding() -> dict[str, Any]:
    return {
        "name": "current_profile_snapshot_at_historical_anchor",
        "verdict": "SUSPICIOUS",
        "block": "backend/admin_tools/roadmap_teacher.py:385",
        "why": (
            "Teacher dataset loads CustomerProfile at build time for historical anchors. "
            "This is not direct label leakage, but it is temporal contamination because profile fields are not versioned at anchor time."
        ),
        "minimal_fix": (
            "Persist profile snapshots at anchor time or restrict training to profile fields known to be stable across time."
        ),
    }


def _category_findings(
    *,
    category: str,
    selected_features: list[str],
    model_metrics: dict[str, float],
    shortcut_metrics: dict[str, Any],
) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    direct_selected = [name for name in selected_features if _is_direct_leak_feature(name)]
    if direct_selected:
        findings.append(
            {
                "name": "direct_teacher_label_feature_selected",
                "verdict": "LEAKING",
                "features": direct_selected,
                "why": "Direct target/teacher fields are present in the trained feature list.",
                "minimal_fix": "Remove all target_/teacher_/candidate/y columns from feature selection and retrain.",
            }
        )

    suspicious_selected = [name for name in selected_features if name in SUSPICIOUS_PREFIX_FEATURES or name.startswith("seen_")]
    prefix_acc = float((shortcut_metrics.get("prefix_signature") or {}).get("test_accuracy") or 0.0)
    prev_acc = float((shortcut_metrics.get("prev_step_prior") or {}).get("test_accuracy") or 0.0)
    seed_acc = float((shortcut_metrics.get("seed_only") or {}).get("test_accuracy") or 0.0)
    model_acc = float(model_metrics.get("test_accuracy") or 0.0)
    if suspicious_selected and (prefix_acc >= 0.99 or prev_acc >= 0.98 or seed_acc >= 0.98 or model_acc >= 0.995):
        findings.append(
            {
                "name": "prefix_state_shortcut_risk",
                "verdict": "SUSPICIOUS",
                "features": suspicious_selected,
                "why": (
                    "The teacher policy is close to deterministic once prefix-state fields are known. "
                    "That makes very high accuracy weak evidence of generalization."
                ),
                "minimal_fix": (
                    "Add an ablation without remaining_action_count/full seen_* mask and report shortcut baselines alongside model accuracy."
                ),
            }
        )

    return findings


def _category_verdict(findings: list[dict[str, Any]]) -> str:
    if any(str(item.get("verdict")) == "LEAKING" for item in findings):
        return "LEAKING"
    if findings:
        return "SUSPICIOUS"
    return "SAFE"


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Roadmap Initial Planner Leakage Audit",
        "",
        f"- dataset_dir: `{report['dataset_dir']}`",
        f"- model_root: `{report['model_root']}`",
        f"- overall_verdict: **{report['overall_verdict']}**",
        "",
        "## Global Findings",
    ]
    for finding in report.get("global_findings") or []:
        lines.append(f"- {finding['verdict']}: `{finding['name']}` - {finding['why']}")
    lines.extend(["", "## Categories"])
    for category, payload in sorted((report.get("categories") or {}).items()):
        lines.append(f"### {category}")
        lines.append(f"- verdict: **{payload['verdict']}**")
        lines.append(
            f"- model acc val/test: `{payload['model_metrics']['val_accuracy']:.4f}` / `{payload['model_metrics']['test_accuracy']:.4f}`"
        )
        lines.append(
            f"- shortcut acc test: seed_only=`{payload['shortcut_metrics']['seed_only']['test_accuracy']:.4f}` "
            f"position_only=`{payload['shortcut_metrics']['position_only']['test_accuracy']:.4f}` "
            f"prev_step_prior=`{payload['shortcut_metrics']['prev_step_prior']['test_accuracy']:.4f}` "
            f"prefix_signature=`{payload['shortcut_metrics']['prefix_signature']['test_accuracy']:.4f}`"
        )
        lines.append(
            f"- direct leak columns in dataset but excluded from training: `{', '.join(payload['excluded_direct_leak_columns']) or 'none'}`"
        )
        if payload.get("findings"):
            for finding in payload["findings"]:
                lines.append(f"- {finding['verdict']}: `{finding['name']}` - {finding['why']}")
        else:
            lines.append("- SAFE: no direct leakage or strong shortcut signature detected")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


class Command(BaseCommand):
    help = "Audit roadmap initial planner teacher-imitation baseline for leakage and shortcut learning."

    def add_arguments(self, parser):
        parser.add_argument("--dataset-dir", type=str, default="tmp/roadmap_teacher_v1")
        parser.add_argument("--model-root", type=str, default="models/roadmap_initial_planner")
        parser.add_argument("--report-md", type=str, default="reports/roadmap_initial_planner_leakage_audit.md")
        parser.add_argument("--report-json", type=str, default="reports/roadmap_initial_planner_leakage_audit.json")
        parser.add_argument("--categories", type=str, default="")

    def handle(self, *args, **options):
        if pd is None:
            raise CommandError("pandas is required")

        dataset_dir = resolve_path(str(options["dataset_dir"]))
        model_root = resolve_path(str(options["model_root"]))
        report_md_path = resolve_path(str(options["report_md"]))
        report_json_path = resolve_path(str(options["report_json"]))
        report_md_path.parent.mkdir(parents=True, exist_ok=True)
        report_json_path.parent.mkdir(parents=True, exist_ok=True)

        stepwise_df, _sequence_df, dataset_metadata, _splits = load_teacher_dataset(dataset_dir)
        categories = _selected_categories(str(options["categories"]))

        global_findings = [_temporal_profile_finding()]
        overlap = dict((dataset_metadata.get("split_user_overlap_counts") or {}))
        overlap_total = int(sum(int(value or 0) for value in overlap.values()))
        if overlap_total > 0:
            global_findings.append(
                {
                    "name": "user_overlap_between_splits",
                    "verdict": "LEAKING",
                    "why": f"user leakage across splits detected: {overlap}",
                    "minimal_fix": "Rebuild the teacher dataset with strict user-level splits.",
                }
            )

        report: dict[str, Any] = {
            "dataset_dir": str(dataset_dir),
            "model_root": str(model_root),
            "split_user_overlap_counts": overlap,
            "global_findings": global_findings,
            "categories": {},
        }

        overall_verdict = "SAFE"
        for category in categories:
            decisions_df = build_decision_state_dataframe(stepwise_df=stepwise_df, category=category, metadata=dataset_metadata)
            if decisions_df.empty:
                continue
            bundle = load_initial_planner(category, model_root=model_root)
            feature_columns = [str(item) for item in (bundle.get("feature_columns") or []) if str(item)]

            split_map = {
                "train": decisions_df[decisions_df["split"].astype(str) == "train"].copy(),
                "val": decisions_df[decisions_df["split"].astype(str) == "val"].copy(),
                "test": decisions_df[decisions_df["split"].astype(str) == "test"].copy(),
            }
            val_prob = predict_action_probabilities(bundle, split_map["val"][feature_columns]) if not split_map["val"].empty else pd.DataFrame()
            test_prob = predict_action_probabilities(bundle, split_map["test"][feature_columns]) if not split_map["test"].empty else pd.DataFrame()
            val_pred = val_prob.idxmax(axis=1).tolist() if not val_prob.empty else []
            test_pred = test_prob.idxmax(axis=1).tolist() if not test_prob.empty else []
            model_metrics = {
                "val_accuracy": _accuracy(split_map["val"]["label"].astype(str).tolist(), val_pred),
                "test_accuracy": _accuracy(split_map["test"]["label"].astype(str).tolist(), test_pred),
            }

            seed_cols = [name for name in ["seed_action_token"] if name in decisions_df.columns]
            pos_cols = [name for name in ["position"] if name in decisions_df.columns]
            prefix_cols = _prefix_signature_columns(decisions_df, category)

            prev_map = previous_step_prior_map(split_map["train"])
            majority = str(split_map["train"]["label"].astype(str).mode().iloc[0]) if not split_map["train"].empty else "__stop__"
            prev_preds = [prev_map.get(str(item or "__none__"), majority) for item in split_map["test"]["prev_step_1"].astype(str).tolist()]
            shortcut_metrics = {
                "seed_only": {
                    "val_accuracy": _group_majority_accuracy(split_map["train"], split_map["val"], seed_cols)["accuracy"],
                    "test_accuracy": _group_majority_accuracy(split_map["train"], split_map["test"], seed_cols)["accuracy"],
                },
                "position_only": {
                    "val_accuracy": _group_majority_accuracy(split_map["train"], split_map["val"], pos_cols)["accuracy"],
                    "test_accuracy": _group_majority_accuracy(split_map["train"], split_map["test"], pos_cols)["accuracy"],
                },
                "prev_step_prior": {
                    "val_accuracy": _accuracy(
                        split_map["val"]["label"].astype(str).tolist(),
                        [prev_map.get(str(item or "__none__"), majority) for item in split_map["val"]["prev_step_1"].astype(str).tolist()],
                    ),
                    "test_accuracy": _accuracy(split_map["test"]["label"].astype(str).tolist(), prev_preds),
                },
                "prefix_signature": {
                    "val_accuracy": _group_majority_accuracy(split_map["train"], split_map["val"], prefix_cols)["accuracy"],
                    "test_accuracy": _group_majority_accuracy(split_map["train"], split_map["test"], prefix_cols)["accuracy"],
                    "train_row_deterministic_share": _deterministic_prefix_share(split_map["train"], prefix_cols),
                },
            }

            dataset_columns = [str(col) for col in decisions_df.columns]
            excluded_direct_leak_columns = sorted(
                col for col in dataset_columns if _is_direct_leak_feature(col) and col not in set(feature_columns)
            )
            findings = _category_findings(
                category=category,
                selected_features=feature_columns,
                model_metrics=model_metrics,
                shortcut_metrics=shortcut_metrics,
            )
            verdict = _category_verdict(findings)
            if verdict == "LEAKING":
                overall_verdict = "LEAKING"
            elif verdict == "SUSPICIOUS" and overall_verdict != "LEAKING":
                overall_verdict = "SUSPICIOUS"

            report["categories"][category] = {
                "verdict": verdict,
                "feature_count": int(len(feature_columns)),
                "selected_feature_columns": feature_columns,
                "excluded_direct_leak_columns": excluded_direct_leak_columns,
                "model_metrics": model_metrics,
                "shortcut_metrics": shortcut_metrics,
                "findings": findings,
            }

        if overall_verdict != "LEAKING" and any(str(item.get("verdict")) == "SUSPICIOUS" for item in global_findings):
            overall_verdict = "SUSPICIOUS"
        report["overall_verdict"] = overall_verdict

        report_json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        report_md_path.write_text(_markdown(report), encoding="utf-8")
        self.stdout.write(f"[report_roadmap_initial_planner_leakage_audit] json={report_json_path}")
        self.stdout.write(f"[report_roadmap_initial_planner_leakage_audit] md={report_md_path}")
