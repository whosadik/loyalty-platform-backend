from __future__ import annotations

import json
from datetime import timedelta
from copy import deepcopy
from pathlib import Path
from typing import Any

from django.conf import settings
from django.db.utils import DatabaseError, OperationalError
from django.utils import timezone

from roadmap_app.ml_artifact_proof import (
    PROOF_FILE_EVAL,
    PROOF_FILE_METADATA,
    PROOF_FILE_SHADOW,
    PROOF_FILE_UPLIFT_30D,
    PROOF_FILE_UPLIFT_7D,
    artifact_file_path,
    load_json_file,
    proof_bundle_status,
)
from roadmap_app.ml_next_step import nextstep_model_artifact_summary
from roadmap_app.nextstep_decision_quality import build_nextstep_v4_decision_quality_payload
from roadmap_app.nextstep_historical_anchor_context import (
    HistoricalAnchorReadError,
    build_historical_anchor_read_context,
)
from roadmap_app.nextstep_haircare_shampoo_truth_design import (
    build_nextstep_haircare_shampoo_truth_design_payload,
)


POLICY_VERSION = "roadmap_nextstep_v4_targeted_retrain_v1"
DEFAULT_TARGETED_DATA_DIR = Path("data") / "ml" / "roadmap_nextstep_v4_targeted_retrain_v1"
DEFAULT_TARGETED_MODEL_DIR = Path("models") / "roadmap_next_step_v4_targeted_retrain_v1"
DEFAULT_COMPARE_REPORT_STEM = Path("reports") / "roadmap_nextstep_targeted_retrain_comparison"
DEFAULT_HISTORICAL_ANCHOR_COMPARE_REPORT_STEM = (
    Path("reports") / "roadmap_nextstep_v5_historical_anchor_targeted_v1_comparison"
)
DEFAULT_HISTORICAL_ANCHOR_COMPARE_REPORT_JSON = DEFAULT_HISTORICAL_ANCHOR_COMPARE_REPORT_STEM.with_suffix(
    ".json"
)
DEFAULT_TWO_STAGE_GATE_RERUN_REPORT_STEM = (
    Path("reports") / "roadmap_nextstep_v5_gate_rerun_under_two_stage_truth"
)
DEFAULT_BROADER_QUALIFICATION_RERUN_REPORT_STEM = (
    Path("reports") / "roadmap_nextstep_v5_broader_qualification_rerun"
)
SOURCE_PREFERENCE_CHOICES = ["auto", "fresh_db", "cached_artifact"]


TARGETED_TRUTH_SLICES = [
    {
        "category": "skincare",
        "truth_product_type": "mask",
        "probable_causes": [
            "class_imbalance",
            "missing_or_weak_features",
            "baseline_biased_training_target",
        ],
        "diagnosis": "Mask is a tail treatment step; true positives exist, but the model under-ranks them and leaks toward essence/moisturizer-like substitutes.",
    },
    {
        "category": "skincare",
        "truth_product_type": "toner",
        "probable_causes": [
            "class_imbalance",
            "missing_or_weak_features",
            "baseline_biased_training_target",
        ],
        "diagnosis": "Toner loses to moisturizer/mask style alternatives, which points to weak separation for light-prep steps rather than candidate absence.",
    },
    {
        "category": "skincare",
        "truth_product_type": "eye_cream",
        "probable_causes": [
            "class_imbalance",
            "missing_or_weak_features",
        ],
        "diagnosis": "Eye cream is present in candidate sets but gets confused with serum-style actives; this looks like weak slice signal, not missing labels.",
    },
    {
        "category": "haircare",
        "truth_product_type": "shampoo",
        "probable_causes": [
            "baseline_biased_training_target",
            "class_imbalance",
            "threshold_or_calibration_issue",
        ],
        "diagnosis": "The model over-advances to conditioner when the recovered truth still resolves to shampoo, which suggests transition bias and weak repeat-shampoo calibration.",
    },
]


TARGETED_DISAGREEMENT_PAIRS = [
    {
        "category": "haircare",
        "baseline_product_type": "shampoo",
        "model_product_type": "conditioner",
        "probable_causes": [
            "baseline_biased_training_target",
            "threshold_or_calibration_issue",
        ],
        "diagnosis": "This is the clearest harmful disagreement in haircare: the model advances to conditioner when baseline and truth still point to shampoo.",
    },
]


PROTECTED_TRUTH_SLICES = [
    {
        "category": "haircare",
        "truth_product_type": "hair_mask",
        "diagnosis": "Promising treatment slice with positive model wins; preserve while targeting shampoo errors.",
    },
    {
        "category": "haircare",
        "truth_product_type": "hair_oil",
        "diagnosis": "Promising haircare oil slice; avoid collateral regression from stronger shampoo weighting.",
    },
    {
        "category": "skincare",
        "truth_product_type": "essence",
        "diagnosis": "Model has useful wins on essence; targeted retrain should not flatten this into baseline-like behavior.",
    },
    {
        "category": "fragrance",
        "truth_product_type": "cold_evening",
        "diagnosis": "Analysis-only protection bucket. No runtime implication in this block.",
    },
]


REWEIGHT_RULES = [
    {
        "name": "target_skincare_mask_positive",
        "kind": "target_positive",
        "category": "skincare",
        "label": "mask",
        "candidate_type": "mask",
        "y": 1,
        "multiplier": 1.8,
    },
    {
        "name": "target_skincare_mask_vs_essence_negative",
        "kind": "hard_negative",
        "category": "skincare",
        "label": "mask",
        "candidate_type": "essence",
        "y": 0,
        "multiplier": 1.45,
    },
    {
        "name": "target_skincare_toner_positive",
        "kind": "target_positive",
        "category": "skincare",
        "label": "toner",
        "candidate_type": "toner",
        "y": 1,
        "multiplier": 1.8,
    },
    {
        "name": "target_skincare_toner_vs_moisturizer_negative",
        "kind": "hard_negative",
        "category": "skincare",
        "label": "toner",
        "candidate_type": "moisturizer",
        "y": 0,
        "multiplier": 1.45,
    },
    {
        "name": "target_skincare_eyecream_positive",
        "kind": "target_positive",
        "category": "skincare",
        "label": "eye_cream",
        "candidate_type": "eye_cream",
        "y": 1,
        "multiplier": 1.8,
    },
    {
        "name": "target_skincare_eyecream_vs_serum_negative",
        "kind": "hard_negative",
        "category": "skincare",
        "label": "eye_cream",
        "candidate_type": "serum",
        "y": 0,
        "multiplier": 1.45,
    },
    {
        "name": "target_haircare_shampoo_positive",
        "kind": "target_positive",
        "category": "haircare",
        "label": "shampoo",
        "candidate_type": "shampoo",
        "y": 1,
        "multiplier": 1.55,
    },
    {
        "name": "target_haircare_shampoo_vs_conditioner_negative",
        "kind": "hard_negative",
        "category": "haircare",
        "label": "shampoo",
        "candidate_type": "conditioner",
        "y": 0,
        "multiplier": 1.75,
    },
    {
        "name": "protect_haircare_hair_mask_positive",
        "kind": "protect_positive",
        "category": "haircare",
        "label": "hair_mask",
        "candidate_type": "hair_mask",
        "y": 1,
        "multiplier": 1.15,
    },
    {
        "name": "protect_haircare_hair_oil_positive",
        "kind": "protect_positive",
        "category": "haircare",
        "label": "hair_oil",
        "candidate_type": "hair_oil",
        "y": 1,
        "multiplier": 1.15,
    },
    {
        "name": "protect_skincare_essence_positive",
        "kind": "protect_positive",
        "category": "skincare",
        "label": "essence",
        "candidate_type": "essence",
        "y": 1,
        "multiplier": 1.10,
    },
]


def _safe_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _safe_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def targeted_retrain_policy_payload() -> dict[str, Any]:
    return {
        "policy_version": POLICY_VERSION,
        "targeted_truth_slices": TARGETED_TRUTH_SLICES,
        "targeted_disagreement_pairs": TARGETED_DISAGREEMENT_PAIRS,
        "protected_truth_slices": PROTECTED_TRUTH_SLICES,
        "reweight_rules": REWEIGHT_RULES,
        "notes": [
            "No new features are introduced; only existing dataset columns are reweighted.",
            "Fragrance cold_evening is monitored as a protected analysis bucket only.",
            "Rule baseline and runtime configuration remain unchanged.",
        ],
    }


def apply_targeted_retrain_weights(df):
    if "category" not in df.columns or "label" not in df.columns or "candidate_type" not in df.columns or "y" not in df.columns:
        raise ValueError("Dataset must contain category, label, candidate_type, y columns for targeted reweighting.")

    work = df.copy()
    if "sample_weight" in work.columns:
        base_weight = work["sample_weight"].astype(float)
    else:
        base_weight = 1.0
    work["sample_weight_base"] = base_weight
    work["sample_weight_multiplier"] = 1.0
    work["targeted_retrain_bucket"] = "default"

    rule_summary: list[dict[str, Any]] = []
    for rule in REWEIGHT_RULES:
        mask = (
            work["category"].astype(str).str.strip().str.lower().eq(str(rule["category"]))
            & work["label"].astype(str).str.strip().str.lower().eq(str(rule["label"]))
            & work["candidate_type"].astype(str).str.strip().str.lower().eq(str(rule["candidate_type"]))
            & work["y"].astype(int).eq(int(rule["y"]))
        )
        matched = int(mask.sum())
        if matched > 0:
            work.loc[mask, "sample_weight_multiplier"] = (
                work.loc[mask, "sample_weight_multiplier"].astype(float) * float(rule["multiplier"])
            )
            work.loc[mask, "targeted_retrain_bucket"] = str(rule["kind"])
        rule_summary.append(
            {
                "name": str(rule["name"]),
                "kind": str(rule["kind"]),
                "matched_rows": matched,
                "multiplier": float(rule["multiplier"]),
                "category": str(rule["category"]),
                "label": str(rule["label"]),
                "candidate_type": str(rule["candidate_type"]),
                "y": int(rule["y"]),
            }
        )

    work["sample_weight"] = (
        work["sample_weight_base"].astype(float) * work["sample_weight_multiplier"].astype(float)
    ).clip(lower=0.25, upper=4.0)

    summary = {
        "rows_total": int(len(work)),
        "rows_reweighted_total": int((work["sample_weight_multiplier"].astype(float) != 1.0).sum()),
        "bucket_distribution": {
            str(k): int(v)
            for k, v in sorted(
                work["targeted_retrain_bucket"].astype(str).value_counts(dropna=False).to_dict().items(),
                key=lambda kv: (-kv[1], kv[0]),
            )
        },
        "sample_weight_base_summary": {
            "min": round(float(work["sample_weight_base"].min()), 6),
            "max": round(float(work["sample_weight_base"].max()), 6),
            "mean": round(float(work["sample_weight_base"].mean()), 6),
        },
        "sample_weight_summary": {
            "min": round(float(work["sample_weight"].min()), 6),
            "max": round(float(work["sample_weight"].max()), 6),
            "mean": round(float(work["sample_weight"].mean()), 6),
        },
        "rule_summary": rule_summary,
    }
    return work, summary


def candidate_proof_bundle_summary(model_path: str | Path | None) -> dict[str, Any]:
    model_summary = nextstep_model_artifact_summary(model_path)
    model_version = str(model_summary.get("model_version") or "")
    return proof_bundle_status(
        model_path=model_path,
        required_files=[
            PROOF_FILE_METADATA,
            PROOF_FILE_EVAL,
            PROOF_FILE_SHADOW,
            PROOF_FILE_UPLIFT_7D,
            PROOF_FILE_UPLIFT_30D,
        ],
        expected_model_version=model_version or None,
    )


def _load_eval_report(model_path: str | Path | None) -> dict[str, Any]:
    return _safe_dict(load_json_file(artifact_file_path(model_path, PROOF_FILE_EVAL)))


def _extract_category_comparison(base_payload: dict[str, Any], candidate_payload: dict[str, Any], category: str) -> dict[str, Any]:
    base_row = _safe_dict(_safe_dict(base_payload.get("per_category")).get(category))
    candidate_row = _safe_dict(_safe_dict(candidate_payload.get("per_category")).get(category))
    return {
        "category": category,
        "base_rollout_reason": str(base_row.get("rollout_reason") or ""),
        "candidate_rollout_reason": str(candidate_row.get("rollout_reason") or ""),
        "base_diagnosis": _safe_dict(base_row.get("diagnosis")),
        "candidate_diagnosis": _safe_dict(candidate_row.get("diagnosis")),
        "base_model_win_rate": base_row.get("model_win_rate_vs_truth"),
        "candidate_model_win_rate": candidate_row.get("model_win_rate_vs_truth"),
        "base_baseline_win_rate": base_row.get("baseline_win_rate_vs_truth"),
        "candidate_baseline_win_rate": candidate_row.get("baseline_win_rate_vs_truth"),
        "base_both_wrong_rate": base_row.get("both_wrong_rate"),
        "candidate_both_wrong_rate": candidate_row.get("both_wrong_rate"),
        "base_resolved_truth": int(base_row.get("resolved_truth_anchors_total", 0) or 0),
        "candidate_resolved_truth": int(candidate_row.get("resolved_truth_anchors_total", 0) or 0),
        "delta_model_win_rate": _delta(candidate_row.get("model_win_rate_vs_truth"), base_row.get("model_win_rate_vs_truth")),
        "delta_baseline_win_rate": _delta(candidate_row.get("baseline_win_rate_vs_truth"), base_row.get("baseline_win_rate_vs_truth")),
        "delta_both_wrong_rate": _delta(candidate_row.get("both_wrong_rate"), base_row.get("both_wrong_rate")),
    }


def _delta(new_value: Any, old_value: Any) -> float | None:
    try:
        if new_value is None or old_value is None:
            return None
        return round(float(new_value) - float(old_value), 6)
    except Exception:
        return None


def _float_value(value: Any, default: float) -> float:
    try:
        if value is None:
            return float(default)
        return float(value)
    except Exception:
        return float(default)


def _int_value(value: Any, default: int) -> int:
    try:
        if value is None:
            return int(default)
        return int(value)
    except Exception:
        return int(default)


def _slice_lookup(payload: dict[str, Any], *, kind: str, key: str) -> dict[str, Any]:
    direct = _safe_dict(_safe_dict(_safe_dict(payload.get("slice_analysis")).get(kind)).get(key))
    if direct:
        return direct

    category_key, _, remainder = str(key).partition(":")
    per_category = _safe_dict(payload.get("per_category"))
    category_payload = _safe_dict(per_category.get(category_key))

    if kind == "truth_slice_lookup":
        target_truth = remainder
        candidate_rows = _safe_list(category_payload.get("worst_slices")) + _safe_list(
            category_payload.get("promising_slices")
        )
        for row in candidate_rows:
            if str(_safe_dict(row).get("truth_product_type") or "") == target_truth:
                return _safe_dict(row)
        return {}

    if kind == "disagreement_pair_lookup":
        baseline_product_type, _, model_product_type = remainder.partition(":")
        candidate_rows = _safe_list(category_payload.get("worst_disagreement_pairs")) + _safe_list(
            category_payload.get("promising_disagreement_pairs")
        )
        for row in candidate_rows:
            safe_row = _safe_dict(row)
            if (
                str(safe_row.get("baseline_product_type") or "") == baseline_product_type
                and str(safe_row.get("model_product_type") or "") == model_product_type
            ):
                return safe_row
        return {}

    return {}


def _artifact_shampoo_truth_entry(
    *,
    model_path: str | Path,
    days: int,
    historical_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = build_nextstep_haircare_shampoo_truth_design_payload(
        model_path=model_path,
        reference_model_path="",
        days=days,
        include_ga=False,
        historical_context=historical_context,
    )
    return {
        "executive_verdict": _safe_dict(payload.get("executive_verdict")),
        "truth_designs": _safe_dict(_safe_dict(payload.get("candidate")).get("truth_designs")),
        "catalog_safety": _safe_dict(payload.get("catalog_safety")),
    }


def _shampoo_two_stage_snapshot(artifact: dict[str, Any]) -> dict[str, Any]:
    shampoo_payload = _safe_dict(artifact.get("shampoo_truth_design"))
    truth_designs = _safe_dict(shampoo_payload.get("truth_designs"))
    current_gate = _safe_dict(truth_designs.get("current_gate"))
    designs = _safe_dict(truth_designs.get("designs"))
    two_stage = _safe_dict(designs.get("D_two_stage_truth"))
    stage1 = _safe_dict(two_stage.get("stage_1_family"))
    stage2 = _safe_dict(two_stage.get("stage_2_concrete_step"))
    return {
        "current_gate": current_gate,
        "two_stage": two_stage,
        "stage_1_family": stage1,
        "stage_2_concrete_step": stage2,
        "stage_1_outcome": _safe_dict(stage1.get("outcome_matrix")),
        "stage_2_outcome": _safe_dict(stage2.get("outcome_matrix")),
        "resolved_anchors_total": _int_value(two_stage.get("resolved_anchors_total"), 0),
        "unresolved_anchors_total": _int_value(two_stage.get("unresolved_anchors_total"), 0),
        "unresolved_anchors_by_reason": _safe_dict(two_stage.get("unresolved_anchors_by_reason")),
        "shampoo_conditioner_observability": _safe_dict(two_stage.get("shampoo_conditioner_observability")),
        "model_vs_baseline_comparable": bool(two_stage.get("model_vs_baseline_comparable")),
        "current_gate_status": str(_safe_dict(current_gate.get("verdict")).get("status") or ""),
        "two_stage_gate_informativeness": str(two_stage.get("gate_informativeness_status") or ""),
    }


def _compare_shampoo_two_stage_rows(artifacts: dict[str, dict[str, Any]]) -> dict[str, Any]:
    rows = {}
    for label, artifact in artifacts.items():
        snapshot = _shampoo_two_stage_snapshot(artifact)
        rows[label] = {
            "current_gate_status": snapshot.get("current_gate_status"),
            "old_gate_resolved": _int_value(_safe_dict(snapshot.get("current_gate")).get("resolved_anchors_total"), 0),
            "old_gate_unresolved": _int_value(_safe_dict(snapshot.get("current_gate")).get("unresolved_anchors_total"), 0),
            "old_gate_shampoo_truth_rows": _int_value(_safe_dict(snapshot.get("current_gate")).get("standalone_shampoo_truth_rows_total"), 0),
            "old_gate_shampoo_conditioner_rows": _int_value(
                _safe_dict(snapshot.get("current_gate")).get("resolved_shampoo_conditioner_comparable_rows_total"), 0
            ),
            "two_stage_resolved": snapshot.get("resolved_anchors_total"),
            "two_stage_unresolved": snapshot.get("unresolved_anchors_total"),
            "two_stage_unresolved_by_reason": snapshot.get("unresolved_anchors_by_reason"),
            "stage_1_family_model_win_rate": _safe_dict(snapshot.get("stage_1_outcome")).get("model_win_rate"),
            "stage_1_family_baseline_win_rate": _safe_dict(snapshot.get("stage_1_outcome")).get("baseline_win_rate"),
            "stage_1_family_model_wins": _safe_dict(snapshot.get("stage_1_outcome")).get("model_wins"),
            "stage_2_concrete_model_win_rate": _safe_dict(snapshot.get("stage_2_outcome")).get("model_win_rate"),
            "stage_2_concrete_baseline_win_rate": _safe_dict(snapshot.get("stage_2_outcome")).get("baseline_win_rate"),
            "stage_2_concrete_model_wins": _safe_dict(snapshot.get("stage_2_outcome")).get("model_wins"),
            "shampoo_conditioner_observability": snapshot.get("shampoo_conditioner_observability"),
            "two_stage_gate_informativeness": snapshot.get("two_stage_gate_informativeness"),
        }
    return rows


def build_targeted_retrain_comparison_payload(
    *,
    base_model_path: str | Path,
    candidate_model_path: str | Path,
    days: int = 30,
) -> dict[str, Any]:
    now_utc = timezone.now()
    since = now_utc - timedelta(days=int(days))
    historical_context = build_historical_anchor_read_context(
        since=since,
        until=now_utc,
        category="all",
        include_ga=False,
    )
    base_decision_quality = build_nextstep_v4_decision_quality_payload(
        model_path=base_model_path,
        days=days,
        category="all",
        include_ga=False,
        min_slice_size=10,
        historical_context=historical_context,
    )
    candidate_decision_quality = build_nextstep_v4_decision_quality_payload(
        model_path=candidate_model_path,
        days=days,
        category="all",
        include_ga=False,
        min_slice_size=10,
        historical_context=historical_context,
    )
    base_eval = _load_eval_report(base_model_path)
    candidate_eval = _load_eval_report(candidate_model_path)

    targeted_truth_rows = []
    for spec in TARGETED_TRUTH_SLICES:
        key = f"{spec['category']}:{spec['truth_product_type']}"
        targeted_truth_rows.append(
            {
                **spec,
                "base": _slice_lookup(base_decision_quality, kind="truth_slice_lookup", key=key),
                "candidate": _slice_lookup(candidate_decision_quality, kind="truth_slice_lookup", key=key),
            }
        )

    targeted_pair_rows = []
    for spec in TARGETED_DISAGREEMENT_PAIRS:
        key = f"{spec['category']}:{spec['baseline_product_type']}:{spec['model_product_type']}"
        targeted_pair_rows.append(
            {
                **spec,
                "base": _slice_lookup(base_decision_quality, kind="disagreement_pair_lookup", key=key),
                "candidate": _slice_lookup(candidate_decision_quality, kind="disagreement_pair_lookup", key=key),
            }
        )

    protected_rows = []
    for spec in PROTECTED_TRUTH_SLICES:
        key = f"{spec['category']}:{spec['truth_product_type']}"
        protected_rows.append(
            {
                **spec,
                "base": _slice_lookup(base_decision_quality, kind="truth_slice_lookup", key=key),
                "candidate": _slice_lookup(candidate_decision_quality, kind="truth_slice_lookup", key=key),
            }
        )

    category_compare = [
        _extract_category_comparison(base_decision_quality, candidate_decision_quality, category)
        for category in ["haircare", "skincare", "makeup", "fragrance"]
    ]
    return {
        "policy": targeted_retrain_policy_payload(),
        "base_model_path": str(Path(str(base_model_path)).expanduser().resolve()),
        "candidate_model_path": str(Path(str(candidate_model_path)).expanduser().resolve()),
        "base_eval": base_eval,
        "candidate_eval": candidate_eval,
        "base_proof_bundle": candidate_proof_bundle_summary(base_model_path),
        "candidate_proof_bundle": candidate_proof_bundle_summary(candidate_model_path),
        "base_decision_quality": base_decision_quality,
        "candidate_decision_quality": candidate_decision_quality,
        "category_comparison": category_compare,
        "targeted_truth_slices": targeted_truth_rows,
        "targeted_disagreement_pairs": targeted_pair_rows,
        "protected_truth_slices": protected_rows,
    }


def _pct(value: Any) -> str:
    try:
        if value is None:
            return "n/a"
        return f"{float(value) * 100.0:.2f}%"
    except Exception:
        return "n/a"


def _ppt(value: Any) -> str:
    try:
        if value is None:
            return "n/a"
        return f"{float(value) * 100.0:+.2f}pp"
    except Exception:
        return "n/a"


def render_targeted_retrain_comparison_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Roadmap Nextstep Targeted Retrain Comparison",
        "",
        "## Why These Slices Were Targeted",
    ]
    for spec in _safe_list(_safe_dict(payload.get("policy")).get("targeted_truth_slices")):
        causes = ", ".join(_safe_list(spec.get("probable_causes")))
        lines.append(
            f"- `{spec.get('category')}/{spec.get('truth_product_type')}`: {spec.get('diagnosis')} "
            f"Likely causes: {causes}."
        )

    lines.extend(["", "## What Changed In Training"])
    lines.extend(
        [
            "- Strategy: slice-aware reweighting on existing continuation dataset, plus targeted hard negatives for the worst disagreement pair.",
            "- Protected buckets: haircare/hair_mask, haircare/hair_oil, skincare/essence.",
            "- Fragrance/cold_evening remained analysis-only and was not used as a runtime rollout target.",
            "",
            "## Proof Bundle Completeness",
        ]
    )
    for label, key in [("base", "base_proof_bundle"), ("candidate", "candidate_proof_bundle")]:
        proof = _safe_dict(payload.get(key))
        lines.append(
            f"- {label}: required_complete=`{proof.get('required_complete')}` reason=`{proof.get('reason')}`"
        )

    lines.extend(["", "## Category Comparison"])
    lines.append("| category | old_reason | new_reason | old_model_win | new_model_win | old_baseline_win | new_baseline_win | old_both_wrong | new_both_wrong |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for row in _safe_list(payload.get("category_comparison")):
        lines.append(
            f"| {row.get('category')} | {row.get('base_rollout_reason')} | {row.get('candidate_rollout_reason')} "
            f"| {_pct(row.get('base_model_win_rate'))} | {_pct(row.get('candidate_model_win_rate'))} "
            f"| {_pct(row.get('base_baseline_win_rate'))} | {_pct(row.get('candidate_baseline_win_rate'))} "
            f"| {_pct(row.get('base_both_wrong_rate'))} | {_pct(row.get('candidate_both_wrong_rate'))} |"
        )

    lines.extend(["", "## Targeted Slices"])
    lines.append("| slice | old_model_win | new_model_win | old_baseline_win | new_baseline_win | old_net | new_net |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- |")
    for row in _safe_list(payload.get("targeted_truth_slices")):
        old_row = _safe_dict(row.get("base"))
        new_row = _safe_dict(row.get("candidate"))
        lines.append(
            f"| {row.get('category')}/{row.get('truth_product_type')} | {_pct(old_row.get('model_win_rate_vs_truth'))} "
            f"| {_pct(new_row.get('model_win_rate_vs_truth'))} | {_pct(old_row.get('baseline_win_rate_vs_truth'))} "
            f"| {_pct(new_row.get('baseline_win_rate_vs_truth'))} | {old_row.get('net_wins_model_minus_baseline', 'n/a')} "
            f"| {new_row.get('net_wins_model_minus_baseline', 'n/a')} |"
        )

    lines.extend(["", "## Targeted Disagreement Pairs"])
    lines.append("| pair | old_model_win | new_model_win | old_baseline_win | new_baseline_win | old_net | new_net |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- |")
    for row in _safe_list(payload.get("targeted_disagreement_pairs")):
        old_row = _safe_dict(row.get("base"))
        new_row = _safe_dict(row.get("candidate"))
        pair_label = f"{row.get('category')}:{row.get('baseline_product_type')}->{row.get('model_product_type')}"
        lines.append(
            f"| {pair_label} | {_pct(old_row.get('model_win_rate_vs_truth'))} | {_pct(new_row.get('model_win_rate_vs_truth'))} "
            f"| {_pct(old_row.get('baseline_win_rate_vs_truth'))} | {_pct(new_row.get('baseline_win_rate_vs_truth'))} "
            f"| {old_row.get('net_wins_model_minus_baseline', 'n/a')} | {new_row.get('net_wins_model_minus_baseline', 'n/a')} |"
        )

    lines.extend(["", "## Protected Slices"])
    lines.append("| slice | old_model_win | new_model_win | old_net | new_net |")
    lines.append("| --- | --- | --- | --- | --- |")
    for row in _safe_list(payload.get("protected_truth_slices")):
        old_row = _safe_dict(row.get("base"))
        new_row = _safe_dict(row.get("candidate"))
        lines.append(
            f"| {row.get('category')}/{row.get('truth_product_type')} | {_pct(old_row.get('model_win_rate_vs_truth'))} "
            f"| {_pct(new_row.get('model_win_rate_vs_truth'))} | {old_row.get('net_wins_model_minus_baseline', 'n/a')} "
            f"| {new_row.get('net_wins_model_minus_baseline', 'n/a')} |"
        )

    base_eval = _safe_dict(payload.get("base_eval"))
    candidate_eval = _safe_dict(payload.get("candidate_eval"))
    base_metrics = _safe_dict(base_eval.get("metrics_test"))
    candidate_metrics = _safe_dict(candidate_eval.get("metrics_test"))
    lines.extend(["", "## Offline Eval"])
    lines.append("| artifact | recall@1 | ndcg@5 |")
    lines.append("| --- | --- | --- |")
    lines.append(
        f"| old | {_pct(base_metrics.get('recall_at_1'))} | {_pct(base_metrics.get('ndcg_at_5'))} |"
    )
    lines.append(
        f"| new | {_pct(candidate_metrics.get('recall_at_1'))} | {_pct(candidate_metrics.get('ndcg_at_5'))} |"
    )

    lines.extend(["", "## Recommendation"])
    lines.append(
        "- Candidate should continue qualification only if it improves skincare baseline gap materially and does not damage the protected haircare/skincare slices."
    )
    lines.append(
        "- If skincare remains B-like and haircare still carries large both-wrong mass, continuation should stay frozen after this retrain."
    )
    return "\n".join(lines).strip() + "\n"


def _artifact_comparison_entry(
    *,
    label: str,
    model_path: str | Path,
    days: int,
    historical_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    resolved_model_path = str(Path(str(model_path)).expanduser().resolve())
    decision_quality = build_nextstep_v4_decision_quality_payload(
        model_path=resolved_model_path,
        days=days,
        category="all",
        include_ga=False,
        min_slice_size=10,
        historical_context=historical_context,
    )
    return {
        "label": label,
        "model_path": resolved_model_path,
        "eval": _load_eval_report(resolved_model_path),
        "proof_bundle": candidate_proof_bundle_summary(resolved_model_path),
        "decision_quality": decision_quality,
        "shampoo_truth_design": _artifact_shampoo_truth_entry(
            model_path=resolved_model_path,
            days=days,
            historical_context=historical_context,
        ),
    }


def _comparison_slice_rows(
    *,
    artifacts: dict[str, dict[str, Any]],
    kind: str,
    key_builder: Any,
    specs: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for spec in specs:
        key = key_builder(spec)
        row = dict(spec)
        for label, artifact in artifacts.items():
            row[label] = _slice_lookup(_safe_dict(artifact.get("decision_quality")), kind=kind, key=key)
        out.append(row)
    return out


def _compare_category_rows(artifacts: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for category in ["haircare", "skincare", "makeup", "fragrance"]:
        row = {"category": category}
        for label, artifact in artifacts.items():
            category_payload = _safe_dict(_safe_dict(artifact.get("decision_quality")).get("per_category")).get(category) or {}
            row[label] = {
                "rollout_reason": str(category_payload.get("rollout_reason") or ""),
                "diagnosis": _safe_dict(category_payload.get("diagnosis")),
                "model_win_rate": category_payload.get("model_win_rate_vs_truth"),
                "baseline_win_rate": category_payload.get("baseline_win_rate_vs_truth"),
                "both_wrong_rate": category_payload.get("both_wrong_rate"),
                "resolved_truth": int(category_payload.get("resolved_truth_anchors_total", 0) or 0),
            }
        rows.append(row)
    return rows


def _gate_status(name: str, passed: bool, reason: str, details: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": name,
        "passed": bool(passed),
        "reason": str(reason),
        "details": details,
    }


def _acceptance_gates(artifacts: dict[str, dict[str, Any]]) -> dict[str, Any]:
    active = artifacts["active"]
    candidate = artifacts["v5_historical_anchor"]
    active_dq = _safe_dict(active.get("decision_quality"))
    candidate_dq = _safe_dict(candidate.get("decision_quality"))
    active_eval = _safe_dict(active.get("eval"))
    candidate_eval = _safe_dict(candidate.get("eval"))

    active_skincare = _safe_dict(_safe_dict(active_dq.get("per_category")).get("skincare"))
    candidate_skincare = _safe_dict(_safe_dict(candidate_dq.get("per_category")).get("skincare"))
    gate_skincare = not (
        str(candidate_skincare.get("rollout_reason") or "") == "low_uplift"
        and str(_safe_dict(candidate_skincare.get("diagnosis")).get("code") or "") == "B"
    )

    active_shampoo = _shampoo_two_stage_snapshot(active)
    candidate_shampoo = _shampoo_two_stage_snapshot(candidate)
    active_stage1 = _safe_dict(active_shampoo.get("stage_1_outcome"))
    candidate_stage1 = _safe_dict(candidate_shampoo.get("stage_1_outcome"))
    active_stage2 = _safe_dict(active_shampoo.get("stage_2_outcome"))
    candidate_stage2 = _safe_dict(candidate_shampoo.get("stage_2_outcome"))
    gate_haircare_measurable = (
        bool(active_shampoo.get("model_vs_baseline_comparable"))
        and bool(candidate_shampoo.get("model_vs_baseline_comparable"))
        and _int_value(active_shampoo.get("resolved_anchors_total"), 0) > 0
        and _int_value(candidate_shampoo.get("resolved_anchors_total"), 0) > 0
    )
    gate_haircare_unresolved_stable = (
        _int_value(candidate_shampoo.get("unresolved_anchors_total"), 10**9)
        <= _int_value(active_shampoo.get("unresolved_anchors_total"), 10**9)
    )
    gate_haircare_stage1 = (
        _float_value(candidate_stage1.get("model_win_rate"), -1.0)
        >= _float_value(active_stage1.get("model_win_rate"), -1.0)
    )
    gate_haircare_stage2 = (
        _float_value(candidate_stage2.get("model_win_rate"), -1.0)
        > _float_value(active_stage2.get("model_win_rate"), -1.0)
    )
    gate_haircare = (
        gate_haircare_measurable
        and gate_haircare_unresolved_stable
        and gate_haircare_stage1
        and gate_haircare_stage2
    )
    if not gate_haircare_measurable:
        gate_haircare_reason = "haircare_two_stage_truth_not_measurable"
    elif not gate_haircare_unresolved_stable:
        gate_haircare_reason = "haircare_two_stage_unresolved_regressed"
    elif not gate_haircare_stage1:
        gate_haircare_reason = "haircare_two_stage_stage1_regressed"
    elif not gate_haircare_stage2:
        gate_haircare_reason = "haircare_two_stage_stage2_not_improved"
    else:
        gate_haircare_reason = "passed"

    protected_results = []
    protected_passed = True
    for spec in PROTECTED_TRUTH_SLICES[:3]:
        key = f"{spec['category']}:{spec['truth_product_type']}"
        active_row = _slice_lookup(active_dq, kind="truth_slice_lookup", key=key)
        candidate_row = _slice_lookup(candidate_dq, kind="truth_slice_lookup", key=key)
        row_passed = (
            _float_value(candidate_row.get("model_win_rate_vs_truth"), 0.0)
            >= _float_value(active_row.get("model_win_rate_vs_truth"), 0.0)
            and _int_value(candidate_row.get("net_wins_model_minus_baseline"), 0)
            >= _int_value(active_row.get("net_wins_model_minus_baseline"), 0)
        )
        protected_results.append(
            {
                "slice": key,
                "passed": bool(row_passed),
                "active_model_win_rate": active_row.get("model_win_rate_vs_truth"),
                "candidate_model_win_rate": candidate_row.get("model_win_rate_vs_truth"),
                "active_net": active_row.get("net_wins_model_minus_baseline"),
                "candidate_net": candidate_row.get("net_wins_model_minus_baseline"),
            }
        )
        protected_passed = protected_passed and bool(row_passed)

    active_overall = _safe_dict(active_dq.get("overall_enabled_categories"))
    candidate_overall = _safe_dict(candidate_dq.get("overall_enabled_categories"))
    gate_overall = (
        _float_value(candidate_overall.get("net_win_rate_model_minus_baseline"), -1.0)
        >= _float_value(active_overall.get("net_win_rate_model_minus_baseline"), -1.0)
        and _float_value(candidate_overall.get("both_wrong_rate"), 1.0)
        <= _float_value(active_overall.get("both_wrong_rate"), 1.0)
    )

    active_metrics = _safe_dict(active_eval.get("metrics_test"))
    candidate_metrics = _safe_dict(candidate_eval.get("metrics_test"))
    ndcg_delta = _delta(candidate_metrics.get("ndcg_at_5"), active_metrics.get("ndcg_at_5"))
    recall_delta = _delta(candidate_metrics.get("recall_at_1"), active_metrics.get("recall_at_1"))
    gate_offline = (
        (ndcg_delta is not None and float(ndcg_delta) >= -0.005)
        and (recall_delta is not None and float(recall_delta) >= -0.01)
    )

    gates = [
        _gate_status(
            "skincare_not_clearly_B_low_uplift",
            gate_skincare,
            "candidate_skincare_still_B_low_uplift" if not gate_skincare else "passed",
            {
                "active": {
                    "rollout_reason": active_skincare.get("rollout_reason"),
                    "diagnosis": _safe_dict(active_skincare.get("diagnosis")),
                },
                "candidate": {
                    "rollout_reason": candidate_skincare.get("rollout_reason"),
                    "diagnosis": _safe_dict(candidate_skincare.get("diagnosis")),
                },
            },
        ),
        _gate_status(
            "haircare_shampoo_two_stage_truth_improves",
            gate_haircare,
            gate_haircare_reason,
            {
                "active": active_shampoo,
                "candidate": candidate_shampoo,
                "old_gate_semantics": "exact_shampoo_truth_or_shampoo_to_conditioner_pair",
                "new_gate_semantics": {
                    "rule": "D_two_stage_truth",
                    "stage_1": "transition_family_correctness",
                    "stage_2": "exact_downstream_concrete_step_correctness",
                },
            },
        ),
        _gate_status(
            "protected_slices_non_regression",
            protected_passed,
            "protected_slice_regression" if not protected_passed else "passed",
            {"protected_results": protected_results},
        ),
        _gate_status(
            "overall_decision_quality_not_worse_than_active",
            gate_overall,
            "overall_decision_quality_regressed" if not gate_overall else "passed",
            {"active": active_overall, "candidate": candidate_overall},
        ),
        _gate_status(
            "offline_eval_not_materially_worse_than_active",
            gate_offline,
            "offline_eval_regressed" if not gate_offline else "passed",
            {
                "active_metrics_test": active_metrics,
                "candidate_metrics_test": candidate_metrics,
                "thresholds": {"min_ndcg_delta": -0.005, "min_recall_at_1_delta": -0.01},
                "delta_ndcg_at_5": ndcg_delta,
                "delta_recall_at_1": recall_delta,
            },
        ),
    ]
    return {
        "overall_passed": all(bool(gate["passed"]) for gate in gates),
        "gates": gates,
    }


def _runtime_config_snapshot() -> dict[str, Any]:
    return {
        "runtime_freeze_ml": bool(getattr(settings, "ROADMAP_RUNTIME_FREEZE_ML", True)),
        "roadmap_nextstep_model_path": str(getattr(settings, "ROADMAP_NEXTSTEP_MODEL_PATH", "") or ""),
        "roadmap_nextstep_v3_model_path": str(getattr(settings, "ROADMAP_NEXTSTEP_V3_MODEL_PATH", "") or ""),
        "roadmap_nextstep_v4_model_path": str(getattr(settings, "ROADMAP_NEXTSTEP_V4_MODEL_PATH", "") or ""),
        "roadmap_nextstep_v4_shadow_model_path": str(
            getattr(settings, "ROADMAP_NEXTSTEP_V4_SHADOW_MODEL_PATH", "") or ""
        ),
    }


def _catalog_safety_summary(artifacts: dict[str, dict[str, Any]]) -> dict[str, Any]:
    per_artifact: dict[str, Any] = {}
    writes_performed = False
    for label, artifact in artifacts.items():
        safety = _safe_dict(_safe_dict(artifact.get("shampoo_truth_design")).get("catalog_safety"))
        per_artifact[label] = safety
        writes_performed = writes_performed or bool(safety.get("catalog_writes_performed"))
    return {
        "catalog_writes_performed": bool(writes_performed),
        "per_artifact": per_artifact,
    }


def _category_delta_summary(candidate: dict[str, Any], reference: dict[str, Any]) -> dict[str, Any]:
    return {
        "model_win_rate_delta": _delta(candidate.get("model_win_rate"), reference.get("model_win_rate")),
        "baseline_win_rate_delta": _delta(candidate.get("baseline_win_rate"), reference.get("baseline_win_rate")),
        "both_wrong_rate_delta": _delta(candidate.get("both_wrong_rate"), reference.get("both_wrong_rate")),
        "resolved_truth_delta": (
            _int_value(candidate.get("resolved_truth"), 0) - _int_value(reference.get("resolved_truth"), 0)
        ),
    }


def _is_best_against_references(candidate: dict[str, Any], *references: dict[str, Any]) -> bool:
    if not references:
        return False
    candidate_model_win = _float_value(candidate.get("model_win_rate"), -1.0)
    candidate_baseline_win = _float_value(candidate.get("baseline_win_rate"), 1.0)
    candidate_both_wrong = _float_value(candidate.get("both_wrong_rate"), 1.0)
    return all(
        candidate_model_win >= _float_value(reference.get("model_win_rate"), -1.0)
        and candidate_baseline_win <= _float_value(reference.get("baseline_win_rate"), 1.0)
        and candidate_both_wrong <= _float_value(reference.get("both_wrong_rate"), 1.0)
        for reference in references
    )


def _category_qualification_verdict(
    row: dict[str, Any],
    *,
    gate_lookup: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    category = str(row.get("category") or "")
    active = _safe_dict(row.get("active"))
    retrain = _safe_dict(row.get("retrain_v1"))
    candidate = _safe_dict(row.get("v5_historical_anchor"))
    candidate_reason = str(candidate.get("rollout_reason") or "")
    candidate_diagnosis = _safe_dict(candidate.get("diagnosis"))
    diagnosis_code = str(candidate_diagnosis.get("code") or "")
    candidate_best = _is_best_against_references(candidate, active, retrain)
    beats_active = _is_best_against_references(candidate, active)
    beats_retrain = _is_best_against_references(candidate, retrain)
    shampoo_gate = _safe_dict(gate_lookup.get("haircare_shampoo_two_stage_truth_improves"))

    status = "hold"
    direct_answer = "hold"
    next_stage = False
    blocks_next_phase = False

    if category == "haircare":
        if bool(shampoo_gate.get("passed")) and beats_active and beats_retrain:
            status = "candidate_for_next_stage_under_freeze"
            direct_answer = "improved; candidate for next stage under freeze, but not for runtime enablement"
            next_stage = True
        elif _float_value(candidate.get("model_win_rate"), 0.0) > _float_value(active.get("model_win_rate"), 0.0):
            status = "improved_but_shampoo_gate_blocked"
            direct_answer = "improved, but the shampoo acceptance gate still blocks broader progression"
            blocks_next_phase = True
        else:
            status = "hold"
            direct_answer = "still HOLD"
            blocks_next_phase = True
    elif category == "skincare":
        if candidate_reason == "low_uplift" and diagnosis_code == "B":
            status = "hold_low_uplift"
            direct_answer = "still HOLD(low_uplift)"
            blocks_next_phase = True
        elif beats_active and beats_retrain:
            status = "improved_enough_for_next_stage_under_freeze"
            direct_answer = "improved enough for the next qualification stage under freeze; runtime remains HOLD(low_uplift)"
            next_stage = True
        else:
            status = "hold_low_uplift"
            direct_answer = "still HOLD(low_uplift)"
            blocks_next_phase = True
    elif category == "makeup":
        if candidate_reason == "sample_too_small_but_nonzero_control":
            status = "sample_limited_hold"
            direct_answer = "still sample-limited; not good enough yet"
        elif beats_active and beats_retrain:
            status = "candidate_for_next_stage_under_freeze"
            direct_answer = "good enough for next stage under freeze"
            next_stage = True
        else:
            status = "hold"
            direct_answer = "not good enough yet"
            blocks_next_phase = True
    elif category == "fragrance":
        if beats_active and beats_retrain:
            status = "analysis_only_positive_signal"
            direct_answer = "analysis-only bucket shows useful positive signal; still no runtime implication"
        else:
            status = "analysis_only_noisy_signal"
            direct_answer = "analysis-only bucket remains noisy; still no runtime implication"

    return {
        "category": category,
        "status": status,
        "direct_answer": direct_answer,
        "candidate_for_next_stage_under_freeze": bool(next_stage),
        "blocks_next_qualification_phase": bool(blocks_next_phase),
        "candidate_is_best_among_artifacts": bool(candidate_best),
        "candidate_beats_active": bool(beats_active),
        "candidate_beats_retrain_v1": bool(beats_retrain),
        "current_rollout_reason": candidate_reason,
        "candidate_diagnosis": candidate_diagnosis,
        "candidate": candidate,
        "comparison_vs_active": _category_delta_summary(candidate, active),
        "comparison_vs_retrain_v1": _category_delta_summary(candidate, retrain),
    }


def _global_blockers(
    *,
    acceptance: dict[str, Any],
    category_verdicts: dict[str, Any],
) -> list[str]:
    blockers: list[str] = []
    for gate in _safe_list(_safe_dict(acceptance).get("gates")):
        safe_gate = _safe_dict(gate)
        if bool(safe_gate.get("passed")):
            continue
        blockers.append(f"{safe_gate.get('name')}:{safe_gate.get('reason')}")
    for category, summary in sorted(category_verdicts.items()):
        safe_summary = _safe_dict(summary)
        if bool(safe_summary.get("blocks_next_qualification_phase")):
            blockers.append(f"{category}:{safe_summary.get('status')}")
    return blockers


def _broader_qualification_summary(
    *,
    category_rows: list[dict[str, Any]],
    acceptance: dict[str, Any],
) -> dict[str, Any]:
    gate_lookup = {
        str(gate.get("name")): _safe_dict(gate)
        for gate in _safe_list(_safe_dict(acceptance).get("gates"))
    }
    per_category = {
        str(row.get("category") or ""): _category_qualification_verdict(row, gate_lookup=gate_lookup)
        for row in category_rows
    }
    blockers = _global_blockers(acceptance=acceptance, category_verdicts=per_category)
    overall_passed = bool(_safe_dict(acceptance).get("overall_passed"))
    haircare_ready = bool(_safe_dict(per_category.get("haircare")).get("candidate_for_next_stage_under_freeze"))
    skincare_ready = bool(_safe_dict(per_category.get("skincare")).get("candidate_for_next_stage_under_freeze"))
    any_subset_ready = any(
        bool(_safe_dict(summary).get("candidate_for_next_stage_under_freeze"))
        for summary in per_category.values()
    )
    best_overall = overall_passed and haircare_ready and skincare_ready and not blockers

    if best_overall:
        recommendation_code = "A"
        recommendation_label = "continue qualification with v5 as the new best candidate"
    elif any_subset_ready:
        recommendation_code = "C"
        recommendation_label = "continue qualification only for a subset of categories under freeze"
    else:
        recommendation_code = "B"
        recommendation_label = "keep continuation frozen because broader results do not justify progression"

    return {
        "per_category": per_category,
        "global": {
            "recommendation_code": recommendation_code,
            "recommendation_label": recommendation_label,
            "is_v5_new_best_continuation_candidate": bool(best_overall),
            "is_v5_better_enough_than_active_for_next_phase": bool(recommendation_code in {"A", "C"}),
            "some_category_or_gate_still_blocks_progression": bool(blockers),
            "exact_blocker": "; ".join(blockers) if blockers else "none",
            "next_stage_focus_categories": [
                category
                for category, summary in per_category.items()
                if bool(_safe_dict(summary).get("candidate_for_next_stage_under_freeze"))
            ],
            "analysis_only_categories": [
                category
                for category, summary in per_category.items()
                if str(_safe_dict(summary).get("status") or "").startswith("analysis_only_")
            ],
            "runtime_enablement_allowed": False,
            "runtime_enablement_reason": "Runtime ML freeze remains on; this rerun is qualification/reporting only.",
            "remaining_blockers": blockers,
        },
    }


def normalize_historical_anchor_candidate_comparison_payload(payload: dict[str, Any]) -> dict[str, Any]:
    work = deepcopy(_safe_dict(payload))
    if not work:
        raise ValueError("comparison payload must be a dict")

    artifacts = _safe_dict(work.get("artifacts"))
    category_rows = _safe_list(work.get("category_comparison"))
    acceptance = _safe_dict(work.get("acceptance_gates"))

    scope = _safe_dict(work.get("qualification_scope"))
    if not scope:
        work["qualification_scope"] = {
            "mode": "read_only_qualification_rerun_under_runtime_freeze",
            "categories": {
                "primary": ["haircare", "skincare", "makeup"],
                "analysis_only": ["fragrance"],
            },
            "decision_quality_path": "current historical-anchor decision-quality path",
            "uplift_qualification_path": "current uplift/qualification path keyed by exact model_path",
            "shampoo_acceptance_semantics": "D_two_stage_truth",
            "unresolved_anchor_semantics": "fail_closed",
            "runtime_enablement": False,
            "rule_baseline_behavior_changed": False,
        }

    runtime_guardrails = _safe_dict(work.get("runtime_guardrails"))
    if not runtime_guardrails:
        snapshot = _runtime_config_snapshot()
        work["runtime_guardrails"] = {
            "before": snapshot,
            "after": snapshot,
            "runtime_config_changed": False,
        }

    catalog_safety = _safe_dict(work.get("catalog_safety"))
    if not catalog_safety:
        work["catalog_safety"] = (
            _catalog_safety_summary(artifacts)
            if artifacts
            else {
                "catalog_writes_performed": False,
                "per_artifact": {},
            }
        )

    broader = _safe_dict(work.get("broader_qualification"))
    if not broader and category_rows and acceptance:
        work["broader_qualification"] = _broader_qualification_summary(
            category_rows=category_rows,
            acceptance=acceptance,
        )

    provenance = _safe_dict(work.get("report_provenance"))
    if not provenance:
        work["report_provenance"] = {
            "source_of_truth": "fresh_db",
            "report_materialization": "fresh_db_rerun",
            "generated_from": "live_db",
            "fresh_db_attempted": True,
            "fresh_db_succeeded": True,
            "fresh_db_error": "",
            "fresh_db_failure_stage": "",
            "fresh_db_failure_operation": "",
            "cached_artifact_path": "",
            "input_sources": ["live_db", "current_runtime_settings_snapshot"],
            "read_only": True,
        }
    else:
        normalized = dict(provenance)
        normalized.setdefault("fresh_db_failure_stage", "")
        normalized.setdefault("fresh_db_failure_operation", "")
        work["report_provenance"] = normalized

    return work


def load_historical_anchor_candidate_comparison_payload_from_json(
    comparison_json_path: str | Path,
) -> dict[str, Any]:
    resolved_path = Path(str(comparison_json_path)).expanduser().resolve()
    payload = json.loads(resolved_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"comparison payload at {resolved_path} is not a dict")
    return normalize_historical_anchor_candidate_comparison_payload(payload)


def materialize_historical_anchor_candidate_comparison_payload(
    *,
    active_model_path: str | Path,
    retrain_v1_model_path: str | Path,
    candidate_model_path: str | Path,
    days: int = 30,
    source_preference: str = "auto",
    cached_comparison_json_path: str | Path | None = None,
) -> dict[str, Any]:
    source_mode = str(source_preference or "auto").strip().lower()
    if source_mode not in set(SOURCE_PREFERENCE_CHOICES):
        raise ValueError(f"unsupported source_preference={source_preference}")

    cached_path = Path(
        str(cached_comparison_json_path or DEFAULT_HISTORICAL_ANCHOR_COMPARE_REPORT_JSON)
    ).expanduser()
    fresh_db_error = ""
    fresh_db_failure_stage = ""
    fresh_db_failure_operation = ""
    fresh_db_attempted = source_mode in {"auto", "fresh_db"}

    if source_mode in {"auto", "fresh_db"}:
        try:
            payload = build_historical_anchor_candidate_comparison_payload(
                active_model_path=active_model_path,
                retrain_v1_model_path=retrain_v1_model_path,
                candidate_model_path=candidate_model_path,
                days=days,
            )
            payload = normalize_historical_anchor_candidate_comparison_payload(payload)
            payload["report_provenance"] = {
                "source_of_truth": "fresh_db",
                "report_materialization": "fresh_db_rerun",
                "generated_from": "live_db",
                "fresh_db_attempted": True,
                "fresh_db_succeeded": True,
                "fresh_db_error": "",
                "fresh_db_failure_stage": "",
                "fresh_db_failure_operation": "",
                "cached_artifact_path": str(cached_path.resolve()) if cached_path.exists() else "",
                "input_sources": ["live_db", "current_runtime_settings_snapshot"],
                "read_only": True,
            }
            return payload
        except HistoricalAnchorReadError as exc:
            fresh_db_error = str(exc.error_text)
            fresh_db_failure_stage = str(exc.stage)
            fresh_db_failure_operation = str(exc.operation)
            if source_mode == "fresh_db":
                raise
        except (OperationalError, DatabaseError) as exc:
            fresh_db_error = f"{type(exc).__module__}.{type(exc).__name__}: {exc}"
            fresh_db_failure_stage = "unclassified_db_failure"
            fresh_db_failure_operation = "build_historical_anchor_candidate_comparison_payload"
            if source_mode == "fresh_db":
                raise

    payload = load_historical_anchor_candidate_comparison_payload_from_json(cached_path)
    payload["report_provenance"] = {
        "source_of_truth": "cached_artifact",
        "report_materialization": "materialized_from_saved_artifacts",
        "generated_from": "mixed_read_only_inputs" if fresh_db_attempted else "comparison_json",
        "fresh_db_attempted": bool(fresh_db_attempted),
        "fresh_db_succeeded": False,
        "fresh_db_error": fresh_db_error,
        "fresh_db_failure_stage": fresh_db_failure_stage,
        "fresh_db_failure_operation": fresh_db_failure_operation,
        "cached_artifact_path": str(cached_path.resolve()),
        "input_sources": ["comparison_json", "current_runtime_settings_snapshot"],
        "read_only": True,
    }
    return payload


def build_historical_anchor_candidate_comparison_payload(
    *,
    active_model_path: str | Path,
    retrain_v1_model_path: str | Path,
    candidate_model_path: str | Path,
    days: int = 30,
) -> dict[str, Any]:
    runtime_snapshot_before = _runtime_config_snapshot()
    now_utc = timezone.now()
    since = now_utc - timedelta(days=int(days))
    historical_context = build_historical_anchor_read_context(
        since=since,
        until=now_utc,
        category="all",
        include_ga=False,
    )
    artifacts = {
        "active": _artifact_comparison_entry(
            label="active",
            model_path=active_model_path,
            days=days,
            historical_context=historical_context,
        ),
        "retrain_v1": _artifact_comparison_entry(
            label="retrain_v1",
            model_path=retrain_v1_model_path,
            days=days,
            historical_context=historical_context,
        ),
        "v5_historical_anchor": _artifact_comparison_entry(
            label="v5_historical_anchor",
            model_path=candidate_model_path,
            days=days,
            historical_context=historical_context,
        ),
    }
    targeted_truth_rows = _comparison_slice_rows(
        artifacts=artifacts,
        kind="truth_slice_lookup",
        key_builder=lambda spec: f"{spec['category']}:{spec['truth_product_type']}",
        specs=TARGETED_TRUTH_SLICES,
    )
    targeted_pair_rows = _comparison_slice_rows(
        artifacts=artifacts,
        kind="disagreement_pair_lookup",
        key_builder=lambda spec: f"{spec['category']}:{spec['baseline_product_type']}:{spec['model_product_type']}",
        specs=TARGETED_DISAGREEMENT_PAIRS,
    )
    protected_rows = _comparison_slice_rows(
        artifacts=artifacts,
        kind="truth_slice_lookup",
        key_builder=lambda spec: f"{spec['category']}:{spec['truth_product_type']}",
        specs=PROTECTED_TRUTH_SLICES,
    )
    acceptance = _acceptance_gates(artifacts)
    category_comparison = _compare_category_rows(artifacts)
    runtime_snapshot_after = _runtime_config_snapshot()
    payload = {
        "qualification_scope": {
            "mode": "read_only_qualification_rerun_under_runtime_freeze",
            "categories": {
                "primary": ["haircare", "skincare", "makeup"],
                "analysis_only": ["fragrance"],
            },
            "decision_quality_path": "current historical-anchor decision-quality path",
            "uplift_qualification_path": "current uplift/qualification path keyed by exact model_path",
            "shampoo_acceptance_semantics": "D_two_stage_truth",
            "unresolved_anchor_semantics": "fail_closed",
            "runtime_enablement": False,
            "rule_baseline_behavior_changed": False,
        },
        "policy": targeted_retrain_policy_payload(),
        "artifacts": artifacts,
        "runtime_guardrails": {
            "before": runtime_snapshot_before,
            "after": runtime_snapshot_after,
            "runtime_config_changed": runtime_snapshot_before != runtime_snapshot_after,
        },
        "catalog_safety": _catalog_safety_summary(artifacts),
        "category_comparison": category_comparison,
        "targeted_truth_slices": targeted_truth_rows,
        "targeted_disagreement_pairs": targeted_pair_rows,
        "protected_truth_slices": protected_rows,
        "haircare_shampoo_truth_gate_comparison": _compare_shampoo_two_stage_rows(artifacts),
        "acceptance_gates": acceptance,
        "broader_qualification": _broader_qualification_summary(
            category_rows=category_comparison,
            acceptance=acceptance,
        ),
    }
    return normalize_historical_anchor_candidate_comparison_payload(payload)


def render_historical_anchor_candidate_comparison_markdown(payload: dict[str, Any]) -> str:
    artifacts = _safe_dict(payload.get("artifacts"))
    gate_lookup = {
        str(gate.get("name")): _safe_dict(gate)
        for gate in _safe_list(_safe_dict(payload.get("acceptance_gates")).get("gates"))
    }
    shampoo_gate = _safe_dict(gate_lookup.get("haircare_shampoo_two_stage_truth_improves"))
    shampoo_comparison = _safe_dict(payload.get("haircare_shampoo_truth_gate_comparison"))
    scope = _safe_dict(payload.get("qualification_scope"))
    runtime_guardrails = _safe_dict(payload.get("runtime_guardrails"))
    catalog_safety = _safe_dict(payload.get("catalog_safety"))
    broader = _safe_dict(payload.get("broader_qualification"))
    broader_global = _safe_dict(broader.get("global"))
    provenance = _safe_dict(payload.get("report_provenance"))
    lines = [
        "# Roadmap Nextstep v5 Broader Qualification Rerun",
        "",
        "## Executive Verdict",
        f"- scope: `{scope.get('mode')}`",
        f"- report materialization: `{provenance.get('report_materialization')}`",
        f"- source_of_truth: `{provenance.get('source_of_truth')}`",
        f"- generated_from: `{provenance.get('generated_from')}`",
        f"- fresh_db_attempted: `{provenance.get('fresh_db_attempted')}`",
        f"- fresh_db_succeeded: `{provenance.get('fresh_db_succeeded')}`",
        f"- runtime ML freeze remains on: `{_safe_dict(runtime_guardrails.get('after')).get('runtime_freeze_ml')}`",
        f"- runtime config changed during rerun: `{runtime_guardrails.get('runtime_config_changed')}`",
        f"- catalog writes performed: `{catalog_safety.get('catalog_writes_performed')}`",
        f"- rule baseline behavior changed: `{scope.get('rule_baseline_behavior_changed')}`",
        f"- unresolved anchors remain fail-closed: `{scope.get('unresolved_anchor_semantics') == 'fail_closed'}`",
        f"- shampoo gate semantics adopted: `D_two_stage_truth`",
        f"- shampoo gate passed: `{shampoo_gate.get('passed')}`",
        f"- shampoo gate reason: `{shampoo_gate.get('reason')}`",
        f"- overall acceptance passed: `{_safe_dict(payload.get('acceptance_gates')).get('overall_passed')}`",
        f"- v5 is now the best continuation candidate overall: `{broader_global.get('is_v5_new_best_continuation_candidate')}`",
        f"- v5 is better enough than active to continue to the next qualification phase: `{broader_global.get('is_v5_better_enough_than_active_for_next_phase')}`",
        f"- some category or gate still blocks progression: `{broader_global.get('some_category_or_gate_still_blocks_progression')}`",
        f"- exact blocker: `{broader_global.get('exact_blocker')}`",
        f"- final recommendation: `{broader_global.get('recommendation_code')}` {broader_global.get('recommendation_label')}",
        "",
        "## Provenance",
        f"- report_materialization: `{provenance.get('report_materialization')}`",
        f"- source_of_truth: `{provenance.get('source_of_truth')}`",
        f"- generated_from: `{provenance.get('generated_from')}`",
        f"- cached_artifact_path: `{provenance.get('cached_artifact_path')}`",
        f"- fresh_db_failure_stage: `{provenance.get('fresh_db_failure_stage')}`",
        f"- fresh_db_failure_operation: `{provenance.get('fresh_db_failure_operation')}`",
        f"- fresh_db_error: `{provenance.get('fresh_db_error')}`",
        "",
        "## Why These Slices Were Targeted",
    ]
    for spec in _safe_list(_safe_dict(payload.get("policy")).get("targeted_truth_slices")):
        causes = ", ".join(_safe_list(spec.get("probable_causes")))
        lines.append(
            f"- `{spec.get('category')}/{spec.get('truth_product_type')}`: {spec.get('diagnosis')} "
            f"Likely causes: {causes}."
        )

    lines.extend(["", "## Artifacts"])
    for key in ["active", "retrain_v1", "v5_historical_anchor"]:
        artifact = _safe_dict(artifacts.get(key))
        proof = _safe_dict(artifact.get("proof_bundle"))
        lines.append(
            f"- `{key}` path=`{artifact.get('model_path')}` proof_complete=`{proof.get('required_complete')}` "
            f"reason=`{proof.get('reason')}`"
        )

    lines.extend(["", "## Haircare Shampoo Gate"])
    lines.append("- old gate semantics: `exact shampoo truth or shampoo->conditioner pair truth`")
    lines.append("- old gate status: semantically mismatched to immutable historical truth")
    lines.append("- adopted gate semantics: `D_two_stage_truth`")
    lines.append("  stage 1: transition family correctness")
    lines.append("  stage 2: exact downstream concrete-step correctness")
    lines.append("")
    lines.append("| artifact | old_gate_status | old_resolved | old_unresolved | old_shampoo_truth_rows | old_shampoo->conditioner_rows | stage1_model_win | stage2_model_win | two_stage_resolved | two_stage_unresolved |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for key in ["active", "retrain_v1", "v5_historical_anchor"]:
        row = _safe_dict(shampoo_comparison.get(key))
        lines.append(
            f"| {key} | {row.get('current_gate_status')} | {row.get('old_gate_resolved')} | {row.get('old_gate_unresolved')} "
            f"| {row.get('old_gate_shampoo_truth_rows')} | {row.get('old_gate_shampoo_conditioner_rows')} "
            f"| {_pct(row.get('stage_1_family_model_win_rate'))} | {_pct(row.get('stage_2_concrete_model_win_rate'))} "
            f"| {row.get('two_stage_resolved')} | {row.get('two_stage_unresolved')} |"
        )
    if shampoo_gate:
        lines.append("")
        lines.append(f"- final shampoo gate verdict for v5: `{'cleared' if shampoo_gate.get('passed') else 'not_cleared'}`")
        lines.append(f"- gate reason: `{shampoo_gate.get('reason')}`")
        lines.append("- unresolved anchors remain fail-closed under the adopted gate.")

    lines.extend(["", "## Per-Category Qualification Verdict"])
    lines.append("| category | direct_answer | v5_status | current_rollout_reason | v5_diag | vs_active_model_win | vs_active_both_wrong | vs_retrain_model_win | vs_retrain_both_wrong |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for row in _safe_list(payload.get("category_comparison")):
        category = str(row.get("category") or "")
        verdict = _safe_dict(_safe_dict(broader.get("per_category")).get(category))
        lines.append(
            f"| {category} | {verdict.get('direct_answer')} | {verdict.get('status')} | "
            f"{verdict.get('current_rollout_reason')} | {str(_safe_dict(verdict.get('candidate_diagnosis')).get('code') or '')} | "
            f"{_ppt(_safe_dict(verdict.get('comparison_vs_active')).get('model_win_rate_delta'))} | "
            f"{_ppt(_safe_dict(verdict.get('comparison_vs_active')).get('both_wrong_rate_delta'))} | "
            f"{_ppt(_safe_dict(verdict.get('comparison_vs_retrain_v1')).get('model_win_rate_delta'))} | "
            f"{_ppt(_safe_dict(verdict.get('comparison_vs_retrain_v1')).get('both_wrong_rate_delta'))} |"
        )

    lines.extend(["", "## Category Comparison"])
    lines.append("| category | active_reason | retrain_v1_reason | v5_reason | active_model_win | retrain_v1_model_win | v5_model_win |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- |")
    for row in _safe_list(payload.get("category_comparison")):
        active_row = _safe_dict(row.get("active"))
        retrain_row = _safe_dict(row.get("retrain_v1"))
        candidate_row = _safe_dict(row.get("v5_historical_anchor"))
        lines.append(
            f"| {row.get('category')} | {active_row.get('rollout_reason')} | {retrain_row.get('rollout_reason')} "
            f"| {candidate_row.get('rollout_reason')} | {_pct(active_row.get('model_win_rate'))} "
            f"| {_pct(retrain_row.get('model_win_rate'))} | {_pct(candidate_row.get('model_win_rate'))} |"
        )

    lines.extend(["", "## Targeted Slices"])
    lines.append("| slice | active_net | retrain_v1_net | v5_net | active_model_win | retrain_v1_model_win | v5_model_win |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- |")
    for row in _safe_list(payload.get("targeted_truth_slices")):
        active_row = _safe_dict(row.get("active"))
        retrain_row = _safe_dict(row.get("retrain_v1"))
        candidate_row = _safe_dict(row.get("v5_historical_anchor"))
        lines.append(
            f"| {row.get('category')}/{row.get('truth_product_type')} | {active_row.get('net_wins_model_minus_baseline', 'n/a')} "
            f"| {retrain_row.get('net_wins_model_minus_baseline', 'n/a')} | {candidate_row.get('net_wins_model_minus_baseline', 'n/a')} "
            f"| {_pct(active_row.get('model_win_rate_vs_truth'))} | {_pct(retrain_row.get('model_win_rate_vs_truth'))} "
            f"| {_pct(candidate_row.get('model_win_rate_vs_truth'))} |"
        )

    lines.extend(["", "## Protected Slices"])
    lines.append("| slice | active_net | retrain_v1_net | v5_net | active_model_win | retrain_v1_model_win | v5_model_win |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- |")
    for row in _safe_list(payload.get("protected_truth_slices")):
        active_row = _safe_dict(row.get("active"))
        retrain_row = _safe_dict(row.get("retrain_v1"))
        candidate_row = _safe_dict(row.get("v5_historical_anchor"))
        lines.append(
            f"| {row.get('category')}/{row.get('truth_product_type')} | {active_row.get('net_wins_model_minus_baseline', 'n/a')} "
            f"| {retrain_row.get('net_wins_model_minus_baseline', 'n/a')} | {candidate_row.get('net_wins_model_minus_baseline', 'n/a')} "
            f"| {_pct(active_row.get('model_win_rate_vs_truth'))} | {_pct(retrain_row.get('model_win_rate_vs_truth'))} "
            f"| {_pct(candidate_row.get('model_win_rate_vs_truth'))} |"
        )

    lines.extend(["", "## Acceptance Gates"])
    for gate in _safe_list(_safe_dict(payload.get("acceptance_gates")).get("gates")):
        lines.append(
            f"- `{gate.get('name')}` passed=`{gate.get('passed')}` reason=`{gate.get('reason')}`"
        )

    lines.extend(["", "## Offline Eval"])
    lines.append("| artifact | recall@1 | ndcg@5 |")
    lines.append("| --- | --- | --- |")
    for key in ["active", "retrain_v1", "v5_historical_anchor"]:
        eval_payload = _safe_dict(_safe_dict(artifacts.get(key)).get("eval"))
        metrics = _safe_dict(eval_payload.get("metrics_test"))
        lines.append(f"| {key} | {_pct(metrics.get('recall_at_1'))} | {_pct(metrics.get('ndcg_at_5'))} |")

    lines.extend(["", "## Verdict"])
    lines.append(
        f"- Recommendation `{broader_global.get('recommendation_code')}`: {broader_global.get('recommendation_label')}."
    )
    lines.append(
        f"- Next-stage focus categories under freeze: `{', '.join(_safe_list(broader_global.get('next_stage_focus_categories'))) or 'none'}`."
    )
    lines.append(
        f"- Analysis-only categories: `{', '.join(_safe_list(broader_global.get('analysis_only_categories'))) or 'none'}`."
    )
    lines.append(
        f"- Runtime enablement allowed: `{broader_global.get('runtime_enablement_allowed')}`. "
        f"{broader_global.get('runtime_enablement_reason')}"
    )
    if bool(_safe_dict(payload.get("acceptance_gates")).get("overall_passed")):
        lines.append("- v5 satisfies the updated freeze-only acceptance gates and should advance as the broader qualification candidate, not as a runtime artifact.")
    else:
        lines.append("- v5 does not satisfy the updated broader qualification gates; keep continuation frozen until the listed blocker is resolved.")
    return "\n".join(lines).strip() + "\n"
