from __future__ import annotations

import json
from pathlib import Path
from typing import Any

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


POLICY_VERSION = "roadmap_nextstep_v4_targeted_retrain_v1"
DEFAULT_TARGETED_DATA_DIR = Path("data") / "ml" / "roadmap_nextstep_v4_targeted_retrain_v1"
DEFAULT_TARGETED_MODEL_DIR = Path("models") / "roadmap_next_step_v4_targeted_retrain_v1"
DEFAULT_COMPARE_REPORT_STEM = Path("reports") / "roadmap_nextstep_targeted_retrain_comparison"
DEFAULT_HISTORICAL_ANCHOR_COMPARE_REPORT_STEM = (
    Path("reports") / "roadmap_nextstep_v5_historical_anchor_targeted_v1_comparison"
)


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


def build_targeted_retrain_comparison_payload(
    *,
    base_model_path: str | Path,
    candidate_model_path: str | Path,
    days: int = 30,
) -> dict[str, Any]:
    base_decision_quality = build_nextstep_v4_decision_quality_payload(
        model_path=base_model_path,
        days=days,
        category="all",
        include_ga=False,
        min_slice_size=10,
    )
    candidate_decision_quality = build_nextstep_v4_decision_quality_payload(
        model_path=candidate_model_path,
        days=days,
        category="all",
        include_ga=False,
        min_slice_size=10,
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


def _artifact_comparison_entry(*, label: str, model_path: str | Path, days: int) -> dict[str, Any]:
    resolved_model_path = str(Path(str(model_path)).expanduser().resolve())
    decision_quality = build_nextstep_v4_decision_quality_payload(
        model_path=resolved_model_path,
        days=days,
        category="all",
        include_ga=False,
        min_slice_size=10,
    )
    return {
        "label": label,
        "model_path": resolved_model_path,
        "eval": _load_eval_report(resolved_model_path),
        "proof_bundle": candidate_proof_bundle_summary(resolved_model_path),
        "decision_quality": decision_quality,
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

    shampoo_key = "haircare:shampoo"
    pair_key = "haircare:shampoo:conditioner"
    active_shampoo = _slice_lookup(active_dq, kind="truth_slice_lookup", key=shampoo_key)
    candidate_shampoo = _slice_lookup(candidate_dq, kind="truth_slice_lookup", key=shampoo_key)
    active_pair = _slice_lookup(active_dq, kind="disagreement_pair_lookup", key=pair_key)
    candidate_pair = _slice_lookup(candidate_dq, kind="disagreement_pair_lookup", key=pair_key)
    gate_haircare = (
        _int_value(candidate_shampoo.get("net_wins_model_minus_baseline"), -10**9)
        > _int_value(active_shampoo.get("net_wins_model_minus_baseline"), -10**9)
        and _int_value(candidate_pair.get("net_wins_model_minus_baseline"), 0)
        >= _int_value(active_pair.get("net_wins_model_minus_baseline"), 0)
    )

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
            "haircare_shampoo_and_pair_improve",
            gate_haircare,
            "haircare_targeted_slice_not_improved" if not gate_haircare else "passed",
            {
                "active_shampoo": active_shampoo,
                "candidate_shampoo": candidate_shampoo,
                "active_pair": active_pair,
                "candidate_pair": candidate_pair,
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


def build_historical_anchor_candidate_comparison_payload(
    *,
    active_model_path: str | Path,
    retrain_v1_model_path: str | Path,
    candidate_model_path: str | Path,
    days: int = 30,
) -> dict[str, Any]:
    artifacts = {
        "active": _artifact_comparison_entry(label="active", model_path=active_model_path, days=days),
        "retrain_v1": _artifact_comparison_entry(
            label="retrain_v1", model_path=retrain_v1_model_path, days=days
        ),
        "v5_historical_anchor": _artifact_comparison_entry(
            label="v5_historical_anchor", model_path=candidate_model_path, days=days
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
    return {
        "policy": targeted_retrain_policy_payload(),
        "artifacts": artifacts,
        "category_comparison": _compare_category_rows(artifacts),
        "targeted_truth_slices": targeted_truth_rows,
        "targeted_disagreement_pairs": targeted_pair_rows,
        "protected_truth_slices": protected_rows,
        "acceptance_gates": _acceptance_gates(artifacts),
    }


def render_historical_anchor_candidate_comparison_markdown(payload: dict[str, Any]) -> str:
    artifacts = _safe_dict(payload.get("artifacts"))
    lines = [
        "# Roadmap Nextstep v5 Historical Anchor Targeted Comparison",
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
    if bool(_safe_dict(payload.get("acceptance_gates")).get("overall_passed")):
        lines.append("- v5 historical-anchor artifact satisfies the defined acceptance gates and is a meaningful future qualification candidate.")
    else:
        lines.append("- v5 historical-anchor artifact does not satisfy the defined acceptance gates and should not advance runtime qualification.")
    return "\n".join(lines).strip() + "\n"
