from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

try:
    import pandas as pd
except Exception:  # pragma: no cover
    pd = None

from django.core.management.base import BaseCommand, CommandError
from django.utils import timezone

STOP_TOKEN = "__stop__"


def _resolve_path(raw_path: str) -> Path:
    candidate = Path(str(raw_path)).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    cwd_path = (Path.cwd() / candidate).resolve()
    if cwd_path.parent.exists():
        return cwd_path
    return (Path(__file__).resolve().parents[4] / candidate).resolve()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_dataset_bundle(dataset_dir: Path) -> dict[str, Any]:
    metadata_path = dataset_dir / "metadata.json"
    splits_path = dataset_dir / "splits.json"
    parquet_path = dataset_dir / "dataset.parquet"
    csv_path = dataset_dir / "dataset.csv"
    if not metadata_path.exists():
        raise CommandError(f"Missing metadata.json in {dataset_dir}")
    if not splits_path.exists():
        raise CommandError(f"Missing splits.json in {dataset_dir}")
    if parquet_path.exists():
        df = pd.read_parquet(parquet_path)
    elif csv_path.exists():
        df = pd.read_csv(csv_path)
    else:
        raise CommandError(f"Missing dataset.parquet/dataset.csv in {dataset_dir}")
    metadata = _load_json(metadata_path)
    splits = _load_json(splits_path)
    if "t0_utc" in df.columns:
        df["t0_utc"] = pd.to_datetime(df["t0_utc"], utc=True, format="mixed")
    positives = df[df["y"] == 1].copy()
    return {
        "dataset_dir": str(dataset_dir),
        "metadata": metadata,
        "splits": splits,
        "df": df,
        "positives": positives,
    }


def _split_time_bounds(positives: "pd.DataFrame") -> dict[str, Any]:
    bounds: dict[str, Any] = {}
    for split_name in ("train", "val", "test"):
        split_df = positives[positives["split"] == split_name]
        if split_df.empty:
            bounds[split_name] = {
                "decision_points": 0,
                "users": 0,
                "t0_min_utc": None,
                "t0_max_utc": None,
            }
            continue
        bounds[split_name] = {
            "decision_points": int(split_df["decision_id"].nunique()),
            "users": int(split_df["user_id"].nunique()),
            "t0_min_utc": split_df["t0_utc"].min().isoformat().replace("+00:00", "Z"),
            "t0_max_utc": split_df["t0_utc"].max().isoformat().replace("+00:00", "Z"),
        }
    return bounds


def _time_leakage_checks(split_bounds: dict[str, Any]) -> dict[str, Any]:
    def _to_ts(value: Any):
        if not value:
            return None
        return pd.Timestamp(value)

    train_max = _to_ts(split_bounds.get("train", {}).get("t0_max_utc"))
    val_min = _to_ts(split_bounds.get("val", {}).get("t0_min_utc"))
    val_max = _to_ts(split_bounds.get("val", {}).get("t0_max_utc"))
    test_min = _to_ts(split_bounds.get("test", {}).get("t0_min_utc"))
    checks = {
        "train_before_or_equal_val": bool(train_max is None or val_min is None or train_max <= val_min),
        "val_before_or_equal_test": bool(val_max is None or test_min is None or val_max <= test_min),
        "train_before_or_equal_test": bool(train_max is None or test_min is None or train_max <= test_min),
    }
    checks["status"] = "passed" if all(checks.values()) else "failed"
    return checks


def _raw_decision_surface_total(metadata: dict[str, Any]) -> int:
    if metadata.get("raw_plan_refreshed_events") is not None:
        return int(metadata.get("raw_plan_refreshed_events") or 0)
    surface_summary = metadata.get("surface_summary") or {}
    return int(sum(int(payload.get("raw_count") or 0) for payload in surface_summary.values()))


def _label_vocab_by_category(
    positives: "pd.DataFrame",
    candidate_types_by_category: dict[str, list[str]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for category, full_vocab in sorted(candidate_types_by_category.items()):
        category_df = positives[positives["category"] == category]
        labels_all = sorted(category_df["label"].astype(str).unique().tolist()) if not category_df.empty else []
        labels_non_stop = sorted(
            category_df.loc[category_df["label"] != STOP_TOKEN, "label"].astype(str).unique().tolist()
        )
        non_stop_vocab = [candidate for candidate in full_vocab if candidate != STOP_TOKEN]
        result[category] = {
            "all_labels_present": labels_all,
            "non_stop_labels_present": labels_non_stop,
            "all_labels_present_count": int(len(labels_all)),
            "non_stop_labels_present_count": int(len(labels_non_stop)),
            "available_non_stop_actions_count": int(len(non_stop_vocab)),
            "non_stop_action_coverage_ratio": round(
                float(len(labels_non_stop)) / float(len(non_stop_vocab) or 1),
                6,
            ),
        }
    return result


def _class_balance_by_category(positives: "pd.DataFrame") -> dict[str, Any]:
    result: dict[str, Any] = {}
    for category, category_df in positives.groupby("category"):
        total = float(len(category_df) or 1)
        counts = category_df["label"].astype(str).value_counts().to_dict()
        result[str(category)] = {
            str(label): {
                "count": int(count),
                "share": round(float(count) / total, 6),
            }
            for label, count in sorted(counts.items())
        }
    return result


def _positives_by_label_source(positives: "pd.DataFrame", *, non_stop_only: bool) -> dict[str, int]:
    frame = positives if not non_stop_only else positives[positives["label"] != STOP_TOKEN]
    counter = Counter(str(value) for value in frame["label_source"].astype(str).tolist())
    return {key: int(value) for key, value in sorted(counter.items())}


def _split_user_counts(positives: "pd.DataFrame") -> dict[str, int]:
    return {
        split_name: int(positives.loc[positives["split"] == split_name, "user_id"].nunique())
        for split_name in ("train", "val", "test")
    }


def _dataset_metrics(*, bundle: dict[str, Any], dataset_kind: str) -> dict[str, Any]:
    metadata = dict(bundle["metadata"])
    positives = bundle["positives"]
    positives_non_stop = positives[positives["label"] != STOP_TOKEN]
    candidate_types_by_category = dict(metadata.get("candidate_types_by_category") or {})
    split_bounds = _split_time_bounds(positives)
    label_vocab_by_category = _label_vocab_by_category(positives, candidate_types_by_category)
    class_balance_by_category = _class_balance_by_category(positives)
    raw_surface_total = _raw_decision_surface_total(metadata)
    excluded_noisy = int(metadata.get("excluded_noisy_decision_points_count") or 0)
    if dataset_kind == "transitions":
        slices = dict(metadata.get("slices") or {})
        continuation_slice = dict(slices.get("continuation_only") or {})
        initial_slice = dict(slices.get("initial_only") or {})
        continuation_positives_count = int(continuation_slice.get("positives_excluding_stop") or 0)
        initial_decisions_count = int(initial_slice.get("trusted_decisions_total") or 0)
        continuation_decisions_count = int(continuation_slice.get("trusted_decisions_total") or 0)
    else:
        continuation_positives_count = 0
        initial_decisions_count = int(metadata.get("decision_points_total") or 0)
        continuation_decisions_count = 0
    return {
        "dataset_dir": bundle["dataset_dir"],
        "rows_total": int(metadata.get("rows_total") or 0),
        "decision_points_total": int(metadata.get("decision_points_total") or 0),
        "users_total": int(metadata.get("users_total") or 0),
        "positives_excluding_stop": int(len(positives_non_stop)),
        "stop_rate": float(metadata.get("stop_label_rate") or 0.0),
        "positives_by_category": {
            str(key): int(value) for key, value in sorted((metadata.get("positives_by_category") or {}).items())
        },
        "label_source_distribution_all": {
            str(key): int(value) for key, value in sorted((metadata.get("label_source_distribution") or {}).items())
        },
        "positives_by_label_source_non_stop": _positives_by_label_source(positives, non_stop_only=True),
        "fragrance_positives_count": int((metadata.get("positives_by_category") or {}).get("fragrance") or 0),
        "continuation_positives_count": continuation_positives_count,
        "initial_decisions_count": initial_decisions_count,
        "continuation_decisions_count": continuation_decisions_count,
        "decision_type_distribution": {
            str(key): int(value) for key, value in sorted((metadata.get("decision_type_distribution") or {}).items())
        },
        "labels_present_by_category": label_vocab_by_category,
        "split_counts": {str(key): int(value) for key, value in sorted((metadata.get("split_counts") or {}).items())},
        "split_user_counts": _split_user_counts(positives),
        "split_user_overlap_counts": {
            str(key): int(value) for key, value in sorted((metadata.get("split_user_overlap_counts") or {}).items())
        },
        "split_time_bounds": split_bounds,
        "time_leakage_checks": _time_leakage_checks(split_bounds),
        "excluded_noisy_decision_points_count": excluded_noisy,
        "excluded_noisy_decision_points_share": round(
            float(excluded_noisy) / float(raw_surface_total or 1),
            6,
        ),
        "excluded_legacy_bad_fragrance_completions_count": int(
            metadata.get("excluded_legacy_bad_fragrance_completions_count") or 0
        ),
        "class_balance_by_category": class_balance_by_category,
        "leakage_assertions": dict(metadata.get("leakage_assertions") or {}),
        "candidate_types_by_category": candidate_types_by_category,
        "raw_surface_total": raw_surface_total,
    }


def _safe_read_previous_metadata(path: Path) -> dict[str, Any] | None:
    metadata_path = path / "metadata.json"
    if not metadata_path.exists():
        return None
    return _load_json(metadata_path)


def _comparison_against_previous(
    *,
    previous_initial: dict[str, Any] | None,
    previous_transitions: dict[str, Any] | None,
    current_initial: dict[str, Any],
    current_transitions: dict[str, Any],
) -> dict[str, Any]:
    comparison: dict[str, Any] = {"available": bool(previous_initial and previous_transitions)}
    if not comparison["available"]:
        comparison["note"] = "Previous weak-state metadata not found; comparison is unavailable."
        return comparison

    previous_initial_non_stop = int(previous_initial.get("decision_points_total") or 0) - int(
        previous_initial.get("stop_label_count") or 0
    )
    current_initial_non_stop = int(current_initial.get("positives_excluding_stop") or 0)
    previous_transition_combined_non_stop = int(
        ((previous_transitions.get("slices") or {}).get("combined") or {}).get("positives_excluding_stop") or 0
    )
    current_transition_combined_non_stop = int(current_transitions.get("positives_excluding_stop") or 0)
    previous_transition_cont_non_stop = int(
        ((previous_transitions.get("slices") or {}).get("continuation_only") or {}).get("positives_excluding_stop") or 0
    )
    current_transition_cont_non_stop = int(current_transitions.get("continuation_positives_count") or 0)

    improved = []
    if current_initial_non_stop > previous_initial_non_stop:
        improved.append("initial_non_stop_positives")
    if current_initial.get("fragrance_positives_count", 0) > int((previous_initial.get("positives_by_category") or {}).get("fragrance") or 0):
        improved.append("initial_fragrance_positives")
    if current_initial.get("users_total", 0) > int(previous_initial.get("users_total") or 0):
        improved.append("initial_user_coverage")
    if current_transitions.get("decision_points_total", 0) > int(previous_transitions.get("decision_points_total") or 0):
        improved.append("transition_decision_volume")

    still_weak = []
    if current_transition_cont_non_stop <= 0:
        still_weak.append("continuation_non_stop_positives_absent")
    narrow_categories = [
        category
        for category, payload in current_initial.get("labels_present_by_category", {}).items()
        if int(payload.get("non_stop_labels_present_count") or 0) <= 1
    ]
    if narrow_categories:
        still_weak.append(f"initial_label_diversity_narrow:{','.join(sorted(narrow_categories))}")

    comparison["initial"] = {
        "decision_points_total": {
            "before": int(previous_initial.get("decision_points_total") or 0),
            "now": int(current_initial.get("decision_points_total") or 0),
        },
        "rows_total": {
            "before": int(previous_initial.get("rows_total") or 0),
            "now": int(current_initial.get("rows_total") or 0),
        },
        "users_total": {
            "before": int(previous_initial.get("users_total") or 0),
            "now": int(current_initial.get("users_total") or 0),
        },
        "non_stop_positives": {
            "before": previous_initial_non_stop,
            "now": current_initial_non_stop,
        },
        "stop_rate": {
            "before": float(previous_initial.get("stop_label_rate") or 0.0),
            "now": float(current_initial.get("stop_rate") or 0.0),
        },
        "positives_by_category": {
            "before": dict(previous_initial.get("positives_by_category") or {}),
            "now": dict(current_initial.get("positives_by_category") or {}),
        },
    }
    comparison["transitions"] = {
        "decision_points_total": {
            "before": int(previous_transitions.get("decision_points_total") or 0),
            "now": int(current_transitions.get("decision_points_total") or 0),
        },
        "rows_total": {
            "before": int(previous_transitions.get("rows_total") or 0),
            "now": int(current_transitions.get("rows_total") or 0),
        },
        "users_total": {
            "before": int(previous_transitions.get("users_total") or 0),
            "now": int(current_transitions.get("users_total") or 0),
        },
        "combined_non_stop_positives": {
            "before": previous_transition_combined_non_stop,
            "now": current_transition_combined_non_stop,
        },
        "continuation_non_stop_positives": {
            "before": previous_transition_cont_non_stop,
            "now": current_transition_cont_non_stop,
        },
        "fragrance_positives_combined": {
            "before": int(
                ((previous_transitions.get("slices") or {}).get("combined") or {}).get(
                    "fragrance_trusted_positives_count"
                )
                or 0
            ),
            "now": int(current_transitions.get("fragrance_positives_count") or 0),
        },
        "stop_rate": {
            "before": float(previous_transitions.get("stop_label_rate") or 0.0),
            "now": float(current_transitions.get("stop_rate") or 0.0),
        },
    }
    comparison["what_improved"] = improved
    comparison["what_is_still_weak"] = still_weak
    comparison["structural_gap_closed_for_initial"] = bool(current_initial_non_stop >= 100)
    comparison["structural_gap_closed_for_transitions"] = bool(current_transition_cont_non_stop >= 25)
    comparison["seeding_solved_structural_gap"] = (
        "partial"
        if comparison["structural_gap_closed_for_initial"] and not comparison["structural_gap_closed_for_transitions"]
        else ("yes" if comparison["structural_gap_closed_for_initial"] and comparison["structural_gap_closed_for_transitions"] else "no")
    )
    return comparison


def _category_positive_threshold_count(positives_by_category: dict[str, int], threshold: int) -> int:
    return sum(1 for value in positives_by_category.values() if int(value) >= threshold)


def _diverse_category_count(labels_present_by_category: dict[str, Any], minimum_labels: int) -> int:
    return sum(
        1
        for payload in labels_present_by_category.values()
        if int(payload.get("non_stop_labels_present_count") or 0) >= minimum_labels
    )


def _recommended_categories(
    *,
    positives_by_category: dict[str, int],
    labels_present_by_category: dict[str, Any],
    min_positive_support: int,
    min_coverage_ratio: float,
) -> list[str]:
    recommended: list[str] = []
    for category, payload in sorted(labels_present_by_category.items()):
        coverage_ratio = float(payload.get("non_stop_action_coverage_ratio") or 0.0)
        positive_support = int(positives_by_category.get(category) or 0)
        if positive_support >= min_positive_support and coverage_ratio >= min_coverage_ratio:
            recommended.append(category)
    return recommended


def _build_verdict(initial_metrics: dict[str, Any], transitions_metrics: dict[str, Any]) -> dict[str, Any]:
    initial_leakage_ok = (
        str((initial_metrics.get("time_leakage_checks") or {}).get("status") or "") == "passed"
        and str((initial_metrics.get("leakage_assertions") or {}).get("status") or "") == "passed"
    )
    transitions_leakage_ok = (
        str((transitions_metrics.get("time_leakage_checks") or {}).get("status") or "") == "passed"
        and str((transitions_metrics.get("leakage_assertions") or {}).get("status") or "") == "passed"
    )
    initial_categories_ge20 = _category_positive_threshold_count(initial_metrics["positives_by_category"], 20)
    initial_diverse_categories = _diverse_category_count(initial_metrics["labels_present_by_category"], 2)
    fragrance_payload = dict(initial_metrics["labels_present_by_category"].get("fragrance") or {})
    fragrance_diverse = int(fragrance_payload.get("non_stop_labels_present_count") or 0) >= 2
    fragrance_positive_count = int(initial_metrics.get("fragrance_positives_count") or 0)
    continuation_positive_count = int(transitions_metrics.get("continuation_positives_count") or 0)
    initial_recommended_categories = _recommended_categories(
        positives_by_category=initial_metrics["positives_by_category"],
        labels_present_by_category=initial_metrics["labels_present_by_category"],
        min_positive_support=20,
        min_coverage_ratio=0.75,
    )
    transition_recommended_categories = _recommended_categories(
        positives_by_category=transitions_metrics["positives_by_category"],
        labels_present_by_category=transitions_metrics["labels_present_by_category"],
        min_positive_support=40,
        min_coverage_ratio=0.50,
    )

    usable_for_initial_training = bool(
        initial_leakage_ok
        and int(initial_metrics.get("positives_excluding_stop") or 0) >= 100
        and initial_categories_ge20 >= 3
        and initial_diverse_categories >= 2
    )
    usable_for_transition_training = bool(
        transitions_leakage_ok
        and continuation_positive_count >= 25
        and _diverse_category_count(transitions_metrics["labels_present_by_category"], 2) >= 2
    )
    usable_for_multicategory_training = bool(
        usable_for_initial_training and initial_diverse_categories >= 3 and initial_categories_ge20 >= 3
    )
    usable_for_fragrance_training = bool(
        initial_leakage_ok and fragrance_positive_count >= 20 and fragrance_diverse
    )

    if usable_for_initial_training and usable_for_transition_training:
        recommended_next_block = "train_initial_and_transition"
    elif usable_for_initial_training:
        recommended_next_block = "train_initial_only"
    else:
        recommended_next_block = "improve_seeding_first"

    weak_initial_categories = sorted(
        category
        for category in initial_metrics["labels_present_by_category"].keys()
        if category not in initial_recommended_categories
    )
    weak_transition_categories = sorted(
        category
        for category in transitions_metrics["labels_present_by_category"].keys()
        if category not in transition_recommended_categories
    )
    fragrance_label_count = int(fragrance_payload.get("non_stop_labels_present_count") or 0)

    return {
        "usable_for_initial_training": usable_for_initial_training,
        "usable_for_transition_training": usable_for_transition_training,
        "usable_for_multicategory_training": usable_for_multicategory_training,
        "usable_for_fragrance_training": usable_for_fragrance_training,
        "ready_for_initial_planner_training": usable_for_initial_training,
        "ready_for_transition_planner_training": usable_for_transition_training,
        "recommended_next_block": recommended_next_block,
        "why": {
            "initial": (
                "Ready for training: "
                f"{int(initial_metrics.get('positives_excluding_stop') or 0)} non-stop positives with "
                f"{initial_diverse_categories} categories having >=2 non-stop labels. "
                f"Strongest initial slices: {initial_recommended_categories or []}. "
                f"Weaker initial slice: {weak_initial_categories or []}."
            ),
            "transitions": (
                "Ready for transition experiments: "
                f"{continuation_positive_count} continuation non-stop positives. "
                f"Strongest continuation slices: {transition_recommended_categories or []}. "
                f"Weaker continuation slice: {weak_transition_categories or []}."
            ),
            "multicategory": (
                "Multi-category training is viable because initial positives now cover "
                f"{initial_categories_ge20} categories with >=20 positives. "
                f"Makeup remains the narrowest category."
            ),
            "fragrance": (
                "Fragrance initial labels cover all slot actions: "
                f"{fragrance_label_count}/4 with {fragrance_positive_count} positives. "
                "Continuation fragrance exists too, but is skewed toward cold-evening follow-ups."
            ),
        },
        "recommended_first_dataset": "initial" if usable_for_initial_training else None,
        "recommended_first_categories": initial_recommended_categories if usable_for_initial_training else [],
        "recommended_exclusions": (
            weak_initial_categories if usable_for_initial_training and weak_initial_categories else []
        ),
    }


def _write_markdown(
    *,
    output_path: Path,
    initial_metrics: dict[str, Any],
    transitions_metrics: dict[str, Any],
    comparison: dict[str, Any],
    verdict: dict[str, Any],
) -> None:
    def _metric_lines(title: str, metrics: dict[str, Any]) -> list[str]:
        lines = [
            f"## {title}",
            "",
            f"- dataset dir: `{metrics['dataset_dir']}`",
            f"- decision points: **{metrics['decision_points_total']}**",
            f"- rows: **{metrics['rows_total']}**",
            f"- users: **{metrics['users_total']}**",
            f"- positives excluding `__stop__`: **{metrics['positives_excluding_stop']}**",
            f"- stop rate: **{metrics['stop_rate']:.6f}**",
            f"- excluded noisy decision points: **{metrics['excluded_noisy_decision_points_count']}** ({metrics['excluded_noisy_decision_points_share']:.6f})",
            f"- excluded bad fragrance exact completions: **{metrics['excluded_legacy_bad_fragrance_completions_count']}**",
            "",
            "### Positives By Category",
        ]
        for category, value in sorted(metrics["positives_by_category"].items()):
            lines.append(f"- {category}: **{value}**")
        lines.extend(["", "### Non-Stop Positives By Label Source"])
        for label_source, value in sorted(metrics["positives_by_label_source_non_stop"].items()):
            lines.append(f"- {label_source}: **{value}**")
        lines.extend(["", "### Labels Present By Category"])
        for category, payload in sorted(metrics["labels_present_by_category"].items()):
            lines.append(
                f"- {category}: non-stop labels={payload['non_stop_labels_present']} "
                f"({payload['non_stop_labels_present_count']}/{payload['available_non_stop_actions_count']})"
            )
        lines.extend(["", "### Split / Leakage Checks"])
        for split_name, payload in metrics["split_time_bounds"].items():
            lines.append(
                f"- {split_name}: decisions={payload['decision_points']}, users={payload['users']}, "
                f"t0_min={payload['t0_min_utc']}, t0_max={payload['t0_max_utc']}"
            )
        lines.append(f"- time leakage checks: **{metrics['time_leakage_checks']['status']}**")
        lines.append(f"- metadata leakage assertions: **{metrics['leakage_assertions'].get('status', 'unknown')}**")
        lines.append(f"- split user overlap: `{metrics['split_user_overlap_counts']}`")
        lines.extend(["", "### Class Balance By Category"])
        for category, payload in sorted(metrics["class_balance_by_category"].items()):
            summary = ", ".join(f"{label}={values['count']}" for label, values in sorted(payload.items()))
            lines.append(f"- {category}: {summary}")
        return lines

    lines = [
        "# Roadmap Live Dataset Audit",
        "",
        f"- generated_at_utc: `{timezone.now().astimezone().isoformat()}`",
        "",
    ]
    lines.extend(_metric_lines("Initial Planner Dataset", initial_metrics))
    lines.extend([""])
    lines.extend(_metric_lines("Transitions Planner Dataset", transitions_metrics))
    lines.extend(["", "## Comparison vs Previous Weak State", ""])
    if comparison.get("available"):
        initial_cmp = comparison["initial"]
        transitions_cmp = comparison["transitions"]
        lines.extend(
            [
                f"- initial decision points: **{initial_cmp['decision_points_total']['before']} -> {initial_cmp['decision_points_total']['now']}**",
                f"- initial non-stop positives: **{initial_cmp['non_stop_positives']['before']} -> {initial_cmp['non_stop_positives']['now']}**",
                f"- initial stop rate: **{initial_cmp['stop_rate']['before']:.6f} -> {initial_cmp['stop_rate']['now']:.6f}**",
                f"- transitions decision points: **{transitions_cmp['decision_points_total']['before']} -> {transitions_cmp['decision_points_total']['now']}**",
                f"- transitions combined non-stop positives: **{transitions_cmp['combined_non_stop_positives']['before']} -> {transitions_cmp['combined_non_stop_positives']['now']}**",
                f"- continuation non-stop positives: **{transitions_cmp['continuation_non_stop_positives']['before']} -> {transitions_cmp['continuation_non_stop_positives']['now']}**",
                f"- combined fragrance positives: **{transitions_cmp['fragrance_positives_combined']['before']} -> {transitions_cmp['fragrance_positives_combined']['now']}**",
                f"- what improved: `{comparison['what_improved']}`",
                f"- what is still weak: `{comparison['what_is_still_weak']}`",
                f"- seeding solved structural gap: **{comparison['seeding_solved_structural_gap']}**",
            ]
        )
    else:
        lines.append(f"- {comparison.get('note')}")
    lines.extend(
        [
            "",
            "## Verdict",
            "",
            f"- usable_for_initial_training: **{str(verdict['usable_for_initial_training']).lower()}**",
            f"- usable_for_transition_training: **{str(verdict['usable_for_transition_training']).lower()}**",
            f"- usable_for_multicategory_training: **{str(verdict['usable_for_multicategory_training']).lower()}**",
            f"- usable_for_fragrance_training: **{str(verdict['usable_for_fragrance_training']).lower()}**",
            f"- ready_for_initial_planner_training: **{str(verdict['ready_for_initial_planner_training']).lower()}**",
            f"- ready_for_transition_planner_training: **{str(verdict['ready_for_transition_planner_training']).lower()}**",
            f"- recommended_next_block: **{verdict['recommended_next_block']}**",
            "",
            "### Why",
            f"- initial: {verdict['why']['initial']}",
            f"- transitions: {verdict['why']['transitions']}",
            f"- multicategory: {verdict['why']['multicategory']}",
            f"- fragrance: {verdict['why']['fragrance']}",
            "",
            f"- recommended first dataset: `{verdict['recommended_first_dataset']}`",
            f"- recommended first categories: `{verdict['recommended_first_categories']}`",
            f"- recommended exclusions: `{verdict['recommended_exclusions']}`",
        ]
    )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


class Command(BaseCommand):
    help = "Audit rebuilt live planner datasets and compare them against the previous weak-state planner artifacts."

    def add_arguments(self, parser):
        parser.add_argument("--initial-dir", type=str, default="tmp/roadmap_planner_live_initial_v1")
        parser.add_argument("--transitions-dir", type=str, default="tmp/roadmap_planner_live_transitions_v1")
        parser.add_argument("--previous-initial-dir", type=str, default="tmp/roadmap_planner_prodstyle")
        parser.add_argument("--previous-transitions-dir", type=str, default="tmp/roadmap_planner_transitions_v2_3d")
        parser.add_argument("--output-md", type=str, default="reports/roadmap_live_dataset_audit.md")
        parser.add_argument("--output-json", type=str, default="reports/roadmap_live_dataset_audit.json")

    def handle(self, *args, **options):
        if pd is None:
            raise CommandError("pandas is required. Install dependencies from requirements-ml.txt")

        initial_dir = _resolve_path(str(options["initial_dir"]))
        transitions_dir = _resolve_path(str(options["transitions_dir"]))
        previous_initial_dir = _resolve_path(str(options["previous_initial_dir"]))
        previous_transitions_dir = _resolve_path(str(options["previous_transitions_dir"]))
        output_md = _resolve_path(str(options["output_md"]))
        output_json = _resolve_path(str(options["output_json"]))

        initial_bundle = _load_dataset_bundle(initial_dir)
        transitions_bundle = _load_dataset_bundle(transitions_dir)

        initial_metrics = _dataset_metrics(bundle=initial_bundle, dataset_kind="initial")
        transitions_metrics = _dataset_metrics(bundle=transitions_bundle, dataset_kind="transitions")
        comparison = _comparison_against_previous(
            previous_initial=_safe_read_previous_metadata(previous_initial_dir),
            previous_transitions=_safe_read_previous_metadata(previous_transitions_dir),
            current_initial=initial_metrics,
            current_transitions=transitions_metrics,
        )
        verdict = _build_verdict(initial_metrics, transitions_metrics)

        payload = {
            "generated_at_utc": timezone.now().astimezone().isoformat(),
            "initial_dataset": initial_metrics,
            "transitions_dataset": transitions_metrics,
            "comparison_vs_previous_weak_state": comparison,
            "verdict": verdict,
        }

        output_md.parent.mkdir(parents=True, exist_ok=True)
        output_json.parent.mkdir(parents=True, exist_ok=True)
        _write_markdown(
            output_path=output_md,
            initial_metrics=initial_metrics,
            transitions_metrics=transitions_metrics,
            comparison=comparison,
            verdict=verdict,
        )
        output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

        self.stdout.write(f"[report_roadmap_live_dataset_audit] md={output_md}")
        self.stdout.write(f"[report_roadmap_live_dataset_audit] json={output_json}")
