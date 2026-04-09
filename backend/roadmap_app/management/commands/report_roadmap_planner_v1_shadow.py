from __future__ import annotations

import json
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone as dt_timezone
from pathlib import Path
from typing import Any

from django.conf import settings
from django.core.management.base import BaseCommand, CommandError
from django.db.models import Prefetch
from django.test.utils import override_settings
from django.utils import timezone

from roadmap_app.ml_planner import (
    generate_planner_chain,
    planner_model_artifact_summary,
    planner_shadow_report_path_for_model_path,
)
from roadmap_app.models import RoadmapPlan, RoadmapStep
from roadmap_app.services import CATEGORY_RULES


def _unique(values: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        token = str(value or "").strip().lower()
        if not token or token in seen:
            continue
        seen.add(token)
        out.append(token)
    return out


def _prefix_match_rate(left: list[str], right: list[str]) -> float:
    if not right:
        return 0.0
    match = 0
    for a, b in zip(left, right):
        if str(a) != str(b):
            break
        match += 1
    return float(match / max(1, len(right)))


def _safe_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _selected_categories(raw: str) -> list[str]:
    token = str(raw or "all").strip().lower()
    if token in {"", "all"}:
        return list(CATEGORY_RULES.keys())
    if token not in CATEGORY_RULES:
        raise CommandError(f"Unsupported category: {token}")
    return [token]


def _build_markdown(report: dict[str, Any]) -> str:
    lines: list[str] = [
        "# Roadmap Planner V1 Shadow",
        "",
        f"- model_path: `{report.get('model_path')}`",
        f"- model_version: `{report.get('model_version')}`",
        f"- selected_feature_set: `{report.get('selected_feature_set')}`",
        f"- eligible_plans: `{_safe_dict(report.get('overall')).get('eligible_plans')}`",
        f"- exact_chain_match_rate: `{_safe_dict(report.get('overall')).get('exact_chain_match_rate')}`",
        f"- first_step_match_rate: `{_safe_dict(report.get('overall')).get('first_step_match_rate')}`",
        f"- mean_prefix_match_rate: `{_safe_dict(report.get('overall')).get('mean_prefix_match_rate')}`",
        "",
        "## By Category",
        "| category | eligible_plans | exact_chain_match_rate | first_step_match_rate | mean_prefix_match_rate |",
        "| --- | --- | --- | --- | --- |",
    ]
    for category, row in sorted(_safe_dict(report.get("by_category")).items()):
        row_dict = _safe_dict(row)
        lines.append(
            f"| {category} | {row_dict.get('eligible_plans', 0)} | "
            f"{row_dict.get('exact_chain_match_rate')} | {row_dict.get('first_step_match_rate')} | "
            f"{row_dict.get('mean_prefix_match_rate')} |"
        )
    samples = report.get("sample_cases") or []
    if samples:
        lines.extend(["", "## Sample Cases"])
        for row in samples:
            row_dict = _safe_dict(row)
            lines.append(
                f"- plan_id=`{row_dict.get('plan_id')}` category=`{row_dict.get('category')}` "
                f"baseline=`{row_dict.get('baseline_chain')}` predicted=`{row_dict.get('predicted_chain')}`"
            )
    return "\n".join(lines)


class Command(BaseCommand):
    help = "Artifact-local shadow report for roadmap planner_v1 against the current stable runtime baseline."

    def add_arguments(self, parser):
        parser.add_argument("--days", type=int, default=30)
        parser.add_argument("--category", type=str, default="all")
        parser.add_argument(
            "--include-ga",
            action="store_true",
            default=False,
            help='Include users with username starting with "ga_".',
        )
        parser.add_argument("--limit", type=int, default=None)
        parser.add_argument(
            "--model-path",
            type=str,
            default=str(getattr(settings, "ROADMAP_PLANNER_V1_MODEL_PATH", "") or "").strip(),
        )
        parser.add_argument("--sample-cases", type=int, default=20)
        parser.add_argument(
            "--output-json",
            type=str,
            default=None,
        )
        parser.add_argument(
            "--output-md",
            type=str,
            default=None,
        )

    def handle(self, *args, **options):
        days = int(options["days"] or 30)
        if days <= 0:
            raise CommandError("--days must be > 0")
        categories = _selected_categories(str(options["category"] or "all"))
        include_ga = bool(options.get("include_ga"))
        limit = options.get("limit")
        if limit is not None:
            limit = int(limit)
            if limit <= 0:
                raise CommandError("--limit must be > 0")
        sample_cases = max(0, int(options.get("sample_cases") or 20))
        model_path = str(options.get("model_path") or "").strip()
        if not model_path:
            raise CommandError("--model-path is required")

        default_shadow_path = planner_shadow_report_path_for_model_path(model_path)
        output_json = Path(str(options.get("output_json") or default_shadow_path).strip()).resolve()
        output_md = Path(
            str(options.get("output_md") or default_shadow_path.with_suffix(".md")).strip()
        ).resolve()

        since = timezone.now() - timedelta(days=days)
        plan_qs = (
            RoadmapPlan.objects.filter(updated_at__gte=since, category__in=categories, is_active=True)
            .select_related("user")
            .prefetch_related(
                Prefetch(
                    "steps",
                    queryset=RoadmapStep.objects.order_by("step_index", "id"),
                    to_attr="prefetched_steps",
                )
            )
            .order_by("-updated_at", "-id")
        )
        if not include_ga:
            plan_qs = plan_qs.exclude(user__username__startswith="ga_")
        if limit is not None:
            plan_qs = plan_qs[:limit]
        plans = list(plan_qs)

        artifact = planner_model_artifact_summary(model_path)
        overall = Counter()
        category_counts: dict[str, Counter[str]] = defaultdict(Counter)
        category_prefix_rates: dict[str, list[float]] = defaultdict(list)
        sample_rows: list[dict[str, Any]] = []

        with override_settings(
            ROADMAP_RUNTIME_FREEZE_ML=False,
            ROADMAP_PLANNER_V1_MODE="shadow",
            ROADMAP_PLANNER_V1_ENABLED_CATEGORIES=list(CATEGORY_RULES.keys()),
        ):
            for plan in plans:
                category = str(plan.category or "").strip().lower()
                rules = CATEGORY_RULES.get(category)
                if not rules:
                    continue
                steps = list(getattr(plan, "prefetched_steps", []) or [])
                baseline_chain = _unique([str(step.product_type) for step in steps if str(step.product_type).strip()])
                if not baseline_chain:
                    continue
                purchased_types = _unique(
                    [str(step.product_type) for step in steps if step.status == RoadmapStep.Status.COMPLETED]
                )
                owned_types_ordered = _unique(
                    [str(step.product_type) for step in steps if step.status == RoadmapStep.Status.OWNED]
                )
                refresh_caller = str(
                    _safe_dict(_safe_dict(plan.meta).get("context")).get("refresh_caller") or "planner_shadow_report"
                ).strip() or "planner_shadow_report"
                result = generate_planner_chain(
                    user=plan.user,
                    category=category,
                    candidate_types=baseline_chain,
                    purchased_types=purchased_types,
                    owned_types_ordered=owned_types_ordered,
                    min_steps=int(rules["min_steps"]),
                    max_steps=int(rules["max_steps"]),
                    refresh_caller=refresh_caller,
                )
                if str(result.get("decision") or "") != "model_used":
                    category_counts[category]["skipped_plans"] += 1
                    continue
                predicted_chain = _unique([str(x) for x in (result.get("chain") or []) if str(x).strip()])
                if not predicted_chain:
                    category_counts[category]["skipped_plans"] += 1
                    continue

                exact_hit = int(predicted_chain == baseline_chain)
                first_hit = int(
                    bool(predicted_chain and baseline_chain and predicted_chain[0] == baseline_chain[0])
                )
                prefix_rate = _prefix_match_rate(predicted_chain, baseline_chain)

                overall["eligible_plans"] += 1
                overall["exact_chain_hits"] += exact_hit
                overall["first_step_hits"] += first_hit
                category_counts[category]["eligible_plans"] += 1
                category_counts[category]["exact_chain_hits"] += exact_hit
                category_counts[category]["first_step_hits"] += first_hit
                category_prefix_rates[category].append(prefix_rate)

                if sample_cases and (
                    not exact_hit or not first_hit or prefix_rate < 1.0
                ) and len(sample_rows) < sample_cases:
                    sample_rows.append(
                        {
                            "plan_id": int(plan.id),
                            "user_id": int(plan.user_id),
                            "category": category,
                            "baseline_chain": baseline_chain,
                            "predicted_chain": predicted_chain,
                            "purchased_types": purchased_types,
                            "owned_types_ordered": owned_types_ordered,
                        }
                    )

        overall_prefix_rates = [rate for rates in category_prefix_rates.values() for rate in rates]
        report = {
            "generated_at_utc": datetime.now(dt_timezone.utc).isoformat(),
            "model_path": model_path,
            "model_version": str(artifact.get("model_version") or ""),
            "selected_feature_set": str(artifact.get("selected_feature_set") or ""),
            "scope": {
                "days": days,
                "categories": categories,
                "include_ga": include_ga,
                "limit": limit,
                "baseline": "current_active_plan_chain",
                "state_source": "current_plan_step_statuses",
            },
            "overall": {
                "eligible_plans": int(overall.get("eligible_plans", 0)),
                "exact_chain_match_rate": round(
                    float(overall.get("exact_chain_hits", 0)) / max(1, int(overall.get("eligible_plans", 0))),
                    6,
                ),
                "first_step_match_rate": round(
                    float(overall.get("first_step_hits", 0)) / max(1, int(overall.get("eligible_plans", 0))),
                    6,
                ),
                "mean_prefix_match_rate": round(
                    float(sum(overall_prefix_rates)) / max(1, len(overall_prefix_rates)),
                    6,
                ),
            },
            "by_category": {},
            "sample_cases": sample_rows,
        }
        for category in categories:
            counter = category_counts.get(category, Counter())
            prefix_rates = category_prefix_rates.get(category, [])
            eligible_plans = int(counter.get("eligible_plans", 0))
            report["by_category"][category] = {
                "eligible_plans": eligible_plans,
                "skipped_plans": int(counter.get("skipped_plans", 0)),
                "exact_chain_match_rate": round(
                    float(counter.get("exact_chain_hits", 0)) / max(1, eligible_plans),
                    6,
                ),
                "first_step_match_rate": round(
                    float(counter.get("first_step_hits", 0)) / max(1, eligible_plans),
                    6,
                ),
                "mean_prefix_match_rate": round(
                    float(sum(prefix_rates)) / max(1, len(prefix_rates)),
                    6,
                ),
            }

        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_md.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        output_md.write_text(_build_markdown(report), encoding="utf-8")

        self.stdout.write(f"[report_roadmap_planner_v1_shadow] json={output_json}")
        self.stdout.write(f"[report_roadmap_planner_v1_shadow] md={output_md}")
