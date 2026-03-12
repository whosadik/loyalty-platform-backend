from __future__ import annotations

from collections import Counter, defaultdict
from datetime import timedelta
from pathlib import Path
from typing import Any

from django.conf import settings
from django.core.management.base import BaseCommand, CommandError
from django.utils import timezone

from roadmap_app.ml_next_step import (
    nextstep_model_artifact_summary,
    predict_next_product_types_for_model_path,
)
from roadmap_app.models import RoadmapPlan, RoadmapStep


CATEGORY_CHOICES = ["all", "skincare", "haircare", "makeup", "fragrance"]


def _safe_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _safe_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _normalize_product_ids(value: Any) -> list[int]:
    out: list[int] = []
    for item in _safe_list(value):
        try:
            out.append(int(item))
        except Exception:
            continue
    return out


def _runtime_policy_payload(predictions: list[dict[str, Any]]) -> tuple[list[str], dict[str, float]]:
    policy_names: set[str] = set()
    max_abs_bias: dict[str, float] = {}
    for row in predictions:
        if not isinstance(row, dict):
            continue
        for raw_policy in row.get("runtime_policies") or []:
            policy_name = str(raw_policy or "").strip()
            if policy_name:
                policy_names.add(policy_name)
        raw_biases = row.get("runtime_policy_biases")
        if not isinstance(raw_biases, dict):
            continue
        for raw_policy_name, raw_bias_value in raw_biases.items():
            policy_name = str(raw_policy_name or "").strip()
            if not policy_name:
                continue
            try:
                bias_value = abs(float(raw_bias_value or 0.0))
            except Exception:
                continue
            prev = float(max_abs_bias.get(policy_name, 0.0))
            if bias_value > prev:
                max_abs_bias[policy_name] = float(round(bias_value, 6))
    return sorted(policy_names), {
        str(k): float(v) for k, v in sorted(max_abs_bias.items(), key=lambda kv: kv[0])
    }


class Command(BaseCommand):
    help = "Backfill RoadmapPlan.meta.ml.runtime_policies for recent plans without changing roadmap steps."

    def add_arguments(self, parser):
        parser.add_argument("--days", type=int, default=30)
        parser.add_argument("--category", type=str, default="all", choices=CATEGORY_CHOICES)
        parser.add_argument(
            "--include-ga",
            action="store_true",
            default=False,
            help='Include users with username starting with "ga_".',
        )
        parser.add_argument(
            "--active-only",
            action="store_true",
            default=False,
            help="Only scan currently active plans.",
        )
        parser.add_argument("--limit", type=int, default=None)
        parser.add_argument(
            "--model-path",
            type=str,
            default=None,
            help="Optional override for active next-step model path. Defaults to ml.model_path per plan, then ROADMAP_NEXTSTEP_V4_MODEL_PATH.",
        )
        parser.add_argument(
            "--write",
            action="store_true",
            default=False,
            help="Apply runtime policy payloads to DB. Default is dry-run.",
        )

    def handle(self, *args, **options):
        days = int(options["days"] or 30)
        if days <= 0:
            raise CommandError("--days must be > 0")

        category = str(options["category"] or "all").strip().lower()
        include_ga = bool(options["include_ga"])
        active_only = bool(options["active_only"])
        should_write = bool(options["write"])
        limit = options.get("limit")
        if limit is not None:
            limit = int(limit)
            if limit <= 0:
                raise CommandError("--limit must be > 0")

        override_model_path = str(options.get("model_path") or "").strip()
        default_model_path = str(getattr(settings, "ROADMAP_NEXTSTEP_V4_MODEL_PATH", "") or "").strip()

        now_utc = timezone.now()
        since = now_utc - timedelta(days=days)

        qs = RoadmapPlan.objects.filter(updated_at__gte=since, updated_at__lte=now_utc).order_by("-updated_at", "-id")
        if category != "all":
            qs = qs.filter(category=category)
        if active_only:
            qs = qs.filter(is_active=True)
        if not include_ga:
            qs = qs.exclude(user__username__startswith="ga_")
        if limit is not None:
            qs = qs[:limit]

        plan_rows = list(qs.values("id", "user_id", "category", "meta"))
        plan_ids = [int(row["id"]) for row in plan_rows]
        step_types_by_plan: dict[int, list[str]] = defaultdict(list)
        planned_target_by_plan: dict[int, dict[str, Any]] = {}
        seen_by_plan: dict[int, set[str]] = defaultdict(set)
        if plan_ids:
            step_rows = RoadmapStep.objects.filter(plan_id__in=plan_ids).order_by("plan_id", "step_index", "id").values(
                "plan_id",
                "product_type",
                "step_index",
                "status",
            )
            for row in step_rows:
                plan_id = int(row["plan_id"])
                product_type = str(row.get("product_type") or "").strip().lower()
                step_index = int(row.get("step_index") or 0)
                status = str(row.get("status") or "").strip()
                target = planned_target_by_plan.get(plan_id)
                if target is None and product_type and status in {
                    RoadmapStep.Status.MISSING,
                    RoadmapStep.Status.RECOMMENDED,
                }:
                    planned_target_by_plan[plan_id] = {
                        "product_type": product_type,
                        "step_index": step_index,
                    }
                if not product_type or product_type in seen_by_plan[plan_id]:
                    continue
                seen_by_plan[plan_id].add(product_type)
                step_types_by_plan[plan_id].append(product_type)
                if plan_id not in planned_target_by_plan and product_type:
                    planned_target_by_plan[plan_id] = {
                        "product_type": product_type,
                        "step_index": step_index,
                    }

        artifact_cache: dict[str, dict[str, Any]] = {}

        def _artifact_summary_for(path_value: str) -> dict[str, Any]:
            model_path_value = str(Path(path_value).expanduser())
            cached = artifact_cache.get(model_path_value)
            if cached is not None:
                return cached
            cached = nextstep_model_artifact_summary(model_path_value)
            artifact_cache[model_path_value] = cached
            return cached

        scanned = 0
        skipped_counts: Counter[str] = Counter()
        projected_policy_counts: Counter[str] = Counter()
        pending_updates: list[tuple[int, dict[str, Any], str, list[str]]] = []
        preview_rows: list[dict[str, Any]] = []

        for row in plan_rows:
            scanned += 1
            plan_id = int(row["id"])
            user_id = int(row["user_id"])
            category_key = str(row.get("category") or "").strip().lower() or "__unknown__"
            original_meta = _safe_dict(row.get("meta"))
            ml = _safe_dict(original_meta.get("ml"))
            if not ml:
                skipped_counts["missing_ml_payload"] += 1
                continue

            candidate_types = list(step_types_by_plan.get(plan_id) or [])
            if not candidate_types:
                skipped_counts["no_candidate_types"] += 1
                continue

            stored_model_path = str(ml.get("model_path") or "").strip()
            plan_model_path = str(override_model_path or stored_model_path or default_model_path).strip()
            if not plan_model_path:
                skipped_counts["missing_model_path"] += 1
                continue
            artifact = _artifact_summary_for(plan_model_path)
            if (
                not bool(artifact.get("exists"))
                and not override_model_path
                and default_model_path
                and Path(plan_model_path).expanduser() != Path(default_model_path).expanduser()
            ):
                fallback_artifact = _artifact_summary_for(default_model_path)
                if bool(fallback_artifact.get("exists")):
                    plan_model_path = default_model_path
                    artifact = fallback_artifact
            if not bool(artifact.get("exists")):
                skipped_counts["missing_model_file"] += 1
                continue

            context = _safe_dict(original_meta.get("context"))
            context_product_ids = _normalize_product_ids(context.get("post_ctx_product_ids"))
            planned_target = planned_target_by_plan.get(plan_id) or {}
            planned_target_product_type = str(
                planned_target.get("product_type")
                or ml.get("planned_target_product_type")
                or ""
            ).strip().lower()
            planned_target_step_index = int(
                planned_target.get("step_index")
                or ml.get("planned_target_step_index")
                or 0
            )

            predictions = predict_next_product_types_for_model_path(
                plan_model_path,
                user=user_id,
                context_product_ids=context_product_ids,
                category=category_key,
                planned_target_product_type=planned_target_product_type,
                planned_target_step_index=planned_target_step_index,
                candidate_types=candidate_types,
            )
            runtime_policies, runtime_policy_max_abs_bias = _runtime_policy_payload(predictions)
            updated_meta = dict(original_meta)
            updated_ml = dict(ml)
            updated_ml["runtime_policies"] = runtime_policies
            updated_ml["runtime_policy_max_abs_bias"] = runtime_policy_max_abs_bias
            updated_ml["runtime_policy_meta_source"] = "backfill_projection"
            updated_ml["runtime_policy_meta_updated_at"] = now_utc.isoformat()
            updated_meta["ml"] = updated_ml

            for policy_name in runtime_policies:
                projected_policy_counts[str(policy_name)] += 1

            if updated_meta == original_meta:
                skipped_counts["already_up_to_date"] += 1
                continue
            pending_updates.append((plan_id, updated_meta, category_key, runtime_policies))
            if len(preview_rows) < 10:
                preview_rows.append(
                    {
                        "plan_id": plan_id,
                        "category": category_key,
                        "model_version": str(artifact.get("model_version") or ""),
                        "runtime_policies": runtime_policies,
                    }
                )

        updated_count = 0
        if should_write:
            for plan_id, updated_meta, _, _ in pending_updates:
                updated_count += RoadmapPlan.objects.filter(id=plan_id).update(meta=updated_meta)

        self.stdout.write("# Roadmap Runtime Policy Meta Backfill")
        self.stdout.write("")
        self.stdout.write(f"- mode: `{'write' if should_write else 'dry-run'}`")
        self.stdout.write(f"- analysis window: last `{days}` days")
        self.stdout.write(f"- category: `{category}`")
        self.stdout.write(f"- include ga_* users: `{include_ga}`")
        self.stdout.write(f"- active only: `{active_only}`")
        self.stdout.write(f"- default model path: `{override_model_path or default_model_path or 'per-plan only'}`")
        self.stdout.write(f"- plans scanned: `{scanned}`")
        self.stdout.write(f"- plans needing runtime policy backfill: `{len(pending_updates)}`")
        self.stdout.write(f"- plans updated: `{updated_count}`")
        self.stdout.write("")
        self.stdout.write("## Projected runtime policies")
        if projected_policy_counts:
            for key, value in sorted(projected_policy_counts.items(), key=lambda kv: (-kv[1], kv[0])):
                self.stdout.write(f"- {key}: `{int(value)}`")
        else:
            self.stdout.write("- none")
        self.stdout.write("")
        self.stdout.write("## Skipped")
        if skipped_counts:
            for key, value in sorted(skipped_counts.items(), key=lambda kv: (-kv[1], kv[0])):
                self.stdout.write(f"- {key}: `{int(value)}`")
        else:
            self.stdout.write("- none")
        self.stdout.write("")
        self.stdout.write("## Preview")
        if preview_rows:
            for row in preview_rows:
                runtime_policies = ", ".join(row.get("runtime_policies") or []) or "-"
                self.stdout.write(
                    f"- plan `{row['plan_id']}` [{row['category']}] "
                    f"model=`{row['model_version'] or 'n/a'}` runtime_policies=`{runtime_policies}`"
                )
        else:
            self.stdout.write("- none")
