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


class Command(BaseCommand):
    help = "Backfill RoadmapPlan.meta.ml.shadow for recent plans without changing roadmap steps."

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
            help="Optional override for shadow next-step model path. Defaults to ROADMAP_NEXTSTEP_V4_SHADOW_MODEL_PATH.",
        )
        parser.add_argument(
            "--write",
            action="store_true",
            default=False,
            help="Apply shadow payloads to DB. Default is dry-run.",
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

        model_path_raw = str(
            options.get("model_path")
            or getattr(settings, "ROADMAP_NEXTSTEP_V4_SHADOW_MODEL_PATH", "")
            or ""
        ).strip()
        if not model_path_raw:
            raise CommandError("Shadow model path is empty. Set ROADMAP_NEXTSTEP_V4_SHADOW_MODEL_PATH or pass --model-path.")

        model_path = str(Path(model_path_raw).expanduser())
        shadow_artifact = nextstep_model_artifact_summary(model_path)
        if not bool(shadow_artifact.get("exists")):
            raise CommandError(f"Shadow model file not found: {model_path}")

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

        scanned = 0
        skipped_counts: Counter[str] = Counter()
        reason_counts: Counter[str] = Counter()
        pending_updates: list[tuple[int, dict[str, Any], str]] = []
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

            active_model_path = str(ml.get("model_path") or "").strip()
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

            if active_model_path and Path(active_model_path).expanduser() == Path(model_path).expanduser():
                projected_shadow = {
                    "enabled": False,
                    "reason": "shadow_same_as_active",
                    "model_path": model_path,
                    "model_version": str(shadow_artifact.get("model_version") or ""),
                    "selected_feature_set": str(shadow_artifact.get("selected_feature_set") or ""),
                    "planned_target_product_type": planned_target_product_type,
                    "planned_target_step_index": planned_target_step_index,
                    "predictions": [],
                }
            else:
                predictions = predict_next_product_types_for_model_path(
                    model_path,
                    user=user_id,
                    context_product_ids=context_product_ids,
                    category=category_key,
                    planned_target_product_type=planned_target_product_type,
                    planned_target_step_index=planned_target_step_index,
                    candidate_types=candidate_types,
                )
                projected_shadow = {
                    "enabled": True,
                    "reason": "ok" if predictions else "no_predictions_or_model_unavailable",
                    "model_path": model_path,
                    "model_version": str(shadow_artifact.get("model_version") or ""),
                    "selected_feature_set": str(shadow_artifact.get("selected_feature_set") or ""),
                    "planned_target_product_type": planned_target_product_type,
                    "planned_target_step_index": planned_target_step_index,
                    "predictions": list(predictions[:10]),
                }

            updated_meta = dict(original_meta)
            updated_ml = dict(ml)
            updated_ml["shadow"] = projected_shadow
            updated_meta["ml"] = updated_ml
            reason_counts[str(projected_shadow.get("reason") or "__unknown__")] += 1
            if updated_meta == original_meta:
                skipped_counts["already_up_to_date"] += 1
                continue
            pending_updates.append((plan_id, updated_meta, category_key))
            if len(preview_rows) < 10:
                preview_rows.append(
                    {
                        "plan_id": plan_id,
                        "category": category_key,
                        "reason": str(projected_shadow.get("reason") or ""),
                        "shadow_model_version": str(projected_shadow.get("model_version") or ""),
                        "prediction_count": int(len(_safe_list(projected_shadow.get("predictions")))),
                    }
                )

        updated_count = 0
        if should_write:
            for plan_id, updated_meta, _ in pending_updates:
                updated_count += RoadmapPlan.objects.filter(id=plan_id).update(meta=updated_meta)

        self.stdout.write("# Roadmap Shadow Meta Backfill")
        self.stdout.write("")
        self.stdout.write(f"- mode: `{'write' if should_write else 'dry-run'}`")
        self.stdout.write(f"- analysis window: last `{days}` days")
        self.stdout.write(f"- category: `{category}`")
        self.stdout.write(f"- include ga_* users: `{include_ga}`")
        self.stdout.write(f"- active only: `{active_only}`")
        self.stdout.write(f"- shadow model path: `{model_path}`")
        self.stdout.write(f"- shadow model version: `{shadow_artifact.get('model_version') or 'n/a'}`")
        self.stdout.write(f"- plans scanned: `{scanned}`")
        self.stdout.write(f"- plans needing shadow backfill: `{len(pending_updates)}`")
        self.stdout.write(f"- plans updated: `{updated_count}`")
        self.stdout.write("")
        self.stdout.write("## Projected shadow reasons")
        if reason_counts:
            for key, value in sorted(reason_counts.items(), key=lambda kv: (-kv[1], kv[0])):
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
        self.stdout.write("## Example changes")
        if not preview_rows:
            self.stdout.write("- none")
            return
        for row in preview_rows:
            self.stdout.write(
                f"- plan_id=`{row['plan_id']}` category=`{row['category']}` "
                f"reason=`{row['reason']}` shadow_model_version=`{row['shadow_model_version']}` "
                f"predictions=`{row['prediction_count']}`"
            )
