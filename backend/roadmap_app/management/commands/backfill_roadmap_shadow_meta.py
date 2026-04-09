from __future__ import annotations

from collections import Counter, defaultdict
from datetime import timedelta
from typing import Any

from catalog.models import Product
from django.conf import settings
from django.core.management.base import BaseCommand, CommandError
from django.utils import timezone
from roadmap_app.historical_anchor_replay import build_historical_continuation_anchor_records
from roadmap_app.ml_next_step import (
    _load_model_for_path,
    _predict_with_v4_artifact_from_sources,
    nextstep_model_artifact_summary,
    predict_next_product_types_for_model_path,
)
from roadmap_app.models import RoadmapPlan, RoadmapStep
from roadmap_app.shadow_evidence import (
    build_control_evidence_payload,
    build_historical_control_evidence_payload,
    build_historical_shadow_evidence_payload,
    build_shadow_evidence_payload,
    merge_control_evidence_into_meta,
    merge_historical_control_evidence_into_meta,
    merge_historical_shadow_evidence_into_meta,
    merge_shadow_evidence_into_meta,
    normalized_model_path,
)
from transactions.models import TransactionItem
from users_app.models import CustomerProfile


CATEGORY_CHOICES = ["all", "skincare", "haircare", "makeup", "fragrance"]
REPLAY_MODE_CHOICES = ["current_snapshot", "historical_anchors"]


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


def _to_float(value: Any) -> float:
    try:
        return float(value or 0.0)
    except Exception:
        return 0.0


def _normalized_tx_item(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "ts": row["transaction__created_at"],
        "tx_id": int(row["transaction__id"]),
        "tx_total": _to_float(row.get("transaction__total_amount")),
        "product_id": int(row["product_id"]),
        "category": str(row.get("product__category") or "").strip().lower(),
        "product_type": str(row.get("product__product_type") or "").strip().lower(),
        "concerns": row.get("product__concerns") if isinstance(row.get("product__concerns"), list) else [],
        "actives": row.get("product__actives") if isinstance(row.get("product__actives"), list) else [],
        "flags": row.get("product__flags") if isinstance(row.get("product__flags"), list) else [],
        "supported_skin_types": (
            row.get("product__supported_skin_types")
            if isinstance(row.get("product__supported_skin_types"), list)
            else []
        ),
        "attrs": _safe_dict(row.get("product__attrs")),
        "ingredients_inci": str(row.get("product__ingredients_inci") or ""),
        "raw_meta": _safe_dict(row.get("product__raw_meta")),
        "quantity": max(1, int(row.get("quantity") or 0)),
    }


def _historical_context_product_ids(
    *,
    tx_items_desc_by_user: dict[int, list[dict[str, Any]]],
    user_id: int,
    anchor_time,
    limit: int = 50,
) -> list[int]:
    out: list[int] = []
    seen: set[int] = set()
    for row in tx_items_desc_by_user.get(int(user_id), []):
        if row["ts"] > anchor_time:
            continue
        product_id = int(row.get("product_id") or 0)
        if product_id <= 0 or product_id in seen:
            continue
        seen.add(product_id)
        out.append(product_id)
        if len(out) >= int(limit):
            break
    return out


def _historical_items_as_of(
    *,
    tx_items_asc_by_user: dict[int, list[dict[str, Any]]],
    user_id: int,
    anchor_time,
) -> list[dict[str, Any]]:
    return [
        dict(row)
        for row in tx_items_asc_by_user.get(int(user_id), [])
        if row["ts"] <= anchor_time
    ]


class Command(BaseCommand):
    help = "Backfill RoadmapPlan.meta.ml shadow and baseline-control evidence for recent plans without changing roadmap steps."

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
            "--replay-mode",
            type=str,
            default="current_snapshot",
            choices=REPLAY_MODE_CHOICES,
            help="Replay path to backfill. current_snapshot preserves the existing plan-based path; historical_anchors reconstructs immutable PLAN_REFRESHED anchors.",
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
        replay_mode = str(options.get("replay_mode") or "current_snapshot").strip().lower()
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

        model_path = normalized_model_path(model_path_raw)
        shadow_artifact = nextstep_model_artifact_summary(model_path)
        if not bool(shadow_artifact.get("exists")):
            raise CommandError(f"Shadow model file not found: {model_path}")
        configured_shadow_model_path = normalized_model_path(
            getattr(settings, "ROADMAP_NEXTSTEP_V4_SHADOW_MODEL_PATH", "") or ""
        )
        threshold = float(
            getattr(
                settings,
                "ROADMAP_NEXTSTEP_V4_CONFIDENCE_THRESHOLD",
                getattr(settings, "ROADMAP_NEXTSTEP_CONFIDENCE_THRESHOLD", 0.35),
            )
        )

        now_utc = timezone.now()
        since = now_utc - timedelta(days=days)

        if replay_mode == "historical_anchors":
            self._handle_historical_anchor_replay(
                since=since,
                now_utc=now_utc,
                category=category,
                include_ga=include_ga,
                active_only=active_only,
                limit=limit,
                should_write=should_write,
                model_path=model_path,
                shadow_artifact=shadow_artifact,
                threshold=threshold,
            )
            return

        self._handle_current_snapshot_replay(
            since=since,
            now_utc=now_utc,
            category=category,
            include_ga=include_ga,
            active_only=active_only,
            limit=limit,
            should_write=should_write,
            model_path=model_path,
            configured_shadow_model_path=configured_shadow_model_path,
            shadow_artifact=shadow_artifact,
            threshold=threshold,
            days=days,
        )

    def _handle_current_snapshot_replay(
        self,
        *,
        since,
        now_utc,
        category: str,
        include_ga: bool,
        active_only: bool,
        limit: int | None,
        should_write: bool,
        model_path: str,
        configured_shadow_model_path: str,
        shadow_artifact: dict[str, Any],
        threshold: float,
        days: int,
    ) -> None:
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
        baseline_control_by_plan: dict[int, dict[str, Any]] = {}
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
                if not planned_target_by_plan.get(plan_id) and product_type and status in {
                    RoadmapStep.Status.MISSING,
                    RoadmapStep.Status.RECOMMENDED,
                }:
                    planned_target_by_plan[plan_id] = {"product_type": product_type, "step_index": step_index}
                if plan_id not in baseline_control_by_plan and product_type and status in {
                    RoadmapStep.Status.MISSING,
                    RoadmapStep.Status.RECOMMENDED,
                }:
                    baseline_control_by_plan[plan_id] = {
                        "product_type": product_type,
                        "step_index": step_index,
                        "status": status,
                    }
                if not product_type or product_type in seen_by_plan[plan_id]:
                    continue
                seen_by_plan[plan_id].add(product_type)
                step_types_by_plan[plan_id].append(product_type)
                if plan_id not in planned_target_by_plan and product_type:
                    planned_target_by_plan[plan_id] = {"product_type": product_type, "step_index": step_index}

        scanned = 0
        skipped_counts: Counter[str] = Counter()
        reason_counts: Counter[str] = Counter()
        comparable_decision_counts: Counter[str] = Counter()
        control_reason_counts: Counter[str] = Counter()
        control_decision_counts: Counter[str] = Counter()
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
            baseline_control = baseline_control_by_plan.get(plan_id) or {}
            control_product_type = str(baseline_control.get("product_type") or "").strip().lower()
            control_step_index = int(baseline_control.get("step_index") or 0)
            control_step_status = str(baseline_control.get("status") or "").strip().lower()

            predictions = predict_next_product_types_for_model_path(
                model_path,
                user=user_id,
                context_product_ids=context_product_ids,
                category=category_key,
                planned_target_product_type=planned_target_product_type,
                planned_target_step_index=planned_target_step_index,
                candidate_types=candidate_types,
            )
            projected_shadow = build_shadow_evidence_payload(
                model_path=model_path,
                model_version=str(shadow_artifact.get("model_version") or ""),
                selected_feature_set=str(shadow_artifact.get("selected_feature_set") or ""),
                plan_id=plan_id,
                category=category_key,
                plan_updated_at=str(original_meta.get("generated_at") or ""),
                evidence_generated_at=now_utc.isoformat(),
                threshold=threshold,
                candidate_types=candidate_types,
                planned_target_product_type=planned_target_product_type,
                planned_target_step_index=planned_target_step_index,
                prediction_reason="ok" if predictions else "no_predictions_or_model_unavailable",
                predictions=list(predictions[:10]),
                was_model_considered=True,
            )
            projected_control = build_control_evidence_payload(
                model_path=model_path,
                plan_id=plan_id,
                category=category_key,
                plan_updated_at=str(original_meta.get("generated_at") or ""),
                evidence_generated_at=now_utc.isoformat(),
                candidate_types=candidate_types,
                selected_product_type=control_product_type,
                selected_step_index=control_step_index,
                selected_step_status=control_step_status,
                planned_target_product_type=planned_target_product_type,
                planned_target_step_index=planned_target_step_index,
                baseline_source="current_rule_plan",
                was_control_available=bool(control_product_type),
                comparable_reason=("selected_current_plan_next_step" if control_product_type else "no_actionable_step"),
            )

            updated_meta = merge_shadow_evidence_into_meta(
                original_meta,
                projected_shadow,
                set_legacy_shadow=(model_path == configured_shadow_model_path),
            )
            updated_meta = merge_control_evidence_into_meta(updated_meta, projected_control)
            reason_counts[str(projected_shadow.get("reason") or "__unknown__")] += 1
            comparable_decision_counts[str(projected_shadow.get("comparable_decision") or "__unknown__")] += 1
            control_reason_counts[str(projected_control.get("comparable_reason") or "__unknown__")] += 1
            control_decision_counts[str(projected_control.get("comparable_decision") or "__unknown__")] += 1
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
                        "comparable_decision": str(projected_shadow.get("comparable_decision") or ""),
                        "top1_product_type": str(projected_shadow.get("top1_product_type") or ""),
                        "control_product_type": control_product_type,
                        "control_decision": str(projected_control.get("comparable_decision") or ""),
                    }
                )

        updated_count = 0
        if should_write:
            for plan_id, updated_meta, _ in pending_updates:
                updated_count += RoadmapPlan.objects.filter(id=plan_id).update(meta=updated_meta)

        self.stdout.write("# Roadmap Shadow Meta Backfill")
        self.stdout.write("")
        self.stdout.write(f"- replay_mode: `current_snapshot`")
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
        self.stdout.write("## Comparable decisions")
        if comparable_decision_counts:
            for key, value in sorted(comparable_decision_counts.items(), key=lambda kv: (-kv[1], kv[0])):
                self.stdout.write(f"- {key}: `{int(value)}`")
        else:
            self.stdout.write("- none")
        self.stdout.write("")
        self.stdout.write("## Baseline control decisions")
        if control_decision_counts:
            for key, value in sorted(control_decision_counts.items(), key=lambda kv: (-kv[1], kv[0])):
                self.stdout.write(f"- {key}: `{int(value)}`")
        else:
            self.stdout.write("- none")
        self.stdout.write("")
        self.stdout.write("## Baseline control reasons")
        if control_reason_counts:
            for key, value in sorted(control_reason_counts.items(), key=lambda kv: (-kv[1], kv[0])):
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
                f"decision=`{row['comparable_decision']}` top1=`{row['top1_product_type'] or '-'}` "
                f"predictions=`{row['prediction_count']}` control=`{row['control_decision']}` "
                f"baseline_top1=`{row['control_product_type'] or '-'}`"
            )

    def _handle_historical_anchor_replay(
        self,
        *,
        since,
        now_utc,
        category: str,
        include_ga: bool,
        active_only: bool,
        limit: int | None,
        should_write: bool,
        model_path: str,
        shadow_artifact: dict[str, Any],
        threshold: float,
    ) -> None:
        anchors = build_historical_continuation_anchor_records(
            since=since,
            until=now_utc,
            category=category,
            include_ga=include_ga,
        )
        if limit is not None:
            anchors = anchors[:limit]

        plan_ids = sorted({int(anchor["plan_id"]) for anchor in anchors if int(anchor.get("plan_id") or 0) > 0})
        plan_rows = {
            int(row["id"]): row
            for row in RoadmapPlan.objects.filter(id__in=plan_ids).values(
                "id",
                "user_id",
                "category",
                "is_active",
                "meta",
            )
        }
        if active_only:
            anchors = [
                anchor
                for anchor in anchors
                if bool(_safe_dict(plan_rows.get(int(anchor["plan_id"]))).get("is_active"))
            ]

        artifact_obj = _load_model_for_path(model_path)
        artifact_loaded = isinstance(artifact_obj, dict) and str(artifact_obj.get("task") or "") == "roadmap_nextstep_v4_ranking"

        user_ids = sorted({int(anchor["user_id"]) for anchor in anchors if int(anchor.get("user_id") or 0) > 0})
        tx_items_asc_by_user: dict[int, list[dict[str, Any]]] = defaultdict(list)
        tx_items_desc_by_user: dict[int, list[dict[str, Any]]] = defaultdict(list)
        product_ids_for_context: set[int] = set()
        if user_ids and artifact_loaded:
            tx_rows = list(
                TransactionItem.objects.filter(
                    transaction__user_id__in=user_ids,
                    transaction__created_at__lte=now_utc,
                )
                .values(
                    "product_id",
                    "transaction__id",
                    "transaction__created_at",
                    "transaction__total_amount",
                    "product__category",
                    "product__product_type",
                    "quantity",
                    "product__concerns",
                    "product__actives",
                    "product__flags",
                    "product__supported_skin_types",
                    "product__attrs",
                    "product__ingredients_inci",
                    "product__raw_meta",
                    "transaction__user_id",
                )
                .order_by("transaction__user_id", "transaction__created_at", "transaction__id", "product_id")
            )
            for row in tx_rows:
                user_id = int(row["transaction__user_id"])
                item = _normalized_tx_item(row)
                tx_items_asc_by_user[user_id].append(item)
                tx_items_desc_by_user[user_id].append(item)
                product_id = int(item.get("product_id") or 0)
                if product_id > 0:
                    product_ids_for_context.add(product_id)
            for user_id in list(tx_items_desc_by_user.keys()):
                tx_items_desc_by_user[user_id] = sorted(
                    tx_items_desc_by_user[user_id],
                    key=lambda row: (row["ts"], int(row.get("tx_id") or 0), int(row.get("product_id") or 0)),
                    reverse=True,
                )

        product_rows_by_id = {
            int(row["id"]): row
            for row in Product.objects.filter(id__in=sorted(product_ids_for_context)).values(
                "id",
                "category",
                "product_type",
                "concerns",
                "actives",
                "flags",
                "supported_skin_types",
                "attrs",
                "ingredients_inci",
                "raw_meta",
            )
        }
        profile_by_user = {
            int(profile.user_id): profile
            for profile in CustomerProfile.objects.filter(user_id__in=user_ids)
        }
        categories_in_scope = sorted(
            {str(anchor.get("category") or "").strip().lower() for anchor in anchors if str(anchor.get("category") or "").strip()}
        )
        catalog_by_category = {
            category_key: list(
                Product.objects.filter(category=category_key).values(
                    "category",
                    "product_type",
                    "concerns",
                    "actives",
                    "flags",
                    "supported_skin_types",
                    "attrs",
                    "ingredients_inci",
                    "raw_meta",
                )
            )
            for category_key in categories_in_scope
        }

        pending_meta_by_plan = {
            int(plan_id): _safe_dict(row.get("meta"))
            for plan_id, row in plan_rows.items()
        }
        original_meta_by_plan = {
            int(plan_id): _safe_dict(row.get("meta"))
            for plan_id, row in plan_rows.items()
        }

        scanned = 0
        skipped_counts: Counter[str] = Counter()
        reconstruction_counts: Counter[str] = Counter()
        reason_counts: Counter[str] = Counter()
        comparable_decision_counts: Counter[str] = Counter()
        control_reason_counts: Counter[str] = Counter()
        control_decision_counts: Counter[str] = Counter()
        preview_rows: list[dict[str, Any]] = []

        for anchor in anchors:
            scanned += 1
            plan_id = int(anchor.get("plan_id") or 0)
            user_id = int(anchor.get("user_id") or 0)
            category_key = str(anchor.get("category") or "").strip().lower() or "__unknown__"
            plan_row = plan_rows.get(plan_id)
            if not plan_row:
                skipped_counts["missing_plan_row"] += 1
                continue

            original_meta = pending_meta_by_plan.get(plan_id) or _safe_dict(plan_row.get("meta"))
            anchor_created_at = anchor.get("anchor_created_at")
            anchor_key = str(anchor.get("anchor_key") or "")
            reconstruction_reason = str(anchor.get("reconstruction_reason") or "").strip()
            reconstruction_counts[reconstruction_reason or "ok"] += 1

            candidate_types = [
                str(item or "").strip().lower()
                for item in (anchor.get("candidate_types") or [])
                if str(item or "").strip()
            ]
            planned_target_product_type = str(anchor.get("planned_target_product_type") or "").strip().lower()
            planned_target_step_index = int(anchor.get("planned_target_step_index") or 0)
            prediction_reason = reconstruction_reason
            predictions: list[dict[str, Any]] = []
            was_model_considered = bool(candidate_types)

            if artifact_loaded and anchor_created_at is not None and candidate_types:
                context_product_ids = _historical_context_product_ids(
                    tx_items_desc_by_user=tx_items_desc_by_user,
                    user_id=user_id,
                    anchor_time=anchor_created_at,
                )
                items = _historical_items_as_of(
                    tx_items_asc_by_user=tx_items_asc_by_user,
                    user_id=user_id,
                    anchor_time=anchor_created_at,
                )
                context_products = [
                    dict(product_rows_by_id[product_id])
                    for product_id in context_product_ids
                    if int(product_id) in product_rows_by_id
                ]
                predictions = _predict_with_v4_artifact_from_sources(
                    artifact=artifact_obj,
                    category=category_key,
                    now_utc=anchor_created_at,
                    items=items,
                    profile=profile_by_user.get(user_id),
                    context_products=context_products,
                    catalog_products=list(catalog_by_category.get(category_key) or []),
                    planned_target_product_type=planned_target_product_type,
                    planned_target_step_index=planned_target_step_index,
                    candidate_types=candidate_types,
                )
            elif candidate_types:
                context_product_ids = []
                if anchor_created_at is not None:
                    context_product_ids = _historical_context_product_ids(
                        tx_items_desc_by_user=tx_items_desc_by_user,
                        user_id=user_id,
                        anchor_time=anchor_created_at,
                    )
                predictions = predict_next_product_types_for_model_path(
                    model_path,
                    user=user_id,
                    context_product_ids=context_product_ids,
                    category=category_key,
                    planned_target_product_type=planned_target_product_type,
                    planned_target_step_index=planned_target_step_index,
                    candidate_types=candidate_types,
                )

            if predictions and not prediction_reason:
                prediction_reason = "ok"
            elif not prediction_reason:
                prediction_reason = "no_predictions_or_model_unavailable"

            projected_shadow = build_historical_shadow_evidence_payload(
                anchor_key=anchor_key,
                anchor_event_id=int(anchor.get("anchor_event_id") or 0),
                anchor_created_at=anchor_created_at.isoformat() if anchor_created_at is not None else "",
                anchor_source="plan_refreshed",
                reconstruction_reason=reconstruction_reason,
                reconstructed_candidate_types=candidate_types,
                model_path=model_path,
                model_version=str(shadow_artifact.get("model_version") or ""),
                selected_feature_set=str(shadow_artifact.get("selected_feature_set") or ""),
                plan_id=plan_id,
                category=category_key,
                plan_updated_at=anchor_created_at.isoformat() if anchor_created_at is not None else "",
                evidence_generated_at=now_utc.isoformat(),
                threshold=threshold,
                candidate_types=candidate_types,
                planned_target_product_type=planned_target_product_type,
                planned_target_step_index=planned_target_step_index,
                prediction_reason=prediction_reason,
                predictions=list(predictions[:10]),
                was_model_considered=was_model_considered,
            )
            projected_control = build_historical_control_evidence_payload(
                anchor_key=anchor_key,
                anchor_event_id=int(anchor.get("anchor_event_id") or 0),
                anchor_created_at=anchor_created_at.isoformat() if anchor_created_at is not None else "",
                anchor_source="plan_refreshed",
                reconstruction_reason=reconstruction_reason,
                model_path=model_path,
                plan_id=plan_id,
                category=category_key,
                plan_updated_at=anchor_created_at.isoformat() if anchor_created_at is not None else "",
                evidence_generated_at=now_utc.isoformat(),
                candidate_types=candidate_types,
                selected_product_type=planned_target_product_type,
                selected_step_index=planned_target_step_index,
                selected_step_status="recommended",
                planned_target_product_type=planned_target_product_type,
                planned_target_step_index=planned_target_step_index,
                baseline_source="historical_plan_refresh",
                was_control_available=bool(planned_target_product_type),
                comparable_reason=(
                    "selected_historical_next_step"
                    if planned_target_product_type
                    else (reconstruction_reason or "no_actionable_step")
                ),
            )

            updated_meta = merge_historical_shadow_evidence_into_meta(original_meta, projected_shadow)
            updated_meta = merge_historical_control_evidence_into_meta(updated_meta, projected_control)
            pending_meta_by_plan[plan_id] = updated_meta
            reason_counts[str(projected_shadow.get("reason") or "__unknown__")] += 1
            comparable_decision_counts[str(projected_shadow.get("comparable_decision") or "__unknown__")] += 1
            control_reason_counts[str(projected_control.get("comparable_reason") or "__unknown__")] += 1
            control_decision_counts[str(projected_control.get("comparable_decision") or "__unknown__")] += 1

            if len(preview_rows) < 10:
                preview_rows.append(
                    {
                        "plan_id": plan_id,
                        "anchor_key": anchor_key,
                        "category": category_key,
                        "reconstruction_reason": reconstruction_reason or "ok",
                        "reason": str(projected_shadow.get("reason") or ""),
                        "shadow_model_version": str(projected_shadow.get("model_version") or ""),
                        "prediction_count": int(len(_safe_list(projected_shadow.get("predictions")))),
                        "comparable_decision": str(projected_shadow.get("comparable_decision") or ""),
                        "top1_product_type": str(projected_shadow.get("top1_product_type") or ""),
                        "control_product_type": planned_target_product_type,
                        "control_decision": str(projected_control.get("comparable_decision") or ""),
                    }
                )

        pending_updates = [
            (plan_id, updated_meta)
            for plan_id, updated_meta in pending_meta_by_plan.items()
            if updated_meta != original_meta_by_plan.get(plan_id)
        ]

        updated_count = 0
        if should_write:
            for plan_id, updated_meta in pending_updates:
                updated_count += RoadmapPlan.objects.filter(id=plan_id).update(meta=updated_meta)

        self.stdout.write("# Roadmap Shadow Meta Backfill")
        self.stdout.write("")
        self.stdout.write(f"- replay_mode: `historical_anchors`")
        self.stdout.write(f"- mode: `{'write' if should_write else 'dry-run'}`")
        self.stdout.write(f"- category: `{category}`")
        self.stdout.write(f"- include ga_* users: `{include_ga}`")
        self.stdout.write(f"- active only: `{active_only}`")
        self.stdout.write(f"- shadow model path: `{model_path}`")
        self.stdout.write(f"- shadow model version: `{shadow_artifact.get('model_version') or 'n/a'}`")
        self.stdout.write(f"- anchors scanned: `{scanned}`")
        self.stdout.write(f"- plans touched: `{len(pending_meta_by_plan)}`")
        self.stdout.write(f"- plans needing historical backfill: `{len(pending_updates)}`")
        self.stdout.write(f"- plans updated: `{updated_count}`")
        self.stdout.write("")
        self.stdout.write("## Historical reconstruction")
        if reconstruction_counts:
            for key, value in sorted(reconstruction_counts.items(), key=lambda kv: (-kv[1], kv[0])):
                self.stdout.write(f"- {key}: `{int(value)}`")
        else:
            self.stdout.write("- none")
        self.stdout.write("")
        self.stdout.write("## Projected shadow reasons")
        if reason_counts:
            for key, value in sorted(reason_counts.items(), key=lambda kv: (-kv[1], kv[0])):
                self.stdout.write(f"- {key}: `{int(value)}`")
        else:
            self.stdout.write("- none")
        self.stdout.write("")
        self.stdout.write("## Comparable decisions")
        if comparable_decision_counts:
            for key, value in sorted(comparable_decision_counts.items(), key=lambda kv: (-kv[1], kv[0])):
                self.stdout.write(f"- {key}: `{int(value)}`")
        else:
            self.stdout.write("- none")
        self.stdout.write("")
        self.stdout.write("## Baseline control decisions")
        if control_decision_counts:
            for key, value in sorted(control_decision_counts.items(), key=lambda kv: (-kv[1], kv[0])):
                self.stdout.write(f"- {key}: `{int(value)}`")
        else:
            self.stdout.write("- none")
        self.stdout.write("")
        self.stdout.write("## Baseline control reasons")
        if control_reason_counts:
            for key, value in sorted(control_reason_counts.items(), key=lambda kv: (-kv[1], kv[0])):
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
                f"- plan_id=`{row['plan_id']}` anchor=`{row['anchor_key']}` category=`{row['category']}` "
                f"reconstruction=`{row['reconstruction_reason']}` reason=`{row['reason']}` "
                f"shadow_model_version=`{row['shadow_model_version']}` decision=`{row['comparable_decision']}` "
                f"top1=`{row['top1_product_type'] or '-'}` predictions=`{row['prediction_count']}` "
                f"control=`{row['control_decision']}` baseline_top1=`{row['control_product_type'] or '-'}`"
            )
