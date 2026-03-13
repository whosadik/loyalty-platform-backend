from __future__ import annotations

from collections import Counter
from datetime import timedelta
from typing import Any

from django.core.management.base import BaseCommand, CommandError
from django.db import transaction
from django.utils import timezone

from roadmap_app.models import RoadmapPlan
from roadmap_app.services import (
    _build_chain,
    _category_owned,
    _context_product_ids,
    _owned_fragrance_slots,
    _post_ctx_types_by_category,
    _purchased_fragrance_slots,
    _unique,
    get_active_plan,
    refresh_roadmap,
)
from transactions.models import Transaction


CATEGORY_CHOICES = ["all", "skincare", "haircare", "makeup", "fragrance"]


class _DryRunRollback(Exception):
    pass


def _safe_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _build_post_ctx(txn: Transaction) -> dict[str, Any] | None:
    categories: list[str] = []
    product_types: list[str] = []
    product_ids: list[int] = []
    seen_ids: set[int] = set()
    items = list(getattr(txn, "_prefetched_objects_cache", {}).get("items", []) or txn.items.select_related("product").all())
    for item in items:
        product = getattr(item, "product", None)
        if product is None:
            continue
        try:
            pid = int(product.id)
        except Exception:
            continue
        category = str(getattr(product, "category", "") or "").strip().lower()
        product_type = str(getattr(product, "product_type", "") or "").strip().lower()
        if category and category not in categories:
            categories.append(category)
        if product_type and product_type not in product_types:
            product_types.append(product_type)
        if pid > 0 and pid not in seen_ids:
            seen_ids.add(pid)
            product_ids.append(pid)

    if not categories and not product_ids:
        return None

    return {
        "categories": categories,
        "product_types": product_types,
        "product_ids": product_ids,
    }


def _plan_snapshot(plan: RoadmapPlan) -> dict[str, Any]:
    meta = _safe_dict(plan.meta)
    ml = _safe_dict(meta.get("ml"))
    context = _safe_dict(meta.get("context"))
    return {
        "plan_id": int(plan.id),
        "category": str(plan.category or "").strip().lower(),
        "decision": str(ml.get("decision") or "").strip().lower() or "disabled",
        "model_slot": str(ml.get("model_slot") or "active").strip().lower() or "active",
        "planned_target_product_type": str(ml.get("planned_target_product_type") or "").strip().lower(),
        "planned_target_step_index": int(ml.get("planned_target_step_index") or 0),
        "rollout_reason": str(ml.get("rollout_reason") or "").strip().lower(),
        "refresh_caller": str(context.get("refresh_caller") or "").strip().lower(),
        "model_version": str(ml.get("model_version") or "").strip(),
    }


def _missing_plan_snapshot(category: str) -> dict[str, Any]:
    return {
        "plan_id": 0,
        "category": str(category or "").strip().lower(),
        "decision": "missing_plan",
        "model_slot": "__missing_plan__",
        "planned_target_product_type": "",
        "planned_target_step_index": 0,
        "rollout_reason": "",
        "refresh_caller": "",
        "model_version": "",
    }


class Command(BaseCommand):
    help = "Replay recent purchase-context roadmap refreshes to materialize current rollout logic on existing plans."

    def add_arguments(self, parser):
        parser.add_argument("--days", type=int, default=5)
        parser.add_argument("--category", type=str, default="all", choices=CATEGORY_CHOICES)
        parser.add_argument(
            "--include-ga",
            action="store_true",
            default=False,
            help='Include users with username starting with "ga_".',
        )
        parser.add_argument(
            "--include-inactive",
            action="store_true",
            default=False,
            help="Include inactive plans. Default is active plans only.",
        )
        parser.add_argument("--limit", type=int, default=None)
        parser.add_argument(
            "--all-transactions",
            action="store_true",
            default=False,
            help="Replay every matching transaction. Default is latest matching transaction per user.",
        )
        parser.add_argument(
            "--analysis-only",
            action="store_true",
            default=False,
            help="Read-only fast path: analyze current runtime chain/slot via _build_chain without refreshing plans.",
        )
        parser.add_argument(
            "--write",
            action="store_true",
            default=False,
            help="Apply refreshes to DB. Default is dry-run with transaction rollback.",
        )

    def handle(self, *args, **options):
        days = int(options["days"] or 0)
        if days <= 0:
            raise CommandError("--days must be > 0")

        category = str(options["category"] or "all").strip().lower()
        include_ga = bool(options["include_ga"])
        include_inactive = bool(options["include_inactive"])
        replay_all_transactions = bool(options["all_transactions"])
        analysis_only = bool(options["analysis_only"])
        should_write = bool(options["write"])
        if analysis_only and should_write:
            raise CommandError("--analysis-only cannot be combined with --write")

        limit = options.get("limit")
        if limit is not None:
            limit = int(limit)
            if limit <= 0:
                raise CommandError("--limit must be > 0")

        now_utc = timezone.now()
        since = now_utc - timedelta(days=days)

        qs = (
            Transaction.objects.select_related("user")
            .prefetch_related("items__product")
            .filter(created_at__gte=since, created_at__lte=now_utc)
            .order_by("-created_at", "-id")
        )
        if category != "all":
            qs = qs.filter(items__product__category=category)
        if not include_ga:
            qs = qs.exclude(user__username__startswith="ga_")
        qs = qs.distinct()

        txns = list(qs)

        transactions_scanned = 0
        transactions_selected = 0
        replayed_plans = 0
        changed = 0
        updated = 0
        skipped_counts: Counter[str] = Counter()
        transition_counts: Counter[str] = Counter()
        slot_counts: Counter[str] = Counter()
        target_counts: Counter[str] = Counter()
        preview_rows: list[dict[str, Any]] = []
        errors: list[str] = []
        seen_users: set[int] = set()

        for txn in txns:
            transactions_scanned += 1
            if not replay_all_transactions and int(txn.user_id) in seen_users:
                skipped_counts["older_transaction_for_user"] += 1
                continue

            post_ctx = _build_post_ctx(txn)
            if not post_ctx:
                skipped_counts["missing_post_ctx"] += 1
                continue

            categories_to_refresh = list(post_ctx.get("categories") or [])
            if category != "all":
                categories_to_refresh = [category] if category in categories_to_refresh else []
            if not categories_to_refresh:
                skipped_counts["transaction_category_not_matched"] += 1
                continue

            seen_users.add(int(txn.user_id))
            transactions_selected += 1
            if limit is not None and transactions_selected > limit:
                break

            for category_key in categories_to_refresh:
                if not include_inactive:
                    active_plan = get_active_plan(txn.user, category=category_key)
                    if active_plan is not None and not bool(active_plan.is_active):
                        skipped_counts["inactive_plan_filtered"] += 1
                        continue

                replayed_plans += 1
                active_plan = get_active_plan(txn.user, category=category_key)
                before = _plan_snapshot(active_plan) if active_plan else _missing_plan_snapshot(category_key)
                after: dict[str, Any] | None = None

                try:
                    if analysis_only:
                        after = self._analysis_snapshot(
                            user=txn.user,
                            category=category_key,
                            post_ctx=post_ctx,
                            active_plan=active_plan,
                        )
                    elif should_write:
                        refreshed = refresh_roadmap(txn.user, category=category_key, post_ctx=post_ctx)
                        after = _plan_snapshot(refreshed)
                    else:
                        with transaction.atomic():
                            refreshed = refresh_roadmap(txn.user, category=category_key, post_ctx=post_ctx)
                            after = _plan_snapshot(refreshed)
                            raise _DryRunRollback()
                except _DryRunRollback:
                    pass
                except Exception as exc:
                    skipped_counts["refresh_error"] += 1
                    if len(errors) < 10:
                        errors.append(f"txn={txn.id} user={txn.user_id} category={category_key} error={exc}")
                    continue

                if after is None:
                    skipped_counts["missing_after_snapshot"] += 1
                    continue

                slot_counts[str(after["model_slot"] or "active")] += 1
                target_counts[str(after["planned_target_product_type"] or "__none__")] += 1
                transition_key = f"{before['model_slot']} -> {after['model_slot']}"
                transition_counts[transition_key] += 1

                if before != after:
                    changed += 1
                    if should_write:
                        updated += 1
                else:
                    skipped_counts["unchanged_after_replay"] += 1

                if len(preview_rows) < 12:
                    preview_rows.append(
                        {
                            "txn_id": int(txn.id),
                            "plan_id": int(after["plan_id"] or 0),
                            "user_id": int(txn.user_id),
                            "category": str(category_key),
                            "before_slot": before["model_slot"],
                            "after_slot": after["model_slot"],
                            "before_target": before["planned_target_product_type"] or "__none__",
                            "after_target": after["planned_target_product_type"] or "__none__",
                            "decision": after["decision"],
                            "rollout_reason": after["rollout_reason"] or "__none__",
                        }
                    )

        self.stdout.write("# Replay Recent Purchase-Context Roadmaps")
        self.stdout.write("")
        if analysis_only:
            mode = "analysis-only"
        else:
            mode = "write" if should_write else "dry-run"
        self.stdout.write(f"- mode: `{mode}`")
        self.stdout.write(f"- analysis window: last `{days}` days")
        self.stdout.write(f"- category: `{category}`")
        self.stdout.write(f"- include ga_* users: `{include_ga}`")
        self.stdout.write(f"- active only: `{not include_inactive}`")
        self.stdout.write(f"- latest transaction per user: `{not replay_all_transactions}`")
        self.stdout.write(f"- transactions scanned: `{transactions_scanned}`")
        self.stdout.write(f"- transactions selected: `{transactions_selected}`")
        self.stdout.write(f"- plans replayed: `{replayed_plans}`")
        self.stdout.write(f"- changed after replay: `{changed}`")
        self.stdout.write(f"- plans updated: `{updated}`")
        self.stdout.write("")
        self.stdout.write("## Result slots")
        if slot_counts:
            for key, value in sorted(slot_counts.items(), key=lambda kv: (-kv[1], kv[0])):
                self.stdout.write(f"- {key}: `{int(value)}`")
        else:
            self.stdout.write("- none")
        self.stdout.write("")
        self.stdout.write("## Planned targets")
        if target_counts:
            for key, value in sorted(target_counts.items(), key=lambda kv: (-kv[1], kv[0])):
                self.stdout.write(f"- {key}: `{int(value)}`")
        else:
            self.stdout.write("- none")
        self.stdout.write("")
        self.stdout.write("## Slot transitions")
        if transition_counts:
            for key, value in sorted(transition_counts.items(), key=lambda kv: (-kv[1], kv[0])):
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
                self.stdout.write(
                    f"- txn `{row['txn_id']}` plan `{row['plan_id']}` user=`{row['user_id']}` [{row['category']}] "
                    f"slot `{row['before_slot']}` -> `{row['after_slot']}` "
                    f"target `{row['before_target']}` -> `{row['after_target']}` "
                    f"decision=`{row['decision']}` rollout_reason=`{row['rollout_reason']}`"
                )
        else:
            self.stdout.write("- none")
        if errors:
            self.stdout.write("")
            self.stdout.write("## Errors")
            for item in errors:
                self.stdout.write(f"- {item}")

    def _analysis_snapshot(
        self,
        *,
        user,
        category: str,
        post_ctx: dict[str, Any],
        active_plan: RoadmapPlan | None,
    ) -> dict[str, Any]:
        _, _, owned_types_ordered, _ = _category_owned(user, category)
        purchased_by_category = _post_ctx_types_by_category(post_ctx)
        purchased_types = _unique([str(x) for x in purchased_by_category.get(category, [])])
        if category == "fragrance":
            owned_types_ordered = _unique(_owned_fragrance_slots(user))
            purchased_types = _unique(_purchased_fragrance_slots(post_ctx))
        context_product_ids = _context_product_ids(user, post_ctx, limit=50)
        _, _, _, ml_runtime = _build_chain(
            user=user,
            category=category,
            purchased_types=purchased_types,
            owned_types_ordered=owned_types_ordered,
            context_product_ids=context_product_ids,
            refresh_caller="update_roadmap_from_purchase",
        )
        return {
            "plan_id": int(getattr(active_plan, "id", 0) or 0),
            "category": str(category or "").strip().lower(),
            "decision": str(ml_runtime.get("decision") or "").strip().lower() or "disabled",
            "model_slot": str(ml_runtime.get("model_slot") or "active").strip().lower() or "active",
            "planned_target_product_type": str(
                ml_runtime.get("planned_target_product_type") or ""
            ).strip().lower(),
            "planned_target_step_index": int(ml_runtime.get("planned_target_step_index") or 0),
            "rollout_reason": str(ml_runtime.get("rollout_reason") or "").strip().lower(),
            "refresh_caller": "update_roadmap_from_purchase",
            "model_version": str(ml_runtime.get("model_version") or "").strip(),
        }
