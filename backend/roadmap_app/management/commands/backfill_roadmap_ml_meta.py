from __future__ import annotations

from collections import Counter
from datetime import timedelta
from typing import Any

from django.core.management.base import BaseCommand, CommandError
from django.utils import timezone

from roadmap_app.models import RoadmapPlan
from roadmap_app.services import _normalize_plan_meta


CATEGORY_CHOICES = ["all", "skincare", "haircare", "makeup", "fragrance"]
VALID_DECISIONS = {"model_used", "fallback", "disabled"}


def _safe_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _decision_from_meta(meta: dict[str, Any] | None) -> str:
    ml = _safe_dict(_safe_dict(meta).get("ml"))
    decision = str(ml.get("decision") or "").strip().lower()
    if decision in VALID_DECISIONS:
        return decision
    return "missing_ml_meta"


class Command(BaseCommand):
    help = "Normalize historical RoadmapPlan.meta.ml payloads for recent plans."

    def add_arguments(self, parser):
        parser.add_argument("--days", type=int, default=60)
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
            "--write",
            action="store_true",
            default=False,
            help="Apply normalized meta payloads to DB. Default is dry-run.",
        )

    def handle(self, *args, **options):
        days = int(options["days"] or 60)
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

        rows = list(qs.values("id", "category", "meta"))
        before_counts: Counter[str] = Counter()
        projected_counts: Counter[str] = Counter()
        pending_updates: list[tuple[int, dict[str, Any], str, str, str]] = []

        for row in rows:
            plan_id = int(row["id"])
            category_key = str(row.get("category") or "").strip() or "__unknown__"
            original_meta = _safe_dict(row.get("meta"))
            normalized_meta = _normalize_plan_meta(original_meta)
            before_decision = _decision_from_meta(original_meta)
            after_decision = _decision_from_meta(normalized_meta)
            before_counts[before_decision] += 1
            projected_counts[after_decision] += 1
            if normalized_meta != original_meta:
                pending_updates.append((plan_id, normalized_meta, category_key, before_decision, after_decision))

        updated_count = 0
        if should_write:
            for plan_id, normalized_meta, _, _, _ in pending_updates:
                updated_count += RoadmapPlan.objects.filter(id=plan_id).update(meta=normalized_meta)

        examples = pending_updates[:10]
        self.stdout.write("# Roadmap ML Meta Backfill")
        self.stdout.write("")
        self.stdout.write(f"- mode: `{'write' if should_write else 'dry-run'}`")
        self.stdout.write(f"- analysis window: last `{days}` days")
        self.stdout.write(f"- category: `{category}`")
        self.stdout.write(f"- include ga_* users: `{include_ga}`")
        self.stdout.write(f"- active only: `{active_only}`")
        self.stdout.write(f"- plans scanned: `{len(rows)}`")
        self.stdout.write(f"- plans needing normalization: `{len(pending_updates)}`")
        self.stdout.write(f"- plans updated: `{updated_count}`")
        self.stdout.write("")
        self.stdout.write("## Decision counts before")
        for key in ["model_used", "fallback", "disabled", "missing_ml_meta"]:
            self.stdout.write(f"- {key}: `{int(before_counts.get(key, 0))}`")
        self.stdout.write("")
        self.stdout.write("## Decision counts after")
        for key in ["model_used", "fallback", "disabled", "missing_ml_meta"]:
            self.stdout.write(f"- {key}: `{int(projected_counts.get(key, 0))}`")
        self.stdout.write("")
        self.stdout.write("## Example changes")
        if not examples:
            self.stdout.write("- none")
            return
        for plan_id, _, category_key, before_decision, after_decision in examples:
            self.stdout.write(
                f"- plan_id=`{plan_id}` category=`{category_key}` decision: `{before_decision}` -> `{after_decision}`"
            )
