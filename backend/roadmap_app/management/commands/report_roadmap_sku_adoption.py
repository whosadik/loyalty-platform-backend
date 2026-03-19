from __future__ import annotations

import json
from collections import Counter, defaultdict
from datetime import timedelta
from pathlib import Path
from typing import Any

from django.core.management.base import BaseCommand, CommandError
from django.utils import timezone

from catalog.models import Product
from roadmap_app.models import RoadmapEvent


CATEGORY_CHOICES = ["skincare", "haircare", "makeup", "fragrance", "mixed", "all"]
FORMAT_CHOICES = ["md", "json", "both"]
COHORT_MODE_CHOICES = ["fresh", "all"]


def _safe_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _to_int(value: Any) -> int | None:
    try:
        if value is None:
            return None
        return int(value)
    except Exception:
        return None


def _event_key(created_at, event_id: int) -> tuple[Any, int]:
    return created_at, int(event_id or 0)


def _rate(numerator: int, denominator: int) -> float | None:
    if denominator <= 0:
        return None
    return round(float(numerator) / float(denominator), 6)


def _decision_from_ml(ml: dict[str, Any]) -> str:
    decision = str(ml.get("decision") or "").strip().lower()
    return decision or "missing_ml_meta"


def _step_bucket() -> dict[str, int]:
    return {
        "recommended_steps": 0,
        "clicked_after_generated": 0,
        "any_completion": 0,
        "exact_recommended_product_checkout": 0,
        "semantic_alternative_checkout": 0,
        "product_type_checkout": 0,
    }


def _finalize_bucket(bucket: dict[str, int]) -> dict[str, Any]:
    recommended_steps = int(bucket.get("recommended_steps", 0))
    clicked = int(bucket.get("clicked_after_generated", 0))
    any_completion = int(bucket.get("any_completion", 0))
    exact = int(bucket.get("exact_recommended_product_checkout", 0))
    semantic = int(bucket.get("semantic_alternative_checkout", 0))
    product_type = int(bucket.get("product_type_checkout", 0))
    return {
        "recommended_steps": recommended_steps,
        "clicked_after_generated": clicked,
        "any_completion": any_completion,
        "exact_recommended_product_checkout": exact,
        "semantic_alternative_checkout": semantic,
        "product_type_checkout": product_type,
        "click_rate": _rate(clicked, recommended_steps),
        "step_completion_rate": _rate(any_completion, recommended_steps),
        "exact_adoption_rate": _rate(exact, recommended_steps),
        "semantic_alternative_rate": _rate(semantic, recommended_steps),
        "product_type_match_rate": _rate(product_type, recommended_steps),
    }


def _markdown_table(rows: list[dict[str, Any]], columns: list[tuple[str, str]]) -> list[str]:
    lines = [
        "| " + " | ".join(label for _, label in columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in rows:
        values = []
        for key, _label in columns:
            value = row.get(key)
            if isinstance(value, float):
                values.append(f"{value:.6f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return lines


def _build_markdown(payload: dict[str, Any]) -> str:
    params = _safe_dict(payload.get("params"))
    overall = _safe_dict(payload.get("overall"))
    lines: list[str] = [
        "# Roadmap SKU Adoption",
        "",
        f"- Window days: `{params.get('days')}`",
        f"- Category: `{params.get('category')}`",
        f"- include_ga: `{params.get('include_ga')}`",
        f"- cohort_mode: `{params.get('cohort_mode')}`",
        "",
        "## Overall",
        "",
    ]
    overall_rows = [
        {"metric": key, "value": value}
        for key, value in overall.items()
    ]
    lines.extend(_markdown_table(overall_rows, [("metric", "metric"), ("value", "value")]))

    by_type = list(_safe_dict(payload.get("breakdowns")).get("by_step_product_type_rows", [])[:15])
    if by_type:
        lines.extend(["", "## By Step Product Type", ""])
        lines.extend(
            _markdown_table(
                by_type,
                [
                    ("step_product_type", "step_product_type"),
                    ("recommended_steps", "recommended_steps"),
                    ("exact_adoption_rate", "exact_adoption_rate"),
                    ("semantic_alternative_rate", "semantic_alternative_rate"),
                    ("step_completion_rate", "step_completion_rate"),
                ],
            )
        )

    by_product = list(_safe_dict(payload.get("breakdowns")).get("by_recommended_product_rows", [])[:20])
    if by_product:
        lines.extend(["", "## Top Recommended Products", ""])
        lines.extend(
            _markdown_table(
                by_product,
                [
                    ("recommended_product_id", "recommended_product_id"),
                    ("product_name", "product_name"),
                    ("step_product_type", "step_product_type"),
                    ("recommended_steps", "recommended_steps"),
                    ("exact_adoption_rate", "exact_adoption_rate"),
                    ("semantic_alternative_rate", "semantic_alternative_rate"),
                ],
            )
        )
    return "\n".join(lines)


class Command(BaseCommand):
    help = "Read-only report for roadmap recommended-product adoption on next-step generated instances."

    def add_arguments(self, parser):
        parser.add_argument("--days", type=int, default=30)
        parser.add_argument("--category", type=str, default="all", choices=CATEGORY_CHOICES)
        parser.add_argument("--include-ga", action="store_true", default=False)
        parser.add_argument("--format", type=str, default="both", choices=FORMAT_CHOICES)
        parser.add_argument("--cohort-mode", type=str, default="fresh", choices=COHORT_MODE_CHOICES)
        parser.add_argument("--out", type=str, default=None)

    def handle(self, *args, **options):
        days = int(options["days"] or 30)
        if days <= 0:
            raise CommandError("--days must be > 0")
        category = str(options.get("category") or "all").strip().lower()
        include_ga = bool(options.get("include_ga"))
        out_format = str(options.get("format") or "both").strip().lower()
        cohort_mode = str(options.get("cohort_mode") or "fresh").strip().lower()
        out_raw = options.get("out")

        now_utc = timezone.now()
        since = now_utc - timedelta(days=days)

        event_qs = RoadmapEvent.objects.filter(
            created_at__gte=since,
            created_at__lte=now_utc,
            event_type__in=[
                RoadmapEvent.Type.PLAN_REFRESHED,
                RoadmapEvent.Type.STEP_GENERATED,
                RoadmapEvent.Type.STEP_CLICKED,
                RoadmapEvent.Type.STEP_COMPLETED,
            ],
        )
        if not include_ga:
            event_qs = event_qs.exclude(user__username__startswith="ga_")

        rows = list(
            event_qs.order_by("user_id", "step_id", "created_at", "id").values(
                "id",
                "user_id",
                "plan_id",
                "step_id",
                "event_type",
                "created_at",
                "context",
            )
        )

        plan_refreshes: dict[int, list[dict[str, Any]]] = defaultdict(list)
        generated_by_key: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
        generated_by_plan_step: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
        clicks_by_key: dict[tuple[int, int], list[tuple[Any, int]]] = defaultdict(list)
        completions_by_key: dict[tuple[int, int], list[tuple[Any, int, str, int | None, str | None]]] = defaultdict(list)

        for row in rows:
            ctx = _safe_dict(row.get("context"))
            event_type = str(row.get("event_type") or "")
            user_id = _to_int(row.get("user_id"))
            event_id = _to_int(row.get("id")) or 0
            created_at = row.get("created_at")
            if user_id is None or created_at is None:
                continue

            if event_type == RoadmapEvent.Type.PLAN_REFRESHED:
                plan_id = _to_int(row.get("plan_id")) or _to_int(ctx.get("plan_id"))
                if plan_id is None:
                    continue
                event_category = str(ctx.get("category") or "").strip().lower() or "__unknown__"
                if category != "all" and event_category != category:
                    continue
                plan_refreshes[int(plan_id)].append(
                    {
                        "event_id": int(event_id),
                        "plan_id": int(plan_id),
                        "generated_at": created_at,
                        "category": event_category,
                        "next_step_id": _to_int(ctx.get("next_step_id")),
                        "ml_decision": _decision_from_ml(_safe_dict(ctx.get("ml"))),
                        "plan_source": str(ctx.get("source") or "roadmap_v1"),
                    }
                )
                continue

            step_id = _to_int(row.get("step_id")) or _to_int(ctx.get("step_id"))
            if step_id is None:
                continue
            key = (int(user_id), int(step_id))

            if event_type == RoadmapEvent.Type.STEP_GENERATED:
                event_category = str(ctx.get("category") or "").strip().lower() or "__unknown__"
                if category != "all" and event_category != category:
                    continue
                plan_id = _to_int(row.get("plan_id")) or _to_int(ctx.get("plan_id"))
                if plan_id is None:
                    continue
                generated = {
                    "generated_event_id": int(event_id),
                    "user_id": int(user_id),
                    "plan_id": int(plan_id),
                    "step_id": int(step_id),
                    "generated_at": created_at,
                    "category": event_category,
                    "step_product_type": str(ctx.get("product_type") or "").strip().lower() or "__unknown__",
                    "step_index": _to_int(ctx.get("step_index")),
                    "recommended_product_id": _to_int(ctx.get("recommended_product_id")),
                    "has_recommendation": bool(ctx.get("has_recommendation")) or _to_int(ctx.get("recommended_product_id")) is not None,
                    "ml_decision": _decision_from_ml(_safe_dict(ctx.get("ml"))),
                    "plan_source": str(ctx.get("plan_source") or "roadmap_v1"),
                }
                generated_by_key[key].append(generated)
                generated_by_plan_step[(int(plan_id), int(step_id))].append(generated)
            elif event_type == RoadmapEvent.Type.STEP_CLICKED:
                clicks_by_key[key].append((created_at, int(event_id)))
            elif event_type == RoadmapEvent.Type.STEP_COMPLETED:
                match_meta = _safe_dict(ctx.get("match_meta"))
                completions_by_key[key].append(
                    (
                        created_at,
                        int(event_id),
                        str(ctx.get("matched_by") or "").strip().lower(),
                        _to_int(match_meta.get("purchased_product_id")) or _to_int(ctx.get("purchased_product_id")),
                        str(match_meta.get("purchased_product_type") or ctx.get("product_type") or "").strip().lower() or None,
                    )
                )

        next_step_generated_event_ids: set[int] = set()
        for items in generated_by_plan_step.values():
            items.sort(key=lambda item: _event_key(item.get("generated_at"), int(item.get("generated_event_id") or 0)))
        for refreshes in plan_refreshes.values():
            refreshes.sort(key=lambda item: _event_key(item.get("generated_at"), int(item.get("event_id") or 0)))
            for idx, refresh in enumerate(refreshes):
                next_step_id = _to_int(refresh.get("next_step_id"))
                if next_step_id is None:
                    continue
                refresh_key = _event_key(refresh.get("generated_at"), int(refresh.get("event_id") or 0))
                next_refresh_key = (
                    _event_key(refreshes[idx + 1].get("generated_at"), int(refreshes[idx + 1].get("event_id") or 0))
                    if idx + 1 < len(refreshes)
                    else None
                )
                for generated in generated_by_plan_step.get((int(refresh["plan_id"]), int(next_step_id)), []):
                    generated_key = _event_key(generated.get("generated_at"), int(generated.get("generated_event_id") or 0))
                    if generated_key < refresh_key:
                        continue
                    if next_refresh_key is not None and generated_key >= next_refresh_key:
                        break
                    next_step_generated_event_ids.add(int(generated.get("generated_event_id") or 0))
                    break

        instances: list[dict[str, Any]] = []
        for key, items in generated_by_key.items():
            items.sort(key=lambda item: _event_key(item["generated_at"], int(item.get("generated_event_id") or 0)))
            for idx, item in enumerate(items):
                generated_event_id = int(item.get("generated_event_id") or 0)
                if generated_event_id not in next_step_generated_event_ids:
                    continue
                if not bool(item.get("has_recommendation")) or _to_int(item.get("recommended_product_id")) is None:
                    continue
                if cohort_mode == "fresh" and str(item.get("ml_decision") or "") == "missing_ml_meta":
                    continue

                start_key = _event_key(item["generated_at"], generated_event_id)
                next_generated_key = (
                    _event_key(items[idx + 1]["generated_at"], int(items[idx + 1].get("generated_event_id") or 0))
                    if idx + 1 < len(items)
                    else None
                )

                clicks = [
                    click
                    for click in clicks_by_key.get(key, [])
                    if _event_key(click[0], click[1]) >= start_key
                    and (next_generated_key is None or _event_key(click[0], click[1]) < next_generated_key)
                ]
                completions = [
                    completion
                    for completion in completions_by_key.get(key, [])
                    if _event_key(completion[0], completion[1]) >= start_key
                    and (next_generated_key is None or _event_key(completion[0], completion[1]) < next_generated_key)
                ]

                recommended_product_id = int(item["recommended_product_id"])
                semantic_alternatives = [
                    completion
                    for completion in completions
                    if str(completion[2] or "") == "semantic_content_match"
                ]
                exact_match = any(str(completion[2] or "") == "recommended_product_id" for completion in completions)
                product_type_match = any(str(completion[2] or "") == "product_type" for completion in completions)

                first_semantic = semantic_alternatives[0] if semantic_alternatives else None
                purchased_alt_product_id = int(first_semantic[3]) if first_semantic and first_semantic[3] else None
                purchased_alt_product_type = str(first_semantic[4] or "") if first_semantic and first_semantic[4] else None

                instances.append(
                    {
                        **item,
                        "clicked_after_generated": bool(clicks),
                        "any_completion": bool(completions),
                        "exact_recommended_product_checkout": bool(exact_match),
                        "semantic_alternative_checkout": bool(semantic_alternatives),
                        "product_type_checkout": bool(product_type_match),
                        "purchased_alt_product_id": purchased_alt_product_id,
                        "purchased_alt_product_type": purchased_alt_product_type,
                    }
                )

        recommended_product_ids = sorted(
            {
                int(item["recommended_product_id"])
                for item in instances
                if _to_int(item.get("recommended_product_id")) is not None
            }
        )
        alt_product_ids = sorted(
            {
                int(item["purchased_alt_product_id"])
                for item in instances
                if _to_int(item.get("purchased_alt_product_id")) is not None
            }
        )
        product_rows = {
            int(row["id"]): row
            for row in Product.objects.filter(id__in=(recommended_product_ids + alt_product_ids)).values(
                "id", "name", "brand", "category", "product_type"
            )
        }

        overall_bucket = _step_bucket()
        by_category: dict[str, dict[str, int]] = defaultdict(_step_bucket)
        by_step_product_type: dict[str, dict[str, int]] = defaultdict(_step_bucket)
        by_ml_decision: dict[str, dict[str, int]] = defaultdict(_step_bucket)
        by_recommended_product: dict[int, dict[str, Any]] = {}
        alt_counts_by_recommended_product: dict[int, Counter[int]] = defaultdict(Counter)

        for item in instances:
            for bucket in (
                overall_bucket,
                by_category[str(item.get("category") or "__unknown__")],
                by_step_product_type[str(item.get("step_product_type") or "__unknown__")],
                by_ml_decision[str(item.get("ml_decision") or "missing_ml_meta")],
            ):
                bucket["recommended_steps"] += 1
                bucket["clicked_after_generated"] += int(bool(item.get("clicked_after_generated")))
                bucket["any_completion"] += int(bool(item.get("any_completion")))
                bucket["exact_recommended_product_checkout"] += int(bool(item.get("exact_recommended_product_checkout")))
                bucket["semantic_alternative_checkout"] += int(bool(item.get("semantic_alternative_checkout")))
                bucket["product_type_checkout"] += int(bool(item.get("product_type_checkout")))

            recommended_product_id = int(item["recommended_product_id"])
            product_bucket = by_recommended_product.setdefault(
                recommended_product_id,
                {
                    "recommended_product_id": recommended_product_id,
                    "product_name": str(product_rows.get(recommended_product_id, {}).get("name") or ""),
                    "brand": str(product_rows.get(recommended_product_id, {}).get("brand") or ""),
                    "product_type": str(product_rows.get(recommended_product_id, {}).get("product_type") or ""),
                    "step_product_type": str(item.get("step_product_type") or "__unknown__"),
                    **_step_bucket(),
                },
            )
            product_bucket["recommended_steps"] += 1
            product_bucket["clicked_after_generated"] += int(bool(item.get("clicked_after_generated")))
            product_bucket["any_completion"] += int(bool(item.get("any_completion")))
            product_bucket["exact_recommended_product_checkout"] += int(bool(item.get("exact_recommended_product_checkout")))
            product_bucket["semantic_alternative_checkout"] += int(bool(item.get("semantic_alternative_checkout")))
            product_bucket["product_type_checkout"] += int(bool(item.get("product_type_checkout")))

            alt_pid = _to_int(item.get("purchased_alt_product_id"))
            if alt_pid is not None and bool(item.get("semantic_alternative_checkout")):
                alt_counts_by_recommended_product[recommended_product_id][int(alt_pid)] += 1

        by_recommended_product_rows: list[dict[str, Any]] = []
        for recommended_product_id, bucket in by_recommended_product.items():
            finalized = _finalize_bucket(bucket)
            alt_rows = []
            for alt_pid, count in alt_counts_by_recommended_product.get(recommended_product_id, Counter()).most_common(5):
                alt_product = product_rows.get(int(alt_pid), {})
                alt_rows.append(
                    {
                        "purchased_product_id": int(alt_pid),
                        "name": str(alt_product.get("name") or ""),
                        "brand": str(alt_product.get("brand") or ""),
                        "product_type": str(alt_product.get("product_type") or ""),
                        "count": int(count),
                    }
                )
            by_recommended_product_rows.append(
                {
                    "recommended_product_id": int(recommended_product_id),
                    "product_name": str(bucket.get("product_name") or ""),
                    "brand": str(bucket.get("brand") or ""),
                    "product_type": str(bucket.get("product_type") or ""),
                    "step_product_type": str(bucket.get("step_product_type") or ""),
                    **finalized,
                    "top_semantic_alternatives": alt_rows,
                }
            )

        by_recommended_product_rows.sort(
            key=lambda row: (
                -int(row.get("recommended_steps") or 0),
                -float(row.get("exact_adoption_rate") or 0.0),
                int(row.get("recommended_product_id") or 0),
            )
        )

        payload: dict[str, Any] = {
            "generated_at_utc": now_utc.isoformat(),
            "window_start_utc": since.isoformat(),
            "window_end_utc": now_utc.isoformat(),
            "params": {
                "days": days,
                "category": category,
                "include_ga": include_ga,
                "format": out_format,
                "cohort_mode": cohort_mode,
            },
            "overall": _finalize_bucket(overall_bucket),
            "breakdowns": {
                "by_category": {
                    key: _finalize_bucket(bucket)
                    for key, bucket in sorted(by_category.items(), key=lambda kv: kv[0])
                },
                "by_step_product_type": {
                    key: _finalize_bucket(bucket)
                    for key, bucket in sorted(by_step_product_type.items(), key=lambda kv: kv[0])
                },
                "by_step_product_type_rows": [
                    {"step_product_type": key, **_finalize_bucket(bucket)}
                    for key, bucket in sorted(
                        by_step_product_type.items(),
                        key=lambda kv: (-int(kv[1].get("recommended_steps", 0)), kv[0]),
                    )
                ],
                "by_ml_decision": {
                    key: _finalize_bucket(bucket)
                    for key, bucket in sorted(by_ml_decision.items(), key=lambda kv: kv[0])
                },
                "by_recommended_product_rows": by_recommended_product_rows,
            },
            "notes": [
                "Read-only report: no DB writes.",
                "Uses PLAN_REFRESHED.next_step_id + STEP_GENERATED to isolate next-step-only instances.",
                "Exact adoption is STEP_COMPLETED.matched_by=recommended_product_id.",
                "Semantic alternative is STEP_COMPLETED.matched_by=semantic_content_match.",
                "Product-type checkout is STEP_COMPLETED.matched_by=product_type.",
            ],
        }

        markdown = _build_markdown(payload)
        json_text = json.dumps(payload, ensure_ascii=False, indent=2)

        out_stem = Path(out_raw) if out_raw else Path("reports") / f"roadmap_sku_adoption_{days}d_{category}"
        wrote_paths: list[Path] = []

        if out_format in {"md", "both"}:
            if out_raw:
                md_path = out_stem.with_suffix(".md")
                md_path.parent.mkdir(parents=True, exist_ok=True)
                md_path.write_text(markdown, encoding="utf-8")
                wrote_paths.append(md_path)
            else:
                self.stdout.write(markdown)

        if out_format in {"json", "both"}:
            if out_raw:
                json_path = out_stem.with_suffix(".json")
                json_path.parent.mkdir(parents=True, exist_ok=True)
                json_path.write_text(json_text, encoding="utf-8")
                wrote_paths.append(json_path)
            else:
                if out_format == "json":
                    self.stdout.write(json_text)
                else:
                    self.stdout.write("\n---\n")
                    self.stdout.write(json_text)

        for path in wrote_paths:
            self.stdout.write(f"[report_roadmap_sku_adoption] wrote: {path}")
