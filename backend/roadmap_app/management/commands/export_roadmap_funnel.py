from __future__ import annotations

import csv
import json
from datetime import timedelta, timezone as dt_timezone
from pathlib import Path
from typing import Any

from django.core.management.base import BaseCommand, CommandError
from django.db.models import Q, Sum
from django.utils import timezone

from catalog.models import Product
from roadmap_app.fragrance_slots import SLOTS, slot_of_fragrance
from roadmap_app.models import RoadmapEvent, RoadmapStep
from transactions.models import Transaction, TransactionItem


CATEGORY_CHOICES = ["skincare", "haircare", "makeup", "fragrance", "mixed", "all"]
FORMAT_CHOICES = ["csv", "parquet"]


def _to_int(value: Any) -> int | None:
    try:
        if value is None:
            return None
        return int(value)
    except Exception:
        return None


def _safe_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _safe_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _dt_to_utc_iso(value) -> str | None:
    if not value:
        return None
    return value.astimezone(dt_timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _why_text(why: Any) -> str:
    if isinstance(why, list):
        text = " | ".join(str(x) for x in why[:3])
    elif isinstance(why, dict):
        text = " | ".join(f"{k}:{v}" for k, v in list(why.items())[:3])
    else:
        text = str(why or "")
    text = text.strip()
    return text[:200]


def _latency_sec(start_dt, end_dt) -> int | None:
    if not start_dt or not end_dt:
        return None
    delta = (end_dt - start_dt).total_seconds()
    return int(delta) if delta >= 0 else None


class Command(BaseCommand):
    help = "Read-only export of Roadmap funnel instances for analytics/ML."

    def add_arguments(self, parser):
        parser.add_argument("--days", type=int, default=30)
        parser.add_argument("--category", type=str, default="all", choices=CATEGORY_CHOICES)
        parser.add_argument(
            "--include-ga",
            action="store_true",
            default=False,
            help='Include users with username starting with "ga_".',
        )
        parser.add_argument("--out", type=str, default=None)
        parser.add_argument("--format", type=str, default="csv", choices=FORMAT_CHOICES)
        parser.add_argument("--k", type=int, default=10)
        parser.add_argument("--tz", type=str, default="UTC", choices=["UTC"])

    def handle(self, *args, **options):
        days = int(options["days"] or 30)
        if days <= 0:
            raise CommandError("--days must be > 0")
        category = str(options["category"] or "all").strip().lower()
        include_ga = bool(options["include_ga"])
        out_path_raw = options.get("out")
        fmt = str(options["format"] or "csv").strip().lower()
        k = int(options["k"] or 10)
        if k <= 0:
            raise CommandError("--k must be > 0")

        now_utc = timezone.now().astimezone(dt_timezone.utc)
        since = now_utc - timedelta(days=days)
        since90 = now_utc - timedelta(days=90)

        default_ext = "parquet" if fmt == "parquet" else "csv"
        default_out = Path("reports") / f"roadmap_funnel_{days}d_{category}.{default_ext}"
        out_path = Path(out_path_raw) if out_path_raw else default_out
        out_path.parent.mkdir(parents=True, exist_ok=True)

        event_qs = RoadmapEvent.objects.filter(
            created_at__gte=since,
            step_id__isnull=False,
            event_type__in=[
                RoadmapEvent.Type.STEP_EXPOSED,
                RoadmapEvent.Type.STEP_CLICKED,
                RoadmapEvent.Type.STEP_SKIPPED,
                RoadmapEvent.Type.STEP_COMPLETED,
            ],
        )
        if category != "all":
            event_qs = event_qs.filter(step__plan__category=category)
        if not include_ga:
            event_qs = event_qs.exclude(user__username__startswith="ga_")

        # One row per (user_id, step_id)
        instances: dict[tuple[int, int], dict[str, Any]] = {}
        for row in event_qs.order_by("user_id", "step_id", "created_at", "id").values(
            "user_id", "step_id", "event_type", "created_at", "context"
        ):
            user_id = int(row["user_id"])
            step_id = int(row["step_id"])
            key = (user_id, step_id)
            rec = instances.get(key)
            if rec is None:
                rec = {
                    "user_id": user_id,
                    "step_id": step_id,
                    "first_exposed_at": None,
                    "first_clicked_at": None,
                    "first_skipped_at": None,
                    "first_completed_at": None,
                    "has_exposed": 0,
                    "has_clicked": 0,
                    "has_skipped": 0,
                    "has_completed": 0,
                    "expose_count": 0,
                    "exposed_from_offers": 0,
                    "offer_assignment_id": None,
                    "transaction_id": None,
                }
                instances[key] = rec

            event_type = str(row["event_type"])
            dt = row.get("created_at")
            ctx = _safe_dict(row.get("context"))

            if event_type == RoadmapEvent.Type.STEP_EXPOSED:
                rec["has_exposed"] = 1
                rec["expose_count"] += 1
                if rec["first_exposed_at"] is None:
                    rec["first_exposed_at"] = dt
                sources_raw = ctx.get("sources")
                if isinstance(sources_raw, list):
                    sources = {str(x).strip().lower() for x in sources_raw if str(x).strip()}
                else:
                    sources = set()
                offer_assignment_id = _to_int(ctx.get("offer_assignment_id"))
                if offer_assignment_id is not None:
                    if rec["offer_assignment_id"] is None:
                        rec["offer_assignment_id"] = offer_assignment_id
                if offer_assignment_id is not None or ("offers" in sources):
                    rec["exposed_from_offers"] = 1
            elif event_type == RoadmapEvent.Type.STEP_CLICKED:
                rec["has_clicked"] = 1
                if rec["first_clicked_at"] is None:
                    rec["first_clicked_at"] = dt
            elif event_type == RoadmapEvent.Type.STEP_SKIPPED:
                rec["has_skipped"] = 1
                if rec["first_skipped_at"] is None:
                    rec["first_skipped_at"] = dt
            elif event_type == RoadmapEvent.Type.STEP_COMPLETED:
                rec["has_completed"] = 1
                if rec["first_completed_at"] is None:
                    rec["first_completed_at"] = dt
                tx_id = _to_int(ctx.get("transaction_id"))
                if tx_id is not None and rec["transaction_id"] is None:
                    rec["transaction_id"] = tx_id

        if not instances:
            rows: list[dict[str, Any]] = []
            headers = [
                "user_id",
                "plan_id",
                "step_id",
                "category",
                "step_index",
                "step_product_type",
                "step_status_at_export",
                "cadence",
                "recommended_product_id",
                "suggestions_count",
                "why_text",
                "is_fragrance_slot",
                "first_exposed_at",
                "first_clicked_at",
                "first_skipped_at",
                "first_completed_at",
                "has_exposed",
                "has_clicked",
                "has_skipped",
                "has_completed",
                "latency_click_sec",
                "latency_complete_sec",
                "expose_count",
                "exposed_from_offers",
                "offer_assignment_id",
                "transaction_id",
                "rec_product_type",
                "rec_brand",
                "rec_price",
                "rec_in_stock",
                "computed_slot_of_recommended",
                "last_k_purchases_product_ids",
                "last_k_purchases_product_types",
                "last_k_purchases_categories",
                "days_since_last_purchase_in_category",
                "tx_count_90d_category",
                "tx_amount_90d_category",
            ]
            self._write_output(fmt=fmt, out_path=out_path, rows=rows, headers=headers)
            self.stdout.write(f"[export_roadmap_funnel] file={out_path}")
            self.stdout.write("[export_roadmap_funnel] rows_exported=0")
            self.stdout.write("[export_roadmap_funnel] has_exposed=0 has_clicked=0 has_completed=0 has_skipped=0")
            return

        step_ids = sorted({int(key[1]) for key in instances.keys()})
        step_qs = (
            RoadmapStep.objects.filter(id__in=step_ids)
            .select_related("plan", "recommended_product")
            .values(
                "id",
                "plan_id",
                "plan__user_id",
                "plan__category",
                "step_index",
                "product_type",
                "status",
                "cadence",
                "recommended_product_id",
                "suggestions",
                "why",
            )
        )
        step_map = {int(row["id"]): row for row in step_qs}

        rec_product_ids = {
            int(step["recommended_product_id"])
            for step in step_map.values()
            if step.get("recommended_product_id") is not None
        }
        rec_product_map = {
            int(row["id"]): row
            for row in Product.objects.filter(id__in=rec_product_ids).values(
                "id", "product_type", "brand", "price", "in_stock", "attrs", "raw_meta"
            )
        }

        last_k_cache: dict[int, dict[str, str]] = {}
        ctx90_cache: dict[tuple[int, str], dict[str, Any]] = {}

        def user_last_k(user_id: int) -> dict[str, str]:
            cached = last_k_cache.get(user_id)
            if cached is not None:
                return cached
            items = list(
                TransactionItem.objects.filter(transaction__user_id=user_id)
                .order_by("-transaction__created_at", "-id")
                .values("product_id", "product__product_type", "product__category")[:k]
            )
            product_ids = [int(x["product_id"]) for x in items]
            product_types = [str(x.get("product__product_type") or "") for x in items]
            categories = [str(x.get("product__category") or "") for x in items]
            out = {
                "last_k_purchases_product_ids": json.dumps(product_ids, ensure_ascii=False),
                "last_k_purchases_product_types": json.dumps(product_types, ensure_ascii=False),
                "last_k_purchases_categories": json.dumps(categories, ensure_ascii=False),
            }
            last_k_cache[user_id] = out
            return out

        def user_category_90d(user_id: int, cat: str) -> dict[str, Any]:
            key = (user_id, cat)
            cached = ctx90_cache.get(key)
            if cached is not None:
                return cached

            last_item = (
                TransactionItem.objects.filter(transaction__user_id=user_id, product__category=cat)
                .order_by("-transaction__created_at")
                .values("transaction__created_at")
                .first()
            )
            if last_item and last_item.get("transaction__created_at"):
                days_since = (now_utc.date() - last_item["transaction__created_at"].astimezone(dt_timezone.utc).date()).days
            else:
                days_since = None

            tx_ids_90 = list(
                TransactionItem.objects.filter(
                    transaction__user_id=user_id,
                    product__category=cat,
                    transaction__created_at__gte=since90,
                )
                .values_list("transaction_id", flat=True)
                .distinct()
            )
            tx_count_90d = len(tx_ids_90)
            if tx_ids_90:
                amount_agg = Transaction.objects.filter(user_id=user_id, id__in=tx_ids_90).aggregate(
                    total=Sum("total_amount")
                )
                tx_amount_90d = float(amount_agg.get("total") or 0.0)
            else:
                tx_amount_90d = 0.0

            out = {
                "days_since_last_purchase_in_category": days_since,
                "tx_count_90d_category": tx_count_90d,
                "tx_amount_90d_category": round(tx_amount_90d, 2),
            }
            ctx90_cache[key] = out
            return out

        rows: list[dict[str, Any]] = []
        for (user_id, step_id), agg in instances.items():
            step = step_map.get(int(step_id))
            if not step:
                continue

            step_category = str(step.get("plan__category") or "")
            step_product_type = str(step.get("product_type") or "")
            is_fragrance_slot = int(step_category == "fragrance" and step_product_type in SLOTS)

            recommended_product_id = _to_int(step.get("recommended_product_id"))
            rec = rec_product_map.get(recommended_product_id) if recommended_product_id is not None else None
            if rec:
                rec_product_type = str(rec.get("product_type") or "")
                rec_brand = str(rec.get("brand") or "")
                rec_price = float(rec.get("price") or 0.0)
                rec_in_stock = 1 if bool(rec.get("in_stock")) else 0
                computed_slot_of_recommended = (
                    slot_of_fragrance(
                        _safe_dict(rec.get("attrs")),
                        raw_meta=_safe_dict(rec.get("raw_meta")),
                    )
                    if step_category == "fragrance"
                    else None
                )
            else:
                rec_product_type = None
                rec_brand = None
                rec_price = None
                rec_in_stock = None
                computed_slot_of_recommended = None

            last_k_ctx = user_last_k(user_id)
            ctx90 = user_category_90d(user_id, step_category)

            row = {
                "user_id": user_id,
                "plan_id": _to_int(step.get("plan_id")),
                "step_id": step_id,
                "category": step_category,
                "step_index": _to_int(step.get("step_index")),
                "step_product_type": step_product_type,
                "step_status_at_export": str(step.get("status") or ""),
                "cadence": step.get("cadence"),
                "recommended_product_id": recommended_product_id,
                "suggestions_count": len(_safe_list(step.get("suggestions"))),
                "why_text": _why_text(step.get("why")),
                "is_fragrance_slot": is_fragrance_slot,
                "first_exposed_at": _dt_to_utc_iso(agg.get("first_exposed_at")),
                "first_clicked_at": _dt_to_utc_iso(agg.get("first_clicked_at")),
                "first_skipped_at": _dt_to_utc_iso(agg.get("first_skipped_at")),
                "first_completed_at": _dt_to_utc_iso(agg.get("first_completed_at")),
                "has_exposed": int(agg.get("has_exposed") or 0),
                "has_clicked": int(agg.get("has_clicked") or 0),
                "has_skipped": int(agg.get("has_skipped") or 0),
                "has_completed": int(agg.get("has_completed") or 0),
                "latency_click_sec": _latency_sec(
                    agg.get("first_exposed_at"),
                    agg.get("first_clicked_at"),
                ),
                "latency_complete_sec": _latency_sec(
                    agg.get("first_exposed_at"),
                    agg.get("first_completed_at"),
                ),
                "expose_count": int(agg.get("expose_count") or 0),
                "exposed_from_offers": int(agg.get("exposed_from_offers") or 0),
                "offer_assignment_id": _to_int(agg.get("offer_assignment_id")),
                "transaction_id": _to_int(agg.get("transaction_id")),
                "rec_product_type": rec_product_type,
                "rec_brand": rec_brand,
                "rec_price": rec_price,
                "rec_in_stock": rec_in_stock,
                "computed_slot_of_recommended": computed_slot_of_recommended,
                **last_k_ctx,
                **ctx90,
            }
            rows.append(row)

        rows.sort(key=lambda x: (int(x["user_id"]), int(x["step_id"])))
        headers = list(rows[0].keys()) if rows else []
        self._write_output(fmt=fmt, out_path=out_path, rows=rows, headers=headers)

        has_exposed = sum(int(x["has_exposed"]) for x in rows)
        has_clicked = sum(int(x["has_clicked"]) for x in rows)
        has_completed = sum(int(x["has_completed"]) for x in rows)
        has_skipped = sum(int(x["has_skipped"]) for x in rows)

        self.stdout.write(f"[export_roadmap_funnel] file={out_path}")
        self.stdout.write(f"[export_roadmap_funnel] rows_exported={len(rows)}")
        self.stdout.write(
            "[export_roadmap_funnel] "
            f"has_exposed={has_exposed} has_clicked={has_clicked} "
            f"has_completed={has_completed} has_skipped={has_skipped}"
        )

    def _write_output(self, *, fmt: str, out_path: Path, rows: list[dict[str, Any]], headers: list[str]) -> None:
        if fmt == "csv":
            with out_path.open("w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=headers)
                writer.writeheader()
                for row in rows:
                    writer.writerow(row)
            return

        if fmt == "parquet":
            try:
                import pyarrow as pa
                import pyarrow.parquet as pq
            except Exception as exc:
                raise CommandError(
                    "Parquet export requires pyarrow. Install dependency or use --format csv."
                ) from exc
            table = pa.Table.from_pylist(rows)
            pq.write_table(table, str(out_path))
            return

        raise CommandError(f"Unsupported format: {fmt}")
