from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from django.contrib.auth import get_user_model
from django.core.cache import cache
from django.core.management import call_command
from django.core.management.base import BaseCommand, CommandError
from django.db import transaction
from django.utils import timezone as dj_timezone

from admin_tools.goldapple_catalog import export_queryset_jsonl, json_default
from catalog.models import Product
from loyalty.models import LoyaltyAccount, LoyaltyLedgerEntry, Tier
from offers.models import OfferAssignment, OfferEvent
from offers.services import _load_products_for_recs, get_or_assign_next_offer
from recs_analytics.models import RecommendationEvent
from roadmap_app.fragrance_slots import slot_of_fragrance
from roadmap_app.models import RoadmapEvent, RoadmapPlan, RoadmapStep
from roadmap_app.services import refresh_roadmap
from transactions.models import CartItem, OwnedProduct, Transaction, TransactionItem, WishlistItem
from users_app.models import CustomerProfile


ARTIFACT_FIELDS = [
    "source_product_id_raw",
    "name",
    "brand",
    "price",
    "currency",
    "category",
    "product_type",
    "concerns",
    "attrs",
    "actives",
    "flags",
    "supported_skin_types",
    "strength",
    "in_stock",
    "step",
    "image_url",
    "image_urls",
    "description",
    "application_text",
    "ingredients_inci",
    "volume_raw",
    "raw_meta",
]


class Command(BaseCommand):
    help = "Backup disposable demo/dev history, clear synthetic catalog/history, and import confident Goldapple curated rows."

    def add_arguments(self, parser):
        parser.add_argument(
            "--artifact",
            type=str,
            default="data/catalog/curated/goldapple_300_products/goldapple_curated_catalog_confident.jsonl",
            help="Path to confident-only curated Goldapple artifact JSONL.",
        )
        parser.add_argument(
            "--backup-dir",
            type=str,
            default="backups/demo_fresh_rebuild",
            help="Directory where disposable-table backups will be written.",
        )
        parser.add_argument(
            "--inventory-output",
            type=str,
            default="reports/project_inventory_after_fresh_rebuild.md",
            help="Inventory report path after successful import.",
        )
        parser.add_argument("--skip-smoke", action="store_true", help="Skip runtime smoke checks after import.")
        parser.add_argument("--execute", action="store_true", help="Apply reset/import. Without this flag the command is dry-run.")

    def handle(self, *args, **options):
        artifact_path = Path(str(options["artifact"])).resolve()
        backup_root = Path(str(options["backup_dir"])).resolve()
        inventory_output = str(options["inventory_output"]).strip()
        execute = bool(options.get("execute"))
        skip_smoke = bool(options.get("skip_smoke"))

        confident_rows = self._load_confident_rows(artifact_path)
        if not confident_rows:
            raise CommandError(f"No confident rows found in artifact: {artifact_path}")

        plan = self._reset_plan()
        coverage = self._coverage_from_rows(confident_rows)

        self.stdout.write(f"artifact={artifact_path}")
        self.stdout.write(f"confident_rows={len(confident_rows)}")
        self.stdout.write(f"tables_to_clear={json.dumps(plan['counts'], ensure_ascii=False, default=json_default)}")
        self.stdout.write(f"products_after_import_by_category={json.dumps(coverage['by_category'], ensure_ascii=False, default=json_default)}")
        self.stdout.write(f"fragrance_slot_distribution={json.dumps(coverage['fragrance_slots'], ensure_ascii=False, default=json_default)}")

        if not execute:
            self.stdout.write("ready_to_execute=True")
            self.stdout.write("dry_run_only=True")
            return

        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        backup_dir = backup_root / timestamp
        backup_dir.mkdir(parents=True, exist_ok=True)

        backup_manifest = self._backup_disposable_state(backup_dir)
        reset_manifest = self._clear_disposable_state()
        import_manifest = self._import_confident_products(confident_rows)

        cache.clear()
        smoke = {"skipped": True}
        if not skip_smoke:
            smoke = self._run_smoke_checks()

        if inventory_output:
            call_command("report_project_inventory", output=inventory_output)

        summary = {
            "artifact": str(artifact_path),
            "backup_dir": str(backup_dir),
            "backup_manifest": backup_manifest,
            "reset_manifest": reset_manifest,
            "import_manifest": import_manifest,
            "coverage": self._coverage_from_db(),
            "smoke": smoke,
            "inventory_output": inventory_output,
            "completed_at": datetime.now(timezone.utc).isoformat(),
        }
        summary_path = backup_dir / "fresh_rebuild_summary.json"
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=json_default), encoding="utf-8")

        self.stdout.write(self.style.SUCCESS("Fresh rebuild completed"))
        self.stdout.write(f"backup_dir={backup_dir}")
        self.stdout.write(f"summary_path={summary_path}")
        self.stdout.write(f"inventory_output={inventory_output}")
        self.stdout.write(f"coverage_after_import={json.dumps(summary['coverage'], ensure_ascii=False, default=json_default)}")
        self.stdout.write(f"smoke={json.dumps(smoke, ensure_ascii=False, default=json_default)}")

    def _load_confident_rows(self, artifact_path: Path) -> list[dict]:
        if not artifact_path.exists():
            raise CommandError(f"Artifact not found: {artifact_path}")
        rows: list[dict] = []
        with artifact_path.open("r", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                if row.get("import_status") and row.get("import_status") != "confident":
                    continue
                rows.append(row)
        return rows

    def _reset_plan(self) -> dict[str, dict[str, int]]:
        counts = {
            "products": Product.objects.count(),
            "transactions": Transaction.objects.count(),
            "transaction_items": TransactionItem.objects.count(),
            "owned_products": OwnedProduct.objects.count(),
            "wishlist_items": WishlistItem.objects.count(),
            "cart_items": CartItem.objects.count(),
            "offer_assignments": OfferAssignment.objects.count(),
            "offer_events": OfferEvent.objects.count(),
            "recommendation_events": RecommendationEvent.objects.count(),
            "roadmap_plans": RoadmapPlan.objects.count(),
            "roadmap_steps": RoadmapStep.objects.count(),
            "roadmap_events": RoadmapEvent.objects.count(),
            "loyalty_accounts": LoyaltyAccount.objects.count(),
            "loyalty_ledger": LoyaltyLedgerEntry.objects.count(),
        }
        return {"counts": counts}

    def _coverage_from_rows(self, rows: list[dict]) -> dict[str, dict]:
        by_category = Counter()
        by_product_type = Counter()
        in_stock = Counter()
        fragrance_slots = Counter()
        for row in rows:
            category = str(row.get("category") or "")
            product_type = str(row.get("product_type") or "")
            by_category[category] += 1
            by_product_type[f"{category}:{product_type}"] += 1
            in_stock[str(bool(row.get("in_stock"))).lower()] += 1
            if category == "fragrance":
                fragrance_slots[slot_of_fragrance(row.get("attrs") or {}, raw_meta=row.get("raw_meta") or {})] += 1
        return {
            "by_category": dict(by_category),
            "by_product_type": dict(by_product_type),
            "in_stock": dict(in_stock),
            "fragrance_slots": dict(fragrance_slots),
        }

    def _backup_disposable_state(self, backup_dir: Path) -> dict[str, int]:
        manifest: dict[str, int] = {}
        targets = [
            ("products", Product.objects.all()),
            ("transactions", Transaction.objects.all()),
            ("transaction_items", TransactionItem.objects.all()),
            ("owned_products", OwnedProduct.objects.all()),
            ("wishlist_items", WishlistItem.objects.all()),
            ("cart_items", CartItem.objects.all()),
            ("offer_assignments", OfferAssignment.objects.all()),
            ("offer_events", OfferEvent.objects.all()),
            ("recommendation_events", RecommendationEvent.objects.all()),
            ("roadmap_plans", RoadmapPlan.objects.all()),
            ("roadmap_steps", RoadmapStep.objects.all()),
            ("roadmap_events", RoadmapEvent.objects.all()),
            ("loyalty_accounts", LoyaltyAccount.objects.all()),
            ("loyalty_ledger", LoyaltyLedgerEntry.objects.all()),
        ]
        for name, queryset in targets:
            manifest[name] = export_queryset_jsonl(queryset, backup_dir / f"{name}.jsonl")
        manifest_path = backup_dir / "backup_manifest.json"
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, default=json_default), encoding="utf-8")
        return manifest

    def _clear_disposable_state(self) -> dict[str, int]:
        manifest: dict[str, int] = {}
        delete_plan = [
            ("offer_events", OfferEvent),
            ("recommendation_events", RecommendationEvent),
            ("roadmap_events", RoadmapEvent),
            ("wishlist_items", WishlistItem),
            ("cart_items", CartItem),
            ("transaction_items", TransactionItem),
            ("owned_products", OwnedProduct),
            ("roadmap_steps", RoadmapStep),
            ("offer_assignments", OfferAssignment),
            ("roadmap_plans", RoadmapPlan),
            ("transactions", Transaction),
            ("loyalty_ledger", LoyaltyLedgerEntry),
            ("products", Product),
        ]
        for name, model in delete_plan:
            count = model.objects.count()
            model.objects.all().delete()
            manifest[name] = count

        base_tier = Tier.objects.order_by("threshold_spend_90d", "id").first()
        updated = LoyaltyAccount.objects.count()
        if base_tier is not None:
            LoyaltyAccount.objects.all().update(points_balance=0, tier=base_tier)
        else:
            LoyaltyAccount.objects.all().update(points_balance=0)
        manifest["loyalty_accounts_reset"] = updated
        return manifest

    def _import_confident_products(self, rows: list[dict]) -> dict[str, int]:
        to_create: list[Product] = []
        for row in rows:
            source_raw = str(row.get("source_product_id_raw") or row.get("source_row") or "").strip()
            payload = {
                "source_product_id": f"ga:{source_raw}" if source_raw else f"ga:row:{row.get('source_row')}",
                "name": row.get("name") or "",
                "brand": row.get("brand") or "",
                "price": row.get("price"),
                "currency": row.get("currency") or "",
                "category": row.get("category") or "",
                "product_type": row.get("product_type") or "",
                "concerns": row.get("concerns") or [],
                "attrs": row.get("attrs") or {},
                "actives": row.get("actives") or [],
                "flags": row.get("flags") or [],
                "supported_skin_types": row.get("supported_skin_types") or [],
                "strength": row.get("strength") or Product.Strength.LOW,
                "in_stock": bool(row.get("in_stock")),
                "step": row.get("step") or "",
                "image_url": row.get("image_url") or "",
                "image_urls": row.get("image_urls") or [],
                "description": row.get("description") or "",
                "application_text": row.get("application_text") or "",
                "ingredients_inci": row.get("ingredients_inci") or "",
                "volume_raw": row.get("volume_raw") or "",
                "raw_meta": {
                    **(row.get("raw_meta") or {}),
                    "curation_status": "confident",
                    "curation_reason": row.get("import_reason") or "",
                },
            }
            to_create.append(Product(**payload))

        Product.objects.bulk_create(to_create, batch_size=500)
        return {"created_products": len(to_create)}

    def _coverage_from_db(self) -> dict[str, dict]:
        rows = list(Product.objects.values("category", "product_type", "in_stock", "attrs", "raw_meta"))
        by_category = Counter()
        by_product_type = Counter()
        in_stock = Counter()
        fragrance_slots = Counter()
        for row in rows:
            category = str(row.get("category") or "")
            product_type = str(row.get("product_type") or "")
            by_category[category] += 1
            by_product_type[f"{category}:{product_type}"] += 1
            in_stock[str(bool(row.get("in_stock"))).lower()] += 1
            if category == "fragrance":
                fragrance_slots[slot_of_fragrance(row.get("attrs") or {}, raw_meta=row.get("raw_meta") or {})] += 1
        return {
            "products_total": len(rows),
            "by_category": dict(by_category),
            "by_product_type": dict(by_product_type),
            "in_stock": dict(in_stock),
            "fragrance_slots": dict(fragrance_slots),
        }

    def _run_smoke_checks(self) -> dict[str, object]:
        User = get_user_model()
        result: dict[str, object] = {
            "products_total": Product.objects.count(),
            "products_api_count": Product.objects.count(),
        }
        products_for_recs = _load_products_for_recs()
        result["recs_products_count"] = len(products_for_recs)

        with transaction.atomic():
            sample_user = User.objects.order_by("id").first()
            created_user = False
            if sample_user is None:
                sample_user = User.objects.create_user(username="catalog_smoke_user", password="demo12345")
                created_user = True
            CustomerProfile.objects.get_or_create(user=sample_user)

            roadmap_counts = {}
            for category in ("haircare", "skincare", "makeup", "fragrance"):
                plan = refresh_roadmap(sample_user, category=category, post_ctx=None)
                roadmap_counts[category] = plan.steps.count()
            result["roadmap_step_counts"] = roadmap_counts

            assignment = get_or_assign_next_offer(
                sample_user,
                now=dj_timezone.now(),
                context_steps=None,
                post_ctx=None,
                roadmap_ctx=None,
            )
            result["next_offer_assignment_created"] = bool(assignment)
            result["next_offer_assignment_id"] = int(assignment.id) if assignment else None
            result["smoke_user_created"] = created_user
            transaction.set_rollback(True)

        return result
