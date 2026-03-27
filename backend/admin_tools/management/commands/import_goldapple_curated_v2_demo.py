from __future__ import annotations

import json
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any

from django.contrib.auth import get_user_model
from django.core.cache import cache
from django.core.management import call_command
from django.core.management.base import CommandError
from django.db import transaction
from rest_framework.test import APIClient

from admin_tools.goldapple_catalog import json_default
from admin_tools.goldapple_curated_v2_runtime import (
    FRAGRANCE_RETAIL_TYPES,
    ROADMAP_ACTION_TYPES,
    audit_curated_v2_workbook,
    build_final_verdict_md,
    build_runtime_import_report_md,
    coverage_from_rows,
)
from admin_tools.management.commands.fresh_rebuild_demo_catalog import Command as FreshRebuildCommand
from catalog.models import Product
from loyalty.models import LoyaltyAccount, Tier
from offers.models import CampaignBudget, Offer
from roadmap_app.fragrance_slots import SLOTS as FRAGRANCE_SLOTS, slot_of_fragrance
from roadmap_app.models import RoadmapPlan
from users_app.models import CustomerProfile


MAKEUP_SMOKE_ORDER = ("foundation", "mascara", "blush", "lipstick", "eyeshadow", "primer", "setting_spray")


class Command(FreshRebuildCommand):
    help = "Fresh import the curated_v2 Goldapple workbook into a disposable demo/dev DB and run runtime smoke checks."

    def add_arguments(self, parser):
        parser.add_argument(
            "--workbook",
            type=str,
            default="data/catalog/goldapple_300_products_curated_v2.xlsx",
            help="Curated v2 workbook path.",
        )
        parser.add_argument(
            "--backup-dir",
            type=str,
            default="backups/curated_v2_runtime_import",
            help="Directory where disposable-table backups will be written.",
        )
        parser.add_argument(
            "--inventory-output",
            type=str,
            default="reports/project_inventory_after_curated_v2_import.md",
            help="Inventory report path after successful import.",
        )
        parser.add_argument(
            "--report-md",
            type=str,
            default="reports/goldapple_catalog_curated_v2_runtime_import.md",
            help="Markdown report path.",
        )
        parser.add_argument(
            "--report-json",
            type=str,
            default="reports/goldapple_catalog_curated_v2_runtime_import.json",
            help="JSON report path.",
        )
        parser.add_argument(
            "--verdict-md",
            type=str,
            default="reports/goldapple_catalog_curated_v2_runtime_verdict.md",
            help="Final verdict markdown path.",
        )
        parser.add_argument("--skip-smoke", action="store_true", help="Skip runtime smoke checks after import.")
        parser.add_argument("--execute", action="store_true", help="Apply reset/import. Without this flag the command is dry-run.")

    def handle(self, *args, **options):
        workbook_path = str(options["workbook"])
        backup_root = Path(str(options["backup_dir"])).resolve()
        inventory_output = str(options["inventory_output"]).strip()
        report_md_path = Path(str(options["report_md"])).resolve()
        report_json_path = Path(str(options["report_json"])).resolve()
        verdict_md_path = Path(str(options["verdict_md"])).resolve()
        execute = bool(options.get("execute"))
        skip_smoke = bool(options.get("skip_smoke"))

        audit = audit_curated_v2_workbook(workbook_path)
        before_counts = self._reset_plan()["counts"]
        report: dict[str, Any] = {
            "executed": False,
            "workbook": str(Path(workbook_path).resolve()),
            "audit": audit,
            "import": {
                "products_before": int(before_counts.get("products", 0)),
                "counts_before_reset": before_counts,
            },
            "smoke": {"skipped": True},
        }

        report["verdict"] = self._build_verdict(report)
        self._write_reports(report, report_md_path=report_md_path, report_json_path=report_json_path, verdict_md_path=verdict_md_path)

        self.stdout.write(f"workbook={report['workbook']}")
        self.stdout.write(f"rows_total={audit['rows_total']}")
        self.stdout.write(f"rows_valid={audit['rows_valid']}")
        self.stdout.write(f"duplicate_groups={audit['duplicate_groups_count']}")
        self.stdout.write(f"blocking_invalid_rows={len(audit['blocking_issues'])}")

        if not execute:
            self.stdout.write(f"ready_to_execute={str(bool(audit['ready_to_import']))}")
            self.stdout.write("dry_run_only=True")
            self.stdout.write(f"report_md={report_md_path}")
            self.stdout.write(f"report_json={report_json_path}")
            self.stdout.write(f"verdict_md={verdict_md_path}")
            return

        if not audit["ready_to_import"]:
            raise CommandError("Curated workbook is not ready to import. Check blocking issues in the audit report.")

        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        backup_dir = backup_root / timestamp
        backup_dir.mkdir(parents=True, exist_ok=True)

        backup_manifest = self._backup_disposable_state(backup_dir)
        reset_manifest = self._clear_disposable_state()
        import_manifest = self._import_confident_products(audit["normalized_rows"])
        cache.clear()

        smoke = {"skipped": True}
        if not skip_smoke:
            smoke = self._run_api_runtime_smoke_checks()

        if inventory_output:
            call_command("report_project_inventory", output=inventory_output)

        report = {
            "executed": True,
            "workbook": str(Path(workbook_path).resolve()),
            "audit": audit,
            "import": {
                "products_before": int(before_counts.get("products", 0)),
                "products_after": int(Product.objects.count()),
                "backup_dir": str(backup_dir),
                "backup_manifest": backup_manifest,
                "reset_manifest": reset_manifest,
                "import_manifest": import_manifest,
                "coverage_after": self._coverage_from_db_extended(),
                "inventory_output": inventory_output,
            },
            "smoke": smoke,
        }
        report["verdict"] = self._build_verdict(report)
        self._write_reports(report, report_md_path=report_md_path, report_json_path=report_json_path, verdict_md_path=verdict_md_path)

        summary_path = backup_dir / "runtime_import_summary.json"
        summary_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=json_default), encoding="utf-8")

        self.stdout.write(self.style.SUCCESS("Curated v2 demo import completed"))
        self.stdout.write(f"backup_dir={backup_dir}")
        self.stdout.write(f"summary_path={summary_path}")
        self.stdout.write(f"report_md={report_md_path}")
        self.stdout.write(f"report_json={report_json_path}")
        self.stdout.write(f"verdict_md={verdict_md_path}")
        self.stdout.write(f"safe_for_runtime_catalog={report['verdict']['safe_for_runtime_catalog']}")

    def _coverage_from_db_extended(self) -> dict[str, Any]:
        rows = list(
            Product.objects.values(
                "category",
                "product_type",
                "in_stock",
                "attrs",
                "raw_meta",
                "name",
                "brand",
                "price",
                "currency",
                "concerns",
                "actives",
                "flags",
                "supported_skin_types",
                "strength",
                "step",
                "image_url",
                "image_urls",
                "description",
                "application_text",
                "ingredients_inci",
                "volume_raw",
            )
        )
        return coverage_from_rows(rows)

    def _write_reports(self, report: dict[str, Any], *, report_md_path: Path, report_json_path: Path, verdict_md_path: Path) -> None:
        report_md_path.parent.mkdir(parents=True, exist_ok=True)
        report_json_path.parent.mkdir(parents=True, exist_ok=True)
        verdict_md_path.parent.mkdir(parents=True, exist_ok=True)
        report_md_path.write_text(build_runtime_import_report_md(report), encoding="utf-8")
        report_json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=json_default), encoding="utf-8")
        verdict_md_path.write_text(build_final_verdict_md(report), encoding="utf-8")

    def _build_verdict(self, report: dict[str, Any]) -> dict[str, Any]:
        audit = report.get("audit") or {}
        smoke = report.get("smoke") or {}
        blockers: list[str] = []
        dataset_notes = [
            "project-DB dataset rebuild is blocked until new non-trivial demo history is seeded on top of the fresh catalog"
        ]
        if audit.get("required_headers_missing"):
            blockers.append(f"missing headers: {audit['required_headers_missing']}")
        if audit.get("blocking_issues"):
            blockers.append(f"blocking invalid rows: {len(audit['blocking_issues'])}")
        if report.get("executed") and not smoke.get("skipped") and (smoke.get("failures") or []):
            blockers.extend(str(item) for item in (smoke.get("failures") or []))
        safe_for_demo_catalog = bool(audit.get("ready_to_import"))
        safe_for_runtime_catalog = bool(report.get("executed")) and not bool(smoke.get("skipped")) and not blockers
        return {
            "safe_for_demo_catalog": safe_for_demo_catalog,
            "safe_for_runtime_catalog": safe_for_runtime_catalog,
            "safe_for_dataset_rebuild": False,
            "blocking_issues": blockers
            or (
                []
                if report.get("executed")
                else ["runtime smoke not executed yet; dataset rebuild still blocked until new demo history exists"]
            ),
            "dataset_rebuild_notes": dataset_notes,
        }

    def _run_api_runtime_smoke_checks(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "catalog_endpoints": {},
            "roadmap_endpoints": {},
            "offer_endpoints": {},
            "checkout_scenarios": {},
            "semantic_checks": {},
            "failures": [],
        }
        with transaction.atomic():
            self._ensure_smoke_offer_fixtures()
            anon_client = APIClient()
            result["catalog_endpoints"]["/api/products"] = self._check_response_status(
                anon_client.get("/api/products", follow=True), expected=200, label="GET /api/products", failures=result["failures"]
            )
            result["catalog_endpoints"]["/api/products?category=haircare"] = self._check_response_status(
                anon_client.get("/api/products", {"category": "haircare"}, follow=True),
                expected=200,
                label="GET /api/products?category=haircare",
                failures=result["failures"],
            )
            result["catalog_endpoints"]["/api/products?category=fragrance"] = self._check_response_status(
                anon_client.get("/api/products", {"category": "fragrance"}, follow=True),
                expected=200,
                label="GET /api/products?category=fragrance",
                failures=result["failures"],
            )

            _, preflight_client = self._build_smoke_user("preflight")
            result["offer_endpoints"]["/api/me/next-offer"] = self._check_response_status(
                preflight_client.get("/api/me/next-offer"),
                expected=200,
                label="GET /api/me/next-offer",
                failures=result["failures"],
            )

            for category in ("haircare", "skincare", "makeup", "fragrance"):
                category_user, category_client = self._build_smoke_user(category)
                roadmap_key = f"/api/me/roadmap?category={category}"
                roadmap_resp = category_client.get("/api/me/roadmap", {"category": category})
                roadmap_ok = self._check_response_status(
                    roadmap_resp,
                    expected=200,
                    label=f"GET {roadmap_key}",
                    failures=result["failures"],
                )
                result["roadmap_endpoints"][roadmap_key] = roadmap_ok
                result["checkout_scenarios"][category] = self._run_checkout_scenario(
                    category_user,
                    category_client,
                    category,
                    failures=result["failures"],
                )

            transaction.set_rollback(True)
        return result

    def _check_response_status(self, response, *, expected: int, label: str, failures: list[str]) -> dict[str, Any]:
        ok = int(response.status_code) == int(expected)
        if not ok:
            failures.append(f"{label} returned {response.status_code}")
        return {"status_code": int(response.status_code), "ok": ok}

    def _build_smoke_user(self, suffix: str) -> tuple[Any, APIClient]:
        User = get_user_model()
        stamp = datetime.now(timezone.utc).strftime("%H%M%S%f")
        username = f"curated_v2_smoke_{suffix}_{stamp}"
        user = User.objects.create_user(username=username, password="demo12345")
        profile, _ = CustomerProfile.objects.get_or_create(user=user)
        profile.skin_type = CustomerProfile.SkinType.NORMAL
        profile.goals = ["hydration"]
        profile.avoid_flags = []
        profile.budget = CustomerProfile.Budget.MEDIUM
        profile.hair_profile = {"hair_type": "normal", "scalp_type": "normal", "hair_thickness": "medium"}
        profile.makeup_profile = {"finish": "natural"}
        profile.fragrance_profile = {"preferred_families": ["citrus", "woody"], "preferred_intensity": "medium"}
        profile.save()

        bronze, _ = Tier.objects.get_or_create(
            name="Bronze",
            defaults={"threshold_spend_90d": 0, "points_rate": Decimal("0.10")},
        )
        LoyaltyAccount.objects.get_or_create(user=user, defaults={"tier": bronze, "points_balance": 0})

        client = APIClient()
        client.force_authenticate(user)
        return user, client

    def _ensure_smoke_offer_fixtures(self) -> None:
        default_campaign, _ = CampaignBudget.objects.get_or_create(
            name="default",
            defaults={
                "weekly_limit": Decimal("1000.00"),
                "weekly_spent": Decimal("0.00"),
                "priority": 100,
                "is_active": True,
            },
        )
        fragrance_campaign, _ = CampaignBudget.objects.get_or_create(
            name="fragrance_crosssell",
            defaults={
                "weekly_limit": Decimal("1000.00"),
                "weekly_spent": Decimal("0.00"),
                "priority": 50,
                "is_active": True,
                "allowed_categories": ["fragrance"],
            },
        )
        skincare_campaign, _ = CampaignBudget.objects.get_or_create(
            name="skincare_retention",
            defaults={
                "weekly_limit": Decimal("1000.00"),
                "weekly_spent": Decimal("0.00"),
                "priority": 60,
                "is_active": True,
                "allowed_categories": ["skincare"],
            },
        )
        makeup_campaign, _ = CampaignBudget.objects.get_or_create(
            name="makeup_push",
            defaults={
                "weekly_limit": Decimal("1000.00"),
                "weekly_spent": Decimal("0.00"),
                "priority": 70,
                "is_active": True,
                "allowed_categories": ["makeup"],
            },
        )

        def _ensure_offer(*, name: str, campaign: CampaignBudget, allowed_categories: list[str], allowed_product_types: list[str], target_scope: str) -> None:
            if Offer.objects.filter(name=name, offer_type=Offer.Type.DISCOUNT).exists():
                return
            Offer.objects.create(
                name=name,
                offer_type=Offer.Type.DISCOUNT,
                value=Decimal("10.00"),
                estimated_cost=Decimal("4.00"),
                is_active=True,
                target_scope=target_scope,
                cooldown_days=0,
                expires_in_days=7,
                allowed_categories=allowed_categories,
                allowed_product_types=allowed_product_types,
                campaign=campaign,
            )

        _ensure_offer(
            name="CuratedV2 Haircare Next Step",
            campaign=default_campaign,
            allowed_categories=["haircare"],
            allowed_product_types=list(ROADMAP_ACTION_TYPES["haircare"]),
            target_scope="product_type",
        )
        _ensure_offer(
            name="CuratedV2 Skincare Next Step",
            campaign=skincare_campaign,
            allowed_categories=["skincare"],
            allowed_product_types=list(ROADMAP_ACTION_TYPES["skincare"]),
            target_scope="product_type",
        )
        _ensure_offer(
            name="CuratedV2 Makeup Next Step",
            campaign=makeup_campaign,
            allowed_categories=["makeup"],
            allowed_product_types=list(ROADMAP_ACTION_TYPES["makeup"]),
            target_scope="product_type",
        )
        _ensure_offer(
            name="CuratedV2 Fragrance Slot",
            campaign=fragrance_campaign,
            allowed_categories=["fragrance"],
            allowed_product_types=list(FRAGRANCE_RETAIL_TYPES),
            target_scope="product_id",
        )

    def _run_checkout_scenario(self, user, client: APIClient, category: str, *, failures: list[str]) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "seed_product_id": None,
            "seed_product_type": None,
            "checkout_status_code": None,
            "roadmap_status_code_after_checkout": None,
            "next_offer_status_code_after_checkout": None,
            "roadmap_step_types_after_checkout": [],
            "recommended_product_ids": [],
            "recommended_product_product_types": [],
            "next_step_product_type": None,
            "next_step_has_recommended_product": False,
        }

        product = self._pick_seed_product(category)
        if product is None:
            failures.append(f"{category}: no in-stock seed product available")
            payload["ok"] = False
            return payload

        payload["seed_product_id"] = int(product.id)
        payload["seed_product_type"] = str(product.product_type)

        checkout_resp = client.post(
            "/api/checkout",
            {"channel": "offline", "items": [{"product": int(product.id), "quantity": 1}]},
            format="json",
        )
        payload["checkout_status_code"] = int(checkout_resp.status_code)
        if checkout_resp.status_code != 201:
            failures.append(f"{category}: checkout returned {checkout_resp.status_code}")
            payload["ok"] = False
            return payload

        roadmap_resp = client.get("/api/me/roadmap", {"category": category})
        payload["roadmap_status_code_after_checkout"] = int(roadmap_resp.status_code)
        if roadmap_resp.status_code != 200:
            failures.append(f"{category}: roadmap after checkout returned {roadmap_resp.status_code}")
            payload["ok"] = False
            return payload

        next_offer_resp = client.get("/api/me/next-offer")
        payload["next_offer_status_code_after_checkout"] = int(next_offer_resp.status_code)
        if next_offer_resp.status_code != 200:
            failures.append(f"{category}: next-offer after checkout returned {next_offer_resp.status_code}")

        roadmap_data = getattr(roadmap_resp, "data", None) or {}
        steps = roadmap_data.get("steps") or []
        summary = roadmap_data.get("summary") or {}
        next_step = summary.get("next_step") or {}
        payload["roadmap_step_types_after_checkout"] = [str(step.get("product_type") or "") for step in steps]
        payload["next_step_product_type"] = str(next_step.get("product_type") or "") or None
        payload["next_step_has_recommended_product"] = bool(next_step.get("recommended_product"))

        recommended_rows = []
        for step in steps:
            recommended = step.get("recommended_product") or {}
            if not recommended:
                continue
            recommended_rows.append(recommended)
        payload["recommended_product_ids"] = [int(item["id"]) for item in recommended_rows]
        payload["recommended_product_product_types"] = [str(item.get("product_type") or "") for item in recommended_rows]

        if category == "haircare":
            self._validate_haircare_semantics(payload, failures=failures)
        elif category == "skincare":
            self._validate_skincare_semantics(payload, failures=failures)
        elif category == "makeup":
            self._validate_makeup_semantics(payload, failures=failures)
        elif category == "fragrance":
            self._validate_fragrance_semantics(user, payload, next_offer_resp=next_offer_resp, failures=failures)

        payload["ok"] = not any(str(item).startswith(f"{category}:") for item in failures)
        return payload

    def _pick_seed_product(self, category: str) -> Product | None:
        qs = Product.objects.filter(category=category, in_stock=True).order_by("id")
        if category == "haircare":
            return qs.filter(product_type="shampoo").first() or qs.first()
        if category == "skincare":
            return qs.filter(product_type="cleanser").first() or qs.first()
        if category == "makeup":
            for product_type in MAKEUP_SMOKE_ORDER:
                product = qs.filter(product_type=product_type).first()
                if product is not None:
                    return product
            return qs.first()
        if category == "fragrance":
            return qs.filter(product_type__in=FRAGRANCE_RETAIL_TYPES).first() or qs.first()
        return qs.first()

    def _validate_haircare_semantics(self, payload: dict[str, Any], *, failures: list[str]) -> None:
        step_types = payload["roadmap_step_types_after_checkout"]
        allowed = set(ROADMAP_ACTION_TYPES["haircare"])
        if not step_types or any(step_type not in allowed for step_type in step_types):
            failures.append("haircare: roadmap step types escaped the canonical haircare ontology")
        if payload["seed_product_type"] == "shampoo" and payload["next_step_product_type"] == "shampoo":
            failures.append("haircare: shampoo checkout left shampoo as next step")
        if payload["seed_product_type"] == "shampoo" and not payload["next_step_has_recommended_product"]:
            failures.append("haircare: next step after shampoo checkout has no recommended product")
        if not {"conditioner", "hair_mask", "hair_oil"} & set(step_types):
            failures.append("haircare: conditioner/hair_mask/hair_oil chain is missing after import")

    def _validate_skincare_semantics(self, payload: dict[str, Any], *, failures: list[str]) -> None:
        step_types = payload["roadmap_step_types_after_checkout"]
        allowed = set(ROADMAP_ACTION_TYPES["skincare"])
        if not step_types or any(step_type not in allowed for step_type in step_types):
            failures.append("skincare: roadmap step types escaped the canonical skincare ontology")
        if not {"serum", "moisturizer", "spf"} <= set(step_types):
            failures.append("skincare: cleanser/serum/moisturizer/spf chain is incomplete after import")
        if payload["next_step_product_type"] and payload["next_step_product_type"] not in allowed:
            failures.append("skincare: next step product_type is outside skincare ontology")
        if payload["next_step_product_type"] and not payload["next_step_has_recommended_product"]:
            failures.append("skincare: next step has no recommended product")

    def _validate_makeup_semantics(self, payload: dict[str, Any], *, failures: list[str]) -> None:
        step_types = payload["roadmap_step_types_after_checkout"]
        allowed = set(ROADMAP_ACTION_TYPES["makeup"])
        if not step_types or any(step_type not in allowed for step_type in step_types):
            failures.append("makeup: roadmap step types escaped the canonical makeup ontology")
        if payload["next_step_product_type"] and payload["next_step_product_type"] not in allowed:
            failures.append("makeup: next step product_type is outside makeup ontology")
        if payload["next_step_product_type"] and not payload["next_step_has_recommended_product"]:
            failures.append("makeup: next step has no recommended product")

    def _validate_fragrance_semantics(self, user, payload: dict[str, Any], *, next_offer_resp, failures: list[str]) -> None:
        step_types = payload["roadmap_step_types_after_checkout"]
        if not step_types or any(step_type not in FRAGRANCE_SLOTS for step_type in step_types):
            failures.append("fragrance: roadmap steps are not slot-level")

        recommended_types = set(payload["recommended_product_product_types"])
        if any(product_type not in FRAGRANCE_RETAIL_TYPES for product_type in recommended_types):
            failures.append("fragrance: recommended catalog products are not retail-level types")

        if payload["next_step_product_type"] and payload["next_step_product_type"] not in FRAGRANCE_SLOTS:
            failures.append("fragrance: next step is not slot-level")
        if payload["next_step_product_type"] and not payload["next_step_has_recommended_product"]:
            failures.append("fragrance: next step has no recommended product")

        plan = (
            RoadmapPlan.objects.filter(user=user, category="fragrance", is_active=True)
            .order_by("-id")
            .first()
        )
        if plan is not None:
            rows = list(plan.steps.select_related("recommended_product").order_by("step_index"))
            for step in rows:
                if step.recommended_product_id is None:
                    continue
                slot = slot_of_fragrance(step.recommended_product.attrs or {}, raw_meta=step.recommended_product.raw_meta or {})
                if slot != step.product_type:
                    failures.append("fragrance: recommended_product slot mismatch after import")
                    break

        next_offer = getattr(next_offer_resp, "data", None) or {}
        target = (next_offer.get("target") or {}) if isinstance(next_offer, dict) else {}
        payload["next_offer_target"] = target
        if target:
            if target.get("category") == "fragrance" and target.get("scope") == "product_id":
                if target.get("product_type") not in FRAGRANCE_SLOTS:
                    failures.append("fragrance: next-offer target.product_type is not slot-level")
                actual_product_type = target.get("actual_product_type")
                if actual_product_type not in FRAGRANCE_RETAIL_TYPES:
                    failures.append("fragrance: next-offer target.actual_product_type is not retail-level")
                product_id = target.get("value")
                try:
                    offer_product = Product.objects.get(id=product_id)
                except Product.DoesNotExist:
                    failures.append("fragrance: next-offer target product_id does not exist")
                else:
                    slot = slot_of_fragrance(offer_product.attrs or {}, raw_meta=offer_product.raw_meta or {})
                    if slot != target.get("product_type"):
                        failures.append("fragrance: next-offer target product_id does not match slot")
            else:
                failures.append("fragrance: next-offer did not return a slot-aware product_id target")
