from __future__ import annotations

import csv
import io
import json
from collections import Counter
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from django.apps import apps
from django.conf import settings
from django.contrib.auth import get_user_model
from django.core.management import call_command
from django.core.management.base import BaseCommand, CommandError
from django.db import connection, transaction

from catalog.models import Product
from offers.models import CampaignBudget, Offer, OfferAssignment, OfferEvent
from roadmap_app.models import RoadmapEvent, RoadmapPlan, RoadmapStep
from transactions.models import OwnedProduct, Transaction, TransactionItem
from users_app.models import CustomerProfile


FRAGRANCE_SLOT_TYPES = {"warm_day", "warm_evening", "cold_day", "cold_evening"}
NULL_TOKENS = {"", "null", "none", "nan", "nat"}
SUMMARY_PREVIEW_LINES = 14
SAMPLE_VALIDATION_LIMIT = 1000


def is_nullish(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip().lower() in NULL_TOKENS
    return False


def parse_str(value: Any, default: str = "") -> str:
    if is_nullish(value):
        return default
    return str(value).strip()


def parse_optional_str(value: Any) -> str | None:
    if is_nullish(value):
        return None
    return str(value).strip()


def parse_bool(*, file_name: str, row_no: int, column: str, value: Any, default: bool = False) -> bool:
    if is_nullish(value):
        return bool(default)
    text = str(value).strip().lower()
    if text in {"1", "true", "t", "yes", "y"}:
        return True
    if text in {"0", "false", "f", "no", "n"}:
        return False
    raise CommandError(f"{file_name}:{row_no} invalid bool in '{column}': {value!r}")


def _parse_decimal_value(file_name: str, row_no: int, column: str, value: Any) -> Decimal:
    try:
        return Decimal(str(value).strip())
    except (InvalidOperation, ValueError, TypeError):
        raise CommandError(f"{file_name}:{row_no} invalid decimal in '{column}': {value!r}")


def parse_int(
    *,
    file_name: str,
    row_no: int,
    column: str,
    value: Any,
    required: bool = False,
    default: int | None = 0,
) -> int | None:
    if is_nullish(value):
        if required and default is None:
            raise CommandError(f"{file_name}:{row_no} required integer '{column}' is empty")
        return default
    dec = _parse_decimal_value(file_name, row_no, column, value)
    if dec != dec.to_integral_value():
        raise CommandError(f"{file_name}:{row_no} non-integer value in '{column}': {value!r}")
    return int(dec)


def parse_decimal(
    *,
    file_name: str,
    row_no: int,
    column: str,
    value: Any,
    required: bool = False,
    default: Decimal | None = Decimal("0"),
) -> Decimal | None:
    if is_nullish(value):
        if required and default is None:
            raise CommandError(f"{file_name}:{row_no} required decimal '{column}' is empty")
        return default
    return _parse_decimal_value(file_name, row_no, column, value)


def parse_float(
    *,
    file_name: str,
    row_no: int,
    column: str,
    value: Any,
    required: bool = False,
    default: float | None = 0.0,
) -> float | None:
    if is_nullish(value):
        if required and default is None:
            raise CommandError(f"{file_name}:{row_no} required float '{column}' is empty")
        return default
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        raise CommandError(f"{file_name}:{row_no} invalid float in '{column}': {value!r}")


def parse_dt(
    *,
    file_name: str,
    row_no: int,
    column: str,
    value: Any,
    required: bool = False,
    default: datetime | None = None,
) -> datetime | None:
    if is_nullish(value):
        if required and default is None:
            raise CommandError(f"{file_name}:{row_no} required datetime '{column}' is empty")
        return default
    text = str(value).strip()
    if text.endswith("Z"):
        text = f"{text[:-1]}+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        raise CommandError(f"{file_name}:{row_no} invalid datetime in '{column}': {value!r}")
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    else:
        parsed = parsed.astimezone(timezone.utc)
    return parsed


def parse_json_field(
    *,
    file_name: str,
    row_no: int,
    column: str,
    value: Any,
    expected: str,
) -> dict[str, Any] | list[Any]:
    if expected not in {"dict", "list"}:
        raise ValueError("expected must be 'dict' or 'list'")
    fallback: dict[str, Any] | list[Any] = {} if expected == "dict" else []
    if is_nullish(value):
        return fallback

    if isinstance(value, str):
        text = value.strip()
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as exc:
            raise CommandError(
                f"{file_name}:{row_no} invalid JSON in '{column}': {exc.msg}"
            )
    else:
        parsed = value

    if parsed is None:
        return fallback
    if expected == "dict" and not isinstance(parsed, dict):
        raise CommandError(f"{file_name}:{row_no} JSON field '{column}' must be object")
    if expected == "list" and not isinstance(parsed, list):
        raise CommandError(f"{file_name}:{row_no} JSON field '{column}' must be array")
    return parsed


def load_csv(path: Path, *, file_name: str, required_columns: set[str]):
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = set(reader.fieldnames or [])
        missing = sorted(required_columns.difference(fieldnames))
        if missing:
            raise CommandError(f"{file_name} missing required columns: {missing}")
        for row_no, row in enumerate(reader, start=2):
            yield row_no, row


def bulk_insert(model, objects: list[Any], *, batch_size: int) -> int:
    if not objects:
        return 0
    model.objects.bulk_create(objects, batch_size=batch_size)
    return len(objects)


def validate_transaction_items_have_products() -> None:
    invalid = TransactionItem.objects.exclude(product_id__in=Product.objects.values("id")).count()
    if invalid:
        raise CommandError(f"Validation failed: transaction_items with missing product_id: {invalid}")


def validate_roadmap_steps_have_plans() -> None:
    invalid = RoadmapStep.objects.exclude(plan_id__in=RoadmapPlan.objects.values("id")).count()
    if invalid:
        raise CommandError(f"Validation failed: roadmap_steps with missing plan_id: {invalid}")


def validate_offer_assignments_have_offers() -> None:
    invalid = OfferAssignment.objects.exclude(offer_id__in=Offer.objects.values("id")).count()
    if invalid:
        raise CommandError(f"Validation failed: offer_assignments with missing offer_id: {invalid}")


def validate_fragrance_slot_targets() -> None:
    invalid_ids: list[int] = []
    qs = OfferAssignment.objects.filter(
        target__scope="product_id",
        target__category="fragrance",
        target__product_type__in=sorted(FRAGRANCE_SLOT_TYPES),
    ).values("id", "target")
    for row in qs.iterator():
        target = row["target"]
        if not isinstance(target, dict):
            invalid_ids.append(int(row["id"]))
            continue
        actual = target.get("actual_product_type")
        if not isinstance(actual, str) or not actual.strip():
            invalid_ids.append(int(row["id"]))
    if invalid_ids:
        preview = ", ".join(str(x) for x in invalid_ids[:10])
        raise CommandError(
            "Validation failed: fragrance slot targets with empty "
            f"target.actual_product_type (count={len(invalid_ids)}, sample={preview})"
        )


def validate_roadmap_step_exposed_context() -> None:
    invalid_ids: list[int] = []
    qs = RoadmapEvent.objects.filter(event_type=RoadmapEvent.Type.STEP_EXPOSED).values("id", "context")
    for row in qs.iterator():
        context = row["context"]
        event_id = int(row["id"])
        if not isinstance(context, dict):
            invalid_ids.append(event_id)
            continue
        category = context.get("category")
        if not isinstance(category, str) or not category.strip():
            invalid_ids.append(event_id)
            continue
        sources = context.get("sources")
        if isinstance(sources, list):
            normalized = {str(x).strip().lower() for x in sources if str(x).strip()}
            if "offers" in normalized:
                offer_assignment_id = context.get("offer_assignment_id")
                if is_nullish(offer_assignment_id):
                    invalid_ids.append(event_id)
    if invalid_ids:
        preview = ", ".join(str(x) for x in invalid_ids[:10])
        raise CommandError(
            "Validation failed: RoadmapEvent.STEP_EXPOSED context integrity issue "
            f"(count={len(invalid_ids)}, sample={preview})"
        )


class Command(BaseCommand):
    help = "Import synthetic roadmap/offers/recommendation dataset from CSV files."

    IMPORT_ORDER = [
        "products.csv",
        "users.csv",
        "customer_profiles.csv",
        "transactions.csv",
        "transaction_items.csv",
        "owned_products.csv",
        "roadmap_plans.csv",
        "roadmap_steps.csv",
        "campaign_budgets.csv",
        "offers.csv",
        "offer_assignments.csv",
        "offer_events.csv",
        "roadmap_events.csv",
        "recommendation_events.csv",
    ]

    REQUIRED_COLUMNS = {
        "products.csv": {
            "id",
            "name",
            "brand",
            "price",
            "source_product_id",
            "currency",
            "category",
            "product_type",
            "concerns",
            "attrs",
            "step",
            "actives",
            "flags",
            "supported_skin_types",
            "strength",
            "in_stock",
            "image_url",
            "image_urls",
            "description",
            "application_text",
            "ingredients_inci",
            "volume_raw",
            "raw_meta",
            "created_at",
            "updated_at",
        },
        "users.csv": {"user_id", "username", "segment", "favorite_category", "created_at"},
        "customer_profiles.csv": {
            "user_id",
            "skin_type",
            "goals",
            "avoid_flags",
            "budget",
            "hair_profile",
            "makeup_profile",
            "fragrance_profile",
        },
        "transactions.csv": {
            "transaction_id",
            "user_id",
            "created_at",
            "total_amount",
            "channel",
            "idempotency_key",
            "pricing_meta",
        },
        "transaction_items.csv": {
            "transaction_item_id",
            "transaction_id",
            "user_id",
            "product_id",
            "quantity",
            "unit_price",
        },
        "owned_products.csv": {
            "owned_product_id",
            "user_id",
            "product_id",
            "quantity_total",
            "is_active",
            "last_acquired_at",
            "acquired_at",
            "source",
        },
        "roadmap_plans.csv": {
            "plan_id",
            "user_id",
            "category",
            "is_active",
            "version",
            "meta",
            "created_at",
            "updated_at",
        },
        "roadmap_steps.csv": {
            "step_id",
            "plan_id",
            "step_index",
            "product_type",
            "status",
            "recommended_product_id",
            "suggestions",
            "score",
            "confidence",
            "why",
            "cadence",
            "created_at",
            "updated_at",
        },
        "campaign_budgets.csv": {
            "campaign_id",
            "name",
            "weekly_limit",
            "weekly_spent",
            "priority",
            "is_active",
            "allowed_steps",
            "allowed_categories",
        },
        "offers.csv": {
            "offer_id",
            "is_active",
            "campaign_id",
            "name",
            "offer_type",
            "value",
            "min_total_spend_90d",
            "allowed_steps",
            "estimated_cost",
            "cooldown_days",
            "expires_in_days",
            "allowed_categories",
            "allowed_product_types",
            "target_scope",
            "created_at",
        },
        "offer_assignments.csv": {
            "assignment_id",
            "user_id",
            "offer_id",
            "assigned_at",
            "expires_at",
            "reason",
            "is_active",
            "is_redeemed",
            "redeemed_transaction_id",
            "superseded_at",
            "superseded_by",
            "target",
        },
        "offer_events.csv": {
            "offer_event_id",
            "assignment_id",
            "user_id",
            "offer_id",
            "campaign_name",
            "event_type",
            "event_key",
            "event_version",
            "created_at",
            "request_id",
            "context",
        },
        "roadmap_events.csv": {
            "roadmap_event_id",
            "created_at",
            "user_id",
            "plan_id",
            "step_id",
            "event_type",
            "request_id",
            "context",
        },
        "recommendation_events.csv": {
            "rec_event_id",
            "created_at",
            "user_id",
            "action",
            "page",
            "section_key",
            "request_id",
            "product_id",
            "algo_mode",
            "score",
            "components",
            "context",
        },
    }

    def add_arguments(self, parser):
        parser.add_argument(
            "--path",
            required=True,
            help="Path to directory with extracted CSV files (not zip).",
        )
        parser.add_argument("--truncate", action="store_true", help="Delete existing data before import.")
        parser.add_argument(
            "--i-understand-its-destructive",
            action="store_true",
            help="Required with --truncate to confirm destructive operation.",
        )
        parser.add_argument("--dry-run", action="store_true", help="Parse/validate only. No DB writes.")
        parser.add_argument("--batch-size", type=int, default=5000, help="bulk_create batch size.")

    def handle(self, *args, **options):
        self.warnings: list[str] = []

        dataset_dir = self.resolve_dataset_path(str(options["path"]))
        dry_run = bool(options["dry_run"])
        truncate = bool(options["truncate"])
        destructive_ack = bool(options["i_understand_its_destructive"])
        batch_size = int(options["batch_size"])

        if batch_size <= 0:
            raise CommandError("--batch-size must be > 0")
        if truncate and not destructive_ack:
            raise CommandError(
                "--truncate requires --i-understand-its-destructive. "
                "Refusing to run destructive operation."
            )

        recommendation_model = self.get_recommendation_event_model()
        if recommendation_model is None:
            self.warn(
                "RecommendationEvent model/table is unavailable. "
                "recommendation_events.csv will be skipped."
            )

        self.ensure_files_exist(dataset_dir)

        if dry_run:
            self.stdout.write(
                f"[import_synth_dataset] Dry-run mode. Parsing CSV from: {dataset_dir}"
            )
            rows_per_file = self.run_dry_run(
                dataset_dir=dataset_dir,
                recommendation_model=recommendation_model,
            )
            self.print_rows_per_file(rows_per_file)
            self.stdout.write(self.style.SUCCESS("Dry-run completed successfully. No DB writes were made."))
            self.stdout.write(f"Warnings: {len(self.warnings)}")
            return

        if truncate:
            self.stdout.write("[import_synth_dataset] Truncating target tables...")
            self.truncate_tables(recommendation_model=recommendation_model)

        self.stdout.write(f"[import_synth_dataset] Importing CSV from: {dataset_dir}")
        rows_loaded: dict[str, int] = {}

        rows_loaded["products.csv"] = self.import_products(dataset_dir, batch_size)
        rows_loaded["users.csv"] = self.import_users(dataset_dir, batch_size)
        rows_loaded["customer_profiles.csv"] = self.import_customer_profiles(dataset_dir, batch_size)
        rows_loaded["transactions.csv"] = self.import_transactions(dataset_dir, batch_size)
        rows_loaded["transaction_items.csv"] = self.import_transaction_items(dataset_dir, batch_size)
        rows_loaded["owned_products.csv"] = self.import_owned_products(dataset_dir, batch_size)
        rows_loaded["roadmap_plans.csv"] = self.import_roadmap_plans(dataset_dir, batch_size)
        rows_loaded["roadmap_steps.csv"] = self.import_roadmap_steps(dataset_dir, batch_size)
        rows_loaded["campaign_budgets.csv"] = self.import_campaign_budgets(dataset_dir, batch_size)
        rows_loaded["offers.csv"] = self.import_offers(dataset_dir, batch_size)
        rows_loaded["offer_assignments.csv"] = self.import_offer_assignments(dataset_dir, batch_size)
        rows_loaded["offer_events.csv"] = self.import_offer_events(dataset_dir, batch_size)
        rows_loaded["roadmap_events.csv"] = self.import_roadmap_events(dataset_dir, batch_size)
        if recommendation_model is not None:
            rows_loaded["recommendation_events.csv"] = self.import_recommendation_events(
                dataset_dir, batch_size, recommendation_model
            )
        else:
            rows_loaded["recommendation_events.csv"] = 0

        self.stdout.write("[import_synth_dataset] Running post-import validations...")
        validate_transaction_items_have_products()
        validate_roadmap_steps_have_plans()
        validate_offer_assignments_have_offers()
        validate_fragrance_slot_targets()
        validate_roadmap_step_exposed_context()

        reset_tables = [
            "auth_user",
            "catalog_product",
            "transactions_transaction",
            "transactions_transactionitem",
            "transactions_ownedproduct",
            "offers_offer",
            "offers_offerassignment",
            "offers_offerevent",
            "roadmap_app_roadmapplan",
            "roadmap_app_roadmapstep",
            "roadmap_app_roadmapevent",
            "offers_campaignbudget",
            "users_app_customerprofile",
        ]
        if recommendation_model is not None:
            reset_tables.append(recommendation_model._meta.db_table)
        self.stdout.write("[import_synth_dataset] Running: python manage.py fix_postgres_sequences --apply ...")
        call_command("fix_postgres_sequences", apply=True, tables=reset_tables)

        self.print_rows_per_file(rows_loaded)
        self.print_totals(recommendation_model=recommendation_model)
        self.stdout.write(f"Warnings: {len(self.warnings)}")

        self.stdout.write("[import_synth_dataset] Running: python manage.py report_roadmap_quality --days 1")
        self.print_roadmap_quality_summary()
        self.stdout.write(self.style.SUCCESS("[import_synth_dataset] Import completed."))

    def resolve_dataset_path(self, raw_path: str) -> Path:
        input_path = Path(raw_path).expanduser()
        candidates: list[Path]
        if input_path.is_absolute():
            candidates = [input_path]
        else:
            candidates = [
                (Path.cwd() / input_path),
                (Path(settings.BASE_DIR) / input_path),
                (Path(settings.BASE_DIR).parent / input_path),
            ]

        for candidate in candidates:
            candidate_resolved = candidate.resolve()
            if candidate_resolved.exists() and candidate_resolved.is_dir():
                return candidate_resolved

        tried = ", ".join(str(x.resolve()) for x in candidates)
        raise CommandError(f"Dataset path not found or not a directory. Tried: {tried}")

    def ensure_files_exist(self, dataset_dir: Path) -> None:
        missing = [f for f in self.IMPORT_ORDER if not (dataset_dir / f).exists()]
        if missing:
            raise CommandError(f"Missing required CSV files: {missing}")

    def warn(self, message: str) -> None:
        self.warnings.append(message)
        self.stdout.write(self.style.WARNING(f"WARNING: {message}"))

    def get_recommendation_event_model(self):
        try:
            model = apps.get_model("recs_analytics", "RecommendationEvent")
        except LookupError:
            return None
        table_names = connection.introspection.table_names()
        if model._meta.db_table not in table_names:
            return None
        return model
    def parse_product_row(self, row: dict[str, Any], row_no: int, file_name: str) -> dict[str, Any]:
        attrs = parse_json_field(
            file_name=file_name,
            row_no=row_no,
            column="attrs",
            value=row.get("attrs"),
            expected="dict",
        )
        raw_meta = parse_json_field(
            file_name=file_name,
            row_no=row_no,
            column="raw_meta",
            value=row.get("raw_meta"),
            expected="dict",
        )
        return {
            "id": parse_int(file_name=file_name, row_no=row_no, column="id", value=row.get("id"), required=True, default=None),
            "name": parse_str(row.get("name")),
            "brand": parse_str(row.get("brand")),
            "price": parse_decimal(
                file_name=file_name,
                row_no=row_no,
                column="price",
                value=row.get("price"),
                default=None,
            ),
            "source_product_id": parse_str(row.get("source_product_id")),
            "currency": parse_str(row.get("currency")),
            "category": parse_str(row.get("category"), default=Product.Category.SKINCARE),
            "product_type": parse_str(row.get("product_type")),
            "concerns": parse_json_field(
                file_name=file_name, row_no=row_no, column="concerns", value=row.get("concerns"), expected="list"
            ),
            "attrs": attrs,
            "step": parse_str(row.get("step")),
            "actives": parse_json_field(
                file_name=file_name, row_no=row_no, column="actives", value=row.get("actives"), expected="list"
            ),
            "flags": parse_json_field(
                file_name=file_name, row_no=row_no, column="flags", value=row.get("flags"), expected="list"
            ),
            "supported_skin_types": parse_json_field(
                file_name=file_name,
                row_no=row_no,
                column="supported_skin_types",
                value=row.get("supported_skin_types"),
                expected="list",
            ),
            "strength": parse_str(row.get("strength"), default=Product.Strength.LOW),
            "in_stock": parse_bool(
                file_name=file_name, row_no=row_no, column="in_stock", value=row.get("in_stock"), default=True
            ),
            "image_url": parse_str(row.get("image_url")),
            "image_urls": parse_json_field(
                file_name=file_name, row_no=row_no, column="image_urls", value=row.get("image_urls"), expected="list"
            ),
            "description": parse_str(row.get("description")),
            "application_text": parse_str(row.get("application_text")),
            "ingredients_inci": parse_str(row.get("ingredients_inci")),
            "volume_raw": parse_str(row.get("volume_raw")),
            "raw_meta": raw_meta,
            "created_at": parse_dt(
                file_name=file_name, row_no=row_no, column="created_at", value=row.get("created_at"), required=True
            ),
            "updated_at": parse_dt(
                file_name=file_name, row_no=row_no, column="updated_at", value=row.get("updated_at"), required=True
            ),
        }

    def parse_user_row(self, row: dict[str, Any], row_no: int, file_name: str) -> dict[str, Any]:
        user_id = parse_int(
            file_name=file_name, row_no=row_no, column="user_id", value=row.get("user_id"), required=True, default=None
        )
        created_at = parse_dt(
            file_name=file_name,
            row_no=row_no,
            column="created_at",
            value=row.get("created_at"),
            default=datetime.now(timezone.utc),
        )
        return {
            "id": user_id,
            "username": parse_str(row.get("username"), default=f"user_{user_id}"),
            "password": "!",
            "date_joined": created_at,
            "is_active": True,
            "is_staff": False,
            "is_superuser": False,
        }

    def parse_customer_profile_row(self, row: dict[str, Any], row_no: int, file_name: str) -> dict[str, Any]:
        return {
            "user_id": parse_int(
                file_name=file_name, row_no=row_no, column="user_id", value=row.get("user_id"), required=True, default=None
            ),
            "skin_type": parse_str(row.get("skin_type"), default=CustomerProfile.SkinType.NORMAL),
            "goals": parse_json_field(
                file_name=file_name, row_no=row_no, column="goals", value=row.get("goals"), expected="list"
            ),
            "avoid_flags": parse_json_field(
                file_name=file_name, row_no=row_no, column="avoid_flags", value=row.get("avoid_flags"), expected="list"
            ),
            "budget": parse_str(row.get("budget"), default=CustomerProfile.Budget.MEDIUM),
            "hair_profile": parse_json_field(
                file_name=file_name, row_no=row_no, column="hair_profile", value=row.get("hair_profile"), expected="dict"
            ),
            "makeup_profile": parse_json_field(
                file_name=file_name,
                row_no=row_no,
                column="makeup_profile",
                value=row.get("makeup_profile"),
                expected="dict",
            ),
            "fragrance_profile": parse_json_field(
                file_name=file_name,
                row_no=row_no,
                column="fragrance_profile",
                value=row.get("fragrance_profile"),
                expected="dict",
            ),
        }

    def parse_transaction_row(self, row: dict[str, Any], row_no: int, file_name: str) -> dict[str, Any]:
        return {
            "id": parse_int(
                file_name=file_name,
                row_no=row_no,
                column="transaction_id",
                value=row.get("transaction_id"),
                required=True,
                default=None,
            ),
            "user_id": parse_int(
                file_name=file_name, row_no=row_no, column="user_id", value=row.get("user_id"), required=True, default=None
            ),
            "created_at": parse_dt(
                file_name=file_name, row_no=row_no, column="created_at", value=row.get("created_at"), required=True
            ),
            "total_amount": parse_decimal(
                file_name=file_name, row_no=row_no, column="total_amount", value=row.get("total_amount"), default=Decimal("0")
            ),
            "channel": parse_str(row.get("channel"), default="offline"),
            "idempotency_key": parse_optional_str(row.get("idempotency_key")),
            "pricing_meta": parse_json_field(
                file_name=file_name, row_no=row_no, column="pricing_meta", value=row.get("pricing_meta"), expected="dict"
            ),
        }

    def parse_transaction_item_row(self, row: dict[str, Any], row_no: int, file_name: str) -> dict[str, Any]:
        return {
            "id": parse_int(
                file_name=file_name,
                row_no=row_no,
                column="transaction_item_id",
                value=row.get("transaction_item_id"),
                required=True,
                default=None,
            ),
            "transaction_id": parse_int(
                file_name=file_name,
                row_no=row_no,
                column="transaction_id",
                value=row.get("transaction_id"),
                required=True,
                default=None,
            ),
            "product_id": parse_int(
                file_name=file_name, row_no=row_no, column="product_id", value=row.get("product_id"), required=True, default=None
            ),
            "quantity": parse_int(
                file_name=file_name, row_no=row_no, column="quantity", value=row.get("quantity"), default=0
            ),
            "unit_price": parse_decimal(
                file_name=file_name, row_no=row_no, column="unit_price", value=row.get("unit_price"), default=Decimal("0")
            ),
            "user_id": parse_int(
                file_name=file_name, row_no=row_no, column="user_id", value=row.get("user_id"), default=None
            ),
        }

    def parse_owned_product_row(self, row: dict[str, Any], row_no: int, file_name: str) -> dict[str, Any]:
        now_utc = datetime.now(timezone.utc)
        return {
            "id": parse_int(
                file_name=file_name,
                row_no=row_no,
                column="owned_product_id",
                value=row.get("owned_product_id"),
                required=True,
                default=None,
            ),
            "user_id": parse_int(
                file_name=file_name, row_no=row_no, column="user_id", value=row.get("user_id"), required=True, default=None
            ),
            "product_id": parse_int(
                file_name=file_name, row_no=row_no, column="product_id", value=row.get("product_id"), required=True, default=None
            ),
            "quantity_total": parse_int(
                file_name=file_name, row_no=row_no, column="quantity_total", value=row.get("quantity_total"), default=0
            ),
            "is_active": parse_bool(
                file_name=file_name, row_no=row_no, column="is_active", value=row.get("is_active"), default=True
            ),
            "last_acquired_at": parse_dt(
                file_name=file_name, row_no=row_no, column="last_acquired_at", value=row.get("last_acquired_at"), default=now_utc
            ),
            "acquired_at": parse_dt(
                file_name=file_name, row_no=row_no, column="acquired_at", value=row.get("acquired_at"), default=now_utc
            ),
            "source": parse_str(row.get("source"), default="import"),
        }

    def parse_roadmap_plan_row(self, row: dict[str, Any], row_no: int, file_name: str) -> dict[str, Any]:
        return {
            "id": parse_int(
                file_name=file_name, row_no=row_no, column="plan_id", value=row.get("plan_id"), required=True, default=None
            ),
            "user_id": parse_int(
                file_name=file_name, row_no=row_no, column="user_id", value=row.get("user_id"), required=True, default=None
            ),
            "category": parse_str(row.get("category"), default=RoadmapPlan.Category.SKINCARE),
            "is_active": parse_bool(
                file_name=file_name, row_no=row_no, column="is_active", value=row.get("is_active"), default=True
            ),
            "version": parse_int(
                file_name=file_name, row_no=row_no, column="version", value=row.get("version"), default=1
            ),
            "meta": parse_json_field(
                file_name=file_name, row_no=row_no, column="meta", value=row.get("meta"), expected="dict"
            ),
            "created_at": parse_dt(
                file_name=file_name, row_no=row_no, column="created_at", value=row.get("created_at"), required=True
            ),
            "updated_at": parse_dt(
                file_name=file_name, row_no=row_no, column="updated_at", value=row.get("updated_at"), required=True
            ),
        }

    def parse_roadmap_step_row(self, row: dict[str, Any], row_no: int, file_name: str) -> dict[str, Any]:
        return {
            "id": parse_int(
                file_name=file_name, row_no=row_no, column="step_id", value=row.get("step_id"), required=True, default=None
            ),
            "plan_id": parse_int(
                file_name=file_name, row_no=row_no, column="plan_id", value=row.get("plan_id"), required=True, default=None
            ),
            "step_index": parse_int(
                file_name=file_name, row_no=row_no, column="step_index", value=row.get("step_index"), default=0
            ),
            "product_type": parse_str(row.get("product_type")),
            "status": parse_str(row.get("status"), default=RoadmapStep.Status.MISSING),
            "recommended_product_id": parse_int(
                file_name=file_name, row_no=row_no, column="recommended_product_id", value=row.get("recommended_product_id"), default=None
            ),
            "suggestions": parse_json_field(
                file_name=file_name, row_no=row_no, column="suggestions", value=row.get("suggestions"), expected="list"
            ),
            "score": parse_float(
                file_name=file_name, row_no=row_no, column="score", value=row.get("score"), default=None
            ),
            "confidence": parse_float(
                file_name=file_name, row_no=row_no, column="confidence", value=row.get("confidence"), default=None
            ),
            "why": parse_json_field(
                file_name=file_name, row_no=row_no, column="why", value=row.get("why"), expected="list"
            ),
            "cadence": parse_str(row.get("cadence"), default=""),
            "created_at": parse_dt(
                file_name=file_name, row_no=row_no, column="created_at", value=row.get("created_at"), required=True
            ),
            "updated_at": parse_dt(
                file_name=file_name, row_no=row_no, column="updated_at", value=row.get("updated_at"), required=True
            ),
        }
    def parse_campaign_budget_row(self, row: dict[str, Any], row_no: int, file_name: str) -> dict[str, Any]:
        return {
            "id": parse_int(
                file_name=file_name, row_no=row_no, column="campaign_id", value=row.get("campaign_id"), required=True, default=None
            ),
            "name": parse_str(row.get("name")),
            "weekly_limit": parse_decimal(
                file_name=file_name, row_no=row_no, column="weekly_limit", value=row.get("weekly_limit"), default=Decimal("0")
            ),
            "weekly_spent": parse_decimal(
                file_name=file_name, row_no=row_no, column="weekly_spent", value=row.get("weekly_spent"), default=Decimal("0")
            ),
            "priority": parse_int(
                file_name=file_name, row_no=row_no, column="priority", value=row.get("priority"), default=100
            ),
            "is_active": parse_bool(
                file_name=file_name, row_no=row_no, column="is_active", value=row.get("is_active"), default=True
            ),
            "allowed_steps": parse_json_field(
                file_name=file_name, row_no=row_no, column="allowed_steps", value=row.get("allowed_steps"), expected="list"
            ),
            "allowed_categories": parse_json_field(
                file_name=file_name,
                row_no=row_no,
                column="allowed_categories",
                value=row.get("allowed_categories"),
                expected="list",
            ),
        }

    def parse_offer_row(self, row: dict[str, Any], row_no: int, file_name: str) -> dict[str, Any]:
        return {
            "id": parse_int(
                file_name=file_name, row_no=row_no, column="offer_id", value=row.get("offer_id"), required=True, default=None
            ),
            "is_active": parse_bool(
                file_name=file_name, row_no=row_no, column="is_active", value=row.get("is_active"), default=True
            ),
            "campaign_id": parse_int(
                file_name=file_name, row_no=row_no, column="campaign_id", value=row.get("campaign_id"), default=None
            ),
            "name": parse_str(row.get("name")),
            "offer_type": parse_str(row.get("offer_type"), default=Offer.Type.DISCOUNT),
            "value": parse_decimal(
                file_name=file_name, row_no=row_no, column="value", value=row.get("value"), default=Decimal("0")
            ),
            "min_total_spend_90d": parse_decimal(
                file_name=file_name,
                row_no=row_no,
                column="min_total_spend_90d",
                value=row.get("min_total_spend_90d"),
                default=Decimal("0"),
            ),
            "allowed_steps": parse_json_field(
                file_name=file_name, row_no=row_no, column="allowed_steps", value=row.get("allowed_steps"), expected="list"
            ),
            "estimated_cost": parse_decimal(
                file_name=file_name, row_no=row_no, column="estimated_cost", value=row.get("estimated_cost"), default=Decimal("0")
            ),
            "cooldown_days": parse_int(
                file_name=file_name, row_no=row_no, column="cooldown_days", value=row.get("cooldown_days"), default=0
            ),
            "expires_in_days": parse_int(
                file_name=file_name, row_no=row_no, column="expires_in_days", value=row.get("expires_in_days"), default=0
            ),
            "allowed_categories": parse_json_field(
                file_name=file_name,
                row_no=row_no,
                column="allowed_categories",
                value=row.get("allowed_categories"),
                expected="list",
            ),
            "allowed_product_types": parse_json_field(
                file_name=file_name,
                row_no=row_no,
                column="allowed_product_types",
                value=row.get("allowed_product_types"),
                expected="list",
            ),
            "target_scope": parse_str(row.get("target_scope"), default="cart"),
            "created_at": parse_dt(
                file_name=file_name, row_no=row_no, column="created_at", value=row.get("created_at"), required=True
            ),
        }

    def parse_offer_assignment_row(self, row: dict[str, Any], row_no: int, file_name: str) -> dict[str, Any]:
        return {
            "id": parse_int(
                file_name=file_name,
                row_no=row_no,
                column="assignment_id",
                value=row.get("assignment_id"),
                required=True,
                default=None,
            ),
            "user_id": parse_int(
                file_name=file_name, row_no=row_no, column="user_id", value=row.get("user_id"), required=True, default=None
            ),
            "offer_id": parse_int(
                file_name=file_name, row_no=row_no, column="offer_id", value=row.get("offer_id"), required=True, default=None
            ),
            "assigned_at": parse_dt(
                file_name=file_name, row_no=row_no, column="assigned_at", value=row.get("assigned_at"), required=True
            ),
            "expires_at": parse_dt(
                file_name=file_name, row_no=row_no, column="expires_at", value=row.get("expires_at"), default=None
            ),
            "reason": parse_json_field(
                file_name=file_name, row_no=row_no, column="reason", value=row.get("reason"), expected="dict"
            ),
            "is_active": parse_bool(
                file_name=file_name, row_no=row_no, column="is_active", value=row.get("is_active"), default=True
            ),
            "is_redeemed": parse_bool(
                file_name=file_name, row_no=row_no, column="is_redeemed", value=row.get("is_redeemed"), default=False
            ),
            "redeemed_transaction_id": parse_int(
                file_name=file_name,
                row_no=row_no,
                column="redeemed_transaction_id",
                value=row.get("redeemed_transaction_id"),
                default=None,
            ),
            "superseded_at": parse_dt(
                file_name=file_name, row_no=row_no, column="superseded_at", value=row.get("superseded_at"), default=None
            ),
            "superseded_by_id": parse_int(
                file_name=file_name, row_no=row_no, column="superseded_by", value=row.get("superseded_by"), default=None
            ),
            "target": parse_json_field(
                file_name=file_name, row_no=row_no, column="target", value=row.get("target"), expected="dict"
            ),
        }

    def parse_offer_event_row(self, row: dict[str, Any], row_no: int, file_name: str) -> dict[str, Any]:
        return {
            "id": parse_int(
                file_name=file_name,
                row_no=row_no,
                column="offer_event_id",
                value=row.get("offer_event_id"),
                required=True,
                default=None,
            ),
            "assignment_id": parse_int(
                file_name=file_name,
                row_no=row_no,
                column="assignment_id",
                value=row.get("assignment_id"),
                required=True,
                default=None,
            ),
            "user_id": parse_int(
                file_name=file_name, row_no=row_no, column="user_id", value=row.get("user_id"), required=True, default=None
            ),
            "offer_id": parse_int(
                file_name=file_name, row_no=row_no, column="offer_id", value=row.get("offer_id"), required=True, default=None
            ),
            "campaign_name": parse_str(row.get("campaign_name")),
            "event_type": parse_str(row.get("event_type")),
            "event_key": parse_optional_str(row.get("event_key")),
            "event_version": parse_int(
                file_name=file_name, row_no=row_no, column="event_version", value=row.get("event_version"), default=1
            ),
            "created_at": parse_dt(
                file_name=file_name, row_no=row_no, column="created_at", value=row.get("created_at"), required=True
            ),
            "request_id": parse_optional_str(row.get("request_id")),
            "context": parse_json_field(
                file_name=file_name, row_no=row_no, column="context", value=row.get("context"), expected="dict"
            ),
        }

    def parse_roadmap_event_row(self, row: dict[str, Any], row_no: int, file_name: str) -> dict[str, Any]:
        return {
            "id": parse_int(
                file_name=file_name,
                row_no=row_no,
                column="roadmap_event_id",
                value=row.get("roadmap_event_id"),
                required=True,
                default=None,
            ),
            "created_at": parse_dt(
                file_name=file_name, row_no=row_no, column="created_at", value=row.get("created_at"), required=True
            ),
            "user_id": parse_int(
                file_name=file_name, row_no=row_no, column="user_id", value=row.get("user_id"), required=True, default=None
            ),
            "plan_id": parse_int(
                file_name=file_name, row_no=row_no, column="plan_id", value=row.get("plan_id"), default=None
            ),
            "step_id": parse_int(
                file_name=file_name, row_no=row_no, column="step_id", value=row.get("step_id"), default=None
            ),
            "event_type": parse_str(row.get("event_type")),
            "request_id": parse_optional_str(row.get("request_id")),
            "context": parse_json_field(
                file_name=file_name, row_no=row_no, column="context", value=row.get("context"), expected="dict"
            ),
        }

    def parse_recommendation_event_row(self, row: dict[str, Any], row_no: int, file_name: str) -> dict[str, Any]:
        return {
            "id": parse_int(
                file_name=file_name, row_no=row_no, column="rec_event_id", value=row.get("rec_event_id"), required=True, default=None
            ),
            "created_at": parse_dt(
                file_name=file_name, row_no=row_no, column="created_at", value=row.get("created_at"), required=True
            ),
            "user_id": parse_int(
                file_name=file_name, row_no=row_no, column="user_id", value=row.get("user_id"), required=True, default=None
            ),
            "action": parse_str(row.get("action")),
            "page": parse_str(row.get("page"), default="home"),
            "section_key": parse_optional_str(row.get("section_key")),
            "request_id": parse_optional_str(row.get("request_id")),
            "product_id": parse_int(
                file_name=file_name, row_no=row_no, column="product_id", value=row.get("product_id"), required=True, default=None
            ),
            "algo_mode": parse_optional_str(row.get("algo_mode")),
            "score": parse_float(
                file_name=file_name, row_no=row_no, column="score", value=row.get("score"), default=None
            ),
            "components": parse_json_field(
                file_name=file_name, row_no=row_no, column="components", value=row.get("components"), expected="dict"
            ),
            "context": parse_json_field(
                file_name=file_name, row_no=row_no, column="context", value=row.get("context"), expected="dict"
            ),
        }

    def import_simple_file(
        self,
        *,
        dataset_dir: Path,
        file_name: str,
        model,
        parser,
        batch_size: int,
    ) -> int:
        rows_loaded = 0
        buffer: list[Any] = []
        path = dataset_dir / file_name
        required = self.REQUIRED_COLUMNS[file_name]

        with transaction.atomic():
            for row_no, row in load_csv(path, file_name=file_name, required_columns=required):
                parsed = parser(row, row_no, file_name)
                buffer.append(model(**parsed))
                if len(buffer) >= batch_size:
                    rows_loaded += bulk_insert(model, buffer, batch_size=batch_size)
                    buffer.clear()
            if buffer:
                rows_loaded += bulk_insert(model, buffer, batch_size=batch_size)

        return rows_loaded

    def import_products(self, dataset_dir: Path, batch_size: int) -> int:
        return self.import_simple_file(
            dataset_dir=dataset_dir,
            file_name="products.csv",
            model=Product,
            parser=self.parse_product_row,
            batch_size=batch_size,
        )

    def import_users(self, dataset_dir: Path, batch_size: int) -> int:
        user_model = get_user_model()
        return self.import_simple_file(
            dataset_dir=dataset_dir,
            file_name="users.csv",
            model=user_model,
            parser=self.parse_user_row,
            batch_size=batch_size,
        )

    def import_customer_profiles(self, dataset_dir: Path, batch_size: int) -> int:
        return self.import_simple_file(
            dataset_dir=dataset_dir,
            file_name="customer_profiles.csv",
            model=CustomerProfile,
            parser=self.parse_customer_profile_row,
            batch_size=batch_size,
        )

    def import_transactions(self, dataset_dir: Path, batch_size: int) -> int:
        return self.import_simple_file(
            dataset_dir=dataset_dir,
            file_name="transactions.csv",
            model=Transaction,
            parser=self.parse_transaction_row,
            batch_size=batch_size,
        )

    def import_transaction_items(self, dataset_dir: Path, batch_size: int) -> int:
        rows_loaded = 0
        buffer: list[TransactionItem] = []
        path = dataset_dir / "transaction_items.csv"
        required = self.REQUIRED_COLUMNS["transaction_items.csv"]

        with transaction.atomic():
            for row_no, row in load_csv(path, file_name="transaction_items.csv", required_columns=required):
                parsed = self.parse_transaction_item_row(row, row_no, "transaction_items.csv")
                parsed.pop("user_id", None)
                buffer.append(TransactionItem(**parsed))
                if len(buffer) >= batch_size:
                    rows_loaded += bulk_insert(TransactionItem, buffer, batch_size=batch_size)
                    buffer.clear()
            if buffer:
                rows_loaded += bulk_insert(TransactionItem, buffer, batch_size=batch_size)
        return rows_loaded

    def import_owned_products(self, dataset_dir: Path, batch_size: int) -> int:
        return self.import_simple_file(
            dataset_dir=dataset_dir,
            file_name="owned_products.csv",
            model=OwnedProduct,
            parser=self.parse_owned_product_row,
            batch_size=batch_size,
        )

    def import_roadmap_plans(self, dataset_dir: Path, batch_size: int) -> int:
        return self.import_simple_file(
            dataset_dir=dataset_dir,
            file_name="roadmap_plans.csv",
            model=RoadmapPlan,
            parser=self.parse_roadmap_plan_row,
            batch_size=batch_size,
        )

    def import_roadmap_steps(self, dataset_dir: Path, batch_size: int) -> int:
        return self.import_simple_file(
            dataset_dir=dataset_dir,
            file_name="roadmap_steps.csv",
            model=RoadmapStep,
            parser=self.parse_roadmap_step_row,
            batch_size=batch_size,
        )

    def import_campaign_budgets(self, dataset_dir: Path, batch_size: int) -> int:
        return self.import_simple_file(
            dataset_dir=dataset_dir,
            file_name="campaign_budgets.csv",
            model=CampaignBudget,
            parser=self.parse_campaign_budget_row,
            batch_size=batch_size,
        )

    def import_offers(self, dataset_dir: Path, batch_size: int) -> int:
        return self.import_simple_file(
            dataset_dir=dataset_dir,
            file_name="offers.csv",
            model=Offer,
            parser=self.parse_offer_row,
            batch_size=batch_size,
        )
    def import_offer_assignments(self, dataset_dir: Path, batch_size: int) -> int:
        rows_loaded = 0
        buffer: list[OfferAssignment] = []
        superseded_links: list[tuple[int, int]] = []
        path = dataset_dir / "offer_assignments.csv"
        required = self.REQUIRED_COLUMNS["offer_assignments.csv"]

        with transaction.atomic():
            for row_no, row in load_csv(path, file_name="offer_assignments.csv", required_columns=required):
                parsed = self.parse_offer_assignment_row(row, row_no, "offer_assignments.csv")
                superseded_by_id = parsed.pop("superseded_by_id")
                if superseded_by_id is not None:
                    superseded_links.append((int(parsed["id"]), int(superseded_by_id)))
                parsed["superseded_by_id"] = None
                buffer.append(OfferAssignment(**parsed))
                if len(buffer) >= batch_size:
                    rows_loaded += bulk_insert(OfferAssignment, buffer, batch_size=batch_size)
                    buffer.clear()
            if buffer:
                rows_loaded += bulk_insert(OfferAssignment, buffer, batch_size=batch_size)

            if superseded_links:
                all_ids = {src for src, _ in superseded_links}
                all_ids.update(dst for _, dst in superseded_links)
                existing_ids = set(
                    OfferAssignment.objects.filter(id__in=all_ids).values_list("id", flat=True)
                )
                missing_refs = sorted(dst for _, dst in superseded_links if dst not in existing_ids)
                if missing_refs:
                    raise CommandError(
                        "offer_assignments.csv contains unknown superseded_by ids: "
                        f"{missing_refs[:10]}"
                    )

                update_buffer: list[OfferAssignment] = []
                for assignment_id, superseded_by_id in superseded_links:
                    update_buffer.append(
                        OfferAssignment(id=assignment_id, superseded_by_id=superseded_by_id)
                    )
                    if len(update_buffer) >= batch_size:
                        OfferAssignment.objects.bulk_update(
                            update_buffer, ["superseded_by"], batch_size=batch_size
                        )
                        update_buffer.clear()
                if update_buffer:
                    OfferAssignment.objects.bulk_update(
                        update_buffer, ["superseded_by"], batch_size=batch_size
                    )

        return rows_loaded

    def import_offer_events(self, dataset_dir: Path, batch_size: int) -> int:
        return self.import_simple_file(
            dataset_dir=dataset_dir,
            file_name="offer_events.csv",
            model=OfferEvent,
            parser=self.parse_offer_event_row,
            batch_size=batch_size,
        )

    def import_roadmap_events(self, dataset_dir: Path, batch_size: int) -> int:
        return self.import_simple_file(
            dataset_dir=dataset_dir,
            file_name="roadmap_events.csv",
            model=RoadmapEvent,
            parser=self.parse_roadmap_event_row,
            batch_size=batch_size,
        )

    def import_recommendation_events(self, dataset_dir: Path, batch_size: int, recommendation_model) -> int:
        return self.import_simple_file(
            dataset_dir=dataset_dir,
            file_name="recommendation_events.csv",
            model=recommendation_model,
            parser=self.parse_recommendation_event_row,
            batch_size=batch_size,
        )

    def run_dry_run(self, *, dataset_dir: Path, recommendation_model) -> dict[str, int]:
        parser_map = {
            "products.csv": self.parse_product_row,
            "users.csv": self.parse_user_row,
            "customer_profiles.csv": self.parse_customer_profile_row,
            "transactions.csv": self.parse_transaction_row,
            "transaction_items.csv": self.parse_transaction_item_row,
            "owned_products.csv": self.parse_owned_product_row,
            "roadmap_plans.csv": self.parse_roadmap_plan_row,
            "roadmap_steps.csv": self.parse_roadmap_step_row,
            "campaign_budgets.csv": self.parse_campaign_budget_row,
            "offers.csv": self.parse_offer_row,
            "offer_assignments.csv": self.parse_offer_assignment_row,
            "offer_events.csv": self.parse_offer_event_row,
            "roadmap_events.csv": self.parse_roadmap_event_row,
            "recommendation_events.csv": self.parse_recommendation_event_row,
        }

        rows_per_file: dict[str, int] = {}
        sample_rows: dict[str, list[dict[str, Any]]] = {file_name: [] for file_name in self.IMPORT_ORDER}
        id_sets: dict[str, set[int]] = {
            "products": set(),
            "transactions": set(),
            "plans": set(),
            "offers": set(),
        }
        aggregates = {
            "products_by_category": Counter(),
            "offer_events_by_type": Counter(),
            "roadmap_events_by_type": Counter(),
            "recommendation_by_action": Counter(),
            "transaction_total_sum": Decimal("0"),
        }

        for file_name in self.IMPORT_ORDER:
            if file_name == "recommendation_events.csv" and recommendation_model is None:
                rows_per_file[file_name] = 0
                continue

            parser = parser_map[file_name]
            required = self.REQUIRED_COLUMNS[file_name]
            path = dataset_dir / file_name
            row_count = 0
            for row_no, row in load_csv(path, file_name=file_name, required_columns=required):
                parsed = parser(row, row_no, file_name)
                row_count += 1
                if len(sample_rows[file_name]) < SAMPLE_VALIDATION_LIMIT:
                    sample_rows[file_name].append(parsed)

                if file_name == "products.csv":
                    id_sets["products"].add(int(parsed["id"]))
                    aggregates["products_by_category"][str(parsed["category"])] += 1
                elif file_name == "transactions.csv":
                    id_sets["transactions"].add(int(parsed["id"]))
                    aggregates["transaction_total_sum"] += parsed["total_amount"] or Decimal("0")
                elif file_name == "roadmap_plans.csv":
                    id_sets["plans"].add(int(parsed["id"]))
                elif file_name == "offers.csv":
                    id_sets["offers"].add(int(parsed["id"]))
                elif file_name == "offer_events.csv":
                    aggregates["offer_events_by_type"][str(parsed["event_type"])] += 1
                elif file_name == "roadmap_events.csv":
                    aggregates["roadmap_events_by_type"][str(parsed["event_type"])] += 1
                elif file_name == "recommendation_events.csv":
                    aggregates["recommendation_by_action"][str(parsed["action"])] += 1

            rows_per_file[file_name] = row_count

        sample_errors = self.validate_samples(sample_rows=sample_rows, id_sets=id_sets)
        if sample_errors:
            raise CommandError("Dry-run sample validations failed:\n- " + "\n- ".join(sample_errors[:20]))

        self.stdout.write("[dry-run] Sample validations: OK")
        self.stdout.write(f"[dry-run] products_by_category: {dict(aggregates['products_by_category'])}")
        self.stdout.write(f"[dry-run] transaction_total_sum: {aggregates['transaction_total_sum']}")
        self.stdout.write(f"[dry-run] offer_events_by_type: {dict(aggregates['offer_events_by_type'])}")
        self.stdout.write(f"[dry-run] roadmap_events_by_type: {dict(aggregates['roadmap_events_by_type'])}")
        if recommendation_model is not None:
            self.stdout.write(
                f"[dry-run] recommendation_by_action: {dict(aggregates['recommendation_by_action'])}"
            )

        return rows_per_file

    def validate_samples(
        self,
        *,
        sample_rows: dict[str, list[dict[str, Any]]],
        id_sets: dict[str, set[int]],
    ) -> list[str]:
        errors: list[str] = []

        for row in sample_rows["transaction_items.csv"]:
            tid = int(row["transaction_id"])
            pid = int(row["product_id"])
            if tid not in id_sets["transactions"]:
                errors.append(f"transaction_items sample references missing transaction_id={tid}")
            if pid not in id_sets["products"]:
                errors.append(f"transaction_items sample references missing product_id={pid}")

        for row in sample_rows["roadmap_steps.csv"]:
            plan_id = int(row["plan_id"])
            if plan_id not in id_sets["plans"]:
                errors.append(f"roadmap_steps sample references missing plan_id={plan_id}")

        for row in sample_rows["offer_assignments.csv"]:
            offer_id = int(row["offer_id"])
            if offer_id not in id_sets["offers"]:
                errors.append(f"offer_assignments sample references missing offer_id={offer_id}")

            target = row.get("target")
            if not isinstance(target, dict):
                continue
            scope = str(target.get("scope") or "")
            category = str(target.get("category") or "")
            product_type = str(target.get("product_type") or "")
            if (
                scope == "product_id"
                and category == "fragrance"
                and product_type in FRAGRANCE_SLOT_TYPES
            ):
                actual = target.get("actual_product_type")
                if not isinstance(actual, str) or not actual.strip():
                    errors.append(
                        "offer_assignments sample missing target.actual_product_type for fragrance slot"
                    )

        for row in sample_rows["roadmap_events.csv"]:
            if row.get("event_type") != RoadmapEvent.Type.STEP_EXPOSED:
                continue
            context = row.get("context")
            if not isinstance(context, dict):
                errors.append("roadmap_events sample STEP_EXPOSED context is not a dict")
                continue
            category = context.get("category")
            if not isinstance(category, str) or not category.strip():
                errors.append("roadmap_events sample STEP_EXPOSED context.category is empty")
            sources = context.get("sources")
            if isinstance(sources, list):
                normalized = {str(x).strip().lower() for x in sources if str(x).strip()}
                if "offers" in normalized and is_nullish(context.get("offer_assignment_id")):
                    errors.append(
                        "roadmap_events sample STEP_EXPOSED with sources=offers missing context.offer_assignment_id"
                    )

        return errors

    def truncate_tables(self, *, recommendation_model) -> None:
        user_model = get_user_model()
        models = [
            OfferEvent,
            RoadmapEvent,
            OfferAssignment,
            TransactionItem,
            OwnedProduct,
            RoadmapStep,
            RoadmapPlan,
            Transaction,
            Offer,
            CampaignBudget,
            CustomerProfile,
            Product,
            user_model,
        ]
        if recommendation_model is not None:
            models.insert(2, recommendation_model)

        with transaction.atomic():
            for model in models:
                deleted, _ = model.objects.all().delete()
                self.stdout.write(f"  cleared {model._meta.label}: {deleted}")

    def print_rows_per_file(self, rows_per_file: dict[str, int]) -> None:
        self.stdout.write("Rows loaded per file:")
        for file_name in self.IMPORT_ORDER:
            value = rows_per_file.get(file_name, 0)
            self.stdout.write(f"  {file_name}: {value}")

    def print_totals(self, *, recommendation_model) -> None:
        self.stdout.write("Totals:")
        totals = [
            ("Product", Product.objects.count()),
            ("Transaction", Transaction.objects.count()),
            ("TransactionItem", TransactionItem.objects.count()),
            ("OwnedProduct", OwnedProduct.objects.count()),
            ("RoadmapPlan", RoadmapPlan.objects.count()),
            ("RoadmapStep", RoadmapStep.objects.count()),
            ("Offer", Offer.objects.count()),
            ("OfferAssignment", OfferAssignment.objects.count()),
            ("OfferEvent", OfferEvent.objects.count()),
            ("RoadmapEvent", RoadmapEvent.objects.count()),
        ]
        if recommendation_model is not None:
            totals.append(("RecommendationEvent", recommendation_model.objects.count()))
        else:
            totals.append(("RecommendationEvent", "skipped"))
        for label, value in totals:
            self.stdout.write(f"  {label}: {value}")

    def print_roadmap_quality_summary(self) -> None:
        output = io.StringIO()
        try:
            call_command("report_roadmap_quality", days=1, stdout=output)
        except Exception as exc:
            raise CommandError(f"Failed to run report_roadmap_quality --days 1: {exc}")

        lines = output.getvalue().splitlines()
        if not lines:
            self.stdout.write("  report_roadmap_quality produced no output.")
            return

        start = None
        for idx, line in enumerate(lines):
            if line.strip() == "## 1) Summary":
                start = idx
                break

        if start is None:
            preview = lines[:SUMMARY_PREVIEW_LINES]
        else:
            end = len(lines)
            for idx in range(start + 1, len(lines)):
                if lines[idx].startswith("## "):
                    end = idx
                    break
            preview = lines[start:end][:SUMMARY_PREVIEW_LINES]

        self.stdout.write("report_roadmap_quality summary preview:")
        for line in preview:
            self.stdout.write(f"  {line}")
