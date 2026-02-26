from __future__ import annotations

from bisect import bisect_left
from collections import defaultdict
from datetime import timedelta
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from django.contrib.auth import get_user_model
from django.contrib.auth.hashers import make_password
from django.core.management.base import BaseCommand, CommandError
from django.db import transaction as db_tx

from catalog.models import Product
from offers.models import CampaignBudget, Offer, OfferAssignment, OfferEvent
from recs_analytics.models import RecommendationEvent
from transactions.models import OwnedProduct, Transaction, TransactionItem
from users_app.models import CustomerProfile

try:
    import pandas as pd
except Exception as exc:  # pragma: no cover
    pd = None
    PANDAS_IMPORT_ERROR = exc
else:
    PANDAS_IMPORT_ERROR = None


IMPORT_PAGE = "import"
EVENT_SOURCE = "goldapple_deep"


def _clean_str(v: Any) -> str:
    if pd.isna(v):
        return ""
    return str(v).strip()


def _to_int(v: Any, default: int = 0) -> int:
    try:
        if pd.isna(v):
            return int(default)
        return int(float(v))
    except Exception:
        return int(default)


def _to_decimal(v: Any, default: str = "0.00") -> Decimal:
    if pd.isna(v):
        return Decimal(default)
    try:
        return Decimal(str(v)).quantize(Decimal("0.01"))
    except (InvalidOperation, ValueError, TypeError):
        return Decimal(default)


def _read_xlsx(path: Path) -> "pd.DataFrame":
    return pd.read_excel(path)


def _normalize_orders(df: "pd.DataFrame") -> "pd.DataFrame":
    required = {"order_id", "user_id", "created_at", "source_product_id", "quantity", "unit_price", "channel"}
    missing = required.difference(df.columns)
    if missing:
        raise CommandError(f"orders_items_deep.xlsx missing columns: {sorted(missing)}")

    out = df.copy()
    out["order_id"] = out["order_id"].map(_clean_str)
    out["user_id"] = out["user_id"].map(_clean_str)
    out["source_product_id"] = out["source_product_id"].map(_clean_str)
    out["channel"] = out["channel"].map(_clean_str).replace("", "online")
    out["created_dt"] = pd.to_datetime(out["created_at"], errors="coerce", utc=True)
    out["quantity"] = pd.to_numeric(out["quantity"], errors="coerce").fillna(1).clip(lower=1).round().astype(int)
    out["unit_price"] = pd.to_numeric(out["unit_price"], errors="coerce").fillna(0.0)
    out["line_total"] = out["quantity"] * out["unit_price"]
    out = out[
        (out["order_id"] != "")
        & (out["user_id"] != "")
        & (out["source_product_id"] != "")
        & (out["created_dt"].notna())
    ].copy()
    return out


def _normalize_offer_assignments(df: "pd.DataFrame") -> "pd.DataFrame":
    required = {
        "assignment_id",
        "user_id",
        "assigned_at",
        "campaign_name",
        "offer_type",
        "target_scope",
        "offer_value",
        "estimated_cost",
        "cooldown_days",
        "expires_in_days",
        "label_redeemed",
    }
    missing = required.difference(df.columns)
    if missing:
        raise CommandError(f"offer_assignments_deep.xlsx missing columns: {sorted(missing)}")

    out = df.copy()
    out["assignment_id"] = out["assignment_id"].map(_clean_str)
    out["user_id"] = out["user_id"].map(_clean_str)
    out["campaign_name"] = out["campaign_name"].map(_clean_str).replace("", "import_campaign")
    out["offer_type"] = out["offer_type"].map(_clean_str).str.lower()
    out["target_scope"] = out["target_scope"].map(_clean_str).str.lower()
    out["assigned_dt"] = pd.to_datetime(out["assigned_at"], errors="coerce", utc=True)
    out["offer_value"] = pd.to_numeric(out["offer_value"], errors="coerce").fillna(0.0)
    out["estimated_cost"] = pd.to_numeric(out["estimated_cost"], errors="coerce").fillna(0.0)
    out["cooldown_days"] = pd.to_numeric(out["cooldown_days"], errors="coerce").fillna(0).clip(lower=0).astype(int)
    out["expires_in_days"] = pd.to_numeric(out["expires_in_days"], errors="coerce").fillna(7).clip(lower=0).astype(int)
    out["label_redeemed"] = pd.to_numeric(out["label_redeemed"], errors="coerce").fillna(0).astype(int)
    out["label_redeemed"] = (out["label_redeemed"] > 0).astype(int)

    valid_types = {Offer.Type.DISCOUNT, Offer.Type.POINTS_MULTIPLIER}
    out.loc[~out["offer_type"].isin(valid_types), "offer_type"] = Offer.Type.DISCOUNT

    valid_scopes = {"cart", "category", "product_type", "product_id"}
    out.loc[~out["target_scope"].isin(valid_scopes), "target_scope"] = "cart"

    out = out[(out["assignment_id"] != "") & (out["user_id"] != "") & (out["assigned_dt"].notna())].copy()
    out = out.drop_duplicates("assignment_id", keep="last")
    return out


def _normalize_offer_events(df: "pd.DataFrame") -> "pd.DataFrame":
    required = {"assignment_id", "event_type", "created_at"}
    missing = required.difference(df.columns)
    if missing:
        raise CommandError(f"offer_events_deep.xlsx missing columns: {sorted(missing)}")

    out = df.copy()
    out["assignment_id"] = out["assignment_id"].map(_clean_str)
    out["event_type"] = out["event_type"].map(_clean_str).str.lower()
    out["created_dt"] = pd.to_datetime(out["created_at"], errors="coerce", utc=True)
    allowed = {
        OfferEvent.Type.ASSIGNED,
        OfferEvent.Type.EXPOSED,
        OfferEvent.Type.CLICKED,
        OfferEvent.Type.REDEEMED,
        OfferEvent.Type.EXPIRED,
    }
    out = out[(out["assignment_id"] != "") & (out["event_type"].isin(allowed)) & (out["created_dt"].notna())].copy()
    return out


def _normalize_recs_events(df: "pd.DataFrame") -> "pd.DataFrame":
    required = {"event_time", "event_type", "user_id", "source_product_id", "price", "brand", "category_code"}
    missing = required.difference(df.columns)
    if missing:
        raise CommandError(f"events_recs_deep.xlsx missing columns: {sorted(missing)}")

    out = df.copy()
    out["event_type"] = out["event_type"].map(_clean_str).str.lower()
    out["user_id"] = out["user_id"].map(_clean_str)
    out["source_product_id"] = out["source_product_id"].map(_clean_str)
    out["created_dt"] = pd.to_datetime(out["event_time"], errors="coerce", utc=True)
    out["brand"] = out["brand"].map(_clean_str)
    out["category_code"] = out["category_code"].map(_clean_str).str.lower()
    out["price_num"] = pd.to_numeric(out["price"], errors="coerce")

    action_map = {
        "view": RecommendationEvent.Action.IMPRESSION,
        "cart": RecommendationEvent.Action.ADD_TO_CART,
        "purchase": RecommendationEvent.Action.PURCHASE_ATTRIBUTED,
    }
    out["action"] = out["event_type"].map(action_map)
    out = out[
        (out["user_id"] != "")
        & (out["source_product_id"] != "")
        & (out["created_dt"].notna())
        & (out["action"].notna())
    ].copy()
    return out


class Command(BaseCommand):
    help = "Import GoldApple deep Excel events into Django DB tables."

    def add_arguments(self, parser):
        parser.add_argument(
            "--input_dir",
            default="data/goldapple_user_events_excels_deep",
            help="Directory with events_recs_deep.xlsx, offer_assignments_deep.xlsx, offer_events_deep.xlsx, orders_items_deep.xlsx",
        )
        parser.add_argument(
            "--user_prefix",
            default="ga_",
            help="Username prefix for imported users.",
        )
        parser.add_argument(
            "--user_password",
            default="demo12345",
            help="Password for newly created imported users (local dev only).",
        )
        parser.add_argument(
            "--source",
            default=EVENT_SOURCE,
            help="Source marker written into JSON context.",
        )
        parser.add_argument(
            "--batch_size",
            type=int,
            default=2000,
            help="Batch size for bulk inserts/updates.",
        )
        parser.add_argument(
            "--clear",
            action="store_true",
            help="Delete previously imported users by prefix before import (recommended for full reload).",
        )

    def handle(self, *args, **options):
        if pd is None:
            raise CommandError(f"pandas is required for this command: {PANDAS_IMPORT_ERROR}")

        input_dir = Path(options["input_dir"]).resolve()
        if not input_dir.exists():
            raise CommandError(f"Input dir not found: {input_dir}")

        source = str(options["source"]).strip() or EVENT_SOURCE
        user_prefix = str(options["user_prefix"]).strip()
        user_password = str(options.get("user_password") or "demo12345")
        if not user_prefix:
            raise CommandError("user_prefix cannot be empty")
        batch_size = max(100, int(options["batch_size"] or 2000))

        p_orders = input_dir / "orders_items_deep.xlsx"
        p_offer_assign = input_dir / "offer_assignments_deep.xlsx"
        p_offer_events = input_dir / "offer_events_deep.xlsx"
        p_recs = input_dir / "events_recs_deep.xlsx"
        for p in [p_orders, p_offer_assign, p_offer_events, p_recs]:
            if not p.exists():
                raise CommandError(f"Required file not found: {p}")

        self.stdout.write("Reading XLSX files...")
        orders_df = _normalize_orders(_read_xlsx(p_orders))
        assignments_df = _normalize_offer_assignments(_read_xlsx(p_offer_assign))
        offer_events_df = _normalize_offer_events(_read_xlsx(p_offer_events))
        recs_df = _normalize_recs_events(_read_xlsx(p_recs))

        self.stdout.write(
            f"Loaded rows: orders={len(orders_df)}, assignments={len(assignments_df)}, offer_events={len(offer_events_df)}, recs={len(recs_df)}"
        )

        all_external_users = sorted(
            set(orders_df["user_id"].tolist())
            | set(assignments_df["user_id"].tolist())
            | set(recs_df["user_id"].tolist())
        )
        if not all_external_users:
            raise CommandError("No users found in source files")

        if options.get("clear"):
            User = get_user_model()
            q = User.objects.filter(username__startswith=user_prefix)
            to_delete = q.count()
            if to_delete:
                q.delete()
            self.stdout.write(f"Cleared users by prefix '{user_prefix}': {to_delete}")

        user_id_map = self._ensure_users(
            all_external_users,
            user_prefix=user_prefix,
            user_password=user_password,
            batch_size=batch_size,
        )
        imported_user_ids = list(user_id_map.values())

        product_map = {
            _clean_str(src): int(pid)
            for pid, src in Product.objects.exclude(source_product_id="").values_list("id", "source_product_id")
            if _clean_str(src)
        }
        if not product_map:
            raise CommandError("Catalog is empty or source_product_id is not populated. Import catalog first.")

        orders_df["django_user_id"] = orders_df["user_id"].map(user_id_map)
        orders_df["product_id"] = orders_df["source_product_id"].map(product_map)

        recs_df["django_user_id"] = recs_df["user_id"].map(user_id_map)
        recs_df["product_id"] = recs_df["source_product_id"].map(product_map)

        missing_orders_products = int(orders_df["product_id"].isna().sum())
        missing_recs_products = int(recs_df["product_id"].isna().sum())
        if missing_orders_products or missing_recs_products:
            raise CommandError(
                f"Missing product mapping. orders_missing={missing_orders_products}, recs_missing={missing_recs_products}. "
                f"Make sure catalog source_product_id covers all source_product_id from events."
            )

        orders_df["product_id"] = orders_df["product_id"].astype(int)
        recs_df["product_id"] = recs_df["product_id"].astype(int)
        assignments_df["django_user_id"] = assignments_df["user_id"].map(user_id_map)

        _, user_tx_timeline = self._import_transactions(
            orders_df=orders_df,
            source=source,
            imported_user_ids=imported_user_ids,
            batch_size=batch_size,
        )

        assignment_id_map, assignment_assigned_at_map = self._import_offer_assignments(
            assignments_df=assignments_df,
            source=source,
            batch_size=batch_size,
        )

        redeemed_ts_map = self._redeemed_ts_by_external_assignment(offer_events_df)
        self._attach_redeemed_transactions(
            assignments_df=assignments_df,
            assignment_id_map=assignment_id_map,
            assignment_assigned_at_map=assignment_assigned_at_map,
            redeemed_ts_map=redeemed_ts_map,
            user_tx_timeline=user_tx_timeline,
            batch_size=batch_size,
        )

        self._import_offer_events(
            offer_events_df=offer_events_df,
            assignment_id_map=assignment_id_map,
            source=source,
            batch_size=batch_size,
        )

        self._import_recommendation_events(
            recs_df=recs_df,
            imported_user_ids=imported_user_ids,
            source=source,
            batch_size=batch_size,
        )

        RecommendationEvent.objects.filter(page="tmp_bulk").delete()

        self.stdout.write(self.style.SUCCESS("Import completed."))
        self.stdout.write(
            f"DB counts: users={len(imported_user_ids)}, txns={Transaction.objects.filter(user_id__in=imported_user_ids).count()}, "
            f"txn_items={TransactionItem.objects.filter(transaction__user_id__in=imported_user_ids).count()}, "
            f"offer_assignments={OfferAssignment.objects.filter(user_id__in=imported_user_ids).count()}, "
            f"offer_events={OfferEvent.objects.filter(user_id__in=imported_user_ids).count()}, "
            f"recs={RecommendationEvent.objects.filter(user_id__in=imported_user_ids, page=IMPORT_PAGE, section_key=source).count()}"
        )

    def _ensure_users(
        self,
        external_user_ids: list[str],
        *,
        user_prefix: str,
        user_password: str,
        batch_size: int,
    ) -> dict[str, int]:
        User = get_user_model()
        usernames = [f"{user_prefix}{ext_id}" for ext_id in external_user_ids]
        existing = dict(User.objects.filter(username__in=usernames).values_list("username", "id"))

        to_create = []
        hashed_password = make_password(user_password)
        for username in usernames:
            if username in existing:
                continue
            to_create.append(
                User(
                    username=username,
                    password=hashed_password,
                    is_active=True,
                )
            )
        if to_create:
            User.objects.bulk_create(to_create, batch_size=batch_size)

        user_map = dict(User.objects.filter(username__in=usernames).values_list("username", "id"))
        profile_objs = [CustomerProfile(user_id=uid) for uid in user_map.values()]
        CustomerProfile.objects.bulk_create(profile_objs, batch_size=batch_size, ignore_conflicts=True)

        return {ext_id: int(user_map[f"{user_prefix}{ext_id}"]) for ext_id in external_user_ids}

    def _import_transactions(
        self,
        *,
        orders_df: "pd.DataFrame",
        source: str,
        imported_user_ids: list[int],
        batch_size: int,
    ) -> tuple[dict[str, int], dict[int, list[tuple[Any, int]]]]:
        self.stdout.write("Importing transactions + items + owned products...")

        agg = (
            orders_df.groupby("order_id", as_index=False)
            .agg(
                django_user_id=("django_user_id", "first"),
                created_dt=("created_dt", "min"),
                channel=("channel", "first"),
                total_amount=("line_total", "sum"),
            )
            .copy()
        )
        agg["idempotency_key"] = agg["order_id"].map(lambda x: f"import:{source}:order:{x}")

        idem_keys = agg["idempotency_key"].tolist()
        existing_rows = Transaction.objects.filter(
            user_id__in=imported_user_ids,
            idempotency_key__in=idem_keys,
        ).values("id", "user_id", "idempotency_key")
        tx_by_key = {(int(r["user_id"]), str(r["idempotency_key"])): int(r["id"]) for r in existing_rows}

        to_create = []
        for r in agg.itertuples(index=False):
            key = (int(r.django_user_id), str(r.idempotency_key))
            if key in tx_by_key:
                continue
            to_create.append(
                Transaction(
                    user_id=int(r.django_user_id),
                    idempotency_key=str(r.idempotency_key),
                    total_amount=_to_decimal(r.total_amount),
                    channel=_clean_str(r.channel) or "online",
                )
            )
        if to_create:
            Transaction.objects.bulk_create(to_create, batch_size=batch_size)

        all_rows = Transaction.objects.filter(
            user_id__in=imported_user_ids,
            idempotency_key__in=idem_keys,
        ).values("id", "user_id", "idempotency_key")
        tx_by_key = {(int(r["user_id"]), str(r["idempotency_key"])): int(r["id"]) for r in all_rows}

        order_to_txn: dict[str, int] = {}
        tx_updates = []
        for r in agg.itertuples(index=False):
            key = (int(r.django_user_id), str(r.idempotency_key))
            tx_id = tx_by_key[key]
            order_to_txn[str(r.order_id)] = tx_id
            tx_updates.append(
                Transaction(
                    id=tx_id,
                    total_amount=_to_decimal(r.total_amount),
                    channel=_clean_str(r.channel) or "online",
                    created_at=r.created_dt.to_pydatetime(),
                )
            )
        if tx_updates:
            Transaction.objects.bulk_update(
                tx_updates,
                ["total_amount", "channel", "created_at"],
                batch_size=batch_size,
            )

        txn_ids = list(order_to_txn.values())
        TransactionItem.objects.filter(transaction_id__in=txn_ids).delete()
        item_objs = []
        for r in orders_df.itertuples(index=False):
            tx_id = order_to_txn[str(r.order_id)]
            item_objs.append(
                TransactionItem(
                    transaction_id=tx_id,
                    product_id=int(r.product_id),
                    quantity=_to_int(r.quantity, default=1),
                    unit_price=_to_decimal(r.unit_price),
                )
            )
        if item_objs:
            TransactionItem.objects.bulk_create(item_objs, batch_size=batch_size)

        OwnedProduct.objects.filter(user_id__in=imported_user_ids).delete()
        owned_agg = (
            orders_df.groupby(["django_user_id", "product_id"], as_index=False)
            .agg(
                qty=("quantity", "sum"),
                first_dt=("created_dt", "min"),
                last_dt=("created_dt", "max"),
            )
            .copy()
        )
        owned_objs = []
        for r in owned_agg.itertuples(index=False):
            owned_objs.append(
                OwnedProduct(
                    user_id=int(r.django_user_id),
                    product_id=int(r.product_id),
                    quantity_total=_to_int(r.qty, default=0),
                    is_active=True,
                    acquired_at=r.first_dt.to_pydatetime(),
                    last_acquired_at=r.last_dt.to_pydatetime(),
                    source="import",
                )
            )
        if owned_objs:
            OwnedProduct.objects.bulk_create(owned_objs, batch_size=batch_size)

        user_tx_timeline: dict[int, list[tuple[Any, int]]] = defaultdict(list)
        for r in agg.itertuples(index=False):
            tx_id = order_to_txn[str(r.order_id)]
            user_tx_timeline[int(r.django_user_id)].append((r.created_dt.to_pydatetime(), tx_id))
        for uid in list(user_tx_timeline.keys()):
            user_tx_timeline[uid].sort(key=lambda x: x[0])

        self.stdout.write(
            f"Transactions imported: {len(order_to_txn)}, items={len(item_objs)}, owned_products={len(owned_objs)}"
        )
        return order_to_txn, user_tx_timeline

    def _import_offer_assignments(
        self,
        *,
        assignments_df: "pd.DataFrame",
        source: str,
        batch_size: int,
    ) -> tuple[dict[str, int], dict[str, Any]]:
        self.stdout.write("Importing offer assignments...")

        campaign_names = sorted(set(assignments_df["campaign_name"].tolist()))
        existing_campaigns = {
            str(name): int(cid)
            for cid, name in CampaignBudget.objects.filter(name__in=campaign_names).values_list("id", "name")
        }
        to_create_campaigns = []
        for name in campaign_names:
            if name in existing_campaigns:
                continue
            to_create_campaigns.append(
                CampaignBudget(
                    name=name,
                    is_active=True,
                    priority=100,
                    weekly_limit=Decimal("1000000000.00"),
                    weekly_spent=Decimal("0.00"),
                )
            )
        if to_create_campaigns:
            CampaignBudget.objects.bulk_create(to_create_campaigns, batch_size=batch_size)
        campaign_map = {
            str(name): int(cid)
            for cid, name in CampaignBudget.objects.filter(name__in=campaign_names).values_list("id", "name")
        }

        def offer_key(row: Any) -> tuple[Any, ...]:
            return (
                int(campaign_map[str(row.campaign_name)]),
                str(row.offer_type),
                str(row.target_scope),
                str(_to_decimal(row.offer_value)),
                str(_to_decimal(row.estimated_cost)),
                _to_int(row.cooldown_days, default=0),
                _to_int(row.expires_in_days, default=7),
            )

        existing_offers: dict[tuple[Any, ...], int] = {}
        campaign_ids = list(set(campaign_map.values()))
        for o in Offer.objects.filter(campaign_id__in=campaign_ids).values(
            "id",
            "campaign_id",
            "offer_type",
            "target_scope",
            "value",
            "estimated_cost",
            "cooldown_days",
            "expires_in_days",
        ):
            k = (
                int(o["campaign_id"]),
                str(o["offer_type"]),
                str(o["target_scope"]),
                str(_to_decimal(o["value"])),
                str(_to_decimal(o["estimated_cost"])),
                int(o["cooldown_days"]),
                int(o["expires_in_days"]),
            )
            existing_offers[k] = int(o["id"])

        new_offer_objs = []
        for r in assignments_df.itertuples(index=False):
            k = offer_key(r)
            if k in existing_offers:
                continue
            campaign_name = str(r.campaign_name)
            nm = f"[IMPORTED] {campaign_name} {r.offer_type} {r.target_scope} {str(_to_decimal(r.offer_value))}"
            new_offer_objs.append(
                Offer(
                    campaign_id=int(k[0]),
                    name=nm[:200],
                    offer_type=str(r.offer_type),
                    target_scope=str(r.target_scope),
                    value=_to_decimal(r.offer_value),
                    estimated_cost=_to_decimal(r.estimated_cost),
                    cooldown_days=int(k[5]),
                    expires_in_days=int(k[6]),
                    is_active=True,
                )
            )
            existing_offers[k] = -1
        if new_offer_objs:
            Offer.objects.bulk_create(new_offer_objs, batch_size=batch_size)

        existing_offers = {}
        for o in Offer.objects.filter(campaign_id__in=campaign_ids).values(
            "id",
            "campaign_id",
            "offer_type",
            "target_scope",
            "value",
            "estimated_cost",
            "cooldown_days",
            "expires_in_days",
        ):
            k = (
                int(o["campaign_id"]),
                str(o["offer_type"]),
                str(o["target_scope"]),
                str(_to_decimal(o["value"])),
                str(_to_decimal(o["estimated_cost"])),
                int(o["cooldown_days"]),
                int(o["expires_in_days"]),
            )
            existing_offers[k] = int(o["id"])

        existing_ext_map: dict[str, int] = {}
        for row in OfferAssignment.objects.filter(
            user_id__in=assignments_df["django_user_id"].unique().tolist()
        ).values("id", "reason"):
            reason = row["reason"] or {}
            ext_id = ((reason.get("import") or {}).get("assignment_external_id") if isinstance(reason, dict) else None)
            ext_id = _clean_str(ext_id)
            if ext_id and ext_id not in existing_ext_map:
                existing_ext_map[ext_id] = int(row["id"])

        create_objs = []
        update_objs = []
        for r in assignments_df.itertuples(index=False):
            ext_id = str(r.assignment_id)
            k = offer_key(r)
            offer_id = int(existing_offers[k])
            assigned_dt = r.assigned_dt.to_pydatetime()
            expires_at = assigned_dt + timedelta(days=_to_int(r.expires_in_days, default=7))
            target = {
                "scope": str(r.target_scope),
                "source": source,
                "external_assignment_id": ext_id,
            }
            reason = {
                "source": source,
                "import": {
                    "assignment_external_id": ext_id,
                    "campaign_name": str(r.campaign_name),
                    "label_redeemed": int(r.label_redeemed),
                    "assigned_at": assigned_dt.isoformat(),
                },
            }
            redeemed = bool(_to_int(r.label_redeemed, default=0))

            if ext_id in existing_ext_map:
                update_objs.append(
                    OfferAssignment(
                        id=int(existing_ext_map[ext_id]),
                        user_id=int(r.django_user_id),
                        offer_id=offer_id,
                        target=target,
                        reason=reason,
                        expires_at=expires_at,
                        is_redeemed=redeemed,
                        assigned_at=assigned_dt,
                    )
                )
            else:
                create_objs.append(
                    OfferAssignment(
                        user_id=int(r.django_user_id),
                        offer_id=offer_id,
                        target=target,
                        reason=reason,
                        expires_at=expires_at,
                        is_redeemed=redeemed,
                        assigned_at=assigned_dt,
                    )
                )
        if create_objs:
            OfferAssignment.objects.bulk_create(create_objs, batch_size=batch_size)
            OfferAssignment.objects.bulk_update(create_objs, ["assigned_at"], batch_size=batch_size)
        if update_objs:
            OfferAssignment.objects.bulk_update(
                update_objs,
                ["user", "offer", "target", "reason", "expires_at", "is_redeemed", "assigned_at"],
                batch_size=batch_size,
            )

        assignment_id_map: dict[str, int] = {}
        assignment_assigned_at_map: dict[str, Any] = {}
        for row in OfferAssignment.objects.filter(
            user_id__in=assignments_df["django_user_id"].unique().tolist()
        ).values("id", "assigned_at", "reason"):
            reason = row["reason"] or {}
            ext_id = ((reason.get("import") or {}).get("assignment_external_id") if isinstance(reason, dict) else None)
            ext_id = _clean_str(ext_id)
            if not ext_id:
                continue
            assignment_id_map[ext_id] = int(row["id"])
            assignment_assigned_at_map[ext_id] = row["assigned_at"]

        self.stdout.write(
            f"Offer assignments imported: {len(assignment_id_map)} (created={len(create_objs)}, updated={len(update_objs)})"
        )
        return assignment_id_map, assignment_assigned_at_map

    def _redeemed_ts_by_external_assignment(self, offer_events_df: "pd.DataFrame") -> dict[str, Any]:
        redeemed_rows = offer_events_df[offer_events_df["event_type"] == OfferEvent.Type.REDEEMED].copy()
        if redeemed_rows.empty:
            return {}
        grouped = redeemed_rows.groupby("assignment_id", as_index=False).agg(created_dt=("created_dt", "min"))
        return {str(r.assignment_id): r.created_dt.to_pydatetime() for r in grouped.itertuples(index=False)}

    def _attach_redeemed_transactions(
        self,
        *,
        assignments_df: "pd.DataFrame",
        assignment_id_map: dict[str, int],
        assignment_assigned_at_map: dict[str, Any],
        redeemed_ts_map: dict[str, Any],
        user_tx_timeline: dict[int, list[tuple[Any, int]]],
        batch_size: int,
    ) -> None:
        self.stdout.write("Linking redeemed assignments to transactions...")

        def nearest_txn(user_id: int, target_dt: Any, max_days: int) -> int | None:
            timeline = user_tx_timeline.get(user_id) or []
            if not timeline:
                return None
            times = [t for t, _ in timeline]
            idx = bisect_left(times, target_dt)
            best_id = None
            best_delta = None
            left = max(0, idx - 4)
            right = min(len(timeline), idx + 4)
            for i in range(left, right):
                dt_i, tx_id_i = timeline[i]
                delta = abs(dt_i - target_dt)
                if best_delta is None or delta < best_delta:
                    best_delta = delta
                    best_id = tx_id_i
            if best_delta is None:
                return None
            if best_delta <= timedelta(days=max_days):
                return int(best_id)
            # Fallback: choose first transaction after assignment/redeem time.
            if idx < len(timeline):
                return int(timeline[idx][1])
            return int(best_id) if best_id is not None else None

        updates = []
        for r in assignments_df.itertuples(index=False):
            if int(r.label_redeemed) != 1:
                continue
            ext_id = str(r.assignment_id)
            assignment_id = assignment_id_map.get(ext_id)
            if not assignment_id:
                continue
            assigned_at = assignment_assigned_at_map.get(ext_id)
            if assigned_at is None:
                assigned_at = r.assigned_dt.to_pydatetime()
            target_dt = redeemed_ts_map.get(ext_id) or assigned_at
            max_days = max(3, _to_int(r.expires_in_days, default=7) + 2)
            tx_id = nearest_txn(int(r.django_user_id), target_dt, max_days=max_days)
            updates.append(
                OfferAssignment(
                    id=int(assignment_id),
                    is_redeemed=True,
                    redeemed_transaction_id=tx_id,
                )
            )
        if updates:
            OfferAssignment.objects.bulk_update(updates, ["is_redeemed", "redeemed_transaction_id"], batch_size=batch_size)
        self.stdout.write(f"Redeemed links updated: {len(updates)}")

    def _import_offer_events(
        self,
        *,
        offer_events_df: "pd.DataFrame",
        assignment_id_map: dict[str, int],
        source: str,
        batch_size: int,
    ) -> None:
        self.stdout.write("Importing offer events...")

        assignment_rows = OfferAssignment.objects.filter(id__in=assignment_id_map.values()).values(
            "id",
            "user_id",
            "offer_id",
            "offer__campaign__name",
        )
        assignment_info = {
            int(r["id"]): {
                "user_id": int(r["user_id"]),
                "offer_id": int(r["offer_id"]),
                "campaign_name": _clean_str(r["offer__campaign__name"]) or "none",
            }
            for r in assignment_rows
        }
        ext_to_assignment_id = {ext: int(aid) for ext, aid in assignment_id_map.items()}

        grouped = (
            offer_events_df.groupby(["assignment_id", "event_type"], as_index=False)
            .agg(
                first_dt=("created_dt", "min"),
                last_dt=("created_dt", "max"),
                rows_count=("event_type", "size"),
            )
            .copy()
        )
        grouped["assignment_db_id"] = grouped["assignment_id"].map(ext_to_assignment_id)
        grouped = grouped[grouped["assignment_db_id"].notna()].copy()
        grouped["assignment_db_id"] = grouped["assignment_db_id"].astype(int)

        existing = {
            (int(r["assignment_id"]), str(r["event_type"])): int(r["id"])
            for r in OfferEvent.objects.filter(
                assignment_id__in=grouped["assignment_db_id"].tolist()
            ).values("id", "assignment_id", "event_type")
        }

        create_objs = []
        update_objs = []
        for r in grouped.itertuples(index=False):
            assignment_id = int(r.assignment_db_id)
            et = str(r.event_type)
            info = assignment_info.get(assignment_id)
            if not info:
                continue
            ctx = {
                "source": source,
                "import_rows_count": int(r.rows_count),
                "first_event_at": r.first_dt.to_pydatetime().isoformat(),
                "last_event_at": r.last_dt.to_pydatetime().isoformat(),
            }
            key = (assignment_id, et)
            if key in existing:
                update_objs.append(
                    OfferEvent(
                        id=int(existing[key]),
                        assignment_id=assignment_id,
                        user_id=int(info["user_id"]),
                        offer_id=int(info["offer_id"]),
                        campaign_name=str(info["campaign_name"]),
                        event_type=et,
                        created_at=r.first_dt.to_pydatetime(),
                        context=ctx,
                    )
                )
            else:
                create_objs.append(
                    OfferEvent(
                        assignment_id=assignment_id,
                        user_id=int(info["user_id"]),
                        offer_id=int(info["offer_id"]),
                        campaign_name=str(info["campaign_name"]),
                        event_type=et,
                        context=ctx,
                        created_at=r.first_dt.to_pydatetime(),
                    )
                )
        if create_objs:
            OfferEvent.objects.bulk_create(create_objs, batch_size=batch_size)
            OfferEvent.objects.bulk_update(create_objs, ["created_at", "context"], batch_size=batch_size)
        if update_objs:
            OfferEvent.objects.bulk_update(
                update_objs,
                ["user", "offer", "campaign_name", "created_at", "context"],
                batch_size=batch_size,
            )
        self.stdout.write(f"Offer events imported: {len(create_objs) + len(update_objs)}")

    def _import_recommendation_events(
        self,
        *,
        recs_df: "pd.DataFrame",
        imported_user_ids: list[int],
        source: str,
        batch_size: int,
    ) -> None:
        self.stdout.write("Importing recommendation events...")
        RecommendationEvent.objects.filter(
            user_id__in=imported_user_ids,
            page=IMPORT_PAGE,
            section_key=source,
        ).delete()

        total_rows = len(recs_df)
        created = 0
        for start in range(0, total_rows, batch_size):
            chunk = recs_df.iloc[start : start + batch_size]
            objs = []
            created_ats = []
            for r in chunk.itertuples(index=False):
                ctx = {
                    "source": source,
                    "source_event_type": str(r.event_type),
                    "brand": _clean_str(r.brand),
                    "category_code": _clean_str(r.category_code),
                }
                if not pd.isna(r.price_num):
                    ctx["price"] = float(r.price_num)
                objs.append(
                    RecommendationEvent(
                        user_id=int(r.django_user_id),
                        action=str(r.action),
                        product_id=int(r.product_id),
                        page=IMPORT_PAGE,
                        section_key=source,
                        algo_mode="import",
                        context=ctx,
                    )
                )
                created_ats.append(r.created_dt.to_pydatetime())
            if not objs:
                continue
            with db_tx.atomic():
                RecommendationEvent.objects.bulk_create(objs, batch_size=batch_size)
                for i, obj in enumerate(objs):
                    obj.created_at = created_ats[i]
                RecommendationEvent.objects.bulk_update(objs, ["created_at"], batch_size=batch_size)
            created += len(objs)
            if created % (batch_size * 10) == 0 or created == total_rows:
                self.stdout.write(f"  rec events progress: {created}/{total_rows}")

        self.stdout.write(f"Recommendation events imported: {created}")
