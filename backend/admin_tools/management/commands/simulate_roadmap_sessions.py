from __future__ import annotations

import copy
import json
import random
import time
from collections import Counter, defaultdict
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import date, datetime, time as dt_time, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any
from unittest.mock import patch

from django.conf import settings
from django.contrib.auth import get_user_model
from django.core.management import call_command
from django.core.management.base import BaseCommand, CommandError
from django.db import IntegrityError, close_old_connections, connection
from django.db.models import Count, Max, Sum
from django.test.utils import override_settings
from django.utils import timezone as dj_timezone
from rest_framework.test import APIClient

from admin_tools.sim.sim_utils import (
    clamp_prob,
    make_request_id,
    patched_now,
    sample_poisson,
    weighted_choice,
)
from admin_tools.postgres_sequences import apply_postgres_sequences, inspect_postgres_sequences
from catalog.models import Product
from loyalty.models import LoyaltyAccount, Tier
from offers.models import OfferAssignment
from roadmap_app.fragrance_slots import SLOTS, slot_of_fragrance
from roadmap_app.models import RoadmapEvent, RoadmapStep
from transactions.models import OwnedProduct, Transaction, TransactionItem
from users_app.models import CustomerProfile


CATEGORIES = ["skincare", "haircare", "makeup", "fragrance"]
BASE_CATEGORY_WEIGHTS = {
    "skincare": 0.45,
    "haircare": 0.25,
    "makeup": 0.20,
    "fragrance": 0.10,
}
MIN_IN_STOCK_PER_CATEGORY = 10
MIN_FRAGRANCE_PER_SLOT = 5
ML_READY_POSITIVES = 200
ML_READY_FRAGRANCE_POSITIVES = 30


@dataclass(frozen=True)
class SegmentConfig:
    name: str
    sessions_mult: float
    complete_mult: float
    click_mult: float
    skip_mult: float
    redeem_mult: float
    order_prob: float
    basket_bonus: int


SEGMENT_CONFIGS: dict[str, SegmentConfig] = {
    "new": SegmentConfig(
        name="new",
        sessions_mult=0.75,
        complete_mult=0.75,
        click_mult=0.80,
        skip_mult=1.10,
        redeem_mult=0.80,
        order_prob=0.55,
        basket_bonus=0,
    ),
    "active": SegmentConfig(
        name="active",
        sessions_mult=1.00,
        complete_mult=1.00,
        click_mult=1.00,
        skip_mult=1.00,
        redeem_mult=1.00,
        order_prob=0.65,
        basket_bonus=0,
    ),
    "at_risk": SegmentConfig(
        name="at_risk",
        sessions_mult=0.60,
        complete_mult=0.80,
        click_mult=0.70,
        skip_mult=1.20,
        redeem_mult=0.85,
        order_prob=0.45,
        basket_bonus=0,
    ),
    "vip": SegmentConfig(
        name="vip",
        sessions_mult=1.45,
        complete_mult=1.30,
        click_mult=1.25,
        skip_mult=0.80,
        redeem_mult=1.20,
        order_prob=0.85,
        basket_bonus=1,
    ),
}


@dataclass
class UserState:
    segment: SegmentConfig
    favorite_category: str
    profile: CustomerProfile | None = None


def _safe_int(value: Any) -> int | None:
    try:
        if value is None:
            return None
        return int(value)
    except Exception:
        return None


def _safe_json(resp) -> dict[str, Any]:
    data = getattr(resp, "data", None)
    if isinstance(data, dict):
        return data
    if isinstance(data, list):
        return {"items": data}
    try:
        text = resp.content.decode("utf-8")
    except Exception:
        return {}
    try:
        parsed = json.loads(text)
    except Exception:
        return {}
    return parsed if isinstance(parsed, dict) else {"items": parsed}


class Command(BaseCommand):
    """
    README
    ======
    Offline simulator for realistic roadmap/offer/checkout user sessions.
    All writes go through existing API endpoints via DRF APIClient.

    Base run:
      python manage.py simulate_roadmap_sessions --days 120 --start-date 2025-11-01 --users 5000 --include-ga --seed 42

    Quick smoke:
      python manage.py simulate_roadmap_sessions --days 1 --users 5 --seed 1 --avg-sessions 1.0

    To reach 5k+ STEP_COMPLETED:
      - users >= 5000
      - days >= 90
      - avg-sessions ~0.25..0.4
      - p-complete-next-step >= 0.35
      - max-orders-per-session >= 1
      - p-roadmap-get >= 0.8 (to keep next_step available)

    Reliability notes:
      - DRF throttling is disabled by default via --disable-throttle.
      - HTTP 429 has retry/backoff (3 attempts, exponential + jitter).
      - HTTP >=400 details are logged to reports/sim_errors_<timestamp>.jsonl
    """

    help = "Simulate realistic roadmap/offers/checkout sessions through DRF APIClient with virtual time."

    def add_arguments(self, parser):
        parser.add_argument("--days", type=int, required=True)
        parser.add_argument("--start-date", type=str, default=None)
        parser.add_argument("--users", type=int, required=True)
        parser.add_argument("--include-ga", action="store_true", default=False)
        parser.add_argument("--seed", type=int, default=42)

        parser.add_argument("--avg-sessions", type=float, default=0.25)
        parser.add_argument("--max-orders-per-session", type=int, default=1)
        parser.add_argument("--max-items", type=int, default=3)

        parser.add_argument("--p-roadmap-get", type=float, default=0.9)
        parser.add_argument("--p-next-offer-get", type=float, default=0.7)
        parser.add_argument("--p-step-click", type=float, default=0.15)
        parser.add_argument("--p-step-skip", type=float, default=0.05)
        parser.add_argument("--p-complete-next-step", type=float, default=0.35)
        parser.add_argument("--p-redeem-offer", type=float, default=0.12)

        parser.add_argument("--batch-users", type=int, default=200)
        parser.add_argument("--progress-every", type=int, default=100)
        parser.add_argument("--dry-run", action="store_true", default=False)
        parser.add_argument(
            "--disable-throttle",
            dest="disable_throttle",
            action="store_true",
            default=True,
            help="Disable DRF throttling during simulation run (default: enabled).",
        )
        parser.add_argument(
            "--enable-throttle",
            dest="disable_throttle",
            action="store_false",
            help="Keep normal DRF throttling (opposite of --disable-throttle).",
        )
        parser.add_argument(
            "--max-errors",
            type=int,
            default=1000,
            help="Max number of HTTP errors to persist into JSONL log.",
        )
        parser.add_argument(
            "--stop-on-fatal",
            action="store_true",
            default=False,
            help="Stop command immediately when HTTP 5xx is detected.",
        )

    def handle(self, *args, **options):
        started = time.monotonic()
        rng = random.Random(int(options["seed"]))
        self._request_rng = rng

        days = int(options["days"])
        if days <= 0:
            raise CommandError("--days must be > 0")

        users_n = int(options["users"])
        if users_n <= 0:
            raise CommandError("--users must be > 0")

        max_orders_per_session = int(options["max_orders_per_session"])
        if max_orders_per_session <= 0:
            raise CommandError("--max-orders-per-session must be > 0")

        max_items = int(options["max_items"])
        if max_items <= 0:
            raise CommandError("--max-items must be > 0")

        batch_users = int(options["batch_users"])
        if batch_users <= 0:
            raise CommandError("--batch-users must be > 0")

        progress_every = int(options["progress_every"])
        if progress_every <= 0:
            raise CommandError("--progress-every must be > 0")

        avg_sessions = float(options["avg_sessions"])
        if avg_sessions < 0:
            raise CommandError("--avg-sessions must be >= 0")

        start_date = self._resolve_start_date(options.get("start_date"), days=days)
        include_ga = bool(options["include_ga"])
        dry_run = bool(options["dry_run"])
        disable_throttle = bool(options["disable_throttle"])
        max_errors = int(options["max_errors"])
        stop_on_fatal = bool(options["stop_on_fatal"])
        if max_errors <= 0:
            raise CommandError("--max-errors must be > 0")

        self._max_errors = max_errors
        self._stop_on_fatal = stop_on_fatal
        self._error_counts: Counter[tuple[str, int, str]] = Counter()
        self._error_rows_written = 0
        self._error_log_path: Path | None = None

        p_roadmap_get = clamp_prob(options["p_roadmap_get"])
        p_next_offer_get = clamp_prob(options["p_next_offer_get"])
        p_step_click = clamp_prob(options["p_step_click"])
        p_step_skip = clamp_prob(options["p_step_skip"])
        p_complete_next_step = clamp_prob(options["p_complete_next_step"])
        p_redeem_offer = clamp_prob(options["p_redeem_offer"])

        self.stdout.write(
            "simulate_roadmap_sessions config: "
            f"days={days}, start_date={start_date.isoformat()}, users={users_n}, include_ga={include_ga}, "
            f"seed={int(options['seed'])}, avg_sessions={avg_sessions}, disable_throttle={disable_throttle}"
        )
        self._soft_fix_ownedproduct_sequence()

        product_cache = ProductPoolCache(rng=rng)
        coverage = self._validate_catalog_coverage(product_cache)
        for line in coverage:
            self.stdout.write(line)

        users = self._select_or_create_users(
            users_n=users_n,
            include_ga=include_ga,
            batch_size=batch_users,
        )
        user_ids = [int(u.id) for u in users]
        self._ensure_profiles_and_loyalty(user_ids=user_ids, rng=rng)

        if dry_run:
            est_sessions = float(users_n * days * avg_sessions)
            est_tx = est_sessions * 0.65
            est_roadmap_get = est_sessions * p_roadmap_get
            est_next_offer_get = est_sessions * p_next_offer_get
            est_completions = est_tx * p_complete_next_step
            self.stdout.write(self.style.WARNING("Dry-run: no writes, only estimates and checks."))
            self.stdout.write(
                f"Estimated volume: sessions~{int(est_sessions):,}, transactions~{int(est_tx):,}, "
                f"roadmap_get~{int(est_roadmap_get):,}, next_offer_get~{int(est_next_offer_get):,}, "
                f"step_completed~{int(est_completions):,}"
            )
            return

        base_counts = self._snapshot_counts()
        run_nonce = str(base_counts["transactions"])
        counters: Counter[str] = Counter()
        warnings: list[str] = []
        idem_counter = 0
        processed_users = 0

        runtime_rf = copy.deepcopy(getattr(settings, "REST_FRAMEWORK", {}))
        if disable_throttle:
            runtime_rf["DEFAULT_THROTTLE_CLASSES"] = []
            runtime_rf["DEFAULT_THROTTLE_RATES"] = {}

        self._prepare_error_log()
        try:
            with override_settings(REST_FRAMEWORK=runtime_rf):
                throttle_ctx = (
                    patch("rest_framework.throttling.UserRateThrottle.allow_request", return_value=True)
                    if disable_throttle
                    else nullcontext()
                )
                with throttle_ctx:
                    for batch_start in range(0, len(users), batch_users):
                        batch = users[batch_start : batch_start + batch_users]
                        states = self._build_user_states(
                            users=batch,
                            rng=rng,
                            start_date=start_date,
                        )
                        for user in batch:
                            state = states.get(int(user.id))
                            if not state:
                                continue
                            client = APIClient()
                            client.force_authenticate(user=user)
                            idem_counter = self._simulate_user_days(
                                client=client,
                                user=user,
                                state=state,
                                start_date=start_date,
                                days=days,
                                avg_sessions=avg_sessions,
                                max_orders_per_session=max_orders_per_session,
                                max_items=max_items,
                                p_roadmap_get=p_roadmap_get,
                                p_next_offer_get=p_next_offer_get,
                                p_step_click=p_step_click,
                                p_step_skip=p_step_skip,
                                p_complete_next_step=p_complete_next_step,
                                p_redeem_offer=p_redeem_offer,
                                counters=counters,
                                warnings=warnings,
                                product_cache=product_cache,
                                rng=rng,
                                idem_counter=idem_counter,
                                run_nonce=run_nonce,
                            )
                            processed_users += 1
                            if processed_users % progress_every == 0:
                                elapsed = time.monotonic() - started
                                completed_now = self._count_completed()
                                completed_delta = completed_now - base_counts["roadmap_completed"]
                                self.stdout.write(
                                    f"Progress {processed_users}/{len(users)} users, {elapsed:.1f}s | "
                                    f"tx_created={counters['tx_created']}, offer_exposed={counters['offer_exposed']}, "
                                    f"roadmap_exposed={counters['roadmap_exposed']}, roadmap_clicked={counters['roadmap_clicked']}, "
                                    f"roadmap_skipped={counters['roadmap_skipped']}, roadmap_completed={completed_delta}"
                                )
                        close_old_connections()
        finally:
            self._close_error_log()

        final_counts = self._snapshot_counts()
        deltas = {k: final_counts[k] - base_counts.get(k, 0) for k in final_counts}
        added_completed = deltas.get("roadmap_completed", 0)
        added_completed_fragrance = deltas.get("roadmap_completed_fragrance", 0)

        self.stdout.write(self.style.SUCCESS("Simulation complete."))
        self.stdout.write(
            "Added counts: "
            f"transactions={deltas['transactions']}, transaction_items={deltas['transaction_items']}, "
            f"owned_products={deltas['owned_products']}, roadmap_step_exposed={deltas['roadmap_exposed']}, "
            f"roadmap_step_clicked={deltas['roadmap_clicked']}, roadmap_step_skipped={deltas['roadmap_skipped']}, "
            f"roadmap_step_completed={added_completed}, offer_exposed={deltas['offer_exposed']}, "
            f"offer_clicked={deltas['offer_clicked']}, offer_redeemed={deltas['offer_redeemed']}"
        )
        self.stdout.write(f"Warnings: {len(warnings)}")
        for line in warnings[:10]:
            self.stdout.write(self.style.WARNING(f"- {line}"))
        if len(warnings) > 10:
            self.stdout.write(self.style.WARNING(f"... and {len(warnings) - 10} more"))
        self._print_error_summary()

        self.stdout.write("Running report_roadmap_quality --days 7")
        call_command("report_roadmap_quality", days=7, include_ga=include_ga)
        self.stdout.write("Running report_roadmap_quality --days 30")
        call_command("report_roadmap_quality", days=30, include_ga=include_ga)

        if (
            final_counts["roadmap_completed"] >= ML_READY_POSITIVES
            and final_counts["roadmap_completed_fragrance"] >= ML_READY_FRAGRANCE_POSITIVES
        ):
            self.stdout.write("")
            self.stdout.write("ML next steps:")
            self.stdout.write(
                "python manage.py build_roadmap_ml_dataset --days 180 --include-ga --k 10"
            )
            self.stdout.write(
                "python manage.py train_roadmap_nextstep_model --data-dir data/ml/roadmap_nextstep "
                "--model-dir models/roadmap_next_step_v2"
            )
        else:
            self.stdout.write(
                self.style.WARNING(
                    "ML threshold not reached yet. Need at least "
                    f"{ML_READY_POSITIVES} STEP_COMPLETED total and "
                    f"{ML_READY_FRAGRANCE_POSITIVES} fragrance STEP_COMPLETED."
                )
            )
            self.stdout.write(
                f"Current STEP_COMPLETED={final_counts['roadmap_completed']}, "
                f"fragrance STEP_COMPLETED={final_counts['roadmap_completed_fragrance']}"
            )
            self.stdout.write(
                f"Added in this run: STEP_COMPLETED={added_completed}, "
                f"fragrance STEP_COMPLETED={added_completed_fragrance}"
            )

    def _prepare_error_log(self) -> None:
        reports_dir = (Path(settings.BASE_DIR).parent / "reports").resolve()
        reports_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
        self._error_log_path = reports_dir / f"sim_errors_{ts}.jsonl"
        self._error_log_handle = self._error_log_path.open("a", encoding="utf-8")

    def _close_error_log(self) -> None:
        handle = getattr(self, "_error_log_handle", None)
        if handle:
            try:
                handle.close()
            except Exception:
                pass
        self._error_log_handle = None

    def _extract_first_error_key(self, payload: dict[str, Any]) -> str:
        if not isinstance(payload, dict) or not payload:
            return "__content__"
        details = payload.get("details")
        if isinstance(details, dict):
            msg = details.get("message")
            if isinstance(msg, str) and msg.strip():
                return msg.strip()[:220]
        elif isinstance(details, str) and details.strip():
            return details.strip()[:220]

        message = payload.get("message")
        if isinstance(message, str) and message.strip():
            return message.strip()[:220]

        code = payload.get("code")
        if isinstance(code, str) and code.strip():
            return code.strip()[:220]

        keys = [str(k) for k in payload.keys()]
        keys.sort()
        return keys[0] if keys else "__content__"

    def _log_http_error(
        self,
        *,
        endpoint: str,
        method: str,
        status_code: int,
        request_id: str,
        user_id: int | None,
        request_payload: dict[str, Any] | None,
        response,
        response_payload: dict[str, Any] | None = None,
        force: bool = False,
    ) -> None:
        if not hasattr(self, "_error_counts"):
            self._error_counts = Counter()
        if not hasattr(self, "_error_rows_written"):
            self._error_rows_written = 0
        if not hasattr(self, "_max_errors"):
            self._max_errors = 1000
        if int(status_code) < 400 and not bool(force):
            return

        body: dict[str, Any] = {}
        if isinstance(response_payload, dict):
            body = response_payload
        else:
            body = _safe_json(response)
            if not body:
                try:
                    raw_text = response.content.decode("utf-8", errors="ignore")
                except Exception:
                    raw_text = ""
                body = {"raw": raw_text[:500]}
        first_key = self._extract_first_error_key(body)
        self._error_counts[(str(endpoint), int(status_code), first_key)] += 1

        if self._error_rows_written < int(self._max_errors):
            event = {
                "ts_utc": datetime.now(tz=timezone.utc).isoformat().replace("+00:00", "Z"),
                "endpoint": str(endpoint),
                "method": str(method).upper(),
                "status_code": int(status_code),
                "first_error_key": first_key,
                "request_id": str(request_id),
                "user_id": int(user_id) if user_id is not None else None,
                "request_payload": request_payload if isinstance(request_payload, dict) else {},
                "response": body,
            }
            handle = getattr(self, "_error_log_handle", None)
            if handle:
                handle.write(json.dumps(event, ensure_ascii=False) + "\n")
                handle.flush()
            self._error_rows_written += 1

        if int(status_code) >= 500 and bool(self._stop_on_fatal):
            raise CommandError(
                f"Fatal HTTP {status_code} from {method.upper()} {endpoint}. "
                f"See error log: {self._error_log_path}"
            )

    def _log_checkout_skip_no_eligible(
        self,
        *,
        user_id: int,
        request_id: str,
        request_payload: dict[str, Any],
        assignment_id: int,
        target: dict[str, Any],
    ) -> None:
        body = {
            "code": "skipped",
            "details": {"message": "no_eligible_item_for_assignment"},
            "reason": "no_eligible_item_for_assignment",
            "assignment_id": int(assignment_id),
            "target": target if isinstance(target, dict) else {},
        }
        self._log_http_error(
            endpoint="/api/checkout",
            method="POST",
            status_code=0,
            request_id=request_id,
            user_id=user_id,
            request_payload=request_payload,
            response=None,
            response_payload=body,
            force=True,
        )

    def _print_error_summary(self) -> None:
        if not self._error_counts:
            self.stdout.write("HTTP errors captured: 0")
            return
        self.stdout.write(
            f"HTTP errors captured: {sum(self._error_counts.values())} "
            f"(logged_rows={self._error_rows_written}, max_errors={self._max_errors})"
        )
        if self._error_log_path:
            self.stdout.write(f"Error log JSONL: {self._error_log_path}")
        self.stdout.write("Error summary by (endpoint, status, first_error_key):")
        sorted_rows = sorted(self._error_counts.items(), key=lambda kv: (-kv[1], kv[0]))
        for (endpoint, status, first_key), count in sorted_rows[:20]:
            self.stdout.write(f"  {count} | {endpoint} | {status} | {first_key}")

    def _request_with_retry(
        self,
        *,
        client: APIClient,
        method: str,
        path: str,
        sim_now: datetime,
        request_id: str,
        user_id: int | None,
        payload: dict[str, Any] | None,
        format_value: str | None = None,
    ):
        max_attempts = 3
        attempt = 0
        last_response = None
        while attempt < max_attempts:
            call_now = sim_now + timedelta(milliseconds=attempt)
            with patched_now(call_now):
                if method == "GET":
                    last_response = client.get(path, data=payload or {}, HTTP_X_REQUEST_ID=request_id)
                elif method == "POST":
                    last_response = client.post(
                        path,
                        data=payload or {},
                        format=format_value or "json",
                        HTTP_X_REQUEST_ID=request_id,
                    )
                elif method == "PATCH":
                    last_response = client.patch(
                        path,
                        data=payload or {},
                        format=format_value or "json",
                        HTTP_X_REQUEST_ID=request_id,
                    )
                else:
                    raise ValueError(f"Unsupported method: {method}")
            if int(getattr(last_response, "status_code", 0)) != 429:
                break
            attempt += 1
            if attempt >= max_attempts:
                break
            backoff = (0.05 * (2 ** (attempt - 1))) + float(self._request_rng.random() * 0.03)
            time.sleep(backoff)

        status_code = int(getattr(last_response, "status_code", 0) or 0)
        self._log_http_error(
            endpoint=path,
            method=method,
            status_code=status_code,
            request_id=request_id,
            user_id=user_id,
            request_payload=payload or {},
            response=last_response,
        )
        return last_response

    def _resolve_start_date(self, raw_value: str | None, *, days: int) -> date:
        if raw_value:
            try:
                return date.fromisoformat(str(raw_value))
            except ValueError as exc:
                raise CommandError(f"Invalid --start-date (expected YYYY-MM-DD): {raw_value}") from exc
        today_utc = dj_timezone.now().date()
        return today_utc - timedelta(days=max(1, days))

    def _soft_fix_ownedproduct_sequence(self) -> None:
        if connection.vendor != "postgresql":
            return
        table_name = OwnedProduct._meta.db_table
        rows = inspect_postgres_sequences(tables=[table_name])
        if not rows:
            return
        row = rows[0]
        if row.get("status") == "error":
            self.stdout.write(
                self.style.WARNING(
                    "Could not inspect OwnedProduct sequence drift. "
                    "Run `python manage.py fix_postgres_sequences --dry-run` manually."
                )
            )
            return
        if not row.get("drifted"):
            return
        self.stdout.write(
            self.style.WARNING(
                "Detected out-of-sync PostgreSQL sequence for transactions_ownedproduct. "
                "Applying auto-fix before simulation."
            )
        )
        apply_postgres_sequences(tables=[table_name])

    def _validate_catalog_coverage(self, product_cache: "ProductPoolCache") -> list[str]:
        out: list[str] = []
        counts: dict[str, int] = {}
        for cat in CATEGORIES:
            cnt = int(Product.objects.filter(category=cat, in_stock=True).count())
            counts[cat] = cnt
            if cnt < MIN_IN_STOCK_PER_CATEGORY:
                raise CommandError(
                    f"Catalog coverage too low for category='{cat}': {cnt} in_stock "
                    f"(need >= {MIN_IN_STOCK_PER_CATEGORY})."
                )
        out.append(
            "Catalog in_stock by category: "
            + ", ".join([f"{cat}={counts[cat]}" for cat in CATEGORIES])
        )

        slot_counts = product_cache.ensure_fragrance_slots()
        for slot in SLOTS:
            c = int(len(slot_counts.get(slot, [])))
            if c < MIN_FRAGRANCE_PER_SLOT:
                raise CommandError(
                    f"Fragrance slot coverage too low for slot='{slot}': {c} products "
                    f"(need >= {MIN_FRAGRANCE_PER_SLOT})."
                )
        out.append(
            "Fragrance in_stock by slot: "
            + ", ".join([f"{slot}={len(slot_counts.get(slot, []))}" for slot in SLOTS])
        )
        return out

    def _select_or_create_users(self, *, users_n: int, include_ga: bool, batch_size: int):
        User = get_user_model()
        selected = []
        if include_ga:
            selected = list(User.objects.filter(username__startswith="ga_").order_by("id")[:users_n])
            if len(selected) < users_n:
                need = users_n - len(selected)
                created = self._ensure_named_users(prefix="ga_sim_", count=need, batch_size=batch_size)
                selected.extend(created)
        else:
            selected = self._ensure_named_users(prefix="sim_", count=users_n, batch_size=batch_size)
        selected = selected[:users_n]
        if len(selected) != users_n:
            raise CommandError(f"Could not prepare {users_n} users (got {len(selected)}).")
        self.stdout.write(f"Simulation users prepared: {len(selected)}")
        return selected

    def _ensure_named_users(self, *, prefix: str, count: int, batch_size: int):
        User = get_user_model()
        existing = list(User.objects.filter(username__startswith=prefix).order_by("username", "id")[:count])
        if len(existing) >= count:
            return existing[:count]

        existing_names = set(
            User.objects.filter(username__startswith=prefix).values_list("username", flat=True)
        )
        need = count - len(existing)
        to_create = []
        next_idx = 1
        while len(to_create) < need:
            username = f"{prefix}{next_idx:06d}"
            next_idx += 1
            if username in existing_names:
                continue
            existing_names.add(username)
            to_create.append(User(username=username, is_active=True))
        if to_create:
            self._reset_pk_sequence_if_needed(User)
            try:
                User.objects.bulk_create(to_create, batch_size=batch_size)
            except IntegrityError:
                # Sequence can drift after manual-id imports; fix and retry once.
                self._reset_pk_sequence_if_needed(User)
                User.objects.bulk_create(to_create, batch_size=batch_size)
        created = list(
            User.objects.filter(username__in=[u.username for u in to_create]).order_by("username", "id")
        )
        rows = (existing + created)[:count]
        return rows

    def _reset_pk_sequence_if_needed(self, model_cls):
        if connection.vendor != "postgresql":
            return
        table = model_cls._meta.db_table
        pk_col = model_cls._meta.pk.column
        with connection.cursor() as cursor:
            cursor.execute("SELECT pg_get_serial_sequence(%s, %s)", [table, pk_col])
            seq_row = cursor.fetchone()
            seq_name = seq_row[0] if seq_row else None
            if not seq_name:
                return
            cursor.execute(f"SELECT COALESCE(MAX({pk_col}), 1) FROM {table}")
            max_row = cursor.fetchone()
            max_id = int(max_row[0] or 1) if max_row else 1
            cursor.execute("SELECT setval(%s, %s, true)", [seq_name, max_id])

    def _ensure_profiles_and_loyalty(self, *, user_ids: list[int], rng: random.Random):
        if not user_ids:
            return
        tier = Tier.objects.filter(name="Bronze").first()
        if not tier:
            tier = Tier.objects.create(
                name="Bronze",
                threshold_spend_90d=Decimal("0.00"),
                points_rate=Decimal("1.00"),
            )

        existing_profile_ids = set(
            CustomerProfile.objects.filter(user_id__in=user_ids).values_list("user_id", flat=True)
        )
        new_profiles = [CustomerProfile(user_id=uid) for uid in user_ids if uid not in existing_profile_ids]
        if new_profiles:
            CustomerProfile.objects.bulk_create(new_profiles, batch_size=1000, ignore_conflicts=True)

        existing_loyalty_ids = set(
            LoyaltyAccount.objects.filter(user_id__in=user_ids).values_list("user_id", flat=True)
        )
        new_accounts = [
            LoyaltyAccount(user_id=uid, tier_id=tier.id, points_balance=0)
            for uid in user_ids
            if uid not in existing_loyalty_ids
        ]
        if new_accounts:
            LoyaltyAccount.objects.bulk_create(new_accounts, batch_size=1000, ignore_conflicts=True)

        profiles = list(CustomerProfile.objects.filter(user_id__in=user_ids))
        to_update = []
        for cp in profiles:
            changed = False
            if not cp.skin_type:
                cp.skin_type = rng.choice(
                    [
                        CustomerProfile.SkinType.NORMAL,
                        CustomerProfile.SkinType.COMBINATION,
                        CustomerProfile.SkinType.DRY,
                        CustomerProfile.SkinType.OILY,
                        CustomerProfile.SkinType.SENSITIVE,
                    ]
                )
                changed = True
            if not cp.budget:
                cp.budget = rng.choice(
                    [CustomerProfile.Budget.LOW, CustomerProfile.Budget.MEDIUM, CustomerProfile.Budget.HIGH]
                )
                changed = True
            if not cp.goals:
                cp.goals = rng.sample(["hydration", "acne", "anti_aging", "brightening", "soothing"], k=2)
                changed = True
            if not cp.hair_profile:
                cp.hair_profile = {
                    "hair_type": rng.choice(["straight", "wavy", "curly", "coily"]),
                    "concerns": rng.sample(["frizz", "dryness", "damage", "oiliness"], k=2),
                }
                changed = True
            if not cp.makeup_profile:
                cp.makeup_profile = {
                    "finish_pref": rng.choice(["natural", "matte", "dewy"]),
                    "coverage_pref": rng.choice(["light", "medium", "full"]),
                }
                changed = True
            if not cp.fragrance_profile:
                cp.fragrance_profile = {
                    "liked_families": rng.sample(
                        ["fresh", "floral", "woody", "oriental", "gourmand"],
                        k=2,
                    ),
                    "intensity_pref": rng.choice(["light", "medium", "strong"]),
                }
                changed = True
            if changed:
                to_update.append(cp)
        if to_update:
            CustomerProfile.objects.bulk_update(
                to_update,
                fields=[
                    "skin_type",
                    "budget",
                    "goals",
                    "hair_profile",
                    "makeup_profile",
                    "fragrance_profile",
                    "updated_at",
                ],
                batch_size=1000,
            )

    def _build_user_states(
        self,
        *,
        users: list[Any],
        rng: random.Random,
        start_date: date,
    ) -> dict[int, UserState]:
        user_ids = [int(u.id) for u in users]
        profiles = {
            int(cp.user_id): cp
            for cp in CustomerProfile.objects.filter(user_id__in=user_ids)
        }
        tx_stats_qs = (
            Transaction.objects.filter(user_id__in=user_ids)
            .values("user_id")
            .annotate(
                tx_count=Count("id"),
                total_spend=Sum("total_amount"),
                last_tx_at=Max("created_at"),
            )
        )
        tx_stats: dict[int, dict[str, Any]] = {
            int(row["user_id"]): row for row in tx_stats_qs
        }
        cat_counts_qs = (
            TransactionItem.objects.filter(transaction__user_id__in=user_ids)
            .values("transaction__user_id", "product__category")
            .annotate(cnt=Count("id"))
        )
        category_counts: dict[int, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        for row in cat_counts_qs:
            uid = int(row["transaction__user_id"])
            cat = str(row["product__category"] or "")
            if cat in CATEGORIES:
                category_counts[uid][cat] += int(row["cnt"] or 0)

        out: dict[int, UserState] = {}
        start_dt_utc = datetime.combine(start_date, dt_time.min, tzinfo=timezone.utc)
        for user in users:
            uid = int(user.id)
            stat = tx_stats.get(uid, {})
            tx_count = int(stat.get("tx_count") or 0)
            total_spend = Decimal(str(stat.get("total_spend") or "0"))
            last_tx = stat.get("last_tx_at")
            recency_days = 9999
            if isinstance(last_tx, datetime):
                recency_days = max(0, int((start_dt_utc - last_tx).days))
            if tx_count == 0:
                segment = SEGMENT_CONFIGS["new"]
            elif tx_count >= 25 or total_spend >= Decimal("1500"):
                segment = SEGMENT_CONFIGS["vip"]
            elif recency_days >= 45:
                segment = SEGMENT_CONFIGS["at_risk"]
            else:
                segment = SEGMENT_CONFIGS["active"]

            cp = profiles.get(uid)
            fav = self._infer_favorite_category(
                profile=cp,
                category_counts=category_counts.get(uid, {}),
                rng=rng,
            )
            out[uid] = UserState(segment=segment, favorite_category=fav, profile=cp)
        return out

    def _infer_favorite_category(
        self,
        *,
        profile: CustomerProfile | None,
        category_counts: dict[str, int],
        rng: random.Random,
    ) -> str:
        if category_counts:
            best = sorted(
                category_counts.items(),
                key=lambda kv: (-int(kv[1]), kv[0]),
            )[0][0]
            if best in CATEGORIES:
                return best
        if profile:
            if profile.makeup_profile:
                return "makeup"
            if profile.hair_profile:
                return "haircare"
            if profile.fragrance_profile:
                return "fragrance"
            if profile.goals:
                return "skincare"
        return rng.choice(CATEGORIES)

    def _simulate_user_days(
        self,
        *,
        client: APIClient,
        user: Any,
        state: UserState,
        start_date: date,
        days: int,
        avg_sessions: float,
        max_orders_per_session: int,
        max_items: int,
        p_roadmap_get: float,
        p_next_offer_get: float,
        p_step_click: float,
        p_step_skip: float,
        p_complete_next_step: float,
        p_redeem_offer: float,
        counters: Counter[str],
        warnings: list[str],
        product_cache: "ProductPoolCache",
        rng: random.Random,
        idem_counter: int,
        run_nonce: str,
    ) -> int:
        uid = int(user.id)
        for day_index in range(days):
            day_date = start_date + timedelta(days=day_index)
            lam = max(0.0, avg_sessions * state.segment.sessions_mult)
            sessions = min(2, sample_poisson(rng, lam))
            if sessions <= 0:
                continue
            for session_index in range(sessions):
                session_now = datetime.combine(
                    day_date,
                    dt_time(
                        hour=rng.randint(8, 23),
                        minute=rng.randint(0, 59),
                        second=rng.randint(0, 59),
                    ),
                    tzinfo=timezone.utc,
                )
                idem_counter = self._simulate_single_session(
                    client=client,
                    user_id=uid,
                    day_index=day_index,
                    session_index=session_index,
                    sim_now=session_now,
                    state=state,
                    max_orders_per_session=max_orders_per_session,
                    max_items=max_items,
                    p_roadmap_get=p_roadmap_get,
                    p_next_offer_get=p_next_offer_get,
                    p_step_click=p_step_click,
                    p_step_skip=p_step_skip,
                    p_complete_next_step=p_complete_next_step,
                    p_redeem_offer=p_redeem_offer,
                    counters=counters,
                    warnings=warnings,
                    product_cache=product_cache,
                    rng=rng,
                    idem_counter=idem_counter,
                    run_nonce=run_nonce,
                )
        return idem_counter

    def _simulate_single_session(
        self,
        *,
        client: APIClient,
        user_id: int,
        day_index: int,
        session_index: int,
        sim_now: datetime,
        state: UserState,
        max_orders_per_session: int,
        max_items: int,
        p_roadmap_get: float,
        p_next_offer_get: float,
        p_step_click: float,
        p_step_skip: float,
        p_complete_next_step: float,
        p_redeem_offer: float,
        counters: Counter[str],
        warnings: list[str],
        product_cache: "ProductPoolCache",
        rng: random.Random,
        idem_counter: int,
        run_nonce: str,
    ) -> int:
        request_index = 0

        def rid(suffix: str) -> str:
            nonlocal request_index
            request_index += 1
            return make_request_id(
                day_index=day_index,
                user_id=user_id,
                session_index=session_index,
                event_index=request_index,
                suffix=suffix,
            )

        category = self._choose_category(state=state, rng=rng)
        next_step: dict[str, Any] = {}
        next_step_row: dict[str, Any] = {}
        assignment_id: int | None = None
        assignment_target: dict[str, Any] = {}
        step_skipped = False

        if rng.random() < 0.05:
            refresh_resp = self._api_post(
                client=client,
                path="/api/me/roadmap/refresh",
                body={"category": category},
                sim_now=sim_now,
                request_id=rid("refresh"),
                user_id=user_id,
            )
            if refresh_resp.status_code not in (200, 201):
                self._warn(warnings, f"roadmap/refresh status={refresh_resp.status_code} user_id={user_id}")

        if rng.random() < p_roadmap_get:
            roadmap_resp = self._api_get(
                client=client,
                path="/api/me/roadmap",
                params={"category": category},
                sim_now=sim_now + timedelta(seconds=20),
                request_id=rid("roadmap"),
                user_id=user_id,
            )
            if roadmap_resp.status_code == 200:
                counters["roadmap_exposed"] += 1
                payload = _safe_json(roadmap_resp)
                summary = payload.get("summary") if isinstance(payload.get("summary"), dict) else {}
                next_step = summary.get("next_step") if isinstance(summary.get("next_step"), dict) else {}
                steps = payload.get("steps") if isinstance(payload.get("steps"), list) else []
                if next_step:
                    sid = _safe_int(next_step.get("id"))
                    if sid is not None:
                        next_step_row = next(
                            (x for x in steps if _safe_int((x or {}).get("id")) == sid),
                            {},
                        )
            else:
                self._warn(warnings, f"roadmap/get status={roadmap_resp.status_code} user_id={user_id}")

        if rng.random() < p_next_offer_get:
            next_offer_resp = self._api_get(
                client=client,
                path="/api/me/next-offer",
                params=None,
                sim_now=sim_now + timedelta(seconds=40),
                request_id=rid("nextoffer"),
                user_id=user_id,
            )
            if next_offer_resp.status_code == 200:
                offer_payload = _safe_json(next_offer_resp)
                aid = _safe_int(offer_payload.get("assignment_id"))
                if aid:
                    counters["offer_exposed"] += 1
                    assignment_id = aid
                    assignment_target = (
                        offer_payload.get("target")
                        if isinstance(offer_payload.get("target"), dict)
                        else {}
                    )
                    offer_click_prob = clamp_prob(p_step_click * state.segment.click_mult * 0.6)
                    if rng.random() < offer_click_prob:
                        click_resp = self._api_post(
                            client=client,
                            path="/api/offers/click",
                            body={
                                "assignment_id": int(aid),
                                "context": {"simulator": "roadmap_sessions_v1"},
                            },
                            sim_now=sim_now + timedelta(seconds=45),
                            request_id=rid("offerclick"),
                            user_id=user_id,
                        )
                        if click_resp.status_code == 200:
                            counters["offer_clicked"] += 1
                        else:
                            self._warn(
                                warnings,
                                f"offers/click status={click_resp.status_code} user_id={user_id} assignment_id={aid}",
                            )
            else:
                self._warn(warnings, f"next-offer status={next_offer_resp.status_code} user_id={user_id}")

        step_id = _safe_int(next_step.get("id")) if next_step else None
        click_prob = clamp_prob(p_step_click * state.segment.click_mult)
        skip_prob = clamp_prob(p_step_skip * state.segment.skip_mult)
        if step_id is not None:
            action_roll = rng.random()
            if action_roll < skip_prob:
                patch_resp = self._api_patch(
                    client=client,
                    path=f"/api/me/roadmap/steps/{step_id}",
                    body={"status": "skipped"},
                    sim_now=sim_now + timedelta(seconds=55),
                    request_id=rid("stepskip"),
                    user_id=user_id,
                )
                if patch_resp.status_code == 200:
                    counters["roadmap_skipped"] += 1
                    step_skipped = True
                else:
                    self._warn(
                        warnings,
                        f"roadmap/step skip status={patch_resp.status_code} user_id={user_id} step_id={step_id}",
                    )
            elif action_roll < (skip_prob + click_prob):
                click_resp = self._api_post(
                    client=client,
                    path=f"/api/me/roadmap/steps/{step_id}/click",
                    body={},
                    sim_now=sim_now + timedelta(seconds=58),
                    request_id=rid("stepclick"),
                    user_id=user_id,
                )
                if click_resp.status_code == 200:
                    counters["roadmap_clicked"] += 1
                else:
                    self._warn(
                        warnings,
                        f"roadmap/step click status={click_resp.status_code} user_id={user_id} step_id={step_id}",
                    )

        order_prob = clamp_prob(state.segment.order_prob)
        if rng.random() > order_prob:
            return idem_counter

        orders_count = 1
        if max_orders_per_session > 1 and rng.random() < 0.20:
            orders_count = rng.randint(1, max_orders_per_session)

        for order_idx in range(orders_count):
            next_step_pt = str((next_step_row or next_step).get("product_type") or "").strip()
            can_complete = bool(next_step_pt and not step_skipped)
            complete_prob = clamp_prob(p_complete_next_step * state.segment.complete_mult)
            should_complete = can_complete and (rng.random() < complete_prob)

            items_payload, chosen_product_ids, target_product_id = self._build_checkout_items(
                category=category,
                next_step=next_step,
                next_step_row=next_step_row,
                should_complete=should_complete,
                max_items=max_items,
                basket_bonus=state.segment.basket_bonus,
                rng=rng,
                product_cache=product_cache,
            )
            if not items_payload:
                self._warn(
                    warnings,
                    f"no checkout items user_id={user_id} category={category} should_complete={should_complete}",
                )
                continue
            items_payload = self._sanitize_checkout_items(items_payload, product_cache=product_cache)
            if not items_payload:
                self._warn(
                    warnings,
                    f"checkout items invalid after sanitize user_id={user_id} category={category}",
                )
                continue

            apply_candidate_id = None
            redeem_prob = clamp_prob(p_redeem_offer * state.segment.redeem_mult)
            if assignment_id and rng.random() < redeem_prob:
                apply_candidate_id = int(assignment_id)

            idem_counter += 1
            idem_key = (
                f"sim-{run_nonce}-{day_index}-{user_id}-{session_index}-{order_idx}-{idem_counter}"
            )[:64]
            checkout_request_id = rid("checkout")
            body, apply_assignment_used = self._build_checkout_payload(
                user_id=user_id,
                category=category,
                base_items=items_payload,
                base_product_ids=chosen_product_ids,
                max_items=max_items,
                assignment_id=apply_candidate_id,
                assignment_target=assignment_target,
                idempotency_key=idem_key,
                request_id=checkout_request_id,
                warnings=warnings,
                rng=rng,
                product_cache=product_cache,
            )
            apply_assignment_id = (
                int(body.get("apply_assignment_id"))
                if body.get("apply_assignment_id") is not None
                else None
            )
            if apply_candidate_id and not apply_assignment_used:
                counters["offer_apply_skipped_no_eligible"] += 1

            checkout_resp = self._api_post(
                client=client,
                path="/api/checkout",
                body=body,
                sim_now=sim_now + timedelta(minutes=1, seconds=order_idx * 8),
                request_id=checkout_request_id,
                user_id=user_id,
            )
            if checkout_resp.status_code != 201 and apply_assignment_id:
                idem_counter += 1
                retry_key = (
                    f"sim-retry-{run_nonce}-{day_index}-{user_id}-{session_index}-{order_idx}-{idem_counter}"
                )[:64]
                retry_body = {
                    "channel": "web",
                    "items": items_payload,
                    "idempotency_key": retry_key,
                }
                checkout_resp = self._api_post(
                    client=client,
                    path="/api/checkout",
                    body=retry_body,
                    sim_now=sim_now + timedelta(minutes=1, seconds=order_idx * 8 + 1),
                    request_id=rid("checkoutretry"),
                    user_id=user_id,
                )
            if checkout_resp.status_code == 201:
                counters["tx_created"] += 1
                if apply_assignment_id:
                    counters["offer_redeemed"] += 1
            else:
                counters["checkout_failed"] += 1
                self._warn(
                    warnings,
                    f"checkout status={checkout_resp.status_code} user_id={user_id} body_items={len(items_payload)}",
                )
                continue

            if should_complete and target_product_id:
                counters["roadmap_complete_attempts"] += 1

        return idem_counter

    def _choose_category(self, *, state: UserState, rng: random.Random) -> str:
        weights = dict(BASE_CATEGORY_WEIGHTS)
        fav = state.favorite_category if state.favorite_category in CATEGORIES else "skincare"
        weights[fav] = float(weights.get(fav, 0.0)) + 0.25

        cp = state.profile
        if cp:
            if cp.hair_profile:
                weights["haircare"] += 0.05
            if cp.makeup_profile:
                weights["makeup"] += 0.05
            if cp.fragrance_profile:
                weights["fragrance"] += 0.04
            goals = cp.goals if isinstance(cp.goals, list) else []
            if goals:
                weights["skincare"] += 0.08

        if state.segment.name == "vip":
            weights["fragrance"] += 0.03
            weights["makeup"] += 0.02
        if state.segment.name == "at_risk":
            weights["skincare"] += 0.03

        return weighted_choice(rng=rng, weights=weights, default_key=fav)

    def _build_checkout_items(
        self,
        *,
        category: str,
        next_step: dict[str, Any],
        next_step_row: dict[str, Any],
        should_complete: bool,
        max_items: int,
        basket_bonus: int,
        rng: random.Random,
        product_cache: "ProductPoolCache",
    ) -> tuple[list[dict[str, int]], list[int], int | None]:
        target_pid = None
        step_product_type = str((next_step_row or next_step).get("product_type") or "").strip()
        if should_complete and step_product_type:
            target_pid = self._pick_product_for_next_step(
                category=category,
                step_product_type=step_product_type,
                next_step=next_step,
                next_step_row=next_step_row,
                rng=rng,
                product_cache=product_cache,
            )

        if target_pid is None:
            noise_type = None
            if step_product_type and rng.random() < 0.30:
                noise_type = product_cache.random_other_type(
                    category=category,
                    exclude_product_type=step_product_type,
                    rng=rng,
                )
            target_pid = product_cache.pick_random_product(
                category=category,
                product_type=noise_type,
                rng=rng,
            )
        if target_pid is None:
            return [], [], None

        basket_size = 1 + rng.randint(0, max(0, max_items - 1))
        basket_size = min(max_items, basket_size + max(0, basket_bonus))
        items: list[dict[str, int]] = [{"product": int(target_pid), "quantity": 1}]
        chosen: list[int] = [int(target_pid)]

        while len(items) < basket_size:
            same_category = rng.random() < 0.80
            pick_cat = category if same_category else rng.choice(CATEGORIES)
            pid = product_cache.pick_random_product(category=pick_cat, product_type=None, rng=rng)
            if pid is None or int(pid) in chosen:
                continue
            qty = 1 if rng.random() < 0.90 else 2
            items.append({"product": int(pid), "quantity": qty})
            chosen.append(int(pid))

        return items, chosen, int(target_pid)

    def _load_assignment_target(
        self,
        *,
        user_id: int,
        assignment_id: int,
        fallback_target: dict[str, Any],
    ) -> dict[str, Any]:
        row = (
            OfferAssignment.objects.filter(id=int(assignment_id), user_id=int(user_id))
            .values("target")
            .first()
        )
        db_target = row.get("target") if isinstance(row, dict) else None
        if isinstance(db_target, dict):
            return db_target
        if isinstance(fallback_target, dict):
            return fallback_target
        return {}

    def _pick_eligible_product_for_assignment(
        self,
        *,
        target: dict[str, Any],
        rng: random.Random,
        product_cache: "ProductPoolCache",
    ) -> int | None:
        target = target if isinstance(target, dict) else {}
        scope = str(target.get("scope") or "cart").strip().lower()
        value = target.get("value")
        target_category = str(target.get("category") or "").strip().lower()
        target_pt = str(target.get("product_type") or "").strip().lower()
        slot_target = bool(target_category == "fragrance" and target_pt in SLOTS)

        if scope == "cart":
            return None

        if scope == "product_id":
            target_pid = _safe_int(value)
            if target_pid is None:
                return None
            meta = product_cache.product_meta(int(target_pid))
            if not meta or not bool(meta.get("in_stock")):
                return None
            if target_category and str(meta.get("category") or "") != target_category:
                return None
            return int(target_pid)

        if slot_target and scope in {"product_type", "category"}:
            return product_cache.pick_random_slot_product(target_pt, rng=rng)

        if scope == "product_type":
            wanted = str(value or target_pt).strip().lower()
            if not wanted:
                return None
            if target_category:
                return product_cache.pick_random_product(
                    category=target_category,
                    product_type=wanted,
                    rng=rng,
                )
            return product_cache.pick_random_product_by_type(product_type=wanted, rng=rng)

        if scope == "category":
            wanted_cat = str(value or "").strip().lower() or target_category
            if not wanted_cat:
                return None
            return product_cache.pick_random_product(
                category=wanted_cat,
                product_type=None,
                rng=rng,
            )

        return None

    def _fill_checkout_items_to_max(
        self,
        *,
        items: list[dict[str, int]],
        chosen_product_ids: list[int],
        category: str,
        max_items: int,
        rng: random.Random,
        product_cache: "ProductPoolCache",
    ) -> tuple[list[dict[str, int]], list[int]]:
        tries = 0
        seen = {int(x) for x in chosen_product_ids}
        while len(items) < int(max_items) and tries < int(max_items * 40):
            tries += 1
            same_category = rng.random() < 0.80
            pick_cat = category if same_category else rng.choice(CATEGORIES)
            pid = product_cache.pick_random_product(category=pick_cat, product_type=None, rng=rng)
            if pid is None or int(pid) in seen:
                continue
            qty = 1 if rng.random() < 0.90 else 2
            items.append({"product": int(pid), "quantity": int(qty)})
            chosen_product_ids.append(int(pid))
            seen.add(int(pid))
        return items, chosen_product_ids

    def _build_checkout_payload(
        self,
        *,
        user_id: int,
        category: str,
        base_items: list[dict[str, int]],
        base_product_ids: list[int],
        max_items: int,
        assignment_id: int | None,
        assignment_target: dict[str, Any],
        idempotency_key: str,
        request_id: str,
        warnings: list[str],
        rng: random.Random,
        product_cache: "ProductPoolCache",
    ) -> tuple[dict[str, Any], bool]:
        items = self._sanitize_checkout_items(base_items, product_cache=product_cache)
        chosen = [int(x) for x in base_product_ids if str(x).strip()]
        if not items:
            return {"channel": "web", "items": [], "idempotency_key": idempotency_key}, False

        apply_assignment = False
        target = {}
        scope = "cart"
        if assignment_id is not None:
            target = self._load_assignment_target(
                user_id=user_id,
                assignment_id=int(assignment_id),
                fallback_target=assignment_target,
            )
            scope = str((target or {}).get("scope") or "cart").strip().lower()
            if scope == "cart":
                apply_assignment = True
            else:
                eligible_pid = None
                for _ in range(20):
                    candidate = self._pick_eligible_product_for_assignment(
                        target=target,
                        rng=rng,
                        product_cache=product_cache,
                    )
                    if candidate is not None:
                        eligible_pid = int(candidate)
                        break

                if eligible_pid is None:
                    self._warn(
                        warnings,
                        "checkout apply skipped: no eligible item for assignment "
                        f"user_id={user_id} assignment_id={assignment_id}",
                    )
                    self._log_checkout_skip_no_eligible(
                        user_id=user_id,
                        request_id=request_id,
                        request_payload={
                            "channel": "web",
                            "items": items,
                            "idempotency_key": idempotency_key,
                            "apply_assignment_id": int(assignment_id),
                        },
                        assignment_id=int(assignment_id),
                        target=target,
                    )
                else:
                    qty = 1
                    for row in items:
                        if _safe_int((row or {}).get("product")) == int(eligible_pid):
                            qty = max(1, _safe_int((row or {}).get("quantity")) or 1)
                            break
                    new_items: list[dict[str, int]] = [
                        {"product": int(eligible_pid), "quantity": int(qty)}
                    ]
                    new_chosen: list[int] = [int(eligible_pid)]
                    for row in items:
                        pid = _safe_int((row or {}).get("product"))
                        if pid is None or int(pid) in set(new_chosen):
                            continue
                        q = max(1, _safe_int((row or {}).get("quantity")) or 1)
                        if len(new_items) >= int(max_items):
                            break
                        new_items.append({"product": int(pid), "quantity": int(q)})
                        new_chosen.append(int(pid))
                    items, chosen = self._fill_checkout_items_to_max(
                        items=new_items,
                        chosen_product_ids=new_chosen,
                        category=category,
                        max_items=max_items,
                        rng=rng,
                        product_cache=product_cache,
                    )
                    apply_assignment = True

        if apply_assignment:
            chosen_from_items = [
                int(_safe_int((row or {}).get("product")))
                for row in items
                if _safe_int((row or {}).get("product")) is not None
            ]
            items, chosen = self._fill_checkout_items_to_max(
                items=items,
                chosen_product_ids=chosen_from_items,
                category=category,
                max_items=max_items,
                rng=rng,
                product_cache=product_cache,
            )

        payload: dict[str, Any] = {
            "channel": "web",
            "items": items,
            "idempotency_key": idempotency_key,
        }
        if apply_assignment and assignment_id is not None:
            payload["apply_assignment_id"] = int(assignment_id)
        return payload, bool(apply_assignment)

    def _pick_product_for_next_step(
        self,
        *,
        category: str,
        step_product_type: str,
        next_step: dict[str, Any],
        next_step_row: dict[str, Any],
        rng: random.Random,
        product_cache: "ProductPoolCache",
    ) -> int | None:
        rec_pid = _safe_int((next_step_row or next_step).get("recommended_product_id"))
        if rec_pid is not None:
            meta = product_cache.product_meta(rec_pid)
            if meta and meta["in_stock"] and str(meta["category"]) == category:
                if category == "fragrance":
                    slot = str(meta.get("slot") or "")
                    if slot == step_product_type:
                        return int(rec_pid)
                else:
                    if str(meta["product_type"]) == step_product_type:
                        return int(rec_pid)

        suggestions = (next_step_row or {}).get("suggestions")
        if isinstance(suggestions, list):
            for raw_pid in suggestions[:8]:
                spid = _safe_int(raw_pid)
                if spid is None:
                    continue
                meta = product_cache.product_meta(spid)
                if not meta or not meta["in_stock"] or str(meta["category"]) != category:
                    continue
                if category == "fragrance":
                    if str(meta.get("slot") or "") == step_product_type:
                        return int(spid)
                elif str(meta["product_type"]) == step_product_type:
                    return int(spid)

        if category == "fragrance" and step_product_type in SLOTS:
            return product_cache.pick_random_slot_product(step_product_type, rng=rng)
        return product_cache.pick_random_product(
            category=category,
            product_type=step_product_type,
            rng=rng,
        )

    def _assignment_matches_items(
        self,
        *,
        target: dict[str, Any],
        purchased_product_ids: list[int],
        product_cache: "ProductPoolCache",
    ) -> bool:
        if not purchased_product_ids:
            return False
        scope = str((target or {}).get("scope") or "cart")
        value = (target or {}).get("value")
        category = str((target or {}).get("category") or "").strip()
        target_pt = str((target or {}).get("product_type") or "").strip()

        if scope == "cart":
            return True

        if scope == "product_id":
            target_pid = _safe_int(value)
            if target_pid and int(target_pid) in set(int(x) for x in purchased_product_ids):
                return True
            if target_pt:
                for pid in purchased_product_ids:
                    meta = product_cache.product_meta(int(pid))
                    if not meta:
                        continue
                    if category and str(meta["category"]) != category:
                        continue
                    if category == "fragrance" and target_pt in SLOTS:
                        if str(meta.get("slot") or "") == target_pt:
                            return True
                    elif str(meta["product_type"]) == target_pt:
                        return True
            return False

        if scope == "product_type":
            wanted = str(value or target_pt).strip()
            if not wanted:
                return False
            for pid in purchased_product_ids:
                meta = product_cache.product_meta(int(pid))
                if not meta:
                    continue
                if category and str(meta["category"]) != category:
                    continue
                if category == "fragrance" and wanted in SLOTS:
                    if str(meta.get("slot") or "") == wanted:
                        return True
                elif str(meta["product_type"]) == wanted:
                    return True
            return False

        if scope == "category":
            wanted_cat = category or str(value or "").strip()
            if not wanted_cat:
                return False
            for pid in purchased_product_ids:
                meta = product_cache.product_meta(int(pid))
                if meta and str(meta["category"]) == wanted_cat:
                    return True
            return False

        return False

    def _sanitize_checkout_items(
        self,
        items: list[dict[str, Any]],
        *,
        product_cache: "ProductPoolCache",
    ) -> list[dict[str, int]]:
        out: list[dict[str, int]] = []
        seen: set[int] = set()
        for row in items or []:
            pid = _safe_int((row or {}).get("product"))
            qty = _safe_int((row or {}).get("quantity")) or 1
            if pid is None:
                continue
            if qty <= 0:
                qty = 1
            meta = product_cache.product_meta(int(pid))
            if not meta or not bool(meta.get("in_stock")):
                continue
            if int(pid) in seen:
                continue
            seen.add(int(pid))
            out.append({"product": int(pid), "quantity": int(qty)})
        return out

    def _api_get(
        self,
        *,
        client: APIClient,
        path: str,
        params: dict[str, Any] | None,
        sim_now: datetime,
        request_id: str,
        user_id: int | None = None,
    ):
        return self._request_with_retry(
            client=client,
            method="GET",
            path=path,
            sim_now=sim_now,
            request_id=request_id,
            user_id=user_id,
            payload=params or {},
        )

    def _api_post(
        self,
        *,
        client: APIClient,
        path: str,
        body: dict[str, Any],
        sim_now: datetime,
        request_id: str,
        user_id: int | None = None,
    ):
        return self._request_with_retry(
            client=client,
            method="POST",
            path=path,
            sim_now=sim_now,
            request_id=request_id,
            user_id=user_id,
            payload=body,
            format_value="json",
        )

    def _api_patch(
        self,
        *,
        client: APIClient,
        path: str,
        body: dict[str, Any],
        sim_now: datetime,
        request_id: str,
        user_id: int | None = None,
    ):
        return self._request_with_retry(
            client=client,
            method="PATCH",
            path=path,
            sim_now=sim_now,
            request_id=request_id,
            user_id=user_id,
            payload=body,
            format_value="json",
        )

    def _warn(self, warnings: list[str], text: str):
        if len(warnings) < 200:
            warnings.append(text)

    def _count_completed(self) -> int:
        return int(
            RoadmapEvent.objects.filter(event_type=RoadmapEvent.Type.STEP_COMPLETED).count()
        )

    def _snapshot_counts(self) -> dict[str, int]:
        from offers.models import OfferEvent

        completed_fragrance = (
            RoadmapEvent.objects.filter(event_type=RoadmapEvent.Type.STEP_COMPLETED, plan__category="fragrance")
            .count()
        )
        return {
            "transactions": int(Transaction.objects.count()),
            "transaction_items": int(TransactionItem.objects.count()),
            "owned_products": int(OwnedProduct.objects.count()),
            "roadmap_exposed": int(
                RoadmapEvent.objects.filter(event_type=RoadmapEvent.Type.STEP_EXPOSED).count()
            ),
            "roadmap_clicked": int(
                RoadmapEvent.objects.filter(event_type=RoadmapEvent.Type.STEP_CLICKED).count()
            ),
            "roadmap_skipped": int(
                RoadmapEvent.objects.filter(event_type=RoadmapEvent.Type.STEP_SKIPPED).count()
            ),
            "roadmap_completed": int(
                RoadmapEvent.objects.filter(event_type=RoadmapEvent.Type.STEP_COMPLETED).count()
            ),
            "roadmap_completed_fragrance": int(completed_fragrance),
            "offer_exposed": int(
                OfferEvent.objects.filter(event_type=OfferEvent.Type.EXPOSED).count()
            ),
            "offer_clicked": int(
                OfferEvent.objects.filter(event_type=OfferEvent.Type.CLICKED).count()
            ),
            "offer_redeemed": int(
                OfferEvent.objects.filter(event_type=OfferEvent.Type.REDEEMED).count()
            ),
            "offer_assignments": int(OfferAssignment.objects.count()),
        }


class ProductPoolCache:
    def __init__(self, *, rng: random.Random):
        self.rng = rng
        self._by_cat: dict[str, list[int]] = {}
        self._by_cat_type: dict[tuple[str, str], list[int]] = {}
        self._by_type: dict[str, list[int]] = {}
        self._types_by_cat: dict[str, list[str]] = {}
        self._slot_products: dict[str, list[int]] | None = None
        self._meta_cache: dict[int, dict[str, Any]] = {}

    def ensure_fragrance_slots(self) -> dict[str, list[int]]:
        if self._slot_products is not None:
            return self._slot_products
        slots: dict[str, list[int]] = {slot: [] for slot in SLOTS}
        rows = Product.objects.filter(category="fragrance", in_stock=True).values(
            "id",
            "category",
            "product_type",
            "attrs",
            "raw_meta",
            "in_stock",
        )
        for row in rows.iterator():
            pid = int(row["id"])
            slot = slot_of_fragrance(
                attrs=row.get("attrs") or {},
                raw_meta=row.get("raw_meta") if isinstance(row.get("raw_meta"), dict) else {},
            )
            self._meta_cache[pid] = {
                "category": str(row.get("category") or ""),
                "product_type": str(row.get("product_type") or ""),
                "in_stock": bool(row.get("in_stock")),
                "slot": slot,
            }
            if slot in slots:
                slots[slot].append(pid)
        self._slot_products = slots
        return slots

    def product_meta(self, product_id: int) -> dict[str, Any] | None:
        pid = int(product_id)
        if pid in self._meta_cache:
            return self._meta_cache[pid]
        row = (
            Product.objects.filter(id=pid)
            .values("id", "category", "product_type", "in_stock", "attrs", "raw_meta")
            .first()
        )
        if not row:
            return None
        slot = None
        if str(row.get("category") or "") == "fragrance":
            slot = slot_of_fragrance(
                attrs=row.get("attrs") or {},
                raw_meta=row.get("raw_meta") if isinstance(row.get("raw_meta"), dict) else {},
            )
        meta = {
            "category": str(row.get("category") or ""),
            "product_type": str(row.get("product_type") or ""),
            "in_stock": bool(row.get("in_stock")),
            "slot": slot,
        }
        self._meta_cache[pid] = meta
        return meta

    def _load_category(self, category: str) -> list[int]:
        key = str(category)
        if key not in self._by_cat:
            self._by_cat[key] = list(
                Product.objects.filter(category=key, in_stock=True).values_list("id", flat=True)
            )
        return self._by_cat[key]

    def _load_cat_type(self, category: str, product_type: str) -> list[int]:
        key = (str(category), str(product_type))
        if key not in self._by_cat_type:
            self._by_cat_type[key] = list(
                Product.objects.filter(
                    category=key[0],
                    product_type=key[1],
                    in_stock=True,
                ).values_list("id", flat=True)
            )
        return self._by_cat_type[key]

    def _load_type(self, product_type: str) -> list[int]:
        key = str(product_type)
        if key not in self._by_type:
            self._by_type[key] = list(
                Product.objects.filter(
                    product_type=key,
                    in_stock=True,
                ).values_list("id", flat=True)
            )
        return self._by_type[key]

    def _load_types_by_category(self, category: str) -> list[str]:
        key = str(category)
        if key not in self._types_by_cat:
            self._types_by_cat[key] = [
                str(x)
                for x in Product.objects.filter(category=key, in_stock=True)
                .values_list("product_type", flat=True)
                .distinct()
            ]
        return self._types_by_cat[key]

    def pick_random_product(
        self,
        *,
        category: str,
        product_type: str | None,
        rng: random.Random,
    ) -> int | None:
        rows = []
        if product_type:
            rows = self._load_cat_type(category, product_type)
        if not rows:
            rows = self._load_category(category)
        if not rows:
            return None
        return int(rng.choice(rows))

    def pick_random_product_by_type(self, *, product_type: str, rng: random.Random) -> int | None:
        rows = self._load_type(product_type)
        if not rows:
            return None
        return int(rng.choice(rows))

    def random_other_type(
        self,
        *,
        category: str,
        exclude_product_type: str,
        rng: random.Random,
    ) -> str | None:
        all_types = self._load_types_by_category(category)
        candidates = [x for x in all_types if str(x) != str(exclude_product_type)]
        if not candidates:
            return None
        return str(rng.choice(candidates))

    def pick_random_slot_product(self, slot: str, *, rng: random.Random) -> int | None:
        slots = self.ensure_fragrance_slots()
        rows = slots.get(str(slot), [])
        if not rows:
            return None
        return int(rng.choice(rows))

