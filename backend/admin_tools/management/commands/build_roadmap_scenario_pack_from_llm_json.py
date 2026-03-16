from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any

from django.core.management.base import BaseCommand, CommandError

from admin_tools.management.commands.generate_roadmap_scenario_pack import (
    CSV_HEADERS,
    OPTIONAL_EMPTY_FILES,
    _decimal_str,
    _iso,
    _json,
    _normalize_step_status,
    _resolve_out_dir,
    _write_csv,
)


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise CommandError(f"--input file does not exist: {path}") from exc
    except json.JSONDecodeError as exc:
        raise CommandError(f"--input contains invalid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise CommandError("--input root JSON must be an object")
    return payload


def _expect_list(value: Any, *, field: str) -> list[Any]:
    if not isinstance(value, list):
        raise CommandError(f"{field} must be an array")
    return value


def _expect_dict(value: Any, *, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise CommandError(f"{field} must be an object")
    return value


def _token(value: Any, *, field: str) -> str:
    token = str(value or "").strip()
    if not token:
        raise CommandError(f"{field} must be a non-empty string")
    return token


def _recommended_planned_step(scenario: dict[str, Any]) -> dict[str, Any]:
    for step in list(scenario.get("steps") or []):
        if _normalize_step_status(step.get("status")) == "recommended":
            return step
    raise CommandError(f"{scenario.get('slug') or 'scenario'}: missing recommended step")


def _validate_payload(payload: dict[str, Any]) -> tuple[str, list[dict[str, Any]], list[dict[str, Any]]]:
    scenario_set = _token(payload.get("scenario_set"), field="scenario_set")
    category = _token(payload.get("category"), field="category").lower()
    shared_catalog = _expect_list(payload.get("shared_catalog"), field="shared_catalog")
    scenarios = _expect_list(payload.get("scenarios"), field="scenarios")
    if not shared_catalog:
        raise CommandError("shared_catalog must not be empty")
    if not scenarios:
        raise CommandError("scenarios must not be empty")

    product_keys: set[str] = set()
    for idx, product in enumerate(shared_catalog, start=1):
        product = _expect_dict(product, field=f"shared_catalog[{idx}]")
        product_key = _token(product.get("product_key"), field=f"shared_catalog[{idx}].product_key")
        if product_key in product_keys:
            raise CommandError(f"duplicate product_key in shared_catalog: {product_key}")
        product_keys.add(product_key)
        product_category = _token(product.get("category"), field=f"shared_catalog[{idx}].category").lower()
        if product_category != category:
            raise CommandError(
                f"shared_catalog[{idx}].category={product_category!r} does not match root category={category!r}"
            )

    scenario_slugs: set[str] = set()
    for idx, scenario in enumerate(scenarios, start=1):
        scenario = _expect_dict(scenario, field=f"scenarios[{idx}]")
        slug = _token(scenario.get("slug"), field=f"scenarios[{idx}].slug")
        if slug in scenario_slugs:
            raise CommandError(f"duplicate scenario slug: {slug}")
        scenario_slugs.add(slug)

        _token(scenario.get("segment"), field=f"scenarios[{idx}].segment")
        expected_next = _token(
            scenario.get("expected_next_product_type"),
            field=f"scenarios[{idx}].expected_next_product_type",
        ).lower()

        transactions = _expect_list(scenario.get("transactions"), field=f"scenarios[{idx}].transactions")
        steps = _expect_list(scenario.get("steps"), field=f"scenarios[{idx}].steps")
        events = _expect_list(scenario.get("events"), field=f"scenarios[{idx}].events")
        if not (1 <= len(transactions) <= 8):
            raise CommandError(f"{slug}: transactions count must be between 1 and 8")
        if not (2 <= len(steps) <= 8):
            raise CommandError(f"{slug}: steps count must be between 2 and 8")
        if len(events) < 2:
            raise CommandError(f"{slug}: events count must be >= 2")

        rec_steps = [step for step in steps if _normalize_step_status(step.get("status")) == "recommended"]
        if len(rec_steps) != 1:
            raise CommandError(f"{slug}: must contain exactly one recommended step")
        planned_step = rec_steps[0]
        planned_step_index = int(planned_step.get("step_index") or 0)
        planned_product_type = _token(planned_step.get("product_type"), field=f"{slug}.planned_step.product_type").lower()
        if expected_next != planned_product_type:
            raise CommandError(
                f"{slug}: expected_next_product_type={expected_next!r} "
                f"does not match recommended step product_type={planned_product_type!r}"
            )

        for tx_idx, tx in enumerate(transactions, start=1):
            tx = _expect_dict(tx, field=f"{slug}.transactions[{tx_idx}]")
            items = _expect_list(tx.get("items"), field=f"{slug}.transactions[{tx_idx}].items")
            if not items:
                raise CommandError(f"{slug}: transactions[{tx_idx}].items must not be empty")
            for item_idx, item in enumerate(items, start=1):
                item = _expect_dict(item, field=f"{slug}.transactions[{tx_idx}].items[{item_idx}]")
                product_key = _token(
                    item.get("product_key"),
                    field=f"{slug}.transactions[{tx_idx}].items[{item_idx}].product_key",
                )
                if product_key not in product_keys:
                    raise CommandError(f"{slug}: unknown product_key in transactions: {product_key}")

        for step_idx, step in enumerate(steps, start=1):
            step = _expect_dict(step, field=f"{slug}.steps[{step_idx}]")
            recommended_product_key = _token(
                step.get("recommended_product_key"),
                field=f"{slug}.steps[{step_idx}].recommended_product_key",
            )
            if recommended_product_key not in product_keys:
                raise CommandError(f"{slug}: unknown recommended_product_key in steps: {recommended_product_key}")

        outcome_tag = _token(scenario.get("outcome_tag"), field=f"{slug}.outcome_tag")
        event_types: list[str] = []
        for event_idx, event in enumerate(events, start=1):
            event = _expect_dict(event, field=f"{slug}.events[{event_idx}]")
            event_type = _token(event.get("event_type"), field=f"{slug}.events[{event_idx}].event_type")
            event_types.append(event_type)
            step_index = event.get("step_index")
            if event_type == "roadmap_plan_refreshed":
                if step_index is not None:
                    raise CommandError(f"{slug}: roadmap_plan_refreshed must have step_index=null")
            else:
                if int(step_index or 0) != planned_step_index:
                    raise CommandError(
                        f"{slug}: event {event_type} must point to recommended step_index={planned_step_index}"
                    )

            context = event.get("context") or {}
            if not isinstance(context, dict):
                raise CommandError(f"{slug}: event context must be an object")
            if event_type == "roadmap_step_completed":
                match_meta = context.get("match_meta") or {}
                if not isinstance(match_meta, dict):
                    raise CommandError(f"{slug}: roadmap_step_completed.match_meta must be an object")
                for field_name in ("recommended_product_key", "purchased_product_key"):
                    product_key = _token(
                        match_meta.get(field_name),
                        field=f"{slug}.roadmap_step_completed.match_meta.{field_name}",
                    )
                    if product_key not in product_keys:
                        raise CommandError(f"{slug}: unknown {field_name} in match_meta: {product_key}")

        if outcome_tag == "completed_exact" and "roadmap_step_completed" not in event_types:
            raise CommandError(f"{slug}: completed_exact requires roadmap_step_completed")
        if outcome_tag == "completed_semantic":
            completed = [ev for ev in events if str(ev.get("event_type")) == "roadmap_step_completed"]
            if not completed:
                raise CommandError(f"{slug}: completed_semantic requires roadmap_step_completed")
            completed_ctx = completed[0].get("context") or {}
            match_meta = completed_ctx.get("match_meta") or {}
            if str(completed_ctx.get("matched_by") or "") != "semantic_content_match":
                raise CommandError(f"{slug}: completed_semantic must use matched_by=semantic_content_match")
            if str(match_meta.get("recommended_product_key") or "") == str(match_meta.get("purchased_product_key") or ""):
                raise CommandError(f"{slug}: semantic_content_match requires different recommended and purchased product keys")
        if outcome_tag == "clicked_no_purchase":
            if "roadmap_step_clicked" not in event_types or "roadmap_step_completed" in event_types:
                raise CommandError(f"{slug}: clicked_no_purchase must include click and exclude completed")
        if outcome_tag == "exposed_no_click":
            if "roadmap_step_exposed" not in event_types:
                raise CommandError(f"{slug}: exposed_no_click must include exposed event")
            if "roadmap_step_clicked" in event_types or "roadmap_step_completed" in event_types:
                raise CommandError(f"{slug}: exposed_no_click must not include click/completed")
        if outcome_tag == "skipped" and "roadmap_step_skipped" not in event_types:
            raise CommandError(f"{slug}: skipped must include roadmap_step_skipped")

    return scenario_set, shared_catalog, scenarios


class Command(BaseCommand):
    help = "Build import-compatible roadmap scenario pack from a validated LLM JSON payload."

    def add_arguments(self, parser):
        parser.add_argument("--input", type=str, required=True)
        parser.add_argument("--out-dir", type=str, required=True)
        parser.add_argument("--replicas", type=int, default=1)
        parser.add_argument("--days-ago-start", type=int, default=75)
        parser.add_argument("--id-base", type=int, default=970000)

    def handle(self, *args, **options):
        input_path = Path(str(options.get("input") or "")).expanduser().resolve()
        out_dir = _resolve_out_dir(str(options.get("out_dir") or ""))
        replicas = int(options.get("replicas") or 1)
        days_ago_start = int(options.get("days_ago_start") or 75)
        id_base = int(options.get("id_base") or 970000)

        if replicas <= 0:
            raise CommandError("--replicas must be > 0")
        if days_ago_start <= 0:
            raise CommandError("--days-ago-start must be > 0")
        if id_base <= 0:
            raise CommandError("--id-base must be > 0")

        payload = _load_json(input_path)
        scenario_set_name, shared_catalog, scenarios = _validate_payload(payload)
        category = str(payload.get("category") or "").strip().lower()

        now_utc = datetime.now(timezone.utc).replace(microsecond=0)
        base_t0 = now_utc - timedelta(days=days_ago_start)
        created_at = base_t0 - timedelta(days=60)
        out_dir.mkdir(parents=True, exist_ok=True)

        product_rows: list[dict[str, Any]] = []
        product_id_by_key: dict[str, int] = {}
        product_price_by_id: dict[int, Decimal] = {}
        for offset, product in enumerate(shared_catalog, start=1):
            product_id = id_base + offset
            product_key = str(product["product_key"]).strip()
            product_id_by_key[product_key] = product_id
            price_dec = Decimal(str(product.get("price") or "0"))
            product_price_by_id[product_id] = price_dec
            product_rows.append({
                "id": product_id,
                "name": str(product.get("name") or product_key),
                "brand": str(product.get("brand") or "Scenario Lab"),
                "price": _decimal_str(price_dec),
                "source_product_id": f"llm::{product_key}",
                "currency": str(product.get("currency") or "KZT"),
                "category": category,
                "product_type": str(product.get("product_type") or ""),
                "concerns": _json(product.get("concerns") or []),
                "attrs": _json(product.get("attrs") or {}),
                "step": "",
                "actives": _json(product.get("actives") or []),
                "flags": _json(product.get("flags") or []),
                "supported_skin_types": _json(product.get("supported_skin_types") or ["normal"]),
                "strength": str(product.get("strength") or "low"),
                "in_stock": "true" if bool(product.get("in_stock", True)) else "false",
                "image_url": str(product.get("image_url") or f"https://example.com/llm/{product_key}.jpg"),
                "image_urls": _json(product.get("image_urls") or [f"https://example.com/llm/{product_key}.jpg"]),
                "description": str(product.get("description") or f"LLM synthetic scenario product for {product_key}."),
                "application_text": str(product.get("application_text") or "Use as directed in the roadmap scenario."),
                "ingredients_inci": str(product.get("ingredients_inci") or ""),
                "volume_raw": str(product.get("volume_raw") or "250 ml"),
                "raw_meta": _json(product.get("raw_meta") or {}),
                "created_at": _iso(created_at),
                "updated_at": _iso(created_at),
            })

        file_rows: dict[str, list[dict[str, Any]]] = {name: [] for name in CSV_HEADERS}
        file_rows["products.csv"] = product_rows

        next_user_id = id_base + 10000
        next_tx_id = id_base + 20000
        next_tx_item_id = id_base + 30000
        next_owned_id = id_base + 40000
        next_plan_id = id_base + 50000
        next_step_id = id_base + 60000
        next_roadmap_event_id = id_base + 70000

        owned_agg: dict[tuple[int, int], dict[str, Any]] = {}
        expected_transition_counts: Counter[str] = Counter()
        outcome_tag_counts: Counter[str] = Counter()
        scenario_instances: list[dict[str, Any]] = []

        for replica_idx in range(replicas):
            for scenario_idx, scenario in enumerate(scenarios, start=1):
                t0 = base_t0 + timedelta(days=(replica_idx * 14) + ((scenario_idx - 1) * 5))
                user_id = next_user_id
                next_user_id += 1
                plan_id = next_plan_id
                next_plan_id += 1
                username = f"llm_{scenario['slug']}_r{replica_idx + 1}"
                profile = dict(scenario.get("profile") or {})
                planned_step = _recommended_planned_step(scenario)
                expected_next = str(scenario.get("expected_next_product_type") or "").strip().lower()

                file_rows["users.csv"].append({
                    "user_id": user_id,
                    "username": username,
                    "segment": str(scenario.get("segment") or scenario["slug"]),
                    "favorite_category": category,
                    "created_at": _iso(t0 - timedelta(days=45)),
                })
                file_rows["customer_profiles.csv"].append({
                    "user_id": user_id,
                    "skin_type": str(profile.get("skin_type") or "normal"),
                    "goals": _json(profile.get("goals") or []),
                    "avoid_flags": _json(profile.get("avoid_flags") or []),
                    "budget": str(profile.get("budget") or "medium"),
                    "hair_profile": _json(profile.get("hair_profile") or {}),
                    "makeup_profile": _json(profile.get("makeup_profile") or {}),
                    "fragrance_profile": _json(profile.get("fragrance_profile") or {}),
                })

                step_id_by_index: dict[int, int] = {}
                latest_product_times: list[tuple[datetime, int]] = []

                for tx_spec in list(scenario.get("transactions") or []):
                    tx_time = t0 + timedelta(days=float(tx_spec.get("offset_days") or 0))
                    transaction_id = next_tx_id
                    next_tx_id += 1
                    total = Decimal("0")
                    for item in list(tx_spec.get("items") or []):
                        product_id = int(product_id_by_key[str(item["product_key"])])
                        quantity = int(item.get("quantity") or 1)
                        unit_price = Decimal(product_price_by_id[product_id])
                        total += unit_price * quantity
                        file_rows["transaction_items.csv"].append({
                            "transaction_item_id": next_tx_item_id,
                            "transaction_id": transaction_id,
                            "user_id": user_id,
                            "product_id": product_id,
                            "quantity": quantity,
                            "unit_price": _decimal_str(unit_price),
                        })
                        next_tx_item_id += 1
                        if tx_time <= t0:
                            latest_product_times.append((tx_time, product_id))
                        aggregate = owned_agg.setdefault((user_id, product_id), {
                            "quantity_total": 0,
                            "acquired_at": tx_time,
                            "last_acquired_at": tx_time,
                        })
                        aggregate["quantity_total"] = int(aggregate["quantity_total"]) + quantity
                        if tx_time < aggregate["acquired_at"]:
                            aggregate["acquired_at"] = tx_time
                        if tx_time > aggregate["last_acquired_at"]:
                            aggregate["last_acquired_at"] = tx_time

                    file_rows["transactions.csv"].append({
                        "transaction_id": transaction_id,
                        "user_id": user_id,
                        "created_at": _iso(tx_time),
                        "total_amount": _decimal_str(total),
                        "channel": str(tx_spec.get("channel") or "offline"),
                        "idempotency_key": f"llm::{scenario_set_name}::{scenario['slug']}::r{replica_idx + 1}::tx{transaction_id}",
                        "pricing_meta": _json({
                            "source": "build_roadmap_scenario_pack_from_llm_json",
                            "scenario_key": scenario["slug"],
                            "scenario_set": scenario_set_name,
                            "replica": replica_idx + 1,
                        }),
                    })

                latest_context_product_ids = [product_id for _ts, product_id in sorted(latest_product_times)[-3:]]
                plan_meta = {
                    "source": "llm_scenario_json",
                    "scenario_set": scenario_set_name,
                    "scenario_key": scenario["slug"],
                    "context": {"post_ctx_product_ids": latest_context_product_ids},
                    "ml": {
                        "mode": "v4_ranking",
                        "decision": "model_used",
                        "model_path": "synthetic://llm_scenario_json",
                        "model_version": f"synthetic_{scenario_set_name}",
                        "model_slot": "active",
                        "selected_feature_set": "full",
                        "planned_target_product_type": str(planned_step.get("product_type") or ""),
                        "planned_target_step_index": int(planned_step.get("step_index") or 0),
                        "predictions": [{"candidate_type": expected_next, "score": 0.93}],
                    },
                }
                file_rows["roadmap_plans.csv"].append({
                    "plan_id": plan_id,
                    "user_id": user_id,
                    "category": category,
                    "is_active": "true",
                    "version": 1,
                    "meta": _json(plan_meta),
                    "created_at": _iso(t0 - timedelta(days=2)),
                    "updated_at": _iso(t0),
                })

                for step in list(scenario.get("steps") or []):
                    step_id = next_step_id
                    next_step_id += 1
                    step_index = int(step["step_index"])
                    step_id_by_index[step_index] = step_id
                    rec_key = str(step.get("recommended_product_key") or "")
                    rec_id = product_id_by_key.get(rec_key)
                    file_rows["roadmap_steps.csv"].append({
                        "step_id": step_id,
                        "plan_id": plan_id,
                        "step_index": step_index,
                        "product_type": str(step.get("product_type") or ""),
                        "status": _normalize_step_status(str(step.get("status") or "")),
                        "recommended_product_id": rec_id or "",
                        "suggestions": _json(([{"product_id": rec_id, "score": float(step.get("score") or 0.0)}] if rec_id else [])),
                        "score": f"{float(step.get('score') or 0.0):.4f}",
                        "confidence": f"{float(step.get('confidence') or 0.0):.4f}",
                        "why": _json(step.get("why") or []),
                        "cadence": str(step.get("cadence") or ""),
                        "created_at": _iso(t0 - timedelta(days=1)),
                        "updated_at": _iso(t0),
                    })

                for event_spec in list(scenario.get("events") or []):
                    step_id = step_id_by_index.get(int(event_spec["step_index"])) if event_spec.get("step_index") is not None else None
                    created_at_event = t0 + timedelta(hours=float(event_spec.get("offset_hours") or 0))
                    context = dict(event_spec.get("context") or {})
                    context["scenario_key"] = scenario["slug"]
                    context["scenario_set"] = scenario_set_name
                    if event_spec["event_type"] != "roadmap_plan_refreshed":
                        context.setdefault("category", category)
                    if event_spec["event_type"] == "roadmap_step_exposed":
                        context.setdefault("sources", ["roadmap_api"])
                    if event_spec["event_type"] == "roadmap_step_completed":
                        match_meta = dict(context.get("match_meta") or {})
                        rec_key = str(match_meta.pop("recommended_product_key", "") or "")
                        pur_key = str(match_meta.pop("purchased_product_key", "") or "")
                        if rec_key:
                            match_meta["recommended_product_id"] = int(product_id_by_key[rec_key])
                        if pur_key:
                            match_meta["purchased_product_id"] = int(product_id_by_key[pur_key])
                        context["match_meta"] = match_meta
                    file_rows["roadmap_events.csv"].append({
                        "roadmap_event_id": next_roadmap_event_id,
                        "created_at": _iso(created_at_event),
                        "user_id": user_id,
                        "plan_id": plan_id,
                        "step_id": step_id or "",
                        "event_type": str(event_spec["event_type"]),
                        "request_id": f"llm::{scenario_set_name}::{scenario['slug']}::r{replica_idx + 1}::evt{next_roadmap_event_id}",
                        "context": _json(context),
                    })
                    next_roadmap_event_id += 1

                expected_transition_counts[expected_next] += 1
                outcome_tag_counts[str(scenario.get("outcome_tag") or "unknown")] += 1
                scenario_instances.append({
                    "scenario_key": scenario["slug"],
                    "replica": replica_idx + 1,
                    "username": username,
                    "user_id": user_id,
                    "plan_id": plan_id,
                    "t0_utc": _iso(t0),
                    "expected_next_product_type": expected_next,
                    "planned_target_product_type": str(planned_step.get("product_type") or ""),
                    "planned_target_step_index": int(planned_step.get("step_index") or 0),
                    "outcome_tag": str(scenario.get("outcome_tag") or "unknown"),
                })

        for (user_id, product_id), aggregate in sorted(owned_agg.items(), key=lambda item: (item[0][0], item[0][1])):
            file_rows["owned_products.csv"].append({
                "owned_product_id": next_owned_id,
                "user_id": user_id,
                "product_id": product_id,
                "quantity_total": int(aggregate["quantity_total"]),
                "is_active": "true",
                "last_acquired_at": _iso(aggregate["last_acquired_at"]),
                "acquired_at": _iso(aggregate["acquired_at"]),
                "source": "llm_scenario_pack",
            })
            next_owned_id += 1

        for file_name in OPTIONAL_EMPTY_FILES:
            file_rows[file_name] = []

        for file_name, headers in CSV_HEADERS.items():
            _write_csv(out_dir / file_name, headers, list(file_rows.get(file_name) or []))

        summary = {
            "generated_at_utc": _iso(now_utc),
            "source_json": str(input_path),
            "scenario_set": scenario_set_name,
            "category": category,
            "replicas": replicas,
            "users_count": len(file_rows["users.csv"]),
            "products_count": len(file_rows["products.csv"]),
            "transactions_count": len(file_rows["transactions.csv"]),
            "transaction_items_count": len(file_rows["transaction_items.csv"]),
            "owned_products_count": len(file_rows["owned_products.csv"]),
            "roadmap_plans_count": len(file_rows["roadmap_plans.csv"]),
            "roadmap_steps_count": len(file_rows["roadmap_steps.csv"]),
            "roadmap_events_count": len(file_rows["roadmap_events.csv"]),
            "expected_next_distribution": {str(k): int(v) for k, v in sorted(expected_transition_counts.items())},
            "outcome_tag_distribution": {str(k): int(v) for k, v in sorted(outcome_tag_counts.items())},
            "scenario_instances": scenario_instances,
        }
        (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        (out_dir / "README.md").write_text(
            "\n".join([
                f"LLM-derived roadmap scenario pack: {scenario_set_name}",
                "",
                f"Source JSON: {input_path}",
                "",
                "Validate:",
                f"python manage.py import_synth_dataset --path {out_dir} --dry-run",
                "",
                "Import into disposable DB:",
                f"python manage.py import_synth_dataset --path {out_dir} --truncate --i-understand-its-destructive",
                "",
                f"Expected next distribution: {summary['expected_next_distribution']}",
                f"Outcome tags: {summary['outcome_tag_distribution']}",
            ]),
            encoding="utf-8",
        )

        self.stdout.write(f"[build_roadmap_scenario_pack_from_llm_json] out_dir={out_dir}")
        self.stdout.write(
            "[build_roadmap_scenario_pack_from_llm_json] "
            f"users={summary['users_count']} tx={summary['transactions_count']} roadmap_events={summary['roadmap_events_count']}"
        )
        self.stdout.write(
            "[build_roadmap_scenario_pack_from_llm_json] "
            f"expected_next_distribution={summary['expected_next_distribution']}"
        )
