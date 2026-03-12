from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from django.conf import settings
from django.core.management.base import BaseCommand, CommandError

from roadmap_app.ml_next_step import (
    _load_model_for_path,
    _predict_with_v4_artifact_from_sources,
    nextstep_model_artifact_summary,
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _resolve_pack_path(raw_path: str) -> Path:
    candidate = Path(str(raw_path)).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    cwd_path = (Path.cwd() / candidate).resolve()
    if cwd_path.exists():
        return cwd_path
    return (_repo_root() / candidate).resolve()


def _parse_json(raw: Any, *, default: Any) -> Any:
    if raw in (None, ""):
        return default
    if isinstance(raw, (dict, list)):
        return raw
    try:
        value = json.loads(str(raw))
    except Exception:
        return default
    if isinstance(default, dict):
        return value if isinstance(value, dict) else {}
    if isinstance(default, list):
        return value if isinstance(value, list) else []
    return value


def _parse_dt(raw: Any) -> datetime | None:
    text = str(raw or "").strip()
    if not text:
        return None
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).astimezone(timezone.utc)
    except Exception:
        return None


def _to_int(raw: Any, *, default: int = 0) -> int:
    try:
        return int(raw)
    except Exception:
        return int(default)


def _to_float(raw: Any, *, default: float = 0.0) -> float:
    try:
        return float(raw)
    except Exception:
        return float(default)


def _rate(numerator: int, denominator: int) -> float:
    if int(denominator) <= 0:
        return 0.0
    return round(float(numerator) / float(denominator), 6)


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _load_pack(pack_path: Path) -> dict[str, Any]:
    summary_path = pack_path / "summary.json"
    if not summary_path.exists():
        raise CommandError(f"summary.json not found under {pack_path}")

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    required = [
        "products.csv",
        "customer_profiles.csv",
        "transactions.csv",
        "transaction_items.csv",
        "roadmap_plans.csv",
        "roadmap_steps.csv",
        "roadmap_events.csv",
    ]
    missing = [name for name in required if not (pack_path / name).exists()]
    if missing:
        raise CommandError(f"Scenario pack is incomplete: missing {', '.join(missing)}")

    product_rows = []
    products_by_id: dict[int, dict[str, Any]] = {}
    for row in _read_csv_rows(pack_path / "products.csv"):
        product = {
            "id": _to_int(row.get("id")),
            "category": str(row.get("category") or "").strip().lower(),
            "product_type": str(row.get("product_type") or "").strip().lower(),
            "concerns": _parse_json(row.get("concerns"), default=[]),
            "actives": _parse_json(row.get("actives"), default=[]),
            "flags": _parse_json(row.get("flags"), default=[]),
            "supported_skin_types": _parse_json(row.get("supported_skin_types"), default=[]),
            "attrs": _parse_json(row.get("attrs"), default={}),
            "ingredients_inci": str(row.get("ingredients_inci") or ""),
            "raw_meta": _parse_json(row.get("raw_meta"), default={}),
        }
        product_rows.append(product)
        if int(product["id"]) > 0:
            products_by_id[int(product["id"])] = product

    profiles_by_user_id: dict[int, dict[str, Any]] = {}
    for row in _read_csv_rows(pack_path / "customer_profiles.csv"):
        user_id = _to_int(row.get("user_id"))
        profiles_by_user_id[user_id] = {
            "user_id": user_id,
            "skin_type": str(row.get("skin_type") or ""),
            "goals": _parse_json(row.get("goals"), default=[]),
            "avoid_flags": _parse_json(row.get("avoid_flags"), default=[]),
            "budget": str(row.get("budget") or ""),
            "hair_profile": _parse_json(row.get("hair_profile"), default={}),
            "makeup_profile": _parse_json(row.get("makeup_profile"), default={}),
            "fragrance_profile": _parse_json(row.get("fragrance_profile"), default={}),
        }

    tx_by_id: dict[int, dict[str, Any]] = {}
    for row in _read_csv_rows(pack_path / "transactions.csv"):
        tx_id = _to_int(row.get("transaction_id"))
        tx_by_id[tx_id] = {
            "transaction_id": tx_id,
            "user_id": _to_int(row.get("user_id")),
            "created_at": _parse_dt(row.get("created_at")),
            "total_amount": _to_float(row.get("total_amount")),
            "channel": str(row.get("channel") or ""),
        }

    items_by_user_id: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in _read_csv_rows(pack_path / "transaction_items.csv"):
        tx_id = _to_int(row.get("transaction_id"))
        tx = tx_by_id.get(tx_id)
        if not tx or tx.get("created_at") is None:
            continue
        product = products_by_id.get(_to_int(row.get("product_id")))
        if not product:
            continue
        user_id = _to_int(row.get("user_id")) or int(tx["user_id"])
        items_by_user_id[user_id].append(
            {
                "ts": tx["created_at"],
                "tx_id": tx_id,
                "tx_total": float(tx["total_amount"]),
                "category": str(product.get("category") or ""),
                "product_type": str(product.get("product_type") or ""),
                "concerns": list(product.get("concerns") or []),
                "actives": list(product.get("actives") or []),
                "flags": list(product.get("flags") or []),
                "supported_skin_types": list(product.get("supported_skin_types") or []),
                "attrs": dict(product.get("attrs") or {}),
                "ingredients_inci": str(product.get("ingredients_inci") or ""),
                "raw_meta": dict(product.get("raw_meta") or {}),
                "quantity": max(1, _to_int(row.get("quantity"), default=1)),
            }
        )
    for user_items in items_by_user_id.values():
        user_items.sort(key=lambda item: (item["ts"], int(item["tx_id"])))

    plans_by_id: dict[int, dict[str, Any]] = {}
    for row in _read_csv_rows(pack_path / "roadmap_plans.csv"):
        plan_id = _to_int(row.get("plan_id"))
        plans_by_id[plan_id] = {
            "plan_id": plan_id,
            "user_id": _to_int(row.get("user_id")),
            "category": str(row.get("category") or "").strip().lower(),
            "meta": _parse_json(row.get("meta"), default={}),
            "created_at": _parse_dt(row.get("created_at")),
            "updated_at": _parse_dt(row.get("updated_at")),
        }

    steps_by_plan_id: dict[int, list[dict[str, Any]]] = defaultdict(list)
    steps_by_id: dict[int, dict[str, Any]] = {}
    for row in _read_csv_rows(pack_path / "roadmap_steps.csv"):
        step = {
            "step_id": _to_int(row.get("step_id")),
            "plan_id": _to_int(row.get("plan_id")),
            "step_index": _to_int(row.get("step_index")),
            "product_type": str(row.get("product_type") or "").strip().lower(),
            "status": str(row.get("status") or "").strip().lower(),
            "recommended_product_id": _to_int(row.get("recommended_product_id")),
        }
        steps_by_plan_id[int(step["plan_id"])].append(step)
        steps_by_id[int(step["step_id"])] = step
    for plan_steps in steps_by_plan_id.values():
        plan_steps.sort(key=lambda row: (int(row["step_index"]), int(row["step_id"])))

    events_by_plan_id: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in _read_csv_rows(pack_path / "roadmap_events.csv"):
        plan_id = _to_int(row.get("plan_id"))
        event = {
            "roadmap_event_id": _to_int(row.get("roadmap_event_id")),
            "plan_id": plan_id,
            "user_id": _to_int(row.get("user_id")),
            "step_id": _to_int(row.get("step_id")),
            "event_type": str(row.get("event_type") or "").strip().lower(),
            "created_at": _parse_dt(row.get("created_at")),
            "context": _parse_json(row.get("context"), default={}),
        }
        events_by_plan_id[plan_id].append(event)
    for plan_events in events_by_plan_id.values():
        plan_events.sort(
            key=lambda row: (
                row.get("created_at") or datetime.min.replace(tzinfo=timezone.utc),
                int(row.get("roadmap_event_id") or 0),
            )
        )

    return {
        "summary": summary,
        "products": product_rows,
        "products_by_id": products_by_id,
        "profiles_by_user_id": profiles_by_user_id,
        "items_by_user_id": items_by_user_id,
        "plans_by_id": plans_by_id,
        "steps_by_plan_id": steps_by_plan_id,
        "steps_by_id": steps_by_id,
        "events_by_plan_id": events_by_plan_id,
    }


def _top_prediction(predictions: list[dict[str, Any]]) -> dict[str, Any]:
    return predictions[0] if predictions else {}


def _stats_row(payload: dict[str, int], *, label: str) -> dict[str, Any]:
    episodes = int(payload.get("episodes", 0))
    outcome_eligible = int(payload.get("outcome_eligible", 0))
    return {
        label: label,
        "episodes": episodes,
        "agreement_rate": _rate(int(payload.get("agreement", 0)), episodes),
        "active_expected_hit_rate": _rate(int(payload.get("active_expected_hits", 0)), episodes),
        "active_expected_hit_rate_at3": _rate(int(payload.get("active_expected_hits_at3", 0)), episodes),
        "candidate_expected_hit_rate": _rate(int(payload.get("candidate_expected_hits", 0)), episodes),
        "candidate_expected_hit_rate_at3": _rate(int(payload.get("candidate_expected_hits_at3", 0)), episodes),
        "outcome_eligible": outcome_eligible,
        "active_outcome_hit_rate": _rate(int(payload.get("active_outcome_hits", 0)), outcome_eligible),
        "active_outcome_hit_rate_at3": _rate(int(payload.get("active_outcome_hits_at3", 0)), outcome_eligible),
        "candidate_outcome_hit_rate": _rate(int(payload.get("candidate_outcome_hits", 0)), outcome_eligible),
        "candidate_outcome_hit_rate_at3": _rate(int(payload.get("candidate_outcome_hits_at3", 0)), outcome_eligible),
    }


def _render_markdown(payload: dict[str, Any]) -> str:
    lines: list[str] = []
    params = payload.get("params", {})
    summary = payload.get("summary", {})
    compare = payload.get("compare", {})
    lines.append("# Roadmap Scenario Replay")
    lines.append("")
    lines.append(
        f"- pack: `{params.get('path')}`"
    )
    lines.append(
        f"- episodes: `{summary.get('episodes_total', 0)}`"
    )
    lines.append(
        f"- active: `{summary.get('active_model_version')}`"
    )
    lines.append(
        f"- candidate: `{summary.get('candidate_model_version')}`"
    )
    lines.append("")
    lines.append("| metric | active | candidate |")
    lines.append("| --- | ---: | ---: |")
    lines.append(f"| expected hit rate | {compare.get('active_expected_hit_rate', 0.0):.4f} | {compare.get('candidate_expected_hit_rate', 0.0):.4f} |")
    lines.append(f"| expected hit rate @3 | {compare.get('active_expected_hit_rate_at3', 0.0):.4f} | {compare.get('candidate_expected_hit_rate_at3', 0.0):.4f} |")
    lines.append(f"| outcome hit rate | {compare.get('active_outcome_hit_rate', 0.0):.4f} | {compare.get('candidate_outcome_hit_rate', 0.0):.4f} |")
    lines.append(f"| outcome hit rate @3 | {compare.get('active_outcome_hit_rate_at3', 0.0):.4f} | {compare.get('candidate_outcome_hit_rate_at3', 0.0):.4f} |")
    lines.append("")

    by_scenario = payload.get("by_scenario_key", [])
    if by_scenario:
        lines.append("## By Scenario")
        lines.append("")
        lines.append("| scenario | episodes | active expected | candidate expected | active outcome | candidate outcome |")
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
        for row in by_scenario:
            lines.append(
                f"| {row.get('scenario_key')} | {row.get('episodes', 0)} | "
                f"{row.get('active_expected_hit_rate', 0.0):.4f} | {row.get('candidate_expected_hit_rate', 0.0):.4f} | "
                f"{row.get('active_outcome_hit_rate', 0.0):.4f} | {row.get('candidate_outcome_hit_rate', 0.0):.4f} |"
            )
        lines.append("")

    swap_rows = payload.get("swap_rows", [])
    if swap_rows:
        lines.append("## Swaps")
        lines.append("")
        lines.append("| active top1 | candidate top1 | plans |")
        lines.append("| --- | --- | ---: |")
        for row in swap_rows[:10]:
            lines.append(
                f"| {row.get('active_top1')} | {row.get('candidate_top1')} | {row.get('plans', 0)} |"
            )
        lines.append("")

    return "\n".join(lines).strip() + "\n"


class Command(BaseCommand):
    help = "Replay active and candidate roadmap next-step models on a synthetic scenario pack."

    def add_arguments(self, parser):
        parser.add_argument("--path", type=str, required=True)
        parser.add_argument(
            "--active-model-path",
            type=str,
            default=str(getattr(settings, "ROADMAP_NEXTSTEP_V4_MODEL_PATH", "") or ""),
        )
        parser.add_argument(
            "--candidate-model-path",
            type=str,
            default=str(getattr(settings, "ROADMAP_NEXTSTEP_V4_SHADOW_MODEL_PATH", "") or ""),
        )
        parser.add_argument("--format", type=str, choices=["json", "md"], default="json")
        parser.add_argument("--out", type=str, default="")

    def handle(self, *args, **options):
        pack_path = _resolve_pack_path(str(options.get("path") or ""))
        if not pack_path.exists() or not pack_path.is_dir():
            raise CommandError(f"Scenario pack path not found: {pack_path}")

        active_model_path = str(options.get("active_model_path") or "").strip()
        candidate_model_path = str(options.get("candidate_model_path") or "").strip()
        if not active_model_path:
            raise CommandError("--active-model-path is required")
        if not candidate_model_path:
            raise CommandError("--candidate-model-path is required")

        active_artifact = _load_model_for_path(active_model_path)
        candidate_artifact = _load_model_for_path(candidate_model_path)
        if not isinstance(active_artifact, dict) or str(active_artifact.get("task") or "") != "roadmap_nextstep_v4_ranking":
            raise CommandError("Active model path does not point to a roadmap_nextstep_v4_ranking artifact")
        if not isinstance(candidate_artifact, dict) or str(candidate_artifact.get("task") or "") != "roadmap_nextstep_v4_ranking":
            raise CommandError("Candidate model path does not point to a roadmap_nextstep_v4_ranking artifact")

        pack = _load_pack(pack_path)
        summary = pack["summary"]
        products = list(pack["products"])
        products_by_id = dict(pack["products_by_id"])
        profiles_by_user_id = dict(pack["profiles_by_user_id"])
        items_by_user_id = dict(pack["items_by_user_id"])
        plans_by_id = dict(pack["plans_by_id"])
        steps_by_plan_id = dict(pack["steps_by_plan_id"])
        steps_by_id = dict(pack["steps_by_id"])
        events_by_plan_id = dict(pack["events_by_plan_id"])

        scenario_instances = list(summary.get("scenario_instances") or [])
        if not scenario_instances:
            raise CommandError("summary.json does not contain scenario_instances")

        aggregate = {
            "episodes": 0,
            "agreement": 0,
            "active_expected_hits": 0,
            "active_expected_hits_at3": 0,
            "candidate_expected_hits": 0,
            "candidate_expected_hits_at3": 0,
            "outcome_eligible": 0,
            "active_outcome_hits": 0,
            "active_outcome_hits_at3": 0,
            "candidate_outcome_hits": 0,
            "candidate_outcome_hits_at3": 0,
        }
        by_scenario_stats: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        by_expected_stats: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        swap_stats: dict[tuple[str, str], dict[str, int]] = defaultdict(lambda: defaultdict(int))
        episode_rows: list[dict[str, Any]] = []

        for instance in scenario_instances:
            plan_id = _to_int(instance.get("plan_id"))
            user_id = _to_int(instance.get("user_id"))
            plan = plans_by_id.get(plan_id)
            if not plan:
                continue
            category = str(plan.get("category") or "haircare").strip().lower()
            t0 = _parse_dt(instance.get("t0_utc")) or plan.get("updated_at")
            if t0 is None:
                continue

            plan_meta = plan.get("meta") or {}
            context_meta = plan_meta.get("context") if isinstance(plan_meta, dict) else {}
            ml_meta = plan_meta.get("ml") if isinstance(plan_meta, dict) else {}
            context_product_ids = [
                _to_int(raw)
                for raw in (context_meta.get("post_ctx_product_ids") or [])
                if _to_int(raw) > 0
            ]
            context_products = [
                dict(products_by_id[product_id])
                for product_id in context_product_ids
                if product_id in products_by_id
            ]

            expected_next = str(instance.get("expected_next_product_type") or "").strip().lower()
            planned_target_product_type = str(
                instance.get("planned_target_product_type")
                or (ml_meta.get("planned_target_product_type") if isinstance(ml_meta, dict) else "")
                or ""
            ).strip().lower()
            planned_target_step_index = _to_int(
                instance.get("planned_target_step_index")
                or (ml_meta.get("planned_target_step_index") if isinstance(ml_meta, dict) else 0),
                default=0,
            )

            history_items = [
                dict(item)
                for item in (items_by_user_id.get(user_id) or [])
                if item.get("ts") is not None and item["ts"] <= t0
            ]
            profile = profiles_by_user_id.get(user_id) or {}

            plan_events = [
                dict(event)
                for event in (events_by_plan_id.get(plan_id) or [])
                if event.get("created_at") is not None and event["created_at"] >= t0
            ]
            completed_events = [
                event for event in plan_events if str(event.get("event_type") or "") == "roadmap_step_completed"
            ]
            first_completed = completed_events[0] if completed_events else None
            actual_completed_product_type = ""
            actual_matched_by = ""
            if first_completed:
                event_context = first_completed.get("context") or {}
                actual_completed_product_type = str(event_context.get("product_type") or "").strip().lower()
                actual_matched_by = str(event_context.get("matched_by") or "").strip().lower()
                if not actual_completed_product_type:
                    step = steps_by_id.get(_to_int(first_completed.get("step_id")))
                    actual_completed_product_type = str((step or {}).get("product_type") or "").strip().lower()

            active_predictions = _predict_with_v4_artifact_from_sources(
                artifact=active_artifact,
                category=category,
                now_utc=t0,
                items=history_items,
                profile=profile,
                context_products=context_products,
                catalog_products=products,
                planned_target_product_type=planned_target_product_type,
                planned_target_step_index=planned_target_step_index,
                candidate_types=None,
            )
            candidate_predictions = _predict_with_v4_artifact_from_sources(
                artifact=candidate_artifact,
                category=category,
                now_utc=t0,
                items=history_items,
                profile=profile,
                context_products=context_products,
                catalog_products=products,
                planned_target_product_type=planned_target_product_type,
                planned_target_step_index=planned_target_step_index,
                candidate_types=None,
            )

            active_top1 = str(_top_prediction(active_predictions).get("candidate_type") or "").strip().lower()
            candidate_top1 = str(_top_prediction(candidate_predictions).get("candidate_type") or "").strip().lower()
            active_top3 = {
                str(row.get("candidate_type") or "").strip().lower()
                for row in active_predictions[:3]
                if str(row.get("candidate_type") or "").strip()
            }
            candidate_top3 = {
                str(row.get("candidate_type") or "").strip().lower()
                for row in candidate_predictions[:3]
                if str(row.get("candidate_type") or "").strip()
            }
            active_expected_hit = int(bool(expected_next) and active_top1 == expected_next)
            active_expected_hit_at3 = int(bool(expected_next) and expected_next in active_top3)
            candidate_expected_hit = int(bool(expected_next) and candidate_top1 == expected_next)
            candidate_expected_hit_at3 = int(bool(expected_next) and expected_next in candidate_top3)
            outcome_eligible = int(bool(actual_completed_product_type))
            active_outcome_hit = int(outcome_eligible and active_top1 == actual_completed_product_type)
            active_outcome_hit_at3 = int(outcome_eligible and actual_completed_product_type in active_top3)
            candidate_outcome_hit = int(outcome_eligible and candidate_top1 == actual_completed_product_type)
            candidate_outcome_hit_at3 = int(outcome_eligible and actual_completed_product_type in candidate_top3)
            agreement = int(bool(active_top1) and bool(candidate_top1) and active_top1 == candidate_top1)

            aggregate["episodes"] += 1
            aggregate["agreement"] += agreement
            aggregate["active_expected_hits"] += active_expected_hit
            aggregate["active_expected_hits_at3"] += active_expected_hit_at3
            aggregate["candidate_expected_hits"] += candidate_expected_hit
            aggregate["candidate_expected_hits_at3"] += candidate_expected_hit_at3
            aggregate["outcome_eligible"] += outcome_eligible
            aggregate["active_outcome_hits"] += active_outcome_hit
            aggregate["active_outcome_hits_at3"] += active_outcome_hit_at3
            aggregate["candidate_outcome_hits"] += candidate_outcome_hit
            aggregate["candidate_outcome_hits_at3"] += candidate_outcome_hit_at3

            scenario_key = str(instance.get("scenario_key") or "").strip()
            if scenario_key:
                stats = by_scenario_stats[scenario_key]
                stats["episodes"] += 1
                stats["agreement"] += agreement
                stats["active_expected_hits"] += active_expected_hit
                stats["active_expected_hits_at3"] += active_expected_hit_at3
                stats["candidate_expected_hits"] += candidate_expected_hit
                stats["candidate_expected_hits_at3"] += candidate_expected_hit_at3
                stats["outcome_eligible"] += outcome_eligible
                stats["active_outcome_hits"] += active_outcome_hit
                stats["active_outcome_hits_at3"] += active_outcome_hit_at3
                stats["candidate_outcome_hits"] += candidate_outcome_hit
                stats["candidate_outcome_hits_at3"] += candidate_outcome_hit_at3

            if expected_next:
                stats = by_expected_stats[expected_next]
                stats["episodes"] += 1
                stats["agreement"] += agreement
                stats["active_expected_hits"] += active_expected_hit
                stats["active_expected_hits_at3"] += active_expected_hit_at3
                stats["candidate_expected_hits"] += candidate_expected_hit
                stats["candidate_expected_hits_at3"] += candidate_expected_hit_at3
                stats["outcome_eligible"] += outcome_eligible
                stats["active_outcome_hits"] += active_outcome_hit
                stats["active_outcome_hits_at3"] += active_outcome_hit_at3
                stats["candidate_outcome_hits"] += candidate_outcome_hit
                stats["candidate_outcome_hits_at3"] += candidate_outcome_hit_at3

            if active_top1 and candidate_top1 and active_top1 != candidate_top1:
                stats = swap_stats[(active_top1, candidate_top1)]
                stats["plans"] += 1
                stats["active_expected_hits"] += active_expected_hit
                stats["active_expected_hits_at3"] += active_expected_hit_at3
                stats["candidate_expected_hits"] += candidate_expected_hit
                stats["candidate_expected_hits_at3"] += candidate_expected_hit_at3
                stats["outcome_eligible"] += outcome_eligible
                stats["active_outcome_hits"] += active_outcome_hit
                stats["active_outcome_hits_at3"] += active_outcome_hit_at3
                stats["candidate_outcome_hits"] += candidate_outcome_hit
                stats["candidate_outcome_hits_at3"] += candidate_outcome_hit_at3

            episode_rows.append(
                {
                    "scenario_key": scenario_key,
                    "replica": _to_int(instance.get("replica"), default=0),
                    "user_id": user_id,
                    "plan_id": plan_id,
                    "category": category,
                    "t0_utc": t0.isoformat().replace("+00:00", "Z"),
                    "expected_next_product_type": expected_next,
                    "planned_target_product_type": planned_target_product_type,
                    "planned_target_step_index": planned_target_step_index,
                    "actual_completed_product_type": actual_completed_product_type,
                    "actual_matched_by": actual_matched_by,
                    "active_top1": active_top1,
                    "candidate_top1": candidate_top1,
                    "agreement": agreement,
                    "active_expected_hit": active_expected_hit,
                    "active_expected_hit_at3": active_expected_hit_at3,
                    "candidate_expected_hit": candidate_expected_hit,
                    "candidate_expected_hit_at3": candidate_expected_hit_at3,
                    "active_outcome_hit": active_outcome_hit,
                    "active_outcome_hit_at3": active_outcome_hit_at3,
                    "candidate_outcome_hit": candidate_outcome_hit,
                    "candidate_outcome_hit_at3": candidate_outcome_hit_at3,
                    "active_top3": sorted(active_top3),
                    "candidate_top3": sorted(candidate_top3),
                    "active_predictions": active_predictions[:3],
                    "candidate_predictions": candidate_predictions[:3],
                    "context_product_ids": context_product_ids,
                    "history_tx_items": len(history_items),
                    "future_plan_events": Counter(str(event.get("event_type") or "") for event in plan_events),
                    "step_count": len(steps_by_plan_id.get(plan_id) or []),
                }
            )

        by_scenario_rows = []
        for scenario_key, stats in sorted(by_scenario_stats.items()):
            row = _stats_row(stats, label="scenario_key")
            row["scenario_key"] = scenario_key
            by_scenario_rows.append(row)

        by_expected_rows = []
        for expected_next, stats in sorted(by_expected_stats.items()):
            row = _stats_row(stats, label="expected_next_product_type")
            row["expected_next_product_type"] = expected_next
            by_expected_rows.append(row)

        swap_rows = []
        for (active_top1, candidate_top1), stats in sorted(
            swap_stats.items(),
            key=lambda kv: (-int(kv[1].get("plans", 0)), kv[0][0], kv[0][1]),
        ):
            plans = int(stats.get("plans", 0))
            outcome_eligible = int(stats.get("outcome_eligible", 0))
            swap_rows.append(
                {
                    "active_top1": active_top1,
                    "candidate_top1": candidate_top1,
                    "plans": plans,
                    "active_expected_hit_rate": _rate(int(stats.get("active_expected_hits", 0)), plans),
                    "active_expected_hit_rate_at3": _rate(int(stats.get("active_expected_hits_at3", 0)), plans),
                    "candidate_expected_hit_rate": _rate(int(stats.get("candidate_expected_hits", 0)), plans),
                    "candidate_expected_hit_rate_at3": _rate(int(stats.get("candidate_expected_hits_at3", 0)), plans),
                    "outcome_eligible": outcome_eligible,
                    "active_outcome_hit_rate": _rate(int(stats.get("active_outcome_hits", 0)), outcome_eligible),
                    "active_outcome_hit_rate_at3": _rate(int(stats.get("active_outcome_hits_at3", 0)), outcome_eligible),
                    "candidate_outcome_hit_rate": _rate(int(stats.get("candidate_outcome_hits", 0)), outcome_eligible),
                    "candidate_outcome_hit_rate_at3": _rate(int(stats.get("candidate_outcome_hits_at3", 0)), outcome_eligible),
                }
            )

        compare = {
            "episodes_scored": int(aggregate["episodes"]),
            "agreement_rate": _rate(int(aggregate["agreement"]), int(aggregate["episodes"])),
            "active_expected_hit_rate": _rate(
                int(aggregate["active_expected_hits"]),
                int(aggregate["episodes"]),
            ),
            "active_expected_hit_rate_at3": _rate(
                int(aggregate["active_expected_hits_at3"]),
                int(aggregate["episodes"]),
            ),
            "candidate_expected_hit_rate": _rate(
                int(aggregate["candidate_expected_hits"]),
                int(aggregate["episodes"]),
            ),
            "candidate_expected_hit_rate_at3": _rate(
                int(aggregate["candidate_expected_hits_at3"]),
                int(aggregate["episodes"]),
            ),
            "outcome_eligible": int(aggregate["outcome_eligible"]),
            "active_outcome_hit_rate": _rate(
                int(aggregate["active_outcome_hits"]),
                int(aggregate["outcome_eligible"]),
            ),
            "active_outcome_hit_rate_at3": _rate(
                int(aggregate["active_outcome_hits_at3"]),
                int(aggregate["outcome_eligible"]),
            ),
            "candidate_outcome_hit_rate": _rate(
                int(aggregate["candidate_outcome_hits"]),
                int(aggregate["outcome_eligible"]),
            ),
            "candidate_outcome_hit_rate_at3": _rate(
                int(aggregate["candidate_outcome_hits_at3"]),
                int(aggregate["outcome_eligible"]),
            ),
        }
        compare["candidate_minus_active_expected_hit_rate"] = round(
            float(compare["candidate_expected_hit_rate"]) - float(compare["active_expected_hit_rate"]),
            6,
        )
        compare["candidate_minus_active_expected_hit_rate_at3"] = round(
            float(compare["candidate_expected_hit_rate_at3"]) - float(compare["active_expected_hit_rate_at3"]),
            6,
        )
        compare["candidate_minus_active_outcome_hit_rate"] = round(
            float(compare["candidate_outcome_hit_rate"]) - float(compare["active_outcome_hit_rate"]),
            6,
        )
        compare["candidate_minus_active_outcome_hit_rate_at3"] = round(
            float(compare["candidate_outcome_hit_rate_at3"]) - float(compare["active_outcome_hit_rate_at3"]),
            6,
        )

        active_summary = nextstep_model_artifact_summary(active_model_path)
        candidate_summary = nextstep_model_artifact_summary(candidate_model_path)
        payload = {
            "params": {
                "path": str(pack_path),
                "active_model_path": active_model_path,
                "candidate_model_path": candidate_model_path,
                "format": str(options.get("format") or "json"),
            },
            "pack_summary": summary,
            "summary": {
                "episodes_total": int(aggregate["episodes"]),
                "active_model_version": str(active_summary.get("model_version") or ""),
                "candidate_model_version": str(candidate_summary.get("model_version") or ""),
            },
            "active_model": active_summary,
            "candidate_model": candidate_summary,
            "compare": compare,
            "by_scenario_key": by_scenario_rows,
            "by_expected_next_product_type": by_expected_rows,
            "swap_rows": swap_rows,
            "episode_rows": episode_rows,
        }

        fmt = str(options.get("format") or "json").strip().lower()
        output_text = (
            json.dumps(payload, ensure_ascii=False, indent=2)
            if fmt == "json"
            else _render_markdown(payload)
        )

        out_path_raw = str(options.get("out") or "").strip()
        if out_path_raw:
            out_path = _resolve_pack_path(out_path_raw)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(output_text, encoding="utf-8")

        self.stdout.write(output_text)
