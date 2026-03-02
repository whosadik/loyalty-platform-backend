from __future__ import annotations

import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from django.apps import apps
from django.conf import settings
from django.contrib.auth import get_user_model
from django.core.management.base import BaseCommand
from django.db import models
from django.db.models import Count, Q

from catalog.models import Product
from offers.models import OfferAssignment, OfferEvent
from recs_analytics.models import RecommendationEvent
from recs_app.reranker import get_reranker_model_version
from roadmap_app.models import RoadmapPlan, RoadmapStep
from transactions.models import OwnedProduct, Transaction, TransactionItem


def _default_repr(field: models.Field) -> str:
    if not field.has_default():
        return "NO_DEFAULT"
    default = field.default
    if callable(default):
        name = getattr(default, "__name__", default.__class__.__name__)
        if name in {"dict", "list", "set", "tuple"}:
            return f"{name}()"
        return f"callable:{name}"
    if default is None:
        return "None"
    return repr(default)


def _field_type_repr(field: models.Field) -> str:
    if not getattr(field, "is_relation", False):
        return field.get_internal_type()
    rel_model = "?"
    try:
        rel_model = field.related_model._meta.label
    except Exception:
        pass
    on_delete = getattr(getattr(field, "remote_field", None), "on_delete", None)
    on_delete_name = getattr(on_delete, "__name__", str(on_delete)) if on_delete else "?"
    if isinstance(field, models.OneToOneField):
        return f"OneToOne({rel_model}, on_delete={on_delete_name})"
    if isinstance(field, models.ForeignKey):
        return f"ForeignKey({rel_model}, on_delete={on_delete_name})"
    return f"Relation({rel_model})"


def _model_schema_rows(model: type[models.Model]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for field in model._meta.fields:
        out.append(
            {
                "field": field.name,
                "type": _field_type_repr(field),
                "null": bool(getattr(field, "null", False)),
                "blank": bool(getattr(field, "blank", False)),
                "default": _default_repr(field),
            }
        )
    return out


def _markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    if not rows:
        rows = [["-"] * len(headers)]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(x) for x in row) + " |")
    return "\n".join(lines)


def _to_sample(v: Any, max_len: int = 120) -> str:
    if v is None:
        s = "null"
    elif isinstance(v, (dict, list)):
        s = json.dumps(v, ensure_ascii=False, sort_keys=True)
    else:
        s = str(v)
    s = s.replace("\n", " ").strip()
    if len(s) > max_len:
        s = s[: max_len - 3] + "..."
    return s


def _is_non_empty(field_name: str, value: Any) -> bool:
    if value is None:
        return False
    if field_name == "in_stock":
        return True
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, dict, tuple, set)):
        return len(value) > 0
    return True


def _fill_rate(field_name: str, total: int) -> tuple[int, float]:
    non_empty = 0
    for value in Product.objects.values_list(field_name, flat=True).iterator():
        if _is_non_empty(field_name, value):
            non_empty += 1
    pct = round((non_empty / total * 100.0), 2) if total else 0.0
    return non_empty, pct


def _json_key_stats(category: str, field_name: str, max_keys: int) -> list[dict[str, Any]]:
    counts: Counter[str] = Counter()
    examples: dict[str, list[str]] = defaultdict(list)
    for obj in Product.objects.filter(category=category).values_list(field_name, flat=True).iterator():
        if not isinstance(obj, dict):
            continue
        for key, value in obj.items():
            k = str(key)
            counts[k] += 1
            sample = _to_sample(value, max_len=90)
            if sample and sample not in examples[k] and len(examples[k]) < 3:
                examples[k].append(sample)
    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    if max_keys > 0:
        ranked = ranked[:max_keys]
    return [{"key": k, "count": c, "examples": examples.get(k, [])} for k, c in ranked]


def _rec_context_key_stats(max_rows: int = 40) -> list[tuple[str, int]]:
    c: Counter[str] = Counter()
    for ctx in RecommendationEvent.objects.values_list("context", flat=True).iterator():
        if not isinstance(ctx, dict):
            continue
        for key in ctx.keys():
            c[str(key)] += 1
    return sorted(c.items(), key=lambda kv: (-kv[1], kv[0]))[:max_rows]


def _rec_page_section_algo_stats(max_rows: int = 50) -> list[tuple[str, str, str, int]]:
    c: Counter[tuple[str, str, str]] = Counter()
    qs = RecommendationEvent.objects.values("page", "section_key", "algo_mode", "context")
    for row in qs.iterator():
        ctx = row.get("context") or {}
        algo_used = ""
        if isinstance(ctx, dict):
            algo_used = str(ctx.get("algo_used") or "").strip()
        if not algo_used:
            algo_used = str(row.get("algo_mode") or "").strip()
        c[(str(row.get("page") or ""), str(row.get("section_key") or ""), algo_used)] += 1
    ranked = sorted(c.items(), key=lambda kv: (-kv[1], kv[0]))
    return [(p, s, a, cnt) for (p, s, a), cnt in ranked[:max_rows]]


def _fragrance_feature_coverage() -> dict[str, Any]:
    rows = list(
        Product.objects.filter(category=Product.Category.FRAGRANCE).values(
            "attrs",
            "raw_meta",
            "description",
        )
    )
    total = len(rows)
    attrs_keys = Counter()
    raw_keys = Counter()
    desc_non_empty = 0
    words = Counter()
    for row in rows:
        attrs = row.get("attrs") or {}
        raw_meta = row.get("raw_meta") or {}
        if isinstance(attrs, dict):
            for k in attrs.keys():
                attrs_keys[str(k)] += 1
        if isinstance(raw_meta, dict):
            for k in raw_meta.keys():
                raw_keys[str(k)] += 1
        desc = str(row.get("description") or "").strip().lower()
        if desc:
            desc_non_empty += 1
            for w in ["summer", "winter", "spring", "autumn", "day", "evening", "night", "intense", "light"]:
                if w in desc:
                    words[w] += 1
    def pct(v: int) -> float:
        return round((v / total * 100.0), 2) if total else 0.0
    return {
        "total": total,
        "attrs_keys": dict(attrs_keys),
        "raw_keys": dict(raw_keys),
        "description_non_empty_pct": pct(desc_non_empty),
        "description_keyword_hits": dict(words),
        "scent_family_pct": pct(attrs_keys.get("scent_family", 0)),
        "intensity_pct": pct(attrs_keys.get("intensity", 0)),
        "notes_pct": pct(attrs_keys.get("notes", 0)),
        "season_attr_pct": pct(attrs_keys.get("season", 0) + attrs_keys.get("seasons", 0)),
        "occasion_attr_pct": pct(attrs_keys.get("occasion", 0) + attrs_keys.get("occasions", 0)),
        "day_night_attr_pct": pct(attrs_keys.get("day_night", 0)),
    }


def _model_artifacts(root: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for pkl in sorted(root.glob("**/model.pkl")):
        meta_path = pkl.parent / "metadata.json"
        version = ""
        if meta_path.exists():
            try:
                version = str(json.loads(meta_path.read_text(encoding="utf-8")).get("model_version") or "")
            except Exception:
                version = ""
        out.append({"path": str(pkl), "has_meta": meta_path.exists(), "version": version})
    return out


class Command(BaseCommand):
    help = "Generate project inventory report (schema/data/api/ml) in markdown."

    def add_arguments(self, parser):
        parser.add_argument("--output", type=str, default="", help="Optional markdown output file path.")
        parser.add_argument("--max-json-keys", type=int, default=30)
        parser.add_argument("--max-rec-breakdown", type=int, default=50)

    def handle(self, *args, **options):
        max_json_keys = int(options["max_json_keys"])
        max_rec_breakdown = int(options["max_rec_breakdown"])
        project_root = Path(settings.BASE_DIR).parent.resolve()

        User = get_user_model()
        has_username = any(f.name == "username" for f in User._meta.fields)

        users_total = User.objects.count()
        users_ga = User.objects.filter(username__startswith="ga_").count() if has_username else 0

        products_total = Product.objects.count()
        products_by_category = list(
            Product.objects.values("category").annotate(total=Count("id")).order_by("category")
        )

        tx_total = Transaction.objects.count()
        tx_ga = Transaction.objects.filter(user__username__startswith="ga_").count() if has_username else 0
        tx_items_total = TransactionItem.objects.count()
        owned_total = OwnedProduct.objects.count()

        offer_assign_total = OfferAssignment.objects.count()
        offer_events_total = OfferEvent.objects.count()
        offer_events_by_type = list(
            OfferEvent.objects.values("event_type").annotate(total=Count("id")).order_by("event_type")
        )

        rec_events_total = RecommendationEvent.objects.count()
        rec_breakdown = _rec_page_section_algo_stats(max_rows=max_rec_breakdown)
        rec_context_keys = _rec_context_key_stats(max_rows=60)

        roadmap_plans_total = RoadmapPlan.objects.count()
        roadmap_steps_total = RoadmapStep.objects.count()

        fill_fields = [
            "category",
            "product_type",
            "brand",
            "price",
            "in_stock",
            "attrs",
            "concerns",
            "actives",
            "supported_skin_types",
            "image_url",
            "description",
            "ingredients_inci",
        ]
        fill_rows = []
        for field_name in fill_fields:
            non_empty, pct = _fill_rate(field_name, products_total)
            fill_rows.append((field_name, non_empty, pct))

        json_keys_by_category: dict[str, dict[str, list[dict[str, Any]]]] = {}
        for category in [
            Product.Category.FRAGRANCE,
            Product.Category.HAIRCARE,
            Product.Category.SKINCARE,
            Product.Category.MAKEUP,
        ]:
            json_keys_by_category[category] = {
                "attrs": _json_key_stats(category, "attrs", max_json_keys),
                "raw_meta": _json_key_stats(category, "raw_meta", max_json_keys),
            }

        fragrance_cov = _fragrance_feature_coverage()
        model_artifacts = _model_artifacts(project_root / "models")
        active_recs_model_version = get_reranker_model_version()
        active_recs_model_path = str(getattr(settings, "RECS_RERANKER_MODEL_PATH", "") or "")
        active_roadmap_model_path = str(getattr(settings, "ROADMAP_NEXTSTEP_MODEL_PATH", "") or "")

        schema_models: list[tuple[str, type[models.Model]]] = [
            ("catalog.Product", Product),
            ("transactions.Transaction", Transaction),
            ("transactions.TransactionItem", TransactionItem),
            ("transactions.OwnedProduct", OwnedProduct),
            ("offers.Offer", apps.get_model("offers", "Offer")),
            ("offers.OfferAssignment", OfferAssignment),
            ("offers.OfferEvent", OfferEvent),
            ("recs_analytics.RecommendationEvent", RecommendationEvent),
            ("roadmap_app.RoadmapPlan", RoadmapPlan),
            ("roadmap_app.RoadmapStep", RoadmapStep),
        ]

        lines: list[str] = []
        lines.append("# Inventory Report")
        lines.append("")
        lines.append(
            f"_Generated at: {datetime.now(timezone.utc).isoformat()} | DB: {settings.DATABASES['default'].get('NAME')}_"
        )
        lines.append("")
        lines.append("## Summary")
        lines.append(f"- Users: **{users_total}** (ga_*: **{users_ga}**)")
        lines.append(f"- Products: **{products_total}**")
        lines.append(f"- Transactions: **{tx_total}** (ga_*: **{tx_ga}**)")
        lines.append(f"- TransactionItems: **{tx_items_total}**; OwnedProducts: **{owned_total}**")
        lines.append(f"- OfferAssignments: **{offer_assign_total}**; OfferEvents: **{offer_events_total}**")
        lines.append(f"- RecommendationEvents: **{rec_events_total}**")
        lines.append(f"- RoadmapPlans: **{roadmap_plans_total}**; RoadmapSteps: **{roadmap_steps_total}**")
        lines.append(
            f"- Active recs model: `{active_recs_model_path}` (version: `{active_recs_model_version or 'n/a'}`)"
        )
        lines.append(f"- Active roadmap model path: `{active_roadmap_model_path}`")
        lines.append("")

        lines.append("## DB Schema snapshot")
        lines.append("")
        for model_name, model in schema_models:
            lines.append(f"### {model_name}")
            rows = _model_schema_rows(model)
            lines.append(
                _markdown_table(
                    ["field", "type", "null", "blank", "default"],
                    [
                        [
                            r["field"],
                            r["type"],
                            "yes" if r["null"] else "no",
                            "yes" if r["blank"] else "no",
                            r["default"],
                        ]
                        for r in rows
                    ],
                )
            )
            lines.append("")

        lines.append("### Product fields used by subsystems")
        lines.append("- recs: `id, name, brand, price, category, product_type, concerns, attrs, actives, flags, supported_skin_types, strength, in_stock`")
        lines.append("- roadmap: `id, category, product_type, in_stock, brand, attrs` + recs payload fields")
        lines.append("- offers target selection: `id, category, product_type, in_stock` directly; recommendation-based targeting reuses recs payload")
        lines.append("")

        lines.append("### Product fill-rate")
        lines.append(_markdown_table(["field", "non-empty", "non-empty %"], [[f, n, p] for f, n, p in fill_rows]))
        lines.append("")

        lines.append("### attrs/raw_meta keys with sample values")
        for category, payload in json_keys_by_category.items():
            lines.append(f"#### category = {category}")
            for field_name in ["attrs", "raw_meta"]:
                lines.append(f"- `{field_name}`")
                lines.append(
                    _markdown_table(
                        ["key", "count", "examples"],
                        [
                            [row["key"], row["count"], "; ".join(row["examples"]) or "-"]
                            for row in payload[field_name]
                        ],
                    )
                )
                lines.append("")

        lines.append("### OfferEvent idempotency")
        lines.append("- Unique constraint: `event_key` unique when not null (`uq_offer_event_key_not_null`).")
        lines.append("- Key strategy in `offers/events.py`:")
        lines.append("  - explicit `idempotency_key`")
        lines.append("  - else request-scoped key with `request_id`")
        lines.append("  - else one-shot key for `assigned/redeemed/expired/superseded`")
        lines.append("")

        lines.append("### RecommendationEvent context keys")
        lines.append(_markdown_table(["context key", "count"], [[k, c] for k, c in rec_context_keys]))
        lines.append("")
        lines.append("### Roadmap statuses")
        lines.append(f"- Status: {', '.join([x[0] for x in RoadmapStep.Status.choices])}")
        lines.append(f"- Cadence: {', '.join([x[0] for x in RoadmapStep.Cadence.choices])}")
        lines.append("")

        lines.append("## Data volume")
        lines.append("")
        lines.append(
            _markdown_table(
                ["metric", "value"],
                [
                    ["users_total", users_total],
                    ["users_ga_prefix", users_ga],
                    ["products_total", products_total],
                    ["transactions_total", tx_total],
                    ["transactions_ga_prefix", tx_ga],
                    ["transaction_items_total", tx_items_total],
                    ["owned_products_total", owned_total],
                    ["offer_assignments_total", offer_assign_total],
                    ["offer_events_total", offer_events_total],
                    ["rec_events_total", rec_events_total],
                    ["roadmap_plans_total", roadmap_plans_total],
                    ["roadmap_steps_total", roadmap_steps_total],
                ],
            )
        )
        lines.append("")
        lines.append("### Products by category")
        lines.append(
            _markdown_table(["category", "count"], [[r["category"], r["total"]] for r in products_by_category])
        )
        lines.append("")
        lines.append("### Offer events by type")
        lines.append(
            _markdown_table(["event_type", "count"], [[r["event_type"], r["total"]] for r in offer_events_by_type])
        )
        lines.append("")
        lines.append("### Recommendation events by page/section/algo_used")
        lines.append(
            _markdown_table(["page", "section_key", "algo_used", "count"], [[p, s, a, c] for p, s, a, c in rec_breakdown])
        )
        lines.append("")

        lines.append("## API map")
        lines.append("")
        lines.append(
            _markdown_table(
                ["endpoint", "file/class", "key params/body", "events", "A/B & guardrail"],
                [
                    ["/api/products", "backend/catalog/views.py::ProductViewSet.list", "category/product_type/brand/in_stock", "none", "none"],
                    ["/api/products/{id}", "backend/catalog/views.py::ProductViewSet.retrieve", "id", "none", "none"],
                    ["/api/me/profile", "backend/users_app/views.py::MeProfileView", "GET; PUT profile fields", "none", "none"],
                    ["/api/me/favorite-category", "backend/users_app/views.py::MeFavoriteCategoryView", "GET", "none", "none"],
                    ["/api/me/recommendations", "backend/recs_app/views.py::MeRecommendationsView", "category/product_type/limit/algo", "none", "reranker A/B + guardrail"],
                    ["/api/me/recommendations/bundle", "backend/recs_app/views.py::MeBundleView", "product_id/limit/algo", "none", "reranker A/B + guardrail"],
                    ["/api/me/recommendations/home", "backend/recs_app/views.py::HomeRecommendationsView", "limit/category/product_type/price/algo", "RecommendationEvent impression", "reranker A/B + guardrail"],
                    ["/api/me/recommendations/event", "backend/recs_analytics/views.py::RecEventCreateView", "action/product_id/page/section_key/context", "RecommendationEvent click/add_to_cart", "inherits experiment context"],
                    ["/api/me/next-offer", "backend/offers/views.py::MeNextOfferView", "GET", "OfferEvent exposed (+assigned/expired/superseded via service)", "none"],
                    ["/api/me/offers", "backend/offers/views.py::MeOffersView", "GET", "OfferEvent exposed", "none"],
                    ["/api/offers/click", "backend/offers/views.py::OfferClickView", "assignment_id/context", "OfferEvent clicked", "none"],
                    ["/api/offers/redeem", "backend/offers/views.py::RedeemOfferView", "assignment_id/transaction_id", "OfferEvent redeemed", "none"],
                    ["/api/checkout", "backend/checkout_app/views.py::CheckoutView", "items + optional idempotency/apply_assignment_id/redeem_points", "OfferEvent redeemed/assigned + RecommendationEvent purchase_attributed", "none"],
                    ["/api/checkout/preview", "backend/checkout_app/views.py::CheckoutPreviewView", "items + optional offer/points", "none", "none"],
                    ["/api/me/roadmap", "backend/roadmap_app/views.py::MeRoadmapView", "category", "none", "none"],
                    ["/api/me/roadmap/refresh", "backend/roadmap_app/views.py::MeRoadmapRefreshView", "category", "none", "none"],
                    ["/api/me/roadmap/steps/{id}", "backend/roadmap_app/views.py::MeRoadmapStepPatchView", "status", "none", "none"],
                    ["/api/admin/overview", "backend/admin_tools/views.py::AdminOverviewView", "GET", "none", "none"],
                    ["/api/admin/metrics", "backend/analytics_app/views.py::AdminMetricsView", "GET", "none", "none"],
                    ["/api/admin/recs/experiments", "backend/admin_tools/views.py::AdminRecsExperimentsView", "days/experiment_id/variant", "none", "none"],
                    ["/api/admin/health", "backend/admin_tools/views.py::AdminHealthView", "GET", "none", "none"],
                ],
            )
        )
        lines.append("")

        lines.append("## ML map")
        lines.append("")
        lines.append("### model.pkl inventory")
        lines.append(
            _markdown_table(
                ["model path", "has metadata.json", "metadata.model_version"],
                [[row["path"], "yes" if row["has_meta"] else "no", row["version"] or "-"] for row in model_artifacts],
            )
        )
        lines.append("")
        lines.append("### Runtime model loading")
        lines.append("- recs: `backend/recs_app/reranker.py` loads `RECS_RERANKER_MODEL_PATH`; version from sibling metadata.json (`model_version`) or filename stem")
        lines.append("- roadmap: `backend/roadmap_app/ml_next_step.py` loads `ROADMAP_NEXTSTEP_MODEL_PATH` (explicit model_version not surfaced)")
        lines.append("- offer redemption model artifacts exist but are not wired into backend inference path")
        lines.append("")
        lines.append(_markdown_table(
            ["setting", "value"],
            [
                ["RECS_RERANKER_MODEL_PATH", active_recs_model_path],
                ["active_recs_model_version", active_recs_model_version or "n/a"],
                ["ROADMAP_NEXTSTEP_MODEL_PATH", active_roadmap_model_path],
                ["RECS_AB_ENABLED", bool(getattr(settings, "RECS_AB_ENABLED", False))],
                ["RECS_GUARDRAIL_ENABLED", bool(getattr(settings, "RECS_GUARDRAIL_ENABLED", False))],
            ],
        ))
        lines.append("")
        lines.append("### Training scripts")
        lines.append(
            _markdown_table(
                ["script", "purpose", "inputs", "outputs"],
                [
                    ["ml/training/run_recs_pipeline.py", "full recs pipeline orchestration", "processed/raw_glob/project_db + hyperparams", "model/co_map/splits + baseline/reranker reports (txt/json)"],
                    ["ml/training/run_offer_pipeline.py", "offer pipeline orchestration", "project_db export window", "offer_train.parquet + offer model.pkl + metadata.json"],
                    ["ml/training/export_project_training_data.py", "export training datasets from Django DB", "out_dir/days", "interactions/items/transactions_items/rec_events/offer_events/offer_train parquet"],
                    ["ml/training/build_next_purchase_dataset.py", "build supervised next-purchase dataset", "interactions + items parquet", "next_purchase_ds.parquet"],
                    ["ml/training/train_reranker_lr.py", "train rec reranker (lr/hgb) + co_map artifact", "interactions/items/ds + training params", "model.pkl/co_map.pkl/metadata.json/train_users.txt/test_users.txt"],
                    ["ml/training/eval_cooc_baseline.py", "evaluate cooc baseline", "interactions/items/ds + user splits", "cooc baseline report txt/json"],
                    ["ml/training/eval_reranker.py", "evaluate reranker", "interactions/items/ds/model + user splits", "reranker report txt/json"],
                    ["ml/training/train_offer_redemption_lr.py", "train offer redemption logistic model", "offer_train.parquet", "model.pkl + metadata.json"],
                    ["ml/training/sanity_check_candidates.py", "retrieval sanity/coverage check", "interactions/items/ds + retrieval params", "console summary (coverage)"],
                ],
            )
        )
        lines.append("")

        lines.append("## Roadmap ML readiness")
        lines.append("")
        lines.append("### Current roadmap logic")
        lines.append("- chain = category rules (`base` + `optional`) + purchased/owned/recent/catalog types, then dynamically trimmed/extended")
        lines.append("- optional ML next-step predictions can reorder and expand chain (confidence threshold based)")
        lines.append("- recommended_product/suggestions use `ml_logic.recommender.recommend` with `_cooccurrence_90d` and stock/ownership filters")
        lines.append("- checkout builds `roadmap_ctx` and passes it into offers target selection; offers can supersede active assignment if roadmap next step differs")
        lines.append("")
        lines.append("### Fragrance signals available")
        lines.append(
            _markdown_table(
                ["signal", "coverage %", "comment"],
                [
                    ["attrs.scent_family", fragrance_cov["scent_family_pct"], "explicit"],
                    ["attrs.intensity", fragrance_cov["intensity_pct"], "explicit"],
                    ["attrs.notes", fragrance_cov["notes_pct"], "explicit"],
                    ["attrs.season/seasons", fragrance_cov["season_attr_pct"], "explicit season field"],
                    ["attrs.occasion/occasions", fragrance_cov["occasion_attr_pct"], "explicit occasion field"],
                    ["attrs.day_night", fragrance_cov["day_night_attr_pct"], "explicit day/evening field"],
                    ["description non-empty", fragrance_cov["description_non_empty_pct"], "fallback text signal only"],
                ],
            )
        )
        lines.append("")
        lines.append("Description keyword hits (fragrance):")
        lines.append(
            _markdown_table(
                ["keyword", "count"],
                [[k, v] for k, v in sorted(fragrance_cov["description_keyword_hits"].items(), key=lambda kv: (-kv[1], kv[0]))],
            )
        )
        lines.append("")
        lines.append("### Missing data pieces (no implementation)")
        lines.append("- candidate slots: `warm_day`, `warm_evening`, `cold_day`, `cold_evening`")
        lines.append("- fragrance fields: `notes(top/heart/base)`, `family/subfamily`, `intensity(normalized)`, `season`, `occasion`")
        lines.append("- roadmap event labels: step impression/click/skip/accept and downstream conversion linkage")
        lines.append("")

        lines.append("## Risks / conflicts")
        empty_attrs = Product.objects.filter(Q(attrs={}) | Q(attrs__isnull=True)).count()
        empty_product_type = Product.objects.filter(Q(product_type="") | Q(product_type__isnull=True)).count()
        out_of_stock = Product.objects.filter(in_stock=False).count()
        exposed_cnt = OfferEvent.objects.filter(event_type=OfferEvent.Type.EXPOSED).count()
        assigned_cnt = OfferEvent.objects.filter(event_type=OfferEvent.Type.ASSIGNED).count()
        superseded_cnt = OfferAssignment.objects.filter(superseded_at__isnull=False).count()
        lines.append(
            _markdown_table(
                ["risk", "signal", "impact"],
                [
                    ["data sparsity", f"empty attrs={empty_attrs}/{products_total}; empty product_type={empty_product_type}/{products_total}", "weaker rec/roadmap/offer targeting"],
                    ["catalog stock state", f"in_stock=false: {out_of_stock}/{products_total}", "candidate pool shrink and fallback pressure"],
                    ["campaign conflicts", f"superseded assignments: {superseded_cnt}", "roadmap-driven supersede + budget refund path complexity"],
                    ["exposure inflation", f"offer_exposed={exposed_cnt}, offer_assigned={assigned_cnt}", "EXPOSED can grow fast due repeated GET endpoints"],
                ],
            )
        )
        lines.append("")

        lines.append("## Missing pieces checklist")
        lines.append("- [ ] Structured fragrance ontology for season/daypart/occasion/intensity/notes.")
        lines.append("- [ ] Roadmap interaction telemetry for supervised learning.")
        lines.append("- [ ] Explicit dedup policy for offer exposure analytics windows.")
        lines.append("- [ ] Data quality monitor for critical product fields (attrs/product_type/in_stock).")
        lines.append("")

        report = "\n".join(lines).strip() + "\n"
        output = str(options.get("output") or "").strip()
        if output:
            out_path = Path(output).resolve()
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(report, encoding="utf-8")
            self.stdout.write(self.style.SUCCESS(f"Report written: {out_path}"))
        self.stdout.write(report)
