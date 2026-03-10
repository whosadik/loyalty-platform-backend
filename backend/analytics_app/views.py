import os
from datetime import datetime, timedelta, timezone
from collections import Counter
from decimal import Decimal

from django.conf import settings
from django.core.cache import cache
from django.db import connection
from django.db.models import Count, Sum
from django.utils import timezone as dj_timezone
from django.utils.dateparse import parse_date
from rest_framework.views import APIView
from rest_framework.response import Response

from offers.models import OfferAssignment, CampaignBudget, OfferEvent
from loyalty.models import LoyaltyAccount, LoyaltyLedgerEntry
from routines.models import RoutineSnapshot
from transactions.models import Transaction

from ml_logic.next_best_reward import compute_rfm, segment
from backend.permissions import HasStaffPermission
from recs_analytics.admin_metrics import recs_metrics_30d
from recs_analytics.models import RecommendationEvent
from offers.admin_metrics import (
    campaigns_metrics_30d,
    offers_events_kpis,
    offers_metrics_30d,
    offers_promo_efficiency_30d,
)


def _parse_datetime_bounds(date_from_raw: str | None, date_to_raw: str | None):
    date_from = parse_date(date_from_raw or "") if date_from_raw else None
    date_to = parse_date(date_to_raw or "") if date_to_raw else None

    dt_from = None
    dt_to = None

    if date_from:
        dt_from = datetime.combine(date_from, datetime.min.time()).replace(tzinfo=timezone.utc)
    if date_to:
        dt_to = datetime.combine(date_to, datetime.max.time()).replace(tzinfo=timezone.utc)

    if dt_from and dt_to and dt_to < dt_from:
        dt_from, dt_to = dt_to, dt_from

    return dt_from, dt_to


def _clean_filter(raw):
    if raw is None:
        return None
    value = str(raw).strip()
    return value or None


def _apply_range(qs, field: str, dt_from, dt_to):
    if dt_from:
        qs = qs.filter(**{f"{field}__gte": dt_from})
    if dt_to:
        qs = qs.filter(**{f"{field}__lte": dt_to})
    return qs


def _offers_events_kpis_from_queryset(events_qs, now):
    def counts(days: int):
        since = now - timedelta(days=days)
        subset = events_qs.filter(created_at__gte=since)
        exposed = subset.filter(event_type=OfferEvent.Type.EXPOSED).count()
        clicked = subset.filter(event_type=OfferEvent.Type.CLICKED).count()
        redeemed = subset.filter(event_type=OfferEvent.Type.REDEEMED).count()
        redemption_rate = (redeemed / exposed) if exposed else 0.0
        ctr = (clicked / exposed) if exposed else 0.0
        return exposed, clicked, redeemed, redemption_rate, ctr

    e7, c7, r7, red_rate7, ctr7 = counts(7)
    e30, c30, r30, red_rate30, ctr30 = counts(30)

    since30 = now - timedelta(days=30)
    agg = {}
    rows = (
        events_qs.filter(
            created_at__gte=since30,
            event_type__in=[OfferEvent.Type.EXPOSED, OfferEvent.Type.CLICKED, OfferEvent.Type.REDEEMED],
        )
        .values("campaign_name", "event_type")
        .annotate(cnt=Count("id"))
        .order_by("campaign_name")
    )
    for row in rows:
        name = row["campaign_name"] or "none"
        bucket = agg.setdefault(name, {"campaign_name": name, "exposed": 0, "clicked": 0, "redeemed": 0})
        if row["event_type"] == OfferEvent.Type.EXPOSED:
            bucket["exposed"] += int(row["cnt"])
        elif row["event_type"] == OfferEvent.Type.CLICKED:
            bucket["clicked"] += int(row["cnt"])
        elif row["event_type"] == OfferEvent.Type.REDEEMED:
            bucket["redeemed"] += int(row["cnt"])

    by_campaign_30d = []
    for item in agg.values():
        exposed = item["exposed"]
        clicked = item["clicked"]
        redeemed = item["redeemed"]
        by_campaign_30d.append(
            {
                **item,
                "ctr_clicks_exposed": round((clicked / exposed), 4) if exposed else 0.0,
                "redemption_rate_exposed": round((redeemed / exposed), 4) if exposed else 0.0,
            }
        )
    by_campaign_30d.sort(key=lambda x: (-x["exposed"], x["campaign_name"]))

    return {
        "exposed_7d": int(e7),
        "clicked_7d": int(c7),
        "redeemed_7d": int(r7),
        "ctr_clicks_exposed_7d": round(ctr7, 4),
        "redemption_rate_exposed_7d": round(red_rate7, 4),
        "exposed_30d": int(e30),
        "clicked_30d": int(c30),
        "redeemed_30d": int(r30),
        "ctr_clicks_exposed_30d": round(ctr30, 4),
        "redemption_rate_exposed_30d": round(red_rate30, 4),
        "by_campaign_30d": by_campaign_30d,
    }


def _offers_v3_from_assignments(assignments_qs, now):
    since = now - timedelta(days=30)
    qs = assignments_qs.filter(assigned_at__gte=since)

    total = qs.count()
    redeemed = qs.filter(is_redeemed=True).count()
    redemption_rate = (redeemed / total) if total else 0.0

    picked = {"bundle": 0, "post_purchase_rules": 0, "fallback": 0, "unknown": 0}
    bundle_mode = {"cooccurrence": 0, "fallback": 0, "unknown": 0}
    cat_dist = {}

    for assignment in qs.only("id", "target", "reason"):
        target = assignment.target or {}
        picked_raw = target.get("picked_via") or (assignment.reason or {}).get("picked_via") or ""
        picked_via = str(picked_raw).lower()
        if picked_via == "bundle":
            picked_via = "bundle"
        elif picked_via.startswith("post_purchase_rules"):
            picked_via = "post_purchase_rules"
        elif picked_via:
            picked_via = "fallback"
        else:
            picked_via = "unknown"
        picked[picked_via] += 1

        if target.get("picked_via") == "bundle":
            mode = target.get("bundle_mode") or "unknown"
            if mode not in bundle_mode:
                mode = "unknown"
            bundle_mode[mode] += 1

        category = target.get("category")
        if category:
            cat_dist[category] = cat_dist.get(category, 0) + 1

    return {
        "assignments_30d": total,
        "redemptions_30d": redeemed,
        "redemption_rate_30d": round(redemption_rate, 4),
        "picked_via_distribution_30d": picked,
        "bundle_mode_distribution_30d": bundle_mode,
        "offer_target_category_distribution_30d": cat_dist,
    }


def _campaigns_metrics_from_assignments(assignments_qs, now):
    since = now - timedelta(days=30)
    rows = (
        assignments_qs.filter(assigned_at__gte=since)
        .values("offer__campaign__name")
        .annotate(assignments=Count("id"))
    )
    # Django cannot use Count("id", filter=...) with dynamic annotation this way in older engines;
    # do it in a second pass for portability.
    grouped = {}
    for row in rows:
        name = row["offer__campaign__name"] or "none"
        grouped[name] = {"campaign": name, "assignments_30d": int(row["assignments"] or 0), "redemptions_30d": 0}

    red_rows = (
        assignments_qs.filter(assigned_at__gte=since, is_redeemed=True)
        .values("offer__campaign__name")
        .annotate(redemptions=Count("id"))
    )
    for row in red_rows:
        name = row["offer__campaign__name"] or "none"
        grouped.setdefault(name, {"campaign": name, "assignments_30d": 0, "redemptions_30d": 0})
        grouped[name]["redemptions_30d"] = int(row["redemptions"] or 0)

    out = []
    for item in grouped.values():
        assignments = int(item["assignments_30d"])
        redemptions = int(item["redemptions_30d"])
        out.append(
            {
                "campaign": item["campaign"],
                "assignments_30d": assignments,
                "redemptions_30d": redemptions,
                "redemption_rate": round(redemptions / assignments, 4) if assignments else 0.0,
            }
        )
    out.sort(key=lambda row: (-row["assignments_30d"], row["campaign"]))

    current = []
    for campaign in CampaignBudget.objects.all().order_by("priority", "name"):
        left = float(Decimal(str(campaign.weekly_limit)) - Decimal(str(campaign.weekly_spent)))
        current.append(
            {
                "campaign": campaign.name,
                "priority": campaign.priority,
                "is_active": bool(campaign.is_active),
                "weekly_limit": float(campaign.weekly_limit),
                "weekly_spent": float(campaign.weekly_spent),
                "weekly_left": left,
            }
        )

    return {"window_days": 30, "current_week": current, "last_30d": out}


def _offers_promo_efficiency_from_events(events_qs, now, *, channel: str | None = None):
    since = now - timedelta(days=30)
    synthetic_channel = "import_synthetic"

    redeemed_assignment_ids_qs = events_qs.filter(
        created_at__gte=since,
        event_type=OfferEvent.Type.REDEEMED,
    ).values("assignment_id")

    assignments_qs = (
        OfferAssignment.objects.filter(id__in=redeemed_assignment_ids_qs)
        .select_related("offer", "offer__campaign")
        .only("id", "redeemed_transaction_id", "offer__estimated_cost", "offer__campaign__name")
    )
    if channel:
        channel_tx_ids = Transaction.objects.filter(channel=channel).values("id")
        assignments_qs = assignments_qs.filter(redeemed_transaction_id__in=channel_tx_ids)

    assignments = list(assignments_qs)
    if not assignments:
        return {
            "window_days": 30,
            "redeemed_count": 0,
            "estimated_cost_total": 0.0,
            "redeemed_revenue_total": 0.0,
            "promo_efficiency": 0.0,
            "redeemed_with_transaction_count": 0,
            "redeemed_without_transaction_count": 0,
            "redeemed_with_real_transaction_count": 0,
            "redeemed_with_synthetic_transaction_count": 0,
            "by_campaign_30d": [],
        }

    txn_ids = [a.redeemed_transaction_id for a in assignments if a.redeemed_transaction_id]
    txns_qs = Transaction.objects.filter(id__in=txn_ids).values_list("id", "total_amount", "channel")
    if channel:
        txns_qs = txns_qs.filter(channel=channel)
    txn_map = {
        int(tx_id): {"amount": Decimal(str(amount or 0)), "channel": str(tx_channel or "")}
        for tx_id, amount, tx_channel in txns_qs
    }

    by_campaign = {}
    est_total = Decimal("0")
    revenue_total = Decimal("0")
    with_txn_total = 0
    without_txn_total = 0
    with_real_txn_total = 0
    with_synthetic_txn_total = 0

    for assignment in assignments:
        campaign_name = getattr(getattr(assignment.offer, "campaign", None), "name", None) or "none"
        bucket = by_campaign.setdefault(
            campaign_name,
            {
                "campaign_name": campaign_name,
                "redeemed_count": 0,
                "estimated_cost_total": Decimal("0"),
                "redeemed_revenue_total": Decimal("0"),
                "redeemed_with_transaction_count": 0,
                "redeemed_without_transaction_count": 0,
                "redeemed_with_real_transaction_count": 0,
                "redeemed_with_synthetic_transaction_count": 0,
            },
        )

        cost = Decimal(str(assignment.offer.estimated_cost or 0))
        bucket["redeemed_count"] += 1
        bucket["estimated_cost_total"] += cost
        est_total += cost

        tx_meta = txn_map.get(int(assignment.redeemed_transaction_id or 0))
        if tx_meta is None:
            bucket["redeemed_without_transaction_count"] += 1
            without_txn_total += 1
            continue

        amount = tx_meta["amount"]
        bucket["redeemed_revenue_total"] += amount
        revenue_total += amount
        bucket["redeemed_with_transaction_count"] += 1
        with_txn_total += 1

        if tx_meta["channel"] == synthetic_channel:
            bucket["redeemed_with_synthetic_transaction_count"] += 1
            with_synthetic_txn_total += 1
        else:
            bucket["redeemed_with_real_transaction_count"] += 1
            with_real_txn_total += 1

    by_campaign_30d = []
    for item in by_campaign.values():
        cost = item["estimated_cost_total"]
        revenue = item["redeemed_revenue_total"]
        by_campaign_30d.append(
            {
                "campaign_name": item["campaign_name"],
                "redeemed_count": int(item["redeemed_count"]),
                "estimated_cost_total": float(cost),
                "redeemed_revenue_total": float(revenue),
                "redeemed_with_transaction_count": int(item["redeemed_with_transaction_count"]),
                "redeemed_without_transaction_count": int(item["redeemed_without_transaction_count"]),
                "redeemed_with_real_transaction_count": int(item["redeemed_with_real_transaction_count"]),
                "redeemed_with_synthetic_transaction_count": int(item["redeemed_with_synthetic_transaction_count"]),
                "promo_efficiency": round(float(revenue / cost), 4) if cost > 0 else 0.0,
            }
        )
    by_campaign_30d.sort(key=lambda x: (-x["redeemed_count"], x["campaign_name"]))
    efficiency = (revenue_total / est_total) if est_total > 0 else Decimal("0")

    return {
        "window_days": 30,
        "redeemed_count": int(len(assignments)),
        "estimated_cost_total": float(est_total),
        "redeemed_revenue_total": float(revenue_total),
        "promo_efficiency": round(float(efficiency), 4) if est_total > 0 else 0.0,
        "redeemed_with_transaction_count": int(with_txn_total),
        "redeemed_without_transaction_count": int(without_txn_total),
        "redeemed_with_real_transaction_count": int(with_real_txn_total),
        "redeemed_with_synthetic_transaction_count": int(with_synthetic_txn_total),
        "by_campaign_30d": by_campaign_30d,
    }


def _recs_metrics_from_queryset(events_qs):
    rows_section = events_qs.values("page", "section_key", "action").annotate(c=Count("id"))
    agg_section = {}
    for row in rows_section:
        key = f'{row["page"]}:{row["section_key"] or "none"}'
        bucket = agg_section.setdefault(key, {"impression": 0, "click": 0, "add_to_cart": 0, "purchase_attributed": 0})
        bucket[row["action"]] = int(row["c"])

    rows_algo = events_qs.values("algo_mode", "action").annotate(c=Count("id"))
    agg_algo = {}
    for row in rows_algo:
        algo = str(row["algo_mode"] or "unknown").strip().lower()
        if algo.startswith("reranker"):
            algo = "reranker"
        elif algo.startswith("cooc") or algo in {"cooccurrence", "fallback", "recommend"}:
            algo = "cooc"
        elif not algo:
            algo = "unknown"
        bucket = agg_algo.setdefault(algo, {"impression": 0, "click": 0, "add_to_cart": 0, "purchase_attributed": 0})
        bucket[row["action"]] = int(row["c"])

    def with_rates(raw):
        impression = int(raw.get("impression", 0) or 0)
        click = int(raw.get("click", 0) or 0)
        purchase = int(raw.get("purchase_attributed", 0) or 0)
        return {
            "impression": impression,
            "click": click,
            "add_to_cart": int(raw.get("add_to_cart", 0) or 0),
            "purchase_attributed": purchase,
            "ctr": round(click / impression, 4) if impression else 0.0,
            "conversion": round(purchase / impression, 4) if impression else 0.0,
        }

    return {
        "window_days": 30,
        "by_section": {k: with_rates(v) for k, v in agg_section.items()},
        "by_algo": {k: with_rates(v) for k, v in agg_algo.items()},
        "by_experiment": {},
    }

class AdminMetricsView(APIView):
    permission_classes = [HasStaffPermission.with_perm("view_metrics")]

    def get(self, request):
        date_from_raw = _clean_filter(request.query_params.get("date_from"))
        date_to_raw = _clean_filter(request.query_params.get("date_to"))
        category = _clean_filter(request.query_params.get("category"))
        offer_type = _clean_filter(request.query_params.get("offer_type"))
        channel = _clean_filter(request.query_params.get("channel"))

        dt_from, dt_to = _parse_datetime_bounds(date_from_raw, date_to_raw)
        has_filtered_request = any(
            value is not None for value in (dt_from, dt_to, category, offer_type, channel)
        )

        ttl = int(getattr(settings, "ADMIN_METRICS_CACHE_TTL_SECONDS", 60))
        db_name = connection.settings_dict.get("NAME", "default")
        filter_signature = "|".join(
            [
                date_from_raw or "",
                date_to_raw or "",
                category or "",
                offer_type or "",
                channel or "",
            ]
        )
        cache_key = f"admin:metrics:v2:{db_name}:{os.getpid()}:{filter_signature}"
        if ttl > 0:
            cached = cache.get(cache_key)
            if cached is not None:
                return Response(cached)

        now = dt_to or dj_timezone.now()
        since_7d = now - timedelta(days=7)
        since_30d = now - timedelta(days=30)
        if has_filtered_request:
            assignments_qs = OfferAssignment.objects.all()
            events_qs = OfferEvent.objects.all()
            transactions_qs = Transaction.objects.all()
            recs_qs = RecommendationEvent.objects.select_related("product").all()
            routines_qs = RoutineSnapshot.objects.all()
            ledger_qs = LoyaltyLedgerEntry.objects.all()

            assignments_qs = _apply_range(assignments_qs, "assigned_at", dt_from, dt_to)
            events_qs = _apply_range(events_qs, "created_at", dt_from, dt_to)
            transactions_qs = _apply_range(transactions_qs, "created_at", dt_from, dt_to)
            recs_qs = _apply_range(recs_qs, "created_at", dt_from, dt_to)
            routines_qs = _apply_range(routines_qs, "created_at", dt_from, dt_to)
            ledger_qs = _apply_range(ledger_qs, "created_at", dt_from, dt_to)

            if category:
                assignments_qs = assignments_qs.filter(target__category=category)
                events_qs = events_qs.filter(assignment__target__category=category)
                tx_ids_by_category = Transaction.objects.filter(items__product__category=category).values("id")
                transactions_qs = transactions_qs.filter(id__in=tx_ids_by_category)
                recs_qs = recs_qs.filter(product__category=category)

            if offer_type:
                assignments_qs = assignments_qs.filter(offer__offer_type=offer_type)
                events_qs = events_qs.filter(offer__offer_type=offer_type)

            if channel:
                transactions_qs = transactions_qs.filter(channel=channel)
                channel_tx_ids = Transaction.objects.filter(channel=channel).values("id")
                assignments_qs = assignments_qs.filter(redeemed_transaction_id__in=channel_tx_ids)
                events_qs = events_qs.filter(assignment__redeemed_transaction_id__in=channel_tx_ids)

            assignments_total = assignments_qs.count()
            redemptions_total = assignments_qs.filter(is_redeemed=True).count()
            redemption_rate = (redemptions_total / assignments_total) if assignments_total else 0.0

            assignments_7d = assignments_qs.filter(assigned_at__gte=since_7d).count()
            redemptions_7d = assignments_qs.filter(assigned_at__gte=since_7d, is_redeemed=True).count()

            earned_points = (
                ledger_qs.filter(entry_type=LoyaltyLedgerEntry.Type.EARN)
                .aggregate(s=Sum("points_delta"))["s"]
                or 0
            )
            redeemed_points = (
                ledger_qs.filter(entry_type=LoyaltyLedgerEntry.Type.REDEEM)
                .aggregate(s=Sum("points_delta"))["s"]
                or 0
            )

            snapshots = routines_qs.filter(created_at__gte=since_30d).values_list("missing_steps", flat=True)
            c = Counter()
            for ms in snapshots:
                for step in (ms or []):
                    c[step] += 1
            top_missing_steps = [{"step": k, "count": v} for k, v in c.most_common(10)]

            user_ids = list(
                transactions_qs.filter(created_at__gte=since_30d).values_list("user_id", flat=True).distinct()
            )
            seg_counter = Counter()
            for uid in user_ids[:500]:
                txs = list(transactions_qs.filter(user_id=uid).values("created_at", "total_amount"))
                rfm = compute_rfm(txs, now)
                seg_counter[segment(rfm)] += 1
            segment_distribution = [{"segment": k, "count": v} for k, v in seg_counter.most_common()]

            offers_v3_payload = _offers_v3_from_assignments(assignments_qs, now)
            events_kpis_payload = _offers_events_kpis_from_queryset(events_qs, now)
            promo_efficiency_payload = _offers_promo_efficiency_from_events(events_qs, now, channel=channel)
            campaigns_payload = _campaigns_metrics_from_assignments(assignments_qs, now)
            recs_payload = _recs_metrics_from_queryset(recs_qs.filter(created_at__gte=since_30d))
        else:
            # Offers
            assignments_total = OfferAssignment.objects.count()
            redemptions_total = OfferAssignment.objects.filter(is_redeemed=True).count()
            redemption_rate = (redemptions_total / assignments_total) if assignments_total else 0.0

            assignments_7d = OfferAssignment.objects.filter(assigned_at__gte=since_7d).count()
            redemptions_7d = OfferAssignment.objects.filter(assigned_at__gte=since_7d, is_redeemed=True).count()

        # Budget
        budget, _ = CampaignBudget.objects.get_or_create(
            name="default",
            defaults={"weekly_limit": 1000, "weekly_spent": 0},
        )

        budget_left = float(budget.weekly_limit) - float(budget.weekly_spent)

        # Loyalty: ledger sums
        if not has_filtered_request:
            earned_points = (
                LoyaltyLedgerEntry.objects.filter(entry_type=LoyaltyLedgerEntry.Type.EARN)
                .aggregate(s=Sum("points_delta"))["s"]
                or 0
            )
            redeemed_points = (
                LoyaltyLedgerEntry.objects.filter(entry_type=LoyaltyLedgerEntry.Type.REDEEM)
                .aggregate(s=Sum("points_delta"))["s"]
                or 0
            )
        # redeemed_points is negative
        redeemed_points_abs = abs(int(redeemed_points))

        # Loyalty: tiers distribution
        tier_dist_qs = (
            LoyaltyAccount.objects.select_related("tier")
            .values("tier__name")
            .annotate(cnt=Count("id"))
            .order_by("-cnt")
        )
        tier_distribution = {row["tier__name"] or "None": row["cnt"] for row in tier_dist_qs}

        # Routine: top missing steps (last 30 days)
        if not has_filtered_request:
            snapshots = RoutineSnapshot.objects.filter(created_at__gte=since_30d).values_list("missing_steps", flat=True)
            c = Counter()
            for ms in snapshots:
                for step in (ms or []):
                    c[step] += 1
            top_missing_steps = [{"step": k, "count": v} for k, v in c.most_common(10)]

        # Segments distribution (approx): compute from last 90 days transactions per user
        # MVP: compute for users that have at least one transaction in last 30 days
        if not has_filtered_request:
            user_ids = list(
                Transaction.objects.filter(created_at__gte=since_30d).values_list("user_id", flat=True).distinct()
            )
            seg_counter = Counter()
            for uid in user_ids[:500]:  # hard cap to keep it fast
                txs = list(
                    Transaction.objects.filter(user_id=uid).values("created_at", "total_amount")
                )
                rfm = compute_rfm(txs, now)
                seg_counter[segment(rfm)] += 1

            segment_distribution = [{"segment": k, "count": v} for k, v in seg_counter.most_common()]

        def repeat_rate(days: int):
            since = now - timedelta(days=days)
            if has_filtered_request:
                per_user = (
                    transactions_qs.filter(created_at__gte=since)
                    .values("user_id")
                    .annotate(txn_count=Count("id"))
                )
                active_users = per_user.count()
                repeat_users = per_user.filter(txn_count__gte=2).count()
                rate = (repeat_users / active_users) if active_users else 0.0
                return active_users, repeat_users, rate

            per_user = (
                Transaction.objects.filter(created_at__gte=since)
                .values("user_id")
                .annotate(txn_count=Count("id"))
            )
            active_users = per_user.count()
            repeat_users = per_user.filter(txn_count__gte=2).count()
            rate = (repeat_users / active_users) if active_users else 0.0
            return active_users, repeat_users, rate

        au30, ru30, rr30 = repeat_rate(30)
        au60, ru60, rr60 = repeat_rate(60)
        au90, ru90, rr90 = repeat_rate(90)

        payload = {
            "offers": {
                "assignments_total": assignments_total,
                "redemptions_total": redemptions_total,
                "redemption_rate": round(redemption_rate, 4),
                "assignments_7d": assignments_7d,
                "redemptions_7d": redemptions_7d,
                "offers_v3": offers_v3_payload if has_filtered_request else offers_metrics_30d(),
                "events_kpis": events_kpis_payload if has_filtered_request else offers_events_kpis(),
                "promo_efficiency_30d": promo_efficiency_payload if has_filtered_request else offers_promo_efficiency_30d(),
            },
            "budget": {
                "weekly_limit": float(budget.weekly_limit),
                "weekly_spent": float(budget.weekly_spent),
                "weekly_left": float(budget_left),
            },
            "loyalty": {
                "earned_points_total": int(earned_points),
                "redeemed_points_total": int(redeemed_points_abs),
                "tier_distribution": tier_distribution,
            },
            "routines": {
                "top_missing_steps_30d": top_missing_steps,
            },
            "segments": {
                "distribution_30d": segment_distribution,
                "users_sampled": len(user_ids[:500]),
            },
            "retention": {
                "repeat_purchase_rate_30d": round(rr30, 4),
                "repeat_purchase_rate_60d": round(rr60, 4),
                "repeat_purchase_rate_90d": round(rr90, 4),
                "active_users_30d": int(au30),
                "repeat_users_30d": int(ru30),
                "active_users_60d": int(au60),
                "repeat_users_60d": int(ru60),
                "active_users_90d": int(au90),
                "repeat_users_90d": int(ru90),
            },
            "recs": recs_payload if has_filtered_request else recs_metrics_30d(),
            "campaigns": campaigns_payload if has_filtered_request else campaigns_metrics_30d(),
        }
        if ttl > 0:
            cache.set(cache_key, payload, timeout=ttl)
        return Response(payload)
