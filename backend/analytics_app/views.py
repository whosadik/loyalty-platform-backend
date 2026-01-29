from datetime import datetime, timedelta, timezone
from collections import Counter

from django.db.models import Count, Sum
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.permissions import IsAdminUser

from offers.models import OfferAssignment, CampaignBudget
from loyalty.models import LoyaltyAccount, LoyaltyLedgerEntry
from routines.models import RoutineSnapshot
from transactions.models import Transaction

from ml_logic.next_best_reward import compute_rfm, segment
from offers.admin_metrics import offers_metrics_30d
from backend.permissions import HasStaffPermission
from recs_analytics.admin_metrics import recs_metrics_30d

class AdminMetricsView(APIView):
    permission_classes = [HasStaffPermission.with_perm("view_metrics")]

    def get(self, request):
        now = datetime.now(timezone.utc)
        since_7d = now - timedelta(days=7)
        since_30d = now - timedelta(days=30)

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
        snapshots = RoutineSnapshot.objects.filter(created_at__gte=since_30d).values_list("missing_steps", flat=True)
        c = Counter()
        for ms in snapshots:
            for step in (ms or []):
                c[step] += 1
        top_missing_steps = [{"step": k, "count": v} for k, v in c.most_common(10)]

        # Segments distribution (approx): compute from last 90 days transactions per user
        # MVP: compute for users that have at least one transaction in last 30 days
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

        return Response(
            {
                "offers": {
                    "assignments_total": assignments_total,
                    "redemptions_total": redemptions_total,
                    "redemption_rate": round(redemption_rate, 4),
                    "assignments_7d": assignments_7d,
                    "redemptions_7d": redemptions_7d,
                    "offers_v3": offers_metrics_30d(),
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
                "recs": recs_metrics_30d(),
            }
        )
