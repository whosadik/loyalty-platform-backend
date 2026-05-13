from __future__ import annotations

import os
import uuid
from datetime import timedelta
from decimal import Decimal, InvalidOperation

from django.conf import settings
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
from django.db import transaction as db_tx
from django.db.models import DecimalField, ExpressionWrapper, F, IntegerField, Q, Sum, Value
from django.db.models.functions import Coalesce
from django.utils import timezone

from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import serializers, status
from rest_framework.parsers import MultiPartParser, FormParser

from drf_spectacular.types import OpenApiTypes
from drf_spectacular.utils import OpenApiParameter, OpenApiResponse, extend_schema, inline_serializer

from backend.api_serializers import ApiErrorSerializer
from backend.permissions import HasStaffPermission
from catalog.models import Product
from offers.models import CampaignBudget
from offers.serializers_admin import CampaignSerializer


CampaignDetailResponseSerializer = inline_serializer(
    name="AdminCampaignDetailResponse",
    fields={
        "ok": serializers.BooleanField(),
        "campaign": CampaignSerializer(),
    },
)

CampaignListResponseSerializer = inline_serializer(
    name="AdminCampaignListResponse",
    fields={
        "ok": serializers.BooleanField(),
        "results": CampaignSerializer(many=True),
    },
)

CampaignPatchRequestSerializer = inline_serializer(
    name="AdminCampaignPatchRequest",
    fields={
        "name": serializers.CharField(required=False),
        "is_active": serializers.BooleanField(required=False),
        "campaign_type": serializers.ChoiceField(choices=CampaignBudget.Type.choices, required=False),
        "priority": serializers.IntegerField(required=False),
        "weekly_limit": serializers.DecimalField(required=False, max_digits=12, decimal_places=2),
        "start_date": serializers.DateField(required=False, allow_null=True),
        "end_date": serializers.DateField(required=False, allow_null=True),
        "allowed_categories": serializers.ListField(child=serializers.CharField(), required=False),
        "allowed_steps": serializers.ListField(child=serializers.CharField(), required=False),
        "allowed_brands": serializers.ListField(child=serializers.CharField(), required=False),
        "allowed_product_ids": serializers.ListField(child=serializers.IntegerField(), required=False),
        "tiers": serializers.ListField(child=serializers.CharField(), required=False),
        "recommendation_rules": serializers.JSONField(required=False),
        "promo_text": serializers.CharField(required=False, allow_blank=True),
        "banner_url": serializers.URLField(required=False, allow_blank=True),
        "reset_weekly_spent": serializers.BooleanField(required=False),
    },
)


class AdminCampaignListCreateView(APIView):
    """
    GET: requires view_metrics
    POST: requires manage_campaigns
    """

    def get_permissions(self):
        if self.request.method == "POST":
            return [HasStaffPermission.with_perm("manage_campaigns")()]
        return [HasStaffPermission.with_perm("view_metrics")()]

    @extend_schema(
        tags=["Admin"],
        description="List campaigns (budgets) with optional filters.",
        parameters=[
            OpenApiParameter(name="is_active", required=False, type=bool),
            OpenApiParameter(name="campaign_type", required=False, type=str, description="personal|public"),
            OpenApiParameter(name="name", required=False, type=str, description="substring match"),
            OpenApiParameter(name="ordering", required=False, type=str, description="priority|name|-priority|-name"),
        ],
        responses={
            200: CampaignListResponseSerializer,
        },
    )
    def get(self, request):
        qs = CampaignBudget.objects.all()

        is_active = request.query_params.get("is_active")
        if is_active is not None:
            qs = qs.filter(is_active=str(is_active).lower() in {"1", "true", "yes", "on"})

        campaign_type = request.query_params.get("campaign_type")
        if campaign_type in {CampaignBudget.Type.PERSONAL, CampaignBudget.Type.PUBLIC}:
            qs = qs.filter(campaign_type=campaign_type)

        name = request.query_params.get("name")
        if name:
            qs = qs.filter(name__icontains=name)

        ordering = request.query_params.get("ordering") or "priority"
        if ordering in {"priority", "-priority", "name", "-name", "id", "-id"}:
            qs = qs.order_by(ordering, "id")
        else:
            qs = qs.order_by("priority", "id")

        data = CampaignSerializer(qs, many=True).data
        return Response({"ok": True, "results": data})

    @extend_schema(
        tags=["Admin"],
        description="Create campaign (budget).",
        request=CampaignSerializer,
        responses={
            201: CampaignDetailResponseSerializer,
            400: OpenApiResponse(response=ApiErrorSerializer),
        },
    )
    def post(self, request):
        s = CampaignSerializer(data=request.data)
        s.is_valid(raise_exception=True)
        c = s.save()
        return Response({"ok": True, "campaign": CampaignSerializer(c).data}, status=status.HTTP_201_CREATED)


class AdminCampaignDetailView(APIView):
    """
    GET: requires view_metrics
    PATCH: requires manage_campaigns
    """

    def get_permissions(self):
        if self.request.method == "PATCH":
            return [HasStaffPermission.with_perm("manage_campaigns")()]
        return [HasStaffPermission.with_perm("view_metrics")()]

    def _get(self, pk: int) -> CampaignBudget:
        return CampaignBudget.objects.get(pk=pk)

    @extend_schema(
        tags=["Admin"],
        description="Get campaign details.",
        responses={200: CampaignDetailResponseSerializer},
    )
    def get(self, request, pk: int):
        c = self._get(pk)
        return Response({"ok": True, "campaign": CampaignSerializer(c).data})

    @extend_schema(
        tags=["Admin"],
        description="Patch campaign fields. Optional: reset_weekly_spent=true resets weekly_spent and week_start_date.",
        request=CampaignPatchRequestSerializer,
        responses={
            200: CampaignDetailResponseSerializer,
            400: OpenApiResponse(response=ApiErrorSerializer),
        },
    )
    def patch(self, request, pk: int):
        with db_tx.atomic():
            c = CampaignBudget.objects.select_for_update().get(pk=pk)

            reset = request.data.get("reset_weekly_spent")
            if reset in {True, "true", "1", 1, "yes", "on"}:
                # reset to current week start if field exists
                if hasattr(c, "week_start_date"):
                    now = timezone.now()
                    ws = (now - timedelta(days=now.weekday())).date()
                    c.week_start_date = ws
                c.weekly_spent = 0
                c.save(update_fields=["weekly_spent"] + (["week_start_date"] if hasattr(c, "week_start_date") else []))

            serializer_data = request.data.copy()
            serializer_data.pop("reset_weekly_spent", None)
            s = CampaignSerializer(instance=c, data=serializer_data, partial=True)
            s.is_valid(raise_exception=True)
            c = s.save()

        return Response({"ok": True, "campaign": CampaignSerializer(c).data})


ALLOWED_BANNER_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".gif"}
MAX_BANNER_SIZE_BYTES = 5 * 1024 * 1024  # 5 MB


class AdminCampaignBannerUploadView(APIView):
    """
    POST multipart/form-data with field `file` — saves image to MEDIA_ROOT/campaign_banners/
    and sets campaign.banner_url to its URL.
    """

    permission_classes = [HasStaffPermission.with_perm("manage_campaigns")]
    parser_classes = [MultiPartParser, FormParser]

    @extend_schema(
        tags=["Admin"],
        description="Upload campaign banner image. Returns updated campaign.",
        responses={
            200: CampaignDetailResponseSerializer,
            400: OpenApiResponse(response=ApiErrorSerializer),
        },
    )
    def post(self, request, pk: int):
        upload = request.FILES.get("file")
        if upload is None:
            return Response({"ok": False, "message": "file is required"}, status=400)

        ext = os.path.splitext(upload.name)[1].lower()
        if ext not in ALLOWED_BANNER_EXTENSIONS:
            return Response(
                {"ok": False, "message": f"unsupported extension {ext}; allowed: {sorted(ALLOWED_BANNER_EXTENSIONS)}"},
                status=400,
            )

        if upload.size > MAX_BANNER_SIZE_BYTES:
            return Response(
                {"ok": False, "message": f"file too large ({upload.size} bytes); max {MAX_BANNER_SIZE_BYTES}"},
                status=400,
            )

        with db_tx.atomic():
            campaign = CampaignBudget.objects.select_for_update().get(pk=pk)
            filename = f"campaign_banners/{pk}_{uuid.uuid4().hex}{ext}"
            saved_path = default_storage.save(filename, ContentFile(upload.read()))

            media_url = settings.MEDIA_URL
            if not media_url.endswith("/"):
                media_url += "/"
            relative_url = f"{media_url}{saved_path}"
            absolute_url = request.build_absolute_uri(relative_url)

            campaign.banner_url = absolute_url
            campaign.save(update_fields=["banner_url"])

        return Response({"ok": True, "campaign": CampaignSerializer(campaign).data})


class AdminCampaignPublishView(APIView):
    """
    POST: requires manage_campaigns
    """

    permission_classes = [HasStaffPermission.with_perm("manage_campaigns")]

    @extend_schema(
        tags=["Admin"],
        description="Publish campaign by activating it.",
        responses={200: CampaignDetailResponseSerializer},
    )
    def post(self, request, pk: int):
        with db_tx.atomic():
            campaign = CampaignBudget.objects.select_for_update().get(pk=pk)

            update_fields = []
            if not campaign.is_active:
                campaign.is_active = True
                update_fields.append("is_active")

            if hasattr(campaign, "week_start_date") and campaign.week_start_date is None:
                now = timezone.now()
                campaign.week_start_date = (now - timedelta(days=now.weekday())).date()
                update_fields.append("week_start_date")

            if update_fields:
                campaign.save(update_fields=update_fields)

        return Response({"ok": True, "campaign": CampaignSerializer(campaign).data})


class AdminCampaignRecommendationsView(APIView):
    """
    Finds slow-moving products that match a campaign scope and returns discount/action recommendations.
    """

    permission_classes = [HasStaffPermission.with_perm("view_metrics")]

    @extend_schema(
        tags=["Admin"],
        description="Recommend products/brands for a campaign when sales are below KPI.",
        parameters=[
            OpenApiParameter(name="period_days", required=False, type=int),
            OpenApiParameter(name="min_units_sold", required=False, type=int),
            OpenApiParameter(name="min_revenue", required=False, type=str),
            OpenApiParameter(name="category", required=False, type=str),
            OpenApiParameter(name="brand", required=False, type=str),
            OpenApiParameter(name="limit", required=False, type=int),
        ],
        responses={200: OpenApiTypes.OBJECT},
    )
    def get(self, request, pk: int):
        campaign = CampaignBudget.objects.get(pk=pk)
        rules = campaign.recommendation_rules if isinstance(campaign.recommendation_rules, dict) else {}

        period_days = _positive_int(
            request.query_params.get("period_days", rules.get("period_days")),
            default=30,
            minimum=1,
            maximum=365,
        )
        min_units_sold = _positive_int(
            request.query_params.get("min_units_sold", rules.get("min_units_sold")),
            default=3,
            minimum=0,
            maximum=100000,
        )
        min_revenue = _decimal_value(
            request.query_params.get("min_revenue", rules.get("min_revenue")),
            default=Decimal("0"),
        )
        limit = _positive_int(request.query_params.get("limit"), default=20, minimum=1, maximum=100)
        since = timezone.now() - timedelta(days=period_days)

        qs = Product.objects.filter(in_stock=True)
        campaign_categories = [str(x).strip() for x in (campaign.allowed_categories or []) if str(x).strip()]
        campaign_brands = [str(x).strip() for x in (getattr(campaign, "allowed_brands", []) or []) if str(x).strip()]
        campaign_product_ids = _clean_int_list(getattr(campaign, "allowed_product_ids", []) or [])
        campaign_product_types = [str(x).strip() for x in (campaign.allowed_steps or []) if str(x).strip()]

        category = (request.query_params.get("category") or "").strip()
        brand = (request.query_params.get("brand") or "").strip()

        if campaign_product_ids:
            qs = qs.filter(id__in=campaign_product_ids)
        if category:
            qs = qs.filter(category=category)
        elif campaign_categories:
            qs = qs.filter(category__in=campaign_categories)
        if brand:
            qs = qs.filter(brand__iexact=brand)
        elif campaign_brands:
            brand_q = Q()
            for item in campaign_brands:
                brand_q |= Q(brand__iexact=item)
            qs = qs.filter(brand_q)
        if campaign_product_types:
            qs = qs.filter(product_type__in=campaign_product_types)

        revenue_expr = ExpressionWrapper(
            F("transactionitem__quantity") * F("transactionitem__unit_price"),
            output_field=DecimalField(max_digits=14, decimal_places=2),
        )
        qs = qs.annotate(
            units_sold=Coalesce(
                Sum(
                    "transactionitem__quantity",
                    filter=Q(transactionitem__transaction__created_at__gte=since),
                ),
                Value(0),
                output_field=IntegerField(),
            ),
            revenue=Coalesce(
                Sum(
                    revenue_expr,
                    filter=Q(transactionitem__transaction__created_at__gte=since),
                ),
                Value(Decimal("0")),
                output_field=DecimalField(max_digits=14, decimal_places=2),
            ),
        ).filter(Q(units_sold__lt=min_units_sold) | Q(revenue__lt=min_revenue))

        products = list(qs.order_by("units_sold", "revenue", "-id")[:limit])
        product_payload = [_recommendation_product_payload(p, period_days, min_units_sold, min_revenue) for p in products]
        brand_payload = _brand_recommendation_payload(product_payload)

        return Response(
            {
                "ok": True,
                "campaign_id": campaign.id,
                "rules": {
                    "period_days": period_days,
                    "min_units_sold": min_units_sold,
                    "min_revenue": str(min_revenue),
                },
                "count": len(product_payload),
                "products": product_payload,
                "brands": brand_payload,
            }
        )


def _positive_int(value, *, default: int, minimum: int, maximum: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = default
    return max(minimum, min(maximum, parsed))


def _decimal_value(value, *, default: Decimal) -> Decimal:
    try:
        parsed = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError):
        parsed = default
    if not parsed.is_finite() or parsed < 0:
        return default
    return parsed


def _clean_int_list(values) -> list[int]:
    out: list[int] = []
    for value in values or []:
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            continue
        if parsed > 0 and parsed not in out:
            out.append(parsed)
    return out


def _recommendation_product_payload(
    product: Product,
    period_days: int,
    min_units_sold: int,
    min_revenue: Decimal,
) -> dict:
    units_sold = int(getattr(product, "units_sold", 0) or 0)
    revenue = Decimal(str(getattr(product, "revenue", Decimal("0")) or "0"))
    if units_sold == 0:
        discount = 15
    elif min_units_sold > 0 and units_sold <= max(1, min_units_sold // 2):
        discount = 10
    else:
        discount = 5
    reasons = []
    if units_sold < min_units_sold:
        reasons.append(f"sold {units_sold} units in {period_days}d, KPI {min_units_sold}")
    if revenue < min_revenue:
        reasons.append(f"revenue {revenue} below KPI {min_revenue}")
    return {
        "product_id": product.id,
        "name": product.name,
        "brand": product.brand,
        "category": product.category,
        "product_type": product.product_type,
        "price": str(product.price) if product.price is not None else None,
        "units_sold": units_sold,
        "revenue": str(revenue),
        "recommended_action": "discount",
        "recommended_discount_percent": discount,
        "reason": "; ".join(reasons) or "below campaign KPI",
    }


def _brand_recommendation_payload(products: list[dict]) -> list[dict]:
    buckets: dict[str, dict] = {}
    for item in products:
        brand = str(item.get("brand") or "").strip()
        if not brand:
            continue
        bucket = buckets.setdefault(
            brand,
            {
                "brand": brand,
                "products_count": 0,
                "product_ids": [],
                "units_sold": 0,
                "revenue": Decimal("0"),
                "recommended_discount_percent": 5,
            },
        )
        bucket["products_count"] += 1
        bucket["product_ids"].append(item["product_id"])
        bucket["units_sold"] += int(item.get("units_sold") or 0)
        bucket["revenue"] += Decimal(str(item.get("revenue") or "0"))
        bucket["recommended_discount_percent"] = max(
            int(bucket["recommended_discount_percent"]),
            int(item.get("recommended_discount_percent") or 0),
        )

    out = []
    for item in buckets.values():
        out.append(
            {
                **item,
                "revenue": str(item["revenue"]),
                "reason": f"{item['products_count']} slow products in this brand",
            }
        )
    out.sort(key=lambda x: (-int(x["products_count"]), int(x["units_sold"]), x["brand"]))
    return out
