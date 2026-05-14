"""Manual image fill: a tiny admin page + endpoint for pasting image URLs.

A separate file so it doesn't tangle with the main views. The page is server-
rendered HTML at /api/catalog/admin/manual-images/, and submissions POST to
/api/catalog/admin/attach_image_by_url/.

Auth: staff users only (Django session auth — log in via /admin/).
"""
from __future__ import annotations

import re
import urllib.parse
from pathlib import Path

import requests
from django.conf import settings
from django.contrib.admin.views.decorators import staff_member_required
from django.http import HttpRequest, HttpResponse, JsonResponse
from django.shortcuts import render
from django.utils.decorators import method_decorator
from django.views import View
from django.views.decorators.csrf import csrf_exempt

from .models import Product


_IMAGE_EXTS = (".webp", ".jpg", ".jpeg", ".png", ".avif", ".gif")
_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    ),
    "Accept": "image/avif,image/webp,image/*,*/*",
}


def _pick_extension(url: str, content_type: str) -> str:
    """Pick a sensible extension from URL or response Content-Type."""
    m = re.search(r"\.(webp|jpg|jpeg|png|avif|gif)(?:\?|$)", url, flags=re.I)
    if m:
        return "." + m.group(1).lower()
    ct = (content_type or "").lower().split(";")[0].strip()
    mapping = {
        "image/webp": ".webp",
        "image/jpeg": ".jpg",
        "image/jpg": ".jpg",
        "image/png": ".png",
        "image/avif": ".avif",
        "image/gif": ".gif",
    }
    return mapping.get(ct, ".jpg")


def _download_to(product_id: int, url: str) -> tuple[bool, str, str]:
    """Download `url` and save as media/products/<product_id>.<ext>.

    Returns (ok, relative_path, message).
    """
    referer = "/".join(url.split("/")[:3]) + "/" if url.startswith("http") else ""
    headers = dict(_HEADERS)
    if referer:
        headers["Referer"] = referer
    try:
        resp = requests.get(url, headers=headers, timeout=20, stream=True)
    except requests.RequestException as e:
        return False, "", f"network error: {e}"
    if resp.status_code != 200:
        return False, "", f"HTTP {resp.status_code}"
    ext = _pick_extension(url, resp.headers.get("Content-Type", ""))
    if ext not in _IMAGE_EXTS:
        return False, "", f"unsupported content-type: {resp.headers.get('Content-Type')}"

    out_dir = Path(settings.MEDIA_ROOT) / "products"
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / f"{product_id}{ext}"
    tmp = dest.with_suffix(dest.suffix + ".part")
    size = 0
    with tmp.open("wb") as fh:
        for chunk in resp.iter_content(8192):
            if chunk:
                fh.write(chunk)
                size += len(chunk)
    if size < 1024:
        tmp.unlink(missing_ok=True)
        return False, "", f"file too small ({size}b)"
    # Clean sibling files with other extensions so the gallery is consistent.
    for other in _IMAGE_EXTS:
        sib = out_dir / f"{product_id}{other}"
        if sib != dest and sib.exists():
            sib.unlink()
    tmp.replace(dest)
    return True, f"products/{product_id}{ext}", f"saved {size//1024} KB"


@method_decorator(csrf_exempt, name="dispatch")
@method_decorator(staff_member_required, name="dispatch")
class AttachImageByUrlView(View):
    """POST {product_id, image_url} → downloads and attaches to product."""

    def post(self, request: HttpRequest) -> JsonResponse:
        try:
            import json

            payload = json.loads(request.body or "{}")
        except ValueError:
            return JsonResponse({"ok": False, "error": "invalid JSON"}, status=400)
        product_id = payload.get("product_id")
        url = (payload.get("image_url") or "").strip()
        if not product_id or not url:
            return JsonResponse({"ok": False, "error": "product_id and image_url required"}, status=400)
        try:
            product = Product.objects.get(id=int(product_id))
        except (Product.DoesNotExist, ValueError, TypeError):
            return JsonResponse({"ok": False, "error": "product not found"}, status=404)

        ok, rel_path, msg = _download_to(product.id, url)
        if not ok:
            return JsonResponse({"ok": False, "error": msg}, status=400)
        product.image.name = rel_path
        product.image_url = f"{settings.MEDIA_URL}{rel_path}"
        # Drop any stale extra image_urls so the gallery starts fresh.
        product.image_urls = [product.image_url]
        product.save(update_fields=["image", "image_url", "image_urls", "updated_at"])
        return JsonResponse(
            {
                "ok": True,
                "image_url": product.image_url,
                "size_msg": msg,
            }
        )


@staff_member_required
def manual_images_page(request: HttpRequest) -> HttpResponse:
    """Server-rendered HTML page for pasting image URLs."""
    mode = (request.GET.get("mode") or "needs").lower()
    brand_filter = (request.GET.get("brand") or "").strip()

    qs = Product.objects.all().order_by("brand", "name")
    if mode == "needs":
        # "Needs image": no image file at all OR image_url points to a remote
        # URL (the stale pcdn.goldapple.ru ones are the bulk).
        local_prefix = settings.MEDIA_URL
        media_root = Path(settings.MEDIA_ROOT)
        products: list[Product] = []
        for p in qs:
            url = p.image_url or ""
            if not url:
                products.append(p)
                continue
            if not url.startswith(local_prefix):
                products.append(p)
                continue
            # local — check file actually exists
            rel = url.replace(local_prefix, "", 1).lstrip("/")
            if not (media_root / rel).exists():
                products.append(p)
        qs = products  # type: ignore[assignment]
    else:
        qs = list(qs)

    if brand_filter:
        qs = [p for p in qs if (p.brand or "").lower() == brand_filter.lower()]

    brands = sorted({(p.brand or "").strip() for p in Product.objects.all() if p.brand})

    rows: list[dict] = []
    for p in qs[:500]:
        current = p.image_url or ""
        # Quote the search queries (brand+name) once per row.
        q_brand_name = urllib.parse.quote(f"{p.brand} {p.name}".strip())
        rows.append({
            "id": p.id,
            "brand": p.brand or "",
            "name": p.name or "",
            "category": p.category or "",
            "product_type": p.product_type or "",
            "current_image_url": current,
            "source_product_id": p.source_product_id or "",
            "search_ga": f"https://goldapple.kz/search?q={q_brand_name}",
            "search_google": f"https://www.google.com/search?q={q_brand_name}&tbm=isch",
            "search_wb": f"https://www.wildberries.ru/catalog/0/search.aspx?search={q_brand_name}",
            "search_yandex": f"https://yandex.com/images/search?text={q_brand_name}",
        })

    return render(
        request,
        "catalog/manual_images.html",
        {
            "rows": rows,
            "total_count": len(rows),
            "all_count": Product.objects.count(),
            "mode": mode,
            "brand_filter": brand_filter,
            "brands": brands,
        },
    )
