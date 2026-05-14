"""Fetch product images from Goldapple → Osima → Wildberries.

For every product the command tries each source in order until one returns a
usable image set. Images are saved to ``MEDIA_ROOT/products/<id>.<ext>`` AND
attached to ``Product.image`` (Django ImageField). That way the catalog +
media folder + DB dump can be moved between hosts unchanged: as long as the
new host has the same ``media/products/...`` tree, every image keeps working.

Run examples
------------
    # Full catalog, normal order (GA → Osima → WB):
    python manage.py fetch_product_images

    # Only products missing an image, skip GA (e.g. anti-bot blocking):
    python manage.py fetch_product_images --skip-existing --skip-ga

    # Test on a subset:
    python manage.py fetch_product_images --ids 1627,1660 --headless

    # Dry-run (no DB / disk writes) to verify selectors:
    python manage.py fetch_product_images --limit 5 --dry-run

Run notes
---------
Goldapple is protected by a Group-IB anti-bot challenge. The command launches
a *headed* Chromium with a persistent profile (``.ga_browser_profile/`` in the
repo root). The first time you run it, solve any captcha in the visible
browser window — the challenge cookies are stored in the profile, so
subsequent runs proceed without further interaction. Use ``--headless`` only
after the profile has already cleared the challenge.

Catalog portability
-------------------
* ``Product.image`` stores ``products/<id>.<ext>`` in the DB.
* Image files live in ``MEDIA_ROOT/products/``.
* Dump the DB (``pg_dump`` or ``loyalty.dump``) AND copy ``backend/media/``
  to the new host. After ``pg_restore``, the new host serves the same URLs.
"""
from __future__ import annotations

import os
import re
import time
import urllib.parse
from contextlib import suppress
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Playwright's sync API spins up its own asyncio loop under the hood; without
# this flag Django refuses to run sync ORM calls from inside that thread.
# We're using sync_playwright explicitly so it's safe.
os.environ.setdefault("DJANGO_ALLOW_ASYNC_UNSAFE", "true")

import requests
from django.conf import settings
from django.core.management.base import BaseCommand, CommandError

from catalog.models import Product


# ---------------------------------------------------------------------------
# Constants

GA_PROFILE_DIR_NAME = ".ga_browser_profile"
GA_HOMEPAGE = "https://goldapple.kz/"
GA_PRODUCT_URL = "https://goldapple.kz/{sku}"
GA_HEX_IMG_MAIN = "696d674d61696e"  # "imgMain"
GA_HEX_TONE = "746f6e65"  # "tone"

OSIMA_HOMEPAGE = "https://osima.kz/"
OSIMA_SEARCH_URL = "https://osima.kz/catalog?q={query}"
# Product pages live at /catalog/<category>/<subcategory>/<slug> (4+ segments).
# Category pages have one segment after /catalog/ — we explicitly exclude them.
OSIMA_PRODUCT_HREF_RE = re.compile(r"^/catalog/[^/?#]+/[^/?#]+/[^/?#]+/?(?:[?#]|$)")

WB_SEARCH_URL = "https://search.wb.ru/exactmatch/ru/common/v14/search"
WB_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "ru-RU,ru;q=0.9",
    "Origin": "https://www.wildberries.ru",
    "Referer": "https://www.wildberries.ru/",
    "Connection": "keep-alive",
}
MAX_BASKET = 40
# Fail-fast on WB rate limits: at most 1 retry with a short pause, then give
# up on this product. Without this, a single 429 spiral can park the whole
# pipeline for ~6 minutes (12+24+48+96+192s).
SEARCH_RETRIES = 1
SEARCH_BACKOFF_BASE = 6.0


# ---------------------------------------------------------------------------
# Small helpers

def _ext_from_url(url: str) -> str:
    m = re.search(r"\.(webp|jpg|jpeg|png)(?:\?|$)", url, flags=re.I)
    return ("." + m.group(1).lower()) if m else ".jpg"


def _extract_filename(url: str) -> str:
    return url.rsplit("/", 1)[-1].split("?")[0]


def _extract_ga_sku(product: Product) -> str:
    """The new catalog stores bare numeric source_product_ids (e.g. ``19000085505``).
    The old catalog used the ``ga:`` prefix. Accept both."""
    spid = (product.source_product_id or "").strip()
    if not spid:
        return ""
    if spid.startswith("ga:"):
        spid = spid.split(":", 1)[1]
    if spid.isdigit() and len(spid) >= 6:
        return spid
    return ""


# ---------------------------------------------------------------------------
# Stats container

@dataclass
class Stats:
    ga_ok: int = 0
    osima_ok: int = 0
    wb_ok: int = 0
    no_match: int = 0
    skipped: int = 0
    errors: int = 0
    by_failure: dict[str, int] = field(default_factory=dict)

    def summary(self) -> str:
        return (
            f"ga={self.ga_ok}  osima={self.osima_ok}  wb={self.wb_ok}  "
            f"no_match={self.no_match}  skipped={self.skipped}  errors={self.errors}"
        )


# ---------------------------------------------------------------------------
# Goldapple scrape

def _ga_scrape_product(page, sku: str, log) -> list[str]:
    """Return list of pcdn.goldapple.ru CDN image URLs for ``sku``. [] if dead."""
    captured: list[str] = []

    def on_request(request):
        u = request.url
        if (
            "pcdn.goldapple.ru" in u
            and f"/p/p/{sku}/" in u
            and re.search(r"\.(webp|jpg|jpeg|png)(?:$|\?)", u, re.I)
        ):
            captured.append(u)

    page.on("request", on_request)
    try:
        page.goto(GA_PRODUCT_URL.format(sku=sku), wait_until="domcontentloaded", timeout=60000)

        # Wait for any anti-bot to settle.
        for _ in range(20):
            page.wait_for_timeout(1000)
            t = page.title()
            if "checking" not in t.lower():
                break

        # Wait for at least one CDN image to mount.
        with suppress(Exception):
            page.wait_for_selector(f'img[src*="/p/p/{sku}/"]', timeout=10000)
        page.wait_for_timeout(2000)

        for y in (400, 1200, 2200):
            page.evaluate(f"window.scrollTo(0, {y})")
            page.wait_for_timeout(600)
        page.evaluate("window.scrollTo(0, 0)")
        page.wait_for_timeout(500)

        dom_imgs = page.evaluate(
            """(sku) => Array.from(document.querySelectorAll('img'))
                .map(i => i.src)
                .filter(s => s && s.includes('/p/p/' + sku + '/'))""",
            sku,
        ) or []
        captured.extend(dom_imgs)
    finally:
        page.remove_listener("request", on_request)

    # Dedup, prefer full-HD .webp.
    by_key: dict[str, str] = {}
    for u in captured:
        fname = _extract_filename(u)
        key = re.sub(r"fullhd(?=\.[a-z]+$)", "", fname, flags=re.I)
        key = re.sub(r"\.[a-z]+$", "", key, flags=re.I)
        prev = by_key.get(key)
        if prev is None:
            by_key[key] = u
            continue
        prev_full = "fullhd" in prev.lower()
        cur_full = "fullhd" in u.lower()
        if cur_full and not prev_full:
            by_key[key] = u
        elif cur_full == prev_full and u.lower().endswith(".webp") and not prev.lower().endswith(".webp"):
            by_key[key] = u

    urls = list(by_key.values())
    urls.sort(
        key=lambda u: (
            0 if GA_HEX_IMG_MAIN in u else (2 if GA_HEX_TONE in u else 1),
        )
    )
    return urls


def _ga_download(ctx, url: str, dest: Path, log) -> bool:
    try:
        resp = ctx.request.get(
            url,
            headers={"Referer": "https://goldapple.kz/", "Accept": "image/avif,image/webp,image/*,*/*"},
            timeout=20000,
        )
    except Exception as e:
        log(f"      GA DL err: {e!r}")
        return False
    if resp.status != 200:
        log(f"      GA DL status {resp.status}")
        return False
    body = resp.body()
    if not body or len(body) < 1024:
        log(f"      GA DL too small ({len(body) if body else 0}b)")
        return False
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    tmp.write_bytes(body)
    tmp.replace(dest)
    return True


# ---------------------------------------------------------------------------
# Osima scrape

def _osima_normalize(s: str) -> set[str]:
    return {
        t
        for t in "".join(c.lower() if c.isalnum() or c.isspace() else " " for c in (s or "")).split()
        if len(t) >= 3
    }


def _osima_plausible(brand: str, name: str, page_text: str) -> bool:
    text = (page_text or "").lower()
    if brand and brand.lower() in text:
        return True
    overlap = _osima_normalize(name) & _osima_normalize(page_text)
    return len(overlap) >= 2


def _osima_search(page, brand: str, name: str, log) -> list[str]:
    """Find a matching product on osima.kz and return its image URLs."""
    # Single query only — the name-only fallback rarely worked but always
    # cost +15-30s per product. If brand+name doesn't match, the brand isn't
    # carried by osima and we move on to WB.
    queries = []
    if brand and name:
        queries.append(f"{brand} {name}")
    elif brand:
        queries.append(brand)
    elif name:
        queries.append(name)

    for q in queries:
        url = OSIMA_SEARCH_URL.format(query=urllib.parse.quote(q))
        log(f"      Osima search: {q!r}")
        try:
            # Osima is an SPA — wait for network to settle, then a short
            # extra wait for the final client render.
            page.goto(url, wait_until="networkidle", timeout=45000)
        except Exception as e:
            log(f"      Osima nav err: {e!r}")
            continue
        page.wait_for_timeout(2500)

        # Check the count text — if "Найдено: 0", skip immediately.
        found_count = page.evaluate(
            """() => {
                const m = (document.body.innerText || '').match(/Найдено[:\\s]*(\\d+)/i);
                return m ? parseInt(m[1], 10) : null;
            }"""
        )
        if found_count == 0:
            log("      Osima: no results")
            continue

        # Collect candidate product hrefs (4+ segment /catalog/.../.../...) and
        # rank them by brand/name overlap so we don't grab the first random card.
        candidates: list[dict] = page.evaluate(
            """() => {
                const out = [];
                const seen = new Set();
                document.querySelectorAll('a[href*="/catalog/"]').forEach(a => {
                    let href = a.getAttribute('href') || '';
                    if (!href) return;
                    // Normalize: strip query / hash for the path regex test below.
                    const path = href.split('?')[0].split('#')[0];
                    const segs = path.split('/').filter(Boolean);
                    // Expect: ['catalog', cat, subcat, slug] → 4 segments minimum.
                    if (segs.length < 4 || segs[0] !== 'catalog') return;
                    if (seen.has(href)) return;
                    seen.add(href);
                    // Pull surrounding text for matching.
                    let text = (a.innerText || a.textContent || '').trim();
                    if (!text) {
                        const card = a.closest('article, li, div');
                        if (card) text = (card.innerText || '').trim();
                    }
                    out.push({ href, text });
                });
                return out;
            }"""
        ) or []
        if not candidates:
            log("      Osima: no product cards on results page")
            continue

        brand_lower = (brand or "").lower()
        name_tokens = _osima_normalize(name)
        # The cleanest match is the URL slug `brand-name`, e.g. for
        # brand="Heimish" name="all clean balm" → slug "heimish-all-clean-balm".
        expected_slug = "-".join(
            t for t in _osima_normalize(f"{brand} {name}") if t
        )

        def score(c: dict) -> tuple[int, int]:
            txt = (c.get("text") or "").lower()
            href = c.get("href") or ""
            last = href.rsplit("/", 1)[-1].split("?")[0]
            s = 0
            if brand_lower and brand_lower in txt:
                s += 5
            s += len(name_tokens & _osima_normalize(txt))
            if expected_slug and last == expected_slug:
                s += 100  # exact slug wins
            elif expected_slug and last.startswith(expected_slug + "-"):
                s += 50   # variant (mandarin, 23-ton, …) — close but not exact
            # Penalty for longer (probably variant) slugs.
            penalty = -len(last)
            return (s, penalty)

        candidates.sort(key=score, reverse=True)
        best = candidates[0]
        top_score = score(best)[0]
        if top_score < 1:
            log(f"      Osima: no card matches brand/name (top: {best.get('text', '')[:60]!r})")
            continue
        href = best["href"]
        if href.startswith("/"):
            href = urllib.parse.urljoin("https://osima.kz/", href)
        log(f"      Osima match (score={top_score}): {best.get('text', '')[:60]!r} -> {href}")

        try:
            page.goto(href, wait_until="networkidle", timeout=45000)
        except Exception as e:
            log(f"      Osima product nav err: {e!r}")
            continue
        page.wait_for_timeout(1800)

        page_text = page.evaluate("() => document.body.innerText || ''")
        if not _osima_plausible(brand, name, page_text):
            log("      Osima: product page doesn't match brand/name, skipping")
            continue

        # Osima serves product images from a dedicated CDN — prefer those.
        images = page.evaluate(
            """() => {
                const out = [];
                const seen = new Set();
                const collect = (src) => {
                    if (!src || src.startsWith('data:')) return;
                    if (seen.has(src)) return;
                    if (!/\\.(jpe?g|png|webp)/i.test(src)) return;
                    seen.add(src);
                    out.push(src);
                };
                document.querySelectorAll('img').forEach(img => {
                    collect(img.currentSrc || img.src || img.dataset.src || '');
                });
                document.querySelectorAll('source[srcset]').forEach(s => {
                    const first = (s.srcset || '').split(',')[0].trim().split(' ')[0];
                    collect(first);
                });
                return out;
            }"""
        ) or []

        good: list[str] = []
        for src in images:
            if src.startswith("//"):
                src = "https:" + src
            elif src.startswith("/"):
                src = urllib.parse.urljoin("https://osima.kz/", src)
            sl = src.lower()
            # Reject obvious chrome.
            if any(bad in sl for bad in ("logo", "favicon", "sprite", "placeholder")):
                continue
            # Reject post / blog images.
            if "/posts/" in sl:
                continue
            good.append(src)

        # Prefer the product CDN; fall back to any survivors.
        cdn = [s for s in good if "osima-images.object.pscloud.io/products/" in s]
        chosen = cdn if cdn else good
        # Dedup preserving order.
        seen: set[str] = set()
        deduped: list[str] = []
        for s in chosen:
            if s in seen:
                continue
            seen.add(s)
            deduped.append(s)
        if not deduped:
            log("      Osima: no usable images on product page")
            continue
        return deduped[:8]

    return []


def _osima_download(ctx, url: str, dest: Path, log) -> bool:
    try:
        resp = ctx.request.get(
            url,
            headers={"Referer": "https://osima.kz/", "Accept": "image/avif,image/webp,image/*,*/*"},
            timeout=20000,
        )
    except Exception as e:
        log(f"      Osima DL err: {e!r}")
        return False
    if resp.status != 200:
        log(f"      Osima DL status {resp.status}")
        return False
    body = resp.body()
    if not body or len(body) < 1024:
        log(f"      Osima DL too small ({len(body) if body else 0}b)")
        return False
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    tmp.write_bytes(body)
    tmp.replace(dest)
    return True


# ---------------------------------------------------------------------------
# Wildberries fallback (HTTP API)

def _wb_tokens(s: str) -> set[str]:
    return {
        t
        for t in "".join(c.lower() if c.isalnum() or c.isspace() else " " for c in (s or "")).split()
        if len(t) >= 3
    }


def _wb_plausible(brand: str, name: str, result_name: str) -> bool:
    rn = (result_name or "").lower()
    if brand and brand.lower() in rn:
        return True
    overlap = len(_wb_tokens(name) & _wb_tokens(result_name))
    # Loosened from 2 → 1: many K-beauty / niche brands have terse product
    # names where a single shared distinctive token (e.g. "polish", "balm")
    # is still a usable signal.
    return overlap >= 1


def _wb_search(session: requests.Session, query: str, log) -> list[dict] | None:
    url = WB_SEARCH_URL + "?" + urllib.parse.urlencode({
        "appType": "1", "curr": "rub", "dest": "-1257786", "query": query,
        "resultset": "catalog", "spp": "30", "suppressSpellcheck": "false",
    })
    last_status: int | None = None
    for attempt in range(SEARCH_RETRIES + 1):
        try:
            r = session.get(url, timeout=15)
        except requests.RequestException as e:
            log(f"      WB net err: {e!r}")
            if attempt < SEARCH_RETRIES:
                time.sleep(SEARCH_BACKOFF_BASE)
                continue
            return None
        if r.status_code == 200:
            try:
                data = r.json()
            except ValueError:
                return None
            return ((data.get("data") or {}).get("products")) or data.get("products") or []
        last_status = r.status_code
        if r.status_code == 429 and attempt < SEARCH_RETRIES:
            log(f"      WB 429, wait {SEARCH_BACKOFF_BASE:.0f}s (1 retry)")
            time.sleep(SEARCH_BACKOFF_BASE)
            continue
        break
    if last_status == 429:
        log("      WB still rate-limited, skipping product")
    else:
        log(f"      WB status {last_status}, skipping product")
    return None


def _wb_find_basket(session: requests.Session, nm: int, basket_cache: dict[int, str]) -> str | None:
    vol = nm // 100000
    if vol in basket_cache:
        return basket_cache[vol]
    part = nm // 1000
    for b in range(1, MAX_BASKET + 1):
        try:
            r = session.head(
                f"https://basket-{b:02d}.wbbasket.ru/vol{vol}/part{part}/{nm}/images/big/1.webp",
                timeout=4,
            )
        except requests.RequestException:
            continue
        if r.status_code == 200:
            basket_cache[vol] = f"{b:02d}"
            return basket_cache[vol]
    return None


def _wb_download(session: requests.Session, url: str, dest: Path) -> bool:
    try:
        r = session.get(url, timeout=15, stream=True)
    except requests.RequestException:
        return False
    if r.status_code != 200:
        return False
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with tmp.open("wb") as f:
        for chunk in r.iter_content(8192):
            if chunk:
                f.write(chunk)
    if tmp.stat().st_size < 1024:
        tmp.unlink(missing_ok=True)
        return False
    tmp.replace(dest)
    return True


# ---------------------------------------------------------------------------
# DB attach helpers

def _attach_to_product(
    product: Product,
    main_local: Path,
    extras: list[Path],
    log,
) -> None:
    """Set product.image / image_url / image_urls and save."""
    media_root = Path(settings.MEDIA_ROOT)
    rel_main = main_local.relative_to(media_root).as_posix()
    product.image.name = rel_main
    product.image_url = f"{settings.MEDIA_URL}{rel_main}"
    gallery = [product.image_url]
    for ex in extras:
        rel_ex = ex.relative_to(media_root).as_posix()
        gallery.append(f"{settings.MEDIA_URL}{rel_ex}")
    product.image_urls = gallery
    product.save(update_fields=["image", "image_url", "image_urls", "updated_at"])


# ---------------------------------------------------------------------------
# Command

class Command(BaseCommand):
    help = "Fetch product images: Goldapple → Osima → Wildberries."

    def add_arguments(self, parser):
        parser.add_argument("--ids", default="", help="Comma-separated product ids to process.")
        parser.add_argument("--limit", type=int, default=0)
        parser.add_argument("--skip-existing", action="store_true", help="Skip products whose image file already exists locally.")
        parser.add_argument("--only-missing", action="store_true", help="Process only products without any image attached in DB.")
        parser.add_argument("--dry-run", action="store_true")
        parser.add_argument("--max-extra-pics", type=int, default=4)
        # Source toggles
        parser.add_argument("--skip-ga", action="store_true")
        parser.add_argument("--skip-osima", action="store_true")
        parser.add_argument("--skip-wb", action="store_true")
        # GA + browser
        parser.add_argument("--ga-delay", type=float, default=2.0)
        parser.add_argument("--osima-delay", type=float, default=2.0)
        parser.add_argument("--wb-delay", type=float, default=8.0)
        parser.add_argument("--headless", action="store_true", help="Run Chromium headless. Anti-bot will likely block GA & Osima until the profile has cleared the challenge once with a head.")
        parser.add_argument("--warmup-seconds", type=int, default=180, help="How long to wait on GA homepage for the anti-bot challenge.")
        parser.add_argument("--no-browser", action="store_true", help="Skip GA + Osima entirely. Use only the Wildberries HTTP API.")

    def handle(self, *args, **opts):
        # Resolve product set
        qs = Product.objects.all().order_by("id")
        if opts["ids"]:
            ids = [int(x) for x in opts["ids"].split(",") if x.strip()]
            qs = qs.filter(id__in=ids)
        if opts["only_missing"]:
            qs = qs.filter(image="").filter(image_url="")
        if opts["limit"] > 0:
            qs = qs[: opts["limit"]]
        products = list(qs)
        total = len(products)

        media_root = Path(settings.MEDIA_ROOT)
        out_dir = media_root / "products"
        out_dir.mkdir(parents=True, exist_ok=True)

        wb_session = requests.Session()
        wb_session.headers.update(WB_HEADERS)
        with suppress(requests.RequestException):
            wb_session.get("https://www.wildberries.ru/", timeout=10)
        basket_cache: dict[int, str] = {}

        stats = Stats()
        log = self.stdout.write

        no_browser = opts["no_browser"]
        skip_ga = opts["skip_ga"]
        skip_osima = opts["skip_osima"]
        skip_wb = opts["skip_wb"]

        log(f"=== fetch_product_images ===")
        log(f"products to process: {total}")
        log(f"output dir: {out_dir}")
        log(f"sources: ga={'on' if not (skip_ga or no_browser) else 'OFF'}, "
            f"osima={'on' if not (skip_osima or no_browser) else 'OFF'}, "
            f"wb={'on' if not skip_wb else 'OFF'}")
        if opts["dry_run"]:
            log("DRY RUN: no DB or filesystem writes")
        log("")

        if no_browser:
            self._run_browserless(products, out_dir, wb_session, basket_cache, opts, stats, log)
        else:
            try:
                from playwright.sync_api import sync_playwright  # noqa: F401
            except ImportError as e:
                raise CommandError(
                    "playwright not installed: pip install playwright && playwright install chromium"
                ) from e
            self._run_with_browser(products, out_dir, wb_session, basket_cache, opts, stats, log)

        log("")
        log(f"DONE  {stats.summary()}")
        if stats.by_failure:
            log("failures by reason:")
            for k, v in sorted(stats.by_failure.items(), key=lambda kv: -kv[1]):
                log(f"  {k}: {v}")

    # ------------------------------------------------------------------
    # browserless path (WB only)

    def _run_browserless(self, products, out_dir, wb_session, basket_cache, opts, stats: Stats, log):
        for idx, product in enumerate(products, 1):
            tag = f"[{idx}/{len(products)}] id={product.id} {product.brand!r} / {product.name[:40]!r}"
            if opts["skip_existing"] and self._existing_image(product, out_dir):
                stats.skipped += 1
                log(f"{tag}  SKIP (image exists)")
                continue
            log(f"{tag}")
            ok = self._try_wb(product, out_dir, wb_session, basket_cache, opts, stats, log)
            if not ok:
                stats.no_match += 1
                stats.by_failure["wb_no_match"] = stats.by_failure.get("wb_no_match", 0) + 1
                log("    NO MATCH")
            time.sleep(max(0.5, opts["wb_delay"] / 2))

    # ------------------------------------------------------------------
    # browser path (GA + Osima + WB)

    def _run_with_browser(self, products, out_dir, wb_session, basket_cache, opts, stats: Stats, log):
        from playwright.sync_api import sync_playwright

        profile_dir = Path(settings.BASE_DIR).parent / GA_PROFILE_DIR_NAME
        profile_dir.mkdir(exist_ok=True)

        with sync_playwright() as p:
            ctx = p.chromium.launch_persistent_context(
                user_data_dir=str(profile_dir),
                channel="chrome",
                headless=opts["headless"],
                viewport={"width": 1366, "height": 900},
                locale="ru-RU",
                args=["--disable-blink-features=AutomationControlled"],
            )
            page = ctx.pages[0] if ctx.pages else ctx.new_page()

            ga_ready = self._warmup_ga(page, opts["warmup_seconds"], log) if not opts["skip_ga"] else False

            for idx, product in enumerate(products, 1):
                tag = f"[{idx}/{len(products)}] id={product.id} {product.brand!r} / {product.name[:40]!r}"
                if opts["skip_existing"] and self._existing_image(product, out_dir):
                    stats.skipped += 1
                    log(f"{tag}  SKIP (image exists)")
                    continue
                log(f"{tag}")

                # 1. Goldapple
                if not opts["skip_ga"] and ga_ready:
                    sku = _extract_ga_sku(product)
                    if sku:
                        if self._try_ga(page, ctx, product, sku, out_dir, opts, stats, log):
                            time.sleep(opts["ga_delay"])
                            continue
                        time.sleep(opts["ga_delay"])
                    else:
                        log("    GA: no SKU")

                # 2. Osima
                if not opts["skip_osima"]:
                    if self._try_osima(page, ctx, product, out_dir, opts, stats, log):
                        time.sleep(opts["osima_delay"])
                        continue
                    time.sleep(opts["osima_delay"])

                # 3. Wildberries
                if not opts["skip_wb"]:
                    if self._try_wb(product, out_dir, wb_session, basket_cache, opts, stats, log):
                        time.sleep(max(0.5, opts["wb_delay"] / 2))
                        continue

                stats.no_match += 1
                stats.by_failure["all_no_match"] = stats.by_failure.get("all_no_match", 0) + 1
                log("    NO MATCH (all sources failed)")

            ctx.close()

    # ------------------------------------------------------------------
    # source attempts

    def _try_ga(self, page, ctx, product: Product, sku: str, out_dir: Path, opts, stats: Stats, log) -> bool:
        log(f"    GA sku={sku}")
        try:
            urls = _ga_scrape_product(page, sku, log)
        except Exception as e:
            log(f"      GA scrape error: {e!r}")
            stats.errors += 1
            return False
        if not urls:
            log("      GA: no images on page")
            return False
        if opts["dry_run"]:
            stats.ga_ok += 1
            log(f"      GA OK (dry) {len(urls)} imgs: {urls[0]}")
            return True
        main_local = out_dir / f"{product.id}{_ext_from_url(urls[0])}"
        if not _ga_download(ctx, urls[0], main_local, log):
            return False
        extras: list[Path] = []
        for i, u in enumerate(urls[1 : 1 + opts["max_extra_pics"]], start=2):
            ext = _ext_from_url(u)
            p_local = out_dir / f"{product.id}_{i}{ext}"
            if _ga_download(ctx, u, p_local, log):
                extras.append(p_local)
        self._clean_sibling_extensions(product.id, main_local, out_dir)
        _attach_to_product(product, main_local, extras, log)
        stats.ga_ok += 1
        log(f"      GA OK ({1 + len(extras)} imgs)")
        return True

    def _try_osima(self, page, ctx, product: Product, out_dir: Path, opts, stats: Stats, log) -> bool:
        urls = _osima_search(page, product.brand or "", product.name or "", log)
        if not urls:
            stats.by_failure["osima_no_match"] = stats.by_failure.get("osima_no_match", 0) + 1
            return False
        if opts["dry_run"]:
            stats.osima_ok += 1
            log(f"      Osima OK (dry) {len(urls)} imgs: {urls[0]}")
            return True
        main_local = out_dir / f"{product.id}{_ext_from_url(urls[0])}"
        if not _osima_download(ctx, urls[0], main_local, log):
            return False
        extras: list[Path] = []
        for i, u in enumerate(urls[1 : 1 + opts["max_extra_pics"]], start=2):
            ext = _ext_from_url(u)
            p_local = out_dir / f"{product.id}_{i}{ext}"
            if _osima_download(ctx, u, p_local, log):
                extras.append(p_local)
        self._clean_sibling_extensions(product.id, main_local, out_dir)
        _attach_to_product(product, main_local, extras, log)
        stats.osima_ok += 1
        log(f"      Osima OK ({1 + len(extras)} imgs)")
        return True

    def _try_wb(self, product: Product, out_dir: Path, session, basket_cache, opts, stats: Stats, log) -> bool:
        # Try several query variations: brand+name first (most specific), then
        # name alone (catches sellers that translate the brand to Cyrillic),
        # then brand alone (last resort — picks something from the brand line).
        brand = product.brand or ""
        name = product.name or ""
        queries: list[str] = []
        if brand and name:
            queries.append(f"{brand} {name}")
            queries.append(name)
        elif brand:
            queries.append(brand)
        elif name:
            queries.append(name)
        if not queries:
            return False

        chosen: dict[str, Any] | None = None
        chosen_query = ""
        for q in queries:
            log(f"    WB search: {q!r}")
            items = _wb_search(session, q, log)
            if not items:
                continue
            for it in items[:10]:
                if _wb_plausible(brand, name, it.get("name") or ""):
                    chosen = it
                    chosen_query = q
                    break
            if chosen is not None:
                break
            log(f"      WB: top {min(10, len(items))} results don't match")
        if chosen is None:
            return False
        nm = chosen["id"]
        pics = max(1, int(chosen.get("pics") or 1))
        basket = _wb_find_basket(session, nm, basket_cache)
        if basket is None:
            log("      WB: basket not found")
            return False
        vol = nm // 100000
        part = nm // 1000
        base = f"https://basket-{basket}.wbbasket.ru/vol{vol}/part{part}/{nm}/images/big"
        # Image priority: on Wildberries the first image is almost always a
        # marketing collage with text. Prefer 2/3/4 as the main shot and only
        # fall back to /1.webp when the seller uploaded a single image.
        if pics >= 2:
            ordered = [2, 3, 4, 1] + list(range(5, pics + 1))
        else:
            ordered = [1]
        ordered = ordered[: max(1, min(len(ordered), opts["max_extra_pics"] + 1))]
        if opts["dry_run"]:
            stats.wb_ok += 1
            log(f"      WB OK (dry) nm={nm} pics={pics} order={ordered}: {base}/{ordered[0]}.webp")
            return True
        main_local = out_dir / f"{product.id}.webp"
        main_idx = None
        for idx in ordered:
            if _wb_download(session, f"{base}/{idx}.webp", main_local):
                main_idx = idx
                break
        if main_idx is None:
            log("      WB: main download failed")
            return False
        extras: list[Path] = []
        # Pull the remaining ordered images as extras, skipping the one we
        # chose for main.
        extra_slot = 2
        for idx in ordered:
            if idx == main_idx:
                continue
            ep = out_dir / f"{product.id}_{extra_slot}.webp"
            if _wb_download(session, f"{base}/{idx}.webp", ep):
                extras.append(ep)
                extra_slot += 1
        self._clean_sibling_extensions(product.id, main_local, out_dir)
        _attach_to_product(product, main_local, extras, log)
        stats.wb_ok += 1
        log(f"      WB OK ({1 + len(extras)} imgs) nm={nm} main=#{main_idx} q={chosen_query!r}")
        return True

    # ------------------------------------------------------------------
    # utils

    def _warmup_ga(self, page, warmup_seconds: int, log) -> bool:
        log("warming up Goldapple (headed Chrome, persistent profile)...")
        try:
            page.goto(GA_HOMEPAGE, wait_until="domcontentloaded", timeout=60000)
        except Exception as e:
            log(f"  homepage nav error: {e!r}")
            return False
        cleared = False
        for sec in range(0, max(warmup_seconds, 5), 2):
            page.wait_for_timeout(2000)
            t = page.title()
            if "checking" not in t.lower():
                log(f"  GA cleared at {sec}s, title={t!r}")
                cleared = True
                break
        if not cleared:
            log(f"  WARNING: GA anti-bot still blocking after {warmup_seconds}s.")
            log("  Solve any visible captcha in the browser window manually, then re-run.")
        return cleared

    @staticmethod
    def _existing_image(product: Product, out_dir: Path) -> bool:
        for ext in (".webp", ".jpg", ".jpeg", ".png"):
            if (out_dir / f"{product.id}{ext}").exists():
                return True
        return False

    @staticmethod
    def _clean_sibling_extensions(product_id: int, main_path: Path, out_dir: Path) -> None:
        for ext in (".webp", ".jpg", ".jpeg", ".png"):
            sib = out_dir / f"{product_id}{ext}"
            if sib != main_path and sib.exists():
                sib.unlink()
