"""Re-fetch product images from Goldapple via headed Playwright with a persistent
Chrome profile. Falls back to Wildberries search for SKUs that no longer exist
on Goldapple.

The Goldapple website is protected by a Group-IB anti-bot challenge. A fresh
headless browser cannot pass it, but a persistent Chrome profile keeps the
fingerprint cookies between runs, so the challenge only has to clear on the
first run (during the homepage warmup) and the same context can then iterate
over every product page.

Images are saved to MEDIA_ROOT/products/<product_id>.<ext> and the product's
image_url / image_urls are rewritten to /media/products/... paths.
"""
from __future__ import annotations

import re
import time
import urllib.parse
from pathlib import Path

import requests
from django.conf import settings
from django.core.management.base import BaseCommand, CommandError

from catalog.models import Product


GA_PROFILE_DIR_NAME = ".ga_browser_profile"
GA_PRODUCT_URL = "https://goldapple.ru/{sku}"
GA_HOMEPAGE = "https://goldapple.ru/"

# Hex-encoded filename prefixes used by GA's CDN naming scheme.
HEX_IMG_MAIN = "696d674d61696e"  # "imgMain"
HEX_TONE = "746f6e65"  # "tone"

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
SEARCH_RETRIES = 5
SEARCH_BACKOFF_BASE = 12.0


def _ext_from_url(url: str) -> str:
    m = re.search(r"\.(webp|jpg|jpeg|png)(?:\?|$)", url, flags=re.I)
    return ("." + m.group(1).lower()) if m else ".jpg"


def _is_main_image(url: str) -> bool:
    return HEX_IMG_MAIN in url


def _is_tone_image(url: str) -> bool:
    return HEX_TONE in url


def _extract_filename(url: str) -> str:
    return url.rsplit("/", 1)[-1].split("?")[0]


# Note: GA's SPA always contains a hidden "Опс... Товар не найден" element in
# the DOM, so we cannot detect a dead product page by looking for that string.
# Instead we treat a product as dead if no /p/p/<sku>/ image URL ever appears
# in the network or DOM after waiting.


# ----- Goldapple scrape -----

def _ga_scrape_product(page, sku: str, scroll: bool = True, log=print) -> list[str]:
    """Open product page; return list of pcdn.goldapple.ru CDN image URLs for
    THIS sku (deduped, full-HD preferred, main first, tones last). [] if dead."""
    captured: list[str] = []

    def on_request(request):
        u = request.url
        if "pcdn.goldapple.ru" in u and f"/p/p/{sku}/" in u and re.search(r"\.(webp|jpg|jpeg|png)(?:$|\?)", u, re.I):
            captured.append(u)

    page.on("request", on_request)
    try:
        page.goto(GA_PRODUCT_URL.format(sku=sku), wait_until="domcontentloaded", timeout=60000)

        # Wait for challenge to clear (shouldn't trigger after warmup, but guard anyway)
        for _ in range(20):
            page.wait_for_timeout(1000)
            t = page.title()
            if "checking" not in t.lower():
                break

        # Wait for product images to start loading
        try:
            page.wait_for_selector(f'img[src*="/p/p/{sku}/"]', timeout=10000)
        except Exception:
            pass
        page.wait_for_timeout(2500)

        if scroll:
            for y in (400, 1000, 1800, 2600):
                page.evaluate(f"window.scrollTo(0, {y})")
                page.wait_for_timeout(700)
            page.evaluate("window.scrollTo(0, 0)")
            page.wait_for_timeout(800)

        # Also collect from rendered <img> tags (in case some never went through network listener)
        dom_imgs = page.evaluate(
            """(sku) => Array.from(document.querySelectorAll('img'))
                .map(i => i.src)
                .filter(s => s && s.includes('/p/p/' + sku + '/'))""",
            sku,
        ) or []
        captured.extend(dom_imgs)
    finally:
        page.remove_listener("request", on_request)

    # Dedup, prefer full-HD; we want one URL per "logical" image. We treat URLs
    # whose filenames differ only by the trailing 'fullhd' suffix as the same.
    by_key: dict[str, str] = {}
    for u in captured:
        fname = _extract_filename(u)
        # Strip 'fullhd' marker for grouping
        key = re.sub(r"fullhd(?=\.[a-z]+$)", "", fname, flags=re.I)
        # Group by directory-less stem so .jpg and .webp variants aren't both kept
        key = re.sub(r"\.[a-z]+$", "", key, flags=re.I)
        # Prefer the URL that has 'fullhd' (higher resolution); among equals keep webp
        prev = by_key.get(key)
        if prev is None:
            by_key[key] = u
            continue
        prev_full = "fullhd" in prev.lower()
        cur_full = "fullhd" in u.lower()
        if cur_full and not prev_full:
            by_key[key] = u
        elif cur_full == prev_full:
            # same-tier; prefer .webp
            if u.lower().endswith(".webp") and not prev.lower().endswith(".webp"):
                by_key[key] = u

    urls = list(by_key.values())
    # Order: main first, then non-tone gallery, tones last
    urls.sort(key=lambda u: (0 if _is_main_image(u) else (2 if _is_tone_image(u) else 1)))
    return urls


def _ga_download(ctx, url: str, dest: Path, log=print) -> bool:
    """Download an image using the browser's request context (carries cookies + referer)."""
    try:
        resp = ctx.request.get(
            url,
            headers={"Referer": "https://goldapple.ru/", "Accept": "image/avif,image/webp,image/*,*/*"},
            timeout=20000,
        )
    except Exception as e:
        log(f"      DL err: {e!r}")
        return False
    if resp.status != 200:
        log(f"      DL status {resp.status}")
        return False
    body = resp.body()
    if not body:
        return False
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    tmp.write_bytes(body)
    tmp.replace(dest)
    return True


# ----- Wildberries fallback -----

def _wb_tokens(s: str) -> set[str]:
    return {t for t in "".join(c.lower() if c.isalnum() or c.isspace() else " " for c in (s or "")).split() if len(t) >= 3}


def _wb_plausible(brand: str, name: str, result_name: str) -> bool:
    rn = (result_name or "").lower()
    if brand and brand.lower() in rn:
        return True
    return len(_wb_tokens(name) & _wb_tokens(result_name)) >= 2


def _wb_search(session: requests.Session, query: str, log=print) -> list[dict] | None:
    url = WB_SEARCH_URL + "?" + urllib.parse.urlencode({
        "appType": "1", "curr": "rub", "dest": "-1257786", "query": query,
        "resultset": "catalog", "spp": "30", "suppressSpellcheck": "false",
    })
    for attempt in range(SEARCH_RETRIES):
        try:
            r = session.get(url, timeout=15)
        except requests.RequestException as e:
            log(f"      WB net err: {e!r}, retry {SEARCH_BACKOFF_BASE}s")
            time.sleep(SEARCH_BACKOFF_BASE)
            continue
        if r.status_code == 200:
            try:
                data = r.json()
            except ValueError:
                return None
            return ((data.get("data") or {}).get("products")) or data.get("products") or []
        if r.status_code == 429:
            wait = SEARCH_BACKOFF_BASE * (2**attempt)
            log(f"      WB 429, wait {wait:.0f}s ({attempt + 1}/{SEARCH_RETRIES})")
            time.sleep(wait)
            continue
        log(f"      WB unexpected {r.status_code}")
        return None
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
    tmp.replace(dest)
    return True


def _wb_fallback(
    session: requests.Session,
    product: Product,
    out_dir: Path,
    basket_cache: dict[int, str],
    max_extra: int,
    delay: float,
    log,
) -> tuple[str, list[str]] | None:
    query = f"{product.brand} {product.name}".strip()
    if not query:
        return None
    items = _wb_search(session, query, log)
    if not items:
        return None
    chosen = next((p for p in items[:5] if _wb_plausible(product.brand, product.name, p.get("name") or "")), None)
    if chosen is None:
        return None
    nm = chosen["id"]
    pics = max(1, int(chosen.get("pics") or 1))
    basket = _wb_find_basket(session, nm, basket_cache)
    if basket is None:
        return None
    vol = nm // 100000
    part = nm // 1000
    base = f"https://basket-{basket}.wbbasket.ru/vol{vol}/part{part}/{nm}/images/big"
    main_path = out_dir / f"{product.id}.webp"
    if not _wb_download(session, f"{base}/1.webp", main_path):
        return None
    main_url = f"{settings.MEDIA_URL}products/{product.id}.webp"
    extra: list[str] = []
    for i in range(2, 2 + min(max_extra, max(0, pics - 1))):
        ep = out_dir / f"{product.id}_{i}.webp"
        if _wb_download(session, f"{base}/{i}.webp", ep):
            extra.append(f"{settings.MEDIA_URL}products/{product.id}_{i}.webp")
    time.sleep(max(0.5, delay / 4))
    return main_url, extra


# ----- Command -----

class Command(BaseCommand):
    help = "Refetch product images from Goldapple (with Wildberries fallback)."

    def add_arguments(self, parser):
        parser.add_argument("--limit", type=int, default=0)
        parser.add_argument("--ids", type=str, default="")
        parser.add_argument("--delay", type=float, default=2.0, help="Delay between GA product pages")
        parser.add_argument("--wb-delay", type=float, default=8.0, help="Delay between WB search calls")
        parser.add_argument("--skip-existing", action="store_true",
                            help="Skip products whose local main image file already exists")
        parser.add_argument("--no-wb-fallback", action="store_true")
        parser.add_argument("--wb-only", action="store_true",
                            help="Skip Goldapple entirely; use Wildberries for every product")
        parser.add_argument("--dry-run", action="store_true")
        parser.add_argument("--max-extra-pics", type=int, default=4)
        parser.add_argument("--headless", action="store_true",
                            help="Try headless mode (Group-IB will likely block)")
        parser.add_argument("--warmup-seconds", type=int, default=180,
                            help="How long to wait on the GA homepage for the anti-bot challenge")

    def handle(self, *args, **opts):
        wb_only = opts["wb_only"]
        if not wb_only:
            try:
                from playwright.sync_api import sync_playwright  # noqa: F401
            except ImportError as e:
                raise CommandError("playwright not installed: pip install playwright && playwright install chromium") from e

        delay = opts["delay"]
        wb_delay = opts["wb_delay"]
        limit = opts["limit"]
        ids = [int(x) for x in opts["ids"].split(",") if x.strip()] if opts["ids"] else []
        dry_run = opts["dry_run"]
        skip_existing = opts["skip_existing"]
        max_extra = opts["max_extra_pics"]
        no_wb = opts["no_wb_fallback"]
        headless = opts["headless"]
        warmup_seconds = opts["warmup_seconds"]

        qs = Product.objects.all().order_by("id")
        if ids:
            qs = qs.filter(id__in=ids)
        if limit > 0:
            qs = qs[:limit]

        # GA processes only ga: prefixed source_product_ids.
        products = list(qs)
        total = len(products)

        media_root = Path(settings.MEDIA_ROOT)
        out_dir = media_root / "products"
        out_dir.mkdir(parents=True, exist_ok=True)

        profile_dir = Path(settings.BASE_DIR).parent / GA_PROFILE_DIR_NAME
        profile_dir.mkdir(exist_ok=True)

        # WB session for fallback
        wb_session = requests.Session()
        wb_session.headers.update(WB_HEADERS)
        try:
            wb_session.get("https://www.wildberries.ru/", timeout=10)
        except requests.RequestException:
            pass
        basket_cache: dict[int, str] = {}

        stats = {"ga_ok": 0, "wb_ok": 0, "no_match": 0, "fail": 0, "skipped": 0}

        if wb_only:
            self.stdout.write(f"WB-ONLY mode. Processing {total} products, output: {out_dir}")
            if dry_run:
                self.stdout.write("DRY RUN: no DB or filesystem writes")
            for idx, product in enumerate(products, 1):
                tag = f"[{idx}/{total}] id={product.id} {product.brand!r} / {product.name[:40]!r}"
                local_main = out_dir / f"{product.id}.webp"
                if skip_existing and local_main.exists():
                    stats["skipped"] += 1
                    self.stdout.write(f"{tag}  SKIP (file exists)")
                    continue
                self.stdout.write(f"{tag}  WB query={product.brand} {product.name[:40]!r}")
                self._maybe_wb(product, out_dir, wb_session, basket_cache, no_wb, dry_run, max_extra, wb_delay, stats)
                time.sleep(max(0.5, wb_delay - 1))
            self.stdout.write("")
            self.stdout.write(
                f"DONE  ga_ok={stats['ga_ok']}  wb_ok={stats['wb_ok']}  no_match={stats['no_match']}  fail={stats['fail']}  skipped={stats['skipped']}"
            )
            return

        from playwright.sync_api import sync_playwright
        with sync_playwright() as p:
            ctx = p.chromium.launch_persistent_context(
                user_data_dir=str(profile_dir),
                channel="chrome",
                headless=headless,
                viewport={"width": 1366, "height": 900},
                locale="ru-RU",
                args=["--disable-blink-features=AutomationControlled"],
            )
            page = ctx.pages[0] if ctx.pages else ctx.new_page()

            # Warm up: go to homepage and wait for anti-bot to clear
            self.stdout.write("warming up Goldapple (headed Chrome, persistent profile)...")
            try:
                page.goto(GA_HOMEPAGE, wait_until="domcontentloaded", timeout=60000)
            except Exception as e:
                self.stdout.write(f"  homepage navigation error: {e!r}")

            cleared = False
            for sec in range(0, max(warmup_seconds, 5), 2):
                page.wait_for_timeout(2000)
                t = page.title()
                if "checking" not in t.lower():
                    self.stdout.write(f"  cleared at {sec}s, title={t!r}")
                    cleared = True
                    break
            if not cleared:
                self.stdout.write(f"  WARNING: anti-bot still blocking after {warmup_seconds}s")
                self.stdout.write("  HINT: solve any visible captcha in the browser window, then re-run.")
                self.stdout.write("  WB fallback will be used for everything.")

            self.stdout.write(f"\nprocessing {total} products, output: {out_dir}")
            if dry_run:
                self.stdout.write("DRY RUN: no DB or filesystem writes")

            for idx, product in enumerate(products, 1):
                tag = f"[{idx}/{total}] id={product.id} {product.brand!r} / {product.name[:40]!r}"
                local_main = out_dir / f"{product.id}.webp"
                local_main_jpg = out_dir / f"{product.id}.jpg"

                if skip_existing and (local_main.exists() or local_main_jpg.exists()):
                    stats["skipped"] += 1
                    self.stdout.write(f"{tag}  SKIP (file exists)")
                    continue

                spid = product.source_product_id or ""
                sku = spid.split(":", 1)[1] if spid.startswith("ga:") else ""
                if not sku:
                    self.stdout.write(f"{tag}  no GA SKU, trying WB...")
                    self._maybe_wb(product, out_dir, wb_session, basket_cache, no_wb, dry_run, max_extra, wb_delay, stats)
                    continue

                self.stdout.write(f"{tag}  GA sku={sku}")
                if not cleared:
                    # Skip GA, go straight to WB.
                    self._maybe_wb(product, out_dir, wb_session, basket_cache, no_wb, dry_run, max_extra, wb_delay, stats)
                    continue

                try:
                    urls = _ga_scrape_product(page, sku, log=self.stdout.write)
                except Exception as e:
                    self.stdout.write(f"    GA scrape error: {e!r}")
                    urls = []

                if not urls:
                    self.stdout.write("    GA: not found / no images")
                    self._maybe_wb(product, out_dir, wb_session, basket_cache, no_wb, dry_run, max_extra, wb_delay, stats)
                    continue

                if dry_run:
                    self.stdout.write(f"    GA OK ({len(urls)} imgs): {urls[0]}")
                    stats["ga_ok"] += 1
                    time.sleep(delay)
                    continue

                # Download main + up to max_extra extras
                main_url = urls[0]
                main_ext = _ext_from_url(main_url)
                main_local = out_dir / f"{product.id}{main_ext}"
                if not _ga_download(ctx, main_url, main_local, log=self.stdout.write):
                    self.stdout.write("    GA main download failed, falling back to WB")
                    self._maybe_wb(product, out_dir, wb_session, basket_cache, no_wb, dry_run, max_extra, wb_delay, stats)
                    continue

                local_main_url = f"{settings.MEDIA_URL}products/{product.id}{main_ext}"
                gallery_urls = [local_main_url]
                # Clean up sibling main file with the *other* extension if it exists from a prior run
                for other_ext in (".webp", ".jpg", ".jpeg", ".png"):
                    if other_ext != main_ext:
                        sib = out_dir / f"{product.id}{other_ext}"
                        if sib.exists():
                            sib.unlink()

                for i, u in enumerate(urls[1 : 1 + max_extra], start=2):
                    ext = _ext_from_url(u)
                    p_local = out_dir / f"{product.id}_{i}{ext}"
                    if _ga_download(ctx, u, p_local, log=self.stdout.write):
                        gallery_urls.append(f"{settings.MEDIA_URL}products/{product.id}_{i}{ext}")

                product.image_url = local_main_url
                product.image_urls = gallery_urls
                product.save(update_fields=["image_url", "image_urls", "updated_at"])
                stats["ga_ok"] += 1
                self.stdout.write(f"    GA OK ({len(gallery_urls)} imgs)")
                time.sleep(delay)

            ctx.close()

        self.stdout.write("")
        self.stdout.write(
            f"DONE  ga_ok={stats['ga_ok']}  wb_ok={stats['wb_ok']}  no_match={stats['no_match']}  fail={stats['fail']}  skipped={stats['skipped']}"
        )

    # WB fallback path; mutates stats and writes DB.
    def _maybe_wb(self, product, out_dir, session, basket_cache, no_wb, dry_run, max_extra, wb_delay, stats):
        if no_wb:
            stats["no_match"] += 1
            return
        result = _wb_fallback(session, product, out_dir, basket_cache, max_extra, wb_delay, self.stdout.write)
        if result is None:
            stats["no_match"] += 1
            self.stdout.write("    WB: no plausible match")
            return
        main_url, extras = result
        if dry_run:
            stats["wb_ok"] += 1
            self.stdout.write(f"    WB OK (dry-run): {main_url}")
            return
        product.image_url = main_url
        product.image_urls = [main_url, *extras]
        product.save(update_fields=["image_url", "image_urls", "updated_at"])
        stats["wb_ok"] += 1
        self.stdout.write(f"    WB OK ({1 + len(extras)} imgs)")
