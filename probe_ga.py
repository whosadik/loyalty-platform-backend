"""Probe one GA product page; capture only its gallery images."""
import sys
from pathlib import Path
from playwright.sync_api import sync_playwright

PROFILE_DIR = Path(__file__).parent / ".ga_browser_profile"
PROFILE_DIR.mkdir(exist_ok=True)

# Try several known-active SKUs from our DB to see which still exist on GA
SKUS = ["19000436878", "19000326439", "19000053850", "19000021580", "19000294997"]

with sync_playwright() as p:
    ctx = p.chromium.launch_persistent_context(
        user_data_dir=str(PROFILE_DIR),
        channel="chrome",
        headless=False,
        viewport={"width": 1366, "height": 900},
        locale="ru-RU",
        args=["--disable-blink-features=AutomationControlled"],
    )
    page = ctx.new_page()

    for sku in SKUS:
        url = f"https://goldapple.ru/{sku}"
        print(f"\n=== {url} ===")
        page.goto(url, wait_until="domcontentloaded", timeout=60000)

        for i in range(20):
            page.wait_for_timeout(1500)
            t = page.title()
            if "checking" not in t.lower():
                break
        page.wait_for_timeout(3000)

        print("title:", page.title()[:80])
        print("final url:", page.url)

        # Try to find gallery images that reference THIS sku
        own_imgs = page.evaluate(
            """(sku) => Array.from(document.querySelectorAll('img'))
                .map(i => ({src: i.src, w: i.naturalWidth, h: i.naturalHeight}))
                .filter(x => x.src && x.src.includes('/p/p/' + sku + '/'))""",
            sku,
        )
        print(f"  own gallery <img> ({len(own_imgs)}):")
        for x in own_imgs[:8]:
            print("   ", x)

        # H1 / product title from page
        h1 = page.evaluate("() => document.querySelector('h1')?.innerText || ''")
        print("  page h1:", h1[:120])

    print("\nclosing in 3s...")
    page.wait_for_timeout(3000)
    ctx.close()
