"""Probe Wildberries search API with session warmup and pacing."""
import urllib.parse
import time
import requests

samples = [
    ("YOPE", "skinimally"),
    ("YERMA", "ampoule serum hyaluronic acid"),
    ("3INA", "the tinted moisturizer"),
    ("Cosworker", "cerahome earlycare ph gel cleanser"),
]


def wb_image_url(nm_id: int) -> str:
    short = nm_id // 100000
    vol = nm_id // 100000
    part = nm_id // 1000
    if 0 <= short <= 143: basket = "01"
    elif short <= 287: basket = "02"
    elif short <= 431: basket = "03"
    elif short <= 719: basket = "04"
    elif short <= 1007: basket = "05"
    elif short <= 1061: basket = "06"
    elif short <= 1115: basket = "07"
    elif short <= 1169: basket = "08"
    elif short <= 1313: basket = "09"
    elif short <= 1601: basket = "10"
    elif short <= 1655: basket = "11"
    elif short <= 1919: basket = "12"
    elif short <= 2045: basket = "13"
    elif short <= 2189: basket = "14"
    elif short <= 2405: basket = "15"
    elif short <= 2621: basket = "16"
    elif short <= 2837: basket = "17"
    elif short <= 3053: basket = "18"
    elif short <= 3269: basket = "19"
    elif short <= 3485: basket = "20"
    elif short <= 3701: basket = "21"
    elif short <= 3917: basket = "22"
    elif short <= 4133: basket = "23"
    elif short <= 4349: basket = "24"
    elif short <= 4565: basket = "25"
    elif short <= 4781: basket = "26"
    else: basket = "27"
    return f"https://basket-{basket}.wbbasket.ru/vol{vol}/part{part}/{nm_id}/images/big/1.webp"


s = requests.Session()
s.headers.update({
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120.0 Safari/537.36",
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "ru-RU,ru;q=0.9",
    "Origin": "https://www.wildberries.ru",
    "Referer": "https://www.wildberries.ru/",
    "Connection": "keep-alive",
})

print("warming up wildberries.ru session...")
try:
    r = s.get("https://www.wildberries.ru/", timeout=15)
    print(" warmup status:", r.status_code, "cookies:", list(s.cookies.keys()))
except Exception as e:
    print(" warmup error:", e)

# Try alternative search hosts and slower pacing
print("\nwaiting 30s for rate-limit window to reset...")
time.sleep(30)

hosts = ["search.wb.ru", "u-search.wb.ru"]

for brand, name in samples:
    query = f"{brand} {name}".strip()
    print(f"\n=== {query} ===")
    success = False
    for host in hosts:
        url = (
            f"https://{host}/exactmatch/ru/common/v14/search?"
            + urllib.parse.urlencode({
                "appType": "1",
                "curr": "rub",
                "dest": "-1257786",
                "query": query,
                "resultset": "catalog",
                "spp": "30",
                "suppressSpellcheck": "false",
            })
        )
        try:
            r = s.get(url, timeout=15)
            print(f"  [{host}] status: {r.status_code}")
            if r.status_code != 200:
                continue
            data = r.json()
            products = ((data.get("data") or {}).get("products")) or data.get("products") or []
            print(f"    products: {len(products)}")
            for p in products[:2]:
                nm = p.get("id")
                img = wb_image_url(nm)
                ir = s.head(img, timeout=5)
                print(f"    nm={nm}  brand={p.get('brand')!r:<25} name={(p.get('name') or '')[:50]!r}  img={ir.status_code}")
                print(f"      {img}")
            success = True
            break
        except Exception as e:
            print(f"  [{host}] err:", repr(e))
    if not success:
        print("  FAILED on all hosts")
    time.sleep(5)
