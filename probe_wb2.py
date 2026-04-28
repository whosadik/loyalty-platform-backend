"""Inspect full WB search response for image clues + try basket variants."""
import json, time, requests, urllib.parse

s = requests.Session()
s.headers.update({
    "User-Agent": "Mozilla/5.0 Chrome/120.0",
    "Accept": "*/*",
    "Referer": "https://www.wildberries.ru/",
})

q = "YERMA ampoule serum hyaluronic acid"
url = f"https://search.wb.ru/exactmatch/ru/common/v14/search?{urllib.parse.urlencode({'appType':'1','curr':'rub','dest':'-1257786','query':q,'resultset':'catalog','spp':'30'})}"
r = s.get(url, timeout=15)
print("status:", r.status_code)
data = r.json()
products = ((data.get("data") or {}).get("products")) or data.get("products") or []
print("found:", len(products))
if products:
    p0 = products[0]
    print("\nfull first product:")
    print(json.dumps(p0, indent=2, ensure_ascii=False)[:2000])
    nm = p0["id"]
    vol = nm // 100000
    part = nm // 1000
    # try different size paths and extensions
    paths = [
        f"vol{vol}/part{part}/{nm}/images/big/1.webp",
        f"vol{vol}/part{part}/{nm}/images/big/1.jpg",
        f"vol{vol}/part{part}/{nm}/images/c246x328/1.webp",
        f"vol{vol}/part{part}/{nm}/images/c516x688/1.webp",
        f"vol{vol}/part{part}/{nm}/info/ru/card.json",
    ]
    print("\nprobing baskets 1..40 for each path scheme...")
    for path in paths:
        for b in range(1, 41):
            u = f"https://basket-{b:02d}.wbbasket.ru/{path}"
            try:
                rr = s.head(u, timeout=3)
                if rr.status_code == 200:
                    print(f"  HIT  basket-{b:02d}  -> {u}")
                    break
            except Exception:
                pass
        else:
            print(f"  miss {path}")
        time.sleep(0.2)
