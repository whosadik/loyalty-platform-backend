"""Probe correct WB basket for high nm_ids."""
import requests, time

# nm_ids from previous run that returned 404 with my sharding
nm_ids = [795608067, 763436712]

s = requests.Session()
s.headers.update({"User-Agent": "Mozilla/5.0", "Referer": "https://www.wildberries.ru/"})

for nm in nm_ids:
    vol = nm // 100000
    part = nm // 1000
    print(f"\nnm={nm} (short={nm//100000})")
    for b in range(1, 35):
        url = f"https://basket-{b:02d}.wbbasket.ru/vol{vol}/part{part}/{nm}/images/big/1.webp"
        try:
            r = s.head(url, timeout=4)
            if r.status_code == 200:
                print(f"  basket-{b:02d}: 200  -> {url}")
                break
        except Exception:
            continue
    else:
        print("  no basket worked!")
    time.sleep(0.5)
