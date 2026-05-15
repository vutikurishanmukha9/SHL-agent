import json, httpx, random
with open('data/shl_catalog.json', encoding='utf-8') as f:
    data = json.load(f)
urls = [item['url'] for item in data]
sample = random.sample(urls, 5)
for u in sample:
    try:
        r = httpx.head(u, follow_redirects=True, timeout=10.0, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"})
        print(f"{u} -> {r.status_code}")
    except Exception as e:
        print(f"{u} -> ERROR: {e}")
