import json
import re
import time
from pathlib import Path
from urllib.parse import urljoin, urlencode

from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright, Page, Browser

# ── Config ────────────────────────────────────────────────────────────
BASE_URL     = "https://www.shl.com"
CATALOG_BASE = "https://www.shl.com/solutions/products/product-catalog/"
OUTPUT_FILE  = "shl_catalog.json"
PAGE_SIZE    = 12       # SHL default items per page
MAX_PAGES    = 100      # safety cap (~1200 assessments max)
CRAWL_DELAY  = 1.2      # seconds between requests (be polite)

TEST_TYPE_MAP = {
    "A": "Ability & Aptitude",
    "B": "Biodata & Situational Judgement",
    "C": "Competencies",
    "D": "Development & 360",
    "E": "Assessment Exercises",
    "K": "Knowledge & Skills",
    "P": "Personality & Behavior",
    "S": "Situational Judgement",
}

# CSS selectors tried in order to find the main content zone.
# First match wins; if none match we fall back to full soup.
CONTENT_SELECTORS = [
    "main",
    "article",
    '[class*="product-detail"]',
    '[class*="catalog-detail"]',
    '[class*="product-content"]',
    '[class*="content-area"]',
    ".content-wrapper",
    "#content",
]

# Secondary safety-net: skip paragraphs that contain these terms.
# Only used as a fallback when no content zone could be isolated.
BLACKLIST_TERMS = [
    "cookie",
    "privacy policy",
    "advertising",
    "analytics",
    "personalization",
    "opt out",
    "opt-out",
    "social media",
    "website experience",
    "we use cookies",
    "gdpr",
    "data protection",
]


# ── Browser helpers ───────────────────────────────────────────────────

def make_page(browser: Browser) -> Page:
    ctx = browser.new_context(
        user_agent=(
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        ),
        locale="en-US",
        viewport={"width": 1280, "height": 900},
    )
    page = ctx.new_page()
    # Block images/fonts — speeds things up, content unaffected
    page.route(
        "**/*.{png,jpg,jpeg,gif,webp,svg,woff,woff2,ttf}",
        lambda r: r.abort(),
    )
    return page


def fetch_html(page: Page, url: str, wait_selector: str | None = None) -> str:
    """Navigate to *url* and return rendered HTML."""
    page.goto(url, wait_until="domcontentloaded", timeout=60_000)
    if wait_selector:
        try:
            page.wait_for_selector(wait_selector, timeout=8_000)
        except Exception:
            pass
    time.sleep(0.5)
    return page.content()


# ── Content-zone isolation ────────────────────────────────────────────

def get_content_root(soup: BeautifulSoup):
    """
    Return the most specific content container we can find,
    or None if none of the known selectors match.
    Using a scoped root means cookie banners, navbars, footers,
    and overlays are excluded structurally — no blacklist needed.
    """
    for selector in CONTENT_SELECTORS:
        node = soup.select_one(selector)
        if node:
            return node
    return None


def clean_paragraphs(root, max_paras: int = 3) -> list[str]:
    """
    Extract up to *max_paras* clean paragraphs from *root*.
    If *root* is a scoped content zone the blacklist is rarely needed;
    it acts as a secondary safety net when we fell back to full soup.
    """
    parts: list[str] = []
    seen: set[str] = set()

    for p in root.find_all("p"):
        text = p.get_text(" ", strip=True)

        if len(text) < 50:
            continue

        lower = text.lower()
        if any(term in lower for term in BLACKLIST_TERMS):
            continue

        if text in seen:
            continue

        seen.add(text)
        parts.append(text)

        if len(parts) >= max_paras:
            break

    return parts


# ── Step 1: Collect all listing links ────────────────────────────────

def extract_links_from_soup(soup: BeautifulSoup) -> list[dict]:
    """Return (name, url) pairs for Individual Test Solution detail pages."""
    links: list[dict] = []
    seen: set[str] = set()

    for a in soup.find_all("a", href=True):
        href: str = a["href"]
        if "/product-catalog/view/" in href:
            full_url = urljoin(BASE_URL, href)
            name = a.get_text(strip=True)
            if full_url not in seen and len(name) > 2:
                seen.add(full_url)
                links.append({"name": name, "url": full_url})

    return links


def scrape_all_listing_links(page: Page) -> list[dict]:
    """
    Paginate through the Individual Test Solutions catalog
    and return every (name, url) found.
    """
    all_links: list[dict] = []
    seen_urls: set[str] = set()

    print("\n[Phase 1] Collecting catalog listing pages …")

    for page_num in range(MAX_PAGES):
        start  = page_num * PAGE_SIZE
        params = urlencode({"start": start, "type": 1})
        url    = f"{CATALOG_BASE}?{params}"

        print(f"  Page {page_num + 1}: {url}")
        html = fetch_html(
            page, url,
            wait_selector="table, .custom-table, [class*='catalog']",
        )
        soup = BeautifulSoup(html, "html.parser")

        page_links = extract_links_from_soup(soup)
        new_links  = [l for l in page_links if l["url"] not in seen_urls]

        if not new_links:
            print(f"  → No new links — pagination complete at page {page_num + 1}.")
            break

        for l in new_links:
            seen_urls.add(l["url"])
        all_links.extend(new_links)
        print(f"  → +{len(new_links)} new | running total: {len(all_links)}")

        time.sleep(CRAWL_DELAY)

    return all_links


# ── Step 2: Parse each detail page ───────────────────────────────────

def parse_test_types(text: str) -> list[str]:
    """
    Extract test type codes from page text.
    SHL renders them as 'Test Type: A K P' or as clickable badge letters.
    """
    # Pattern 1: explicit "Test Type:" label
    m = re.search(r"Test Type[s]?[:\s·]+([A-Z](?:\s+[A-Z])*)", text)
    if m:
        codes = m.group(1).split()
        return [c for c in codes if c in TEST_TYPE_MAP]

    # Pattern 2: legend-style "A Ability & Aptitude"
    found = re.findall(
        r"\b([ABCDEKPS])\s+(?:Ability|Biodata|Competenc|Development|Assessment|Knowledge|Personality|Situational)",
        text,
    )
    return list(dict.fromkeys(found))   # deduplicate, preserve order


def parse_measures(body: str) -> str:
    """
    Extract what the assessment measures/assesses/evaluates.
    Tries increasingly broad patterns and rejects cookie/ad noise.
    """
    MEASURES_BLACKLIST = [
        "advertising",
        "cookie",
        "analytics",
        "website",
        "marketing",
    ]

    patterns = [
        r"this assessment (?:measures?|assesses?|evaluates?)\s+(.{20,300}?)(?:\.|\Z)",
        r"the (?:test|assessment) measures?\s+(.{20,300}?)(?:\.|\Z)",
        r"(?:measures?|assesses?)\s+(?:a candidate['\u2019]s\s+)?(.{20,300}?)(?:\.|\Z)",
        r"(?:evaluates?)\s+(.{20,300}?)(?:\.|\Z)",
    ]

    for pattern in patterns:
        match = re.search(pattern, body, re.IGNORECASE)
        if match:
            extracted = match.group(1).strip()
            lower = extracted.lower()
            if not any(term in lower for term in MEASURES_BLACKLIST):
                return extracted

    return ""


def parse_detail_page(url: str, fallback_name: str, page: Page) -> dict:
    """Fetch and parse one assessment detail page. Returns a structured dict."""
    record: dict = {"url": url}

    try:
        html = fetch_html(page, url)
    except Exception as exc:
        print(f"    ⚠  Failed to fetch {url}: {exc}")
        return {"name": fallback_name, "url": url, "error": str(exc)}

    soup = BeautifulSoup(html, "html.parser")

    # ── Isolate content zone (the core fix) ───────────────────────────
    content_root = get_content_root(soup)
    search_root  = content_root or soup   # graceful fallback

    # Single clean body string — used by ALL subsequent extractions.
    # This is what eliminates cookie/nav/footer noise from every regex.
    body = search_root.get_text(" ", strip=True)

    # ── Name ──────────────────────────────────────────────────────────
    # Prefer h1 inside content zone; fall back to page-level h1.
    h1 = search_root.find("h1") or soup.find("h1")
    record["name"] = h1.get_text(strip=True) if h1 else fallback_name

    # ── Description ───────────────────────────────────────────────────
    desc_parts = clean_paragraphs(search_root, max_paras=3)
    record["description"] = " ".join(desc_parts)

    # ── What it measures ──────────────────────────────────────────────
    record["measures"] = parse_measures(body)

    # ── Test types ────────────────────────────────────────────────────
    record["test_types"]       = parse_test_types(body)
    record["test_type_labels"] = [TEST_TYPE_MAP[t] for t in record["test_types"]]

    # ── Duration ──────────────────────────────────────────────────────
    dur = re.search(
        r"(?:Approximate Completion Time|Duration|Timing)[^\d]*(\d+)\s*(?:min|minutes?)",
        body, re.IGNORECASE,
    )
    record["duration_minutes"] = int(dur.group(1)) if dur else None

    # ── Remote testing flag ───────────────────────────────────────────
    record["remote_testing"] = bool(
        re.search(r"remote\s*testing", body, re.IGNORECASE)
    )

    # ── Adaptive / IRT ────────────────────────────────────────────────
    record["adaptive"] = bool(
        re.search(r"\badaptive\b|\bIRT\b", body, re.IGNORECASE)
    )

    # ── Job levels ────────────────────────────────────────────────────
    level_keywords = [
        "Entry-Level", "Graduate", "Mid-Professional",
        "Professional Individual Contributor", "Manager",
        "Director", "Executive",
    ]
    record["job_levels"] = [kw for kw in level_keywords if kw.lower() in body.lower()]

    # ── Languages ─────────────────────────────────────────────────────

    LANGUAGE_BLACKLIST = [
        "assessment length",
        "approximate completion",
        "time in minutes",
        "speak to our team",
    ]

    lang_m = re.search(
        r"(?:Languages?|Available in)[:\s]+((?:[A-Za-z ()\-]+,?\s*){2,})",
        body,
        re.IGNORECASE,
    )

    if lang_m:
        raw_langs = [
            l.strip()
            for l in re.split(r",|\n", lang_m.group(1))
            if l.strip()
        ]

        clean_langs = []

        for lang in raw_langs:
            lower = lang.lower()

            if any(bad in lower for bad in LANGUAGE_BLACKLIST):
                continue

            if len(lang) < 2:
                continue

            clean_langs.append(lang)

        record["languages"] = clean_langs[:20]

    else:
        record["languages"] = []

    # ── Debug flag: note if we fell back to full soup ─────────────────
    if content_root is None:
        record["_fallback_root"] = True

    return record

# ── Step 3: Full pipeline ─────────────────────────────────────────────

def scrape() -> list[dict]:
    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True)
        page    = make_page(browser)

        # Phase 1: collect all listing links
        links = scrape_all_listing_links(page)
        print(f"\nTotal individual test solution links found: {len(links)}")

        # Phase 2: scrape detail pages
        catalog: list[dict] = []
        print("\n[Phase 2] Scraping detail pages …\n")

        fallback_count = 0

        for i, link in enumerate(links, 1):
            print(f"  [{i:>4}/{len(links)}] {link['name'][:60]}")
            record = parse_detail_page(link["url"], link["name"], page)
            catalog.append(record)

            if record.get("_fallback_root"):
                fallback_count += 1
                print(f"    ⚠  No content zone found — used full-page fallback")

            time.sleep(CRAWL_DELAY)

            # Checkpoint every 50 items so progress isn't lost on crashes
            if i % 50 == 0:
                _checkpoint(catalog, i)

        browser.close()

        if fallback_count:
            print(
                f"\n⚠  {fallback_count}/{len(links)} pages used full-page fallback "
                f"(no content zone matched). Consider adding their container selectors "
                f"to CONTENT_SELECTORS."
            )

    return catalog


def _checkpoint(catalog: list[dict], n: int) -> None:
    path = f"shl_catalog_checkpoint_{n}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(catalog, f, indent=2, ensure_ascii=False)
    print(f"    [checkpoint] saved {n} items → {path}")


# ── Step 4: Save & summarise ──────────────────────────────────────────

def save(catalog: list[dict]) -> None:
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(catalog, f, indent=2, ensure_ascii=False)
    print(f"\nSaved {len(catalog)} assessments → {OUTPUT_FILE}")


def summarise(catalog: list[dict]) -> None:
    print("\n" + "=" * 60)
    print("CATALOG SUMMARY")
    print("=" * 60)
    print(f"Total : {len(catalog)}")

    type_counts: dict[str, int] = {}
    for item in catalog:
        for t in item.get("test_types", []):
            type_counts[t] = type_counts.get(t, 0) + 1

    print("\nBy test type:")
    for code, label in TEST_TYPE_MAP.items():
        n = type_counts.get(code, 0)
        if n:
            print(f"  {code}  {label:<35} {n}")

    durations = [d["duration_minutes"] for d in catalog if d.get("duration_minutes")]
    if durations:
        avg = sum(durations) / len(durations)
        print(
            f"\nDuration — min:{min(durations)}m  max:{max(durations)}m  "
            f"avg:{avg:.0f}m  ({len(durations)}/{len(catalog)} have duration)"
        )

    remote_n    = sum(1 for d in catalog if d.get("remote_testing"))
    errors      = [d for d in catalog if "error" in d]
    fallbacks   = [d for d in catalog if d.get("_fallback_root")]

    print(f"Remote testing available : {remote_n}")
    print(f"Failed / errored pages   : {len(errors)}")
    print(f"Full-page fallback used  : {len(fallbacks)}")

    print("\nSample (first 5):")
    for item in catalog[:5]:
        print(
            f"  • {item['name'][:50]:<50} "
            f"types={item.get('test_types')}  "
            f"desc_len={len(item.get('description', ''))}"
        )


# ── Quick smoke-test (single URL) ────────────────────────────────────

def smoke_test(url: str) -> None:
    """
    Run against a single known URL and pretty-print the result.
    Use this BEFORE launching the full crawl to verify extraction quality.

    Usage:
        python shl_scraper.py smoke https://www.shl.com/solutions/products/product-catalog/view/...
    """
    print(f"\n[Smoke test] {url}\n")
    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True)
        page    = make_page(browser)
        record  = parse_detail_page(url, "SMOKE_TEST", page)
        browser.close()

    print(json.dumps(record, indent=2, ensure_ascii=False))

    # Sanity checks
    print("\n── Sanity checks ──")
    desc = record.get("description", "")
    if any(t in desc.lower() for t in ["cookie", "privacy", "advertising"]):
        print("❌  FAIL: description still contains cookie/privacy garbage")
    elif len(desc) < 50:
        print("⚠   WARN: description is very short — content zone may not have matched")
    else:
        print("✓  description looks clean")

    if record.get("_fallback_root"):
        print("⚠   WARN: fell back to full-page soup — add a selector for this page")
    else:
        print("✓  content zone was isolated successfully")


# ── Entry point ───────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    if len(sys.argv) == 3 and sys.argv[1] == "smoke":
        smoke_test(sys.argv[2])
    else:
        catalog = scrape()
        save(catalog)
        summarise(catalog)
        print("\n✓ Scraping complete.")