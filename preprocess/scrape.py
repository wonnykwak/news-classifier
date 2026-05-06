"""
Collect Fox News and NBC News headlines.
Strategies:
  1. RSS feeds — headline comes directly from feed <title> tag (fast, clean)
  2. News sitemaps — NBC sitemap-news has titles; Fox fastchanging has URLs only
  3. Category pages — scrape article links from section landing pages
  4. Individual articles — scrape <h1> for URLs collected without a headline
     (opt-in via --scrape-articles or --sitemap-limit N)
Output: url,headline CSV compatible with url_with_headlines.csv
"""

from __future__ import annotations

import time
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    )
}
DELAY = 1.2  # seconds between requests

SM_NS = "http://www.sitemaps.org/schemas/sitemap/0.9"
NEWS_NS = "http://www.google.com/schemas/sitemap-news/0.9"
ATOM_NS = "http://www.w3.org/2005/Atom"

# ── RSS feeds ──────────────────────────────────────────────────────────────────

FOXNEWS_RSS = [
    "https://moxie.foxnews.com/google-publisher/latest.xml",
    "https://moxie.foxnews.com/google-publisher/politics.xml",
    "https://moxie.foxnews.com/google-publisher/us.xml",
    "https://moxie.foxnews.com/google-publisher/world.xml",
    "https://moxie.foxnews.com/google-publisher/health.xml",
    "https://moxie.foxnews.com/google-publisher/science.xml",
    "https://moxie.foxnews.com/google-publisher/tech.xml",
    "https://moxie.foxnews.com/google-publisher/entertainment.xml",
    "https://moxie.foxnews.com/google-publisher/sports.xml",
    "https://moxie.foxnews.com/google-publisher/opinion.xml",
    "https://moxie.foxnews.com/google-publisher/lifestyle.xml",
    "https://moxie.foxnews.com/google-publisher/media.xml",
]

NBC_RSS = [
    "https://feeds.nbcnews.com/nbcnews/public/news",
    "https://feeds.nbcnews.com/nbcnews/public/politics",
    "https://feeds.nbcnews.com/nbcnews/public/health",
    "https://feeds.nbcnews.com/nbcnews/public/business",
    "https://feeds.nbcnews.com/nbcnews/public/tech",
    "https://feeds.nbcnews.com/nbcnews/public/investigations",
]

# ── Category landing pages ─────────────────────────────────────────────────────

FOXNEWS_CATEGORIES = [
    "https://www.foxnews.com/politics",
    "https://www.foxnews.com/us",
    "https://www.foxnews.com/world",
    "https://www.foxnews.com/health",
    "https://www.foxnews.com/science",
    "https://www.foxnews.com/tech",
    "https://www.foxnews.com/entertainment",
    "https://www.foxnews.com/sports",
    "https://www.foxnews.com/opinion",
    "https://www.foxnews.com/lifestyle",
    "https://www.foxnews.com/media",
    "https://www.foxnews.com/business",
]

NBC_CATEGORIES = [
    "https://www.nbcnews.com/politics",
    "https://www.nbcnews.com/us-news",
    "https://www.nbcnews.com/world",
    "https://www.nbcnews.com/health",
    "https://www.nbcnews.com/science",
    "https://www.nbcnews.com/tech-media",
    "https://www.nbcnews.com/entertainment",
    "https://www.nbcnews.com/sports",
    "https://www.nbcnews.com/business",
    "https://www.nbcnews.com/opinion",
    "https://www.nbcnews.com/investigations",
    "https://www.nbcnews.com/nbc-out",
]


# ── Helpers ────────────────────────────────────────────────────────────────────

def _fetch(url: str, timeout: int = 15) -> requests.Response | None:
    try:
        resp = requests.get(url, headers=HEADERS, timeout=timeout)
        resp.raise_for_status()
        return resp
    except requests.RequestException as exc:
        print(f"  Warning: {url} → {exc}")
        return None


def _xml_text(el) -> str:
    return (el.text or "").strip() if el is not None else ""


# ── Strategy 1: RSS feeds ──────────────────────────────────────────────────────

def _scrape_rss_feeds(feed_urls: list[str]) -> list[dict]:
    records: list[dict] = []
    for feed_url in feed_urls:
        print(f"  {feed_url}")
        resp = _fetch(feed_url)
        if resp is None:
            time.sleep(DELAY)
            continue
        try:
            root = ET.fromstring(resp.content)
            items = root.findall(".//item")
            if not items:
                items = root.findall(f".//{{{ATOM_NS}}}entry")
            for item in items:
                title_el = item.find("title")
                if title_el is None:
                    title_el = item.find(f"{{{ATOM_NS}}}title")
                link_el = item.find("link")
                if link_el is None:
                    link_el = item.find(f"{{{ATOM_NS}}}link")
                title = _xml_text(title_el)
                link = _xml_text(link_el) or (link_el.get("href", "") if link_el is not None else "")
                if title and link:
                    records.append({"url": link.strip(), "headline": title})
        except ET.ParseError as exc:
            print(f"  Parse error on {feed_url}: {exc}")
        time.sleep(DELAY)
    return records


# ── Strategy 2: News sitemaps ──────────────────────────────────────────────────

def _scrape_nbc_news_sitemap() -> list[dict]:
    """NBC's sitemap-news includes news:title for ~80 recent articles."""
    records: list[dict] = []
    url = "https://www.nbcnews.com/sitemap/nbcnews/sitemap-news"
    print(f"  NBC news sitemap: {url}")
    resp = _fetch(url)
    if resp is None:
        return records
    try:
        root = ET.fromstring(resp.content)
        for url_el in root.findall(f"{{{SM_NS}}}url"):
            loc = _xml_text(url_el.find(f"{{{SM_NS}}}loc"))
            title_el = url_el.find(f".//{{{NEWS_NS}}}title")
            title = _xml_text(title_el)
            if loc and title:
                records.append({"url": loc, "headline": title})
    except ET.ParseError as exc:
        print(f"  Parse error: {exc}")
    return records


def _get_fox_sitemap_urls(limit: int) -> list[str]:
    """Return up to `limit` article URLs from Fox News articles sitemap."""
    urls: list[str] = []
    resp = _fetch("https://www.foxnews.com/sitemap.xml?type=articles", timeout=20)
    if resp is None:
        return urls
    try:
        root = ET.fromstring(resp.content)
        for loc_el in root.findall(f".//{{{SM_NS}}}loc"):
            url = _xml_text(loc_el)
            if url and "foxnews.com" in url:
                urls.append(url)
            if len(urls) >= limit:
                break
    except ET.ParseError:
        pass
    return urls


def _get_nbc_monthly_sitemap_urls(limit: int) -> list[str]:
    """Return up to `limit` article URLs from NBC monthly article sitemaps (most recent first)."""
    urls: list[str] = []
    index_resp = _fetch("https://www.nbcnews.com/sitemap/nbcnews/sitemap-index", timeout=15)
    if index_resp is None:
        return urls
    try:
        root = ET.fromstring(index_resp.content)
        monthly = sorted(
            [_xml_text(l) for l in root.findall(f".//{{{SM_NS}}}loc") if "article.xml" in (_xml_text(l))],
            reverse=True,
        )
    except ET.ParseError:
        return urls

    for sm_url in monthly:
        if len(urls) >= limit:
            break
        resp = _fetch(sm_url, timeout=15)
        if resp is None:
            time.sleep(DELAY)
            continue
        try:
            root = ET.fromstring(resp.content)
            for loc_el in root.findall(f".//{{{SM_NS}}}loc"):
                url = _xml_text(loc_el)
                if url and "nbcnews.com" in url:
                    urls.append(url)
                if len(urls) >= limit:
                    break
        except ET.ParseError:
            pass
        time.sleep(DELAY)
    return urls


# ── Strategy 3: Category pages ─────────────────────────────────────────────────

def _scrape_category_pages() -> list[dict]:
    records: list[dict] = []

    fox_base = "https://www.foxnews.com"
    for cat_url in FOXNEWS_CATEGORIES:
        print(f"  {cat_url}")
        resp = _fetch(cat_url)
        if resp is None:
            time.sleep(DELAY)
            continue
        soup = BeautifulSoup(resp.text, "lxml")
        for tag in soup.select("article h3 a, article h2 a, .title a, h2.title a"):
            headline = tag.get_text(strip=True)
            href = tag.get("href", "")
            if not headline or not href:
                continue
            full_url = href if href.startswith("http") else fox_base + href
            if "foxnews.com" in full_url:
                records.append({"url": full_url, "headline": headline})
        time.sleep(DELAY)

    nbc_base = "https://www.nbcnews.com"
    for cat_url in NBC_CATEGORIES:
        print(f"  {cat_url}")
        resp = _fetch(cat_url)
        if resp is None:
            time.sleep(DELAY)
            continue
        soup = BeautifulSoup(resp.text, "lxml")
        for tag in soup.select("h2 a, h3 a, .tease-card__headline a, [class*='headline'] a"):
            headline = tag.get_text(strip=True)
            href = tag.get("href", "")
            if not headline or not href:
                continue
            full_url = href if href.startswith("http") else nbc_base + href
            if "nbcnews.com" in full_url:
                records.append({"url": full_url, "headline": headline})
        time.sleep(DELAY)

    return records


# ── Strategy 4: Individual article pages ─────────────────────────────────────

def _headline_from_article(url: str, soup: BeautifulSoup) -> str:
    if "foxnews.com" in url:
        selectors = [
            "h1.headline.speakable",
            "h1.headline",
            "h1[class*='headline']",
            "h1",
        ]
    else:
        selectors = [
            "h1.article-hero__headline",
            "h1[class*='headline']",
            "h1[class*='ArticleHero']",
            "h1",
        ]
    for sel in selectors:
        el = soup.select_one(sel)
        if el:
            return el.get_text(strip=True)
    return ""


def _scrape_article_pages(
    urls: list[str],
    checkpoint_csv: str | None = None,
    save_every: int = 100,
) -> list[dict]:
    records: list[dict] = []
    total = len(urls)
    print(f"  Scraping {total} individual article pages...")
    for i, url in enumerate(urls):
        resp = _fetch(url)
        if resp is not None:
            soup = BeautifulSoup(resp.text, "lxml")
            headline = _headline_from_article(url, soup)
            if headline:
                records.append({"url": url, "headline": headline})
        if (i + 1) % 50 == 0:
            print(f"    ... {i + 1}/{total} ({len(records)} found)")
        # Append new records to checkpoint every save_every articles
        if checkpoint_csv and len(records) % save_every == 0 and records:
            _append_checkpoint(checkpoint_csv, records[-save_every:])
        time.sleep(DELAY)
    # Flush any remaining records
    if checkpoint_csv and records:
        remainder = len(records) % save_every
        if remainder:
            _append_checkpoint(checkpoint_csv, records[-remainder:])
    return records


def _append_checkpoint(checkpoint_csv: str, new_records: list[dict]) -> None:
    path = Path(checkpoint_csv)
    df = pd.DataFrame(new_records)
    df.to_csv(checkpoint_csv, mode="a", header=not path.exists(), index=False)


# ── Dedup helper ───────────────────────────────────────────────────────────────

def _dedup(df: pd.DataFrame, existing_urls: set[str], existing_headlines: set[str]) -> pd.DataFrame:
    df = df.dropna(subset=["url", "headline"]).copy()
    df.loc[:, "headline"] = df["headline"].str.strip()
    df = df[df["headline"] != ""]
    df = df[~df["url"].isin(existing_urls)]
    df = df[~df["headline"].isin(existing_headlines)]
    return df.drop_duplicates(subset=["url"]).drop_duplicates(subset=["headline"]).reset_index(drop=True)


# ── Main collection entry point ────────────────────────────────────────────────

def collect(
    existing_csv: str = "preprocess/data/url_with_headlines.csv",
    output_csv: str = "preprocess/data/headlines_new.csv",
    scrape_articles: bool = False,
    sitemap_limit: int = 0,
    fox_only: bool = False,
) -> pd.DataFrame:
    """
    Collect new Fox News and NBC News headlines not already in existing_csv.

    Args:
        existing_csv:    Baseline CSV (url,headline) used for deduplication.
        output_csv:      Path to write newly collected records.
        scrape_articles: If True, scrape individual article pages for any URL
                         collected without a headline.
        sitemap_limit:   If > 0, also pull up to this many URLs from Fox/NBC
                         sitemaps and scrape their article pages for headlines.
                         Slow (~1.2s per URL). Set to e.g. 500 for ~10 min run.
    """
    existing_df = (
        pd.read_csv(existing_csv)
        if Path(existing_csv).exists()
        else pd.DataFrame(columns=["url", "headline"])
    )
    existing_urls: set[str] = set(existing_df["url"].dropna().tolist())
    existing_headlines: set[str] = set(existing_df["headline"].dropna().str.strip().tolist())
    print(f"Existing baseline: {len(existing_df)} rows")

    all_records: list[dict] = []

    # Strategy 1: RSS feeds
    print("\n[1/3] RSS feeds...")
    print("  Fox News:")
    all_records.extend(_scrape_rss_feeds(FOXNEWS_RSS))
    print("  NBC News:")
    all_records.extend(_scrape_rss_feeds(NBC_RSS))

    # Strategy 2: News sitemaps with embedded titles
    print("\n[2/3] News sitemaps (with titles)...")
    all_records.extend(_scrape_nbc_news_sitemap())

    # Strategy 3: Category pages
    print("\n[3/3] Category pages...")
    all_records.extend(_scrape_category_pages())

    new_df = _dedup(pd.DataFrame(all_records), existing_urls, existing_headlines)

    # Strategy 4 (opt-in): individual article scraping from sitemaps
    if sitemap_limit > 0:
        # Load checkpoint if it exists (resume after interruption)
        checkpoint_path = output_csv + ".checkpoint.csv"
        if Path(checkpoint_path).exists():
            checkpoint_df = pd.read_csv(checkpoint_path).drop_duplicates(subset=["url"])
            print(f"  Resuming from checkpoint: {len(checkpoint_df)} articles already scraped")
            existing_urls |= set(checkpoint_df["url"].dropna().tolist())
            existing_headlines |= set(checkpoint_df["headline"].dropna().str.strip().tolist())
        else:
            checkpoint_df = pd.DataFrame(columns=["url", "headline"])

        combined_existing = existing_urls | set(new_df["url"].tolist())
        print(f"\n[4/4] Sitemap bulk scrape (limit={sitemap_limit} per source, fox_only={fox_only})...")

        print(f"  Getting Fox article URLs...")
        fox_urls = [u for u in _get_fox_sitemap_urls(sitemap_limit * 2) if u not in combined_existing][:sitemap_limit]
        print(f"  Fox: {len(fox_urls)} new URLs → scraping...")
        fox_article_records = _scrape_article_pages(fox_urls, checkpoint_csv=checkpoint_path)

        if fox_only:
            nbc_article_records = []
        else:
            combined_existing |= {r["url"] for r in fox_article_records}
            print(f"  Getting NBC article URLs...")
            nbc_urls = [u for u in _get_nbc_monthly_sitemap_urls(sitemap_limit * 2) if u not in combined_existing][:sitemap_limit]
            print(f"  NBC: {len(nbc_urls)} new URLs → scraping...")
            nbc_article_records = _scrape_article_pages(nbc_urls, checkpoint_csv=checkpoint_path)

        extra_df = _dedup(
            pd.DataFrame(fox_article_records + nbc_article_records),
            existing_urls | set(new_df["url"].tolist()),
            existing_headlines | set(new_df["headline"].tolist()),
        )
        new_df = (
            pd.concat([new_df, extra_df], ignore_index=True)
              .drop_duplicates(subset=["url"])
              .drop_duplicates(subset=["headline"])
              .reset_index(drop=True)
        )

    # Strategy 4b (opt-in): scrape individual pages for URLs collected without titles
    elif scrape_articles:
        urls_no_headline = [
            r["url"] for r in all_records
            if not r.get("headline") and r.get("url") not in existing_urls
        ]
        if urls_no_headline:
            print(f"\n[4/4] Individual article scraping ({len(urls_no_headline)} URLs)...")
            extra_records = _scrape_article_pages(urls_no_headline)
            extra_df = _dedup(
                pd.DataFrame(extra_records),
                existing_urls | set(new_df["url"].tolist()),
                existing_headlines | set(new_df["headline"].tolist()),
            )
            new_df = (
                pd.concat([new_df, extra_df], ignore_index=True)
                  .drop_duplicates(subset=["url"])
                  .drop_duplicates(subset=["headline"])
                  .reset_index(drop=True)
            )

    # Remove checkpoint file on successful completion
    checkpoint_path = output_csv + ".checkpoint.csv"
    if Path(checkpoint_path).exists():
        Path(checkpoint_path).unlink()

    fox_n = new_df["url"].str.contains("foxnews", na=False).sum()
    nbc_n = new_df["url"].str.contains("nbcnews|msnbc", na=False).sum()
    print(f"\nNew unique records: {len(new_df)}  (Fox: {fox_n}, NBC: {nbc_n})")

    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    new_df.to_csv(output_csv, index=False)
    print(f"Saved new records → {output_csv}")

    merged = (
        pd.concat([existing_df, new_df], ignore_index=True)
          .drop_duplicates(subset=["headline"])
          .reset_index(drop=True)
    )
    merged_path = "preprocess/data/merged_headlines.csv"
    merged.to_csv(merged_path, index=False)
    fox_m = merged["url"].str.contains("foxnews", na=False).sum()
    nbc_m = merged["url"].str.contains("nbcnews|msnbc", na=False).sum()
    print(f"Merged total: {len(merged)} rows  (Fox: {fox_m}, NBC: {nbc_m}) → {merged_path}")

    return new_df


# ── Backward-compat wrapper ────────────────────────────────────────────────────

def scrape(output_path: str = "preprocess/data/headlines.csv") -> pd.DataFrame:
    return collect(output_csv=output_path)


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--existing", default="preprocess/data/url_with_headlines.csv")
    p.add_argument("--output", default="preprocess/data/headlines_new.csv")
    p.add_argument(
        "--scrape-articles",
        action="store_true",
        help="Scrape individual article pages for URLs without a headline",
    )
    p.add_argument(
        "--sitemap-limit",
        type=int,
        default=0,
        metavar="N",
        help=(
            "Pull up to N article URLs from Fox/NBC sitemaps and scrape each "
            "for its headline. Slow (~1.2s/URL). E.g. --sitemap-limit 500 ≈ 10 min."
        ),
    )
    p.add_argument(
        "--fox-only",
        action="store_true",
        help="Only scrape Fox News articles (skip NBC) during sitemap bulk scrape.",
    )
    args = p.parse_args()
    collect(
        existing_csv=args.existing,
        output_csv=args.output,
        scrape_articles=args.scrape_articles,
        sitemap_limit=args.sitemap_limit,
        fox_only=args.fox_only,
    )
