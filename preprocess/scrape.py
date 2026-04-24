import time
import requests
import pandas as pd
from bs4 import BeautifulSoup

HEADERS = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"}
DELAY = 1.0  # seconds between requests

FOXNEWS_CATEGORIES = [
    "https://www.foxnews.com/politics",
    "https://www.foxnews.com/us",
    "https://www.foxnews.com/world",
    "https://www.foxnews.com/health",
    "https://www.foxnews.com/science",
    "https://www.foxnews.com/tech",
    "https://www.foxnews.com/entertainment",
    "https://www.foxnews.com/sports",
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
]


def _get(url: str) -> BeautifulSoup:
    resp = requests.get(url, headers=HEADERS, timeout=10)
    resp.raise_for_status()
    return BeautifulSoup(resp.text, "lxml")


def scrape_foxnews() -> list[dict]:
    records = []
    base = "https://www.foxnews.com"

    for category_url in FOXNEWS_CATEGORIES:
        print(f"  {category_url}")
        try:
            soup = _get(category_url)
            for tag in soup.select("article h3 a, article h2 a, .title a"):
                headline = tag.get_text(strip=True)
                href = tag.get("href", "")
                if not headline:
                    continue
                full_url = href if href.startswith("http") else base + href
                records.append({"url": full_url, "headline": headline})
        except requests.RequestException as e:
            print(f"  Warning: failed to fetch {category_url}: {e}")

        time.sleep(DELAY)

    return records


def scrape_nbc() -> list[dict]:
    records = []
    base = "https://www.nbcnews.com"

    for category_url in NBC_CATEGORIES:
        print(f"  {category_url}")
        try:
            soup = _get(category_url)
            for tag in soup.select("h2 a, h3 a, .tease-card__headline a, .article-hero__headline a"):
                headline = tag.get_text(strip=True)
                href = tag.get("href", "")
                if not headline:
                    continue
                full_url = href if href.startswith("http") else base + href
                records.append({"url": full_url, "headline": headline})
        except requests.RequestException as e:
            print(f"  Warning: failed to fetch {category_url}: {e}")

        time.sleep(DELAY)

    return records


def scrape(output_path: str = "data/headlines.csv") -> pd.DataFrame:
    print("Scraping FoxNews...")
    fox_records = scrape_foxnews()
    print(f"  {len(fox_records)} headlines collected")

    print("Scraping NBC News...")
    nbc_records = scrape_nbc()
    print(f"  {len(nbc_records)} headlines collected")

    df = pd.DataFrame(fox_records + nbc_records)
    df = df.drop_duplicates(subset=["headline"]).reset_index(drop=True)
    df.to_csv(output_path, index=False)
    print(f"Saved {len(df)} total headlines to '{output_path}'")
    return df


if __name__ == "__main__":
    scrape()
