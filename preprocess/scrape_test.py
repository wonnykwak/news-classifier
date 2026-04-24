from scrape import scrape_foxnews, scrape_nbc
import pandas as pd
import os

fox = scrape_foxnews(max_pages=1)
nbc = scrape_nbc(max_pages=1)

print(f"Fox: {len(fox)} headlines")
print(f"NBC: {len(nbc)} headlines")

os.makedirs('data', exist_ok=True)
df = pd.DataFrame(fox + nbc)
df.to_csv('data/headlines.csv', index=False)
print('Saved!')
