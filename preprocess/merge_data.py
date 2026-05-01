import pandas as pd
old = pd.read_csv('preprocess/data/url_with_headlines.csv')
new = pd.read_csv('preprocess/data/headlines.csv')
df = pd.concat([old, new], ignore_index=True)
df = df.drop_duplicates(subset=['headline']).reset_index(drop=True)
print(df.shape)
df.to_csv('preprocess/data/merged_headlines.csv', index=False)
