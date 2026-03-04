# This script downloads all available IMDb metadata for movies/shows with over 50,000 votes.
# It merges all public IMDb datasets and saves the result to 'imdb_full_metadata.csv'.

import pandas as pd
import gzip
import os
import requests

BASENAME = 'https://datasets.imdbws.com/'
DATASETS = [
    'title.basics.tsv.gz',
    'title.ratings.tsv.gz',
    'title.akas.tsv.gz',
    'title.crew.tsv.gz',
    'title.principals.tsv.gz',
    'title.episode.tsv.gz',
    'name.basics.tsv.gz',
]

def download_file(url, filename):
    if not os.path.exists(filename):
        print(f"Downloading {filename}...")
        r = requests.get(url, stream=True)
        with open(filename, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    else:
        print(f"{filename} already exists.")

for fname in DATASETS:
    download_file(BASENAME + fname, fname)

# Load main tables
basics = pd.read_csv('title.basics.tsv.gz', sep='\t', dtype=str, compression='gzip')
ratings = pd.read_csv('title.ratings.tsv.gz', sep='\t', dtype=str, compression='gzip')
crew = pd.read_csv('title.crew.tsv.gz', sep='\t', dtype=str, compression='gzip')
principals = pd.read_csv('title.principals.tsv.gz', sep='\t', dtype=str, compression='gzip')
akas = pd.read_csv('title.akas.tsv.gz', sep='\t', dtype=str, compression='gzip')
name_basics = pd.read_csv('name.basics.tsv.gz', sep='\t', dtype=str, compression='gzip')

# Filter for movies/shows with >50,000 votes (move before merge)
ratings = ratings[ratings['numVotes'].astype(int) > 50000]

# Merge basics and ratings

df = pd.merge(basics, ratings, on='tconst')


# Group by 'tconst' before each left join to avoid duplication
crew = crew.rename(columns={'directors': 'director_ids', 'writers': 'writer_ids'})
crew_grouped = crew.groupby('tconst', as_index=False).first()
df = pd.merge(df, crew_grouped, on='tconst', how='left')

principals_grouped = principals.groupby('tconst', as_index=False).first()
df = pd.merge(df, principals_grouped, on='tconst', how='left')

akas_grouped = akas[['titleId','title','region','language','types','attributes']].groupby('titleId', as_index=False).first()
df = pd.merge(df, akas_grouped, left_on='tconst', right_on='titleId', how='left')

# Merge name info for cast/crew
principals_with_names = pd.merge(principals, name_basics, on='nconst', how='left')
principals_with_names_grouped = principals_with_names.groupby('tconst', as_index=False).first()
df = pd.merge(df, principals_with_names_grouped, on='tconst', how='left')

# Remove duplicates based on 'tconst' (IMDb ID)
df = df.drop_duplicates(subset=['tconst'])
# Save all columns to CSV
out_cols = list(df.columns)
df.to_csv('imdb_full_metadata.csv', index=False, columns=out_cols)
print(f"Saved {len(df)} unique entries with full metadata to imdb_full_metadata.csv.")
