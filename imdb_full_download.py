# This script downloads the IMDb title dataset and ratings dataset, then filters for movies/shows with over 50,000 votes.
# Results are saved to 'imdb_filtered.csv' with Name, Description, Rating, Popularity, Genre.

import pandas as pd
import gzip
import os
import requests

# IMDb dataset URLs
BASENAME = 'https://datasets.imdbws.com/'
TITLE_BASICS = 'title.basics.tsv.gz'
TITLE_RATINGS = 'title.ratings.tsv.gz'

# Download function
def download_file(url, filename):
    if not os.path.exists(filename):
        print(f"Downloading {filename}...")
        r = requests.get(url, stream=True)
        with open(filename, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    else:
        print(f"{filename} already exists.")

# Download datasets
for fname in [TITLE_BASICS, TITLE_RATINGS]:
    download_file(BASENAME + fname, fname)

# Read datasets
title_basics = pd.read_csv(TITLE_BASICS, sep='\t', dtype=str, compression='gzip')
title_ratings = pd.read_csv(TITLE_RATINGS, sep='\t', dtype=str, compression='gzip')

# Merge datasets on tconst
df = pd.merge(title_basics, title_ratings, on='tconst')

# Filter for movies/shows with >50,000 votes
df = df[df['numVotes'].astype(int) > 50000]

# Select relevant columns and rename
df = df[['primaryTitle', 'titleType', 'genres', 'averageRating', 'numVotes', 'startYear', 'originalTitle']]
df = df.rename(columns={
    'primaryTitle': 'Name',
    'averageRating': 'Rating',
    'numVotes': 'Popularity',
    'genres': 'Genre',
    'titleType': 'Type',
    'startYear': 'Year',
    'originalTitle': 'Description' # IMDb does not provide plot in public dataset
})

# Save to CSV
df.to_csv('imdb_filtered.csv', index=False)
print(f"Saved {len(df)} entries to imdb_filtered.csv.")
