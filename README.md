# Movie Recommendation Agent
A local movie search and ranking workspace that explores IMDb metadata with a clean web UI and AI-generated insights. The experience centers on quick lookups, similarity discovery, side-by-side comparisons, and ranked lists.

## What You Can Do
- Search a title or IMDb ID to get structured metadata and a concise summary.
- Request AI description insights that expand context and audience fit.
- Find similar movies with similarity and composite scores.
- Compare two titles side by side with AI commentary.
- Rank a custom set of titles by rating, popularity, and runtime.
- Pull top-ranked titles by genre, year, and content type.

## UI Walkthrough
![Movie Search Home](images/1-%20Movie%20Search%20Opening%20Screen.jpg)

### Basic Movie Description
Look up a title or IMDb ID to view metadata like rating, popularity, genres, and a short description.

![Basic Movie Description](images/2%20-%20Basic%20Movie%20Description%20Part%201.jpg)
![AI Description Insight](images/3%20-%20Basic%20Movie%20Description%20Part%202.jpg)

### Finding Similar Movie
Search by a reference title/ID and return the top similar recommendations with similarity scores.

![Similar Movie Results](images/4%20-%20Finding%20Similar%20Movie%20Part%201.jpg)
![Similarity Detail Insight](images/5%20-%20Finding%20Similar%20Movie%20Part%202.jpg)

### Compare Two Movies
Compare metadata and see AI commentary on genre overlap, differences, and reception.

![Compare Two Movies Results](images/6%20-%20Compare%20Two%20Movies%20Part%201.jpg)
![AI Comparison Insight](images/7%20-%20Compare%20Two%20Movies%20Part%202.jpg)

### Ranking a Set of Movies
Rank a comma-separated list of titles and review an AI rationale for the ordering.

![Ranking a Set Results](images/8%20-%20Ranking%20a%20Set%20of%20Movies%20Part%201.jpg)
![AI Ranking Insight](images/9%20-%20Ranking%20a%20Set%20of%20Movies%20Part%202.jpg)

### Ranking Top Movies
Filter by genre, year, and content type to return top-ranked titles, plus AI ranking notes.

![Ranking Top Movies](images/10%20-%20Ranking%20Top%20Movies%20Part%201.jpg)
![AI Ranking Notes](images/11%20-%20Ranking%20Top%20Movies%20Part%202.jpg)

## Data Integrity Checks
Validate preprocessing outputs for missing values, duplicate IDs, and deterministic feature extraction:

```bash
python preprocessing_validation.py
```

Target specific datasets with machine-readable output:

```bash
python preprocessing_validation.py imdb_filtered.csv imdb_full_metadata.csv --json
```

## Tests
```bash
python -m unittest discover -s tests -p "test_*.py"
```
