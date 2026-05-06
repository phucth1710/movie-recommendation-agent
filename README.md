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

Technical details (from the current implementation):
- Input: a single `reference` string (title or IMDb ID). Autocomplete hits `/suggest?q=...` when the input has at least 2 characters.
- Resolution: `resolve_reference_movie()` attempts, in order: IMDb ID pattern `tt\d+`, exact ID lookup, exact title lookup, then a unique partial-title match; ambiguous partial matches return an error.
- Output fields (`/api/basic-description`): `imdb_id`, `title`, `content_type`, `year`, `rating`, `popularity`, `genre`, `description`, plus a `summary` string constructed from the top 3 genre tokens, year, rating, and vote count.
- Summary logic: `"{title} is a {content_type} from {year} with {genre} themes. Its IMDb rating is {rating} based on {popularity} votes."` (with safe fallbacks).
- AI insight (`/api/basic-description-insight`): uses the structured profile to ask an LLM for a 3-part markdown essay. Requires `OPENAI_API_KEY` and returns a markdown string in `insight`.

![Basic Movie Description](images/2%20-%20Basic%20Movie%20Description%20Part%201.jpg)
![AI Description Insight](images/3%20-%20Basic%20Movie%20Description%20Part%202.jpg)

### Finding Similar Movie
Search by a reference title/ID and return the top similar recommendations with similarity scores.

Technical details (from the current implementation):
- Input: `reference` string sent to `/api/recommend`. Autocomplete uses `/suggest?q=...`.
- Candidate pool: the full local dataset is filtered to allowed output types only: `movie`, `tvMovie`, `tvSeries`, and `tvMiniSeries`.
- Similarity scoring (base): weighted similarity uses genre Jaccard overlap, release-year proximity (max gap 30 years), and rating similarity; default weights are `genre=0.5`, `year=0.2`, `rating=0.3`.
- Scoping: recommendations first select a scoped pool of size `DEFAULT_SCOPE_SIZE=500` using the similarity model, then re-rank to the final `DEFAULT_TOP_K=10`.
- Composite score: ranking uses $composite = 0.7\times similarity + 0.2\times rating\_norm + 0.1\times popularity\_norm$, where rating is normalized to $[0,1]$ and popularity is log-normalized via `log1p`.
- Output fields: each result includes `imdb_id`, `title`, `content_type`, `year`, `rating`, `popularity`, `genre`, `similarity_score`, and `composite_score`. The UI prioritizes same-type results (show->show, movie->movie) and backfills with the other type when needed.
- AI insight (`/api/similar-insight`): given `source_reference` and a candidate (`candidate_imdb_id` or `candidate_title`), returns a long-form similarity explanation in markdown. Requires `OPENAI_API_KEY`.

![Similar Movie Results](images/4%20-%20Finding%20Similar%20Movie%20Part%201.jpg)
![Similarity Detail Insight](images/5%20-%20Finding%20Similar%20Movie%20Part%202.jpg)

### Compare Two Movies
Compare metadata and see AI commentary on genre overlap, differences, and reception.

Technical details (from the current implementation):
- Input: `first_reference` and `second_reference` sent to `/api/compare`.
- Resolution: the same reference resolution rules as Basic Description apply to both inputs.
- Output fields:
	- `first_movie` and `second_movie`: `imdb_id`, `title`, `content_type`, `genre`, `rating`, `popularity`, `year`, and a short natural-language `description`.
	- `comparison`: `shared_genres`, `rating_diff`, `popularity_diff`, `year_diff`, and winner flags `higher_rated`, `more_popular`, `newer` (values: `first`, `second`, or `tie`).
- Diff math: `rating_diff`, `popularity_diff`, and `year_diff` are computed as `first - second`.
- AI insight (`/api/compare-insight`): returns a structured, multi-section comparison essay in markdown. Requires `OPENAI_API_KEY`.

![Compare Two Movies Results](images/6%20-%20Compare%20Two%20Movies%20Part%201.jpg)
![AI Comparison Insight](images/7%20-%20Compare%20Two%20Movies%20Part%202.jpg)

### Ranking a Set of Movies
Rank a comma-separated list of titles and review an AI rationale for the ordering.

Technical details (from the current implementation):
- Input: either `references` (array) or `references_csv` (comma-separated string) sent to `/api/rank-set`.
- Resolution: each entry is resolved using `resolve_reference_movie()`; duplicates are removed by `imdb_id` and unresolved references are returned in `unresolved`.
- Ranking priority: rating, then popularity, then runtime length (descending).
- Runtime source: `Runtime` or `runtimeMinutes`, normalized to an integer.
- Output fields: `input_size`, `resolved_count`, `unresolved`, `ranking_priority`, and `results` entries with `imdb_id`, `title`, `content_type`, `year`, `genre`, `rating`, `popularity`, `length`.
- AI insight (`/api/rank-set-insight`): generates a long-form ranking rationale over the fixed ordering. Requires `OPENAI_API_KEY`.

![Ranking a Set Results](images/8%20-%20Ranking%20a%20Set%20of%20Movies%20Part%201.jpg)
![AI Ranking Insight](images/9%20-%20Ranking%20a%20Set%20of%20Movies%20Part%202.jpg)

### Ranking Top Movies
Filter by genre, year, and content type to return top-ranked titles, plus AI ranking notes.

Technical details (from the current implementation):
- Input: `genre` (optional), `year` (optional), `content_mode` (`both`, `movie`, or `show`), and `top_k` sent to `/api/rank-top`.
- Filtering: `content_mode` scopes by type; `genre` matches normalized genre tokens; `year` is an exact match.
- Ranking priority: rating, then popularity (descending).
- Output fields: `criteria`, `candidate_count`, `top_k`, and `results` entries with `imdb_id`, `title`, `content_type`, `genre`, `rating`, `popularity`, `year`.
- Suggestions: the UI calls `/suggest/genres?q=...` and `/suggest/years?q=...` for interactive pick lists.
- AI insight (`/api/rank-top-insight`): returns a ranked critique for the filtered slice. Requires `OPENAI_API_KEY`.

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
