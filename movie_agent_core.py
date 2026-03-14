"""
movie_agent_core.py
Core logic for a stateless, terminal-based movie recommendation and ranking agent.
This module provides functions for query interpretation, constraint extraction, metadata retrieval,
similarity computation, multi-criteria ranking, and explanation generation.
"""
import pandas as pd
from typing import List, Dict, Optional, Any
from imdb_csv_utils import read_movies, find_movie_by_id, find_movie_by_name, filter_movies

# --- Local Test Run Configuration (no CLI arguments required) ---
DEFAULT_SOURCE_IMDB_ID = "tt0944947"
DEFAULT_SOURCE_TITLE = 'Inception'
DEFAULT_TOP_K = 5
DEFAULT_WEIGHTS = {
    'genre': 0.5,
    'year': 0.2,
    'rating': 0.3,
}

# --- Query Interpretation & Constraint Extraction ---
def parse_query(query: str) -> Dict[str, Any]:
    """Interpret a natural language or structured query and extract constraints/criteria."""
    # Placeholder: In production, use NLP or rule-based parsing
    # Example: {'genre': 'sci-fi', 'min_rating': 7.5, 'max_year': 2010}
    return {}

# --- Metadata Retrieval ---
def get_movie_metadata(imdb_id: str) -> Optional[Dict]:
    """Retrieve metadata for a movie by IMDb id."""
    return find_movie_by_id(imdb_id)

def get_movies_by_constraints(constraints: Dict[str, Any]) -> List[Dict]:
    """Retrieve movies matching constraints (genre, rating, year, etc)."""
    return filter_movies(
        rating_count_min=constraints.get('min_votes'),
        rating_score_min=constraints.get('min_rating'),
        language=constraints.get('language'),
        genre=constraints.get('genre')
    )

# --- Similarity Computation ---
def _get_value(movie: Any, key: str, default: Any = None) -> Any:
    """Read a field from either a dict-like object or an attribute-based object."""
    if isinstance(movie, dict):
        return movie.get(key, default)
    return getattr(movie, key, default)


def _parse_genres(movie: Any) -> set:
    """Parse the movie genre field into a normalized set of tokens."""
    raw_genre = _get_value(movie, 'Genre', '')
    if raw_genre is None:
        return set()
    return {g.strip().lower() for g in str(raw_genre).split(',') if g.strip()}


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def genre_overlap_similarity(movie_a: Any, movie_b: Any) -> float:
    """Compute Jaccard similarity for movie genres in [0, 1]."""
    genres_a = _parse_genres(movie_a)
    genres_b = _parse_genres(movie_b)
    if not genres_a and not genres_b:
        return 0.0
    union = genres_a | genres_b
    if not union:
        return 0.0
    return len(genres_a & genres_b) / len(union)


def release_year_proximity(movie_a: Any, movie_b: Any, max_year_gap: int = 30) -> float:
    """Return year proximity in [0, 1], where 1 means same release year."""
    year_a = _to_float(_get_value(movie_a, 'Year', None), default=-1)
    year_b = _to_float(_get_value(movie_b, 'Year', None), default=-1)
    if year_a < 0 or year_b < 0:
        return 0.0

    gap = abs(year_a - year_b)
    bounded_gap = min(gap, max_year_gap)
    return 1.0 - (bounded_gap / float(max_year_gap))


def normalize_rating_value(rating: Any, min_rating: float = 0.0, max_rating: float = 10.0) -> float:
    """Normalize a rating value to [0, 1]."""
    rating_value = _to_float(rating, default=min_rating)
    if max_rating <= min_rating:
        return 0.0
    normalized = (rating_value - min_rating) / (max_rating - min_rating)
    return max(0.0, min(1.0, normalized))


def rating_similarity(movie_a: Any, movie_b: Any, min_rating: float = 0.0, max_rating: float = 10.0) -> float:
    """Return rating similarity in [0, 1] based on normalized rating distance."""
    ra = normalize_rating_value(_get_value(movie_a, 'Rating', 0), min_rating=min_rating, max_rating=max_rating)
    rb = normalize_rating_value(_get_value(movie_b, 'Rating', 0), min_rating=min_rating, max_rating=max_rating)
    return 1.0 - abs(ra - rb)


def estimate_rating_bounds(movies: List[Any]) -> Dict[str, float]:
    """Estimate rating normalization bounds from a movie collection."""
    ratings = [_to_float(_get_value(m, 'Rating', None), default=float('nan')) for m in movies]
    ratings = [r for r in ratings if pd.notna(r)]
    if not ratings:
        return {'min_rating': 0.0, 'max_rating': 10.0}
    min_rating = min(ratings)
    max_rating = max(ratings)
    if min_rating == max_rating:
        return {'min_rating': 0.0, 'max_rating': 10.0}
    return {'min_rating': min_rating, 'max_rating': max_rating}


def calculate_structured_similarity(
    movie: Any,
    candidate: Any,
    weights: Optional[Dict[str, float]] = None,
    max_year_gap: int = 30,
    min_rating: float = 0.0,
    max_rating: float = 10.0,
) -> float:
    """Weighted similarity using genre overlap, year proximity, and normalized rating."""
    default_weights = {'genre': 0.5, 'year': 0.2, 'rating': 0.3}
    weights = weights or default_weights

    # Normalize weight sum for stability even when partial weight maps are provided.
    genre_w = _to_float(weights.get('genre', default_weights['genre']))
    year_w = _to_float(weights.get('year', default_weights['year']))
    rating_w = _to_float(weights.get('rating', default_weights['rating']))
    total_w = genre_w + year_w + rating_w
    if total_w <= 0:
        genre_w, year_w, rating_w, total_w = 0.5, 0.2, 0.3, 1.0

    genre_score = genre_overlap_similarity(movie, candidate)
    year_score = release_year_proximity(movie, candidate, max_year_gap=max_year_gap)
    rating_score = rating_similarity(movie, candidate, min_rating=min_rating, max_rating=max_rating)

    return (
        (genre_w * genre_score) +
        (year_w * year_score) +
        (rating_w * rating_score)
    ) / total_w


def score_candidates_by_similarity(
    movie: Any,
    candidates: List[Any],
    weights: Optional[Dict[str, float]] = None,
    max_year_gap: int = 30,
) -> List[Dict[str, Any]]:
    """Score candidates and return rows with component scores and total score."""
    bounds = estimate_rating_bounds([movie] + list(candidates))
    rows: List[Dict[str, Any]] = []

    for candidate in candidates:
        genre_score = genre_overlap_similarity(movie, candidate)
        year_score = release_year_proximity(movie, candidate, max_year_gap=max_year_gap)
        rating_score = rating_similarity(
            movie,
            candidate,
            min_rating=bounds['min_rating'],
            max_rating=bounds['max_rating'],
        )
        total_score = calculate_structured_similarity(
            movie,
            candidate,
            weights=weights,
            max_year_gap=max_year_gap,
            min_rating=bounds['min_rating'],
            max_rating=bounds['max_rating'],
        )
        rows.append({
            'movie': candidate,
            'score': total_score,
            'genre_score': genre_score,
            'year_score': year_score,
            'rating_score': rating_score,
        })

    return sorted(rows, key=lambda row: row['score'], reverse=True)


def compute_similarity(movie: Dict, candidates: List[Dict], criteria: List[str]) -> List[Dict]:
    """Compute similarity ranking using structured metadata criteria.

    Supported criteria entries:
    - 'genre'
    - 'year'
    - 'rating'
    """
    criteria_set = {c.strip().lower() for c in criteria} if criteria else {'genre', 'year', 'rating'}
    weights = {
        'genre': 1.0 if 'genre' in criteria_set else 0.0,
        'year': 1.0 if 'year' in criteria_set else 0.0,
        'rating': 1.0 if 'rating' in criteria_set else 0.0,
    }
    scored = score_candidates_by_similarity(movie, candidates, weights=weights)
    return [row['movie'] for row in scored]


def recommend_similar_movies(
    movie: Any,
    candidates: Optional[List[Any]] = None,
    top_k: int = 10,
    weights: Optional[Dict[str, float]] = None,
    max_year_gap: int = 30,
    exclude_same_id: bool = True,
) -> List[Dict[str, Any]]:
    """Return top-K structured similarity recommendations with score breakdowns."""
    if candidates is None:
        candidates = read_movies()

    movie_id = _get_value(movie, 'imdb_id', None)
    if exclude_same_id and movie_id is not None:
        candidates = [c for c in candidates if _get_value(c, 'imdb_id', None) != movie_id]

    scored = score_candidates_by_similarity(
        movie=movie,
        candidates=candidates,
        weights=weights,
        max_year_gap=max_year_gap,
    )
    return scored[:max(0, top_k)]


def recommend_similar_by_id(
    imdb_id: str,
    top_k: int = 10,
    weights: Optional[Dict[str, float]] = None,
    candidates: Optional[List[Any]] = None,
) -> List[Dict[str, Any]]:
    """Recommend similar movies for a given IMDb id."""
    if candidates is None:
        candidates = read_movies()
    target = find_movie_by_id(imdb_id, movies=candidates)
    if target is None:
        return []
    return recommend_similar_movies(target, candidates=candidates, top_k=top_k, weights=weights)


def recommend_similar_by_name(
    name: str,
    top_k: int = 10,
    weights: Optional[Dict[str, float]] = None,
    candidates: Optional[List[Any]] = None,
) -> List[Dict[str, Any]]:
    """Recommend similar movies for a given movie title (exact match, case-insensitive)."""
    if candidates is None:
        candidates = read_movies()
    target = find_movie_by_name(name, movies=candidates)
    if target is None:
        return []
    return recommend_similar_movies(target, candidates=candidates, top_k=top_k, weights=weights)


def explain_similarity_components(
    movie: Any,
    candidate: Any,
    weights: Optional[Dict[str, float]] = None,
    max_year_gap: int = 30,
    min_rating: float = 0.0,
    max_rating: float = 10.0,
) -> Dict[str, float]:
    """Return component-level similarity contributions for interpretation."""
    genre_score = genre_overlap_similarity(movie, candidate)
    year_score = release_year_proximity(movie, candidate, max_year_gap=max_year_gap)
    rating_score = rating_similarity(
        movie,
        candidate,
        min_rating=min_rating,
        max_rating=max_rating,
    )
    total_score = calculate_structured_similarity(
        movie,
        candidate,
        weights=weights,
        max_year_gap=max_year_gap,
        min_rating=min_rating,
        max_rating=max_rating,
    )
    return {
        'genre_score': genre_score,
        'year_score': year_score,
        'rating_score': rating_score,
        'total_score': total_score,
    }

# --- Multi-Criteria Ranking ---
def rank_movies(movies: List[Dict], criteria: List[str]) -> List[Dict]:
    """Rank movies based on multiple criteria (e.g., rating, popularity)."""
    def sort_key(movie):
        keys = []
        for c in criteria:
            if c == 'rating':
                keys.append(float(movie.get('Rating', 0)))
            elif c == 'popularity':
                keys.append(int(movie.get('Popularity', 0)))
            # Add more criteria as needed
        return tuple(keys)
    return sorted(movies, key=sort_key, reverse=True)

# --- Explanation Generation ---
def generate_explanation(movies: List[Dict], criteria: List[str]) -> str:
    """Generate a structured explanation for the ranking/recommendation."""
    explanation = f"Movies ranked by: {', '.join(criteria)}\n"
    for i, movie in enumerate(movies, 1):
        explanation += f"{i}. {movie.get('Name')} (Rating: {movie.get('Rating')}, Popularity: {movie.get('Popularity')})\n"
    return explanation

# --- Comparative Analysis ---
def compare_movies(movie_ids: List[str], criteria: List[str]) -> str:
    """Compare multiple movies based on criteria and return a summary."""
    movies = [find_movie_by_id(mid) for mid in movie_ids]
    ranked = rank_movies([m for m in movies if m], criteria)
    return generate_explanation(ranked, criteria)


def _format_recommendation_rows(rows: List[Dict[str, Any]]) -> str:
    """Render recommendation rows for terminal output."""
    lines: List[str] = []
    for idx, row in enumerate(rows, 1):
        movie = row['movie']
        name = _get_value(movie, 'Name', 'Unknown')
        imdb_id = _get_value(movie, 'imdb_id', 'N/A')
        rating = _get_value(movie, 'Rating', 'N/A')
        year = _get_value(movie, 'Year', 'N/A')
        lines.append(
            f"{idx}. {name} ({year}) [{imdb_id}] "
            f"score={row['score']:.4f} "
            f"(genre={row['genre_score']:.4f}, year={row['year_score']:.4f}, rating={row['rating_score']:.4f}) "
            f"rating={rating}"
        )
    return "\n".join(lines)


def main() -> None:
    """Run similarity recommendation using fixed local settings."""

    movies = read_movies()
    source = None
    if DEFAULT_SOURCE_IMDB_ID:
        source = find_movie_by_id(DEFAULT_SOURCE_IMDB_ID, movies=movies)
    elif DEFAULT_SOURCE_TITLE:
        source = find_movie_by_name(DEFAULT_SOURCE_TITLE, movies=movies)

    if source is None:
        search_value = DEFAULT_SOURCE_IMDB_ID if DEFAULT_SOURCE_IMDB_ID else DEFAULT_SOURCE_TITLE
        print(f"Movie not found: {search_value}")
        return

    weights = dict(DEFAULT_WEIGHTS)
    rows = recommend_similar_movies(source, candidates=movies, top_k=DEFAULT_TOP_K, weights=weights)

    source_name = _get_value(source, 'Name', 'Unknown')
    source_year = _get_value(source, 'Year', 'N/A')
    source_id = _get_value(source, 'imdb_id', 'N/A')
    print(f"Source: {source_name} ({source_year}) [{source_id}]")
    print(f"Top K: {DEFAULT_TOP_K}")
    print(f"Weights: genre={weights['genre']}, year={weights['year']}, rating={weights['rating']}")
    print('Top recommendations:')
    print(_format_recommendation_rows(rows) if rows else 'No recommendations found.')


if __name__ == '__main__':
    main()
