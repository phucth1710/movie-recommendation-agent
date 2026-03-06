"""
movie_agent_core.py
Core logic for a stateless, terminal-based movie recommendation and ranking agent.
This module provides functions for query interpretation, constraint extraction, metadata retrieval,
similarity computation, multi-criteria ranking, and explanation generation.
"""
import pandas as pd
from typing import List, Dict, Optional, Any
from imdb_csv_utils import read_movies, find_movie_by_id, find_movie_by_name, filter_movies

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
def compute_similarity(movie: Dict, candidates: List[Dict], criteria: List[str]) -> List[Dict]:
    """Compute similarity between a movie and candidate movies based on criteria."""
    # Placeholder: simple genre overlap, can be extended
    def score(candidate):
        score = 0
        if 'Genre' in movie and 'Genre' in candidate:
            movie_genres = set(str(movie['Genre']).split(','))
            candidate_genres = set(str(candidate['Genre']).split(','))
            score += len(movie_genres & candidate_genres)
        # Add more criteria as needed
        return score
    return sorted(candidates, key=score, reverse=True)

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
