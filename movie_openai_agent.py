import asyncio
import math
import os
import re
from functools import lru_cache
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict
from agents import Agent, Runner, function_tool, set_default_openai_key

from imdb_csv_utils import find_movie_by_id, find_movie_by_name, read_movies
from movie_agent_core import (
    DEFAULT_SOURCE_IMDB_ID,
    DEFAULT_SOURCE_TITLE,
    recommend_similar_movies,
)


DEFAULT_SCOPE_SIZE = 500
DEFAULT_TOP_K = 10
DEFAULT_WEIGHTS = {
    "genre": 0.5,
    "year": 0.2,
    "rating": 0.3,
}

ALLOWED_SHOW_TYPES = {"tvseries", "tvminiseries"}
ALLOWED_MOVIE_TYPES = {"movie", "tvmovie"}
ALLOWED_OUTPUT_TYPES = ALLOWED_SHOW_TYPES | ALLOWED_MOVIE_TYPES


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _normalized_type(movie: Any) -> str:
    return str(getattr(movie, "Type", "") or "").strip().lower()


def _is_show_type(movie: Any) -> bool:
    return _normalized_type(movie) in ALLOWED_SHOW_TYPES


def _is_movie_type(movie: Any) -> bool:
    return _normalized_type(movie) in ALLOWED_MOVIE_TYPES


def _is_allowed_output_type(movie: Any) -> bool:
    return _normalized_type(movie) in ALLOWED_OUTPUT_TYPES


def _movie_to_dict(movie: Any) -> Dict[str, Any]:
    return {
        "imdb_id": getattr(movie, "imdb_id", None),
        "Name": getattr(movie, "Name", None),
        "Type": getattr(movie, "Type", None),
        "Genre": getattr(movie, "Genre", None),
        "Rating": getattr(movie, "Rating", None),
        "Popularity": getattr(movie, "Popularity", None),
        "Year": getattr(movie, "Year", None),
        "Description": getattr(movie, "Description", None),
    }


@lru_cache(maxsize=1)
def load_movie_universe() -> List[Any]:
    return read_movies()


def resolve_reference_movie(reference: str, movies: List[Any]) -> Optional[Any]:
    normalized = reference.strip()
    if not normalized:
        return None

    # If user entered an IMDb-style id, require exact id match.
    if re.fullmatch(r"tt\d+", normalized, flags=re.IGNORECASE):
        return find_movie_by_id(normalized, movies=movies)

    movie = find_movie_by_id(normalized, movies=movies)
    if movie is not None:
        return movie

    movie = find_movie_by_name(normalized, movies=movies)
    if movie is not None:
        return movie

    # Prevent accidental broad matches like "t" or "a".
    if len(normalized) < 3:
        return None

    needle = normalized.lower()
    partial_matches: List[Any] = []
    for candidate in movies:
        name = str(getattr(candidate, "Name", "")).lower().strip()
        if needle in name:
            partial_matches.append(candidate)

    # Accept partial matching only when unambiguous.
    if len(partial_matches) == 1:
        return partial_matches[0]

    return None


def scope_candidates(reference_movie: Any, movies: List[Any], scope_size: int = DEFAULT_SCOPE_SIZE) -> List[Dict[str, Any]]:
    target_scope = max(1, min(scope_size, max(1, len(movies) - 1)))
    allowed_movies = [m for m in movies if _is_allowed_output_type(m)]
    return recommend_similar_movies(
        reference_movie,
        candidates=allowed_movies,
        top_k=target_scope,
        weights=DEFAULT_WEIGHTS,
    )


def rank_top_from_scoped_pool(scoped_rows: List[Dict[str, Any]], source_movie: Any, top_k: int = DEFAULT_TOP_K) -> List[Dict[str, Any]]:
    if not scoped_rows:
        return []

    max_popularity = max(_safe_int(getattr(row["movie"], "Popularity", 0), 0) for row in scoped_rows)
    max_popularity = max(1, max_popularity)

    ranked_rows: List[Dict[str, Any]] = []
    for row in scoped_rows:
        movie = row["movie"]
        similarity_score = _safe_float(row.get("score", 0.0), 0.0)
        rating_norm = max(0.0, min(1.0, _safe_float(getattr(movie, "Rating", 0.0), 0.0) / 10.0))
        popularity = _safe_int(getattr(movie, "Popularity", 0), 0)
        popularity_norm = math.log1p(max(0, popularity)) / math.log1p(max_popularity)

        composite_score = (0.7 * similarity_score) + (0.2 * rating_norm) + (0.1 * popularity_norm)
        ranked_rows.append(
            {
                "movie": movie,
                "similarity_score": round(similarity_score, 6),
                "composite_score": round(composite_score, 6),
            }
        )

    ranked_rows = sorted(ranked_rows, key=lambda x: x["composite_score"], reverse=True)

    if _is_show_type(source_movie):
        primary = [r for r in ranked_rows if _is_show_type(r["movie"])]
        secondary = [r for r in ranked_rows if _is_movie_type(r["movie"])]
    else:
        primary = [r for r in ranked_rows if _is_movie_type(r["movie"])]
        secondary = [r for r in ranked_rows if _is_show_type(r["movie"])]

    merged = primary + secondary
    return merged[: max(1, top_k)]


@function_tool
def movie_universe_summary() -> Dict[str, Any]:
    movies = load_movie_universe()
    return {
        "total_movies": len(movies),
        "default_scope_size": DEFAULT_SCOPE_SIZE,
        "default_top_k": DEFAULT_TOP_K,
    }


@function_tool
def recommend_movies_from_reference(
    reference: str,
    scope_size: int = DEFAULT_SCOPE_SIZE,
    top_k: int = DEFAULT_TOP_K,
) -> Dict[str, Any]:
    movies = load_movie_universe()
    source_movie = resolve_reference_movie(reference, movies)
    if source_movie is None:
        return {
            "error": f"Reference movie not found: {reference}",
            "total_movies": len(movies),
            "scope_size": 0,
            "top_k": 0,
            "results": [],
        }

    scoped_rows = scope_candidates(source_movie, movies, scope_size=scope_size)
    top_rows = rank_top_from_scoped_pool(scoped_rows, source_movie=source_movie, top_k=top_k)

    output_rows: List[Dict[str, Any]] = []
    for row in top_rows:
        movie = row["movie"]
        output_rows.append(
            {
                "imdb_id": getattr(movie, "imdb_id", None),
                "title": getattr(movie, "Name", None),
                "content_type": getattr(movie, "Type", None),
                "year": _safe_int(getattr(movie, "Year", None), 0),
                "rating": _safe_float(getattr(movie, "Rating", None), 0.0),
                "popularity": _safe_int(getattr(movie, "Popularity", None), 0),
                "genre": getattr(movie, "Genre", None),
                "similarity_score": row["similarity_score"],
                "composite_score": row["composite_score"],
            }
        )

    source = _movie_to_dict(source_movie)
    return {
        "source_movie": {
            "imdb_id": source.get("imdb_id"),
            "title": source.get("Name"),
            "content_type": source.get("Type"),
            "year": source.get("Year"),
            "rating": source.get("Rating"),
        },
        "total_movies": len(movies),
        "scope_size": len(scoped_rows),
        "top_k": len(output_rows),
        "weights": dict(DEFAULT_WEIGHTS),
        "results": output_rows,
    }


class MovieRankingRow(BaseModel):
    model_config = ConfigDict(extra="forbid")
    imdb_id: str
    title: str
    content_type: str
    year: int
    rating: float
    popularity: int
    similarity_score: float
    composite_score: float
    why_it_ranks: str


class MovieRanking(BaseModel):
    model_config = ConfigDict(extra="forbid")
    source_movie: str
    universe: str
    scoped_pool: str
    rankings: List[MovieRankingRow]
    methodology: str = ""
    notes: str = ""


def pretty_report(report: MovieRanking) -> None:
    print("Movie Recommendation Result")
    print(f"Source movie: {report.source_movie}")
    print(f"Universe: {report.universe}")
    print(f"Scoped pool: {report.scoped_pool}")
    print("Rankings (best to worst):")
    for idx, row in enumerate(report.rankings, 1):
        print(f"{idx}. {row.title} ({row.year}) [{row.imdb_id}] score={row.composite_score:.3f}")
        print(f"   Similarity: {row.similarity_score:.3f} | Rating: {row.rating:.1f} | Popularity: {row.popularity}")
        print(f"   Type: {row.content_type}")
        print(f"   Why: {row.why_it_ranks}")
    if report.methodology:
        print("Methodology:")
        print(report.methodology)
    if report.notes:
        print("Notes:")
        print(report.notes)


def build_agent(model: str = "gpt-5-nano") -> Agent:
    instruction = f"""
You are a deterministic movie ranking and recommendation agent.

Tool usage rules:
- ALWAYS call recommend_movies_from_reference(reference, scope_size={DEFAULT_SCOPE_SIZE}, top_k={DEFAULT_TOP_K}) exactly once.
- Optionally call movie_universe_summary() first if you need dataset context.
- Do not use any external data source.

Ranking workflow requirements:
- The full local universe has around 5,448 movies/shows.
- First scope to {DEFAULT_SCOPE_SIZE} candidates using the structured recommend_similar_movies logic already embedded in the tool.
- Then return the best {DEFAULT_TOP_K} movies/shows from the scoped set.
- Only output movies or shows (tvSeries/tvMiniSeries/movie/tvMovie). Never output episodes, games, shorts, or other content types.
- Prioritize same type as the source (show->show first, movie->movie first), then backfill with the other allowed type if fewer than {DEFAULT_TOP_K}.
- Avoid duplicate titles where possible.

Output requirements:
1) source_movie
2) universe
3) scoped_pool
4) rankings (exactly {DEFAULT_TOP_K} when available)
5) methodology
6) notes

Style requirements:
- Be concise, consistent, and factual.
- Do not invent missing values.
- Explain why each top item ranks highly using similarity, rating, and popularity evidence.
"""

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("Please set OPENAI_API_KEY in your environment.")
    set_default_openai_key(api_key)

    return Agent(
        name="Movie recommendation agent",
        instructions=instruction,
        tools=[recommend_movies_from_reference, movie_universe_summary],
        model=model,
        output_type=MovieRanking,
    )


async def main() -> None:
    movies = load_movie_universe()
    default_reference = DEFAULT_SOURCE_IMDB_ID if DEFAULT_SOURCE_IMDB_ID else DEFAULT_SOURCE_TITLE
    print("Enter a reference movie title or IMDb ID.")
    print(f"Press Enter to use default: {default_reference}")
    try:
        user_input = input("> ").strip()
    except EOFError:
        user_input = ""

    reference_movie = user_input if user_input else default_reference
    print(f"Using reference: {reference_movie}")

    if resolve_reference_movie(reference_movie, movies) is None:
        print(f"Movie does not exist in the local dataset: {reference_movie}")
        print("Please enter a valid IMDb ID (for example: tt0133093) or an exact movie/show title.")
        return

    user_prompt = (
        f"Recommend 10 movies and shows similar to {reference_movie}. "
        "Use the 500-item scoped pool strategy and provide the structured ranking output."
    )

    agent = build_agent()
    result = await Runner.run(agent, input=user_prompt)
    pretty_report(result.final_output)


if __name__ == "__main__":
    asyncio.run(main())
