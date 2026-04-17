import math
import os
from typing import Any, Dict, List

from agents import Agent, function_tool, set_default_openai_key
from pydantic import BaseModel, ConfigDict

from movie_agent_core import recommend_similar_movies
from movie_agent_shared import (
    DEFAULT_SCOPE_SIZE,
    DEFAULT_TOP_K,
    DEFAULT_WEIGHTS,
    is_allowed_output_type,
    is_movie_type,
    is_show_type,
    load_movie_universe,
    movie_to_dict,
    safe_float,
    safe_int,
    resolve_reference_movie,
)


def scope_candidates(reference_movie: Any, movies: List[Any], scope_size: int = DEFAULT_SCOPE_SIZE) -> List[Dict[str, Any]]:
    target_scope = max(1, min(scope_size, max(1, len(movies) - 1)))
    allowed_movies = [m for m in movies if is_allowed_output_type(m)]
    return recommend_similar_movies(
        reference_movie,
        candidates=allowed_movies,
        top_k=target_scope,
        weights=DEFAULT_WEIGHTS,
    )


def rank_top_from_scoped_pool(scoped_rows: List[Dict[str, Any]], source_movie: Any, top_k: int = DEFAULT_TOP_K) -> List[Dict[str, Any]]:
    if not scoped_rows:
        return []

    max_popularity = max(safe_int(getattr(row["movie"], "Popularity", 0), 0) for row in scoped_rows)
    max_popularity = max(1, max_popularity)

    ranked_rows: List[Dict[str, Any]] = []
    for row in scoped_rows:
        movie = row["movie"]
        similarity_score = safe_float(row.get("score", 0.0), 0.0)
        rating_norm = max(0.0, min(1.0, safe_float(getattr(movie, "Rating", 0.0), 0.0) / 10.0))
        popularity = safe_int(getattr(movie, "Popularity", 0), 0)
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

    if is_show_type(source_movie):
        primary = [r for r in ranked_rows if is_show_type(r["movie"])]
        secondary = [r for r in ranked_rows if is_movie_type(r["movie"])]
    else:
        primary = [r for r in ranked_rows if is_movie_type(r["movie"])]
        secondary = [r for r in ranked_rows if is_show_type(r["movie"])]

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
                "year": safe_int(getattr(movie, "Year", None), 0),
                "rating": safe_float(getattr(movie, "Rating", None), 0.0),
                "popularity": safe_int(getattr(movie, "Popularity", None), 0),
                "genre": getattr(movie, "Genre", None),
                "similarity_score": row["similarity_score"],
                "composite_score": row["composite_score"],
            }
        )

    source = movie_to_dict(source_movie)
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
