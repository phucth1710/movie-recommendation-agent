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


def _safe_runtime_length(movie: Any) -> int:
    """Get runtime/length in minutes when available; fallback to 0."""
    runtime = getattr(movie, "Runtime", None)
    if runtime in (None, ""):
        runtime = getattr(movie, "runtimeMinutes", None)
    return _safe_int(runtime, 0)


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


def _parse_genre_tokens(raw_genre: Any) -> List[str]:
    if raw_genre is None:
        return []
    return [g.strip() for g in str(raw_genre).split(",") if g.strip()]


def _truncate_text(value: Any, max_len: int = 120) -> str:
    text = str(value or "").strip()
    if len(text) <= max_len:
        return text
    return text[: max(0, max_len - 3)] + "..."


def _build_movie_short_summary(movie: Dict[str, Any]) -> str:
    """Create a compact, human-readable summary from available local metadata."""
    title = str(movie.get("Name") or "This title")
    content_type = str(movie.get("Type") or "title")
    year = _safe_int(movie.get("Year"), 0)
    rating = _safe_float(movie.get("Rating"), 0.0)
    popularity = _safe_int(movie.get("Popularity"), 0)
    genres = _parse_genre_tokens(movie.get("Genre"))

    genre_phrase = " and ".join(genres[:2]) if genres else "mixed genres"
    year_phrase = str(year) if year > 0 else "an unknown year"
    popularity_phrase = f"{popularity:,} votes" if popularity > 0 else "limited vote data"

    return (
        f"{title} is a {content_type} from {year_phrase} centered on {genre_phrase} themes, "
        f"with an IMDb rating of {rating:.1f} and {popularity_phrase}."
    )


def compare_two_movies(first_reference: str, second_reference: str, movies: List[Any]) -> Dict[str, Any]:
    """Compare two movies/shows across core metadata fields."""
    first_movie = resolve_reference_movie(first_reference, movies)
    second_movie = resolve_reference_movie(second_reference, movies)

    if first_movie is None or second_movie is None:
        return {
            "error": "One or both references could not be resolved.",
            "first_found": first_movie is not None,
            "second_found": second_movie is not None,
        }

    first = _movie_to_dict(first_movie)
    second = _movie_to_dict(second_movie)

    first_genres = set(_parse_genre_tokens(first.get("Genre")))
    second_genres = set(_parse_genre_tokens(second.get("Genre")))
    shared_genres = sorted(first_genres & second_genres)

    first_rating = _safe_float(first.get("Rating"), 0.0)
    second_rating = _safe_float(second.get("Rating"), 0.0)
    first_popularity = _safe_int(first.get("Popularity"), 0)
    second_popularity = _safe_int(second.get("Popularity"), 0)
    first_year = _safe_int(first.get("Year"), 0)
    second_year = _safe_int(second.get("Year"), 0)

    return {
        "first_movie": {
            "imdb_id": first.get("imdb_id"),
            "title": first.get("Name"),
            "content_type": first.get("Type"),
            "genre": first.get("Genre"),
            "rating": first_rating,
            "popularity": first_popularity,
            "year": first_year,
            "description": _build_movie_short_summary(first),
        },
        "second_movie": {
            "imdb_id": second.get("imdb_id"),
            "title": second.get("Name"),
            "content_type": second.get("Type"),
            "genre": second.get("Genre"),
            "rating": second_rating,
            "popularity": second_popularity,
            "year": second_year,
            "description": _build_movie_short_summary(second),
        },
        "comparison": {
            "shared_genres": shared_genres,
            "rating_diff": round(first_rating - second_rating, 3),
            "popularity_diff": first_popularity - second_popularity,
            "year_diff": first_year - second_year,
            "higher_rated": "first" if first_rating > second_rating else ("second" if second_rating > first_rating else "tie"),
            "more_popular": "first" if first_popularity > second_popularity else ("second" if second_popularity > first_popularity else "tie"),
            "newer": "first" if first_year > second_year else ("second" if second_year > first_year else "tie"),
        },
    }


def pretty_comparison_report(report: Dict[str, Any]) -> None:
    if report.get("error"):
        print(report["error"])
        return

    first = report["first_movie"]
    second = report["second_movie"]
    comp = report["comparison"]

    left_name = str(first.get("title") or "First")
    right_name = str(second.get("title") or "Second")
    label_width = 14
    left_width = 46
    right_width = 46

    def _row(label: str, left: Any, right: Any) -> str:
        left_txt = str(left or "")
        right_txt = str(right or "")
        return f"{label:<{label_width}} | {left_txt:<{left_width}} | {right_txt:<{right_width}}"

    def _winner_name(winner_key: str) -> str:
        if winner_key == "first":
            return left_name
        if winner_key == "second":
            return right_name
        return "Tie"

    def _signed(value: float, precision: int = 1) -> str:
        return f"{value:+.{precision}f}"

    print("Movie Comparison")
    print(_row("Field", left_name, right_name))
    print("-" * (label_width + left_width + right_width + 6))
    print(_row("IMDb ID", first.get("imdb_id"), second.get("imdb_id")))
    print(_row("Type", first.get("content_type"), second.get("content_type")))
    print(_row("Genre", first.get("genre"), second.get("genre")))
    print(_row("Rating", f"{_safe_float(first.get('rating')):.1f}", f"{_safe_float(second.get('rating')):.1f}"))
    print(_row("Popularity", first.get("popularity"), second.get("popularity")))
    print(_row("Year", first.get("year"), second.get("year")))
    print(_row("Description", _truncate_text(first.get("description")), _truncate_text(second.get("description"))))
    print()
    shared = ", ".join(comp.get("shared_genres", [])) if comp.get("shared_genres") else "None"
    print(f"Shared genres: {shared}")

    rating_diff = _safe_float(comp.get("rating_diff"), 0.0)
    popularity_diff = _safe_int(comp.get("popularity_diff"), 0)
    year_diff = _safe_int(comp.get("year_diff"), 0)

    print("Comparison summary:")
    print(
        f"- Rating: {_winner_name(comp.get('higher_rated', 'tie'))} "
        f"({_signed(abs(rating_diff), 1)} points difference, first-minus-second={_signed(rating_diff, 1)})."
    )
    print(
        f"- Popularity: {_winner_name(comp.get('more_popular', 'tie'))} "
        f"({popularity_diff:+d} votes first-minus-second)."
    )
    print(
        f"- Release year: {_winner_name(comp.get('newer', 'tie'))} "
        f"({year_diff:+d} years first-minus-second)."
    )

    print("Interpretation:")
    if comp.get("higher_rated") == "tie" and comp.get("more_popular") == "tie" and comp.get("newer") == "tie":
        print("- Both titles are effectively matched on rating, popularity, and release timing.")
    else:
        if comp.get("higher_rated") != "tie":
            print(f"- Critical reception edge goes to {_winner_name(comp.get('higher_rated', 'tie'))}.")
        if comp.get("more_popular") != "tie":
            print(f"- Audience scale edge goes to {_winner_name(comp.get('more_popular', 'tie'))}.")
        if comp.get("newer") != "tie":
            print(f"- Recency edge goes to {_winner_name(comp.get('newer', 'tie'))}.")


def rank_top_movies_shows_by_genre_or_year(
    movies: List[Any],
    genre: Optional[str] = None,
    year: Optional[int] = None,
    content_mode: str = "both",
    top_k: int = 10,
) -> Dict[str, Any]:
    """Rank top movies/shows within a genre or year using Rating > Popularity."""
    mode = str(content_mode or "both").strip().lower()
    if mode == "movie":
        scoped = [m for m in movies if _is_movie_type(m)]
    elif mode == "show":
        scoped = [m for m in movies if _is_show_type(m)]
    else:
        mode = "both"
        scoped = [m for m in movies if _is_allowed_output_type(m)]

    normalized_genre = str(genre or "").strip().lower()
    if normalized_genre:
        scoped = [
            m for m in scoped
            if normalized_genre in {g.lower() for g in _parse_genre_tokens(getattr(m, "Genre", ""))}
        ]

    if year is not None:
        scoped = [m for m in scoped if _safe_int(getattr(m, "Year", 0), 0) == year]

    ranked = sorted(
        scoped,
        key=lambda m: (
            _safe_float(getattr(m, "Rating", 0.0), 0.0),
            _safe_int(getattr(m, "Popularity", 0), 0),
        ),
        reverse=True,
    )

    rows: List[Dict[str, Any]] = []
    for movie in ranked[: max(1, top_k)]:
        rows.append(
            {
                "imdb_id": getattr(movie, "imdb_id", None),
                "title": getattr(movie, "Name", None),
                "content_type": getattr(movie, "Type", None),
                "genre": getattr(movie, "Genre", None),
                "rating": _safe_float(getattr(movie, "Rating", 0.0), 0.0),
                "popularity": _safe_int(getattr(movie, "Popularity", 0), 0),
                "year": _safe_int(getattr(movie, "Year", 0), 0),
            }
        )

    return {
        "criteria": {
            "genre": normalized_genre or None,
            "year": year,
            "content_mode": mode,
            "ranking_priority": "rating_then_popularity",
        },
        "candidate_count": len(scoped),
        "top_k": len(rows),
        "results": rows,
    }


def pretty_top_rankings_report(report: Dict[str, Any]) -> None:
    criteria = report.get("criteria", {})
    print("Top Movies/Shows Ranking")
    print(
        "Filter: "
        f"genre={criteria.get('genre') or 'any'}, "
        f"year={criteria.get('year') if criteria.get('year') is not None else 'any'}, "
        f"content={criteria.get('content_mode') or 'both'}"
    )
    print(f"Ranking priority: {criteria.get('ranking_priority', 'rating_then_popularity')}")
    print(f"Candidates found: {report.get('candidate_count', 0)}")

    rows = report.get("results", [])
    if not rows:
        print("No matching movies/shows found for the selected filter.")
        return

    print("Rank | Title | IMDb ID | Type | Year | Rating | Popularity | Genre")
    print("-" * 110)
    for idx, row in enumerate(rows, 1):
        print(
            f"{idx:<4} | "
            f"{str(row.get('title') or ''):<30} | "
            f"{str(row.get('imdb_id') or ''):<10} | "
            f"{str(row.get('content_type') or ''):<10} | "
            f"{_safe_int(row.get('year'), 0):<4} | "
            f"{_safe_float(row.get('rating'), 0.0):<6.1f} | "
            f"{_safe_int(row.get('popularity'), 0):<10} | "
            f"{str(row.get('genre') or '')}"
        )


def rank_user_selected_set(
    movies: List[Any],
    references: List[str],
) -> Dict[str, Any]:
    """Resolve and rank a user-provided set by Rating > Popularity > Length."""
    resolved: List[Any] = []
    unresolved: List[str] = []
    seen_ids = set()

    for raw_ref in references:
        ref = str(raw_ref or "").strip()
        if not ref:
            continue
        movie = resolve_reference_movie(ref, movies)
        if movie is None:
            unresolved.append(ref)
            continue
        imdb_id = str(getattr(movie, "imdb_id", "") or "")
        if imdb_id and imdb_id in seen_ids:
            continue
        if imdb_id:
            seen_ids.add(imdb_id)
        resolved.append(movie)

    ranked = sorted(
        resolved,
        key=lambda m: (
            _safe_float(getattr(m, "Rating", 0.0), 0.0),
            _safe_int(getattr(m, "Popularity", 0), 0),
            _safe_runtime_length(m),
        ),
        reverse=True,
    )

    rows: List[Dict[str, Any]] = []
    for movie in ranked:
        rows.append(
            {
                "imdb_id": getattr(movie, "imdb_id", None),
                "title": getattr(movie, "Name", None),
                "content_type": getattr(movie, "Type", None),
                "year": _safe_int(getattr(movie, "Year", 0), 0),
                "genre": getattr(movie, "Genre", None),
                "rating": _safe_float(getattr(movie, "Rating", 0.0), 0.0),
                "popularity": _safe_int(getattr(movie, "Popularity", 0), 0),
                "length": _safe_runtime_length(movie),
            }
        )

    return {
        "input_size": len([r for r in references if str(r).strip()]),
        "resolved_count": len(rows),
        "unresolved": unresolved,
        "ranking_priority": "rating_then_popularity_then_length",
        "results": rows,
    }


def pretty_user_set_ranking_report(report: Dict[str, Any]) -> None:
    print("User Set Ranking")
    print(f"Ranking priority: {report.get('ranking_priority', 'rating_then_popularity_then_length')}")
    print(f"Input items: {report.get('input_size', 0)} | Resolved: {report.get('resolved_count', 0)}")

    unresolved = report.get("unresolved", [])
    if unresolved:
        print("Unresolved inputs: " + ", ".join(unresolved))

    rows = report.get("results", [])
    if not rows:
        print("No valid movies/shows were resolved from the provided set.")
        return

    print("Rank | Title | IMDb ID | Type | Year | Rating | Popularity | Length(min) | Genre")
    print("-" * 125)
    for idx, row in enumerate(rows, 1):
        print(
            f"{idx:<4} | "
            f"{str(row.get('title') or ''):<30} | "
            f"{str(row.get('imdb_id') or ''):<10} | "
            f"{str(row.get('content_type') or ''):<10} | "
            f"{_safe_int(row.get('year'), 0):<4} | "
            f"{_safe_float(row.get('rating'), 0.0):<6.1f} | "
            f"{_safe_int(row.get('popularity'), 0):<10} | "
            f"{_safe_int(row.get('length'), 0):<11} | "
            f"{str(row.get('genre') or '')}"
        )


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
    print("Choose an action:")
    print("1) Find 10 similar movies/shows")
    print("2) Compare with another movie/show")
    print("3) Rank top movies/shows by genre or year (Rating > Popularity)")
    print("4) Rank a user-provided set (Rating > Popularity > Length)")
    try:
        action = input("> ").strip().lower()
    except EOFError:
        action = ""

    if action in {"4", "set", "list", "custom"}:
        print("What set of moive/show would you like to rank?")
        print("Enter movie titles or IMDb IDs separated by commas.")
        try:
            set_input = input("> ").strip()
        except EOFError:
            set_input = ""

        # Internal working array to store the provided set.
        reference_array = [part.strip() for part in set_input.split(",") if part.strip()]
        if not reference_array:
            print("You must provide at least one movie/show title or IMDb ID.")
            return

        report = rank_user_selected_set(movies=movies, references=reference_array)
        pretty_user_set_ranking_report(report)
        return

    if action in {"3", "top", "rank", "genre", "year"}:
        print("Choose content type:")
        print("1) Movie only")
        print("2) Show only")
        print("3) Both movies and shows")
        try:
            content_choice = input("> ").strip().lower()
        except EOFError:
            content_choice = ""

        if content_choice in {"1", "movie", "m"}:
            content_mode = "movie"
        elif content_choice in {"2", "show", "s", "tv"}:
            content_mode = "show"
        else:
            content_mode = "both"

        print("Choose ranking filter:")
        print("1) Genre")
        print("2) Year")
        try:
            filter_choice = input("> ").strip().lower()
        except EOFError:
            filter_choice = ""

        top_k = DEFAULT_TOP_K
        print(f"How many results? Press Enter to use default: {DEFAULT_TOP_K}")
        try:
            top_k_input = input("> ").strip()
        except EOFError:
            top_k_input = ""
        if top_k_input:
            parsed_top_k = _safe_int(top_k_input, DEFAULT_TOP_K)
            top_k = parsed_top_k if parsed_top_k > 0 else DEFAULT_TOP_K

        if filter_choice in {"1", "genre", "g"}:
            print("Enter a genre (example: Drama, Action, Sci-Fi):")
            try:
                genre_input = input("> ").strip()
            except EOFError:
                genre_input = ""

            if not genre_input:
                print("Genre is required for genre ranking.")
                return

            report = rank_top_movies_shows_by_genre_or_year(
                movies=movies,
                genre=genre_input,
                year=None,
                content_mode=content_mode,
                top_k=top_k,
            )
            pretty_top_rankings_report(report)
            return

        print("Enter a year (example: 2014):")
        try:
            year_input = input("> ").strip()
        except EOFError:
            year_input = ""

        if not year_input:
            print("Year is required for year ranking.")
            return

        year_value = _safe_int(year_input, 0)
        if year_value <= 0:
            print("Invalid year. Please enter a valid numeric year.")
            return

        report = rank_top_movies_shows_by_genre_or_year(
            movies=movies,
            genre=None,
            year=year_value,
            content_mode=content_mode,
            top_k=top_k,
        )
        pretty_top_rankings_report(report)
        return

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

    if action in {"2", "compare", "comparison", "c"}:
        print("Enter the second movie title or IMDb ID to compare with.")
        try:
            second_reference = input("> ").strip()
        except EOFError:
            second_reference = ""

        if not second_reference:
            print("Second movie is required for comparison.")
            return

        comparison = compare_two_movies(reference_movie, second_reference, movies)
        pretty_comparison_report(comparison)
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
