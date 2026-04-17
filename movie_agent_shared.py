import re
from functools import lru_cache
from typing import Any, Dict, List, Optional

from imdb_csv_utils import find_movie_by_id, find_movie_by_name, read_movies
from movie_agent_core import DEFAULT_SOURCE_IMDB_ID, DEFAULT_SOURCE_TITLE


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


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def safe_runtime_length(movie: Any) -> int:
    runtime = getattr(movie, "Runtime", None)
    if runtime in (None, ""):
        runtime = getattr(movie, "runtimeMinutes", None)
    return safe_int(runtime, 0)


def normalized_type(movie: Any) -> str:
    return str(getattr(movie, "Type", "") or "").strip().lower()


def is_show_type(movie: Any) -> bool:
    return normalized_type(movie) in ALLOWED_SHOW_TYPES


def is_movie_type(movie: Any) -> bool:
    return normalized_type(movie) in ALLOWED_MOVIE_TYPES


def is_allowed_output_type(movie: Any) -> bool:
    return normalized_type(movie) in ALLOWED_OUTPUT_TYPES


def movie_to_dict(movie: Any) -> Dict[str, Any]:
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


def parse_genre_tokens(raw_genre: Any) -> List[str]:
    if raw_genre is None:
        return []
    return [g.strip() for g in str(raw_genre).split(",") if g.strip()]


def truncate_text(value: Any, max_len: int = 120) -> str:
    text = str(value or "").strip()
    if len(text) <= max_len:
        return text
    return text[: max(0, max_len - 3)] + "..."


@lru_cache(maxsize=1)
def load_movie_universe() -> List[Any]:
    return read_movies()


def resolve_reference_movie(reference: str, movies: List[Any]) -> Optional[Any]:
    normalized = reference.strip()
    if not normalized:
        return None

    if re.fullmatch(r"tt\d+", normalized, flags=re.IGNORECASE):
        return find_movie_by_id(normalized, movies=movies)

    movie = find_movie_by_id(normalized, movies=movies)
    if movie is not None:
        return movie

    movie = find_movie_by_name(normalized, movies=movies)
    if movie is not None:
        return movie

    if len(normalized) < 3:
        return None

    needle = normalized.lower()
    partial_matches: List[Any] = []
    for candidate in movies:
        name = str(getattr(candidate, "Name", "")).lower().strip()
        if needle in name:
            partial_matches.append(candidate)

    if len(partial_matches) == 1:
        return partial_matches[0]

    return None
