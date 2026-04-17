import re
import os
from functools import lru_cache
from typing import Any, Dict, List, Optional

from agents import Agent, Runner, function_tool, set_default_openai_key
from pydantic import BaseModel, ConfigDict

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


@function_tool
def resolve_movie_reference_tool(reference: str) -> Dict[str, Any]:
    movies = load_movie_universe()
    movie = resolve_reference_movie(reference, movies)
    if movie is None:
        return {"found": False, "reference": reference}
    return {
        "found": True,
        "reference": reference,
        "imdb_id": getattr(movie, "imdb_id", None),
        "title": getattr(movie, "Name", None),
        "content_type": getattr(movie, "Type", None),
        "year": safe_int(getattr(movie, "Year", 0), 0),
    }


class SharedLookupOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    found: bool
    reference: str
    imdb_id: Optional[str] = None
    title: Optional[str] = None
    content_type: Optional[str] = None
    year: Optional[int] = None


def build_shared_agent(model: str = "gpt-5-nano") -> Agent:
    instruction = """
You are a deterministic movie reference lookup agent.

Tool usage rules:
- ALWAYS call resolve_movie_reference_tool(reference) exactly once.
- Do not use external data.

Output rules:
- Return the exact structured lookup result.
"""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("Please set OPENAI_API_KEY in your environment.")
    set_default_openai_key(api_key)

    return Agent(
        name="Shared movie lookup agent",
        instructions=instruction,
        tools=[resolve_movie_reference_tool],
        model=model,
        output_type=SharedLookupOutput,
    )


async def run_shared_lookup_with_agent(reference: str, model: str = "gpt-5-nano") -> Dict[str, Any]:
    agent = build_shared_agent(model=model)
    result = await Runner.run(agent, input=f"Resolve this movie reference: {reference}")
    return result.final_output.model_dump()
