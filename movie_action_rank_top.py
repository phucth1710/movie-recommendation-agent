import os
from typing import Any, Dict, List, Optional

from agents import Agent, Runner, function_tool, set_default_openai_key
from pydantic import BaseModel, ConfigDict

from movie_agent_shared import (
    is_allowed_output_type,
    is_movie_type,
    is_show_type,
    load_movie_universe,
    parse_genre_tokens,
    safe_float,
    safe_int,
)


def rank_top_movies_shows_by_genre_or_year(
    movies: List[Any],
    genre: Optional[str] = None,
    year: Optional[int] = None,
    content_mode: str = "both",
    top_k: int = 10,
) -> Dict[str, Any]:
    mode = str(content_mode or "both").strip().lower()
    if mode == "movie":
        scoped = [m for m in movies if is_movie_type(m)]
    elif mode == "show":
        scoped = [m for m in movies if is_show_type(m)]
    else:
        mode = "both"
        scoped = [m for m in movies if is_allowed_output_type(m)]

    normalized_genre = str(genre or "").strip().lower()
    if normalized_genre:
        scoped = [
            m for m in scoped
            if normalized_genre in {g.lower() for g in parse_genre_tokens(getattr(m, "Genre", ""))}
        ]

    if year is not None:
        scoped = [m for m in scoped if safe_int(getattr(m, "Year", 0), 0) == year]

    ranked = sorted(
        scoped,
        key=lambda m: (
            safe_float(getattr(m, "Rating", 0.0), 0.0),
            safe_int(getattr(m, "Popularity", 0), 0),
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
                "rating": safe_float(getattr(movie, "Rating", 0.0), 0.0),
                "popularity": safe_int(getattr(movie, "Popularity", 0), 0),
                "year": safe_int(getattr(movie, "Year", 0), 0),
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
            f"{safe_int(row.get('year'), 0):<4} | "
            f"{safe_float(row.get('rating'), 0.0):<6.1f} | "
            f"{safe_int(row.get('popularity'), 0):<10} | "
            f"{str(row.get('genre') or '')}"
        )


@function_tool
def rank_top_by_genre_or_year_tool(
    genre: str = "",
    year: int = 0,
    content_mode: str = "both",
    top_k: int = 10,
) -> Dict[str, Any]:
    movies = load_movie_universe()
    return rank_top_movies_shows_by_genre_or_year(
        movies=movies,
        genre=genre or None,
        year=(year if year > 0 else None),
        content_mode=content_mode,
        top_k=top_k,
    )


class RankedTopItem(BaseModel):
    model_config = ConfigDict(extra="forbid")
    imdb_id: str = ""
    title: str = ""
    content_type: str = ""
    genre: str = ""
    rating: float = 0.0
    popularity: int = 0
    year: int = 0


class RankedTopCriteria(BaseModel):
    model_config = ConfigDict(extra="forbid")
    genre: Optional[str] = None
    year: Optional[int] = None
    content_mode: str
    ranking_priority: str


class RankedTopOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    criteria: RankedTopCriteria
    candidate_count: int
    top_k: int
    results: List[RankedTopItem]


def build_rank_top_agent(model: str = "gpt-5-nano") -> Agent:
    instruction = """
You are a deterministic ranking agent for top movies/shows by genre or year.

Tool usage rules:
- ALWAYS call rank_top_by_genre_or_year_tool(genre, year, content_mode, top_k) exactly once.
- Do not use external data.

Output rules:
- Return the exact structured ranking result from the tool.
"""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("Please set OPENAI_API_KEY in your environment.")
    set_default_openai_key(api_key)

    return Agent(
        name="Top ranking agent",
        instructions=instruction,
        tools=[rank_top_by_genre_or_year_tool],
        model=model,
        output_type=RankedTopOutput,
    )


async def run_rank_top_with_agent(
    genre: Optional[str],
    year: Optional[int],
    content_mode: str,
    top_k: int,
    model: str = "gpt-5-nano",
) -> Dict[str, Any]:
    agent = build_rank_top_agent(model=model)
    prompt = (
        "Rank top movies/shows with this filter: "
        f"genre={genre or 'none'}, year={year or 0}, content_mode={content_mode}, top_k={top_k}."
    )
    result = await Runner.run(agent, input=prompt)
    return result.final_output.model_dump()
