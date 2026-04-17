import os
from typing import Any, Dict, List

from agents import Agent, Runner, function_tool, set_default_openai_key
from pydantic import BaseModel, ConfigDict

from movie_agent_shared import load_movie_universe, resolve_reference_movie, safe_float, safe_int, safe_runtime_length


def rank_user_selected_set(
    movies: List[Any],
    references: List[str],
) -> Dict[str, Any]:
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
            safe_float(getattr(m, "Rating", 0.0), 0.0),
            safe_int(getattr(m, "Popularity", 0), 0),
            safe_runtime_length(m),
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
                "year": safe_int(getattr(movie, "Year", 0), 0),
                "genre": getattr(movie, "Genre", None),
                "rating": safe_float(getattr(movie, "Rating", 0.0), 0.0),
                "popularity": safe_int(getattr(movie, "Popularity", 0), 0),
                "length": safe_runtime_length(movie),
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
            f"{safe_int(row.get('year'), 0):<4} | "
            f"{safe_float(row.get('rating'), 0.0):<6.1f} | "
            f"{safe_int(row.get('popularity'), 0):<10} | "
            f"{safe_int(row.get('length'), 0):<11} | "
            f"{str(row.get('genre') or '')}"
        )


@function_tool
def rank_user_selected_set_tool(references_csv: str) -> Dict[str, Any]:
    movies = load_movie_universe()
    references = [part.strip() for part in str(references_csv or "").split(",") if part.strip()]
    return rank_user_selected_set(movies=movies, references=references)


class RankedSetItem(BaseModel):
    model_config = ConfigDict(extra="forbid")
    imdb_id: str = ""
    title: str = ""
    content_type: str = ""
    year: int = 0
    genre: str = ""
    rating: float = 0.0
    popularity: int = 0
    length: int = 0


class RankedSetOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    input_size: int
    resolved_count: int
    unresolved: List[str]
    ranking_priority: str
    results: List[RankedSetItem]


def build_rank_set_agent(model: str = "gpt-5-nano") -> Agent:
    instruction = """
You are a deterministic ranking agent for a user-provided set.

Tool usage rules:
- ALWAYS call rank_user_selected_set_tool(references_csv) exactly once.
- Do not use external data.

Output rules:
- Return the exact structured ranking from the tool.
"""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("Please set OPENAI_API_KEY in your environment.")
    set_default_openai_key(api_key)

    return Agent(
        name="Set ranking agent",
        instructions=instruction,
        tools=[rank_user_selected_set_tool],
        model=model,
        output_type=RankedSetOutput,
    )


async def run_rank_set_with_agent(references: List[str], model: str = "gpt-5-nano") -> Dict[str, Any]:
    references_csv = ", ".join([r for r in references if str(r).strip()])
    agent = build_rank_set_agent(model=model)
    prompt = (
        "Rank this user set by rating then popularity then length. "
        f"references_csv={references_csv}"
    )
    result = await Runner.run(agent, input=prompt)
    return result.final_output.model_dump()
