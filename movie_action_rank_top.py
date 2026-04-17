from typing import Any, Dict, List, Optional

from movie_agent_shared import (
    is_allowed_output_type,
    is_movie_type,
    is_show_type,
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
