from typing import Any, Dict, List

from movie_agent_shared import resolve_reference_movie, safe_float, safe_int, safe_runtime_length


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
