from typing import Any, Dict, List

from movie_agent_shared import (
    movie_to_dict,
    parse_genre_tokens,
    resolve_reference_movie,
    safe_float,
    safe_int,
    truncate_text,
)


def _build_movie_short_summary(movie: Dict[str, Any]) -> str:
    title = str(movie.get("Name") or "This title")
    content_type = str(movie.get("Type") or "title")
    year = safe_int(movie.get("Year"), 0)
    rating = safe_float(movie.get("Rating"), 0.0)
    popularity = safe_int(movie.get("Popularity"), 0)
    genres = parse_genre_tokens(movie.get("Genre"))

    genre_phrase = " and ".join(genres[:2]) if genres else "mixed genres"
    year_phrase = str(year) if year > 0 else "an unknown year"
    popularity_phrase = f"{popularity:,} votes" if popularity > 0 else "limited vote data"

    return (
        f"{title} is a {content_type} from {year_phrase} centered on {genre_phrase} themes, "
        f"with an IMDb rating of {rating:.1f} and {popularity_phrase}."
    )


def compare_two_movies(first_reference: str, second_reference: str, movies: List[Any]) -> Dict[str, Any]:
    first_movie = resolve_reference_movie(first_reference, movies)
    second_movie = resolve_reference_movie(second_reference, movies)

    if first_movie is None or second_movie is None:
        return {
            "error": "One or both references could not be resolved.",
            "first_found": first_movie is not None,
            "second_found": second_movie is not None,
        }

    first = movie_to_dict(first_movie)
    second = movie_to_dict(second_movie)

    first_genres = set(parse_genre_tokens(first.get("Genre")))
    second_genres = set(parse_genre_tokens(second.get("Genre")))
    shared_genres = sorted(first_genres & second_genres)

    first_rating = safe_float(first.get("Rating"), 0.0)
    second_rating = safe_float(second.get("Rating"), 0.0)
    first_popularity = safe_int(first.get("Popularity"), 0)
    second_popularity = safe_int(second.get("Popularity"), 0)
    first_year = safe_int(first.get("Year"), 0)
    second_year = safe_int(second.get("Year"), 0)

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
    print(_row("Rating", f"{safe_float(first.get('rating')):.1f}", f"{safe_float(second.get('rating')):.1f}"))
    print(_row("Popularity", first.get("popularity"), second.get("popularity")))
    print(_row("Year", first.get("year"), second.get("year")))
    print(_row("Description", truncate_text(first.get("description")), truncate_text(second.get("description"))))
    print()
    shared = ", ".join(comp.get("shared_genres", [])) if comp.get("shared_genres") else "None"
    print(f"Shared genres: {shared}")

    rating_diff = safe_float(comp.get("rating_diff"), 0.0)
    popularity_diff = safe_int(comp.get("popularity_diff"), 0)
    year_diff = safe_int(comp.get("year_diff"), 0)

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
