import asyncio

from agents import Runner

from movie_agent_shared import (
    DEFAULT_SCOPE_SIZE,
    DEFAULT_TOP_K,
    DEFAULT_WEIGHTS,
    load_movie_universe,
    resolve_reference_movie,
    safe_int,
)
from movie_action_compare import compare_two_movies, pretty_comparison_report
from movie_action_rank_set import rank_user_selected_set, pretty_user_set_ranking_report
from movie_action_rank_top import rank_top_movies_shows_by_genre_or_year, pretty_top_rankings_report
from movie_action_similar import (
    build_agent,
    movie_universe_summary,
    pretty_report,
    rank_top_from_scoped_pool,
    recommend_movies_from_reference,
    scope_candidates,
)
from movie_agent_core import DEFAULT_SOURCE_IMDB_ID, DEFAULT_SOURCE_TITLE


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
            parsed_top_k = safe_int(top_k_input, DEFAULT_TOP_K)
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

        year_value = safe_int(year_input, 0)
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
