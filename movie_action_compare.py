import os
from typing import Any, Dict, List

from agents import Agent, Runner, function_tool, set_default_openai_key
from pydantic import BaseModel, ConfigDict

from movie_agent_shared import (
    load_movie_universe,
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


@function_tool
def compare_two_movies_from_references(first_reference: str, second_reference: str) -> Dict[str, Any]:
    movies = load_movie_universe()
    return compare_two_movies(first_reference, second_reference, movies)


class ComparedMovie(BaseModel):
    model_config = ConfigDict(extra="forbid")
    imdb_id: str = ""
    title: str = ""
    content_type: str = ""
    genre: str = ""
    rating: float = 0.0
    popularity: int = 0
    year: int = 0
    description: str = ""


class ComparisonCore(BaseModel):
    model_config = ConfigDict(extra="forbid")
    shared_genres: List[str]
    rating_diff: float
    popularity_diff: int
    year_diff: int
    higher_rated: str
    more_popular: str
    newer: str


class ComparisonOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    first_movie: ComparedMovie
    second_movie: ComparedMovie
    comparison: ComparisonCore


class AIComparisonInsight(BaseModel):
    model_config = ConfigDict(extra="forbid")
    overall_impression: str
    genre_and_tone: str
    main_themes: str
    critical_reception: str
    taste_recommendation: str


def build_compare_agent(model: str = "gpt-5.5") -> Agent:
    instruction = """
You are a deterministic movie comparison agent.

Tool usage rules:
- ALWAYS call compare_two_movies_from_references(first_reference, second_reference) exactly once.
- Do not use any external data source.

Output rules:
- Return the structured comparison object from the tool without inventing data.
- Keep all numeric values exact.
"""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("Please set OPENAI_API_KEY in your environment.")
    set_default_openai_key(api_key)

    return Agent(
        name="Movie comparison agent",
        instructions=instruction,
        tools=[compare_two_movies_from_references],
        model=model,
        output_type=ComparisonOutput,
    )


async def run_compare_with_agent(
    first_reference: str,
    second_reference: str,
    model: str = "gpt-5.5",
) -> Dict[str, Any]:
    agent = build_compare_agent(model=model)
    prompt = (
        f"Compare these two references: first='{first_reference}', second='{second_reference}'. "
        "Return the structured comparison output."
    )
    result = await Runner.run(agent, input=prompt)
    return result.final_output.model_dump()


def build_compare_insight_agent(model: str = "gpt-5.5") -> Agent:
    instruction = """
You are an experienced film and television critic writing a comparison essay
for a general audience that already knows both titles.

You will receive structured metadata (title, year, genres, IMDb rating,
popularity, etc.) for two real, well-known works. Treat the metadata as
IDENTIFIERS for looking each title up in your own knowledge — NOT as the
substance of your analysis. You SHOULD use widely known public knowledge of
each title (characters, premise, story arc, tone, cultural footprint,
awards/sales/legacy, audience response) to write a rich, content-driven
comparison. NEVER write phrases like "the data only says…", "based on the
genre data…", "the provided dataset indicates…", "limited information
available", or any similar hedging about the metadata. Do not refer to
"the data", "the profile", "the dataset", or "the metadata" anywhere.

Output fields (return all five, in this order):
- overall_impression
- genre_and_tone
- main_themes
- critical_reception
- taste_recommendation

FORMAT — every field MUST be valid Markdown.
- Use **bold** for emphasis on titles, themes, and key claims.
- Use bullet lists ("- ") where it genuinely helps clarity (especially in
  main_themes), but prose paragraphs are also fine.
- Do NOT include the field name itself as a heading inside the field's value
  (the UI prepends the section title for you).
- Do NOT use H1/H2 hash headings ("#", "##"); use bold inline emphasis only.

Field-specific requirements:

overall_impression — 3 to 6 sentences of flowing prose that opens the essay.
Establish the cultural relationship between the two titles before any
section-by-section breakdown: how people typically discuss them together,
what they obviously share, and what makes the pairing interesting or
unexpected. Name both titles explicitly. End with a hook that leads naturally
into the deeper analysis below. Do NOT split into bullet points here.

genre_and_tone — A meaty paragraph (4-6 sentences) of CONTENT-DRIVEN
analysis. Go beyond the genre labels: discuss how each work actually feels
on screen, the texture of its action, the moral atmosphere, the use of
violence, pacing, scope (intimate vs operatic), and how each title handles
escalation. Compare and contrast specific tonal traits — for example,
political cynicism, body horror, mythic scale, courtly intrigue,
existential dread — that are concretely true of each work.

main_themes — Use Markdown structure here. Write a 1-2 sentence lead-in,
then provide a clearly formatted breakdown of the actual thematic concerns
of EACH title (e.g., power and legitimacy, freedom vs determinism, cycles
of revenge, the cost of war, family, identity). A good shape is:
  **Shared themes**
  - bullet points of themes both works genuinely explore
  **<First title>**
  - bullet points specific to the first title
  **<Second title>**
  - bullet points specific to the second title
Each bullet should be a concrete theme tied to specific story elements,
not a generic genre cliché.

critical_reception — Do NOT rely on IMDb numbers alone. Treat IMDb rating
and vote count as ONE signal among several. You MUST also discuss, drawing
on widely known public knowledge: box-office or streaming/viewership
performance where relevant, awards (Emmy, Oscar, Crunchyroll/anime awards,
Hugo, etc., as appropriate), critical consensus from major outlets,
fandom size and longevity, merchandising/franchise footprint, and lasting
cultural legacy or influence on the medium. Compare the two works on these
dimensions, not just on the rating gap. 5-8 sentences.

taste_recommendation — This is the conclusion of the essay. Synthesize
everything above (overall impression + tone + themes + reception) into a
practical recommendation. Use this shape:
  **Choose <First title> if…** followed by 2-4 sentences explaining the
  kind of viewer, mood, or appetite it best satisfies, referencing the
  specific tonal/thematic traits you identified earlier.
  **Choose <Second title> if…** followed by 2-4 sentences doing the same.
Optionally close with one short sentence on watching both. Do NOT just
re-quote IMDb numbers; lean on the qualitative analysis.

Style:
- Confident, specific, and concrete. No filler, no hedging about data.
- Refer to the works by their real names. Use widely known public knowledge.
- Keep numeric facts (rating, vote count, year) accurate to the input.
"""

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("Please set OPENAI_API_KEY in your environment.")
    set_default_openai_key(api_key)

    return Agent(
        name="Movie comparison insight agent",
        instructions=instruction,
        tools=[],
        model=model,
        output_type=AIComparisonInsight,
    )


async def run_compare_insight_with_agent(
    first_reference: str,
    second_reference: str,
    model: str = "gpt-5.5",
) -> Dict[str, Any]:
    movies = load_movie_universe()
    base = compare_two_movies(first_reference, second_reference, movies)
    if base.get("error"):
        return {"error": base.get("error", "Unable to generate AI insight.")}

    first_movie = base.get("first_movie", {})
    second_movie = base.get("second_movie", {})
    comp = base.get("comparison", {})

    prompt = (
        "Write a critic's comparison essay for the two titles below. The "
        "metadata is provided ONLY to identify the works — treat them as real, "
        "well-known titles and write content-driven analysis from your own "
        "knowledge of them. Do NOT hedge about missing data, do NOT refer to "
        "'the data' or 'the profile', and do NOT lean on the IMDb numbers as "
        "your only evidence.\n\n"
        "Return all five Markdown fields in this order:\n"
        "  1. overall_impression — 3-6 sentences of opening prose that "
        "frames the cultural relationship between the two titles.\n"
        "  2. genre_and_tone — a meaty paragraph (4-6 sentences) of concrete "
        "tonal analysis (atmosphere, pacing, scope, moral texture, how each "
        "work actually feels), going well beyond the genre labels.\n"
        "  3. main_themes — Markdown with **Shared themes**, **<First "
        "title>**, and **<Second title>** subheadings, each followed by "
        "bullet points of specific thematic concerns tied to real story "
        "elements.\n"
        "  4. critical_reception — 5-8 sentences. Use IMDb rating and votes "
        "as ONE signal, then bring in box office / viewership, awards, "
        "critical consensus, fandom and longevity, franchise footprint, and "
        "cultural legacy. Compare the two works on these dimensions.\n"
        "  5. taste_recommendation — the essay's conclusion. Use the shape "
        "**Choose <First title> if…** (2-4 sentences) and **Choose <Second "
        "title> if…** (2-4 sentences), grounded in the tonal/thematic "
        "analysis above, not just the ratings.\n\n"
        f"First title: {first_movie}\n"
        f"Second title: {second_movie}\n"
        f"Comparison metrics: {comp}\n"
    )

    agent = build_compare_insight_agent(model=model)
    result = await Runner.run(agent, input=prompt)
    return result.final_output.model_dump()
