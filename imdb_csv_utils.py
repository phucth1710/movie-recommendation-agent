
import pandas as pd
from typing import List, Optional


class Movie:
    def __init__(self, imdb_id: str, Name: str, Type: str, Genre: str, Rating: float, Popularity: int, Year: int, Description: str):
        self.imdb_id = imdb_id
        self.Name = Name
        self.Type = Type
        self.Genre = Genre
        self.Rating = float(Rating)
        self.Popularity = int(Popularity)
        self.Year = int(Year)
        self.Description = Description

    @classmethod
    def from_dict(cls, d):
        return cls(
            imdb_id=d.get('imdb_id'),
            Name=d.get('Name'),
            Type=d.get('Type'),
            Genre=d.get('Genre'),
            Rating=d.get('Rating'),
            Popularity=d.get('Popularity'),
            Year=d.get('Year'),
            Description=d.get('Description'),
        )

    def to_dict(self):
        return {
            'imdb_id': self.imdb_id,
            'Name': self.Name,
            'Type': self.Type,
            'Genre': self.Genre,
            'Rating': self.Rating,
            'Popularity': self.Popularity,
            'Year': self.Year,
            'Description': self.Description,
        }

    def __repr__(self):
        return f"Movie(imdb_id={self.imdb_id}, Name={self.Name}, Rating={self.Rating}, Year={self.Year})"

CSV_PATH = 'imdb_filtered.csv'

def read_movies() -> List[Movie]:
    """Read all movies from the CSV file and return as a list of Movie objects."""
    df = pd.read_csv(CSV_PATH)
    return [Movie.from_dict(row) for row in df.to_dict(orient='records')]

def find_movie_by_id(imdb_id: str, movies: List[Movie] = None) -> Optional[Movie]:
    """Find a movie by IMDb id (imdb_id column). Optionally provide a list of Movie objects to filter from."""
    if movies is None:
        movies = read_movies()
    for movie in movies:
        if movie.imdb_id == imdb_id:
            return movie
    return None

def find_movie_by_name(name: str, movies: List[Movie] = None) -> Optional[Movie]:
    """Find a movie by name (Name column, case-insensitive). Optionally provide a list of Movie objects to filter from."""
    if movies is None:
        movies = read_movies()
    name_lower = name.lower()
    for movie in movies:
        if movie.Name.lower() == name_lower:
            return movie
    return None

def filter_movies(movies: List[Movie] = None, rating_count_min: int = None, rating_score_min: float = None, language: str = None, genre: str = None) -> List[Movie]:
    """Filter movies by rating count, rating score, language, genre, etc. Optionally provide a list of Movie objects to filter from as the first argument."""
    if movies is None:
        movies = read_movies()
    filtered = []
    for movie in movies:
        if rating_count_min is not None and movie.Popularity < rating_count_min:
            continue
        if rating_score_min is not None and movie.Rating < rating_score_min:
            continue
        if language is not None and hasattr(movie, 'Language') and getattr(movie, 'Language', '').lower() != language.lower():
            continue
        if genre is not None and genre.lower() not in movie.Genre.lower():
            continue
        filtered.append(movie)
    return filtered

def sort_movies(movies: List[Movie] = None, by: str = 'Rating', ascending: bool = False, limit: int = None) -> List[Movie]:
    """Sort movies by a given attribute (default: Rating). Optionally limit the number of results. Optionally provide a list of Movie objects to sort as the first argument."""
    if movies is None:
        movies = read_movies()
    # Check attribute on an instance if possible
    if movies and not hasattr(movies[0], by):
        raise ValueError(f"Attribute '{by}' not found in Movie class.")
    sorted_movies = sorted(movies, key=lambda m: getattr(m, by), reverse=not ascending)
    if limit is not None:
        sorted_movies = sorted_movies[:limit]
    return sorted_movies


def main():
    print("All movies (first 3):")
    all_movies = read_movies()
    print(all_movies[:3])

    print("\nFind movie by IMDb id (first id):")
    if all_movies:
        first_id = all_movies[0].imdb_id
        print(find_movie_by_id(first_id, all_movies))

    print("\nFind movie by name (first name):")
    if all_movies:
        first_name = all_movies[0].Name
        print(find_movie_by_name(first_name, all_movies))

    print("\nFilter movies (Rating >= 8, Popularity >= 100000):")
    filtered = filter_movies(all_movies, rating_count_min=100000, rating_score_min=8)
    print(filtered[:3])

    print("\nSort movies by Rating (top 3):")
    sorted_movies = sort_movies(all_movies, by='Rating', ascending=False, limit=3)
    print(sorted_movies)


if __name__ == "__main__":
    main()
