from imdb import IMDb

def fetch_movie_data(title):
    ia = IMDb()
    movies = ia.search_movie(title)
    if not movies:
        return None
    movie = ia.get_movie(movies[0].movieID)
    data = {
        'Name': movie.get('title'),
        'Description': movie.get('plot outline'),
        'Rating': movie.get('rating'),
        'Popularity': movie.get('votes'),
        'Genre': movie.get('genres'),
    }
    return data

def main():
    titles = ['The Matrix', 'Breaking Bad', 'Inception', 'Game of Thrones']
    results = []
    for title in titles:
        info = fetch_movie_data(title)
        if info:
            results.append(info)
    print("Results: ", results)
    for r in results:
        print(r)

print("Fetching movie data...")
main()