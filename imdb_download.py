from imdb import IMDb

def fetch_movie_data(title):
    ia = IMDb()
    try:
        movies = ia.search_movie(title)
        if not movies:
            print(f"No results found for: {title}")
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
    except Exception as e:
        print(f"Error fetching data for {title}: {e}")
        return None

def main():
    titles = ['The Matrix', 'Breaking Bad', 'Inception', 'Game of Thrones']
    print("Fetching movie data...")
    results = []
    for title in titles:
        info = fetch_movie_data(title)
        if info and info['Popularity']:
            try:
                votes = int(info['Popularity'])
            except Exception:
                votes = 0
            if votes > 500000:
                print(f"\nTitle: {info['Name']}")
                print(f"Description: {info['Description']}")
                print(f"Rating: {info['Rating']}")
                print(f"Popularity (votes): {info['Popularity']}")
                print(f"Genre: {info['Genre']}")
                results.append(info)

if __name__ == '__main__':
    main()