import threading
import time
import webbrowser
import asyncio
from typing import Any, Dict, List

from flask import Flask, jsonify, render_template_string, request

from movie_openai_agent import (
  DEFAULT_SCOPE_SIZE,
  DEFAULT_TOP_K,
  build_agent,
  load_movie_universe,
  resolve_reference_movie,
)
from agents import Runner


HOST = "127.0.0.1"
PORT = 5001
MAX_SUGGESTIONS = 8

app = Flask(__name__)

HTML_PAGE = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Movie Recommendation Search</title>
  <style>
    :root {
      --bg-top: #eef5ff;
      --bg-bottom: #f7fafc;
      --ink: #0f172a;
      --muted: #475569;
      --line: #d7deea;
      --blue: #0b57d0;
      --blue-hover: #0a4ab2;
      --card: #ffffff;
      --row-alt: #f8fbff;
    }

    * { box-sizing: border-box; }

    body {
      margin: 0;
      min-height: 100vh;
      font-family: "Segoe UI", "Helvetica Neue", sans-serif;
      color: var(--ink);
      background: radial-gradient(circle at 25% 20%, #dbeafe 0%, transparent 35%),
                  radial-gradient(circle at 80% 10%, #e0f2fe 0%, transparent 38%),
                  linear-gradient(180deg, var(--bg-top), var(--bg-bottom));
    }

    .wrap {
      max-width: 980px;
      margin: 0 auto;
      padding: 52px 18px 40px;
    }

    .hero {
      text-align: center;
      margin-bottom: 24px;
    }

    .brand {
      margin: 0 0 8px;
      font-weight: 700;
      font-size: clamp(28px, 4vw, 42px);
      letter-spacing: 0.2px;
      color: #083684;
    }

    .subtitle {
      margin: 0;
      color: var(--muted);
      font-size: 15px;
    }

    .search-card {
      background: var(--card);
      border: 1px solid var(--line);
      border-radius: 22px;
      padding: 16px;
      box-shadow: 0 18px 38px rgba(15, 23, 42, 0.08);
      position: relative;
    }

    .search-row {
      display: grid;
      grid-template-columns: 1fr auto;
      gap: 10px;
      align-items: center;
    }

    .search-input {
      width: 100%;
      border: 1px solid #c4d0e5;
      border-radius: 999px;
      padding: 14px 18px;
      font-size: 17px;
      outline: none;
      transition: border-color 0.15s ease, box-shadow 0.15s ease;
    }

    .search-input:focus {
      border-color: var(--blue);
      box-shadow: 0 0 0 4px rgba(11, 87, 208, 0.15);
    }

    .search-btn {
      border: 0;
      border-radius: 999px;
      background: var(--blue);
      color: #fff;
      padding: 13px 22px;
      font-size: 15px;
      font-weight: 600;
      cursor: pointer;
      transition: background 0.15s ease;
    }

    .search-btn:hover { background: var(--blue-hover); }

    .search-btn:disabled {
      background: #7aa7ea;
      cursor: not-allowed;
    }

    .loading {
      margin-top: 12px;
      display: none;
      align-items: center;
      gap: 10px;
      color: #0f2855;
      font-size: 14px;
    }

    .spinner {
      width: 18px;
      height: 18px;
      border: 2px solid #cfe0fb;
      border-top-color: var(--blue);
      border-radius: 50%;
      animation: spin 0.8s linear infinite;
    }

    @keyframes spin {
      to { transform: rotate(360deg); }
    }

    .dropdown {
      margin-top: 8px;
      border: 1px solid var(--line);
      border-radius: 14px;
      background: #fff;
      overflow: hidden;
      display: none;
      max-height: 240px;
      overflow-y: auto;
    }

    .suggestion-item {
      padding: 10px 14px;
      cursor: pointer;
      border-bottom: 1px solid #eef2f8;
      font-size: 14px;
    }

    .suggestion-item:last-child { border-bottom: none; }
    .suggestion-item:hover { background: #eff6ff; }

    .result-card {
      margin-top: 18px;
      background: #fff;
      border: 1px solid var(--line);
      border-radius: 18px;
      box-shadow: 0 8px 24px rgba(15, 23, 42, 0.06);
      overflow: hidden;
      display: none;
    }

    .meta {
      padding: 14px 16px;
      border-bottom: 1px solid #e7edf7;
      color: #0f2855;
      font-size: 14px;
      background: #f8fbff;
    }

    .error {
      margin-top: 12px;
      color: #9f1239;
      font-size: 14px;
      display: none;
    }

    .table-wrap {
      overflow-x: auto;
    }

    table {
      width: 100%;
      border-collapse: collapse;
      min-width: 900px;
      font-size: 14px;
    }

    th, td {
      text-align: left;
      padding: 10px 12px;
      border-bottom: 1px solid #edf2fa;
      vertical-align: top;
    }

    th {
      background: #eef4ff;
      color: #1e3a8a;
      font-weight: 700;
      position: sticky;
      top: 0;
      z-index: 1;
    }

    tbody tr:nth-child(even) { background: var(--row-alt); }
  </style>
</head>
<body>
  <div class="wrap">
    <div class="hero">
      <h1 class="brand">Movie Search</h1>
      <p class="subtitle">Search by title or IMDb ID, then get 10 similar results in a table.</p>
    </div>

    <div class="search-card">
      <div class="search-row">
        <input id="query" class="search-input" placeholder="Search movie title or IMDb ID (e.g. Game of Thrones or tt0944947)" autocomplete="off" />
        <button id="searchBtn" class="search-btn" type="button">Search</button>
      </div>
      <div id="suggestions" class="dropdown"></div>
      <div id="loading" class="loading" aria-live="polite">
        <span class="spinner" aria-hidden="true"></span>
        <span>Searching recommendations...</span>
      </div>
    </div>

    <div id="error" class="error"></div>

    <div id="resultCard" class="result-card">
      <div id="meta" class="meta"></div>
      <div class="table-wrap">
        <table>
          <thead>
            <tr>
              <th>Rank</th>
              <th>Title</th>
              <th>IMDb ID</th>
              <th>Type</th>
              <th>Year</th>
              <th>Rating</th>
              <th>Popularity</th>
              <th>Genre</th>
              <th>Similarity</th>
              <th>Composite</th>
            </tr>
          </thead>
          <tbody id="rows"></tbody>
        </table>
      </div>
    </div>
  </div>

  <script>
    const queryInput = document.getElementById('query');
    const searchBtn = document.getElementById('searchBtn');
    const suggestions = document.getElementById('suggestions');
    const rows = document.getElementById('rows');
    const meta = document.getElementById('meta');
    const resultCard = document.getElementById('resultCard');
    const errorBox = document.getElementById('error');
    const loading = document.getElementById('loading');

    function showError(msg) {
      if (!msg) {
        errorBox.style.display = 'none';
        errorBox.textContent = '';
        return;
      }
      errorBox.textContent = msg;
      errorBox.style.display = 'block';
    }

    function hideSuggestions() {
      suggestions.style.display = 'none';
      suggestions.innerHTML = '';
    }

    function setLoading(isLoading) {
      loading.style.display = isLoading ? 'flex' : 'none';
      searchBtn.disabled = isLoading;
      searchBtn.textContent = isLoading ? 'Searching...' : 'Search';
    }

    async function fetchSuggestions() {
      const q = queryInput.value.trim();
      if (q.length < 2) {
        hideSuggestions();
        return;
      }

      const res = await fetch(`/suggest?q=${encodeURIComponent(q)}`);
      const data = await res.json();

      if (!data.items || data.items.length === 0) {
        hideSuggestions();
        return;
      }

      suggestions.innerHTML = '';
      data.items.forEach(item => {
        const div = document.createElement('div');
        div.className = 'suggestion-item';
        div.textContent = `${item.title} (${item.year || 'N/A'}) [${item.imdb_id}]`;
        div.addEventListener('click', () => {
          queryInput.value = item.title;
          hideSuggestions();
          runSearch();
        });
        suggestions.appendChild(div);
      });
      suggestions.style.display = 'block';
    }

    async function runSearch() {
      const q = queryInput.value.trim();
      if (!q) {
        showError('Please enter a movie title or IMDb ID.');
        return;
      }

      showError('');
      hideSuggestions();
      setLoading(true);

      try {
        const res = await fetch('/recommend', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ reference: q })
        });

        const data = await res.json();
        if (!res.ok || data.error) {
          resultCard.style.display = 'none';
          showError(data.error || 'Failed to fetch recommendations.');
          return;
        }

        const source = data.source_movie || {};
        meta.textContent = `Source: ${source.title || 'N/A'} [${source.imdb_id || 'N/A'}] | Scope: ${data.scope_size} | Top: ${data.top_k}`;

        rows.innerHTML = '';
        (data.results || []).forEach((row, index) => {
          const tr = document.createElement('tr');
          tr.innerHTML = `
            <td>${index + 1}</td>
            <td>${row.title || ''}</td>
            <td>${row.imdb_id || ''}</td>
            <td>${row.content_type || ''}</td>
            <td>${row.year ?? ''}</td>
            <td>${Number(row.rating || 0).toFixed(1)}</td>
            <td>${row.popularity ?? ''}</td>
            <td>${row.genre || ''}</td>
            <td>${Number(row.similarity_score || 0).toFixed(3)}</td>
            <td>${Number(row.composite_score || 0).toFixed(3)}</td>
          `;
          rows.appendChild(tr);
        });

        resultCard.style.display = 'block';
      } catch (err) {
        resultCard.style.display = 'none';
        showError('Network error while fetching recommendations.');
      } finally {
        setLoading(false);
      }
    }

    let timer = null;
    queryInput.addEventListener('input', () => {
      clearTimeout(timer);
      timer = setTimeout(fetchSuggestions, 120);
    });

    queryInput.addEventListener('keydown', (e) => {
      if (e.key === 'Enter') {
        e.preventDefault();
        runSearch();
      }
    });

    searchBtn.addEventListener('click', runSearch);

    document.addEventListener('click', (e) => {
      if (!suggestions.contains(e.target) && e.target !== queryInput) {
        hideSuggestions();
      }
    });
  </script>
</body>
</html>
"""


def _movie_title(movie: Any) -> str:
    return str(getattr(movie, "Name", "") or "").strip()


def _movie_id(movie: Any) -> str:
    return str(getattr(movie, "imdb_id", "") or "").strip()


def _movie_year(movie: Any) -> str:
    value = getattr(movie, "Year", "")
    return str(value) if value is not None else ""


def _safe_float(value: Any, default: float = 0.0) -> float:
  try:
    return float(value)
  except (TypeError, ValueError):
    return default


def _safe_int(value: Any, default: int = 0) -> int:
  try:
    return int(value)
  except (TypeError, ValueError):
    return default


def _suggest_movies(query: str, limit: int = MAX_SUGGESTIONS) -> List[Dict[str, str]]:
    q = query.strip().lower()
    if len(q) < 2:
        return []

    movies = load_movie_universe()

    starts_with: List[Any] = []
    contains: List[Any] = []

    for movie in movies:
        title = _movie_title(movie)
        imdb_id = _movie_id(movie)
        title_l = title.lower()
        imdb_l = imdb_id.lower()

        if q in imdb_l:
            starts_with.append(movie)
            continue

        if title_l.startswith(q) or any(token.startswith(q) for token in title_l.split()):
            starts_with.append(movie)
        elif q in title_l:
            contains.append(movie)

    ranked = starts_with + contains
    seen = set()
    out: List[Dict[str, str]] = []

    for movie in ranked:
        imdb_id = _movie_id(movie)
        if not imdb_id or imdb_id in seen:
            continue
        seen.add(imdb_id)
        out.append({
            "imdb_id": imdb_id,
            "title": _movie_title(movie),
            "year": _movie_year(movie),
        })
        if len(out) >= limit:
            break

    return out


async def _recommend_from_reference_web(reference: str, scope_size: int = 500, top_k: int = 10) -> Dict[str, Any]:
    movies = load_movie_universe()
    source_movie = resolve_reference_movie(reference, movies)
    if source_movie is None:
        return {
            "error": f"Reference movie not found: {reference}",
            "total_movies": len(movies),
            "scope_size": 0,
            "top_k": 0,
            "results": [],
        }

    user_prompt = (
      f"Recommend {top_k} movies and shows similar to {reference}. "
      f"Use the {scope_size}-item scoped pool strategy and provide the structured ranking output."
    )

    agent = build_agent()
    result = await Runner.run(agent, input=user_prompt)
    report = result.final_output

    output_rows: List[Dict[str, Any]] = []
    for row in report.rankings:
      output_rows.append(
        {
          "imdb_id": row.imdb_id,
          "title": row.title,
          "content_type": row.content_type,
          "year": _safe_int(row.year, 0),
          "rating": _safe_float(row.rating, 0.0),
          "popularity": _safe_int(row.popularity, 0),
          "genre": "",
          "similarity_score": _safe_float(row.similarity_score, 0.0),
          "composite_score": _safe_float(row.composite_score, 0.0),
          "why_it_ranks": row.why_it_ranks,
        }
      )

    return {
        "source_movie": {
          "imdb_id": getattr(source_movie, "imdb_id", None),
          "title": report.source_movie,
          "content_type": getattr(source_movie, "Type", None),
          "year": getattr(source_movie, "Year", None),
          "rating": getattr(source_movie, "Rating", None),
        },
        "total_movies": len(movies),
        "scope_size": scope_size,
        "top_k": len(output_rows),
        "weights": {
          "scope_size": scope_size,
          "top_k": top_k,
        },
        "methodology": getattr(report, "methodology", ""),
        "notes": getattr(report, "notes", ""),
        "results": output_rows,
    }


@app.route("/", methods=["GET"])
def index() -> str:
    return render_template_string(HTML_PAGE)


@app.route("/suggest", methods=["GET"])
def suggest() -> Any:
    query = request.args.get("q", "")
    return jsonify({"items": _suggest_movies(query)})


@app.route("/recommend", methods=["POST"])
def recommend() -> Any:
    payload = request.get_json(silent=True) or {}
    reference = str(payload.get("reference", "")).strip()
    if not reference:
        return jsonify({"error": "reference is required"}), 400

    result = asyncio.run(
      _recommend_from_reference_web(
        reference=reference,
        scope_size=DEFAULT_SCOPE_SIZE,
        top_k=DEFAULT_TOP_K,
      )
    )
    if result.get("error"):
        return jsonify(result), 404
    return jsonify(result)


def launch_website() -> None:
    webbrowser.open(f"http://{HOST}:{PORT}", new=2)


def run_server() -> None:
    app.run(host=HOST, port=PORT, debug=False, use_reloader=False)


def main() -> None:
    try:
        import tkinter as tk
    except Exception:
        server_thread = threading.Thread(target=run_server, daemon=True)
        server_thread.start()
        time.sleep(0.6)
        launch_website()
        print(f"Website is running at http://{HOST}:{PORT}")
        print("Press Ctrl+C to stop.")
        server_thread.join()
        return

    root = tk.Tk()
    root.title("Movie Recommendation Website Launcher")
    root.geometry("420x180")

    label = tk.Label(
        root,
        text="Press the button to launch the movie recommendation website.",
        wraplength=360,
        justify="center",
    )
    label.pack(pady=24)

    state = {"started": False}

    def on_launch() -> None:
        if state["started"]:
            launch_website()
            return

        server_thread = threading.Thread(target=run_server, daemon=True)
        server_thread.start()
        time.sleep(0.6)
        launch_website()
        state["started"] = True
        status.config(text=f"Running at http://{HOST}:{PORT}")

    launch_button = tk.Button(root, text="Launch Website", width=20, command=on_launch)
    launch_button.pack(pady=8)

    status = tk.Label(root, text="")
    status.pack(pady=8)

    root.mainloop()


if __name__ == "__main__":
    main()
