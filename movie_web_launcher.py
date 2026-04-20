import threading
import time
import webbrowser
import asyncio
import os
from typing import Any, Dict, List, Optional

from flask import Flask, jsonify, render_template_string, request
from agents import Agent, Runner, set_default_openai_key

from movie_action_compare import compare_two_movies, run_compare_insight_with_agent
from movie_action_rank_set import rank_user_selected_set
from movie_action_rank_top import rank_top_movies_shows_by_genre_or_year
from movie_action_similar import rank_top_from_scoped_pool, scope_candidates
from movie_agent_shared import (
    DEFAULT_SCOPE_SIZE,
    DEFAULT_TOP_K,
    load_movie_universe,
    movie_to_dict,
    parse_genre_tokens,
    resolve_reference_movie,
    safe_float,
    safe_int,
)

HOST = "127.0.0.1"
PORT = 5001
MAX_SUGGESTIONS = 8

app = Flask(__name__)


PAGE_SHELL = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{{ title }}</title>
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
      --nav-bg: rgba(255, 255, 255, 0.86);
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

    .top-nav {
      position: sticky;
      top: 0;
      z-index: 10;
      border-bottom: 1px solid #dbe5f5;
      background: var(--nav-bg);
      backdrop-filter: blur(8px);
    }

    .top-nav-inner {
      max-width: 1120px;
      margin: 0 auto;
      padding: 12px 14px;
      display: flex;
      gap: 10px;
      align-items: center;
      flex-wrap: wrap;
    }

    .nav-brand {
      text-decoration: none;
      color: #083684;
      font-weight: 700;
      font-size: 20px;
      margin-right: 6px;
    }

    .nav-link {
      text-decoration: none;
      color: #1e3a8a;
      border: 1px solid #bfd1ef;
      border-radius: 999px;
      padding: 7px 12px;
      font-size: 13px;
      font-weight: 600;
      background: #f3f8ff;
      transition: all 0.15s ease;
    }

    .nav-link:hover {
      background: #e4efff;
      border-color: #9bb7e9;
    }

    .nav-link.active {
      color: #fff;
      background: var(--blue);
      border-color: var(--blue);
    }

    .wrap {
      max-width: 1120px;
      margin: 0 auto;
      padding: 28px 16px 32px;
    }

    .hero {
      text-align: center;
      margin-bottom: 18px;
    }

    .hero h1 {
      margin: 0 0 8px;
      font-size: clamp(26px, 4vw, 38px);
      color: #083684;
      letter-spacing: 0.2px;
    }

    .hero p {
      margin: 0;
      color: var(--muted);
      font-size: 15px;
    }

    .card {
      background: var(--card);
      border: 1px solid var(--line);
      border-radius: 18px;
      padding: 16px;
      box-shadow: 0 8px 24px rgba(15, 23, 42, 0.06);
    }

    .form-row {
      display: grid;
      grid-template-columns: 1fr auto;
      gap: 10px;
      align-items: center;
    }

    .stack-row {
      display: grid;
      gap: 10px;
    }

    .input {
      width: 100%;
      border: 1px solid #c4d0e5;
      border-radius: 12px;
      padding: 12px 14px;
      font-size: 15px;
      outline: none;
      transition: border-color 0.15s ease, box-shadow 0.15s ease;
      background: #fff;
    }

    .input:focus {
      border-color: var(--blue);
      box-shadow: 0 0 0 4px rgba(11, 87, 208, 0.14);
    }

    .btn {
      border: 0;
      border-radius: 12px;
      background: var(--blue);
      color: #fff;
      padding: 11px 16px;
      font-size: 14px;
      font-weight: 600;
      cursor: pointer;
      transition: background 0.15s ease;
    }

    .btn:hover { background: var(--blue-hover); }
    .btn:disabled { background: #7aa7ea; cursor: not-allowed; }

    .error {
      margin-top: 12px;
      color: #9f1239;
      font-size: 14px;
      display: none;
    }

    .loading {
      margin-top: 10px;
      display: none;
      align-items: center;
      gap: 8px;
      color: #0f2855;
      font-size: 14px;
    }

    .spinner {
      width: 16px;
      height: 16px;
      border: 2px solid #cfe0fb;
      border-top-color: var(--blue);
      border-radius: 50%;
      animation: spin 0.8s linear infinite;
    }

    @keyframes spin {
      to { transform: rotate(360deg); }
    }

    .result-card {
      margin-top: 14px;
      background: #fff;
      border: 1px solid var(--line);
      border-radius: 16px;
      box-shadow: 0 8px 24px rgba(15, 23, 42, 0.06);
      overflow: hidden;
      display: none;
    }

    .meta {
      padding: 12px 14px;
      border-bottom: 1px solid #e7edf7;
      color: #0f2855;
      font-size: 14px;
      background: #f8fbff;
    }

    .table-wrap { overflow-x: auto; }

    table {
      width: 100%;
      border-collapse: collapse;
      min-width: 860px;
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
    }

    .hint {
      margin: 10px 0 0;
      color: #475569;
      font-size: 13px;
    }

    .dropdown {
      margin-top: 8px;
      border: 1px solid var(--line);
      border-radius: 12px;
      background: #fff;
      overflow: hidden;
      display: none;
      max-height: 220px;
      overflow-y: auto;
    }

    .suggestion-item {
      padding: 10px 12px;
      cursor: pointer;
      border-bottom: 1px solid #eef2f8;
      font-size: 14px;
    }

    .suggestion-item:last-child { border-bottom: none; }
    .suggestion-item:hover { background: #eff6ff; }

    .hub-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 12px;
      margin-top: 16px;
    }

    .hub-item {
      display: block;
      text-decoration: none;
      color: inherit;
      border: 1px solid #d6e3f8;
      border-radius: 14px;
      background: #fff;
      padding: 14px;
      transition: transform 0.15s ease, box-shadow 0.15s ease, border-color 0.15s ease;
      box-shadow: 0 8px 16px rgba(15, 23, 42, 0.04);
    }

    .hub-item:hover {
      transform: translateY(-2px);
      border-color: #a9c3ec;
      box-shadow: 0 12px 20px rgba(15, 23, 42, 0.08);
    }

    .hub-item h3 {
      margin: 0 0 8px;
      font-size: 17px;
      color: #1e3a8a;
    }

    .hub-item p {
      margin: 0;
      color: #475569;
      font-size: 14px;
      line-height: 1.45;
    }

    @media (max-width: 860px) {
      .form-row { grid-template-columns: 1fr; }
      .top-nav-inner { gap: 8px; }
      .nav-brand { width: 100%; }
    }
  </style>
</head>
<body>
  <nav class="top-nav">
    <div class="top-nav-inner">
      <a class="nav-brand" href="/">Movie Search</a>
      <a class="nav-link {% if active == 'home' %}active{% endif %}" href="/">Movie Search</a>
      <a class="nav-link {% if active == 'similar' %}active{% endif %}" href="/similar">Finding Similar Movie</a>
      <a class="nav-link {% if active == 'compare' %}active{% endif %}" href="/compare">Compare Two Movies</a>
      <a class="nav-link {% if active == 'rank_set' %}active{% endif %}" href="/rank-set">Ranking a Set of Movies</a>
      <a class="nav-link {% if active == 'rank_top' %}active{% endif %}" href="/rank-top">Ranking Top Movies</a>
      <a class="nav-link {% if active == 'basic' %}active{% endif %}" href="/basic-description">Basic Movie Description</a>
    </div>
  </nav>
  <main class="wrap">
    {{ content | safe }}
  </main>
  {{ script | safe }}
</body>
</html>
"""


HOME_CONTENT = """
<section class="hero">
  <h1>Welcome to Movie Search</h1>
  <p>This website helps you explore and evaluate movies and shows using your local dataset.</p>
</section>

<section class="card">
  <h2 style="margin:0 0 10px; font-size:22px; color:#0f2855;">What You Can Do Here</h2>
  <p style="margin:0; color:#334155; line-height:1.55;">
    Start with a title or IMDb ID, then use focused tools to compare, rank, and inspect titles.
    Each section is specialized, so you can choose the workflow that matches your task.
  </p>

  <div class="hub-grid">
    <a class="hub-item" href="/similar">
      <h3>Finding Similar Movie</h3>
      <p>Find recommendations that are close to a reference title by similarity, rating, and popularity.</p>
    </a>
    <a class="hub-item" href="/compare">
      <h3>Compare Two Movies</h3>
      <p>Compare two references side by side with differences in rating, popularity, genre overlap, and year.</p>
    </a>
    <a class="hub-item" href="/rank-set">
      <h3>Ranking a Set of Movies</h3>
      <p>Input your own list and rank that exact set by rating, popularity, and runtime.</p>
    </a>
    <a class="hub-item" href="/rank-top">
      <h3>Ranking Top Movies</h3>
      <p>Get top entries by filter criteria such as genre, year, and content mode (movie/show/both).</p>
    </a>
    <a class="hub-item" href="/basic-description">
      <h3>Basic Movie Description</h3>
      <p>Retrieve basic profile information for a title including metadata, genres, and a short summary.</p>
    </a>
  </div>
</section>
"""


SIMILAR_CONTENT = """
<section class="hero">
  <h1>Finding Similar Movie</h1>
  <p>Search by title or IMDb ID and return the top similar recommendations.</p>
</section>

<section class="card">
  <div class="form-row">
    <input id="query" class="input" placeholder="Search movie title or IMDb ID (e.g. Game of Thrones or tt0944947)" autocomplete="off" />
    <button id="searchBtn" class="btn" type="button">Search</button>
  </div>
  <div id="suggestions" class="dropdown"></div>
  <div id="loading" class="loading" aria-live="polite">
    <span class="spinner" aria-hidden="true"></span>
    <span>Searching recommendations...</span>
  </div>
  <div id="error" class="error"></div>

  <div id="resultCard" class="result-card">
    <div id="meta" class="meta"></div>
    <div class="table-wrap">
      <table>
        <thead>
          <tr>
            <th>Rank</th>
            <th id="thTitle" data-sort-key="title">Title</th>
            <th>IMDb ID</th>
            <th>Type</th>
            <th>Year</th>
            <th id="thRating" data-sort-key="rating">Rating</th>
            <th id="thPopularity" data-sort-key="popularity">Popularity</th>
            <th>Genre</th>
            <th>Similarity</th>
            <th>Composite</th>
            <th>AI Insight</th>
          </tr>
        </thead>
        <tbody id="rows"></tbody>
      </table>
    </div>
  </div>
</section>
"""


SIMILAR_SCRIPT = """
<script>
  const queryInput = document.getElementById('query');
  const searchBtn = document.getElementById('searchBtn');
  const suggestions = document.getElementById('suggestions');
  const rows = document.getElementById('rows');
  const meta = document.getElementById('meta');
  const resultCard = document.getElementById('resultCard');
  const errorBox = document.getElementById('error');
  const loading = document.getElementById('loading');
  const sortHeaders = [
    document.getElementById('thTitle'),
    document.getElementById('thRating'),
    document.getElementById('thPopularity')
  ];

  const SORT_CYCLES = {
    title: ['asc', 'default'],
    rating: ['desc', 'asc', 'default'],
    popularity: ['desc', 'asc', 'default']
  };

  let defaultRowsData = [];
  let currentRowsData = [];
  let currentSourceReference = '';
  let activeSort = { key: null, stateIndex: 0 };
  const similarAiCache = {};
  const similarAiExpanded = {};
  const similarAiLoading = {};

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

  function resetHeaderLabels() {
    document.getElementById('thTitle').textContent = 'Title';
    document.getElementById('thRating').textContent = 'Rating';
    document.getElementById('thPopularity').textContent = 'Popularity';
  }

  function updateSortHeaderLabel(key, state) {
    resetHeaderLabels();
    if (!key || state === 'default') return;

    if (key === 'title') {
      document.getElementById('thTitle').textContent = 'Title (A-Z)';
      return;
    }
    if (key === 'rating') {
      document.getElementById('thRating').textContent = state === 'desc' ? 'Rating ↓' : 'Rating ↑';
      return;
    }
    if (key === 'popularity') {
      document.getElementById('thPopularity').textContent = state === 'desc' ? 'Popularity ↓' : 'Popularity ↑';
    }
  }

  function renderRows(rowData) {
    currentRowsData = rowData;
    rows.innerHTML = '';
    rowData.forEach((row, index) => {
      const imdbId = row.imdb_id || '';
      const isLoading = !!similarAiLoading[imdbId];
      const hasInsight = !!similarAiCache[imdbId];
      const isExpanded = !!similarAiExpanded[imdbId];
      const buttonLabel = isLoading ? 'Loading...' : (isExpanded ? 'Hide insight' : 'AI insight');

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
        <td><button class="btn" data-ai-insight-btn="1" data-imdb-id="${imdbId}" style="padding:6px 10px; font-size:12px; border-radius:8px;">${buttonLabel}</button></td>
      `;
      rows.appendChild(tr);

      if (isExpanded && hasInsight) {
        const detailTr = document.createElement('tr');
        detailTr.innerHTML = `<td colspan="11" style="background:#f8fbff; color:#334155; line-height:1.45;"><strong>AI Insight:</strong> ${similarAiCache[imdbId]}</td>`;
        rows.appendChild(detailTr);
      }
    });

    rows.querySelectorAll('button[data-ai-insight-btn="1"]').forEach((button) => {
      button.addEventListener('click', async () => {
        const imdbId = button.getAttribute('data-imdb-id') || '';
        const targetRow = (currentRowsData || []).find((r) => String(r.imdb_id || '') === imdbId);
        if (!targetRow) return;

        if (similarAiCache[imdbId]) {
          similarAiExpanded[imdbId] = !similarAiExpanded[imdbId];
          renderRows(currentRowsData);
          return;
        }

        similarAiLoading[imdbId] = true;
        renderRows(currentRowsData);
        try {
          const res = await fetch('/api/similar-insight', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              source_reference: currentSourceReference,
              candidate_imdb_id: targetRow.imdb_id || '',
              candidate_title: targetRow.title || '',
              similarity_score: targetRow.similarity_score || 0,
              composite_score: targetRow.composite_score || 0,
            })
          });
          const data = await res.json();
          if (!res.ok || data.error) {
            similarAiCache[imdbId] = 'AI insight is currently unavailable for this row.';
          } else {
            similarAiCache[imdbId] = data.insight || 'No AI insight returned.';
          }
        } catch (err) {
          similarAiCache[imdbId] = 'AI insight is currently unavailable for this row.';
        } finally {
          similarAiLoading[imdbId] = false;
          similarAiExpanded[imdbId] = true;
          renderRows(currentRowsData);
        }
      });
    });
  }

  function getNextSortState(sortKey) {
    const cycle = SORT_CYCLES[sortKey] || ['default'];
    if (activeSort.key !== sortKey) {
      activeSort = { key: sortKey, stateIndex: 0 };
      return cycle[0];
    }
    const nextIndex = (activeSort.stateIndex + 1) % cycle.length;
    activeSort = { key: sortKey, stateIndex: nextIndex };
    return cycle[nextIndex];
  }

  function applySort(sortKey) {
    if (!defaultRowsData.length) return;

    const state = getNextSortState(sortKey);
    let sorted = [...defaultRowsData];

    if (state === 'default') {
      activeSort = { key: null, stateIndex: 0 };
      updateSortHeaderLabel(null, 'default');
      renderRows(sorted);
      return;
    }

    if (sortKey === 'title') {
      sorted.sort((a, b) => String(a.title || '').localeCompare(String(b.title || ''), undefined, { sensitivity: 'base' }));
    } else if (sortKey === 'rating') {
      sorted.sort((a, b) => Number(a.rating || 0) - Number(b.rating || 0));
    } else if (sortKey === 'popularity') {
      sorted.sort((a, b) => Number(a.popularity || 0) - Number(b.popularity || 0));
    }

    if (state === 'desc') sorted.reverse();

    updateSortHeaderLabel(sortKey, state);
    renderRows(sorted);
  }

  sortHeaders.forEach((th) => {
    th.addEventListener('click', () => {
      const sortKey = th.dataset.sortKey;
      if (sortKey) applySort(sortKey);
    });
  });

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
    data.items.forEach((item) => {
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
      const res = await fetch('/api/recommend', {
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
      currentSourceReference = source.imdb_id || q;

      defaultRowsData = [...(data.results || [])];
      Object.keys(similarAiCache).forEach((key) => { delete similarAiCache[key]; });
      Object.keys(similarAiExpanded).forEach((key) => { delete similarAiExpanded[key]; });
      Object.keys(similarAiLoading).forEach((key) => { delete similarAiLoading[key]; });
      activeSort = { key: null, stateIndex: 0 };
      updateSortHeaderLabel(null, 'default');
      renderRows(defaultRowsData);
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
"""


COMPARE_CONTENT = """
<section class="hero">
  <h1>Compare Two Movies</h1>
  <p>Enter two titles or IMDb IDs to compare metadata and differences.</p>
</section>

<section class="card">
  <div class="stack-row">
    <input id="firstRef" class="input" placeholder="First title or IMDb ID" />
    <div id="firstSuggestions" class="dropdown"></div>
    <input id="secondRef" class="input" placeholder="Second title or IMDb ID" />
    <div id="secondSuggestions" class="dropdown"></div>
    <button id="compareBtn" class="btn" type="button">Compare</button>
  </div>
  <div id="compareLoading" class="loading"><span class="spinner"></span><span>Comparing...</span></div>
  <div id="compareError" class="error"></div>

  <div id="compareResult" class="result-card">
    <div id="compareMeta" class="meta"></div>
    <div class="table-wrap">
      <table>
        <thead>
          <tr>
            <th>Field</th>
            <th id="firstHeader">First</th>
            <th id="secondHeader">Second</th>
          </tr>
        </thead>
        <tbody id="compareRows"></tbody>
      </table>
    </div>
    <div id="compareAiBlock" style="display:none; padding:14px; border-top:1px solid #e7edf7; background:#fbfdff;">
      <h3 style="margin:0 0 10px; font-size:16px; color:#0f2855;">AI Comparison Insight</h3>
      <div id="aiInsightLoading" style="display:none; align-items:center; gap:8px; color:#0f2855; margin:0 0 8px;">
        <span class="spinner" aria-hidden="true"></span>
        <span>Generating AI insight...</span>
      </div>
      <p id="aiGenreTone" style="margin:0 0 8px; color:#334155;"></p>
      <p id="aiThemes" style="margin:0 0 8px; color:#334155;"></p>
      <p id="aiReception" style="margin:0 0 8px; color:#334155;"></p>
      <p id="aiTaste" style="margin:0; color:#0f2855; font-weight:600;"></p>
    </div>
  </div>
</section>
"""


COMPARE_SCRIPT = """
<script>
  const compareBtn = document.getElementById('compareBtn');
  const firstRef = document.getElementById('firstRef');
  const secondRef = document.getElementById('secondRef');
  const compareLoading = document.getElementById('compareLoading');
  const compareError = document.getElementById('compareError');
  const compareResult = document.getElementById('compareResult');
  const compareMeta = document.getElementById('compareMeta');
  const compareRows = document.getElementById('compareRows');
  const firstHeader = document.getElementById('firstHeader');
  const secondHeader = document.getElementById('secondHeader');
  const firstSuggestions = document.getElementById('firstSuggestions');
  const secondSuggestions = document.getElementById('secondSuggestions');
  const compareAiBlock = document.getElementById('compareAiBlock');
  const aiGenreTone = document.getElementById('aiGenreTone');
  const aiThemes = document.getElementById('aiThemes');
  const aiReception = document.getElementById('aiReception');
  const aiTaste = document.getElementById('aiTaste');
  const aiInsightLoading = document.getElementById('aiInsightLoading');

  function setAiInsightLoading() {
    aiInsightLoading.style.display = 'flex';
    aiGenreTone.textContent = '';
    aiThemes.textContent = '';
    aiReception.textContent = '';
    aiTaste.textContent = '';
    compareAiBlock.style.display = 'block';
  }

  function hideAiInsight() {
    compareAiBlock.style.display = 'none';
    aiInsightLoading.style.display = 'none';
    aiGenreTone.textContent = '';
    aiThemes.textContent = '';
    aiReception.textContent = '';
    aiTaste.textContent = '';
  }

  function renderAiInsight(aiInsight) {
    if (!aiInsight) {
      hideAiInsight();
      return;
    }
    aiInsightLoading.style.display = 'none';
    aiGenreTone.textContent = `Genre and Tone: ${aiInsight.genre_and_tone || ''}`;
    aiThemes.textContent = `Main Themes: ${aiInsight.main_themes || ''}`;
    aiReception.textContent = `Critical Reception: ${aiInsight.critical_reception || ''}`;
    aiTaste.textContent = `Taste Recommendation: ${aiInsight.taste_recommendation || ''}`;
    compareAiBlock.style.display = 'block';
  }

  async function fetchAiInsight(first, second) {
    try {
      const res = await fetch('/api/compare-insight', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ first_reference: first, second_reference: second })
      });
      const data = await res.json();
      if (!res.ok || data.error) {
        hideAiInsight();
        return;
      }
      renderAiInsight(data.ai_insight || null);
    } catch (err) {
      hideAiInsight();
    }
  }

  function hideMovieSuggestions(dropdown) {
    dropdown.style.display = 'none';
    dropdown.innerHTML = '';
  }

  async function fetchMovieSuggestions(inputEl, dropdownEl) {
    const q = inputEl.value.trim();
    if (q.length < 2) {
      hideMovieSuggestions(dropdownEl);
      return;
    }

    const res = await fetch(`/suggest?q=${encodeURIComponent(q)}`);
    const data = await res.json();
    const items = data.items || [];
    if (!items.length) {
      hideMovieSuggestions(dropdownEl);
      return;
    }

    dropdownEl.innerHTML = '';
    items.forEach((item) => {
      const div = document.createElement('div');
      div.className = 'suggestion-item';
      div.textContent = `${item.title} (${item.year || 'N/A'}) [${item.imdb_id}]`;
      div.addEventListener('click', () => {
        inputEl.value = item.title;
        hideMovieSuggestions(dropdownEl);
      });
      dropdownEl.appendChild(div);
    });
    dropdownEl.style.display = 'block';
  }

  function setCompareLoading(value) {
    compareLoading.style.display = value ? 'flex' : 'none';
    compareBtn.disabled = value;
  }

  function showCompareError(msg) {
    if (!msg) {
      compareError.style.display = 'none';
      compareError.textContent = '';
      return;
    }
    compareError.style.display = 'block';
    compareError.textContent = msg;
  }

  function appendCompareRow(label, left, right) {
    const tr = document.createElement('tr');
    tr.innerHTML = `<td>${label}</td><td>${left ?? ''}</td><td>${right ?? ''}</td>`;
    compareRows.appendChild(tr);
  }

  async function runCompare() {
    const first = firstRef.value.trim();
    const second = secondRef.value.trim();
    if (!first || !second) {
      showCompareError('Please fill both references.');
      return;
    }

    showCompareError('');
    setCompareLoading(true);

    try {
      const res = await fetch('/api/compare', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ first_reference: first, second_reference: second })
      });
      const data = await res.json();

      if (!res.ok || data.error) {
        compareResult.style.display = 'none';
        showCompareError(data.error || 'Could not compare these references.');
        return;
      }

      const firstMovie = data.first_movie || {};
      const secondMovie = data.second_movie || {};
      const comparison = data.comparison || {};

      firstHeader.textContent = firstMovie.title || 'First';
      secondHeader.textContent = secondMovie.title || 'Second';

      compareMeta.textContent = `Shared genres: ${(comparison.shared_genres || []).join(', ') || 'None'} | Higher rated: ${comparison.higher_rated || 'tie'} | More popular: ${comparison.more_popular || 'tie'}`;
      compareRows.innerHTML = '';

      appendCompareRow('IMDb ID', firstMovie.imdb_id, secondMovie.imdb_id);
      appendCompareRow('Type', firstMovie.content_type, secondMovie.content_type);
      appendCompareRow('Genre', firstMovie.genre, secondMovie.genre);
      appendCompareRow('Year', firstMovie.year, secondMovie.year);
      appendCompareRow('Rating', Number(firstMovie.rating || 0).toFixed(1), Number(secondMovie.rating || 0).toFixed(1));
      appendCompareRow('Popularity', firstMovie.popularity || 0, secondMovie.popularity || 0);
      appendCompareRow('Description', firstMovie.description || '', secondMovie.description || '');
      appendCompareRow('Rating diff (first-second)', comparison.rating_diff ?? 0, '');
      appendCompareRow('Popularity diff (first-second)', comparison.popularity_diff ?? 0, '');
      appendCompareRow('Year diff (first-second)', comparison.year_diff ?? 0, '');

      compareResult.style.display = 'block';
      setAiInsightLoading();
      fetchAiInsight(first, second);
    } catch (err) {
      compareResult.style.display = 'none';
      hideAiInsight();
      showCompareError('Network error while comparing movies.');
    } finally {
      setCompareLoading(false);
    }
  }

  compareBtn.addEventListener('click', runCompare);

  let firstTimer = null;
  let secondTimer = null;
  firstRef.addEventListener('input', () => {
    clearTimeout(firstTimer);
    firstTimer = setTimeout(() => {
      fetchMovieSuggestions(firstRef, firstSuggestions);
    }, 120);
  });

  secondRef.addEventListener('input', () => {
    clearTimeout(secondTimer);
    secondTimer = setTimeout(() => {
      fetchMovieSuggestions(secondRef, secondSuggestions);
    }, 120);
  });

  [firstRef, secondRef].forEach((el) => {
    el.addEventListener('keydown', (e) => {
      if (e.key === 'Enter') {
        e.preventDefault();
        runCompare();
      }
    });
  });

  document.addEventListener('click', (e) => {
    if (!firstSuggestions.contains(e.target) && e.target !== firstRef) {
      hideMovieSuggestions(firstSuggestions);
    }
    if (!secondSuggestions.contains(e.target) && e.target !== secondRef) {
      hideMovieSuggestions(secondSuggestions);
    }
  });
</script>
"""


RANK_SET_CONTENT = """
<section class="hero">
  <h1>Ranking a Set of Movies</h1>
  <p>Provide a comma-separated set of references, then rank only those entries.</p>
</section>

<section class="card">
  <div class="stack-row">
    <textarea id="setInput" class="input" rows="5" placeholder="Example: Inception, The Dark Knight, tt0944947"></textarea>
    <div id="setSuggestions" class="dropdown"></div>
    <button id="rankSetBtn" class="btn" type="button">Rank Set</button>
    <p class="hint">Ranking priority: rating, then popularity, then length.</p>
  </div>
  <div id="rankSetLoading" class="loading"><span class="spinner"></span><span>Ranking...</span></div>
  <div id="rankSetError" class="error"></div>

  <div id="rankSetResult" class="result-card">
    <div id="rankSetMeta" class="meta"></div>
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
            <th>Length</th>
            <th>Genre</th>
          </tr>
        </thead>
        <tbody id="rankSetRows"></tbody>
      </table>
    </div>
    <div id="rankSetAiBlock" style="display:none; padding:14px; border-top:1px solid #e7edf7; background:#fbfdff;">
      <h3 style="margin:0 0 10px; font-size:16px; color:#0f2855;">AI Ranking Insight</h3>
      <div id="rankSetAiLoading" style="display:none; align-items:center; gap:8px; color:#0f2855; margin:0 0 8px;">
        <span class="spinner" aria-hidden="true"></span>
        <span>Generating AI insight...</span>
      </div>
      <p id="rankSetAiText" style="margin:0; color:#334155; line-height:1.5;"></p>
    </div>
  </div>
</section>
"""


RANK_SET_SCRIPT = """
<script>
  const rankSetBtn = document.getElementById('rankSetBtn');
  const setInput = document.getElementById('setInput');
  const rankSetLoading = document.getElementById('rankSetLoading');
  const rankSetError = document.getElementById('rankSetError');
  const rankSetResult = document.getElementById('rankSetResult');
  const rankSetMeta = document.getElementById('rankSetMeta');
  const rankSetRows = document.getElementById('rankSetRows');
  const setSuggestions = document.getElementById('setSuggestions');
  const rankSetAiBlock = document.getElementById('rankSetAiBlock');
  const rankSetAiLoading = document.getElementById('rankSetAiLoading');
  const rankSetAiText = document.getElementById('rankSetAiText');

  function setRankSetAiLoading() {
    rankSetAiBlock.style.display = 'block';
    rankSetAiLoading.style.display = 'flex';
    rankSetAiText.textContent = '';
  }

  function hideRankSetAi() {
    rankSetAiBlock.style.display = 'none';
    rankSetAiLoading.style.display = 'none';
    rankSetAiText.textContent = '';
  }

  function renderRankSetAi(text) {
    if (!text) {
      hideRankSetAi();
      return;
    }
    rankSetAiBlock.style.display = 'block';
    rankSetAiLoading.style.display = 'none';
    rankSetAiText.textContent = text;
  }

  async function fetchRankSetAiInsight(references) {
    try {
      const res = await fetch('/api/rank-set-insight', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ references })
      });
      const data = await res.json();
      if (!res.ok || data.error) {
        hideRankSetAi();
        return;
      }
      renderRankSetAi(data.insight || '');
    } catch (err) {
      hideRankSetAi();
    }
  }

  function hideSetSuggestions() {
    setSuggestions.style.display = 'none';
    setSuggestions.innerHTML = '';
  }

  function getLastReferenceToken(raw) {
    const parts = String(raw || '').split(',');
    return (parts[parts.length - 1] || '').trim();
  }

  function applySetSuggestion(raw, selectedTitle) {
    const parts = String(raw || '').split(',');
    const committed = parts.slice(0, -1).map((v) => v.trim()).filter((v) => v.length > 0);
    committed.push(selectedTitle);
    return `${committed.join(', ')}, `;
  }

  async function fetchSetSuggestions() {
    const token = getLastReferenceToken(setInput.value);
    if (token.length < 2) {
      hideSetSuggestions();
      return;
    }

    const res = await fetch(`/suggest?q=${encodeURIComponent(token)}`);
    const data = await res.json();
    const items = data.items || [];
    if (!items.length) {
      hideSetSuggestions();
      return;
    }

    setSuggestions.innerHTML = '';
    items.forEach((item) => {
      const div = document.createElement('div');
      div.className = 'suggestion-item';
      div.textContent = `${item.title} (${item.year || 'N/A'}) [${item.imdb_id}]`;
      div.addEventListener('click', () => {
        setInput.value = applySetSuggestion(setInput.value, item.title || '');
        hideSetSuggestions();
        setInput.focus();
      });
      setSuggestions.appendChild(div);
    });
    setSuggestions.style.display = 'block';
  }

  function setRankSetLoading(value) {
    rankSetLoading.style.display = value ? 'flex' : 'none';
    rankSetBtn.disabled = value;
  }

  function showRankSetError(msg) {
    if (!msg) {
      rankSetError.style.display = 'none';
      rankSetError.textContent = '';
      return;
    }
    rankSetError.style.display = 'block';
    rankSetError.textContent = msg;
  }

  function parseReferences(raw) {
    return String(raw || '').split(',').map((v) => v.trim()).filter((v) => v.length > 0);
  }

  async function runRankSet() {
    const references = parseReferences(setInput.value);
    if (!references.length) {
      showRankSetError('Please enter at least one reference.');
      return;
    }

    showRankSetError('');
    setRankSetLoading(true);

    try {
      const res = await fetch('/api/rank-set', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ references })
      });
      const data = await res.json();

      if (!res.ok || data.error) {
        rankSetResult.style.display = 'none';
        showRankSetError(data.error || 'Could not rank this set.');
        return;
      }

      const unresolved = (data.unresolved || []).join(', ') || 'None';
      rankSetMeta.textContent = `Input: ${data.input_size || 0} | Resolved: ${data.resolved_count || 0} | Unresolved: ${unresolved}`;

      rankSetRows.innerHTML = '';
      (data.results || []).forEach((row, idx) => {
        const tr = document.createElement('tr');
        tr.innerHTML = `
          <td>${idx + 1}</td>
          <td>${row.title || ''}</td>
          <td>${row.imdb_id || ''}</td>
          <td>${row.content_type || ''}</td>
          <td>${row.year ?? ''}</td>
          <td>${Number(row.rating || 0).toFixed(1)}</td>
          <td>${row.popularity ?? ''}</td>
          <td>${row.length ?? ''}</td>
          <td>${row.genre || ''}</td>
        `;
        rankSetRows.appendChild(tr);
      });

      rankSetResult.style.display = 'block';
      setRankSetAiLoading();
      fetchRankSetAiInsight(references);
    } catch (err) {
      rankSetResult.style.display = 'none';
      hideRankSetAi();
      showRankSetError('Network error while ranking set.');
    } finally {
      setRankSetLoading(false);
    }
  }

  let setTimer = null;
  setInput.addEventListener('input', () => {
    clearTimeout(setTimer);
    setTimer = setTimeout(fetchSetSuggestions, 120);
  });

  rankSetBtn.addEventListener('click', runRankSet);

  document.addEventListener('click', (e) => {
    if (!setSuggestions.contains(e.target) && e.target !== setInput) {
      hideSetSuggestions();
    }
  });
</script>
"""


RANK_TOP_CONTENT = """
<section class="hero">
  <h1>Ranking Top Movies</h1>
  <p>Filter by genre, year, and content mode, then return top-ranked titles.</p>
</section>

<section class="card">
  <div class="stack-row">
    <input id="topGenre" class="input" placeholder="Genre (optional, e.g. Action)" />
    <div id="topGenreSuggestions" class="dropdown"></div>
    <input id="topYear" class="input" placeholder="Year (optional, e.g. 2014)" />
    <div id="topYearSuggestions" class="dropdown"></div>
    <select id="topMode" class="input">
      <option value="both">Both</option>
      <option value="movie">Movie only</option>
      <option value="show">Show only</option>
    </select>
    <input id="topK" class="input" placeholder="Top K (default 10)" value="10" />
    <button id="rankTopBtn" class="btn" type="button">Rank Top</button>
  </div>
  <div id="rankTopLoading" class="loading"><span class="spinner"></span><span>Ranking...</span></div>
  <div id="rankTopError" class="error"></div>

  <div id="rankTopResult" class="result-card">
    <div id="rankTopMeta" class="meta"></div>
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
          </tr>
        </thead>
        <tbody id="rankTopRows"></tbody>
      </table>
    </div>
    <div id="rankTopAiBlock" style="display:none; padding:14px; border-top:1px solid #e7edf7; background:#fbfdff;">
      <h3 style="margin:0 0 10px; font-size:16px; color:#0f2855;">AI Ranking Insight</h3>
      <div id="rankTopAiLoading" style="display:none; align-items:center; gap:8px; color:#0f2855; margin:0 0 8px;">
        <span class="spinner" aria-hidden="true"></span>
        <span>Generating AI insight...</span>
      </div>
      <p id="rankTopAiText" style="margin:0; color:#334155; line-height:1.5;"></p>
    </div>
  </div>
</section>
"""


RANK_TOP_SCRIPT = """
<script>
  const rankTopBtn = document.getElementById('rankTopBtn');
  const topGenre = document.getElementById('topGenre');
  const topYear = document.getElementById('topYear');
  const topMode = document.getElementById('topMode');
  const topK = document.getElementById('topK');
  const rankTopLoading = document.getElementById('rankTopLoading');
  const rankTopError = document.getElementById('rankTopError');
  const rankTopResult = document.getElementById('rankTopResult');
  const rankTopMeta = document.getElementById('rankTopMeta');
  const rankTopRows = document.getElementById('rankTopRows');
  const topGenreSuggestions = document.getElementById('topGenreSuggestions');
  const topYearSuggestions = document.getElementById('topYearSuggestions');
  const rankTopAiBlock = document.getElementById('rankTopAiBlock');
  const rankTopAiLoading = document.getElementById('rankTopAiLoading');
  const rankTopAiText = document.getElementById('rankTopAiText');

  function setRankTopAiLoading() {
    rankTopAiBlock.style.display = 'block';
    rankTopAiLoading.style.display = 'flex';
    rankTopAiText.textContent = '';
  }

  function hideRankTopAi() {
    rankTopAiBlock.style.display = 'none';
    rankTopAiLoading.style.display = 'none';
    rankTopAiText.textContent = '';
  }

  function renderRankTopAi(text) {
    if (!text) {
      hideRankTopAi();
      return;
    }
    rankTopAiBlock.style.display = 'block';
    rankTopAiLoading.style.display = 'none';
    rankTopAiText.textContent = text;
  }

  async function fetchRankTopAiInsight(payload) {
    try {
      const res = await fetch('/api/rank-top-insight', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });
      const data = await res.json();
      if (!res.ok || data.error) {
        hideRankTopAi();
        return;
      }
      renderRankTopAi(data.insight || '');
    } catch (err) {
      hideRankTopAi();
    }
  }

  function hideTopSuggestions(dropdown) {
    dropdown.style.display = 'none';
    dropdown.innerHTML = '';
  }

  async function fetchGenreSuggestions() {
    const q = topGenre.value.trim();
    if (q.length < 1) {
      hideTopSuggestions(topGenreSuggestions);
      return;
    }

    const res = await fetch(`/suggest/genres?q=${encodeURIComponent(q)}`);
    const data = await res.json();
    const items = data.items || [];
    if (!items.length) {
      hideTopSuggestions(topGenreSuggestions);
      return;
    }

    topGenreSuggestions.innerHTML = '';
    items.forEach((item) => {
      const div = document.createElement('div');
      div.className = 'suggestion-item';
      div.textContent = `${item.genre} (${item.count})`;
      div.addEventListener('click', () => {
        topGenre.value = item.genre || '';
        hideTopSuggestions(topGenreSuggestions);
      });
      topGenreSuggestions.appendChild(div);
    });
    topGenreSuggestions.style.display = 'block';
  }

  async function fetchYearSuggestions() {
    const q = topYear.value.trim();
    if (q.length < 1) {
      hideTopSuggestions(topYearSuggestions);
      return;
    }

    const res = await fetch(`/suggest/years?q=${encodeURIComponent(q)}`);
    const data = await res.json();
    const items = data.items || [];
    if (!items.length) {
      hideTopSuggestions(topYearSuggestions);
      return;
    }

    topYearSuggestions.innerHTML = '';
    items.forEach((item) => {
      const div = document.createElement('div');
      div.className = 'suggestion-item';
      div.textContent = `${item.year} (${item.count})`;
      div.addEventListener('click', () => {
        topYear.value = item.year || '';
        hideTopSuggestions(topYearSuggestions);
      });
      topYearSuggestions.appendChild(div);
    });
    topYearSuggestions.style.display = 'block';
  }

  function setRankTopLoading(value) {
    rankTopLoading.style.display = value ? 'flex' : 'none';
    rankTopBtn.disabled = value;
  }

  function showRankTopError(msg) {
    if (!msg) {
      rankTopError.style.display = 'none';
      rankTopError.textContent = '';
      return;
    }
    rankTopError.style.display = 'block';
    rankTopError.textContent = msg;
  }

  async function runRankTop() {
    const genre = topGenre.value.trim();
    const yearText = topYear.value.trim();
    const parsedYear = yearText ? Number(yearText) : 0;
    const parsedTopK = Math.max(1, Number(topK.value || 10));

    if (yearText && (!Number.isInteger(parsedYear) || parsedYear < 0)) {
      showRankTopError('Year must be a positive integer.');
      return;
    }

    if (!Number.isFinite(parsedTopK) || parsedTopK < 1) {
      showRankTopError('Top K must be a positive integer.');
      return;
    }

    showRankTopError('');
    setRankTopLoading(true);
    const requestPayload = {
      genre,
      year: parsedYear || 0,
      content_mode: topMode.value,
      top_k: parsedTopK,
    };

    try {
      const res = await fetch('/api/rank-top', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(requestPayload)
      });
      const data = await res.json();

      if (!res.ok || data.error) {
        rankTopResult.style.display = 'none';
        showRankTopError(data.error || 'Could not rank top movies.');
        return;
      }

      const criteria = data.criteria || {};
      rankTopMeta.textContent = `genre=${criteria.genre || 'any'} | year=${criteria.year ?? 'any'} | content=${criteria.content_mode || 'both'} | candidates=${data.candidate_count || 0}`;

      rankTopRows.innerHTML = '';
      (data.results || []).forEach((row, idx) => {
        const tr = document.createElement('tr');
        tr.innerHTML = `
          <td>${idx + 1}</td>
          <td>${row.title || ''}</td>
          <td>${row.imdb_id || ''}</td>
          <td>${row.content_type || ''}</td>
          <td>${row.year ?? ''}</td>
          <td>${Number(row.rating || 0).toFixed(1)}</td>
          <td>${row.popularity ?? ''}</td>
          <td>${row.genre || ''}</td>
        `;
        rankTopRows.appendChild(tr);
      });

      rankTopResult.style.display = 'block';
      setRankTopAiLoading();
      fetchRankTopAiInsight(requestPayload);
    } catch (err) {
      rankTopResult.style.display = 'none';
      hideRankTopAi();
      showRankTopError('Network error while ranking top movies.');
    } finally {
      setRankTopLoading(false);
    }
  }

  rankTopBtn.addEventListener('click', runRankTop);

  let topGenreTimer = null;
  let topYearTimer = null;
  topGenre.addEventListener('input', () => {
    clearTimeout(topGenreTimer);
    topGenreTimer = setTimeout(fetchGenreSuggestions, 120);
  });

  topYear.addEventListener('input', () => {
    clearTimeout(topYearTimer);
    topYearTimer = setTimeout(fetchYearSuggestions, 120);
  });

  document.addEventListener('click', (e) => {
    if (!topGenreSuggestions.contains(e.target) && e.target !== topGenre) {
      hideTopSuggestions(topGenreSuggestions);
    }
    if (!topYearSuggestions.contains(e.target) && e.target !== topYear) {
      hideTopSuggestions(topYearSuggestions);
    }
  });
</script>
"""


BASIC_CONTENT = """
<section class="hero">
  <h1>Basic Movie Description</h1>
  <p>Look up a title or IMDb ID to get base information and a concise description.</p>
</section>

<section class="card">
  <div class="form-row">
    <input id="basicRef" class="input" placeholder="Title or IMDb ID" />
    <button id="basicBtn" class="btn" type="button">Describe</button>
  </div>
  <div id="basicSuggestions" class="dropdown"></div>
  <div id="basicLoading" class="loading"><span class="spinner"></span><span>Looking up movie...</span></div>
  <div id="basicError" class="error"></div>

  <div id="basicResult" class="result-card">
    <div id="basicMeta" class="meta"></div>
    <div style="padding:14px; color:#334155; line-height:1.55;" id="basicDescription"></div>
    <div class="table-wrap">
      <table>
        <tbody id="basicRows"></tbody>
      </table>
    </div>
    <div id="basicAiBlock" style="display:none; padding:14px; border-top:1px solid #e7edf7; background:#fbfdff;">
      <h3 style="margin:0 0 10px; font-size:16px; color:#0f2855;">AI Description Insight</h3>
      <div id="basicAiLoading" style="display:none; align-items:center; gap:8px; color:#0f2855; margin:0 0 8px;">
        <span class="spinner" aria-hidden="true"></span>
        <span>Generating AI insight...</span>
      </div>
      <p id="basicAiText" style="margin:0; color:#334155; line-height:1.5;"></p>
    </div>
  </div>
</section>
"""


BASIC_SCRIPT = """
<script>
  const basicRef = document.getElementById('basicRef');
  const basicBtn = document.getElementById('basicBtn');
  const basicLoading = document.getElementById('basicLoading');
  const basicError = document.getElementById('basicError');
  const basicResult = document.getElementById('basicResult');
  const basicMeta = document.getElementById('basicMeta');
  const basicRows = document.getElementById('basicRows');
  const basicDescription = document.getElementById('basicDescription');
  const basicSuggestions = document.getElementById('basicSuggestions');
  const basicAiBlock = document.getElementById('basicAiBlock');
  const basicAiLoading = document.getElementById('basicAiLoading');
  const basicAiText = document.getElementById('basicAiText');

  function setBasicAiLoading() {
    basicAiBlock.style.display = 'block';
    basicAiLoading.style.display = 'flex';
    basicAiText.textContent = '';
  }

  function hideBasicAi() {
    basicAiBlock.style.display = 'none';
    basicAiLoading.style.display = 'none';
    basicAiText.textContent = '';
  }

  function renderBasicAi(text) {
    if (!text) {
      hideBasicAi();
      return;
    }
    basicAiBlock.style.display = 'block';
    basicAiLoading.style.display = 'none';
    basicAiText.textContent = text;
  }

  async function fetchBasicAiInsight(reference) {
    try {
      const res = await fetch('/api/basic-description-insight', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ reference })
      });
      const data = await res.json();
      if (!res.ok || data.error) {
        hideBasicAi();
        return;
      }
      renderBasicAi(data.insight || '');
    } catch (err) {
      hideBasicAi();
    }
  }

  function hideBasicSuggestions() {
    basicSuggestions.style.display = 'none';
    basicSuggestions.innerHTML = '';
  }

  async function fetchBasicSuggestions() {
    const q = basicRef.value.trim();
    if (q.length < 2) {
      hideBasicSuggestions();
      return;
    }

    const res = await fetch(`/suggest?q=${encodeURIComponent(q)}`);
    const data = await res.json();
    const items = data.items || [];
    if (!items.length) {
      hideBasicSuggestions();
      return;
    }

    basicSuggestions.innerHTML = '';
    items.forEach((item) => {
      const div = document.createElement('div');
      div.className = 'suggestion-item';
      div.textContent = `${item.title} (${item.year || 'N/A'}) [${item.imdb_id}]`;
      div.addEventListener('click', () => {
        basicRef.value = item.title;
        hideBasicSuggestions();
      });
      basicSuggestions.appendChild(div);
    });

    basicSuggestions.style.display = 'block';
  }

  function setBasicLoading(value) {
    basicLoading.style.display = value ? 'flex' : 'none';
    basicBtn.disabled = value;
  }

  function showBasicError(msg) {
    if (!msg) {
      basicError.style.display = 'none';
      basicError.textContent = '';
      return;
    }
    basicError.style.display = 'block';
    basicError.textContent = msg;
  }

  function addBasicRow(key, value) {
    const tr = document.createElement('tr');
    tr.innerHTML = `<th style="width:220px;">${key}</th><td>${value ?? ''}</td>`;
    basicRows.appendChild(tr);
  }

  async function runBasicLookup() {
    const reference = basicRef.value.trim();
    if (!reference) {
      showBasicError('Please enter a reference.');
      return;
    }

    showBasicError('');
    setBasicLoading(true);

    try {
      const res = await fetch('/api/basic-description', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ reference })
      });
      const data = await res.json();

      if (!res.ok || data.error) {
        basicResult.style.display = 'none';
        showBasicError(data.error || 'Could not resolve this reference.');
        return;
      }

      basicMeta.textContent = `${data.title || 'N/A'} [${data.imdb_id || 'N/A'}]`;
      basicDescription.textContent = data.summary || '';
      basicRows.innerHTML = '';

      addBasicRow('Title', data.title || '');
      addBasicRow('IMDb ID', data.imdb_id || '');
      addBasicRow('Type', data.content_type || '');
      addBasicRow('Year', data.year ?? '');
      addBasicRow('Rating', Number(data.rating || 0).toFixed(1));
      addBasicRow('Popularity', data.popularity ?? '');
      addBasicRow('Genre', data.genre || '');
      addBasicRow('Description', data.description || '');

      basicResult.style.display = 'block';
      setBasicAiLoading();
      fetchBasicAiInsight(reference);
    } catch (err) {
      basicResult.style.display = 'none';
      hideBasicAi();
      showBasicError('Network error while loading description.');
    } finally {
      setBasicLoading(false);
    }
  }

  basicBtn.addEventListener('click', runBasicLookup);
  let basicTimer = null;
  basicRef.addEventListener('input', () => {
    clearTimeout(basicTimer);
    basicTimer = setTimeout(fetchBasicSuggestions, 120);
  });

  basicRef.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') {
      e.preventDefault();
      runBasicLookup();
    }
  });

  document.addEventListener('click', (e) => {
    if (!basicSuggestions.contains(e.target) && e.target !== basicRef) {
      hideBasicSuggestions();
    }
  });
</script>
"""


def _render_page(title: str, active: str, content: str, script: str = "") -> str:
    return render_template_string(
        PAGE_SHELL,
        title=title,
        active=active,
        content=content,
        script=script,
    )


def _movie_title(movie: Any) -> str:
    return str(getattr(movie, "Name", "") or "").strip()


def _movie_id(movie: Any) -> str:
    return str(getattr(movie, "imdb_id", "") or "").strip()


def _movie_year(movie: Any) -> str:
    value = getattr(movie, "Year", "")
    return str(value) if value is not None else ""


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
        out.append(
            {
                "imdb_id": imdb_id,
                "title": _movie_title(movie),
                "year": _movie_year(movie),
            }
        )
        if len(out) >= limit:
            break

    return out


def _suggest_genres(query: str, limit: int = MAX_SUGGESTIONS) -> List[Dict[str, Any]]:
    q = str(query or "").strip().lower()
    if not q:
        return []

    movies = load_movie_universe()
    counts: Dict[str, int] = {}
    labels: Dict[str, str] = {}

    for movie in movies:
        for token in parse_genre_tokens(getattr(movie, "Genre", "")):
            normalized = token.strip().lower()
            if not normalized:
                continue
            counts[normalized] = counts.get(normalized, 0) + 1
            if normalized not in labels:
                labels[normalized] = token.strip()

    matches: List[Dict[str, Any]] = []
    for normalized, count in counts.items():
        if q in normalized:
            matches.append({"genre": labels.get(normalized, normalized.title()), "count": count})

    matches.sort(key=lambda item: (-int(item["count"]), str(item["genre"]).lower()))
    return matches[: max(1, limit)]


def _suggest_years(query: str, limit: int = MAX_SUGGESTIONS) -> List[Dict[str, Any]]:
    q = str(query or "").strip()
    if not q:
        return []

    movies = load_movie_universe()
    counts: Dict[int, int] = {}

    for movie in movies:
        year = safe_int(getattr(movie, "Year", 0), 0)
        if year <= 0:
            continue
        counts[year] = counts.get(year, 0) + 1

    matches: List[Dict[str, Any]] = []
    for year, count in counts.items():
        year_text = str(year)
        if year_text.startswith(q):
            matches.append({"year": year, "count": count})

    matches.sort(key=lambda item: (-int(item["year"]), -int(item["count"])))
    return matches[: max(1, limit)]


def _build_basic_description(reference: str) -> Dict[str, Any]:
    movies = load_movie_universe()
    movie = resolve_reference_movie(reference, movies)
    if movie is None:
        return {"error": f"Reference movie not found: {reference}"}

    data = movie_to_dict(movie)
    title = str(data.get("Name") or "")
    content_type = str(data.get("Type") or "")
    year = safe_int(data.get("Year"), 0)
    rating = safe_float(data.get("Rating"), 0.0)
    popularity = safe_int(data.get("Popularity"), 0)
    genre_text = str(data.get("Genre") or "")
    description = str(data.get("Description") or "")
    genres = parse_genre_tokens(genre_text)

    genre_phrase = ", ".join(genres[:3]) if genres else "mixed genres"
    year_phrase = str(year) if year > 0 else "an unknown year"
    popularity_phrase = f"{popularity:,} votes" if popularity > 0 else "limited vote data"
    summary = (
        f"{title} is a {content_type} from {year_phrase} with {genre_phrase} themes. "
        f"Its IMDb rating is {rating:.1f} based on {popularity_phrase}."
    )

    return {
        "imdb_id": data.get("imdb_id"),
        "title": title,
        "content_type": content_type,
        "year": year,
        "rating": rating,
        "popularity": popularity,
        "genre": genre_text,
        "description": description,
        "summary": summary,
    }


def _recommend_from_reference(reference: str, scope_size: int, top_k: int) -> Dict[str, Any]:
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

    scoped_rows = scope_candidates(source_movie, movies, scope_size=scope_size)
    top_rows = rank_top_from_scoped_pool(scoped_rows, source_movie=source_movie, top_k=top_k)

    output_rows: List[Dict[str, Any]] = []
    for row in top_rows:
        movie = row["movie"]
        output_rows.append(
            {
                "imdb_id": getattr(movie, "imdb_id", None),
                "title": getattr(movie, "Name", None),
                "content_type": getattr(movie, "Type", None),
                "year": safe_int(getattr(movie, "Year", None), 0),
                "rating": safe_float(getattr(movie, "Rating", None), 0.0),
                "popularity": safe_int(getattr(movie, "Popularity", None), 0),
                "genre": getattr(movie, "Genre", None),
                "similarity_score": safe_float(row.get("similarity_score", 0.0), 0.0),
                "composite_score": safe_float(row.get("composite_score", 0.0), 0.0),
            }
        )

    source = movie_to_dict(source_movie)
    return {
        "source_movie": {
            "imdb_id": source.get("imdb_id"),
            "title": source.get("Name"),
            "content_type": source.get("Type"),
            "year": source.get("Year"),
            "rating": source.get("Rating"),
        },
        "total_movies": len(movies),
        "scope_size": len(scoped_rows),
        "top_k": len(output_rows),
        "weights": {
            "scope_size": scope_size,
            "top_k": top_k,
        },
        "results": output_rows,
    }


async def _run_ai_text_insight(prompt: str) -> Dict[str, Any]:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return {"error": "OPENAI_API_KEY is not set"}

    set_default_openai_key(api_key)
    agent = Agent(
        name="Movie web insight agent",
        instructions=(
            "You are a concise movie assistant. "
            "Use only the provided structured data. "
            "Return plain text in 3-6 short sentences."
        ),
        tools=[],
        model="gpt-5-nano",
    )

    result = await Runner.run(agent, input=prompt)
    return {"insight": str(result.final_output).strip()}


@app.route("/", methods=["GET"])
def index() -> str:
    return _render_page(
        title="Movie Search",
        active="home",
        content=HOME_CONTENT,
    )


@app.route("/similar", methods=["GET"])
def similar_page() -> str:
    return _render_page(
        title="Finding Similar Movie",
        active="similar",
        content=SIMILAR_CONTENT,
        script=SIMILAR_SCRIPT,
    )


@app.route("/compare", methods=["GET"])
def compare_page() -> str:
    return _render_page(
        title="Compare Two Movies",
        active="compare",
        content=COMPARE_CONTENT,
        script=COMPARE_SCRIPT,
    )


@app.route("/rank-set", methods=["GET"])
def rank_set_page() -> str:
    return _render_page(
        title="Ranking a Set of Movies",
        active="rank_set",
        content=RANK_SET_CONTENT,
        script=RANK_SET_SCRIPT,
    )


@app.route("/rank-top", methods=["GET"])
def rank_top_page() -> str:
    return _render_page(
        title="Ranking Top Movies",
        active="rank_top",
        content=RANK_TOP_CONTENT,
        script=RANK_TOP_SCRIPT,
    )


@app.route("/basic-description", methods=["GET"])
def basic_page() -> str:
    return _render_page(
        title="Basic Movie Description",
        active="basic",
        content=BASIC_CONTENT,
        script=BASIC_SCRIPT,
    )


@app.route("/suggest", methods=["GET"])
def suggest() -> Any:
    query = request.args.get("q", "")
    return jsonify({"items": _suggest_movies(query)})


@app.route("/suggest/genres", methods=["GET"])
def suggest_genres() -> Any:
    query = request.args.get("q", "")
    return jsonify({"items": _suggest_genres(query)})


@app.route("/suggest/years", methods=["GET"])
def suggest_years() -> Any:
    query = request.args.get("q", "")
    return jsonify({"items": _suggest_years(query)})


@app.route("/api/recommend", methods=["POST"])
def recommend() -> Any:
    payload = request.get_json(silent=True) or {}
    reference = str(payload.get("reference", "")).strip()
    if not reference:
        return jsonify({"error": "reference is required"}), 400

    result = _recommend_from_reference(
        reference=reference,
        scope_size=DEFAULT_SCOPE_SIZE,
        top_k=DEFAULT_TOP_K,
    )
    if result.get("error"):
        return jsonify(result), 404
    return jsonify(result)


@app.route("/api/compare", methods=["POST"])
def compare_api() -> Any:
    payload = request.get_json(silent=True) or {}
    first_reference = str(payload.get("first_reference", "")).strip()
    second_reference = str(payload.get("second_reference", "")).strip()
    if not first_reference or not second_reference:
        return jsonify({"error": "first_reference and second_reference are required"}), 400

    movies = load_movie_universe()
    result = compare_two_movies(first_reference=first_reference, second_reference=second_reference, movies=movies)
    if result.get("error"):
        return jsonify(result), 404

    result["ai_insight"] = None
    return jsonify(result)


@app.route("/api/compare-insight", methods=["POST"])
def compare_insight_api() -> Any:
    payload = request.get_json(silent=True) or {}
    first_reference = str(payload.get("first_reference", "")).strip()
    second_reference = str(payload.get("second_reference", "")).strip()
    if not first_reference or not second_reference:
        return jsonify({"error": "first_reference and second_reference are required"}), 400

    try:
        ai_insight = asyncio.run(
            run_compare_insight_with_agent(
                first_reference=first_reference,
                second_reference=second_reference,
                model="gpt-5-nano",
            )
        )
        if ai_insight.get("error"):
            return jsonify({"error": ai_insight.get("error", "Unable to generate AI insight")}), 502
        return jsonify({"ai_insight": ai_insight})
    except Exception:
        return jsonify({"error": "AI insight is currently unavailable"}), 502


@app.route("/api/similar-insight", methods=["POST"])
def similar_insight_api() -> Any:
    payload = request.get_json(silent=True) or {}
    source_reference = str(payload.get("source_reference", "")).strip()
    candidate_imdb_id = str(payload.get("candidate_imdb_id", "")).strip()
    candidate_title = str(payload.get("candidate_title", "")).strip()

    if not source_reference or (not candidate_imdb_id and not candidate_title):
        return jsonify({"error": "source_reference and candidate movie reference are required"}), 400

    movies = load_movie_universe()
    source_movie = resolve_reference_movie(source_reference, movies)
    candidate_ref = candidate_imdb_id or candidate_title
    candidate_movie = resolve_reference_movie(candidate_ref, movies)

    if source_movie is None or candidate_movie is None:
        return jsonify({"error": "Could not resolve source or candidate movie"}), 404

    source_data = movie_to_dict(source_movie)
    candidate_data = movie_to_dict(candidate_movie)
    similarity_score = safe_float(payload.get("similarity_score", 0.0), 0.0)
    composite_score = safe_float(payload.get("composite_score", 0.0), 0.0)

    prompt = (
        "Explain why the candidate is similar to the source movie. "
        "Mention genre/tone overlap, era/type alignment, and rating/popularity context. "
        "Also comment on what audience might prefer this candidate.\n\n"
        f"Source movie: {source_data}\n"
        f"Candidate movie: {candidate_data}\n"
        f"Scores: similarity_score={similarity_score:.3f}, composite_score={composite_score:.3f}\n"
    )

    try:
        insight = asyncio.run(_run_ai_text_insight(prompt))
        if insight.get("error"):
            return jsonify({"error": insight.get("error")}), 502
        return jsonify({"insight": insight.get("insight", "")})
    except Exception:
        return jsonify({"error": "AI insight is currently unavailable"}), 502


@app.route("/api/rank-set-insight", methods=["POST"])
def rank_set_insight_api() -> Any:
    payload = request.get_json(silent=True) or {}
    references = payload.get("references", [])
    if not isinstance(references, list):
        return jsonify({"error": "references must be a list"}), 400

    normalized_refs = [str(item or "").strip() for item in references if str(item or "").strip()]
    if not normalized_refs:
        return jsonify({"error": "references are required"}), 400

    movies = load_movie_universe()
    report = rank_user_selected_set(movies=movies, references=normalized_refs)

    prompt = (
        "Provide a concise AI insight for this ranked movie set. "
        "Explain why top items rank higher, any trade-offs, and how a user can pick among top 3.\n\n"
        f"Ranking metadata: input_size={report.get('input_size')}, "
        f"resolved_count={report.get('resolved_count')}, "
        f"unresolved={report.get('unresolved')}\n"
        f"Top results: {report.get('results', [])[:10]}"
    )

    try:
        insight = asyncio.run(_run_ai_text_insight(prompt))
        if insight.get("error"):
            return jsonify({"error": insight.get("error")}), 502
        return jsonify({"insight": insight.get("insight", "")})
    except Exception:
        return jsonify({"error": "AI insight is currently unavailable"}), 502


@app.route("/api/rank-top-insight", methods=["POST"])
def rank_top_insight_api() -> Any:
    payload = request.get_json(silent=True) or {}

    genre = str(payload.get("genre", "")).strip() or None
    year_raw = payload.get("year", 0)
    year: Optional[int] = None
    if year_raw not in (None, "", 0):
        try:
            year = int(year_raw)
        except (TypeError, ValueError):
            return jsonify({"error": "year must be an integer"}), 400

    content_mode = str(payload.get("content_mode", "both")).strip().lower() or "both"
    if content_mode not in {"both", "movie", "show"}:
        return jsonify({"error": "content_mode must be one of: both, movie, show"}), 400

    try:
        top_k = int(payload.get("top_k", 10))
    except (TypeError, ValueError):
        return jsonify({"error": "top_k must be an integer"}), 400
    top_k = max(1, top_k)

    movies = load_movie_universe()
    report = rank_top_movies_shows_by_genre_or_year(
        movies=movies,
        genre=genre,
        year=year,
        content_mode=content_mode,
        top_k=top_k,
    )

    prompt = (
        "Provide a concise AI insight for these top-ranked results. "
        "Explain the ranking pattern and highlight why the top picks stand out.\n\n"
        f"Criteria: {report.get('criteria')}\n"
        f"Candidate count: {report.get('candidate_count')}\n"
        f"Top results: {report.get('results', [])[:10]}"
    )

    try:
        insight = asyncio.run(_run_ai_text_insight(prompt))
        if insight.get("error"):
            return jsonify({"error": insight.get("error")}), 502
        return jsonify({"insight": insight.get("insight", "")})
    except Exception:
        return jsonify({"error": "AI insight is currently unavailable"}), 502


@app.route("/api/basic-description-insight", methods=["POST"])
def basic_description_insight_api() -> Any:
    payload = request.get_json(silent=True) or {}
    reference = str(payload.get("reference", "")).strip()
    if not reference:
        return jsonify({"error": "reference is required"}), 400

    base = _build_basic_description(reference)
    if base.get("error"):
        return jsonify(base), 404

    prompt = (
        "Provide a concise AI insight for this movie profile. "
        "Include: who might enjoy it, viewing mood, and one watch-tip.\n\n"
        f"Movie profile: {base}"
    )

    try:
        insight = asyncio.run(_run_ai_text_insight(prompt))
        if insight.get("error"):
            return jsonify({"error": insight.get("error")}), 502
        return jsonify({"insight": insight.get("insight", "")})
    except Exception:
        return jsonify({"error": "AI insight is currently unavailable"}), 502


@app.route("/api/rank-set", methods=["POST"])
def rank_set_api() -> Any:
    payload = request.get_json(silent=True) or {}

    references: List[str] = []
    if isinstance(payload.get("references"), list):
        references = [str(item or "").strip() for item in payload.get("references", []) if str(item or "").strip()]
    elif payload.get("references_csv") is not None:
        references = [
            part.strip()
            for part in str(payload.get("references_csv", "")).split(",")
            if part.strip()
        ]

    if not references:
        return jsonify({"error": "references are required"}), 400

    movies = load_movie_universe()
    result = rank_user_selected_set(movies=movies, references=references)
    return jsonify(result)


@app.route("/api/rank-top", methods=["POST"])
def rank_top_api() -> Any:
    payload = request.get_json(silent=True) or {}

    genre = str(payload.get("genre", "")).strip() or None
    year_raw = payload.get("year", 0)
    year: Optional[int] = None
    if year_raw not in (None, "", 0):
        try:
            year = int(year_raw)
        except (TypeError, ValueError):
            return jsonify({"error": "year must be an integer"}), 400

    content_mode = str(payload.get("content_mode", "both")).strip().lower() or "both"
    if content_mode not in {"both", "movie", "show"}:
        return jsonify({"error": "content_mode must be one of: both, movie, show"}), 400

    try:
        top_k = int(payload.get("top_k", 10))
    except (TypeError, ValueError):
        return jsonify({"error": "top_k must be an integer"}), 400

    top_k = max(1, top_k)

    movies = load_movie_universe()
    result = rank_top_movies_shows_by_genre_or_year(
        movies=movies,
        genre=genre,
        year=year,
        content_mode=content_mode,
        top_k=top_k,
    )
    return jsonify(result)


@app.route("/api/basic-description", methods=["POST"])
def basic_description_api() -> Any:
    payload = request.get_json(silent=True) or {}
    reference = str(payload.get("reference", "")).strip()
    if not reference:
        return jsonify({"error": "reference is required"}), 400

    result = _build_basic_description(reference)
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
    root.geometry("460x200")

    label = tk.Label(
        root,
        text="Press the button to launch the movie recommendation website.",
        wraplength=390,
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
