import concurrent.futures
import os
import pickle

import numpy as np
import requests
import streamlit as st

# load movies and similarity made by v2 file
@st.cache_resource
def load_data():
    m = pickle.load(open("artifacts/movies_list.pkl", "rb"))
    if os.path.exists("artifacts/similarity.npz"):
        s = np.load("artifacts/similarity.npz")["similarity"]
    else:
        s = pickle.load(open("artifacts/similarity.pkl", "rb"))
    return m, s

movies, similarity = load_data()

# get api key from secrets.toml
def get_api_key():
    try:
        key = st.secrets.get("TMDB_API_KEY", "")
        if key:
            return key
    except:
        pass
    return os.getenv("TMDB_API_KEY", "")

api_key = get_api_key()

# recommend function same as v2 file
# using argpartition is faster than sorting all 4800 movies, we only need top 5
def recommend(movie):
    movie = movie.lower().strip()
    matches = movies[movies["title"].str.lower() == movie]
    if matches.empty:
        contains = movies[movies["title"].str.lower().str.contains(movie, na=False)]
        hint = ""
        if len(contains) > 0:
            hint = " Did you mean: " + ", ".join(contains["title"].head(3).tolist()) + "?"
        st.error("Movie not found." + hint)
        return []
    movie_index = matches.index[0]
    distances = similarity[movie_index]
    # argpartition to get top 6 (first one is the movie itself)
    top_idx = np.argpartition(distances, -6)[-6:]
    top_idx = top_idx[np.argsort(distances[top_idx])][::-1]
    # remove the movie itself
    top_idx = [i for i in top_idx if i != movie_index][:5]
    result = []
    for i in top_idx:
        result.append({
            "movie_id": int(movies.iloc[i].movie_id),
            "title": movies.iloc[i].title
        })
    return result

# simple cache for posters so we dont call api again for same movie
poster_cache = {}

def fetch_poster(movie_id, title):
    if not api_key:
        return None, "nokey"
    # return from cache if we already fetched it
    if movie_id in poster_cache:
        return poster_cache[movie_id]
    try:
        from requests.adapters import HTTPAdapter
        from urllib3.util.retry import Retry
        sess = requests.Session()
        retry = Retry(total=3, backoff_factor=0.6, status_forcelist=[429, 500, 502, 503, 504], allowed_methods=["GET"])
        sess.mount("https://", HTTPAdapter(max_retries=retry, pool_connections=10, pool_maxsize=10))
        r = sess.get(f"https://api.themoviedb.org/3/movie/{movie_id}", params={"api_key": api_key, "language": "en-US"}, timeout=8)
        if r.status_code == 404:
            s = sess.get("https://api.themoviedb.org/3/search/movie", params={"api_key": api_key, "query": title, "page": 1}, timeout=8)
            if s.status_code == 200 and s.json().get("results"):
                top = s.json()["results"][0]
                path = top.get("poster_path")
                if path:
                    res = ("https://image.tmdb.org/t/p/w500" + path, "ok")
                    poster_cache[movie_id] = res
                    return res
                else:
                    return None, "noposter"
            return None, "error"
        if r.status_code != 200:
            return None, "error"
        path = r.json().get("poster_path")
        if path:
            res = ("https://image.tmdb.org/t/p/w500" + path, "ok")
            poster_cache[movie_id] = res
            return res
        else:
            return None, "noposter"
    except:
        return None, "error"

# page title
st.set_page_config(page_title="Movie Recommender System", layout="wide")
st.title("Movie Recommender System")

if not api_key:
    st.info("Running offline (titles only). Add TMDB_API_KEY to .streamlit/secrets.toml to enable posters.")

# dropdown for movies
selected = st.selectbox("Pick a movie", movies["title"].values)

if st.button("Recommend"):
    recs = recommend(selected)
    if len(recs) > 0:
        # fetch all 5 posters at same time, much faster than one by one
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as exe:
            future_to_rec = {exe.submit(fetch_poster, r["movie_id"], r["title"]): r for r in recs}
            poster_map = {}
            for fut in concurrent.futures.as_completed(future_to_rec):
                rec = future_to_rec[fut]
                try:
                    poster_map[rec["movie_id"]] = fut.result()
                except:
                    poster_map[rec["movie_id"]] = (None, "error")
        cols = st.columns(5, gap="medium", vertical_alignment="top")
        for c, rec in zip(cols, recs):
            with c:
                if api_key:
                    url, reason = poster_map.get(rec["movie_id"], (None, "error"))
                    if url:
                        st.image(url, use_container_width=True)
                    elif reason == "noposter":
                        st.caption("no poster on TMDB")
                    elif reason == "nokey":
                        pass
                    else:
                        st.caption("poster failed to load, press Recommend again to retry")
                st.markdown(f"<p style='text-align:center; font-weight:600'>{rec['title']}</p>", unsafe_allow_html=True)
