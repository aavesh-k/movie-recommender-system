# Movie Recommender System

Live Demo: **https://mrs-aavesh-k.streamlit.app**

Content-based movie recommender built on the TMDB 5000 dataset. It suggests 5 similar movies from overview, genres, keywords, cast and director using TF-IDF and cosine similarity.

## Features
* Search any of 4800 movies
* 5 instant recommendations with posters from TMDB
* Fast ranking with TF-IDF (1-2 grams) and cosine similarity
* Offline fallback if API key is missing

## How it works
1. Merge `tmdb_5000_movies.csv` and `tmdb_5000_credits.csv` on `id`
2. Keep `genres`, `keywords`, top 3 cast and director
3. Clean overview text and remove spaces in names
4. Build `tags` column with weighting for genres, cast and crew
5. Stem and vectorize with `TfidfVectorizer(max_features=5000)`
6. Compute cosine similarity and save as `similarity.npz` (float32)

## Project Structure
```
.
├── app.py                         # Streamlit app
├── Movie_Recommender_System.py      # training pipeline
├── artifacts/
│   ├── movies_list.pkl
│   └── similarity.npz
├── requirements.txt
└── .streamlit/secrets.toml          # TMDB_API_KEY
```

## Run Locally
```powershell
# 1. create venv
python -m venv venv
.\venv\Scripts\Activate.ps1

# 2. install
pip install -r requirements.txt

# 3. build artifacts (only if you change data or params)
python Movie_Recommender_System.py

# 4. run app
streamlit run app.py
```

Add your TMDB key in `.streamlit/secrets.toml`:
```toml
TMDB_API_KEY = "your_32char_key"
```
Get a free key from https://www.themoviedb.org/settings/api (choose Developer, use dummy app values).

## Deploy
Push to GitHub and import the repo on https://share.streamlit.io with main file `app.py`. Add `TMDB_API_KEY` in App Settings → Secrets.

## Dataset
TMDB 5000 from Kaggle: `tmdb_5000_movies.csv` and `tmdb_5000_credits.csv`.
