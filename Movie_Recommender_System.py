import ast
import os
import pickle
import re

import numpy as np
import pandas as pd
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# same files as your v1
movies = pd.read_csv("tmdb_5000_movies.csv")
credits = pd.read_csv("tmdb_5000_credits.csv")

# merged on id not on title because title is not unique
# in movies file id is called id and in credits it is movie_id
movies = movies.merge(credits, left_on="id", right_on="movie_id", suffixes=("", "_c"))
# after merge title from credits becomes title_c, we keep title from movies file
movies["movie_id"] = movies["id"]
movies["title"] = movies["title"].fillna(movies["original_title"])

# we will keep these columns only
movies = movies[["movie_id", "title", "overview", "genres", "keywords", "cast", "crew"]]

# checking missing values
# print(movies.isnull().sum())
# droped null overview movies because there are only 3 which doesnt affect much
movies.dropna(subset=["overview"], inplace=True)

# reset index so similarity index and dataframe index will be same
movies.reset_index(drop=True, inplace=True)

# as i noticed that genres column data is in weird form so we will change it to ['Action', 'Adventure']
import ast
def convert(obj):
    L = []
    for i in ast.literal_eval(obj):
        L.append(i["name"])
    return L

movies["genres"] = movies["genres"].apply(convert)

# i will do same thing with the keywords column as i done with genres
movies["keywords"] = movies["keywords"].apply(convert)

# we only taking the first three name from the cast because there are too many actors and first three are main
def convertcast(obj):
    L = []
    counter = 0
    for i in ast.literal_eval(obj):
        if counter != 3:
            L.append(i["name"])
            counter += 1
        else:
            break
    return L

movies["cast"] = movies["cast"].apply(convertcast)

# i only want the director name so i will fetch director name only
def fetch_director(obj):
    L = []
    for i in ast.literal_eval(obj):
        if i["job"] == "Director":
            L.append(i["name"])
            break
    return L

movies["crew"] = movies["crew"].apply(fetch_director)

# overview is string so we clean it and split into list
# using re so comma and fullstop will not create different words like century, and century
def clean_overview(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return text.split()

movies["overview"] = movies["overview"].apply(clean_overview)

# now we will remove the spacing from the names so model dont confuse with same first name
# like Sam Worthington and Sam Mendes
movies["genres"] = movies["genres"].apply(lambda x: [i.replace(" ", "") for i in x])
movies["keywords"] = movies["keywords"].apply(lambda x: [i.replace(" ", "") for i in x])
movies["cast"] = movies["cast"].apply(lambda x: [i.replace(" ", "") for i in x])
movies["crew"] = movies["crew"].apply(lambda x: [i.replace(" ", "") for i in x])

# make all words lower because we already removed spaces and difference of capital small should not matter
movies["genres"] = movies["genres"].apply(lambda x: [i.lower() for i in x])
movies["keywords"] = movies["keywords"].apply(lambda x: [i.lower() for i in x])
movies["cast"] = movies["cast"].apply(lambda x: [i.lower() for i in x])
movies["crew"] = movies["crew"].apply(lambda x: [i.lower() for i in x])

# now i will make tags column from overview, genres, keywords, cast and crew
# giving more weight to genres and cast and crew by adding them 2 times
# because overview has many words and it will hide the genres and cast
def make_tags(row):
    overview = row["overview"]
    genres = row["genres"] * 2
    keywords = row["keywords"]
    cast = row["cast"] * 2
    crew = row["crew"] * 2
    all_tags = overview + genres + keywords + cast + crew
    return all_tags

movies["tags"] = movies.apply(make_tags, axis=1)

# now we only need three columns
new_df = movies[["movie_id", "title", "tags"]].copy()

# now i will make this tags column again to string and in lower case
new_df["tags"] = new_df["tags"].apply(lambda x: " ".join(x))
new_df["tags"] = new_df["tags"].apply(lambda x: x.lower())

# stemming so words with same meaning become one word
# like loved, loving, loves all become love
# we do stemming only on tags string, cast and crew are already lower and without space
ps = PorterStemmer()

def stem(text):
    y = []
    for i in text.split():
        y.append(ps.stem(i))
    return " ".join(y)

new_df["tags"] = new_df["tags"].apply(stem)

# now i will use tfidf instead of countvectorizer because overview has many common words
# tfidf will give more importance to rare words like actor name or director
from sklearn.feature_extraction.text import TfidfVectorizer
cv = TfidfVectorizer(max_features=5000, stop_words="english", ngram_range=(1, 2), min_df=2, max_df=0.8)

vectors = cv.fit_transform(new_df["tags"]).toarray()
print(vectors.shape)

# now i will calculate cosine similarity
# similarity will be between 0 and 1 only because tfidf vectors are not negative
# so we can use float32 to save memory, it will make file half in size
from sklearn.metrics.pairwise import cosine_similarity
similarity = cosine_similarity(vectors).astype(np.float32)
print(similarity.shape, similarity.dtype)

# saving for streamlit app so app will load fast and not calculate again
os.makedirs("artifacts", exist_ok=True)
import pickle
pickle.dump(new_df[["movie_id", "title"]], open("artifacts/movies_list.pkl", "wb"))
import numpy as np
np.savez_compressed("artifacts/similarity.npz", similarity=similarity)
print("saved artifacts/movies_list.pkl and artifacts/similarity.npz")

# now i will make a function to recommend 5 similar movies given a movie name
def recommend(movie):
    movie = movie.lower().strip()
    # finding movie index, using lower so Batman Begins and batman begins both work
    matches = new_df[new_df["title"].str.lower() == movie]
    if matches.empty:
        # if not found try to search like contains
        contains = new_df[new_df["title"].str.lower().str.contains(movie, na=False)]
        if len(contains) > 0:
            print("Did you mean:", ", ".join(contains["title"].head(3).tolist()))
        else:
            print("Movie not found")
        return
    movie_index = matches.index[0]
    distances = similarity[movie_index]
    movie_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]
    for i in movie_list:
        print(new_df.iloc[i[0]].title)

# quick testing like in v1
recommend("Batman Begins")
recommend("Avatar")
recommend("Spectre")
