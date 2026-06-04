import pandas as pd
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import streamlit as st

# =====================================================
# LOAD DATASET
# =====================================================

df = pd.read_csv("movies.csv")

print("Dataset Shape:")
print(df.shape)

print("\nColumns:")
print(df.columns)

# =====================================================
# TASK 2
# TEXT PREPROCESSING
# =====================================================

def clean_text(text):

    text = str(text).lower()

    text = re.sub(r"[^a-zA-Z ]", "", text)

    text = re.sub(r"\s+", " ", text)

    return text.strip()

df["overview"] = df["overview"].fillna("")

df["clean_text"] = df["overview"].apply(
    clean_text
)

# =====================================================
# TASK 3
# TF-IDF
# =====================================================

tfidf = TfidfVectorizer(
    stop_words="english",
    max_features=5000,
    ngram_range=(1, 2)
)

tfidf_matrix = tfidf.fit_transform(
    df["clean_text"]
)

print("\nTF-IDF Shape:")
print(tfidf_matrix.shape)

# =====================================================
# TASK 4
# COSINE SIMILARITY
# =====================================================

similarity_matrix = cosine_similarity(
    tfidf_matrix
)

# =====================================================
# TASK 5
# RECOMMENDATION FUNCTION
# =====================================================

def recommend(movie_name, top_n=5):

    movie_name = movie_name.lower()

    movie_index = None

    for index, title in enumerate(df["title"]):

        if title.lower() == movie_name:

            movie_index = index
            break

    if movie_index is None:
        return []

    similarity_scores = list(
        enumerate(
            similarity_matrix[movie_index]
        )
    )

    similarity_scores = sorted(
        similarity_scores,
        key=lambda x: x[1],
        reverse=True
    )

    similarity_scores = similarity_scores[
        1 : top_n + 1
    ]

    recommendations = []

    for movie in similarity_scores:

        recommendations.append(
            df.iloc[movie[0]]["title"]
        )

    return recommendations

# =====================================================
# STREAMLIT UI
# =====================================================

st.title("Movie Recommendation System")

movie = st.selectbox(
    "Select Movie",
    sorted(df["title"].unique())
)

if st.button("Recommend"):

    results = recommend(movie)

    st.subheader("Recommended Movies")

    for item in results:

        st.write(item)