from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

def build_content_based_model(movies, logger):
    """
    Builds a content-based recommendation model using genres and titles.
    """
    # Combine 'title' and 'genres' into a single text feature
    movies["content"] = movies["title"] + " " + movies["genres"].fillna("")

    # Convert text content into TF-IDF vectors
    logger.info("Creating TF-IDF matrix for content-based filtering...")
    tfidf = TfidfVectorizer(stop_words="english")
    tfidf_matrix = tfidf.fit_transform(movies["content"])

    # Compute cosine similarity matrix
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    logger.info("Cosine similarity matrix created.")

    return movies, cosine_sim

def recommend_movies(movie_title, movies, cosine_sim, top_n=10):
    """
    Recommends movies based on content similarity to a given movie.
    """
    # Get the index of the movie that matches the title
    idx = movies[movies["title"].str.contains(movie_title, case=False, na=False)].index
    if len(idx) == 0:
        return f"Movie '{movie_title}' not found in the database."

    idx = idx[0]

    # Get pairwise similarity scores for all movies with the given movie
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort the movies based on similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the indices of the top-n similar movies
    top_indices = [i[0] for i in sim_scores[1:top_n+1]]

    # Return the titles of the top-n similar movies
    return movies.iloc[top_indices][["title", "genres"]]
