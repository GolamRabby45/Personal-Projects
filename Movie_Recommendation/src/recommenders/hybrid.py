import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def hybrid_recommendations(user_id, movie_title, ratings, movies, algo, embeddings, top_n=10, alpha=0.5):
    """
    Combines collaborative filtering and NLP-based content similarity for hybrid recommendations.
    Parameters:
        - user_id: ID of the user for personalized recommendations.
        - movie_title: Movie title for content-based recommendations.
        - ratings: User-item interaction data.
        - movies: Movies metadata with embeddings.
        - algo: Trained collaborative filtering model (Surprise SVD).
        - embeddings: Textual embeddings for content-based filtering.
        - top_n: Number of recommendations to return.
        - alpha: Weight for blending collaborative (alpha) and content-based (1-alpha) scores.
    """
    # Collaborative Filtering: Predict ratings for all movies for the user
    all_movie_ids = movies["movieId"].unique()
    predicted_ratings = {}
    for movie_id in all_movie_ids:
        try:
            predicted_ratings[movie_id] = algo.predict(user_id, movie_id).est
        except:
            predicted_ratings[movie_id] = 0

    # Normalize collaborative scores
    cf_scores = np.array(list(predicted_ratings.values()))
    cf_scores = (cf_scores - cf_scores.min()) / (cf_scores.max() - cf_scores.min() + 1e-8)

    # Content-Based Filtering: Find movies similar to the given movie title
    idx = movies[movies["title"].str.contains(movie_title, case=False, na=False)].index
    if len(idx) == 0:
        return f"Movie '{movie_title}' not found in the database."
    idx = idx[0]

    movie_embedding = embeddings[idx].reshape(1, -1)
    content_sim = cosine_similarity(movie_embedding, embeddings).flatten()

    # Normalize content-based scores
    content_scores = (content_sim - content_sim.min()) / (content_sim.max() - content_sim.min() + 1e-8)

    # Combine scores
    hybrid_scores = alpha * cf_scores + (1 - alpha) * content_scores

    # Get top-N recommendations
    top_indices = np.argsort(hybrid_scores)[::-1][:top_n]

    # Return movie titles and genres
    return movies.iloc[top_indices][["title", "genres"]]
