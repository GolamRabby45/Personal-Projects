from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

def create_text_embeddings(movies, logger):
    """
    Creates text embeddings for movie descriptions or titles using sentence-transformers.
    """
    # Load pre-trained model
    logger.info("Loading sentence-transformers model for embeddings...")
    model = SentenceTransformer('all-MiniLM-L6-v2')  # Compact and efficient

    # Combine title and genres as a text feature (can include descriptions if available)
    movies["text"] = movies["title"] + " " + movies["genres"].fillna("")

    # Generate embeddings
    logger.info("Generating text embeddings...")
    embeddings = model.encode(movies["text"].tolist(), show_progress_bar=True)
    logger.info("Embeddings generated successfully.")

    return movies, embeddings

def recommend_movies_nlp(movie_title, movies, embeddings, top_n=10):
    """
    Recommends movies based on semantic similarity of text embeddings to a given movie.
    """
    # Get the index of the movie that matches the title
    idx = movies[movies["title"].str.contains(movie_title, case=False, na=False)].index
    if len(idx) == 0:
        return f"Movie '{movie_title}' not found in the database."

    idx = idx[0]

    # Compute cosine similarity with all other movies
    movie_embedding = embeddings[idx].reshape(1, -1)
    sim_scores = cosine_similarity(movie_embedding, embeddings).flatten()

    # Sort the movies based on similarity scores
    sim_scores = list(enumerate(sim_scores))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the indices of the top-n similar movies
    top_indices = [i[0] for i in sim_scores[1:top_n+1]]

    # Return the titles and genres of the top-n similar movies
    return movies.iloc[top_indices][["title", "genres"]]
