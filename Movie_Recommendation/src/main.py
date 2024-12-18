import os
import pandas as pd
from utils import setup_logger
from preprocessing import preprocess_data, explore_data
from recommenders.collaborative import build_collaborative_model
from recommenders.content_based import build_content_based_model, recommend_movies
from recommenders.nlp_embeddings import create_text_embeddings, recommend_movies_nlp
from recommenders.hybrid import hybrid_recommendations
from evaluation import evaluate_recommendations

if __name__ == "__main__":
    logger = setup_logger()

    # File paths
    ratings_file = "../data/ratings.csv"
    movies_file = "../data/movies.csv"

    # Check if dataset files exist
    if not os.path.exists(ratings_file):
        logger.error(f"Ratings file not found: {ratings_file}")
        exit(1)
    if not os.path.exists(movies_file):
        logger.error(f"Movies file not found: {movies_file}")
        exit(1)

    # Preprocess datasets
    try:
        ratings, movies, merged_data = preprocess_data(ratings_file, movies_file, logger)
        logger.info(f"Preprocessed ratings: {ratings.shape}, movies: {movies.shape}")
        logger.info(f"Merged data shape: {merged_data.shape}")
    except Exception as e:
        logger.error(f"Error during preprocessing: {e}")
        exit(1)

    # Exploratory analysis
    try:
        explore_data(merged_data, logger)
    except Exception as e:
        logger.error(f"Error during exploratory analysis: {e}")
        exit(1)

    # Collaborative filtering
    try:
        algo, results = build_collaborative_model(ratings, logger)
    except Exception as e:
        logger.error(f"Error during collaborative filtering: {e}")
        exit(1)

    # NLP embeddings
    try:
        movies, embeddings = create_text_embeddings(movies, logger)
    except Exception as e:
        logger.error(f"Error during NLP embeddings generation: {e}")
        exit(1)

    # Hybrid recommendations
    try:
        user_id = 1
        movie_title = "Toy Story (1995)"
        recommendations_hybrid = hybrid_recommendations(
            user_id=user_id,
            movie_title=movie_title,
            ratings=ratings,
            movies=movies,
            algo=algo,
            embeddings=embeddings,
            top_n=10,
            alpha=0.6
        )
        logger.info(f"Hybrid recommendations for user {user_id} and '{movie_title}':\n{recommendations_hybrid}")

        # Evaluate the hybrid model
        relevant_items = ratings[ratings["userId"] == user_id]["movieId"].tolist()
        recommended_items = recommendations_hybrid["title"].tolist()
        metrics = evaluate_recommendations(recommended_items, relevant_items, k_list=[5, 10])
        logger.info(f"Evaluation Metrics:\n{metrics}")
    except Exception as e:
        logger.error(f"Error during hybrid recommendations: {e}")
        exit(1)
