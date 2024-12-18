from surprise import Dataset, Reader, SVD
from surprise.model_selection import cross_validate
import pandas as pd

def build_collaborative_model(ratings, logger):
    """
    Builds and evaluates a collaborative filtering model using Surprise's SVD.
    """
    # Convert ratings data into Surprise format
    reader = Reader(rating_scale=(0.5, 5.0))  # Adjust rating scale based on dataset
    data = Dataset.load_from_df(ratings[["userId", "movieId", "rating"]], reader)

    # Use SVD for matrix factorization
    algo = SVD()

    # Perform cross-validation
    logger.info("Performing cross-validation on SVD model...")
    results = cross_validate(algo, data, measures=["RMSE", "MAE"], cv=5, verbose=False)

    # Log results
    rmse_mean = results["test_rmse"].mean()
    mae_mean = results["test_mae"].mean()
    logger.info(f"Cross-validated RMSE: {rmse_mean:.4f}")
    logger.info(f"Cross-validated MAE: {mae_mean:.4f}")

    return algo, results
