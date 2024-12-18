from evaluation import evaluate_recommendations
from recommenders.hybrid import hybrid_recommendations

def tune_hybrid_model(ratings, movies, algo, embeddings, user_id, relevant_items, logger, alpha_values=[0.1, 0.3, 0.5, 0.7, 0.9]):
    """
    Tunes the blending weight (alpha) for the hybrid recommendation model.
    """
    best_alpha = None
    best_ndcg = -1
    results = {}

    for alpha in alpha_values:
        logger.info(f"Tuning with alpha={alpha}...")

        # Generate hybrid recommendations
        recommendations_hybrid = hybrid_recommendations(
            user_id=user_id,
            movie_title=None,  # Pure collaborative in this case
            ratings=ratings,
            movies=movies,
            algo=algo,
            embeddings=embeddings,
            top_n=10,
            alpha=alpha
        )

        recommended_items = recommendations_hybrid["title"].tolist()
        metrics = evaluate_recommendations(recommended_items, relevant_items, k_list=[5, 10])
        results[alpha] = metrics
        logger.info(f"Metrics for alpha={alpha}: {metrics}")

        # Track the best NDCG
        if metrics["NDCG@10"] > best_ndcg:
            best_ndcg = metrics["NDCG@10"]
            best_alpha = alpha

    logger.info(f"Best alpha={best_alpha} with NDCG@10={best_ndcg:.4f}")
    return best_alpha, results
