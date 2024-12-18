from recommenders.hybrid import hybrid_recommendations

def generate_recommendations(user_id, movie_title, ratings, movies, algo, embeddings, alpha=0.5, top_n=10):
    """
    Generates recommendations for a user or based on a movie using the hybrid model.
    """
    recommendations = hybrid_recommendations(
        user_id=user_id,
        movie_title=movie_title,
        ratings=ratings,
        movies=movies,
        algo=algo,
        embeddings=embeddings,
        top_n=top_n,
        alpha=alpha
    )
    return recommendations
