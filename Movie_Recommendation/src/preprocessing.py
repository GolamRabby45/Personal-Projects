import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def preprocess_data(ratings_file, movies_file, logger):
    # Load datasets
    ratings = pd.read_csv(ratings_file)
    movies = pd.read_csv(movies_file)

    # Check for missing values
    logger.info(f"Missing values in ratings:\n{ratings.isnull().sum()}")
    logger.info(f"Missing values in movies:\n{movies.isnull().sum()}")

    # Drop missing values if any
    ratings.dropna(inplace=True)
    movies.dropna(inplace=True)

    # Merge ratings and movies on movieId
    merged_data = pd.merge(ratings, movies, on="movieId")
    logger.info(f"Merged data shape: {merged_data.shape}")

    return ratings, movies, merged_data

def explore_data(merged_data, logger):
    # Distribution of ratings
    plt.figure(figsize=(8, 6))
    sns.countplot(x="rating", data=merged_data, palette="muted")
    plt.title("Distribution of Ratings")
    plt.xlabel("Rating")
    plt.ylabel("Count")
    plt.savefig("../outputs/rating_distribution.png")
    plt.close()
    logger.info("Saved rating distribution plot to ../outputs/rating_distribution.png")

    # Top movies with most ratings
    top_movies = (
        merged_data.groupby("title")["rating"]
        .count()
        .sort_values(ascending=False)
        .head(10)
    )
    logger.info(f"Top 10 movies with most ratings:\n{top_movies}")

    # Active users with most ratings
    top_users = (
        merged_data.groupby("userId")["rating"]
        .count()
        .sort_values(ascending=False)
        .head(10)
    )
    logger.info(f"Top 10 users with most ratings:\n{top_users}")

    return top_movies, top_users
