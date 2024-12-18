import numpy as np

def precision_at_k(recommended_items, relevant_items, k):
    """
    Precision@K: Fraction of recommended items in the top-K that are relevant.
    """
    recommended_k = recommended_items[:k]
    relevant_set = set(relevant_items)
    precision = len(set(recommended_k) & relevant_set) / len(recommended_k)
    return precision

def recall_at_k(recommended_items, relevant_items, k):
    """
    Recall@K: Fraction of relevant items that are in the top-K recommendations.
    """
    recommended_k = recommended_items[:k]
    relevant_set = set(relevant_items)
    recall = len(set(recommended_k) & relevant_set) / len(relevant_set)
    return recall

def ndcg_at_k(recommended_items, relevant_items, k):
    """
    NDCG@K: Normalized Discounted Cumulative Gain.
    """
    recommended_k = recommended_items[:k]
    dcg = sum([1 / np.log2(idx + 2) if recommended_k[idx] in relevant_items else 0 for idx in range(len(recommended_k))])
    idcg = sum([1 / np.log2(idx + 2) for idx in range(min(len(relevant_items), k))])
    return dcg / idcg if idcg > 0 else 0

def evaluate_recommendations(recommended_items, relevant_items, k_list=[5, 10]):
    """
    Evaluates Precision@K, Recall@K, and NDCG@K for multiple K values.
    """
    metrics = {}
    for k in k_list:
        precision = precision_at_k(recommended_items, relevant_items, k)
        recall = recall_at_k(recommended_items, relevant_items, k)
        ndcg = ndcg_at_k(recommended_items, relevant_items, k)
        metrics[f"Precision@{k}"] = precision
        metrics[f"Recall@{k}"] = recall
        metrics[f"NDCG@{k}"] = ndcg
    return metrics
