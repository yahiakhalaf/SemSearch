import pandas as pd
import numpy as np
from sklearn.metrics import ndcg_score


def load_ground_truth(csv_path):
    """
    Load ground truth relevance data from a CSV file.

    Expected format:
        query,id_1,id_2,...,id_5
    where missing IDs are NaN.

    Args:
        csv_path (str): Path to the evaluation CSV.

    Returns:
        dict: Mapping from query string to list of relevant document IDs (int).
    """
    df_eval = pd.read_csv(csv_path)
    ground_truth = {}

    for _, row in df_eval.iterrows():
        relevant_ids = [
            int(row[col])
            for col in ["id_1", "id_2", "id_3", "id_4", "id_5"]
            if pd.notna(row[col])
        ]
        ground_truth[row["query"]] = relevant_ids

    return ground_truth


def precision_at_k(retrieved_ids, relevant_ids, k=5):
    """
    Compute Precision@K.

    Args:
        retrieved_ids (list): Ordered list of retrieved document IDs.
        relevant_ids (list): List of relevant document IDs.
        k (int): Cut-off rank.

    Returns:
        float: Precision@K.
    """
    if not retrieved_ids:
        return 0.0

    retrieved_k = retrieved_ids[:k]
    relevant_set = set(relevant_ids)
    hits = sum(1 for doc_id in retrieved_k if doc_id in relevant_set)
    return hits / min(k, len(retrieved_k))


def ndcg_at_k(retrieved_ids, relevant_ids, k=5):
    """
    Compute Normalized Discounted Cumulative Gain (NDCG@K).

    Uses ideal ranking where relevance decreases with position in `relevant_ids`.

    Args:
        retrieved_ids (list): Ordered list of retrieved document IDs.
        relevant_ids (list): List of relevant document IDs (in ideal order).
        k (int): Cut-off rank.

    Returns:
        float: NDCG@K score.
    """
    if not retrieved_ids or not relevant_ids:
        return 0.0

    # Ideal relevance: higher rank = higher relevance
    relevance_map = {doc_id: len(relevant_ids) - i for i, doc_id in enumerate(relevant_ids)}
    retrieved_k = retrieved_ids[:k]
    y_true = np.array([[relevance_map.get(doc_id, 0) for doc_id in retrieved_k]])
    y_score = np.array([[k - i for i in range(len(retrieved_k))]])

    if np.sum(y_true) == 0:
        return 0.0

    return float(ndcg_score(y_true, y_score, k=k))


def average_precision(retrieved_ids, relevant_ids):
    """
    Compute Average Precision (AP) for a single query.

    Args:
        retrieved_ids (list): Ordered list of retrieved document IDs.
        relevant_ids (list): List of relevant document IDs.

    Returns:
        float: Average Precision.
    """
    if not relevant_ids:
        return 0.0

    relevant_set = set(relevant_ids)
    score = 0.0
    hits = 0

    for i, doc_id in enumerate(retrieved_ids):
        if doc_id in relevant_set:
            hits += 1
            score += hits / (i + 1)

    return score / len(relevant_set)