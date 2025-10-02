"""Evaluation metrics for retrieval and ranking."""

from typing import Dict, List, Tuple

import numpy as np
from scipy import stats
from sklearn.calibration import calibration_curve
from sklearn.metrics import auc


def ndcg_at_k(relevance_scores: List[float], k: int = 10) -> float:
    """
    Calculate Normalized Discounted Cumulative Gain at k.

    Args:
        relevance_scores: List of relevance scores in ranked order
        k: Cutoff position

    Returns:
        nDCG@k score (0-1, higher is better)
    """
    relevance_scores = np.array(relevance_scores[:k])
    if len(relevance_scores) == 0:
        return 0.0

    # DCG
    discounts = np.log2(np.arange(2, len(relevance_scores) + 2))
    dcg = np.sum(relevance_scores / discounts)

    # IDCG (ideal DCG)
    ideal_scores = np.sort(relevance_scores)[::-1]
    idcg = np.sum(ideal_scores / discounts)

    if idcg == 0:
        return 0.0

    return dcg / idcg


def mrr_at_k(relevance_scores: List[float], k: int = 10) -> float:
    """
    Calculate Mean Reciprocal Rank at k.

    Args:
        relevance_scores: List of relevance scores in ranked order
        k: Cutoff position

    Returns:
        MRR@k score (0-1, higher is better)
    """
    relevance_scores = np.array(relevance_scores[:k])
    for i, score in enumerate(relevance_scores):
        if score > 0:
            return 1.0 / (i + 1)
    return 0.0


def recall_at_k(relevant_ids: set, retrieved_ids: List, k: int = 100) -> float:
    """
    Calculate Recall at k.

    Args:
        relevant_ids: Set of relevant document IDs
        retrieved_ids: List of retrieved document IDs in ranked order
        k: Cutoff position

    Returns:
        Recall@k score (0-1, higher is better)
    """
    if len(relevant_ids) == 0:
        return 0.0

    retrieved_set = set(retrieved_ids[:k])
    num_relevant_retrieved = len(relevant_ids & retrieved_set)
    return num_relevant_retrieved / len(relevant_ids)


def precision_at_k(relevant_ids: set, retrieved_ids: List, k: int = 10) -> float:
    """
    Calculate Precision at k.

    Args:
        relevant_ids: Set of relevant document IDs
        retrieved_ids: List of retrieved document IDs in ranked order
        k: Cutoff position

    Returns:
        Precision@k score (0-1, higher is better)
    """
    if k == 0:
        return 0.0

    retrieved_set = set(retrieved_ids[:k])
    num_relevant_retrieved = len(relevant_ids & retrieved_set)
    return num_relevant_retrieved / k


def expected_calibration_error(
    confidences: np.ndarray, accuracies: np.ndarray, n_bins: int = 10
) -> float:
    """
    Calculate Expected Calibration Error (ECE).

    Args:
        confidences: Predicted confidence scores (0-1)
        accuracies: Binary accuracy (0 or 1)
        n_bins: Number of bins for calibration

    Returns:
        ECE score (0-1, lower is better)
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]

        # Find samples in this bin
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = np.mean(in_bin)

        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(accuracies[in_bin])
            avg_confidence_in_bin = np.mean(confidences[in_bin])
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

    return ece


def kendall_tau(ranking1: List, ranking2: List) -> float:
    """
    Calculate Kendall's Tau correlation between two rankings.

    Args:
        ranking1: First ranking (list of IDs)
        ranking2: Second ranking (list of IDs)

    Returns:
        Kendall's Tau (-1 to 1, higher is better)
    """
    # Create rank dictionaries
    rank1 = {item: i for i, item in enumerate(ranking1)}
    rank2 = {item: i for i, item in enumerate(ranking2)}

    # Find common items
    common_items = set(ranking1) & set(ranking2)
    if len(common_items) < 2:
        return 0.0

    # Get ranks for common items
    ranks1 = [rank1[item] for item in common_items]
    ranks2 = [rank2[item] for item in common_items]

    # Calculate Kendall's Tau
    tau, _ = stats.kendalltau(ranks1, ranks2)
    return tau


def risk_coverage_curve(
    scores: np.ndarray, labels: np.ndarray, n_points: int = 100
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate risk-coverage curve for selective prediction.

    Args:
        scores: Confidence scores (higher = more confident)
        labels: Binary labels (0 or 1)
        n_points: Number of points in the curve

    Returns:
        Tuple of (coverage, accuracy) arrays
    """
    # Sort by confidence (descending)
    sorted_indices = np.argsort(scores)[::-1]
    sorted_labels = labels[sorted_indices]

    coverages = []
    accuracies = []

    for i in range(1, len(sorted_labels) + 1):
        coverage = i / len(sorted_labels)
        accuracy = np.mean(sorted_labels[:i])
        coverages.append(coverage)
        accuracies.append(accuracy)

    # Subsample to n_points
    if len(coverages) > n_points:
        indices = np.linspace(0, len(coverages) - 1, n_points, dtype=int)
        coverages = [coverages[i] for i in indices]
        accuracies = [accuracies[i] for i in indices]

    return np.array(coverages), np.array(accuracies)


def compute_retrieval_metrics(
    query_results: List[Dict],
    k_values: List[int] = [10, 50, 100],
) -> Dict[str, float]:
    """
    Compute comprehensive retrieval metrics for a set of queries.

    Args:
        query_results: List of dicts with keys:
            - 'retrieved_ids': List of retrieved doc IDs
            - 'relevant_ids': Set of relevant doc IDs
            - 'scores': List of relevance scores (0 or 1)
        k_values: List of k values for metrics

    Returns:
        Dictionary of metric names to values
    """
    metrics = {}

    # Compute per-query metrics
    ndcg_scores = {k: [] for k in k_values}
    mrr_scores = {k: [] for k in k_values}
    recall_scores = {k: [] for k in k_values}
    precision_scores = {k: [] for k in k_values}

    for result in query_results:
        retrieved_ids = result["retrieved_ids"]
        relevant_ids = result["relevant_ids"]
        scores = result.get("scores", [1 if rid in relevant_ids else 0 for rid in retrieved_ids])

        for k in k_values:
            ndcg_scores[k].append(ndcg_at_k(scores, k))
            mrr_scores[k].append(mrr_at_k(scores, k))
            recall_scores[k].append(recall_at_k(relevant_ids, retrieved_ids, k))
            precision_scores[k].append(precision_at_k(relevant_ids, retrieved_ids, k))

    # Average across queries
    for k in k_values:
        metrics[f"ndcg@{k}"] = np.mean(ndcg_scores[k])
        metrics[f"mrr@{k}"] = np.mean(mrr_scores[k])
        metrics[f"recall@{k}"] = np.mean(recall_scores[k])
        metrics[f"precision@{k}"] = np.mean(precision_scores[k])

    return metrics

