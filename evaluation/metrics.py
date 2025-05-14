import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple, Set, Union
from collections import defaultdict
import math

from utils.logger import get_logger

logger = get_logger(__name__)


def precision_at_k(recommendations: List[int], relevant_items: Set[int], k: int = 10) -> float:
    """
    Calculate precision@k for a list of recommendations.

    Precision@k is the proportion of recommended items in the top-k set that are relevant.

    Args:
        recommendations: List of recommended item IDs.
        relevant_items: Set of item IDs that are relevant to the user.
        k: Number of recommendations to consider.

    Returns:
        Precision@k value.
    """
    if not recommendations or not relevant_items or k <= 0:
        return 0.0

    # Consider only the top k recommendations
    top_k = recommendations[:k]

    # Count relevant items in the top k
    num_relevant = sum(1 for item in top_k if item in relevant_items)

    # Calculate precision
    return num_relevant / min(k, len(recommendations))


def recall_at_k(recommendations: List[int], relevant_items: Set[int], k: int = 10) -> float:
    """
    Calculate recall@k for a list of recommendations.

    Recall@k is the proportion of relevant items that are in the top-k recommendations.

    Args:
        recommendations: List of recommended item IDs.
        relevant_items: Set of item IDs that are relevant to the user.
        k: Number of recommendations to consider.

    Returns:
        Recall@k value.
    """
    if not recommendations or not relevant_items or k <= 0:
        return 0.0

    # Consider only the top k recommendations
    top_k = recommendations[:k]

    # Count relevant items in the top k
    num_relevant = sum(1 for item in top_k if item in relevant_items)

    # Calculate recall
    return num_relevant / len(relevant_items) if relevant_items else 0.0


def ndcg_at_k(recommendations: List[int], relevant_items: Dict[int, float], k: int = 10) -> float:
    """
    Calculate normalized discounted cumulative gain (NDCG) at k.

    NDCG measures the ranking quality by comparing the DCG to the ideal DCG where
    recommendations are perfectly ranked by relevance.

    Args:
        recommendations: List of recommended item IDs.
        relevant_items: Dictionary mapping item IDs to relevance scores.
        k: Number of recommendations to consider.

    Returns:
        NDCG@k value.
    """
    if not recommendations or not relevant_items or k <= 0:
        return 0.0

    # Consider only the top k recommendations
    top_k = recommendations[:k]

    # Calculate DCG
    dcg = 0.0
    for i, item in enumerate(top_k):
        if item in relevant_items:
            # Use 0-based index for i, so add 1 for position
            dcg += relevant_items[item] / math.log2(i + 2)

    # Calculate ideal DCG (IDCG)
    # Sort relevant items by relevance score
    sorted_relevant = sorted(relevant_items.values(), reverse=True)
    idcg = 0.0
    for i, rel in enumerate(sorted_relevant[:k]):
        idcg += rel / math.log2(i + 2)

    # Calculate NDCG
    if idcg > 0:
        return dcg / idcg
    else:
        return 0.0


def mean_average_precision(recommendations: List[List[int]], relevant_items: List[Set[int]], k: int = 10) -> float:
    """
    Calculate Mean Average Precision (MAP) at k.

    MAP is the mean of Average Precision scores for a set of users.

    Args:
        recommendations: List of lists of recommended item IDs for each user.
        relevant_items: List of sets of relevant item IDs for each user.
        k: Number of recommendations to consider.

    Returns:
        MAP@k value.
    """
    if not recommendations or not relevant_items:
        return 0.0

    # Calculate average precision for each user
    aps = []
    for user_recs, user_rel in zip(recommendations, relevant_items):
        # Skip users with no relevant items
        if not user_rel:
            continue

        # Consider only the top k recommendations
        top_k = user_recs[:k]

        # Calculate average precision
        hits = 0
        sum_precisions = 0.0

        for i, item in enumerate(top_k):
            if item in user_rel:
                hits += 1
                # Calculate precision at current position
                precision_at_i = hits / (i + 1)
                sum_precisions += precision_at_i

        # Calculate average precision
        ap = sum_precisions / len(user_rel) if user_rel else 0.0
        aps.append(ap)

    # Calculate mean of average precisions
    return sum(aps) / len(aps) if aps else 0.0


def mean_reciprocal_rank(recommendations: List[List[int]], relevant_items: List[Set[int]]) -> float:
    """
    Calculate Mean Reciprocal Rank (MRR).

    MRR is the average of the reciprocal ranks of the first relevant item in each list.

    Args:
        recommendations: List of lists of recommended item IDs for each user.
        relevant_items: List of sets of relevant item IDs for each user.

    Returns:
        MRR value.
    """
    if not recommendations or not relevant_items:
        return 0.0

    # Calculate reciprocal rank for each user
    rrs = []
    for user_recs, user_rel in zip(recommendations, relevant_items):
        # Skip users with no relevant items
        if not user_rel:
            continue

        # Find the first relevant item
        for i, item in enumerate(user_recs):
            if item in user_rel:
                # Calculate reciprocal rank (1-based position)
                rr = 1.0 / (i + 1)
                rrs.append(rr)
                break
        else:
            # No relevant items found
            rrs.append(0.0)

    # Calculate mean of reciprocal ranks
    return sum(rrs) / len(rrs) if rrs else 0.0


def hit_rate_at_k(recommendations: List[List[int]], relevant_items: List[Set[int]], k: int = 10) -> float:
    """
    Calculate Hit Rate at k.

    Hit Rate is the proportion of users for whom at least one relevant item is in the top-k recommendations.

    Args:
        recommendations: List of lists of recommended item IDs for each user.
        relevant_items: List of sets of relevant item IDs for each user.
        k: Number of recommendations to consider.

    Returns:
        Hit Rate@k value.
    """
    if not recommendations or not relevant_items:
        return 0.0

    # Count users with at least one hit
    hits = 0

    for user_recs, user_rel in zip(recommendations, relevant_items):
        # Skip users with no relevant items
        if not user_rel:
            continue

        # Consider only the top k recommendations
        top_k = user_recs[:k]

        # Check if any relevant item is in the top k
        if any(item in user_rel for item in top_k):
            hits += 1

    # Calculate hit rate
    return hits / len(recommendations)


def diversity_at_k(recommendations: List[List[int]], item_features: Dict[int, Set[str]], k: int = 10) -> float:
    """
    Calculate recommendation diversity at k.

    Diversity measures the uniqueness of items in the recommendations based on their features.

    Args:
        recommendations: List of lists of recommended item IDs for each user.
        item_features: Dictionary mapping item IDs to sets of feature strings.
        k: Number of recommendations to consider.

    Returns:
        Diversity@k value.
    """
    if not recommendations or not item_features:
        return 0.0

    # Calculate diversity for each user
    user_diversities = []

    for user_recs in recommendations:
        # Consider only the top k recommendations
        top_k = user_recs[:k]

        # Skip users with too few recommendations
        if len(top_k) <= 1:
            continue

        # Calculate pairwise dissimilarity
        total_dissimilarity = 0.0
        num_pairs = 0

        for i in range(len(top_k)):
            for j in range(i + 1, len(top_k)):
                # Get item features
                item_i = top_k[i]
                item_j = top_k[j]

                if item_i in item_features and item_j in item_features:
                    features_i = item_features[item_i]
                    features_j = item_features[item_j]

                    # Calculate Jaccard distance (1 - Jaccard similarity)
                    if features_i and features_j:
                        intersection = len(features_i.intersection(features_j))
                        union = len(features_i.union(features_j))
                        jaccard_similarity = intersection / union
                        jaccard_distance = 1.0 - jaccard_similarity

                        total_dissimilarity += jaccard_distance
                        num_pairs += 1

        # Calculate average dissimilarity
        if num_pairs > 0:
            user_diversity = total_dissimilarity / num_pairs
            user_diversities.append(user_diversity)

    # Calculate mean diversity
    return sum(user_diversities) / len(user_diversities) if user_diversities else 0.0


def novelty_at_k(recommendations: List[List[int]], item_popularity: Dict[int, int], k: int = 10) -> float:
    """
    Calculate recommendation novelty at k.

    Novelty measures how unusual the recommendations are in terms of item popularity.

    Args:
        recommendations: List of lists of recommended item IDs for each user.
        item_popularity: Dictionary mapping item IDs to popularity scores (e.g., number of interactions).
        k: Number of recommendations to consider.

    Returns:
        Novelty@k value.
    """
    if not recommendations or not item_popularity:
        return 0.0

    # Get total popularity (for normalization)
    total_popularity = sum(item_popularity.values())

    if total_popularity == 0:
        return 0.0

    # Calculate novelty for each user
    user_novelties = []

    for user_recs in recommendations:
        # Consider only the top k recommendations
        top_k = user_recs[:k]

        # Skip users with no recommendations
        if not top_k:
            continue

        # Calculate novelty
        user_novelty = 0.0
        count = 0

        for item in top_k:
            if item in item_popularity:
                # Calculate self-information: -log2(probability)
                prob = item_popularity[item] / total_popularity
                self_info = -math.log2(prob) if prob > 0 else 0.0
                user_novelty += self_info
                count += 1

        # Calculate average novelty
        if count > 0:
            user_novelty /= count
            user_novelties.append(user_novelty)

    # Calculate mean novelty
    return sum(user_novelties) / len(user_novelties) if user_novelties else 0.0


def coverage_at_k(recommendations: List[List[int]], all_items: Set[int], k: int = 10) -> float:
    """
    Calculate catalog coverage at k.

    Coverage measures the proportion of available items that appear in at least one user's recommendations.

    Args:
        recommendations: List of lists of recommended item IDs for each user.
        all_items: Set of all available item IDs.
        k: Number of recommendations to consider.

    Returns:
        Coverage@k value.
    """
    if not recommendations or not all_items:
        return 0.0

    # Collect all unique items in the recommendations
    recommended_items = set()

    for user_recs in recommendations:
        # Consider only the top k recommendations
        top_k = user_recs[:k]

        # Add items to the set
        recommended_items.update(top_k)

    # Calculate coverage
    return len(recommended_items) / len(all_items)