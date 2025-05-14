import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple, Set, Union, Callable
from collections import defaultdict

from models.base import BaseRecommender
from evaluation.metrics import (
    precision_at_k, recall_at_k, ndcg_at_k, mean_average_precision,
    mean_reciprocal_rank, hit_rate_at_k, diversity_at_k, novelty_at_k, coverage_at_k, calculate_cumulative_hit_rate
)
from utils.logger import get_logger
from utils.helpers import timer

logger = get_logger(__name__)


class Evaluator:
    """
    Evaluator class for recommendation systems.

    This class provides methods to evaluate recommender systems using various metrics,
    comparing multiple models, and analyzing performance across different user segments.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the evaluator.

        Args:
            config: Dictionary with configuration parameters.
        """
        # Get config values
        self.metrics = config.get('evaluation.metrics',
                                  ['precision', 'recall', 'ndcg', 'hit_rate', 'map', 'mrr'])
        self.k_values = config.get('evaluation.k_values', [1, 5, 10, 20])
        self.relevance_threshold = config.get('evaluation.relevance_threshold', 3.5)

        # Initialize metric functions
        self.metric_functions = {
            'precision': precision_at_k,
            'recall': recall_at_k,
            'ndcg': ndcg_at_k,
            'hit_rate': hit_rate_at_k,
            'map': mean_average_precision,
            'mrr': mean_reciprocal_rank,
            'diversity': diversity_at_k,
            'novelty': novelty_at_k,
            'coverage': coverage_at_k
        }

        logger.info(f"Initialized Evaluator with metrics: {self.metrics}, k values: {self.k_values}")

    @timer
    def evaluate(self,
                 recommender: BaseRecommender,
                 test_data: pd.DataFrame,
                 item_data: Optional[pd.DataFrame] = None,
                 user_data: Optional[pd.DataFrame] = None,
                 user_segments: Optional[Dict[str, List[int]]] = None) -> Dict[str, Any]:
        """
        Evaluate a recommender system on test data.

        Args:
            recommender: Recommender model to evaluate.
            test_data: Test data containing user-item interactions.
            item_data: Optional item data for diversity metrics.
            user_data: Optional user data for personalization metrics.
            user_segments: Optional dictionary mapping segment names to lists of user IDs.

        Returns:
            Dictionary with evaluation results.
        """
        logger.info(f"Evaluating {recommender.name} recommender")

        # Extract column names
        user_col = 'customer_key'
        item_col = 'product_key'
        rating_col = 'rating' if 'rating' in test_data.columns else None

        # Group test data by user
        user_groups = test_data.groupby(user_col)

        # Prepare data structures for evaluation
        all_recommendations = []  # List of recommendation lists
        all_relevant_items = []  # List of sets of relevant items
        all_relevance_scores = []  # List of dictionaries mapping items to relevance scores

        # Create a mapping from user IDs to lists of known relevant items
        user_relevant_items = {}
        user_relevance_scores = {}

        for user_id, group in user_groups:
            # Get all items the user has interacted with
            if rating_col:
                # Items with ratings above threshold are considered relevant
                relevance_mask = group[rating_col] >= self.relevance_threshold
                relevant_items = set(group[item_col][relevance_mask].values)

                # Create a mapping from items to relevance scores
                relevance_scores = {
                    item: rating for item, rating in zip(group[item_col], group[rating_col])
                }
            else:
                # All items are considered relevant with implicit feedback
                relevant_items = set(group[item_col].values)
                relevance_scores = {item: 1.0 for item in relevant_items}

            # Store for this user
            user_relevant_items[user_id] = relevant_items
            user_relevance_scores[user_id] = relevance_scores

        # Generate recommendations for all users
        for user_id, relevant_items in user_relevant_items.items():
            # Get recommendations
            recommendations = recommender.recommend(
                user_id,
                n=max(self.k_values),
                exclude_seen=False,  # Include seen items for fair comparison
                seen_items=None
            )

            # Extract just the item IDs
            recommended_items = [item_id for item_id, _ in recommendations]

            # Store for evaluation
            all_recommendations.append(recommended_items)
            all_relevant_items.append(relevant_items)
            all_relevance_scores.append(user_relevance_scores[user_id])

        # Prepare item features for diversity metrics
        item_features = None
        if 'diversity' in self.metrics and item_data is not None:
            # Extract features as sets of feature strings
            item_features = {}

            # Use category and subcategory if available
            cat_cols = [col for col in item_data.columns if 'category' in col.lower()]

            if cat_cols:
                for _, row in item_data.iterrows():
                    item_id = row[item_col]
                    features = set()

                    for col in cat_cols:
                        if pd.notna(row[col]):
                            features.add(f"{col}_{row[col]}")

                    item_features[item_id] = features

        # Prepare item popularity for novelty metrics
        item_popularity = None
        if 'novelty' in self.metrics:
            # Count occurrences in test data
            item_popularity = test_data[item_col].value_counts().to_dict()

        # All items for coverage metrics
        all_items = None
        if 'coverage' in self.metrics:
            all_items = set(test_data[item_col].unique())

        # Calculate metrics
        results = {}

        cumulative_hit_rates = calculate_cumulative_hit_rate(
            all_recommendations, all_relevant_items, self.k_values
        )

        for k, rate in cumulative_hit_rates.items():
            results[f"cumulative_hit_rate@{k}"] = rate

        for metric in self.metrics:
            if metric in self.metric_functions:
                if metric in ['precision', 'recall', 'ndcg', 'hit_rate']:
                    # Calculate at different k values
                    for k in self.k_values:
                        if metric == 'ndcg':
                            # NDCG needs relevance scores
                            metric_value = self._calculate_metric_for_all_users(
                                self.metric_functions[metric],
                                all_recommendations,
                                all_relevance_scores,
                                k=k
                            )
                        else:
                            # Other metrics just need binary relevance
                            metric_value = self._calculate_metric_for_all_users(
                                self.metric_functions[metric],
                                all_recommendations,
                                all_relevant_items,
                                k=k
                            )

                        results[f"{metric}@{k}"] = metric_value

                elif metric == 'map':
                    # MAP calculated for all k values
                    for k in self.k_values:
                        map_value = mean_average_precision(all_recommendations, all_relevant_items, k=k)
                        results[f"map@{k}"] = map_value

                elif metric == 'mrr':
                    # MRR calculated once
                    mrr_value = mean_reciprocal_rank(all_recommendations, all_relevant_items)
                    results["mrr"] = mrr_value

                elif metric == 'diversity' and item_features:
                    # Diversity calculated for all k values
                    for k in self.k_values:
                        diversity_value = diversity_at_k(all_recommendations, item_features, k=k)
                        results[f"diversity@{k}"] = diversity_value

                elif metric == 'novelty' and item_popularity:
                    # Novelty calculated for all k values
                    for k in self.k_values:
                        novelty_value = novelty_at_k(all_recommendations, item_popularity, k=k)
                        results[f"novelty@{k}"] = novelty_value

                elif metric == 'coverage' and all_items:
                    # Coverage calculated for all k values
                    for k in self.k_values:
                        coverage_value = coverage_at_k(all_recommendations, all_items, k=k)
                        results[f"coverage@{k}"] = coverage_value

        # If user segments provided, calculate per-segment metrics
        if user_segments:
            segment_results = {}

            for segment_name, user_ids in user_segments.items():
                segment_recommendations = []
                segment_relevant_items = []
                segment_relevance_scores = []

                # Filter data for this segment
                for user_id in user_ids:
                    if user_id in user_relevant_items:
                        # Get recommendations
                        recommendations = recommender.recommend(
                            user_id,
                            n=max(self.k_values),
                            exclude_seen=False,
                            seen_items=None
                        )

                        # Extract just the item IDs
                        recommended_items = [item_id for item_id, _ in recommendations]

                        # Store for evaluation
                        segment_recommendations.append(recommended_items)
                        segment_relevant_items.append(user_relevant_items[user_id])
                        segment_relevance_scores.append(user_relevance_scores[user_id])

                # Calculate metrics for this segment
                segment_metrics = {}

                for metric in self.metrics:
                    if metric in ['precision', 'recall', 'hit_rate']:
                        # Calculate at different k values
                        for k in self.k_values:
                            metric_value = self._calculate_metric_for_all_users(
                                self.metric_functions[metric],
                                segment_recommendations,
                                segment_relevant_items,
                                k=k
                            )

                            segment_metrics[f"{metric}@{k}"] = metric_value

                    elif metric == 'ndcg':
                        # NDCG needs relevance scores
                        for k in self.k_values:
                            metric_value = self._calculate_metric_for_all_users(
                                self.metric_functions[metric],
                                segment_recommendations,
                                segment_relevance_scores,
                                k=k
                            )

                            segment_metrics[f"ndcg@{k}"] = metric_value

                # Store segment results
                segment_results[segment_name] = segment_metrics

            # Add segment results to overall results
            results['segments'] = segment_results

        logger.info(f"Evaluation complete. Results: {results}")
        return results

    def _calculate_metric_for_all_users(self,
                                        metric_func: Callable,
                                        all_recommendations: List[List[int]],
                                        all_relevance: Union[List[Set[int]], List[Dict[int, float]]],
                                        **kwargs) -> float:
        """
        Calculate a metric for all users and return the average.

        Args:
            metric_func: Metric function to use.
            all_recommendations: List of recommendation lists for all users.
            all_relevance: List of sets of relevant items or dictionaries of relevance scores.
            **kwargs: Additional arguments for the metric function.

        Returns:
            Average metric value.
        """
        if not all_recommendations or not all_relevance:
            return 0.0

        # Check if the metric is for a single user or all users
        if metric_func.__name__ in ['mean_average_precision', 'mean_reciprocal_rank', 'hit_rate_at_k']:
            # These metrics already calculate for all users
            return metric_func(all_recommendations, all_relevance, **kwargs)

        # Calculate metric for each user
        user_metrics = []

        for i, (recs, rel) in enumerate(zip(all_recommendations, all_relevance)):
            try:
                value = metric_func(recs, rel, **kwargs)
                user_metrics.append(value)
            except Exception as e:
                logger.warning(f"Error calculating metric for user {i}: {str(e)}")

        # Return average
        return sum(user_metrics) / len(user_metrics) if user_metrics else 0.0

    def compare_recommenders(self,
                             recommenders: Dict[str, BaseRecommender],
                             test_data: pd.DataFrame,
                             item_data: Optional[pd.DataFrame] = None,
                             user_data: Optional[pd.DataFrame] = None,
                             user_segments: Optional[Dict[str, List[int]]] = None) -> Dict[str, Dict[str, Any]]:
        """
        Compare multiple recommenders on the same test data.

        Args:
            recommenders: Dictionary mapping names to recommender models.
            test_data: Test data containing user-item interactions.
            item_data: Optional item data for diversity metrics.
            user_data: Optional user data for personalization metrics.
            user_segments: Optional dictionary mapping segment names to lists of user IDs.

        Returns:
            Dictionary with evaluation results for each recommender.
        """
        logger.info(f"Comparing {len(recommenders)} recommenders")

        results = {}

        for name, recommender in recommenders.items():
            logger.info(f"Evaluating recommender: {name}")

            # Evaluate this recommender
            recommender_results = self.evaluate(
                recommender,
                test_data,
                item_data,
                user_data,
                user_segments
            )

            # Store results
            results[name] = recommender_results

        return results