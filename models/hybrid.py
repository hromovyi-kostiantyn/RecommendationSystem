import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple, Union
from collections import defaultdict

from models.base import BaseRecommender
from utils.logger import get_logger

logger = get_logger(__name__)


class HybridRecommender(BaseRecommender):
    """
    Hybrid recommendation model that combines multiple recommendation approaches.

    This model implements a weighted combination of various recommendation algorithms,
    allowing the system to leverage the strengths of different approaches.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the hybrid recommender.

        Args:
            config: Dictionary with configuration parameters.
        """
        super().__init__("Hybrid", config)

        # Get config values specific to HybridRecommender
        self.weights = config.get('models.hybrid.weights', {
            'content_based': 0.3,
            'collaborative': 0.7
        })

        # Normalize weights to sum to 1
        total_weight = sum(self.weights.values())
        if total_weight > 0:
            self.weights = {k: v / total_weight for k, v in self.weights.items()}

        # Initialize component recommenders
        self.recommenders = {}

        logger.info(f"Initialized HybridRecommender with weights: {self.weights}")

    def add_recommender(self, name: str, recommender: BaseRecommender, weight: Optional[float] = None) -> None:
        """
        Add a component recommender to the hybrid model.

        Args:
            name: Name of the recommender.
            recommender: Recommender instance.
            weight: Optional weight to override the default from config.
        """
        self.recommenders[name] = recommender

        # Update weight if provided
        if weight is not None:
            self.weights[name] = weight

            # Re-normalize weights
            total_weight = sum(self.weights.values())
            self.weights = {k: v / total_weight for k, v in self.weights.items()}

        logger.info(f"Added recommender '{name}' to hybrid model with weight {self.weights.get(name, 0)}")

    def fit(self, train_data: pd.DataFrame, **kwargs) -> None:
        """
        Train the hybrid recommender.

        This method doesn't actually train the model since component recommenders
        should be fitted separately before being added to the hybrid model.

        Args:
            train_data: Training data (not used directly).
            **kwargs: Additional keyword arguments.
        """
        # Check if we have any recommenders
        if not self.recommenders:
            raise ValueError("No component recommenders added to hybrid model")

        # Check if all recommenders are fitted
        for name, recommender in self.recommenders.items():
            if not recommender.is_fitted:
                logger.warning(f"Component recommender '{name}' is not fitted")

        # Set ID mappings from first recommender if not already set
        if self.user_mapping is None and next(iter(self.recommenders.values())).user_mapping is not None:
            first_recommender = next(iter(self.recommenders.values()))
            self.set_id_mappings(first_recommender.user_mapping, first_recommender.item_mapping)

        self.is_fitted = True
        logger.info(f"Hybrid recommender prepared with {len(self.recommenders)} component recommenders")

    def predict(self, user_id: int, item_id: int) -> float:
        """
        Predict the rating for a user-item pair using a weighted average of component predictions.

        Args:
            user_id: User ID.
            item_id: Item ID.

        Returns:
            Predicted rating.
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")

        # Get predictions from component recommenders
        predictions = {}
        for name, recommender in self.recommenders.items():
            try:
                predictions[name] = recommender.predict(user_id, item_id)
            except Exception as e:
                logger.warning(f"Error getting prediction from {name}: {str(e)}")
                predictions[name] = 0.0

        # Calculate weighted average
        weighted_sum = 0.0
        total_weight = 0.0

        for name, prediction in predictions.items():
            weight = self.weights.get(name, 0.0)
            weighted_sum += weight * prediction
            total_weight += weight

        # Return weighted average or 0.0 if no valid predictions
        if total_weight > 0:
            return weighted_sum / total_weight
        else:
            return 0.0

    def recommend(self, user_id: int, n: int = None, exclude_seen: bool = True,
                  seen_items: Optional[List[int]] = None) -> List[Tuple[int, float]]:
        """
        Generate hybrid recommendations for a user by combining recommendations from component models.

        Args:
            user_id: User ID.
            n: Number of recommendations to generate (default: self.top_k).
            exclude_seen: Whether to exclude items the user has already interacted with.
            seen_items: List of item IDs the user has already interacted with.

        Returns:
            List of (item_id, score) tuples.
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before generating recommendations")

        if n is None:
            n = self.top_k

        # Get recommendations from component recommenders
        all_recommendations = {}
        for name, recommender in self.recommenders.items():
            try:
                recs = recommender.recommend(user_id, n=n * 2, exclude_seen=exclude_seen, seen_items=seen_items)
                all_recommendations[name] = recs
            except Exception as e:
                logger.warning(f"Error getting recommendations from {name}: {str(e)}")
                all_recommendations[name] = []

        # Combine recommendations using score fusion
        combined_scores = defaultdict(float)

        for name, recs in all_recommendations.items():
            weight = self.weights.get(name, 0.0)

            if weight > 0:
                # Normalize scores to 0-1 range
                scores = [score for _, score in recs]
                if scores:
                    max_score = max(scores)
                    min_score = min(scores)
                    score_range = max_score - min_score

                    for item_id, score in recs:
                        if score_range > 0:
                            # Normalize and add weighted score
                            normalized_score = (score - min_score) / score_range
                            combined_scores[item_id] += weight * normalized_score
                        else:
                            # If all scores are the same, use 1.0
                            combined_scores[item_id] += weight

        # Convert to list and sort by score
        recommendations = [(item_id, score) for item_id, score in combined_scores.items()]
        recommendations.sort(key=lambda x: x[1], reverse=True)

        # Return top n
        return recommendations[:n]

    def recommend_for_user_segments(
            self,
            user_segments: Dict[str, List[int]],
            segment_weights: Optional[Dict[str, Dict[str, float]]] = None,
            n: int = None,
            exclude_seen: bool = True,
            user_interactions: Optional[Dict[int, List[int]]] = None
    ) -> Dict[int, List[Tuple[int, float]]]:
        """
        Generate recommendations for users with segment-specific weights.

        This method allows different weighting schemes for different user segments,
        enabling more personalized recommendations.

        Args:
            user_segments: Dictionary mapping segment names to lists of user IDs.
            segment_weights: Dictionary mapping segment names to dictionaries of recommender weights.
            n: Number of recommendations to generate (default: self.top_k).
            exclude_seen: Whether to exclude items the user has already interacted with.
            user_interactions: Dictionary mapping user IDs to lists of item IDs they've interacted with.

        Returns:
            Dictionary mapping user IDs to lists of (item_id, score) tuples.
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before generating recommendations")

        if n is None:
            n = self.top_k

        if segment_weights is None:
            segment_weights = {}

        # Initialize result dictionary
        all_recommendations = {}

        # Process each segment
        for segment_name, users in user_segments.items():
            logger.info(f"Generating recommendations for segment '{segment_name}' with {len(users)} users")

            # Get segment-specific weights, or use default weights
            weights = segment_weights.get(segment_name, self.weights)

            # Temporarily update weights
            original_weights = self.weights.copy()
            self.weights = weights

            # Generate recommendations for each user in the segment
            for user_id in users:
                seen_items = user_interactions.get(user_id, []) if user_interactions else None

                try:
                    recs = self.recommend(user_id, n=n, exclude_seen=exclude_seen, seen_items=seen_items)
                    all_recommendations[user_id] = recs
                except Exception as e:
                    logger.warning(f"Error generating recommendations for user {user_id}: {str(e)}")
                    all_recommendations[user_id] = []

            # Restore original weights
            self.weights = original_weights

        # Process users not in any segment
        all_users = set(self.user_mapping.keys()) if self.user_mapping else set()
        segmented_users = set()
        for users in user_segments.values():
            segmented_users.update(users)

        unsegmented_users = all_users - segmented_users

        if unsegmented_users:
            logger.info(f"Generating recommendations for {len(unsegmented_users)} users not in any segment")

            for user_id in unsegmented_users:
                seen_items = user_interactions.get(user_id, []) if user_interactions else None

                try:
                    recs = self.recommend(user_id, n=n, exclude_seen=exclude_seen, seen_items=seen_items)
                    all_recommendations[user_id] = recs
                except Exception as e:
                    logger.warning(f"Error generating recommendations for user {user_id}: {str(e)}")
                    all_recommendations[user_id] = []

        return all_recommendations

    def _get_model_specific_data(self) -> Dict[str, Any]:
        """
        Get model-specific data for saving.

        Returns:
            Dictionary with model-specific data.
        """
        return {
            'weights': self.weights,
            'recommenders': {name: {'path': f"recommenders/{name}.pkl"} for name in self.recommenders}
        }

    def _restore_model_specific_data(self, data: Dict[str, Any]) -> None:
        """
        Restore model-specific data after loading.

        This method doesn't actually load the component recommenders,
        which should be loaded separately and added using add_recommender().

        Args:
            data: Dictionary with model-specific data.
        """
        self.weights = data['weights']