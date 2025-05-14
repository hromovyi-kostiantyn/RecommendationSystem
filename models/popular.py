import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from collections import Counter

from models.base import BaseRecommender
from utils.logger import get_logger

logger = get_logger(__name__)


class PopularityRecommender(BaseRecommender):
    """
    Popularity-based recommendation model that recommends the most popular items.

    This baseline model ranks items by their popularity (number of interactions),
    optionally weighted by rating values. It serves as a non-personalized baseline.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the popularity recommender.

        Args:
            config: Dictionary with configuration parameters.
        """
        super().__init__("Popularity", config)

        # Get config values specific to PopularityRecommender
        self.consider_ratings = config.get('models.popularity.consider_ratings', True)

        # Initialize popularity scores
        self.item_popularity = {}
        self.sorted_items = []

        logger.info(f"Initialized PopularityRecommender (consider_ratings={self.consider_ratings})")

    def fit(self, train_data: pd.DataFrame) -> None:
        """
        Train the popularity recommender by calculating item popularities.

        Args:
            train_data: DataFrame with training data containing at least user_col and item_col.
        """
        logger.info("Fitting popularity recommender")

        item_col = 'product_key'

        if self.consider_ratings and 'rating' in train_data.columns:
            # Calculate popularity as sum of ratings
            item_popularity = train_data.groupby(item_col)['rating'].sum().to_dict()
            logger.info("Using sum of ratings as popularity measure")
        else:
            # Calculate popularity as interaction count
            item_popularity = train_data[item_col].value_counts().to_dict()
            logger.info("Using interaction count as popularity measure")

        # Store item popularities
        self.item_popularity = item_popularity

        # Create sorted list of items by popularity
        self.sorted_items = sorted(
            self.item_popularity.keys(),
            key=lambda x: self.item_popularity[x],
            reverse=True
        )

        self.is_fitted = True
        logger.info(f"Popularity recommender fitted with {len(self.item_popularity)} items")

    def predict(self, user_id: int, item_id: int) -> float:
        """
        Predict the popularity score for an item (same for all users).

        Args:
            user_id: User ID (not used).
            item_id: Item ID.

        Returns:
            Popularity score for the item, or 0.0 if not in training data.
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")

        # Return the popularity score if item exists
        return self.item_popularity.get(item_id, 0.0)

    def recommend(self, user_id: int, n: int = None, exclude_seen: bool = True,
                  seen_items: Optional[List[int]] = None) -> List[Tuple[int, float]]:
        """
        Recommend the most popular items to a user.

        For efficiency, this overrides the base class method to directly
        use the pre-computed sorted items list.

        Args:
            user_id: User ID (not used for non-personalized recommendations).
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

        # Start with sorted items
        recommendations = []

        # Filter and collect recommendations
        for item in self.sorted_items:
            # Skip seen items if requested
            if exclude_seen and seen_items and item in seen_items:
                continue

            # Add to recommendations
            score = self.item_popularity[item]
            recommendations.append((item, score))

            # Stop once we have enough
            if len(recommendations) >= n:
                break

        return recommendations

    def _get_model_specific_data(self) -> Dict[str, Any]:
        """
        Get model-specific data for saving.

        Returns:
            Dictionary with model-specific data.
        """
        return {
            'item_popularity': self.item_popularity,
            'sorted_items': self.sorted_items,
            'consider_ratings': self.consider_ratings
        }

    def _restore_model_specific_data(self, data: Dict[str, Any]) -> None:
        """
        Restore model-specific data after loading.

        Args:
            data: Dictionary with model-specific data.
        """
        self.item_popularity = data['item_popularity']
        self.sorted_items = data['sorted_items']
        self.consider_ratings = data['consider_ratings']