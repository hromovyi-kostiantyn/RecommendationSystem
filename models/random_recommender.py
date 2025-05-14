import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
import random

from models.base import BaseRecommender
from utils.logger import get_logger

logger = get_logger(__name__)


class RandomRecommender(BaseRecommender):
    """
    Random recommendation model that generates random recommendations.

    This baseline model assigns random scores to items, which is useful
    as a comparison baseline for more sophisticated algorithms.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the random recommender.

        Args:
            config: Dictionary with configuration parameters.
        """
        super().__init__("Random", config)

        # Get config values specific to RandomRecommender
        self.random_seed = config.get('models.random.seed', 42)
        self.random_state = np.random.RandomState(self.random_seed)

        # Initialize data containers
        self.items = []
        self.min_rating = 1.0
        self.max_rating = 5.0

        logger.info(f"Initialized RandomRecommender with seed {self.random_seed}")

    def fit(self, train_data: pd.DataFrame) -> None:
        """
        'Train' the random recommender by storing the available items.

        Args:
            train_data: DataFrame with training data containing at least user_col and item_col.
        """
        # Store list of unique items
        item_col = 'product_key'
        self.items = list(train_data[item_col].unique())

        # Store rating bounds if available
        if 'rating' in train_data.columns:
            self.min_rating = train_data['rating'].min()
            self.max_rating = train_data['rating'].max()

        # Set random state
        random.seed(self.random_seed)

        self.is_fitted = True
        logger.info(f"Random recommender fitted with {len(self.items)} unique items")

    def predict(self, user_id: int, item_id: int) -> float:
        """
        Generate a random prediction for a user-item pair.

        Args:
            user_id: User ID.
            item_id: Item ID.

        Returns:
            Random rating between min_rating and max_rating.
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")

        # Generate a random rating within the observed range
        return self.random_state.uniform(self.min_rating, self.max_rating)

    def recommend(self, user_id: int, n: int = None, exclude_seen: bool = True,
                  seen_items: Optional[List[int]] = None) -> List[Tuple[int, float]]:
        """
        Generate random recommendations for a user.

        For efficiency, this overrides the base class method to directly
        sample random items rather than scoring all items.

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

        # Create a copy of available items
        available_items = self.items.copy()

        # Exclude seen items if requested
        if exclude_seen and seen_items:
            available_items = [item for item in available_items if item not in seen_items]

        # If we have fewer items than requested, use all available
        n = min(n, len(available_items))

        if n == 0:
            logger.warning(f"No available items for user {user_id}")
            return []

        # Randomly sample items
        selected_items = random.sample(available_items, n)

        # Generate random scores for selected items
        recommendations = []
        for item in selected_items:
            score = self.random_state.uniform(self.min_rating, self.max_rating)
            recommendations.append((item, score))

        # Sort by score for consistency with other recommenders
        recommendations.sort(key=lambda x: x[1], reverse=True)

        return recommendations

    def _get_model_specific_data(self) -> Dict[str, Any]:
        """
        Get model-specific data for saving.

        Returns:
            Dictionary with model-specific data.
        """
        return {
            'items': self.items,
            'min_rating': self.min_rating,
            'max_rating': self.max_rating,
            'random_seed': self.random_seed
        }

    def _restore_model_specific_data(self, data: Dict[str, Any]) -> None:
        """
        Restore model-specific data after loading.

        Args:
            data: Dictionary with model-specific data.
        """
        self.items = data['items']
        self.min_rating = data['min_rating']
        self.max_rating = data['max_rating']
        self.random_seed = data['random_seed']
        self.random_state = np.random.RandomState(self.random_seed)