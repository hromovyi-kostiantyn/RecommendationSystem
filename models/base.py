from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Any, Tuple
import os
import pickle

from utils.logger import get_logger
from utils.helpers import timer, save_object, load_object

logger = get_logger(__name__)


class BaseRecommender(ABC):
    """
    Abstract base class for all recommendation models.

    This class defines the interface that all recommendation models should implement,
    and provides common functionality for saving/loading models and generating recommendations.
    """

    def __init__(self, name: str, config: Dict[str, Any]):
        """
        Initialize the base recommender.

        Args:
            name: Name of the recommender.
            config: Dictionary with configuration parameters.
        """
        self.name = name
        self.config = config
        self.top_k = config.get('models.base.top_k', 10)
        self.user_mapping = None
        self.item_mapping = None
        self.reverse_user_mapping = None
        self.reverse_item_mapping = None
        self.is_fitted = False
        logger.info(f"Initialized {self.name} recommender")

    def set_id_mappings(self, user_mapping: Dict, item_mapping: Dict):
        """
        Set mappings between original IDs and internal indices.

        Args:
            user_mapping: Dictionary mapping user IDs to indices.
            item_mapping: Dictionary mapping item IDs to indices.
        """
        self.user_mapping = user_mapping
        self.item_mapping = item_mapping

        # Create reverse mappings
        self.reverse_user_mapping = {v: k for k, v in user_mapping.items()}
        self.reverse_item_mapping = {v: k for k, v in item_mapping.items()}

        logger.debug(f"Set ID mappings with {len(user_mapping)} users and {len(item_mapping)} items")

    @abstractmethod
    def fit(self, train_data: Any) -> None:
        """
        Train the recommendation model.

        Args:
            train_data: Training data in the format required by the specific model.
        """
        pass

    @abstractmethod
    def predict(self, user_id: int, item_id: int) -> float:
        """
        Predict the rating or score for a specific user-item pair.

        Args:
            user_id: User ID.
            item_id: Item ID.

        Returns:
            Predicted rating or score.
        """
        pass

    def recommend(self, user_id: int, n: int = None, exclude_seen: bool = True,
                  seen_items: Optional[List[int]] = None) -> List[Tuple[int, float]]:
        """
        Generate recommendations for a user.

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

        # Map user ID to internal index if mapping exists
        if self.user_mapping is not None and user_id in self.user_mapping:
            user_idx = self.user_mapping[user_id]
        else:
            user_idx = user_id

        # Get all items
        all_items = list(
            self.reverse_item_mapping.keys() if self.reverse_item_mapping else range(self._get_num_items()))

        # Exclude seen items if requested
        if exclude_seen:
            if seen_items is not None:
                # Convert seen_items to internal indices if mapping exists
                if self.item_mapping is not None:
                    seen_idx = [self.item_mapping[item] for item in seen_items if item in self.item_mapping]
                else:
                    seen_idx = seen_items

                # Filter out seen items
                candidate_items = [item for item in all_items if item not in seen_idx]
            else:
                # If no seen_items provided, use all items
                candidate_items = all_items
        else:
            candidate_items = all_items

        # Generate predictions for all candidate items
        predictions = []
        for item_idx in candidate_items:
            # Map internal index to original item ID for prediction
            if self.reverse_item_mapping is not None:
                item_id_for_predict = self.reverse_item_mapping[item_idx]
            else:
                item_id_for_predict = item_idx

            score = self.predict(user_idx, item_id_for_predict)
            predictions.append((item_idx, score))

        # Sort by score
        predictions.sort(key=lambda x: x[1], reverse=True)

        # Return top n
        top_predictions = predictions[:n]

        # Convert back to original item IDs if mapping exists
        if self.reverse_item_mapping is not None:
            return [(self.reverse_item_mapping[item_idx], score) for item_idx, score in top_predictions]
        else:
            return top_predictions

    def recommend_for_all_users(self, n: int = None, exclude_seen: bool = True,
                                user_interactions: Optional[Dict[int, List[int]]] = None) -> Dict[
        int, List[Tuple[int, float]]]:
        """
        Generate recommendations for all users.

        Args:
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

        all_recommendations = {}

        # Get all users
        all_users = list(
            self.reverse_user_mapping.keys() if self.reverse_user_mapping else range(self._get_num_users()))

        for user_idx in all_users:
            # Get seen items for this user
            if exclude_seen and user_interactions is not None:
                if self.reverse_user_mapping is not None:
                    user_id = self.reverse_user_mapping[user_idx]
                else:
                    user_id = user_idx

                if user_id in user_interactions:
                    seen_items = user_interactions[user_id]
                else:
                    seen_items = []
            else:
                seen_items = None

            # Get recommendations for this user
            if self.reverse_user_mapping is not None:
                user_id = self.reverse_user_mapping[user_idx]
            else:
                user_id = user_idx

            recommendations = self.recommend(user_id, n, exclude_seen, seen_items)
            all_recommendations[user_id] = recommendations

        return all_recommendations

    def _get_num_users(self) -> int:
        """
        Get the number of users in the model.

        This method should be overridden by subclasses if needed.

        Returns:
            Number of users.
        """
        if self.user_mapping is not None:
            return len(self.user_mapping)

        return 0

    def _get_num_items(self) -> int:
        """
        Get the number of items in the model.

        This method should be overridden by subclasses if needed.

        Returns:
            Number of items.
        """
        if self.item_mapping is not None:
            return len(self.item_mapping)

        return 0

    def save(self, filepath: str) -> None:
        """
        Save the model to disk.

        Args:
            filepath: Path to save the model.
        """
        # Create the directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        # Collect model data
        model_data = {
            'name': self.name,
            'config': self.config,
            'user_mapping': self.user_mapping,
            'item_mapping': self.item_mapping,
            'is_fitted': self.is_fitted,
            'model_specific': self._get_model_specific_data()
        }

        # Save to disk
        save_object(model_data, filepath)
        logger.info(f"Model saved to {filepath}")

    def load(self, filepath: str) -> None:
        """
        Load the model from disk.

        Args:
            filepath: Path to the saved model.
        """
        # Load from disk
        model_data = load_object(filepath)

        # Restore model state
        self.name = model_data['name']
        self.config = model_data['config']
        self.user_mapping = model_data['user_mapping']
        self.item_mapping = model_data['item_mapping']

        if self.user_mapping is not None:
            self.reverse_user_mapping = {v: k for k, v in self.user_mapping.items()}

        if self.item_mapping is not None:
            self.reverse_item_mapping = {v: k for k, v in self.item_mapping.items()}

        self.is_fitted = model_data['is_fitted']

        # Restore model-specific data
        self._restore_model_specific_data(model_data['model_specific'])

        logger.info(f"Model loaded from {filepath}")

    @abstractmethod
    def _get_model_specific_data(self) -> Dict[str, Any]:
        """
        Get model-specific data for saving.

        This method should be implemented by each recommender to save
        any additional data needed to restore the model state.

        Returns:
            Dictionary with model-specific data.
        """
        pass

    @abstractmethod
    def _restore_model_specific_data(self, data: Dict[str, Any]) -> None:
        """
        Restore model-specific data after loading.

        This method should be implemented by each recommender to restore
        any additional data needed for the model state.

        Args:
            data: Dictionary with model-specific data.
        """
        pass