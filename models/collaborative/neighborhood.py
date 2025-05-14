import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from scipy.sparse import csr_matrix
import heapq
from collections import defaultdict

from models.base import BaseRecommender
from utils.logger import get_logger

logger = get_logger(__name__)


class NeighborhoodRecommender(BaseRecommender):
    """
    Neighborhood-based collaborative filtering recommender.

    This class implements k-nearest neighbors (KNN) based collaborative filtering,
    supporting both user-based and item-based approaches.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the neighborhood recommender.

        Args:
            config: Dictionary with configuration parameters.
        """
        super().__init__("Neighborhood", config)

        # Get config values specific to NeighborhoodRecommender
        collaborative_config = config.get('models.collaborative.neighborhood', {})
        self.user_based = collaborative_config.get('user_based', True)
        self.k_neighbors = collaborative_config.get('k_neighbors', 30)
        self.min_k = collaborative_config.get('min_k', 1)

        # Similarity options
        sim_options = collaborative_config.get('sim_options', {})
        self.sim_name = sim_options.get('name', 'cosine')
        self.min_support = sim_options.get('min_support', 3)

        # Initialize data containers
        self.user_item_matrix = None
        self.item_user_matrix = None
        self.similarity_matrix = None
        self.means = None

        # Log configuration
        logger.info(
            f"Initialized NeighborhoodRecommender with approach={'user-based' if self.user_based else 'item-based'}, "
            f"k_neighbors={self.k_neighbors}, similarity={self.sim_name}")

    def fit(self, train_data: pd.DataFrame) -> None:
        """
        Train the neighborhood recommender.

        Args:
            train_data: DataFrame with columns [user_id, item_id, rating].
        """
        logger.info("Fitting neighborhood recommender")

        # Extract column names
        user_col = 'customer_key'
        item_col = 'product_key'
        rating_col = 'rating'

        # Ensure rating column exists
        if rating_col not in train_data.columns:
            logger.info("Rating column not found, using implicit feedback (value=1.0)")
            train_data[rating_col] = 1.0

        # Create user-item matrix
        user_item_matrix = train_data.pivot_table(
            index=user_col,
            columns=item_col,
            values=rating_col,
            fill_value=0
        )

        # Store the matrix
        self.user_item_matrix = user_item_matrix

        # Calculate and subtract means if using user-based CF
        if self.user_based:
            # Calculate user means
            user_means = user_item_matrix.mean(axis=1)
            # Create matrix to hold means for each user
            means_matrix = np.outer(user_means, np.ones(user_item_matrix.shape[1]))
            # Store means
            self.means = user_means
            # Normalize ratings by subtracting means
            normalized_matrix = user_item_matrix.values - means_matrix
            # Replace NaN with 0
            normalized_matrix = np.nan_to_num(normalized_matrix, nan=0.0)

            # Create similarity matrix
            similarity_matrix = self._compute_similarity(normalized_matrix)
            self.similarity_matrix = similarity_matrix

            logger.info(f"Built user-based similarity matrix with shape {similarity_matrix.shape}")

        else:  # Item-based CF
            # Transpose the matrix for item-based approach
            item_user_matrix = user_item_matrix.T
            self.item_user_matrix = item_user_matrix

            # Create similarity matrix
            similarity_matrix = self._compute_similarity(item_user_matrix.values)
            self.similarity_matrix = similarity_matrix

            logger.info(f"Built item-based similarity matrix with shape {similarity_matrix.shape}")

        self.is_fitted = True
        logger.info("Neighborhood recommender fitted successfully")

    def _compute_similarity(self, matrix: np.ndarray) -> np.ndarray:
        """
        Compute similarity matrix using the specified similarity measure.

        Args:
            matrix: User-item or item-user matrix.

        Returns:
            Similarity matrix.
        """
        logger.info(f"Computing {self.sim_name} similarity...")

        n_rows = matrix.shape[0]
        similarity = np.zeros((n_rows, n_rows))

        if self.sim_name == 'cosine':
            # Calculate norm for each row
            norms = np.sqrt(np.sum(matrix ** 2, axis=1))

            # Avoid division by zero
            norms[norms == 0] = 1e-10

            # Normalized matrix
            normalized = matrix / norms[:, np.newaxis]

            # Compute cosine similarity
            similarity = np.dot(normalized, normalized.T)

        elif self.sim_name == 'pearson':
            # Center the rows (subtract row mean from each element)
            row_means = np.mean(matrix, axis=1)
            centered = matrix - row_means[:, np.newaxis]

            # Calculate norm for each centered row
            norms = np.sqrt(np.sum(centered ** 2, axis=1))

            # Avoid division by zero
            norms[norms == 0] = 1e-10

            # Normalized centered matrix
            normalized = centered / norms[:, np.newaxis]

            # Compute Pearson correlation
            similarity = np.dot(normalized, normalized.T)

        elif self.sim_name == 'jaccard':
            # Convert to binary matrix
            binary_matrix = (matrix > 0).astype(float)

            # Calculate intersection and union for each pair
            for i in range(n_rows):
                for j in range(i, n_rows):
                    intersection = np.sum((binary_matrix[i] > 0) & (binary_matrix[j] > 0))
                    union = np.sum((binary_matrix[i] > 0) | (binary_matrix[j] > 0))

                    if union > 0:
                        sim = intersection / union
                    else:
                        sim = 0.0

                    similarity[i, j] = sim
                    similarity[j, i] = sim

        else:  # Default to cosine
            logger.warning(f"Unknown similarity measure: {self.sim_name}, using cosine similarity")
            norms = np.sqrt(np.sum(matrix ** 2, axis=1))
            norms[norms == 0] = 1e-10
            normalized = matrix / norms[:, np.newaxis]
            similarity = np.dot(normalized, normalized.T)

        # Set diagonal to 0 to avoid self-similarity
        np.fill_diagonal(similarity, 0)

        return similarity

    def predict(self, user_id: int, item_id: int) -> float:
        """
        Predict the rating for a user-item pair.

        Args:
            user_id: User ID.
            item_id: Item ID.

        Returns:
            Predicted rating.
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")

        # Convert to internal indices if mappings exist
        if self.user_mapping is not None and user_id in self.user_mapping:
            user_idx = self.user_mapping[user_id]
        else:
            user_idx = user_id

        if self.item_mapping is not None and item_id in self.item_mapping:
            item_idx = self.item_mapping[item_id]
        else:
            item_idx = item_id

        # Check if user/item exists in training data
        if user_idx >= self.user_item_matrix.shape[0] or item_idx >= self.user_item_matrix.shape[1]:
            # Return mean rating or 0.0 if no mean exists
            if self.means is not None and user_idx < len(self.means):
                return self.means[user_idx]
            else:
                return 0.0

        # Get user-item matrix indices
        user_idx_matrix = self.user_item_matrix.index.get_loc(user_id) if isinstance(user_id, (int,
                                                                                               float)) and user_id in self.user_item_matrix.index else -1
        item_idx_matrix = self.user_item_matrix.columns.get_loc(item_id) if isinstance(item_id, (int,
                                                                                                 float)) and item_id in self.user_item_matrix.columns else -1

        # If user or item not found, return default prediction
        if user_idx_matrix == -1 or item_idx_matrix == -1:
            if self.means is not None and user_idx_matrix != -1:
                return self.means.iloc[user_idx_matrix]
            else:
                return 0.0

        if self.user_based:
            return self._predict_user_based(user_idx_matrix, item_idx_matrix)
        else:
            return self._predict_item_based(user_idx_matrix, item_idx_matrix)

    def _predict_user_based(self, user_idx: int, item_idx: int) -> float:
        """
        Make a prediction using user-based collaborative filtering.

        Args:
            user_idx: User index in the matrix.
            item_idx: Item index in the matrix.

        Returns:
            Predicted rating.
        """
        # If user has already rated this item, return the actual rating
        if self.user_item_matrix.iloc[user_idx, item_idx] > 0:
            return self.user_item_matrix.iloc[user_idx, item_idx]

        # Get user's mean rating
        user_mean = self.means.iloc[user_idx]

        # Get similarity scores between the user and all other users
        similarities = self.similarity_matrix[user_idx]

        # Get ratings given to the target item by all users
        item_ratings = self.user_item_matrix.iloc[:, item_idx].values

        # Filter out users who haven't rated the item
        mask = item_ratings > 0
        if np.sum(mask) == 0:
            # No users have rated this item, return user's mean
            return user_mean

        # Get similarities and ratings from users who rated the item
        similarities = similarities[mask]
        ratings = item_ratings[mask]
        means = self.means.iloc[mask].values

        # Get top-k neighbors
        if len(similarities) > self.k_neighbors:
            # Get indices of top-k similarities
            top_k_idx = np.argsort(similarities)[-self.k_neighbors:]
            similarities = similarities[top_k_idx]
            ratings = ratings[top_k_idx]
            means = means[top_k_idx]

        # Check if we have enough neighbors
        if len(similarities) < self.min_k or np.sum(similarities) == 0:
            return user_mean

        # Calculate weighted average of deviations from means
        deviations = ratings - means
        weighted_deviations = np.sum(similarities * deviations) / np.sum(similarities)

        # Predict rating
        prediction = user_mean + weighted_deviations

        # Clip to reasonable bounds
        prediction = max(min(prediction, 5.0), 1.0)

        return prediction

    def _predict_item_based(self, user_idx: int, item_idx: int) -> float:
        """
        Make a prediction using item-based collaborative filtering.

        Args:
            user_idx: User index in the matrix.
            item_idx: Item index in the matrix.

        Returns:
            Predicted rating.
        """
        # If user has already rated this item, return the actual rating
        if self.user_item_matrix.iloc[user_idx, item_idx] > 0:
            return self.user_item_matrix.iloc[user_idx, item_idx]

        # Get similarity scores between the target item and all other items
        similarities = self.similarity_matrix[item_idx]

        # Get user's ratings for all items
        user_ratings = self.user_item_matrix.iloc[user_idx].values

        # Filter out items the user hasn't rated
        mask = user_ratings > 0
        if np.sum(mask) == 0:
            # User hasn't rated any items, return default
            return 0.0

        # Get similarities and ratings for items the user has rated
        similarities = similarities[mask]
        ratings = user_ratings[mask]

        # Get top-k neighbors
        if len(similarities) > self.k_neighbors:
            # Get indices of top-k similarities
            top_k_idx = np.argsort(similarities)[-self.k_neighbors:]
            similarities = similarities[top_k_idx]
            ratings = ratings[top_k_idx]

        # Check if we have enough neighbors
        if len(similarities) < self.min_k or np.sum(similarities) == 0:
            return 0.0

        # Calculate weighted average
        prediction = np.sum(similarities * ratings) / np.sum(similarities)

        # Clip to reasonable bounds
        prediction = max(min(prediction, 5.0), 1.0)

        return prediction

    def _get_model_specific_data(self) -> Dict[str, Any]:
        """
        Get model-specific data for saving.

        Returns:
            Dictionary with model-specific data.
        """
        return {
            'user_based': self.user_based,
            'k_neighbors': self.k_neighbors,
            'min_k': self.min_k,
            'sim_name': self.sim_name,
            'min_support': self.min_support,
            'user_item_matrix': self.user_item_matrix.to_dict() if self.user_item_matrix is not None else None,
            'item_user_matrix': self.item_user_matrix.to_dict() if self.item_user_matrix is not None else None,
            'similarity_matrix': self.similarity_matrix.tolist() if self.similarity_matrix is not None else None,
            'means': self.means.to_dict() if self.means is not None else None
        }

    def _restore_model_specific_data(self, data: Dict[str, Any]) -> None:
        """
        Restore model-specific data after loading.

        Args:
            data: Dictionary with model-specific data.
        """
        self.user_based = data['user_based']
        self.k_neighbors = data['k_neighbors']
        self.min_k = data['min_k']
        self.sim_name = data['sim_name']
        self.min_support = data['min_support']

        if data['user_item_matrix'] is not None:
            self.user_item_matrix = pd.DataFrame(data['user_item_matrix'])

        if data['item_user_matrix'] is not None:
            self.item_user_matrix = pd.DataFrame(data['item_user_matrix'])

        if data['similarity_matrix'] is not None:
            self.similarity_matrix = np.array(data['similarity_matrix'])

        if data['means'] is not None:
            self.means = pd.Series(data['means'])