import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from surprise import SVD, Dataset, Reader, accuracy
from surprise.model_selection import train_test_split
from surprise.trainset import Trainset
import pickle

from models.base import BaseRecommender
from utils.logger import get_logger

logger = get_logger(__name__)


class MatrixFactorizationRecommender(BaseRecommender):
    """
    Matrix factorization-based recommendation model.

    This class implements matrix factorization using the Surprise library's SVD algorithm,
    which decomposes the user-item interaction matrix into latent factors.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the matrix factorization recommender.

        Args:
            config: Dictionary with configuration parameters.
        """
        super().__init__("MatrixFactorization", config)

        # Get config values specific to MatrixFactorizationRecommender
        mf_config = config.get('models.collaborative.matrix_fact', {})
        self.n_factors = mf_config.get('n_factors', 50)
        self.n_epochs = mf_config.get('n_epochs', 20)
        self.lr_all = mf_config.get('lr_all', 0.005)
        self.reg_all = mf_config.get('reg_all', 0.02)
        self.biased = mf_config.get('biased', True)

        # Initialize model
        self.model = SVD(
            n_factors=self.n_factors,
            n_epochs=self.n_epochs,
            lr_all=self.lr_all,
            reg_all=self.reg_all,
            biased=self.biased,
            random_state=config.get('data.random_seed', 42)
        )

        # Initialize data containers
        self.trainset = None
        self.raw_ratings = None
        self.global_mean = 0.0

        logger.info(f"Initialized MatrixFactorizationRecommender with n_factors={self.n_factors}, "
                    f"n_epochs={self.n_epochs}, biased={self.biased}")

    def fit(self, train_data: pd.DataFrame) -> None:
        """
        Train the matrix factorization model.

        Args:
            train_data: DataFrame with columns [user_id, item_id, rating].
        """
        logger.info("Fitting matrix factorization recommender")

        # Extract column names
        user_col = 'customer_key'
        item_col = 'product_key'
        rating_col = 'rating'

        # Ensure rating column exists
        if rating_col not in train_data.columns:
            logger.info("Rating column not found, using implicit feedback (value=1.0)")
            train_data[rating_col] = 1.0

        # Create a copy of the dataframe with renamed columns
        ratings_df = train_data[[user_col, item_col, rating_col]].copy()
        ratings_df.columns = ['user', 'item', 'rating']

        # Store ID mappings if not already set
        if self.user_mapping is None:
            unique_users = ratings_df['user'].unique()
            self.user_mapping = {user: i for i, user in enumerate(unique_users)}
            self.reverse_user_mapping = {i: user for user, i in self.user_mapping.items()}

            unique_items = ratings_df['item'].unique()
            self.item_mapping = {item: i for i, item in enumerate(unique_items)}
            self.reverse_item_mapping = {i: item for item, i in self.item_mapping.items()}

            logger.debug(f"Created ID mappings for {len(self.user_mapping)} users "
                         f"and {len(self.item_mapping)} items")

        # Create Surprise dataset
        reader = Reader(rating_scale=(ratings_df['rating'].min(), ratings_df['rating'].max()))
        data = Dataset.load_from_df(ratings_df, reader)

        # Convert to trainset
        self.trainset = data.build_full_trainset()

        # Store global mean
        self.global_mean = self.trainset.global_mean

        # Store raw ratings for prediction fallback
        self.raw_ratings = ratings_df.values.tolist()

        # Train the model
        logger.info(f"Training matrix factorization with {len(self.trainset.ur)} users "
                    f"and {len(self.trainset.ir)} items")
        self.model.fit(self.trainset)

        self.is_fitted = True
        logger.info("Matrix factorization recommender fitted successfully")

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

        # Check if user and item exist in the trainset
        try:
            inner_user_id = self.trainset.to_inner_uid(user_id)
            inner_item_id = self.trainset.to_inner_iid(item_id)

            # Get the prediction
            prediction = self.model.predict(user_id, item_id).est
            return prediction

        except (ValueError, KeyError):
            # If user or item is unknown, return global mean or default
            return self.global_mean if self.global_mean else 0.0

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

        # Check if user exists in the trainset
        try:
            inner_user_id = self.trainset.to_inner_uid(user_id)
        except (ValueError, KeyError):
            logger.warning(f"User {user_id} not found in training data, using fallback strategy")
            # Fallback to most popular items or other strategy
            return []

        # Get all items for prediction
        all_items = list(self.trainset.all_items())

        # If exclude_seen, filter out seen items
        if exclude_seen and seen_items:
            # Convert seen_items to inner ids
            seen_inner_items = []
            for item_id in seen_items:
                try:
                    inner_item_id = self.trainset.to_inner_iid(item_id)
                    seen_inner_items.append(inner_item_id)
                except (ValueError, KeyError):
                    # Item not in trainset, skip
                    pass

            # Filter out seen items
            candidate_items = [item for item in all_items if item not in seen_inner_items]
        else:
            candidate_items = all_items

        # Generate predictions for all candidate items
        predictions = []
        for inner_item_id in candidate_items:
            item_id = self.trainset.to_raw_iid(inner_item_id)
            score = self.predict(user_id, item_id)
            predictions.append((item_id, score))

        # Sort by predicted rating
        predictions.sort(key=lambda x: x[1], reverse=True)

        # Return top n
        return predictions[:n]

    def _get_model_specific_data(self) -> Dict[str, Any]:
        """
        Get model-specific data for saving.

        Returns:
            Dictionary with model-specific data.
        """
        return {
            'n_factors': self.n_factors,
            'n_epochs': self.n_epochs,
            'lr_all': self.lr_all,
            'reg_all': self.reg_all,
            'biased': self.biased,
            'model': pickle.dumps(self.model),
            'global_mean': self.global_mean,
            'raw_ratings': self.raw_ratings
        }

    def _restore_model_specific_data(self, data: Dict[str, Any]) -> None:
        """
        Restore model-specific data after loading.

        Args:
            data: Dictionary with model-specific data.
        """
        self.n_factors = data['n_factors']
        self.n_epochs = data['n_epochs']
        self.lr_all = data['lr_all']
        self.reg_all = data['reg_all']
        self.biased = data['biased']
        self.model = pickle.loads(data['model'])
        self.global_mean = data['global_mean']
        self.raw_ratings = data['raw_ratings']