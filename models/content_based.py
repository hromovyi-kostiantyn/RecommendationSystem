import pickle

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity, linear_kernel
from scipy.sparse import csr_matrix, hstack

from models.base import BaseRecommender
from data.preprocessing import create_tfidf_features
from utils.logger import get_logger

logger = get_logger(__name__)


class ContentBasedRecommender(BaseRecommender):
    """
    Content-based recommendation model that uses item features to make recommendations.

    This model calculates item similarity based on content features (text descriptions,
    categories, etc.) and recommends items similar to those a user has liked in the past.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the content-based recommender.

        Args:
            config: Dictionary with configuration parameters.
        """
        super().__init__("ContentBased", config)

        # Get config values specific to ContentBasedRecommender
        content_config = config.get('models.content_based', {})

        # TF-IDF parameters
        self.tfidf_config = content_config.get('tfidf', {})
        self.max_features = self.tfidf_config.get('max_features', 100)
        self.min_df = self.tfidf_config.get('min_df', 0.01)
        self.max_df = self.tfidf_config.get('max_df', 0.95)

        # Feature configuration
        self.feature_config = content_config.get('features', {})
        self.text_features = self.feature_config.get('text', ['product_name'])
        self.categorical_features = self.feature_config.get('categorical', ['category', 'sub_category'])
        self.numerical_features = self.feature_config.get('numerical', ['cost'])

        # Similarity metric
        self.similarity_metric = content_config.get('similarity_metric', 'cosine')

        # Initialize data containers
        self.item_features = None
        self.item_feature_matrix = None
        self.item_similarity_matrix = None
        self.user_profiles = {}
        self.vectorizers = {}

        logger.info(f"Initialized ContentBasedRecommender with {self.similarity_metric} similarity, "
                    f"using {len(self.text_features)} text features, "
                    f"{len(self.categorical_features)} categorical features, and "
                    f"{len(self.numerical_features)} numerical features")

    def fit(self, train_data: pd.DataFrame, item_data: pd.DataFrame, user_data: Optional[pd.DataFrame] = None) -> None:
        """
        Train the content-based recommender by calculating item features and user profiles.

        Args:
            train_data: DataFrame with columns [user_id, item_id, rating].
            item_data: DataFrame with item features.
            user_data: Optional DataFrame with user features.
        """
        logger.info("Fitting content-based recommender")

        # Extract column names
        user_col = 'customer_key'
        item_col = 'product_key'
        rating_col = 'rating'

        # Ensure rating column exists
        if rating_col not in train_data.columns:
            logger.info("Rating column not found, using implicit feedback (value=1.0)")
            train_data[rating_col] = 1.0

        # Process item features
        self.item_features = item_data.copy()

        # Create feature matrix for items
        self.item_feature_matrix = self._create_item_feature_matrix(self.item_features)

        # Calculate item similarity matrix
        self.item_similarity_matrix = self._calculate_item_similarity(self.item_feature_matrix)

        # Create user profiles based on liked items
        self._create_user_profiles(train_data, user_col, item_col, rating_col)

        self.is_fitted = True
        logger.info("Content-based recommender fitted successfully")

    def _create_item_feature_matrix(self, item_features: pd.DataFrame) -> csr_matrix:
        """
        Create a feature matrix for items based on text, categorical, and numerical features.

        Args:
            item_features: DataFrame with item features.

        Returns:
            Feature matrix as a sparse matrix.
        """
        logger.info("Creating item feature matrix")

        feature_matrices = []

        # Process text features
        for text_feature in self.text_features:
            if text_feature in item_features.columns:
                logger.info(f"Processing text feature: {text_feature}")

                # Create TF-IDF features
                tfidf_df, vectorizer = create_tfidf_features(
                    item_features,
                    text_feature,
                    max_features=self.max_features,
                    min_df=self.min_df,
                    max_df=self.max_df
                )

                # Store vectorizer for future use
                self.vectorizers[text_feature] = vectorizer

                # Add to feature matrices
                feature_matrices.append(csr_matrix(tfidf_df.values))

        # Process categorical features
        for cat_feature in self.categorical_features:
            if cat_feature in item_features.columns:
                logger.info(f"Processing categorical feature: {cat_feature}")

                # Create one-hot encoding
                dummies = pd.get_dummies(
                    item_features[cat_feature],
                    prefix=cat_feature,
                    dummy_na=False
                )

                # Add to feature matrices
                feature_matrices.append(csr_matrix(dummies.values))

        # Process numerical features
        numerical_df = pd.DataFrame()
        for num_feature in self.numerical_features:
            if num_feature in item_features.columns:
                logger.info(f"Processing numerical feature: {num_feature}")

                # Handle missing values
                if item_features[num_feature].isnull().any():
                    # Fill with median
                    item_features[num_feature] = item_features[num_feature].fillna(
                        item_features[num_feature].median()
                    )

                # Normalize to 0-1 scale
                min_val = item_features[num_feature].min()
                max_val = item_features[num_feature].max()

                if max_val > min_val:
                    normalized = (item_features[num_feature] - min_val) / (max_val - min_val)
                    numerical_df[num_feature] = normalized
                else:
                    # If all values are the same, use 0.5
                    numerical_df[num_feature] = 0.5

        if not numerical_df.empty:
            # Add to feature matrices
            feature_matrices.append(csr_matrix(numerical_df.values))

        # Combine all feature matrices
        if feature_matrices:
            combined_matrix = hstack(feature_matrices)
            logger.info(f"Created item feature matrix with shape {combined_matrix.shape}")
            return combined_matrix
        else:
            # If no features were processed successfully, create a dummy matrix
            logger.warning("No features were processed successfully, using dummy matrix")
            return csr_matrix((len(item_features), 1))

    def _calculate_item_similarity(self, feature_matrix: csr_matrix) -> np.ndarray:
        """
        Calculate item similarity matrix based on feature matrix.

        Args:
            feature_matrix: Feature matrix for items.

        Returns:
            Item similarity matrix.
        """
        logger.info(f"Calculating item similarity using {self.similarity_metric} metric")

        if self.similarity_metric == 'cosine':
            # Calculate cosine similarity
            similarity = cosine_similarity(feature_matrix)
        elif self.similarity_metric == 'linear':
            # Calculate dot product (linear kernel)
            similarity = linear_kernel(feature_matrix)
        else:
            # Default to cosine similarity
            logger.warning(f"Unknown similarity metric: {self.similarity_metric}, using cosine similarity")
            similarity = cosine_similarity(feature_matrix)

        # Set diagonal to 0 to avoid self-similarity
        np.fill_diagonal(similarity, 0)

        logger.info(f"Created item similarity matrix with shape {similarity.shape}")
        return similarity

    def _create_user_profiles(self, train_data: pd.DataFrame, user_col: str, item_col: str, rating_col: str) -> None:
        """
        Create user profiles based on their interactions with items.

        Each user profile is represented as a weighted average of item features they've interacted with.

        Args:
            train_data: DataFrame with user-item interactions.
            user_col: Column name for user IDs.
            item_col: Column name for item IDs.
            rating_col: Column name for ratings.
        """
        logger.info("Creating user profiles")

        # Get all unique user IDs
        user_ids = train_data[user_col].unique()

        # Create a user profile for each user
        for user_id in user_ids:
            # Get items rated by the user
            user_data = train_data[train_data[user_col] == user_id]

            if len(user_data) == 0:
                continue

            # Get item indices and ratings
            item_indices = []
            ratings = []

            for _, row in user_data.iterrows():
                item_id = row[item_col]

                # Check if item exists in our feature matrix
                if item_id in self.item_features.index:
                    item_idx = self.item_features.index.get_loc(item_id)
                    item_indices.append(item_idx)
                    ratings.append(row[rating_col])

            if not item_indices:
                continue

            # Convert to numpy arrays
            item_indices = np.array(item_indices)
            ratings = np.array(ratings)

            # Normalize ratings to sum to 1
            if np.sum(ratings) > 0:
                ratings = ratings / np.sum(ratings)

            # Create user profile as weighted average of item features
            user_profile = np.zeros(self.item_similarity_matrix.shape[1])
            for i, item_idx in enumerate(item_indices):
                user_profile += ratings[i] * self.item_similarity_matrix[item_idx]

            # Store user profile
            self.user_profiles[user_id] = user_profile

        logger.info(f"Created {len(self.user_profiles)} user profiles")

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

        # Check if user and item exist
        if user_id not in self.user_profiles:
            logger.warning(f"User {user_id} not found in training data")
            return 0.0

        if item_id not in self.item_features.index:
            logger.warning(f"Item {item_id} not found in training data")
            return 0.0

        # Get user profile and item index
        user_profile = self.user_profiles[user_id]
        item_idx = self.item_features.index.get_loc(item_id)

        # Calculate similarity between user profile and item
        sim_score = np.dot(user_profile, self.item_similarity_matrix[item_idx])

        # Normalize to 1-5 range
        if sim_score < 0:
            sim_score = 0

        return min(5.0, 1.0 + 4.0 * sim_score)

    def recommend(self, user_id: int, n: int = None, exclude_seen: bool = True,
                  seen_items: Optional[List[int]] = None) -> List[Tuple[int, float]]:
        """
        Generate recommendations for a user based on content similarity.

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

        # Check if user exists
        if user_id not in self.user_profiles:
            logger.warning(f"User {user_id} not found in training data, using fallback strategy")
            # Return most valuable items based on some metric (e.g., average similarity)
            avg_sim = np.mean(self.item_similarity_matrix, axis=0)
            top_items = np.argsort(avg_sim)[::-1][:n]

            # Convert indices to item IDs
            item_ids = self.item_features.index.values
            return [(item_ids[idx], avg_sim[idx]) for idx in top_items]

        # Get user profile
        user_profile = self.user_profiles[user_id]

        # Calculate similarity scores for all items
        scores = np.zeros(len(self.item_features))
        for i in range(len(self.item_features)):
            scores[i] = np.dot(user_profile, self.item_similarity_matrix[i])

        # Create item ID and score pairs
        item_scores = list(zip(self.item_features.index, scores))

        # Filter out seen items if requested
        if exclude_seen and seen_items:
            item_scores = [(item, score) for item, score in item_scores if item not in seen_items]

        # Sort by score (descending)
        item_scores.sort(key=lambda x: x[1], reverse=True)

        # Return top n
        return item_scores[:n]

    def _get_model_specific_data(self) -> Dict[str, Any]:
        """
        Get model-specific data for saving.

        Returns:
            Dictionary with model-specific data.
        """
        return {
            'max_features': self.max_features,
            'min_df': self.min_df,
            'max_df': self.max_df,
            'text_features': self.text_features,
            'categorical_features': self.categorical_features,
            'numerical_features': self.numerical_features,
            'similarity_metric': self.similarity_metric,
            'item_features': self.item_features.to_dict() if self.item_features is not None else None,
            'item_feature_matrix': self.item_feature_matrix.todok().toarray() if self.item_feature_matrix is not None else None,
            'item_similarity_matrix': self.item_similarity_matrix.tolist() if self.item_similarity_matrix is not None else None,
            'user_profiles': {str(k): v.tolist() for k, v in self.user_profiles.items()},
            'vectorizers': {k: pickle.dumps(v) for k, v in self.vectorizers.items()}
        }

    def _restore_model_specific_data(self, data: Dict[str, Any]) -> None:
        """
        Restore model-specific data after loading.

        Args:
            data: Dictionary with model-specific data.
        """
        import pickle

        self.max_features = data['max_features']
        self.min_df = data['min_df']
        self.max_df = data['max_df']
        self.text_features = data['text_features']
        self.categorical_features = data['categorical_features']
        self.numerical_features = data['numerical_features']
        self.similarity_metric = data['similarity_metric']

        if data['item_features'] is not None:
            self.item_features = pd.DataFrame(data['item_features'])

        if data['item_feature_matrix'] is not None:
            self.item_feature_matrix = csr_matrix(np.array(data['item_feature_matrix']))

        if data['item_similarity_matrix'] is not None:
            self.item_similarity_matrix = np.array(data['item_similarity_matrix'])

        self.user_profiles = {int(k) if k.isdigit() else k: np.array(v) for k, v in data['user_profiles'].items()}

        self.vectorizers = {k: pickle.loads(v) for k, v in data['vectorizers'].items()}